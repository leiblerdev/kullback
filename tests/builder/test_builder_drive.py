"""The Builder drives many Tasks per call: replay and run over every Task, the root listing up front."""

from __future__ import annotations

import asyncio
import json

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.builder import env_files
from kullback.builder import session as session_mod
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn
from tests.builder.session_fixtures import reply
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def _trace(trace_id: str, prefix: str) -> Trace:
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Give widget w1 the label striped.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=[f"{prefix}1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"widget_id": "w1"}', tool_call_ids=[f"{prefix}1"], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=[f"{prefix}2"], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"widget_id": "w1"}', tool_call_ids=[f"{prefix}2"], raw_ptr=PTR),
        Turn(idx=6, role="assistant", content="Done.", raw_ptr=PTR),
        Turn(idx=7, role="user", content="Thanks, that is the right label.", raw_ptr=PTR),
    ]
    calls = [
        ToolCall(id=f"{prefix}1", name="describe_widget", args={"widget_id": "w1"},
                 result={"widget_id": "w1", "label": "plain"}, raw_ptr=PTR, trace_id=trace_id),
        ToolCall(id=f"{prefix}2", name="rename_widget", args={"widget_id": "w1", "label": "striped"},
                 result={"widget_id": "w1", "label": "striped"}, raw_ptr=PTR, trace_id=trace_id),
    ]
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _workdir(tmp_path):
    """Three Tasks: two with a recorded Trace each, one whose Trace was never stored."""
    root = tmp_path / "work"
    for task_id, run_id in (("widget_task", "rec1"), ("second_task", "rec2"), ("orphan_task", "rec9")):
        write_env(root, task_id=task_id, run_id=run_id)
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for trace_id, prefix in (("rec1", "c"), ("rec2", "d")):
        trace = _trace(trace_id, prefix)
        (traces / f"{trace_id}.json").write_text(json.dumps(trace.model_dump(mode="json", by_alias=True)),
                                                 encoding="utf-8")
    return root


def _tools(root, model=None):
    return {tool.name: tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model)}


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def _candidate():
    return TestModel([reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
                      reply("Done, widget w1 is labelled striped.")], loop=True)


def test_replay_with_no_task_id_reports_every_task_once_and_the_per_tool_rows(tmp_path):
    root = _workdir(tmp_path)
    result = _run(_tools(root)["replay"], {})
    assert not result.is_error, result.content
    tasks = result.details["tasks"]
    assert sorted(task["task_id"] for task in tasks) == ["orphan_task", "second_task", "widget_task"]
    by_task = {task["task_id"]: task for task in tasks}
    assert by_task["widget_task"]["confirmed"] and by_task["second_task"]["confirmed"]
    assert by_task["orphan_task"]["error"]
    assert result.content.startswith("replay of every Task: 2 of 3 Tasks replayed, 2 confirmed References")
    assert "tool describe_widget: fidelity 1.0000 over 2 calls" in result.content
    assert "tool rename_widget: fidelity 1.0000 over 2 calls" in result.content
    assert all(result.content.count(f"task {task_id}:") == 1
               for task_id in ("widget_task", "second_task", "orphan_task"))
    assert "task orphan_task: not replayed" in result.content
    stored = json.loads((root / "replays.json").read_text(encoding="utf-8"))
    assert sorted(stored) == ["second_task", "widget_task"]
    assert len(result.content.splitlines()) < 60


def test_replay_with_no_task_id_names_where_the_worst_task_first_differs(tmp_path):
    root = _workdir(tmp_path)
    db = json.loads((root / "db.json").read_text(encoding="utf-8"))
    db["widgets"]["w1"]["label"] = "dotted"
    (root / "db.json").write_text(json.dumps(db), encoding="utf-8")
    result = _run(_tools(root)["replay"], {})
    assert not result.is_error, result.content
    assert "first differs at describe_widget [label]" in result.content


def test_run_with_no_task_ids_runs_only_tasks_with_a_confirmed_reference_and_names_the_rest(
        tmp_path, monkeypatch):
    root = _workdir(tmp_path)
    tools = _tools(root, model=_candidate())
    replay = _run(tools["replay"], {"task_id": "widget_task"})
    assert not replay.is_error, replay.content
    result = _run(tools["run"], {})
    assert not result.is_error, result.content
    assert {row["task_id"] for row in result.details["runs"]} == {"widget_task"}
    assert result.details["not_run"] == []
    assert result.content.startswith("reroll of 1 Tasks: 1 Runs, 1 finished, 0 Tasks left not run")

    _run(tools["replay"], {})
    monkeypatch.setattr(domain_tools_mod, "RUNS_PER_CALL", 1)
    capped = _run(tools["run"], {})
    assert not capped.is_error, capped.content
    assert [row["task_id"] for row in capped.details["runs"]] == ["second_task"]
    assert capped.details["not_run"] == ["widget_task"]
    assert "1 Tasks left not run" in capped.content
    assert "not run, past the cap of 1 Runs per call: widget_task" in capped.content


def test_run_with_no_confirmed_reference_says_replay_comes_first(tmp_path):
    root = _workdir(tmp_path)
    result = _run(_tools(root, model=_candidate())["run"], {})
    assert not result.is_error, result.content
    assert result.details["runs"] == []
    assert "no open Task has a confirmed Reference, replay first" in result.content


def test_a_run_with_no_verdict_says_examine_derives_the_verifier_first():
    text = domain_tools_mod._render_run(domain_tools_mod.RunResult(
        summary="reroll", runs=[domain_tools_mod.RunRow(run_id="r-0", verdict=None)]))
    assert text.endswith("no Verifier yet: examine derives it from the confirmed Reference, "
                         "then run scores against it")


def test_the_opening_lists_what_the_root_holds_and_nothing_else(tmp_path):
    root = _workdir(tmp_path)
    env_files.expose(root)
    env_files.explode(root)
    opening = session_mod.opening(root)
    (listing,) = [line for line in opening.splitlines() if line.startswith("Your root holds: ")]
    names = listing.removeprefix("Your root holds: ").removesuffix(".").split(", ")
    assert "db.json" in names and "tools/" in names
    assert names == sorted(names)
    env = root / "env"
    assert all((env / name.rstrip("/")).exists() for name in names)
    assert all((env / name.rstrip("/")).is_dir() == name.endswith("/") for name in names)
    assert opening.endswith(session_mod.OPENING)
