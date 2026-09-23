"""The runner as a tool: version, replay names the first differing column, run verdicts a Run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner import tool
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, Verifier, run_path, stored_run_path, write_json
from kullback.runner.world.loading import EnvironmentError
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def _trace() -> Trace:
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="Give widget w1 the label striped.", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=["c1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content='{"widget_id": "w1"}', tool_call_ids=["c1"],
             raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=None, tool_call_ids=["c2"], raw_ptr=PTR),
        Turn(idx=5, role="tool", content='{"widget_id": "w1"}', tool_call_ids=["c2"],
             raw_ptr=PTR),
        Turn(idx=6, role="assistant", content="Done.", raw_ptr=PTR),
        Turn(idx=7, role="user", content="Thanks, that is the right label.", raw_ptr=PTR),
    ]
    calls = [
        ToolCall(id="c1", name="describe_widget", args={"widget_id": "w1"},
                 result={"widget_id": "w1", "label": "plain"}, raw_ptr=PTR, trace_id="rec1"),
        ToolCall(id="c2", name="rename_widget", args={"widget_id": "w1", "label": "striped"},
                 result={"widget_id": "w1", "label": "striped"}, raw_ptr=PTR, trace_id="rec1"),
    ]
    return Trace(trace_id="rec1", raw_hash="r" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _env_with_trace(root: Path) -> Path:
    write_env(root)
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    write_json(traces / "rec1.json", _trace().model_dump(mode="json", by_alias=True))
    return root


def _break_rename(root: Path) -> None:
    """A body that answers one recorded call with the wrong value: every rename lands elsewhere."""
    path = root / "env" / "tools.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("row.label = label", 'row.label = "smudged"')
    text = text.replace('return {"widget_id": widget_id, "label": label}',
                        'return {"widget_id": widget_id, "label": "smudged"}')
    path.write_text(text, encoding="utf-8")


def test_version_is_a_stable_hex_digest():
    first, second = tool.version(), tool.version()
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    assert first == second


def test_replay_confirms_a_straight_body_and_names_the_first_differing_column_of_a_broken_one(tmp_path):
    """A body returning the wrong value on one recorded call is reported by its column."""
    root = _env_with_trace(tmp_path / "env")
    straight = tool.replay(root, "widget_task", workdir=tmp_path / "work")
    assert straight.confirmed, straight.reasons
    assert straight.fidelity == 1.0
    assert all(call.first_differing_column is None for call in straight.calls)
    _break_rename(root)
    report = tool.replay(root, "widget_task", workdir=tmp_path / "work")
    assert not report.confirmed
    assert report.fidelity == 0.5
    renamed = next(call for call in report.calls if call.tool == "rename_widget")
    assert renamed.verdict == "differs"
    assert renamed.first_differing_column == "label"
    described = next(call for call in report.calls if call.tool == "describe_widget")
    assert described.verdict in ("same", "cosmetic", "both_refused")
    assert described.first_differing_column is None


def _describe_trace(trace_id: str, label: str) -> Trace:
    """A Run that only looks widget w1 up and saw it carry `label`."""
    turns = [
        Turn(idx=0, role="assistant", content="Hi! How can I help?", raw_ptr=PTR),
        Turn(idx=1, role="user", content="What label does widget w1 carry?", raw_ptr=PTR),
        Turn(idx=2, role="assistant", content=None, tool_call_ids=[f"{trace_id}-c1"], raw_ptr=PTR),
        Turn(idx=3, role="tool", content=json.dumps({"widget_id": "w1", "label": label}),
             tool_call_ids=[f"{trace_id}-c1"], raw_ptr=PTR),
        Turn(idx=4, role="assistant", content=f"It carries {label}.", raw_ptr=PTR),
        Turn(idx=5, role="user", content="Thanks.", raw_ptr=PTR),
    ]
    calls = [ToolCall(id=f"{trace_id}-c1", name="describe_widget", args={"widget_id": "w1"},
                      result={"widget_id": "w1", "label": label}, raw_ptr=PTR, trace_id=trace_id)]
    return Trace(trace_id=trace_id, raw_hash="d" * 64, ingest_version="1", source="test",
                 turns=turns, tool_calls=calls, raw_ptr=PTR)


def _env_with_three_runs(root: Path) -> Path:
    """The widget Task over three Runs: the reference renames w1, the other two only look it up.

    The second saw w1 as the Starting state holds it; the third saw it carry another label, which
    its own layer of the Task's overlay pins (D213), and it is the anchor's held-out Run.
    """
    _env_with_trace(root)
    write_json(root / "traces" / "rec2.json",
               _describe_trace("rec2", "plain").model_dump(mode="json", by_alias=True))
    write_json(root / "traces" / "rec3.json",
               _describe_trace("rec3", "dotted").model_dump(mode="json", by_alias=True))
    write_json(root / "tasks" / "widget_task.json",
               {"id": "widget_task", "run_ids": ["rec2", "rec1", "rec3"],
                "intent": "give widget w1 the label striped"})
    write_json(root / "overlays" / "widget_task.json", {
        "overlay": {"task_id": "widget_task", "rows": [
            {"table": "widgets", "id": "w1", "version_hash": "seen-plain"}], "steps": []},
        "values": {"seen-plain": {"widget_id": "w1", "label": "plain"},
                   "seen-dotted": {"widget_id": "w1", "label": "dotted"}},
        "runs": {"rec3": {"rows": [{"table": "widgets", "id": "w1", "version_hash": "seen-dotted"}]}},
    })
    write_json(root / "anchor.json", {"held_out": {"widget_task": ["rec3"]}})
    return root


def test_replay_all_reports_every_trace_with_the_reference_first(tmp_path):
    root = _env_with_three_runs(tmp_path / "env")
    reports = tool.replay_all(root, "widget_task", workdir=tmp_path / "work")
    assert [report.trace_id for report in reports] == ["rec1", "rec2", "rec3"]
    assert [report.reference for report in reports] == [True, False, False]


def test_replay_all_replays_each_trace_in_a_fresh_world(tmp_path):
    """The reference renames w1 first; the next Run still reads the label the Starting state holds."""
    root = _env_with_three_runs(tmp_path / "env")
    reports = {report.trace_id: report for report in tool.replay_all(root, "widget_task",
                                                                     workdir=tmp_path / "work")}
    assert reports["rec1"].confirmed, reports["rec1"].reasons
    assert [(call.verdict, call.ours) for call in reports["rec2"].calls] == [
        ("same", '{"label": "plain", "widget_id": "w1"}')]


def test_a_trace_replays_on_its_own_overlay_layer(tmp_path):
    """Two Runs saw w1 in two labels; each is served the version it recorded (D213)."""
    root = _env_with_three_runs(tmp_path / "env")
    third = tool.replay(root, "widget_task", workdir=tmp_path / "work", trace_id="rec3")
    assert [(call.verdict, call.ours) for call in third.calls] == [
        ("same", '{"label": "dotted", "widget_id": "w1"}')]
    second = tool.replay(root, "widget_task", workdir=tmp_path / "work", trace_id="rec2")
    assert [(call.verdict, call.ours) for call in second.calls] == [
        ("same", '{"label": "plain", "widget_id": "w1"}')]


def test_replay_all_marks_the_anchor_runs_held_out(tmp_path):
    root = _env_with_three_runs(tmp_path / "env")
    reports = tool.replay_all(root, "widget_task", workdir=tmp_path / "work")
    assert {report.trace_id: report.held_out for report in reports} == {
        "rec1": False, "rec2": False, "rec3": True}
    assert reports[-1].as_dict()["held_out"] is True


def test_replay_refuses_a_trace_the_task_does_not_hold(tmp_path):
    root = _env_with_three_runs(tmp_path / "env")
    with pytest.raises(EnvironmentError, match="no Trace missing"):
        tool.replay(root, "widget_task", workdir=tmp_path / "work", trace_id="missing")


def test_run_plays_verdicts_and_diffs_one_run(tmp_path):
    root = _env_with_trace(tmp_path / "env")
    model = TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ])
    report = tool.run(root, "widget_task", model, workdir=tmp_path / "work")
    assert report.path == "runs/widget_task-0.jsonl"
    assert (tmp_path / "work" / report.path).is_file()
    assert [taken["name"] for taken in report.routes] == ["describe_widget", "rename_widget"]
    assert {taken["route"] for taken in report.routes} == {"code"}
    assert report.world_diff["widgets"]["w1"] == {
        "before": {"widget_id": "w1", "label": "plain"},
        "after": {"widget_id": "w1", "label": "striped"}}
    assert report.verdict is not None and report.verdict.passed is True
    assert report.runner_version == tool.version()
    assert report.termination_reason in ("agent_stop", "user_stop")
    rerolled = tool.reroll(root, "widget_task", TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True), count=2, workdir=tmp_path / "work")
    assert [item.run_id for item in rerolled] == ["widget_task-0", "widget_task-1"]


def _rename_script():
    """A model that reads the widget and writes the striped label, then stops."""
    return TestModel([
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)


def test_probe_scores_the_write_it_was_told_to_reach_and_lands_under_probes(tmp_path):
    """A probe that calls the write tool directly passes the Verifier and never lands in runs/."""
    root = _env_with_trace(tmp_path / "env")
    verifier = Verifier.model_validate(json.loads(
        (root / "verifiers" / "widget_task.json").read_text(encoding="utf-8")))
    report = tool.probe(root, "widget_task", _rename_script(), verifier=verifier,
                        workdir=tmp_path / "work")
    assert report.run_id == "probe-widget_task"
    assert report.path == "probes/probe-widget_task.jsonl"
    assert (tmp_path / "work" / report.path).is_file()
    assert list((tmp_path / "work" / "runs").glob("*.jsonl")) == []
    assert report.verdict is not None and report.verdict.passed is True


def test_replay_calls_writes_the_given_run_id_under_the_task_and_reseeds_stably(tmp_path):
    """A hand-written call list replays into runs/<task>/ under its own id, same rows twice."""
    root = _env_with_trace(tmp_path / "env")
    calls = [
        {"id": "c1", "name": "describe_widget", "args": {"widget_id": "w1"},
         "requestor": "assistant", "turn": 0},
        {"id": "c2", "name": "rename_widget", "args": {"widget_id": "w1", "label": "striped"},
         "requestor": "assistant", "turn": 0},
    ]
    first = tool.replay_calls(root, "widget_task", calls, run_id="synth-1",
                              workdir=tmp_path / "work")
    assert first.run_id == "synth-1"
    assert first.path == "runs/widget_task/synth-1.jsonl"
    second = tool.replay_calls(root, "widget_task", calls, run_id="synth-1",
                               workdir=tmp_path / "work")
    assert [(call.tool, call.recorded, call.ours) for call in second.calls] == [
        (call.tool, call.recorded, call.ours) for call in first.calls]


def test_reroll_with_prefix_names_its_files_for_the_buyer_and_carries_user_end(tmp_path):
    """Prefixed re-rolls file under the buyer's name and every report carries user_end."""
    root = _env_with_trace(tmp_path / "env")
    reports = tool.reroll(root, "widget_task", _rename_script(), count=2,
                          workdir=tmp_path / "work", prefix="second-path-r0-b0")
    assert [report.run_id for report in reports] == ["second-path-r0-b0-0", "second-path-r0-b0-1"]
    assert all(Path(report.path).name.startswith("second-path-r0-b0-") for report in reports)
    assert all((tmp_path / "work" / report.path).is_file() for report in reports)
    assert all("user_end" in report.as_dict() for report in reports)
    direct = tool.run(root, "widget_task", _rename_script(), workdir=tmp_path / "named", run_id="custom-7")
    assert direct.run_id == "custom-7" and (tmp_path / "named" / "runs" / "custom-7.jsonl").is_file()



def test_every_run_path_a_report_stores_is_relative_to_the_workdir_whatever_the_cwd(tmp_path, monkeypatch):
    """A workdir named relative to the cwd still stores runs/..., which opens against the workdir."""
    root = _env_with_trace(tmp_path / "env")
    monkeypatch.chdir(tmp_path)
    played = tool.run(root, "widget_task", _rename_script(), workdir="work")
    replayed = tool.replay(root, "widget_task", workdir="work")
    assert played.path == "runs/widget_task-0.jsonl"
    assert replayed.path.startswith("runs/widget_task/") and not Path(replayed.path).is_absolute()
    assert all(run_path(tmp_path / "work", report.path).is_file() for report in (played, replayed))


def test_run_path_joins_a_relative_path_to_the_workdir_and_keeps_an_absolute_one(tmp_path):
    """A row an older workdir wrote with an absolute path still opens; a relative one joins the workdir."""
    older = tmp_path / "elsewhere" / "runs" / "a.jsonl"
    assert run_path(tmp_path / "work", "runs/t/a.jsonl") == tmp_path / "work" / "runs" / "t" / "a.jsonl"
    assert run_path(tmp_path / "work", str(older)) == older
    assert stored_run_path(tmp_path / "work", tmp_path / "work" / "runs" / "a.jsonl") == "runs/a.jsonl"
    assert stored_run_path(tmp_path / "work", older) == str(older)
    assert stored_run_path(tmp_path / "work", "") == ""
