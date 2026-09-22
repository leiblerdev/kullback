"""The Builder's domain tools: replay rows call by call, reroll rows run by run, examine delegates."""

from __future__ import annotations

import asyncio
import json

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn
from tests.builder.session_fixtures import reply
from tests.episode.invented import write_env

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)


def _workdir(tmp_path):
    root = write_env(tmp_path / "work")
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
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
    trace = Trace(trace_id="rec1", raw_hash="r" * 64, ingest_version="1", source="test",
                  turns=turns, tool_calls=calls, raw_ptr=PTR)
    (traces / "rec1.json").write_text(json.dumps(trace.model_dump(mode="json", by_alias=True)),
                                      encoding="utf-8")
    return root


def _run(tool, arguments):
    return asyncio.run(tool.run(arguments))


def test_replay_answers_each_recorded_call_with_both_values(tmp_path):
    root = _workdir(tmp_path)
    (replay,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "replay"]
    result = _run(replay, {"task_id": "widget_task"})
    assert not result.is_error, result.content
    body = json.loads(result.content) if result.content.startswith("{") else None
    assert "c1" in result.content and "c2" in result.content
    assert "describe_widget" in result.content and "rename_widget" in result.content
    assert "fidelity" in result.content
    assert body is None  # the rendered text is rows, the full rows ride in details
    assert result.details["fidelity"] == 1.0
    assert result.details["total"] == 2
    first = result.details["rows"][0]
    assert (first["call_id"], first["tool"]) == ("c1", "describe_widget")
    assert first["recorded"] and first["ours"]


def test_replay_text_caps_rows_at_twenty_then_counts(tmp_path):
    assert domain_tools_mod.ROWS_SHOWN == 20
    root = _workdir(tmp_path)
    (replay,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "replay"]
    result = _run(replay, {"task_id": "widget_task"})
    assert "over 2 recorded calls" in result.content


def test_run_reports_each_fresh_run_with_verdict_routes_diff_and_spend(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel([
        reply(None, ("describe_widget", {"widget_id": "w1"})),
        reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
        reply("Done."),
    ])
    (run,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model) if tool.name == "run"]
    result = _run(run, {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    assert len(result.details["runs"]) == 1
    row = result.details["runs"][0]
    assert row["run_id"] == "widget_task-0"
    assert row["routes"]
    assert isinstance(row["world_diff"], dict)
    assert isinstance(row["spend"], dict)
    assert "widget_task-0" in result.content


def test_run_without_a_model_is_an_error_result(tmp_path):
    root = _workdir(tmp_path)
    (run,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "run"]
    result = _run(run, {"task_id": "widget_task", "count": 1})
    assert result.is_error


def test_examine_calls_the_function_the_session_passed_in(tmp_path):
    root = _workdir(tmp_path)
    seen = {}

    def examine_fn(workdir, task_ids):
        seen["workdir"] = workdir
        seen["task_ids"] = task_ids
        return {"summary": "two findings open", "findings": [{"text": "the body misses a call"}]}

    (examine,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, examine_fn=examine_fn)
                  if tool.name == "examine"]
    result = _run(examine, {"task_ids": ["widget_task"]})
    assert not result.is_error, result.content
    assert seen["task_ids"] == ["widget_task"]
    assert "the body misses a call" in result.content


RENAME_BODY = (
    "row = self.db.widgets.get(widget_id)\n"
    "if row is None:\n"
    "    raise KeyError(widget_id)\n"
    "row.label = label\n"
    'return {"widget_id": widget_id, "label": label}'
)


def test_derive_writes_tool_files_and_the_read_surface(tmp_path):
    root = _workdir(tmp_path)
    from kullback.runner.records import write_json as _write_json
    _write_json(root / "bodies.json", {"rename_widget": RENAME_BODY})
    (derive,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root) if tool.name == "derive_world"]
    result = _run(derive, {})
    assert not result.is_error, result.content
    assert (root / "env" / "tools" / "rename_widget.py").is_file()
    assert (root / "env" / "calls" / "rename_widget.jsonl").is_file()
    assert "rename_widget" in result.content
