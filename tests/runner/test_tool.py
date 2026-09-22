"""The runner as a tool: version, replay names the first differing column, run verdicts a Run."""

from __future__ import annotations

import json
from pathlib import Path

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner import tool
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, Verifier, write_json
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


def test_replay_names_the_first_differing_column(tmp_path):
    """A body returning the wrong value on one recorded call is reported by its column."""
    root = _env_with_trace(tmp_path / "env")
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


def test_replay_confirms_where_the_body_answers_straight(tmp_path):
    root = _env_with_trace(tmp_path / "env")
    report = tool.replay(root, "widget_task", workdir=tmp_path / "work")
    assert report.confirmed, report.reasons
    assert report.fidelity == 1.0
    assert all(call.first_differing_column is None for call in report.calls)


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
    assert Path(report.path).is_file()
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
    assert Path(report.path).is_file() and Path(report.path).parent.name == "probes"
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
    assert Path(first.path) == tmp_path / "work" / "runs" / "widget_task" / "synth-1.jsonl"
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
    assert all(Path(report.path).is_file() for report in reports)
    assert all("user_end" in report.as_dict() for report in reports)


def test_the_shared_retail_fixture_loads():
    """The small shared fixture in tests/fixtures parses: simulations beside their tasks."""
    body = json.loads((Path(__file__).resolve().parents[1] / "fixtures"
                       / "tau2_retail_small.json").read_text(encoding="utf-8"))
    assert len(body["simulations"]) >= 1
    wanted = {sim["task_id"] for sim in body["simulations"]}
    assert wanted <= {task["id"] for task in body["tasks"]}
