"""The sandbox replays a gated call's trace prefix first (F50).

The world is an invented widget shelf: one write tool moves a widget to a new status and one read
tool reads a widget back. A recording made after an earlier write of its own trace matches only on
the world that write left, so the gated call runs after the calls its trace made before it.
"""

from __future__ import annotations

import json

from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.builder.env_files import executor
from kullback.gates.bindings import rulings_for
from kullback.runner.records import (
    Column,
    EntitySchema,
    FieldStat,
    RawPtr,
    ToolCall,
    ToolCallError,
    ToolSig,
    as_dict,
)

DB = {"widgets": {"W-1": {"widget_id": "W-1", "status": "new"},
                  "W-2": {"widget_id": "W-2", "status": "new"}}}

SET_BODY = ("row = self.db.widgets[widget_id]\n"
            "row.status = new_status\n"
            "return {\"widget_id\": row.widget_id, \"status\": row.status}\n")
GET_BODY = ("row = self.db.widgets[widget_id]\n"
            "return {\"widget_id\": row.widget_id, \"status\": row.status}\n")


def _schema() -> EntitySchema:
    columns = [Column(table="widgets", name=name, **{"class": "hard"}) for name in ("widget_id", "status")]
    return EntitySchema(tables=["widgets"], columns=columns, id_patterns={"widgets.widget_id": r"^W-\d+$"})


def _sigs() -> list[ToolSig]:
    widget = FieldStat(name="widget_id", types=["str"], optional=False)
    status = FieldStat(name="new_status", types=["str"], optional=False)
    return [ToolSig(name="set_status", description="Move one widget to a new status.",
                    args_fields=[widget, status], kind="write", unclassified=False),
            ToolSig(name="get_widget", description="Read one widget.",
                    args_fields=[widget], kind="read", unclassified=False)]


def _source(set_body: str = SET_BODY) -> str:
    return ce.module_source(_schema(), _sigs(), {"set_status": set_body, "get_widget": GET_BODY})


def _call(call_id: str, trace: str, at: int, name: str, args: dict, result=None, error=None) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, error=error, trace_id=trace,
                    raw_ptr=RawPtr(file_hash="f", sim_index=0, msg_index=at))


WRITE = _call("w1", "t1", 1, "set_status", {"widget_id": "W-1", "new_status": "done"},
              {"widget_id": "W-1", "status": "done"})
READ_AFTER = _call("r1", "t1", 3, "get_widget", {"widget_id": "W-1"}, {"widget_id": "W-1", "status": "done"})
READ_ALONE = _call("r2", "t2", 1, "get_widget", {"widget_id": "W-1"}, {"widget_id": "W-1", "status": "new"})
TABLE = sb.calls_by_trace([WRITE, READ_AFTER, READ_ALONE])


def _status(result: dict) -> str:
    assert result["ok"], result
    return result["value"]["status"]


def test_a_read_after_a_write_in_its_trace_replays_the_recorded_answer_only_with_the_prefix(tmp_path):
    without = sb.Sandbox(_source(), DB, tmp_path / "plain").run([READ_AFTER])
    with_prefix = sb.Sandbox(_source(), DB, tmp_path / "prefix", trace_calls=TABLE).run([READ_AFTER])
    assert _status(without[0]) == "new"
    assert _status(with_prefix[0]) == "done"


def test_the_same_read_on_a_trace_without_the_write_replays_the_starting_state(tmp_path):
    results = sb.Sandbox(_source(), DB, tmp_path, trace_calls=TABLE).run([READ_AFTER, READ_ALONE])
    assert [_status(result) for result in results] == ["done", "new"]


def test_the_memoised_state_key_differs_between_the_two_prefixes(tmp_path):
    box = sb.Sandbox(_source(), DB, tmp_path, trace_calls=TABLE)
    assert box.state_key(READ_AFTER) != box.state_key(READ_ALONE)
    assert box.state_key(READ_ALONE) == sb.Sandbox(_source(), DB, tmp_path / "plain").state_key(READ_ALONE)


def test_the_replay_gate_passes_both_traces_with_the_prefix_and_fails_the_later_read_without(tmp_path):
    reads = [READ_AFTER, READ_ALONE]
    plain = sb.gate_replay_fidelity(sb.Sandbox(_source(), DB, tmp_path / "plain"), reads, _schema())
    prefixed = sb.gate_replay_fidelity(sb.Sandbox(_source(), DB, tmp_path / "prefix", trace_calls=TABLE),
                                       reads, _schema())
    assert not plain.passed
    assert prefixed.passed


def test_a_prefix_call_that_raises_is_skipped_and_the_gated_call_still_runs(tmp_path):
    missing = _call("w0", "t1", 0, "set_status", {"widget_id": "W-9", "new_status": "done"},
                    {"widget_id": "W-9", "status": "done"})
    table = sb.calls_by_trace([missing, WRITE, READ_AFTER])
    results = sb.Sandbox(_source(), DB, tmp_path, trace_calls=table).run([READ_AFTER])
    assert _status(results[0]) == "done"


def test_a_call_the_recording_refused_is_left_out_of_the_prefix(tmp_path):
    refused = _call("w1", "t1", 1, "set_status", {"widget_id": "W-1", "new_status": "done"},
                    error=ToolCallError(**{"class": "invalid_arguments", "payload": "no"}))
    box = sb.Sandbox(_source(), DB, tmp_path, trace_calls=sb.calls_by_trace([refused, READ_AFTER]))
    assert box.prefix_of(READ_AFTER) == []
    assert _status(box.run([READ_AFTER])[0]) == "new"


def test_a_prefix_calling_a_tool_the_module_does_not_hold_is_skipped(tmp_path):
    other = _call("x1", "t1", 2, "archive_widget", {"widget_id": "W-1"}, {"archived": True})
    table = sb.calls_by_trace([WRITE, other, READ_AFTER])
    results = sb.Sandbox(_source(), DB, tmp_path, trace_calls=table).run([READ_AFTER])
    assert _status(results[0]) == "done"


def test_a_second_write_is_diffed_against_the_world_its_prefix_left(tmp_path):
    again = _call("w2", "t1", 5, "set_status", {"widget_id": "W-1", "new_status": "shipped"},
                  {"widget_id": "W-1", "status": "shipped"})
    table = sb.calls_by_trace([WRITE, again])
    diff = sb.Sandbox(_source(), DB, tmp_path, trace_calls=table).run_diff([again])
    row = diff[0]["changed"]["widgets"]["W-1"]
    assert row["before"]["status"] == "done" and row["after"]["status"] == "shipped"


def test_the_prefix_is_the_earlier_calls_of_the_trace_in_recorded_order_whatever_the_input_order(tmp_path):
    later = _call("w3", "t1", 2, "set_status", {"widget_id": "W-2", "new_status": "held"},
                  {"widget_id": "W-2", "status": "held"})
    box = sb.Sandbox(_source(), DB, tmp_path, trace_calls={"t1": [READ_AFTER, later, WRITE]})
    assert [call.id for call in box.prefix_of(READ_AFTER)] == ["w1", "w3"]
    assert box.prefix_of(WRITE) == []
    assert box.prefix_of(READ_ALONE) == []


# --- the live write gate: rulings_for with the session's executor over a workdir ---

def _workdir(tmp_path):
    work, env = tmp_path / "work", tmp_path / "work" / "env"
    (work / "traces").mkdir(parents=True)
    (env / "tools").mkdir(parents=True)
    (work / "schema.json").write_text(json.dumps(as_dict(_schema())), encoding="utf-8")
    (work / "tool_sigs.json").write_text(json.dumps([as_dict(sig) for sig in _sigs()]), encoding="utf-8")
    (work / "db.json").write_text(json.dumps(DB), encoding="utf-8")
    for name, calls in (("t1", [WRITE, READ_AFTER]), ("t2", [READ_ALONE])):
        (work / "traces" / f"{name}.json").write_text(
            json.dumps({"tool_calls": [as_dict(call) for call in calls]}), encoding="utf-8")
    (env / "tools" / "set_status.py").write_text(
        "def set_status(self, widget_id, new_status):\n" + _indented(SET_BODY), encoding="utf-8")
    (env / "tools" / "get_widget.py").write_text(
        "def get_widget(self, widget_id):\n" + _indented(GET_BODY), encoding="utf-8")
    return work, env


def _indented(body: str) -> str:
    return "".join("    " + line + "\n" for line in body.splitlines())


def _replay(rulings) -> bool:
    return next(ruling for ruling in rulings if ruling.stage == "replay_fidelity").passed


def test_the_write_gate_passes_a_read_of_a_row_its_trace_wrote_only_because_the_prefix_ran(tmp_path):
    work, env = _workdir(tmp_path)
    execute = executor(work, env)

    def without_table(source, calls, db):
        return execute(source, calls, db)

    without_table.render = execute.render
    assert _replay(rulings_for(env, "tools/get_widget.py", work, execute=execute))
    assert not _replay(rulings_for(env, "tools/get_widget.py", work, execute=without_table))
