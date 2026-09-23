"""The held-out split applied once, where recordings become evidence (D220).

An invented ferry domain: two Runs of one Task, one of them held out. The point of every test here
is that a stage cannot leak by forgetting, that the world keeps the rows a held-out Run needs while
the values only it witnessed are withheld from the writer, and that what the split costs and finds
is counted rather than guessed.
"""

from __future__ import annotations

from kullback.builder import compile_env, replay_evidence
from kullback.gates.tool_runs import body_memorised_values_gate
from kullback.runner.records import RawPtr, Task, ToolCall, ToolSig, Trace, Turn

PTR = RawPtr(file_hash="ferry", sim_index=0)


def _trace(run_id: str, calls: list[ToolCall], said: str = "book me a crossing") -> Trace:
    return Trace(trace_id=run_id, raw_hash="raw", ingest_version="0", source="test",
                 turns=[Turn(idx=0, role="user", content=said, raw_ptr=PTR)],
                 tool_calls=calls, raw_ptr=PTR)


def _call(name: str, call_id: str, args: dict, result: object = None) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, raw_ptr=PTR)


def _ferry_inputs() -> dict:
    seed = _trace("run_seed", [_call("read_crossing", "c_seed", {"crossing_id": "x1"},
                                     {"crossing_id": "x1", "berth": "north"})])
    held = _trace("run_held", [_call("read_crossing", "c_held", {"crossing_id": "x2"},
                                     {"crossing_id": "x2", "berth": "windward"})],
                  said="what berth is the late crossing on")
    return {"traces": [seed, held],
            "tasks": [Task(id="task_ferry", category_id="cat", run_ids=["run_seed", "run_held"])],
            "sigs": [ToolSig(name="read_crossing", kind="read")]}


# --- the seed set -------------------------------------------------------------

def test_a_readmission_from_a_held_out_run_is_blocked_and_counted(workdir):
    """D191 puts a call the Reference replay failed on back over every filter. The seed set is not
    one of them: a held-out Run's failure stays out of the evidence."""
    inputs = _ferry_inputs()
    traces = inputs["traces"]
    after_write = replay_evidence.after_write_calls(traces, set())
    callers = {"read_crossing": {"assistant"}}
    failed = {"read_crossing": {"c_seed": "differs", "c_held": "differs"}}

    calls, _skipped, from_replay = replay_evidence.evidence_calls(
        traces, {"run_seed"}, after_write, callers, failed)

    assert {c.id for c in calls["read_crossing"]} == {"c_seed"}
    assert from_replay == {}, "the seed call was evidence already; nothing was put back over a filter"


# --- the world stays complete, the provenance does not ------------------------

WITNESSES = {"crossings": {"x1": {"berth": ["run_seed"], "crossing_id": ["run_seed", "run_held"]},
                           "x2": {"berth": ["run_held"], "crossing_id": ["run_held"]},
                           "x3": {"berth": []}}}
DB = {"crossings": {"x1": {"crossing_id": "x1", "berth": "north"},
                    "x2": {"crossing_id": "x2", "berth": "windward"},
                    "x3": {"crossing_id": "x3", "berth": "composed"}}}


def test_a_value_only_a_held_out_run_witnessed_is_named_and_a_shared_or_unwitnessed_one_is_not():
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    assert columns == {"crossings": {"x2": ["berth", "crossing_id"]}}
    assert compile_env.holdout_values(DB, columns) == {"x2": "crossings.crossing_id",
                                                       "windward": "crossings.berth"}
    assert "x3" not in columns.get("crossings", {}), "a composed value was never learned from a Run"


def test_the_body_writer_is_shown_the_column_and_not_the_value_it_was_not_given():
    schema = compile_env.EntitySchema(tables=["crossings"])
    columns = compile_env.holdout_columns(WITNESSES, ["run_held"])

    text = compile_env._lookup_rows_text(schema, DB, [], None, "crossings", "x2", holdout=columns)

    assert "windward" not in text, "the value only a held-out Run witnessed is not shown"
    assert '"berth"' in text and compile_env.MASKED in text, "the column is still there"


def test_the_memorised_values_gate_refuses_a_held_out_only_literal_and_leaves_a_seen_one_alone():
    source = ("class Tools:\n"
              "    def read_crossing(self, crossing_id):\n"
              "        return {'berth': 'windward'}\n")
    holdout = compile_env.holdout_values(DB, compile_env.holdout_columns(WITNESSES, ["run_held"]))

    ruling = body_memorised_values_gate(source, sig=ToolSig(name="read_crossing", kind="read"),
                                        holdout_values=holdout)

    assert not ruling.passed
    assert any("crossings.berth" in failure for failure in ruling.failures)
    assert ruling.metrics["holdout_values"] == len(holdout)

    source = ("class Tools:\n"
              "    def read_crossing(self, crossing_id):\n"
              "        return {'berth': 'north'}\n")
    holdout = compile_env.holdout_values(DB, compile_env.holdout_columns(WITNESSES, ["run_held"]))

    ruling = body_memorised_values_gate(source, sig=ToolSig(name="read_crossing", kind="read"),
                                        holdout_values=holdout)

    assert ruling.passed, "the seed Runs witnessed this value, so the writer was shown it"
