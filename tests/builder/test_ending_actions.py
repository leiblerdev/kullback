"""D282: a tool whose recorded effect is to end the Run with no write leaves a row, and a write atom demands it.

The domain is invented: a help desk where a clerk looks a ticket up, may close it, and may pass the
conversation to a supervisor. Nothing names a corpus, tool or table of any customer.
"""

from __future__ import annotations

import json
from pathlib import Path

from kullback.builder import compile_env, env_files, mine
from kullback.examiner import derive
from kullback.runner import route
from kullback.runner.canon import CanonRules
from kullback.runner.records import Event, RawPtr, Run, ToolCall, Trace, as_dict, write_json
from kullback.runner.target import check_run
from kullback.runner.world import BuiltEnvironment, clock
from kullback.runner.world.loading import load_toolkit

PTR = RawPtr(file_hash="f" * 64, sim_index=0, msg_index=0)
HANDOFF = "pass_to_supervisor"
ANSWER = "Passed on. A supervisor will pick this up."
TICKET = {"ticket_id": "T1", "state": "open", "subject": "printer"}


def _trace(trace_id: str, *, hand_off: bool, then_look: bool = False) -> Trace:
    calls = [ToolCall(id=f"{trace_id}-1", name="look_ticket", args={"ticket_id": "T1"}, result=TICKET,
                      raw_ptr=PTR, trace_id=trace_id)]
    if hand_off:
        calls.append(ToolCall(id=f"{trace_id}-2", name=HANDOFF, args={"summary": f"case {trace_id}"},
                              result=ANSWER, raw_ptr=PTR, trace_id=trace_id))
    else:
        calls.append(ToolCall(id=f"{trace_id}-2", name="close_ticket", args={"ticket_id": "T1"},
                              result=dict(TICKET, state="closed"), raw_ptr=PTR, trace_id=trace_id))
    if then_look:
        calls.append(ToolCall(id=f"{trace_id}-3", name="look_ticket", args={"ticket_id": "T1"},
                              result=TICKET, raw_ptr=PTR, trace_id=trace_id))
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 tool_calls=calls, raw_ptr=PTR)


def _corpus() -> list[Trace]:
    return ([_trace(f"h{n}", hand_off=True) for n in range(4)]
            + [_trace(f"c{n}", hand_off=False) for n in range(3)])


def _mined(traces: list[Trace]):
    sigs = mine.mine_tools(traces)
    schema = mine.mine_schema(traces, write_tools=sorted(s.name for s in sigs if s.kind == "write"))
    return sigs, schema


def test_a_tool_that_ends_its_runs_and_writes_nothing_becomes_a_write_with_an_actions_table():
    traces = _corpus()
    sigs, schema = _mined(traces)
    assert next(s for s in sigs if s.name == HANDOFF).kind != "write"
    moved = compile_env.record_ending_actions(traces, sigs, schema)
    assert moved == [HANDOFF]
    assert next(s for s in sigs if s.name == HANDOFF).kind == "write"
    table = compile_env.action_tools(schema)[HANDOFF]
    assert table in schema.tables
    assert {c.name for c in schema.columns if c.table == table} == {"action_id", "tool", "args", "turn"}
    assert compile_env.record_ending_actions(traces, sigs, schema) == []  # idempotent


def test_a_tool_called_before_the_end_or_answering_rows_is_not_an_ending_tool():
    traces = [_trace(f"h{n}", hand_off=True, then_look=True) for n in range(4)]
    sigs, _ = _mined(traces)
    assert compile_env.ending_tools(traces, sigs) == []
    assert "close_ticket" not in compile_env.ending_tools(_corpus(), _mined(_corpus())[0])


def _final_lookup(trace_id: str) -> Trace:
    """A Trace that looks the ticket up, then closes on a scalar total keyed by the id it just saw."""
    calls = [ToolCall(id=f"{trace_id}-1", name="look_ticket", args={"ticket_id": "T1"}, result=TICKET,
                      raw_ptr=PTR, trace_id=trace_id),
             ToolCall(id=f"{trace_id}-2", name="ticket_total", args={"ticket_id": "T1"}, result=2,
                      raw_ptr=PTR, trace_id=trace_id)]
    return Trace(trace_id=trace_id, raw_hash="r" * 64, ingest_version="1", source="test",
                 tool_calls=calls, raw_ptr=PTR)


def test_a_final_scalar_lookup_keyed_by_an_id_from_the_world_is_not_an_ending_tool():
    traces = [_final_lookup(f"s{n}") for n in range(4)] + [_trace(f"h{n}", hand_off=True) for n in range(4)]
    sigs, schema = _mined(traces)
    assert next(s for s in sigs if s.name == "ticket_total").kind != "write"
    assert compile_env.ending_tools(traces, sigs) == [HANDOFF], "the free-text hand-off still ends the Run"
    assert "ticket_total" not in compile_env.record_ending_actions(traces, sigs, schema)


def test_a_final_lookup_keyed_by_an_id_of_the_starting_state_never_returned_before_is_not_an_ending_tool():
    other = dict(TICKET, ticket_id="T2")
    shown = [ToolCall(id="w-1", name="look_ticket", args={"ticket_id": "T2"}, result=other, raw_ptr=PTR,
                      trace_id="w")]
    first = [Trace(trace_id=f"u{n}", raw_hash="r" * 64, ingest_version="1", source="test", raw_ptr=PTR,
                   tool_calls=[ToolCall(id=f"u{n}-1", name="ticket_total", args={"ticket_id": "T2"},
                                        result=2, raw_ptr=PTR, trace_id=f"u{n}")])
             for n in range(4)]
    traces = first + [Trace(trace_id="w", raw_hash="r" * 64, ingest_version="1", source="test",
                            raw_ptr=PTR, tool_calls=shown)] + [_trace(f"h{n}", hand_off=True) for n in range(4)]
    sigs, _ = _mined(traces)
    assert compile_env.ending_tools(traces, sigs) == [HANDOFF]
    assert compile_env.ending_tools(first, _mined(first)[0], world={"tickets": {"T2": other}}) == []
    assert compile_env.ending_tools(first, _mined(first)[0], world={}) == ["ticket_total"]


def test_a_tool_the_miner_classified_as_a_read_is_not_an_ending_tool():
    traces = _corpus()
    sigs, _ = _mined(traces)
    handoff = next(s for s in sigs if s.name == HANDOFF)
    handoff.kind, handoff.unclassified = "read", False
    assert compile_env.ending_tools(traces, sigs) == []


def _workdir(root: Path) -> Path:
    traces = _corpus()
    sigs, schema = _mined(traces)
    compile_env.record_ending_actions(traces, sigs, schema)
    write_json(root / "schema.json", as_dict(schema))
    write_json(root / "tool_sigs.json", [as_dict(s) for s in sigs])
    write_json(root / "db.json", {"tickets": {"T1": TICKET}})
    write_json(root / "bodies.json", {HANDOFF: f"return {ANSWER!r}",
                                      "look_ticket": "return self.db.tickets[ticket_id].model_dump()"})
    env_files.explode(root)
    env_files.regenerate(root)
    return root


def _router(root: Path) -> route.Router:
    db = json.loads((root / "db.json").read_text())
    toolkit = load_toolkit(BuiltEnvironment(root).toolkit_source(), db)
    clock.start_context(toolkit, 0, None)
    return route.Router(env_tools_module=toolkit, starting_state=db)


def test_the_compiled_ending_tool_writes_its_row_and_answers_the_recorded_text_byte_for_byte(tmp_path):
    root = _workdir(tmp_path)
    router = _router(root)
    router.route("look_ticket", {"ticket_id": "T1"})
    outcome = router.route(HANDOFF, {"summary": "printer jam, needs a supervisor"})
    assert outcome.error is None and outcome.route == "code"
    assert json.dumps(outcome.result) == json.dumps(ANSWER)
    table = compile_env.action_tools(BuiltEnvironment(root).schema)[HANDOFF]
    assert router.world()[table] == {"1": {"action_id": "1", "tool": HANDOFF, "turn": 2,
                                           "args": {"summary": "printer jam, needs a supervisor"}}}


def test_the_tool_file_of_an_ending_tool_says_the_harness_writes_its_row(tmp_path):
    root = _workdir(tmp_path)
    text = (root / "env" / "tools" / f"{HANDOFF}.py").read_text()
    assert "The harness records each call as a row" in text
    assert "The harness records" not in (root / "env" / "tools" / "look_ticket.py").read_text()


def _run(run_id: str, events: list[dict]) -> Run:
    events = events + [{"type": "stop", "payload": {"start_state": {"tickets": {"T1": TICKET}}}}]
    return Run(run_id=run_id, task_id="t1", termination_reason="success",
               events=[Event(idx=n, type=e["type"], payload=e["payload"]) for n, e in enumerate(events)])


def test_an_empty_run_fails_a_verifier_derived_from_a_reference_that_handed_off():
    traces = _corpus()
    sigs, schema = _mined(traces)
    compile_env.record_ending_actions(traces, sigs, schema)
    writes = {s.name for s in sigs if s.kind == "write"}
    reference = _run("ref", [
        {"type": "user_turn", "payload": {"content": "My printer ticket T1 is stuck, I need a person."}},
        {"type": "tool_call", "payload": {"id": "c1", "name": HANDOFF, "args": {"summary": "stuck"},
                                          "kind": "write"}},
        {"type": "tool_result", "payload": {"id": "c1", "result": ANSWER}},
        {"type": "model_call", "payload": {"reply": {"content": "A supervisor will pick this up."}}},
    ])
    empty = _run("empty", [
        {"type": "user_turn", "payload": {"content": "My printer ticket T1 is stuck, I need a person."}},
        {"type": "model_call", "payload": {"reply": {"content": "Sorry to hear that."}}},
    ])
    verifier = derive.derive_verifier("t1", reference, [], CanonRules(), write_tools=writes)
    assert any(a.kind == "required" and (a.target or {}).get("tool") == HANDOFF for a in verifier.atoms)
    assert check_run(verifier, reference, CanonRules(), write_tools=writes)[0]
    assert not check_run(verifier, empty, CanonRules(), write_tools=writes)[0]
