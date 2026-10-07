"""The gates of a Spec in the older demand grammar, written with no Reference: refusals and no-writes
as forbidden writes, confirmations and hand-offs as conduct."""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.records import Run, Verifier
from kullback.runner.verdict import _gates as verdict_gates
from kullback.spec.end_state import gates_of
from tests.spec.fixtures import check, fn, reference_run, spec

WRITES = {"update_item", "delete_item"}


def _gates(*demands):
    return gates_of(spec([check(f"c{n}", demand=demand) for n, demand in enumerate(demands)]), WRITES)


def test_a_refuse_forbids_each_write_tool_from_the_users_turn_and_names_no_end_state():
    gates = _gates({"demand": "refuse"})
    assert sorted((rule.kind, rule.tool, rule.row_id) for rule in gates.forbidden) == [
        ("write", "delete_item", None), ("write", "update_item", None)]
    assert {rule.source.kind for rule in gates.forbidden} == {"user_turn"}
    assert gates.expected == [] and gates.conduct == []


def test_a_no_write_naming_a_tool_and_row_forbids_that_tool_on_that_row():
    gates = _gates({"demand": "no_write", "tool": "delete_item", "entity": "B2"})
    assert [(rule.tool, rule.row_id) for rule in gates.forbidden] == [("delete_item", "B2")]


def test_a_no_write_fails_a_deleting_run_and_the_empty_run_passes_it():
    gates = _gates({"demand": "no_write"})
    run = reference_run()
    deleting = run.model_copy(update={"events": [e.model_copy(update={"payload": dict(e.payload, name="delete_item")})
                                                 if e.payload.get("name") == "update_item" else e
                                                 for e in run.events]})
    empty = Run(run_id="e", task_id="t1", termination_reason="success", events=run.events[:1])
    verifier = Verifier(task_id="t1", atoms=[], forbidden=gates.forbidden)
    assert verdict_gates(verifier, AtomContext(deleting, fn, WRITES))[0] is not None
    assert verdict_gates(verifier, AtomContext(empty, fn, WRITES)) == (None, None)


def test_a_handoff_and_a_confirmation_become_conduct_whatever_any_reference_called():
    gates = _gates({"demand": "handoff", "tool": "call_person"}, {"demand": "ask", "confirm_tool": "delete_item"})
    assert [(rule.kind, rule.tool) for rule in gates.conduct] == [("handoff", "call_person"),
                                                                  ("confirm_before_write", "delete_item")]
