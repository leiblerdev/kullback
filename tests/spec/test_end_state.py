"""A Task's gates from its Spec demands: refusals and no-writes as forbidden writes, unseen conduct counted."""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.records import Run, Verifier
from kullback.runner.verdict import _gates as verdict_gates
from kullback.spec.end_state import gates_of
from kullback.spec.writer import end_state_counts
from tests.spec.fixtures import check, fn, reference_run, spec

WRITES = {"update_item", "delete_item"}
REFUSE = {"demand": "refuse"}


def _gates(tmp_path, *demands, reference=None):
    checks = [check(f"c{n}", demand=demand) for n, demand in enumerate(demands)]
    return gates_of(tmp_path, spec(checks), canon=fn, reference=reference or reference_run(), write_tools=WRITES)


def test_a_refuse_with_no_recorded_refusal_forbids_each_write_tool_the_reference_did_not_call(tmp_path):
    gates = _gates(tmp_path, REFUSE)
    added = [rule for rule in gates.forbidden if rule.source.kind == "user_turn"]
    assert [(rule.kind, rule.tool, rule.row_id) for rule in added] == [("write", "delete_item", None)]
    assert gates.conduct == [] and gates.unsupported == 0


def test_a_no_write_naming_a_tool_and_row_forbids_that_tool_on_that_row(tmp_path):
    gates = _gates(tmp_path, {"demand": "no_write", "tool": "delete_item", "entity": "B2"})
    assert [(rule.tool, rule.row_id) for rule in gates.forbidden if rule.source.kind == "user_turn"] == \
        [("delete_item", "B2")]


def test_without_a_reference_a_no_write_forbids_every_write_tool_and_the_empty_run_passes_it(tmp_path):
    gates = gates_of(tmp_path, spec([check(demand={"demand": "no_write"})]), canon=fn, write_tools=WRITES)
    assert sorted(rule.tool for rule in gates.forbidden) == sorted(WRITES)
    run = reference_run()
    deleting = run.model_copy(update={"events": [e.model_copy(update={"payload": dict(e.payload, name="delete_item")})
                                                 if e.payload.get("name") == "update_item" else e
                                                 for e in run.events]})
    empty = Run(run_id="e", task_id="t1", termination_reason="success", events=run.events[:1])
    verifier = Verifier(task_id="t1", atoms=[], forbidden=gates.forbidden)
    assert verdict_gates(verifier, AtomContext(deleting, fn, WRITES))[0] is not None
    assert verdict_gates(verifier, AtomContext(empty, fn, WRITES)) == (None, None)


def test_a_handoff_or_confirmation_on_a_tool_the_reference_never_called_is_dropped_and_counted(tmp_path):
    gates = _gates(tmp_path, {"demand": "handoff", "tool": "call_person"},
                   {"demand": "ask", "confirm_tool": "delete_item"}, {"demand": "ask", "confirm_tool": "update_item"})
    assert [(rule.kind, rule.tool) for rule in gates.conduct] == [("confirm_before_write", "update_item")]
    assert gates.unseen == 2 and end_state_counts(gates)["unseen"] == 2
