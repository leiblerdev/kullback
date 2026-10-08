"""The Spec writes and repairs the whole Verifier; the Examiner's atoms edits reach it by code (D320).

A tiny invented workdir: one Task whose Reference moves one item to a new slot and renames it, both
values said by the user, so its expected end state carries two sourced cells.
"""

from __future__ import annotations

import json

from kullback.ai.provider import TestModel
from kullback.runner.records import Run, Verifier, write_json
from kullback.spec import review as RV
from kullback.spec import writer as W
from kullback.spec.canfail import can_fail, judge_run
from kullback.spec.schema import load_spec
from tests.spec.fixtures import WRITE, check, spec
from tests.spec.test_writer import MOVE, answer, inputs
from tests.spec.test_writer import _workdir as _environment

TURN = "Please move item A1 to slot seven and call it blue."
START = {"items": {"A1": {"item_id": "A1", "slot": "two", "label": "red"},
                   "B2": {"item_id": "B2", "slot": "four", "label": "green"}}}
END = {"items": {"A1": {"item_id": "A1", "slot": "seven", "label": "blue"},
                 "B2": {"item_id": "B2", "slot": "four", "label": "green"}}}
SAID = {"id": "said_slot", "kind": "required",
        "payload": {"kind": "communicate", "text": "slot seven", "id": "said_slot"}}


def _reference() -> Run:
    events = [{"idx": 0, "type": "user_turn", "payload": {"content": TURN}},
              {"idx": 1, "type": "tool_call", "payload": {"id": "c1", "name": "update_item", "kind": "write",
                                                          "args": {"item_id": "A1", "slot": "seven",
                                                                   "label": "blue"}}},
              {"idx": 2, "type": "tool_result", "payload": {"id": "c1", "result": END["items"]["A1"]}},
              {"idx": 3, "type": "model_call", "payload": {"reply": {"content": "Item A1 is in slot seven."}}},
              {"idx": 4, "type": "stop", "payload": {"start_state": START, "end_state": END}}]
    return Run.model_validate({"run_id": "ref", "task_id": "t1", "trace_id": "rec1",
                               "termination_reason": "success", "events": events})


def _workdir(root):
    """A minimal Environment with the Task's Reference on disk the way the Reference stage leaves it."""
    _environment(root)
    (root / "verifiers" / "t1.json").unlink()
    path = root / "runs" / "t1" / "ref.jsonl"
    path.write_text(_reference().model_dump_json() + "\n")
    write_json(root / "references.json", {"t1": {"references": [{"kind": "recording", "run_id": "ref",
                                                                  "trace_id": "rec1"}]}})
    write_json(root / "replays.json", {"t1": {"rec1": {"run_id": "ref", "trace_id": "rec1", "confirmed": True,
                                                       "path": "runs/t1/ref.jsonl"}}})
    return root


def _written(tmp_path):
    root = _workdir(tmp_path)
    W.write_verifier(spec(), root, inputs())
    return root, Verifier.model_validate_json(W.runner_verifier_path(root, "t1").read_text())


def test_write_verifier_on_a_reference_puts_its_two_sourced_cells_in_expected_and_no_write_predicate(tmp_path):
    root, verifier = _written(tmp_path)
    [state] = verifier.expected
    cells = {cell.field: cell for cell in state.cells}
    assert set(cells) == {"slot", "label"}
    assert all(cell.source is not None and cell.source.kind == "user_turn" for cell in cells.values())
    assert not any(atom.payload.get("kind") in {"write_value", "entity_count"} for atom in verifier.atoms)
    assert not any(getattr(atom, "predicate_src", None) for atom in verifier.atoms)
    assert load_spec(root, "t1").end_state == {"reference": "ref", "end_states": 1, "cells": 2, "unsupported": 0,
                                               "forbidden": len(verifier.forbidden), "conduct": 0, "unseen": 0}


def test_a_write_demand_carrying_a_python_predicate_is_refused_and_sent_back_once():
    coded = dict(MOVE, id="d0", predicate_src="lambda run: run.wrote('A1')")
    model = TestModel([answer("demands", [coded]), answer("demands", [dict(MOVE, id="d0")])])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["write_predicate"] == 1 and written.counts["sent_back"] == 1
    assert len(model.calls) == 2
    assert W.WRITE_PREDICATE in json.dumps(model.calls[1]["messages"][-1])
    assert [c.demand.get("demand") for c in written.spec.checks] == [WRITE["demand"]]


def test_an_atoms_edit_is_applied_by_the_spec_kept_on_it_and_ruled_on_by_the_gates(tmp_path):
    root, _ = _written(tmp_path)
    out = RV.apply_review(root, "t1", [{"kind": "atoms", "task_id": "t1", "add": [SAID], "drop": [],
                                        "why": "the user asked to hear the slot"}])
    assert out["applied"] == 1 and out["refused"] == [] and out["round"] == 1
    assert set(out) >= {"passed", "failing", "pending"}
    saved = load_spec(root, "t1")
    assert saved.atom_edits[0]["add"] == [SAID] and saved.round == 1
    verifier = Verifier.model_validate_json(W.runner_verifier_path(root, "t1").read_text())
    assert "said_slot" in {atom.id for atom in verifier.atoms}
    assert verifier.expected and verifier.expected[0].cells, "the end-state gates stay the Spec's"


def test_an_atoms_edit_adding_a_write_or_a_predicate_is_refused_with_the_end_state_reason(tmp_path):
    root, _ = _written(tmp_path)
    write = {"id": "w", "kind": "required", "payload": {"kind": "write_value", "field": "slot"}}
    coded = dict(SAID, id="p", predicate_src="lambda run: True")
    out = RV.apply_review(root, "t1", [{"kind": "atoms", "add": [write]}, {"kind": "atoms", "add": [coded]}])
    assert out["applied"] == 0 and all(why.startswith(W.WRITE_PREDICATE) for why in out["refused"])
    assert load_spec(root, "t1").atom_edits == []


def test_the_loop_stops_at_the_ceiling_with_the_task_pending_and_the_rounds_on_the_spec(tmp_path):
    root = _environment(tmp_path) or tmp_path
    W.write_verifier(spec(checks=[check()]), root, inputs())
    rounds = [RV.apply_review(root, "t1", [{"kind": "atoms", "drop": ["nothing"]}], ceiling=2) for _ in range(3)]
    assert [r.get("round") for r in rounds[:2]] == [1, 2]
    assert rounds[0]["passed"] is False and rounds[0]["pending"] is False
    assert rounds[1]["pending"] is True and load_spec(root, "t1").set_aside == RV.PENDING
    assert rounds[2]["skipped"] == "set aside", "a pending Task takes no further round"
    assert load_spec(root, "t1").round == 2


def test_route_findings_sends_atoms_and_text_edits_to_the_spec_and_keeps_body_edits_for_the_builder(tmp_path):
    root, _ = _written(tmp_path)
    body = {"kind": "body", "path": "env/tools/update_item.py", "call_id": "c1", "column": "slot",
            "recorded": "seven", "replayed": "two", "why": "the body drops the slot"}
    findings = [{"task_id": "t1", "kind": "other", "text": "x", "edits": [{"kind": "atoms", "add": [SAID]}]},
                {"task_id": "t1", "kind": "fidelity", "text": "y", "edits": [body]}]
    kept, rounds = RV.route_findings(root, findings)
    assert [f["edits"] for f in kept] == [[body]]
    assert [r["task_id"] for r in rounds] == ["t1"]
    assert RV.rounds_line(rounds).startswith("spec repairs: 1 edits on 1 Tasks, ")


def test_a_verifier_with_end_state_gates_passes_its_reference_and_fails_the_empty_and_undone_runs(tmp_path):
    _, verifier = _written(tmp_path)
    assert judge_run(verifier, _reference()) == (True, None)
    result = can_fail(verifier, _reference())
    assert result.passed
    assert [row["stage"] for row in result.rows] == ["verifier_empty_run"] + ["verifier_undone_cell"] * 2


TOLD = check("c2", demand={"demand": "say", "text": "slot seven"}, because="move item A1 to slot seven")


def _told(tmp_path):
    """The workdir with a Verifier of two cells and one fact-told atom, i0, made by check c2."""
    root = _workdir(tmp_path)
    W.write_verifier(spec(checks=[check(), TOLD]), root, inputs())
    return root


def _both_copies(root) -> Verifier:
    text = W.runner_verifier_path(root, "t1").read_text()
    assert (root / "spec" / "verifiers" / "t1.json").read_text() == text
    return Verifier.model_validate_json(text)


def test_an_atom_drop_naming_a_check_id_is_refused_with_the_atom_ids_and_more_than_half_is_refused(tmp_path):
    root = _told(tmp_path)
    assert RV.atom_checks(load_spec(root, "t1"), _both_copies(root)) == {"i0": "c2"}
    out = RV.apply_review(root, "t1", [{"kind": "atoms", "drop": ["c2"]}, {"kind": "atoms", "drop": ["i0"]}])
    assert out["applied"] == 0
    assert out["refused"] == ["drop names no atom of the Verifier: c2; atoms are i0", "drops 1 of 1 atoms and adds 0"]
    assert load_spec(root, "t1").atom_edits == [] and [a.id for a in _both_copies(root).atoms] == ["i0"]


def test_cell_drop_and_allow_rewrite_both_verifier_copies_move_the_gate_row_and_outlive_a_rewrite(tmp_path):
    root = _told(tmp_path)
    undone = lambda: [r for r in RV.gate_row(root, load_spec(root, "t1"))["can_fail"]["rows"]  # noqa: E731
                      if r["stage"] == "verifier_undone_cell"]
    assert len(undone()) == 2
    out = RV.apply_review(root, "t1", [
        {"kind": "cell", "task_id": "t1", "table": "items", "row_id": "A1", "field": "label", "action": "drop",
         "why": "no turn says the label"},
        {"kind": "cell", "task_id": "t1", "table": "items", "row_id": "B2", "action": "allow", "why": "any is fine"},
        {"kind": "cell", "task_id": "t1", "table": "nowhere", "action": "allow", "why": "x"}])
    assert out["applied"] == 2 and out["refused"] == ["the Verifier has no table nowhere; tables are items"]
    [state] = _both_copies(root).expected
    assert [c.field for c in state.cells] == ["slot"] and state.allowed == [{"table": "items", "row_id": "B2"}]
    assert _both_copies(root).verifier_version == "spec1.e2" and len(undone()) == 1
    RV.apply_review(root, "t1", [])
    assert [c.field for c in _both_copies(root).expected[0].cells] == ["slot"], "a later round keeps the edits"
    W.write_verifier(load_spec(root, "t1"), root)
    assert [c.field for c in _both_copies(root).expected[0].cells] == ["slot"], "a router rewrite keeps them"


def test_conduct_add_and_remove_rewrite_the_verifier_and_a_remove_of_a_missing_rule_is_refused(tmp_path):
    root = _told(tmp_path)
    add = {"kind": "conduct", "task_id": "t1", "action": "add", "conduct": "confirm_before_write",
           "tool": "update_item", "why": "policy says confirm before any change"}
    kept, _ = RV.route_findings(root, [{"task_id": "t1", "finding_id": "f-1", "edits": [add]}])
    [rule] = _both_copies(root).conduct
    assert kept == [] and rule.kind == "confirm_before_write" and rule.tool == "update_item"
    assert rule.source.kind == "policy" and rule.source.ptr == {"clause": add["why"], "finding": "f-1"}
    out = RV.apply_review(root, "t1", [dict(add, action="remove"), dict(add, action="remove", conduct="handoff")])
    assert out["refused"] == ["the Verifier has no handoff conduct on update_item"]
    assert _both_copies(root).conduct == []


def test_a_reference_edit_promotes_nothing_and_leaves_an_open_ruling_on_the_trust_row(tmp_path):
    root = _told(tmp_path)
    out = RV.apply_review(root, "t1", [{"kind": "reference", "run_id": "ref", "why": "it did what was asked"},
                                       {"kind": "reference", "run_id": "gone", "why": "no such Run"}])
    assert out["applied"] == 1 and out["refused"][0].endswith("gone is not one")
    review = json.loads((root / "examiner" / "reviews.json").read_text())["t1"]
    assert review["outcome"] == RV.REFERENCE_PROPOSED and review["run_id"] == "ref"
    assert RV.gate_row(root, load_spec(root, "t1"))["gates"]["no_open_ruling"] is False
    assert json.loads((root / "references.json").read_text())["t1"]["references"][0]["run_id"] == "ref"
