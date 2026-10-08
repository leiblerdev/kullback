"""Gates then weighted reward (D329): pass is every gate item holding, the score is 0 past a failed gate and
the weighted mean past the gates, and the sanity item fails a Run that changed a row nobody declared."""

from __future__ import annotations

from kullback.runner.canon import CanonRules
from kullback.runner.records import Atom, EndState, ExpectedCell, Verifier, as_dict
from kullback.runner.target import canon_fn
from kullback.runner.verdict import item_counts, verdict
from kullback.spec.compile import compile_spec
from kullback.spec.schema import Check, Spec, SpecIntent

START = {"items": {"A1": {"status": "open", "note": "x"}, "B2": {"status": "open", "note": "y"}}}


def _run(end, said=""):
    events = [{"idx": 0, "type": "user_turn", "payload": {"content": "Please close item A1."}}]
    if said:
        events.append({"idx": 1, "type": "model_call", "payload": {"content": said}})
    events.append({"idx": len(events), "type": "stop", "payload": {"start_state": START, "end_state": end}})
    return {"run_id": "r", "events": events}


def _end(**changes):
    end = {"items": {key: dict(row) for key, row in START["items"].items()}}
    for key, value in changes.items():
        row, field = key.split("__")
        end["items"][row][field] = value
    return end


def _closed(value="closed"):
    return EndState(cells=[ExpectedCell(table="items", row_id="A1", field="status", value=value)])


def _said(atom_id, text, **fields):
    return Atom(id=atom_id, kind="communicate", predicate_src=f'communicated("{text}")', **fields)


def _items(out):
    return {item.id: item.holds for item in out.items}


def test_a_run_reaching_the_end_state_with_nothing_else_moved_passes_and_scores_one():
    out = verdict(_run(_end(A1__status="closed")), Verifier(task_id="t", expected=[_closed()]))
    assert out.passed and out.score == 1.0
    assert _items(out) == {"state:items.A1.status": True, "sanity": True}


def test_a_stray_write_beside_the_right_end_state_fails_the_sanity_item_and_scores_zero():
    out = verdict(_run(_end(A1__status="closed", B2__note="z")), Verifier(task_id="t", expected=[_closed()]))
    assert not out.passed and out.score == 0.0
    assert out.failing_atom == "gate:sanity:collateral:items.B2.note"
    assert _items(out) == {"state:items.A1.status": True, "sanity": False}


def test_a_run_matching_any_one_of_several_end_states_passes_on_that_one():
    verifier = Verifier(task_id="t", expected=[_closed("archived"), _closed("closed")])
    out = verdict(_run(_end(A1__status="closed")), verifier)
    assert out.passed and out.score == 1.0 and _items(out)["state:items.A1.status"] is True
    assert not verdict(_run(_end(A1__status="gone")), verifier).passed


def test_an_alternative_whose_gates_hold_is_read_before_a_heavier_one_whose_gate_fails():
    gated = EndState(cells=[ExpectedCell(table="items", row_id="A1", field="status", value=1, gate=True),
                            ExpectedCell(table="items", row_id="A1", field="note", value=1, gate=False)])
    heavier = EndState(cells=[ExpectedCell(table="items", row_id="A1", field="status", value=2, gate=True),
                              ExpectedCell(table="items", row_id="A1", field="note", value=0, gate=False,
                                           weight=10.0)])
    end = _end(A1__status=1, A1__note=0)
    out = verdict(_run(end), Verifier(task_id="t", expected=[gated, heavier]))
    assert out.passed and _items(out)["state:items.A1.status"] is True


def test_a_scored_new_row_count_fails_its_item_not_the_run():
    state = EndState(cells=[ExpectedCell(table="items", row_id="A1", field="status", value="closed")],
                     new_rows=[{"table": "items", "where": {"status": "closed"}, "count": 1,
                                "item": "n1", "gate": False, "weight": 2.0}])
    out = verdict(_run(_end(A1__status="closed")), Verifier(task_id="t", expected=[state]))
    assert out.passed is True
    assert _items(out)["new_rows:items:1"] is False
    assert out.score == 1 / 2

def test_a_scored_item_that_fails_lowers_the_score_and_never_the_pass():
    verifier = Verifier(task_id="t", expected=[_closed()], atoms=[_said("say", "closed", gate=False)])
    quiet = verdict(_run(_end(A1__status="closed")), verifier)
    assert quiet.passed and quiet.class_ == "pass"
    assert quiet.score == 2 / 3  # the cell and the sanity item held, the one scored item did not
    assert "scored_fail:say" in quiet.notes
    told = verdict(_run(_end(A1__status="closed"), said="Item A1 is closed now."), verifier)
    assert told.passed and told.score == 1.0


def test_the_gate_items_hold_at_least_half_the_weight_however_many_scored_items_fail():
    atoms = [_said(f"say{n}", f"word{n}", gate=False) for n in range(4)]
    out = verdict(_run(_end(A1__status="closed")), Verifier(task_id="t", expected=[_closed()], atoms=atoms))
    assert out.passed and out.score == 0.5


def test_weights_move_the_score_past_the_gates():
    atoms = [_said("heavy", "closed", gate=False, weight=3.0), _said("light", "absent", gate=False)]
    out = verdict(_run(_end(A1__status="closed"), said="It is closed."),
                  Verifier(task_id="t", expected=[_closed()], atoms=atoms))
    assert out.passed and out.score == (4 + 3) / (4 + 4)  # the two gates scale up to the scored weight 4


def test_a_scored_judge_item_with_no_judge_result_masks_the_score_and_keeps_the_pass():
    judged = Atom(id="j", kind="communicate", judge=True, gate=False)
    verifier = Verifier(task_id="t", expected=[_closed()], atoms=[judged])
    out = verdict(_run(_end(A1__status="closed")), verifier)
    assert out.passed and out.score is None and "score_masked:j" in out.notes
    assert verdict(_run(_end(A1__status="closed")), verifier, judge_results={"j": {"verdict": "pass"}}).score == 1.0
    assert verdict(_run(_end(A1__status="closed")), verifier, judge_results={"j": {"verdict": "fail"}}).score == 2 / 3


def test_a_gate_nobody_can_settle_leaves_no_pass_and_no_score_never_a_false_zero():
    verifier = Verifier(task_id="t", expected=[_closed()], atoms=[Atom(id="j", kind="required", judge=True)])
    out = verdict(_run(_end(A1__status="closed")), verifier)
    assert out.class_ == "not_verdicted" and out.score is None


def test_an_atom_left_at_its_kind_default_dumps_as_before_and_a_new_kind_brings_its_gate():
    plain = Atom(id="a", kind="allowed")
    assert "gate" not in as_dict(plain) and "weight" not in as_dict(plain) and plain.gate is False
    assert plain.model_copy(update={"kind": "required"}).gate is True
    scored = Atom(id="b", kind="communicate", gate=False, weight=2.0)
    assert as_dict(scored)["gate"] is False and Atom.model_validate(as_dict(scored)) == scored
    assert scored.model_copy(update={"kind": "question"}).gate is False


def test_a_checks_gate_and_weight_reach_every_atom_it_compiles_to():
    intent = "The item is closed."
    check = Check(id="c1", kind="communicate", demand={"demand": "say", "text": "closed"}, because=intent,
                  tier="important", gate=False, weight=0.5, policy_line="close items on request")
    spec = Spec(task_id="t", intent=SpecIntent(task_id="t", text=intent), checks=[check])
    atoms = compile_spec(spec, [], canon_fn(CanonRules())).verifier.atoms
    assert atoms and all(atom.gate is False and atom.weight == 0.5 for atom in atoms)
    assert Check.model_validate(as_dict(check)) == check
    assert "gate" not in as_dict(check.model_copy(update={"gate": True}))


def test_item_counts_name_the_gate_and_scored_items_of_a_verifier_with_the_sanity_item():
    verifier = Verifier(task_id="t", expected=[_closed()],
                        atoms=[_said("say", "closed", gate=False), Atom(id="ok", kind="allowed")])
    assert item_counts(verifier) == {"gate": 2, "scored": 1, "end_states": 1}
