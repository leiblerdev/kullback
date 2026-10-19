"""A said amount meets every spelling of it (D427); each check is said once, or refused (D428).

The world is the invented one of tests/spec/test_items.py: two items in slots, a notes table.
"""

from __future__ import annotations

from kullback.runner.records import Verifier
from kullback.runner.verdict import verdict
from kullback.spec import items as I
from kullback.spec.compile import cell_clashes, implies, merged_atoms, merged_cells, merged_new_rows, new_row_overlaps
from tests.spec.test_items import STATE, _run, world


def _item(item_id: str, kind: str, **fields) -> dict:
    return {"id": item_id, "kind": kind, "gate": True, "weight": 1.0, "fact_ids": [], "policy_line": None, **fields}


def _row(item_id: str, expect: dict, row: str = "A1", kind: str = "row_is", **fields) -> dict:
    return _item(item_id, kind, table="items", row=row, find={"item_id": row}, expect=expect, **fields)


def _note(item_id: str, where: dict, **fields) -> dict:
    return _item(item_id, "row_new", table="notes", where=where, count=1, **fields)


def _compiled(items: list[dict]) -> tuple[Verifier, list[str]]:
    """The items compiled as the writer's verifier_of compiles them: cells and atoms folded."""
    states, cells = merged_cells(I.end_states_of(items, world()))
    atoms, folded = merged_atoms([I.atom_of(item) for item in items])
    return Verifier(task_id="t1", atoms=atoms + [I.sanity_atom(states)], expected=states), cells + folded


def _moved(slot: str = "seven") -> dict:
    return {"items": {**STATE["items"], "A1": dict(STATE["items"]["A1"], slot=slot)}, "notes": {}}


def test_a_said_amount_compiles_to_digits_every_spelling_of_it_holds():
    item = _item("s1", "said", values=["$1,194", "$408", "$2,040.50", "12 kg"], not_values=[])
    values = I.atom_of(item).target["values"]
    assert values == ["194", "408", "040.5", "12"]
    for reply in ("We charged $1,194, then $408 and $2,040.50 for 12 kg.", "1194 and 408 usd, 2040.5, 12kg"):
        assert all(value in reply.lower() for value in values)


def test_a_said_amount_passes_a_reply_that_keeps_its_sign_and_comma():
    item = _item("s1", "said", values=["$1,194"], not_values=[])
    verifier, _ = _compiled([_row("m1", {"slot": "seven"}), item])
    results = {result.id: result.holds for result in verdict(_run(_moved(), said="You were charged $1,194."),
                                                             verifier).items}
    assert results["s1"] is True


def test_a_said_bare_integer_or_text_with_a_number_stays_as_written():
    item = _item("s1", "said", values=["4451", "slot 7 and 8", "A1"], not_values=[])
    assert I.atom_of(item).target["values"] == ["4451", "slot 7 and 8", "A1"]


def test_a_said_items_description_keeps_the_values_as_the_user_reads_them():
    item = _item("s1", "said", values=["$1,194"], not_values=["$99"])
    assert "$1,194" in I.atom_of(item).description and "$99" in I.atom_of(item).description


def test_two_atoms_of_one_kind_on_one_target_are_one_atom_keeping_the_first_id():
    first = _item("n1", "not_called", tool="add_note", args={}, gate=False, fact_ids=["f1"])
    again = dict(first, id="n_note", gate=True, fact_ids=["f2"])
    verifier, lines = _compiled([_row("m1", {"slot": "seven"}), first, again])
    ids = [atom.id for atom in verifier.atoms]
    assert "n1" in ids and "n_note" not in ids
    assert next(atom for atom in verifier.atoms if atom.id == "n1").gate is True
    assert lines == ["n1: merged n_note, the same not_called target"]


def test_two_cells_on_one_field_with_one_value_score_as_one_item():
    verifier, lines = _compiled([_row("m1", {"slot": "seven"}), _row("k1", {"slot": "seven"}, kind="row_keeps")])
    states = [result.id for result in verdict(_run(_moved()), verifier).items if result.id.startswith("state:")]
    assert states == ["state:items.A1.slot"] and lines == ["items.A1.slot: merged"]


def test_a_cell_and_a_set_it_satisfies_fold_to_the_stricter_value():
    verifier, _ = _compiled([_row("m1", {"slot": {"one_of": ["seven", "eight"]}}), _row("m2", {"slot": "seven"})])
    assert [cell.value for cell in verifier.expected[0].cells] == ["seven"]
    assert cell_clashes([_row("m1", {"slot": {"not": ["two"]}}), _row("m2", {"slot": "seven"})]) == []


def test_a_weighted_exact_cell_beside_a_gated_set_stays_two_cells_and_never_becomes_a_gate():
    verifier, lines = _compiled([_row("m1", {"slot": {"not": ["two"]}}), _row("m2", {"slot": "seven"}, gate=False)])
    cells = verifier.expected[0].cells
    assert [(cell.value, cell.gate) for cell in cells] == [({"not": ["two"]}, True), ("seven", False)] and lines == []


def test_two_cells_on_one_field_whose_values_differ_are_refused_naming_both_values():
    found = cell_clashes([_row("m1", {"slot": "seven"}), _row("m2", {"slot": "eight"})])
    assert [item_id for item_id, _ in found] == ["m2"]
    assert '"eight"' in found[0][1] and '"seven"' in found[0][1] and "m1" in found[0][1]
    assert ("m2", found[0][1]) in I.contradictions([_row("m1", {"slot": "seven"}), _row("m2", {"slot": "eight"})])


def test_cells_in_two_alternative_end_states_never_clash():
    assert cell_clashes([_row("m1", {"slot": "seven"}, alt=0), _row("m2", {"slot": "eight"}, alt=1)]) == []


def test_a_value_implies_a_set_only_when_every_value_it_admits_the_set_admits():
    assert implies("seven", {"one_of": ["seven", "eight"]}) and not implies({"one_of": ["seven", "eight"]}, "seven")
    assert implies({"not": ["two", "four"]}, {"not": ["two"]}) and not implies({"not": ["two"]}, {"not": ["four"]})
    assert implies({"one_of": ["seven"]}, {"not": ["two"]}) and not implies("two", {"not": ["two"]})


def test_two_new_row_specs_one_row_matches_with_other_counts_are_refused_naming_both():
    wide, narrow = _note("w1", {"item_id": "A1"}), dict(_note("w2", {"item_id": "A1", "text": "moved"}), count=2)
    found = new_row_overlaps([wide, narrow])
    assert [item_id for item_id, _ in found] == ["w2"] and "w1" in found[0][1]
    assert new_row_overlaps([wide, dict(wide, id="w3", count=2)])[0][0] == "w3"
    assert new_row_overlaps([wide, _note("w4", {"item_id": "B2"})]) == []
    assert new_row_overlaps([wide, dict(narrow, table="items")]) == []
    assert new_row_overlaps([wide, dict(narrow, count=1)]) == []


def test_new_row_specs_one_made_row_can_answer_are_one_item_whose_where_is_the_union():
    """D440: a row spread over items (the slot in one, the text in another) is one row the Run makes."""
    slot = _note("w1", {"item_id": "A1"}, fact_ids=["f1"])
    text = _note("w2", {"text": "moved"}, fact_ids=["f2"])
    other = _note("w3", {"item_id": "B2"})
    merged, lines = merged_new_rows([slot, text, other, _row("m1", {"slot": "seven"})])
    assert [item["id"] for item in merged] == ["w1", "w3", "m1"] and lines == ["w1: merged w2, one made notes row"]
    assert merged[0]["where"] == {"item_id": "A1", "text": "moved"} and merged[0]["fact_ids"] == ["f1", "f2"]
    assert merged_new_rows([slot, dict(text, alt=1)])[1] == []
    assert merged_new_rows([slot, dict(text, count=2)])[1] == []


def test_a_weighed_new_row_spec_never_folds_into_a_gated_one():
    gated, weighed = _note("w1", {"item_id": "A1"}), _note("w2", {"text": "moved"}, gate=False)
    merged, lines = merged_new_rows([gated, weighed])
    assert lines == [] and [item["gate"] for item in merged] == [True, False]


def test_one_made_row_answering_merged_new_row_specs_passes_the_verdict():
    items, _ = merged_new_rows([_note("w1", {"item_id": "A1"}), _note("w2", {"text": "moved"})])
    verifier, _ = _compiled(items)
    end = {**STATE, "notes": {"N1": {"item_id": "A1", "text": "moved"}}}
    assert verdict(_run(end), verifier).passed is True