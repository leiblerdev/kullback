"""The slice a scenario is written onto (D420): picked by code, its binding and its end's values checked by code."""

from __future__ import annotations

from pathlib import Path

from kullback.synth import slice as S
from kullback.synth.rule import World

# Keepers own boxes; a box lists the parts it holds; a part names its maker; a ledger row refers to nothing.
STATE = {"keepers": {"k1": {"label": "Rin Sato"}, "k2": {"label": "Ola Berg"}, "k3": {"label": "Ude Moss"},
                     "k4": {"label": "Ivo Lund"}},
         "boxes": {"b1": {"keeper": "k1", "parts": [{"part": "p1", "cost": 12.5}, {"part": "p2", "cost": 30}],
                          "total": 42.5},
                   "b2": {"keeper": "k2", "parts": [{"part": "p2", "cost": 30}], "total": 30},
                   "b3": {"keeper": "k3", "parts": [], "total": 0, "meta": {"keeper_again": "k3"}}},
         "parts": {"p1": {"name": "brass hinge", "maker": "m1"}, "p2": {"name": "oak panel", "maker": "m1"}},
         "makers": {"m1": {"name": "Hollow Works"}},
         "ledger": {"l1": {"note": "opening stock"}}}


def _world(state: dict = STATE) -> World:
    return World(workdir=Path("."), state=state)


def test_the_owner_table_is_the_one_whose_keys_other_rows_name_most_in_scalar_cells_never_inside_lists():
    assert S.Slicer.of(_world()).owner_table == "keepers"


def test_with_no_table_named_by_another_the_owner_table_holds_the_most_columns_recorded_users_give():
    state = {"people": {"x1": {"mail": "a@b.c", "town": "Ely"}}, "things": {"t1": {"town": "Ely", "size": 3}}}
    assert S.Slicer.of(_world(state)).owner_table == ""
    assert S.Slicer.of(_world(state), fields={"mail", "town"}).owner_table == "people"


def test_a_slice_holds_the_owner_the_rows_naming_it_and_their_rows_one_level_down_and_nothing_else():
    cut = S.Slicer.of(_world()).slice_of("k1")
    assert {table: sorted(rows) for table, rows in cut.rows.items()} == {
        "keepers": ["k1"], "boxes": ["b1"], "parts": ["p1", "p2"]}
    assert list(cut.rows)[0] == "keepers" and (cut.owner_table, cut.owner_key) == ("keepers", "k1")
    assert "makers" not in cut.rows and "ledger" not in cut.rows
    assert cut.text().splitlines()[0] == 'keepers k1: {"label":"Rin Sato"}'


def test_a_slice_past_its_size_limit_names_the_rows_it_does_not_show():
    text = S.Slicer.of(_world()).slice_of("k1").text(limit=80)
    assert text.splitlines()[0].startswith("keepers k1:") and "not shown, past the size limit: " in text


def test_the_pick_is_the_same_for_the_same_seed_and_spreads_over_owners_not_used():
    world = _world()
    assert S.pick_slice(world, "a").owner_key == S.pick_slice(world, "a").owner_key
    used: set = set()
    for at in range(3):
        used.add(S.pick_slice(world, f"seed {at}", used).owner_key)
    assert used == {"k1", "k2", "k3"}
    assert S.pick_slice(world, "next", used).owner_key == "k4"
    assert S.pick_slice(world, "again", {"k1", "k2", "k3", "k4"}).owner_key in {"k1", "k2", "k3"}


def test_a_pick_for_a_table_prefers_an_owner_whose_slice_holds_a_row_of_it():
    assert {S.pick_slice(_world(), f"s{at}", table="parts").owner_key for at in range(8)} <= {"k1", "k2"}


def test_a_world_without_an_owner_table_gives_no_slice():
    assert S.pick_slice(_world({"ledger": {"l1": {"note": "x"}}}), "a") is None


def test_the_binding_is_every_slice_row_named_by_key_or_by_a_value_the_owner_does_not_hold_the_owner_always():
    cut = S.Slicer.of(_world()).slice_of("k1")
    assert S.binding_of(cut, "Rin Sato wants help") == {"keepers": ["k1"]}
    assert S.binding_of(cut, "Rin Sato wants the brass hinge swapped", "", "box b1 is emptied") == {
        "keepers": ["k1"], "boxes": ["b1"], "parts": ["p1"]}


def test_an_end_value_outside_the_slice_and_the_quoted_lines_is_named_and_a_sum_of_two_amounts_passes():
    cut = S.Slicer.of(_world()).slice_of("k1")
    assert S.value_refusal(cut, "Box b1 now totals 42.5 after a refund of 12.5.") == ""
    assert S.value_refusal(cut, "A refund of 17.5 is made.") == ""
    assert S.value_refusal(cut, "A refund of 99 is made.") == "99"
    assert S.value_refusal(cut, "A refund of 99 is made.", ["Refunds over 99 need a manager."]) == ""
    assert S.value_refusal(cut, 'The part is now "walnut panel".') == '"walnut panel"'
    assert S.value_refusal(cut, 'The part is now "oak panel".') == ""
    assert S.value_refusal(cut, "Box Q123 is emptied.") == "Q123"
