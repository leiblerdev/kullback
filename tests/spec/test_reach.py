"""A gated value must be reachable (D380): a tool writes it, a recorded write result shows it, the user fixed it,
or a start row holds it under that field. Code refuses any other gated value with the field's reachable values.

The world, the policy, the tools and the facts are invented: items on shelves, a move tool that marks a moved item
with a fixed state word, and a policy that spells that word another way on one line.
"""

from __future__ import annotations

from kullback.spec import effects as E
from kullback.spec import items as I
from kullback.spec import writer_tools
from kullback.spec.schema import FactSource, IntentFact
from kullback.spec.writer import _store_counts, reason_family

STATE = {"items": {"A1": {"item_id": "A1", "state": "shelved", "slot": "two", "weight": 4},
                   "B2": {"item_id": "B2", "state": "shelved", "slot": "four", "weight": 7},
                   "C3": {"item_id": "C3", "state": "moved (pending)", "slot": "one", "weight": 2}},
         "people": {"P1": {"person_id": "P1", "card": "card_11"}, "P2": {"person_id": "P2", "card": "card_22"}}}
# The policy spells the tool's state word wrong on one line, as a real policy may.
POLICY = "# Moves\n\nA moved item shows the state moved (pendng) until it is checked.\n"
FACTS = [IntentFact(id="f1", text="Please move item A1 to slot seven.", stance="volunteered",
                    source=FactSource(recording="rec1", turn=1), witnesses=["rec2"]),
         IntentFact(id="f2", text="Use slot nine then.", stance="accepted", echo=True,
                    source=FactSource(recording="rec1", turn=3))]
TOOLS = [{"name": "move_item", "kind": "write", "args": ["item_id", "slot", "card"]}]
KNOWN = {fact.id: fact for fact in FACTS}


def _move(row: str, slot: str, card: str) -> tuple[dict, dict]:
    """One recorded move: the slot the call passed, the fixed state word, and the card it was paid with."""
    before = STATE["items"][row]
    after = dict(before, slot=slot, state="moved (pending)", paid_with=card)
    return {"items": {row: {"before": before, "after": after}}}, {"item_id": row, "slot": slot, "card": card}


MOVES = [_move("A1", "five", "card_11"), _move("B2", "six", "card_11")]
EFFECTS = {"move_item": E.always([diff for diff, _ in MOVES], [args for _, args in MOVES])}


def world(reach: I.Reach | None = None, results=()) -> I.World:
    return I.World(STATE, POLICY, FACTS, TOOLS,
                   reach=reach if reach is not None else I.reach_of_inputs(EFFECTS, results))


def row_is(expect: dict, gate: bool = True, **extra) -> dict:
    return dict({"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": expect,
                 "gate": gate, "fact_ids": ["f1"]}, **extra)


def keep(item: dict, w: I.World | None = None) -> tuple[dict | None, str]:
    return (w or world()).check(item, KNOWN)


def test_a_tool_learns_the_literal_values_it_writes_and_the_fields_it_copies_from_its_arguments():
    writes = {(row["table"], row["field"]): row for row in EFFECTS["move_item"]["writes"]}
    assert writes[("items", "state")]["values"] == ["moved (pending)"]
    assert writes[("items", "slot")]["from_args"] == ["slot"]
    assert writes[("items", "paid_with")]["from_args"] == ["card"]


def test_a_field_written_with_more_distinct_values_than_the_literal_cap_keeps_no_values():
    diffs = [{"items": {"A1": {"before": {"note": "x"}, "after": {"note": f"n{n}"}}}} for n in range(E.MAX_LITERAL + 1)]
    assert E.literal_writes(diffs) == []
    assert E.literal_writes(diffs[:2]) == [{"table": "items", "field": "note", "values": ["n0", "n1"]}]


def test_a_gated_value_a_tool_writes_is_kept_and_sourced_from_that_tool():
    kept, why = keep(row_is({"state": "moved (pending)"}))
    assert why == "" and kept["gate"] is True
    assert world().reach_of("items", "state", "MOVED  (Pending) ").ptr["seen"] == "tool:move_item"


def test_a_gated_value_only_the_policy_spells_is_refused_naming_the_item_field_value_and_candidates():
    kept, why = keep(row_is({"state": "moved (pendng)"}))
    assert kept is None and I.UNREACHABLE in why
    assert "item s1 state" in why and '"moved (pendng)"' in why and '"moved (pending)"' in why
    assert "weigh the item" in why


def test_a_gated_value_the_user_fixed_is_reachable_but_an_echo_of_the_agent_is_not():
    assert keep(row_is({"slot": "seven"}))[1] == ""
    assert world().reach_of("items", "slot", "seven").kind == "user_turn"
    assert world(I.Reach()).reach_of("items", "slot", "nine") is None


def test_a_field_copied_from_an_argument_reaches_any_value_the_start_state_holds():
    found = world().reach_of("items", "paid_with", "card_22")
    assert found.kind == "tool_result" and found.ptr["seen"] == "arg:move_item.card"
    assert world().reach_of("items", "paid_with", "card_99") is None


def test_a_value_a_recorded_write_result_shows_under_that_field_is_reachable():
    results = [("rec1", "move_item", '{"item_id": "A1", "state": "checked"}')]
    assert world().reach_of("items", "state", "checked") is None
    found = world(results=results).reach_of("items", "state", "checked")
    assert found.kind == "tool_result" and found.ptr["seen"] == "recording:rec1:move_item"


def test_a_start_row_holding_the_value_under_the_same_field_name_reaches_it_and_another_field_does_not():
    assert world(I.Reach()).reach_of("items", "slot", "four").kind == "world"
    assert world(I.Reach()).reach_of("items", "state", "four") is None


def test_numbers_compare_as_numbers_and_a_null_is_always_reachable():
    assert world(I.Reach()).reach_of("items", "weight", 7.0) is not None
    assert world(I.Reach()).reach_of("items", "weight", "7") is not None
    assert world(I.Reach()).reach_of("items", "weight", 8) is None
    assert world(I.Reach()).reach_of("items", "state", None).ptr == {"removed": True}


def test_every_member_of_a_one_of_or_a_list_is_checked_and_a_not_set_never_is():
    assert I.UNREACHABLE in keep(row_is({"state": {"one_of": ["moved (pending)", "moved (pendng)"]}}))[1]
    assert I.UNREACHABLE in keep(row_is({"slot": ["seven", "moved (pendng)"]}))[1]
    assert keep(row_is({"state": {"not": ["moved (pendng)"]}}))[1] == ""


def test_a_new_row_holding_an_unreachable_gated_value_is_refused():
    item = {"id": "n1", "kind": "row_new", "table": "items", "where": {"state": "moved (pendng)"}, "fact_ids": ["f1"]}
    assert I.UNREACHABLE in keep(item)[1]


def test_a_weighed_item_naming_an_unreachable_value_is_kept_with_the_finding_as_a_warning():
    kept, why = keep(row_is({"state": "moved (pendng)"}, gate=False))
    assert kept is not None and why.startswith("warning:") and I.UNREACHABLE in why


def test_a_world_without_reach_checks_no_gated_value_for_reachability():
    assert I.World(STATE, POLICY, FACTS, TOOLS).check(row_is({"state": "moved (pendng)"}), KNOWN)[1] == ""


def test_the_store_counts_refused_and_warned_unreachable_values_and_a_drop_clears_a_warning():
    store = writer_tools.ItemStore(world(), KNOWN)
    store.add([row_is({"state": "moved (pendng)"}), row_is({"state": "moved (pendng)"}, gate=False, id="s2")])
    assert reason_family(store.refused[0]["reason"]) == "unreachable value"
    assert _store_counts(store)["reachability"] == {"refused": 1, "warned": 1}
    store.drop(["s2"])
    assert _store_counts(store)["reachability"] == {"refused": 1, "warned": 0}


def test_a_gated_cell_takes_its_reachable_source_and_an_unreachable_one_none():
    w = world()
    kept, _ = keep(row_is({"state": "moved (pending)"}), w)
    unreached = dict(kept, expect={"state": "moved (pendng)"}, id="s2", find={"item_id": "B2"}, row="B2")
    (state,) = I.end_states_of([kept, unreached], w)
    sources = {cell.row_id: cell.source for cell in state.cells}
    assert sources["A1"].kind == "tool_result" and sources["B2"] is None
