"""What licenses a value the user gave (D438): a rendered fact or the policy names it, a rendered fact names a
row that holds it, or it equals a rendered fact's words after canon, hyphen, space and case folding.

The world is invented: two boxes on two shelves, each with an owner and a size. The agent offers a shelf and
the user says yes (an echo fact), beside what the user said in their own words.
"""

from __future__ import annotations

from kullback.spec import items as I
from kullback.spec.schema import FactSource, IntentFact

STATE = {"boxes": {"K1": {"box_id": "K1", "shelf": "north wing", "size": "34-inch", "owner": "Ada", "tint": "grey"},
                   "K2": {"box_id": "K2", "shelf": "south wing", "size": "12 inch", "owner": "Ben", "tint": "grey"}}}
POLICY = "# Boxes\n\nA box moves only to a shelf with room, after the user says yes.\n"
OFFER = "agrees to put box K2 on the north wing"
PUT = {"id": "s1", "kind": "row_is", "table": "boxes", "find": {"box_id": "K2"}, "expect": {"shelf": "north wing"},
       "fact_ids": ["f1", "f2"]}


def fact(fact_id: str, text: str, echo: bool = False) -> IntentFact:
    return IntentFact(id=fact_id, text=text, stance="accepted" if echo else "volunteered", echo=echo,
                      source=FactSource(recording="rec1", turn=1))


def world(said: str, policy: str = POLICY) -> I.World:
    return I.World(STATE, policy, [fact("f1", said), fact("f2", OFFER, echo=True)])


def check(said: str, item: dict = PUT, policy: str = POLICY) -> tuple[dict, str]:
    w = world(said, policy)
    return w.check(item, {f.id: f for f in w.facts})


def test_a_value_a_rendered_fact_or_a_policy_line_names_is_licensed_and_its_echo_gate_stands():
    assert world("Please move box K2 to the north wing.").licence("north wing") == I.NAMED_LICENCE
    assert world("Move box K2.", POLICY + "Overflow goes to the north wing.\n").licence("north wing") == I.NAMED_LICENCE
    kept, why = check("Please move box K2 to the north wing.")
    assert why == "" and kept["gate"] is True


def test_a_value_on_the_row_a_rendered_fact_names_by_key_or_by_one_picking_value_is_licensed():
    assert world("Move box K2 to the same shelf as box K1.").licence("north wing") == I.ROW_LICENCE
    assert world("Move box K2 to the shelf of the box Ada owns.").licence("north wing") == I.ROW_LICENCE
    kept, why = check("Move box K2 to the shelf of the box Ada owns.")
    assert why == "" and kept["gate"] is True


def test_a_value_two_rows_hold_picks_no_row_so_it_licenses_nothing_held_beside_it():
    assert world("Move box K2 to the shelf of the grey box.").licence("north wing") == ""


def test_a_value_equal_to_the_facts_words_after_canon_hyphen_space_and_case_folding_is_licensed():
    assert world("Move box K2, the 34 Inch one, please.").licence("34-inch") == I.CANON_LICENCE
    assert world("I want the 12-inch box moved.").licence("12 inch") == I.CANON_LICENCE
    assert I.folded("Paid $1,194 for the 34-Inch box.") == "paid 1194 usd for the 34 inch box"


def test_a_value_only_canon_licenses_passes_the_no_invention_rule_that_refused_it_before():
    item = dict(PUT, expect={"size": "12-inch"}, fact_ids=["f1"])
    assert check("Mark box K2 as the 12 inch size.", item)[1] == ""


def test_an_abbreviation_or_a_choice_licenses_nothing_and_its_echo_gate_is_weighed_naming_the_three_ways():
    for said in ("Move box K2 to the N. wing.", "Move box K2 to the emptiest shelf."):
        assert world(said).licence("north wing") == ""
        kept, why = check(said)
        assert kept["gate"] is False and kept["weighed"] == I.ECHO_WEIGHED
        assert 'shelf "north wing" rests only on echo facts' in why and I.LICENCE_WAYS in why


def test_a_value_nothing_holds_and_no_way_licenses_is_refused_naming_the_three_ways():
    kept, why = check("Move box K2 to the east wing.", dict(PUT, expect={"shelf": "west annex"}, fact_ids=["f1"]))
    assert kept is None and 'value "west annex" is not in the world' in why and I.LICENCE_WAYS in why


def test_an_echo_unfounded_or_end_fact_never_licenses_a_value():
    offer_only = I.World(STATE, POLICY, [fact("f2", "agrees to the 34 inch box on box K1's shelf", echo=True)])
    assert offer_only.named_rows() == [] and offer_only.licence("34-inch") == ""
    assert I.World(STATE, POLICY, [fact("f1", "Box K1.").model_copy(update={"unfounded": True})]).named_rows() == []
