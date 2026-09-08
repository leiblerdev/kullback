"""The replay comparison: hard columns after canon, and the ruling names where they part.

The non_trivial ruling is here too, for the one thing only the recording can settle: whether a tool
answers by its arguments or by the world (D165).
"""
from __future__ import annotations

import json

from kullback.gates.tool_runs import (
    body_non_trivial_gate,
    body_replay_fidelity_gate,
    compare_columns,
    compare_results,
    load_readers,
)
from kullback.runner.canon import CanonRules
from kullback.runner.records import Column, EntitySchema, ToolCall
from runner.replay_fixtures import PTR


def _column(name: str, class_: str) -> Column:
    return Column(table="bookings", name=name, **{"class": class_})


SCHEMA = EntitySchema(tables=["bookings"], id_patterns={"bookings.booking_id": r"^#B\d+$"}, columns=[
    _column("booking_id", "hard"), _column("rooms", "hard"), _column("note", "semantic")])


def _booking(size: str, note: str = "sea view") -> dict:
    return {"booking_id": "#B1", "rooms": [{"no": 1, "bed": "double"}, {"no": 2, "bed": size}], "note": note}


def test_a_differing_hard_column_is_reported_at_its_leaf_with_both_values():
    ok, notes = compare_results(SCHEMA, _booking("twin"), _booking("king"))
    assert ok is False
    assert notes == ['rooms[1].bed: ours "king", recorded "twin"']


def test_a_semantic_column_is_reported_by_name_and_does_not_fail():
    ok, notes = compare_results(SCHEMA, _booking("twin"), _booking("twin", "sea  view"))
    assert ok is True and notes == []
    ok, notes = compare_results(SCHEMA, _booking("twin"), _booking("twin", "garden"))
    assert ok is True and notes == ["semantic:note"]


def test_noise_the_learned_precision_absorbs_is_no_difference():
    rules = CanonRules(number_precision=2)
    ours = {"booking_id": "#B1", "rooms": [{"no": 1, "rate": 30.180000000000064}], "note": ""}
    theirs = {"booking_id": "#B1", "rooms": [{"no": 1, "rate": 30.180000000000007}], "note": ""}
    assert compare_results(SCHEMA, theirs, ours, rules) == (True, [])
    assert compare_results(SCHEMA, theirs, ours) == (False, ["rooms[0].rate: ours 30.180000000000064, recorded 30.180000000000007"])


def test_the_fidelity_ruling_carries_the_leaf_and_not_the_semantic_note():
    call = ToolCall(id="c1", name="get_booking", args={"booking_id": "#B1"}, raw_ptr=PTR, result=json.dumps(_booking("twin", "sea view")))
    result = body_replay_fidelity_gate([call], [{"ok": True, "value": _booking("king", "garden")}], SCHEMA)
    assert result.passed is False
    assert result.failures == ['get_booking({"booking_id": "#B1"}): hard columns differ: rooms[1].bed: ours "king", recorded "twin"']
    assert result.metrics["semantic_differences"] == 1


# --- gate 4: non_trivial, and the state-driven tool it may not judge (D165) ---


def _door_call(index: int, answer: str, args: dict) -> ToolCall:
    return ToolCall(id=f"c{index}", name="shutter_state", args=args, raw_ptr=PTR, result=answer)


def test_a_tool_whose_recorded_answer_changed_under_the_same_arguments_is_left_to_replay_fidelity():
    """An argument-free reading of the garage shutter answered "closed" twice and "open" once, because
    a call between them opened it. The gate runs every call on one world, so a right body can only
    answer one way there; the sequence is replay_fidelity's to judge."""
    calls = [_door_call(1, "closed", {}), _door_call(2, "closed", {}), _door_call(3, "open", {})]
    result = body_non_trivial_gate(calls, [{"ok": True, "value": "closed"}] * 3)
    assert result.passed is True
    assert result.metrics["state_driven"] is True and result.metrics["arg_sets_varying"] == 1
    assert "insufficient_evidence" not in result.metrics


def test_a_constant_body_still_fails_when_each_argument_set_had_one_answer_of_its_own():
    """Two argument sets, each answered one way and the two ways differing: the answer comes from the
    arguments, and a body that says the same thing to both is trivial."""
    calls = [_door_call(1, "closed", {"bay": "north"}), _door_call(2, "open", {"bay": "south"})]
    result = body_non_trivial_gate(calls, [{"ok": True, "value": "closed"}] * 2)
    assert result.passed is False
    assert result.failures == ["the body answers every call the same way"]
    assert "state_driven" not in result.metrics
    assert result.metrics["arg_sets"] == 2 and result.metrics["recorded_answers"] == 2


# --- D187: the scoring path honours the column classes, and prose is read into columns ---

READER = (
    "def read(result):\n"
    "    parts = result.split(' | ')\n"
    "    if len(parts) != 3:\n"
    "        return None\n"
    "    return {'lamp_state': parts[0], 'lamp_note': parts[1], 'lamp_reading': parts[2]}\n"
)

LAMP_SCHEMA = EntitySchema(tables=["lamps"], columns=[
    Column(table="lamps", name="lamp_state", **{"class": "hard"}),
    Column(table="lamps", name="lamp_note", **{"class": "semantic"}),
    Column(table="lamps", name="lamp_reading", **{"class": "exempt"})])

LAMP_READERS = load_readers({"proposals": {"technician": {
    "table": "lamps", "columns": ["lamp_state", "lamp_note", "lamp_reading"],
    "readers": [{"tool": "check_lamp", "source": READER}]}}})


def test_two_prose_results_differing_only_in_a_semantic_column_compare_equal():
    ok, notes = compare_results(LAMP_SCHEMA, "on | steady glow | 41", "on | a steady glow | 41",
                                tool="check_lamp", readers=LAMP_READERS)
    assert ok is True and notes == ["semantic:lamp_note"]


def test_two_prose_results_differing_in_a_hard_column_do_not_compare_equal():
    ok, notes = compare_results(LAMP_SCHEMA, "on | steady glow | 41", "off | steady glow | 41",
                                tool="check_lamp", readers=LAMP_READERS)
    assert ok is False
    assert notes == ['read as columns: lamp_state: ours "off", recorded "on"']


def test_a_prose_result_differing_only_in_an_exempt_column_compares_equal():
    ok, notes = compare_results(LAMP_SCHEMA, "on | steady glow | 41", "on | steady glow | 77",
                                tool="check_lamp", readers=LAMP_READERS)
    assert ok is True and notes == ["exempt:lamp_reading"]


def test_a_reader_that_reads_nothing_falls_back_to_comparing_the_strings():
    ok, notes = compare_results(LAMP_SCHEMA, "on, steady", "off, steady",
                                tool="check_lamp", readers=LAMP_READERS)
    assert ok is False
    assert notes == ['string differs: value: ours "off, steady", recorded "on, steady"']


def test_a_returned_rows_exempt_column_cannot_fail_a_replay_and_a_hard_one_does():
    schema = EntitySchema(tables=["orders"], id_patterns={"orders.order_id": r"^#O\d+$"}, columns=[
        Column(table="orders", name="order_id", **{"class": "hard"}),
        Column(table="orders", name="total", **{"class": "hard"}),
        Column(table="orders", name="created_at", **{"class": "exempt"})])
    recorded = {"order_id": "#O7", "total": 12, "created_at": "2031-01-01T00:00:00"}
    later = {"order_id": "#O7", "total": 12, "created_at": "2031-06-02T09:30:00"}
    assert compare_results(schema, recorded, later) == (True, ["exempt:created_at"])
    ok, notes = compare_results(schema, recorded, dict(later, total=13))
    assert ok is False
    assert notes == ['total: ours 13, recorded 12', "exempt:created_at"]


def test_a_semantic_column_takes_the_judge_when_one_is_given():
    asked = []

    def judge(column, ours, theirs):
        asked.append(column)
        return {"verdict": "equivalent"}

    schema = EntitySchema(tables=["lamps"], columns=[
        Column(table="lamps", name="lamp_id", **{"class": "hard"}),
        Column(table="lamps", name="lamp_note", **{"class": "semantic"})])
    recorded = {"lamp_id": "L1", "lamp_note": "steady glow"}
    ours = {"lamp_id": "L1", "lamp_note": "a glow that holds steady"}
    assert compare_columns(schema, "lamps", recorded, ours, judge=judge) == (True, [])
    assert asked == ["lamp_note"]
    assert compare_columns(schema, "lamps", recorded, ours) == (True, ["semantic:lamp_note"])


def test_two_lists_of_keyed_rows_compare_equal_whatever_order_they_come_back_in():
    schema = EntitySchema(tables=["bills"], id_patterns={"bills.bill_id": r"^#L\d+$"}, columns=[
        Column(table="bills", name="bill_id", **{"class": "hard"}),
        Column(table="bills", name="amount", **{"class": "hard"})])
    first = {"bill_id": "#L1", "amount": 10}
    second = {"bill_id": "#L2", "amount": 20}
    assert compare_results(schema, [first, second], [second, first]) == (True, [])
    ok, notes = compare_results(schema, [first, second], [second, dict(first, amount=11)])
    assert ok is False and notes == ['amount: ours 11, recorded 10']


def test_two_lists_of_rows_carrying_no_key_are_still_paired_by_position():
    schema = EntitySchema(tables=["bills"], columns=[
        Column(table="bills", name="amount", **{"class": "hard"})])
    ok, notes = compare_results(schema, [{"amount": 10}, {"amount": 20}], [{"amount": 20}, {"amount": 10}])
    assert ok is False
    assert notes == ['value: ours 20, recorded 10', 'value: ours 10, recorded 20']
