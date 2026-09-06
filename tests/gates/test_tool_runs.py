"""The replay comparison: hard columns after canon, and the ruling names where they part."""
from __future__ import annotations

import json

from kullback.gates.tool_runs import body_replay_fidelity_gate, compare_results
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
