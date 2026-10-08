"""Expected end states derived by code from a Reference Run, and a Run's end state checked against them (D315)."""

from __future__ import annotations

from kullback.runner.expected import (
    alternatives_from_turns,
    contradicts_user,
    expected_from_run,
    forbidden_from,
    match,
    refusals_of,
    unasked_writes,
)
from kullback.runner.records import Verifier
from kullback.runner.verdict import verdict

START = {"orders": {"A1": {"status": "pending", "method": "card_1", "note": "x"},
                    "B2": {"status": "pending", "method": "card_1", "note": "y"}}}


def _end(**changes):
    end = {"orders": {k: dict(v) for k, v in START["orders"].items()}}
    for key, value in changes.items():
        row, field = key.split("__")
        end["orders"][row][field] = value
    return end


def _events(turns, end, results=()):
    events = [{"idx": i, "type": "user_turn", "payload": {"content": text}} for i, text in enumerate(turns)]
    for value in results:
        n = len(events)
        events.append({"idx": n, "type": "tool_call", "payload": {"id": f"c{n}", "name": "lookup", "args": {}}})
        events.append({"idx": n + 1, "type": "tool_result", "payload": {"id": f"c{n}", "result": value}})
    events.append({"idx": len(events), "type": "stop", "payload": {"start_state": START, "end_state": end}})
    return events


def _run(turns, end):
    return {"run_id": "r", "events": _events(turns, end)}


def test_a_value_named_by_the_user_and_one_only_in_a_tool_result_carry_those_two_sources():
    end = _end(A1__status="cancelled", A1__method="gift_9")
    turns = ["Please mark order A1 cancelled."]
    state = expected_from_run(_events(turns, end, results=[{"methods": ["gift_9"]}]), turns)
    assert {cell.field: cell.source.kind for cell in state.cells} == {"status": "user_turn", "method": "tool_result"}
    assert all(cell.row_source.kind == "user_turn" for cell in state.cells)


def test_a_value_with_no_source_is_unsupported():
    turns = ["Change order A1 please."]
    state = expected_from_run(_events(turns, _end(A1__method="gift_9")), turns)
    assert [cell.source for cell in state.cells] == [None]


def test_a_user_turn_naming_two_values_for_one_field_gives_two_end_states_and_either_run_passes():
    turns = ["Refund order A1 to gift_9 or card_7, either is fine."]
    reference = _end(A1__method="gift_9")
    states = alternatives_from_turns(expected_from_run(_events(turns, reference), turns), turns)
    assert sorted(state.cells[0].value for state in states) == ["card_7", "gift_9"]
    verifier = Verifier(task_id="t", expected=states)
    assert Verifier.model_validate_json(verifier.model_dump_json()) == verifier
    for method in ("gift_9", "card_7"):
        assert verdict(_run(turns, _end(A1__method=method)), verifier).passed
    assert not verdict(_run(turns, _end(A1__method="card_1x")), verifier).passed


def test_a_run_that_leaves_the_recorded_row_unchanged_fails_the_expected_gate():
    turns = ["Cancel order A1."]
    state = expected_from_run(_events(turns, _end(A1__status="cancelled")), turns)
    assert match(state, START, START).ok is False
    out = verdict(_run(turns, START), Verifier(task_id="t", expected=[state]))
    assert not out.passed and out.failing_atom == "gate:expected:cell:orders.A1.status"


def test_a_run_moving_a_row_outside_the_expected_set_and_the_allowed_list_fails_on_collateral():
    turns = ["Cancel order A1."]
    state = expected_from_run(_events(turns, _end(A1__status="cancelled")), turns)
    moved = _end(A1__status="cancelled", B2__note="z")
    assert match(state, START, moved).ok is False
    assert match(state, START, moved).why == ["collateral:orders.B2.note"]
    allowed = state.model_copy(update={"allowed": [{"table": "orders", "row_id": "B2", "field": "note"}]})
    assert match(allowed, START, moved).ok is True


def test_a_user_negation_and_a_recorded_refusal_are_the_only_forbidden_writes():
    turns = ["Cancel order A1 but do not touch B2.", "Don't change the note."]
    events = _events(turns, _end(A1__status="cancelled"))
    events.insert(-1, {"idx": 90, "type": "tool_call", "payload": {"id": "z", "name": "refund", "args": {"id": "A1"}}})
    events.insert(-1, {"idx": 91, "type": "tool_result", "payload": {"id": "z", "error": {"message": "not allowed"}}})
    state = expected_from_run(events, turns)
    found = forbidden_from(state, turns, refusals_of(events), start_state=START)
    assert {(f.tool, f.row_id, f.field, f.source.kind) for f in found} == {
        (None, "B2", None, "user_turn"), (None, "A1", "note", "user_turn"), ("refund", "A1", None, "tool_result")}


def test_a_list_cell_is_sourced_by_its_changed_leaves_and_names_the_leaf_no_turn_or_result_shows():
    start = {"orders": {"A1": {"items": [{"sku": "k1", "qty": 1}, {"sku": "k2", "qty": 1}]}}}
    end = {"orders": {"A1": {"items": [{"sku": "k1", "qty": 1}, {"sku": "k9", "qty": 1}]}}}
    events = [{"idx": 0, "type": "user_turn", "payload": {"content": "Swap the second item of A1 for k9."}},
              {"idx": 1, "type": "stop", "payload": {"start_state": start, "end_state": end}}]
    cell = expected_from_run(events, ["Swap the second item of A1 for k9."]).cells[0]
    assert cell.source.kind == "user_turn" and list(cell.leaf_sources) == ["1.sku"]
    turns = ["Swap the second item of A1."]
    events[0]["payload"]["content"] = turns[0]
    cell = expected_from_run(events, turns).cells[0]
    assert cell.source is None and cell.leaf_sources == {"1.sku": None}


# --- D321: consistent with the request, not only available ---

WALLET = {"orders": {"A1": {"method": "card_1"}, "B2": {"method": "gift_9"}, "C3": {"method": "paypal_4"}}}


def _cell_state(field, value, row_id="A1"):
    return expected_from_run(_events([], {"orders": {**START["orders"], row_id: {
        **START["orders"][row_id], field: value}}}), []).model_copy()


def test_a_cell_holding_a_value_the_user_named_is_not_contradicted():
    state = _cell_state("method", "gift_9")
    assert contradicts_user(state, ["Refund A1 to gift_9 please."], WALLET) == []


def test_a_cell_holding_another_value_of_a_column_the_user_named_a_choice_of_contradicts_the_user():
    state = _cell_state("method", "paypal_4")
    turns = ["Refund A1 please.", "Put it on gift_9, not my card."]
    assert contradicts_user(state, turns, WALLET) == ["orders.A1.method"]


def test_a_column_the_user_never_named_a_value_of_is_not_contradicted():
    state = _cell_state("method", "paypal_4")
    assert contradicts_user(state, ["Refund A1 however you like."], WALLET) == []


def _write_run(turns):
    events = [{"idx": i, "type": "user_turn", "payload": {"content": text}} for i, text in enumerate(turns)]
    n = len(events)
    events += [{"idx": n, "type": "tool_call", "payload": {"id": "w", "name": "refund", "args": {"order_id": "Z9"}}},
               {"idx": n + 1, "type": "tool_result", "payload": {"id": "w", "result": {"ok": True}}},
               {"idx": n + 2, "type": "stop", "payload": {"start_state": START, "end_state": START}}]
    return {"run_id": "r", "events": events}


def test_a_write_with_no_request_naming_its_arguments_and_no_yes_before_it_is_unasked():
    assert unasked_writes(_write_run(["Something is wrong with my account."]), (), {"refund"}) == ["refund@1"]


def test_a_write_after_a_user_yes_or_a_request_naming_its_row_is_asked():
    assert unasked_writes(_write_run(["Something is wrong.", "Yes, do that."]), (), {"refund"}) == []
    assert unasked_writes(_write_run(["Refund order Z9."]), (), {"refund"}) == []
