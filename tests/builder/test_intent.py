"""Tests for what builder/intent.py still re-exports: applying an Intent (D47) and the value strip (D196)."""

from __future__ import annotations

import json

from conftest import PTR
from kullback.builder.intent import Intent, apply_intent, strip_intent
from kullback.runner.records import Column, EntitySchema, Task, ToolCall, Trace, Turn


def make_trace(trace_id: str, user_turns: list[str], calls: list[dict]) -> Trace:
    return Trace(
        trace_id=trace_id,
        raw_hash="raw",
        ingest_version="0",
        source="tau2",
        turns=[Turn(idx=i, role="user", content=c, raw_ptr=PTR) for i, c in enumerate(user_turns)],
        tool_calls=[
            ToolCall(id=f"{trace_id}-{i}", name=c["name"], args=c.get("args", {}),
                     result=c.get("result"), raw_ptr=PTR)
            for i, c in enumerate(calls)
        ],
        raw_ptr=PTR,
    )


def cancel_trace(trace_id: str, order: str) -> Trace:
    return make_trace(
        trace_id,
        [f"i want to cancel order {order} because the delivery was late"],
        [
            {
                "name": "cancel_order",
                "args": {"order_id": order, "reason": "late delivery"},
                "result": {"status": "cancelled"},
            }
        ],
    )


def two_run_task() -> tuple[Task, list[Trace]]:
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2")]
    return Task(id="task_1", category_id="cat_1", run_ids=["t1", "t2"]), traces


# --- noun phrases ---


# --- tokenising ---


# --- spans ---


# --- a span that is not the user's words ---


# --- write_intent ---


# --- the rewrite loop ---


# --- whether a recorded Intent still holds for its Task ---


# --- applying the Intent to the Task ---


def test_apply_intent_sets_the_task_name_intent_and_unguarded_mark():
    task, _ = two_run_task()
    intent = Intent(task_id=task.id, text="cancel the order because the delivery was late", grounded=True)
    updated = apply_intent(task, intent)
    assert updated.intent == "cancel the order because the delivery was late"
    assert updated.name == "cancel the order because the delivery was late"
    assert task.intent is None
    task = Task(id="task_1", run_ids=["t1"])
    intent = Intent(task_id=task.id, text="cancel order W1", grounded=True, unguarded=True)
    assert apply_intent(task, intent).unguarded is True


def test_apply_intent_leaves_an_ungrounded_intent_off_the_task():
    task, _ = two_run_task()
    updated = apply_intent(task, Intent(task_id=task.id, text="refund to a gift card", grounded=False))
    assert updated.intent is None
    assert updated.name is None


# --- D196: the strip, before the Intent exists ---


def hire_trace(trace_id: str, ref: str, said: str) -> Trace:
    """One bike-hire recording: the rider says why, the desk reads the hire and refunds the deposit."""
    return make_trace(
        trace_id,
        [said],
        [
            {
                "name": "get_hire",
                "args": {"hire_ref": ref},
                "result": {"hire_ref": ref, "station": "Riverside", "started_on": "2026-03-04",
                           "deposit": 45.0},
            },
            {"name": "refund_hire", "args": {"hire_ref": ref, "deposit": 45.0},
             "result": {"refunded": True}},
        ],
    )


def hire_task() -> tuple[Task, list[Trace]]:
    traces = [hire_trace("h1", "HR-88231", "the hire i started never unlocked, please refund me"),
              hire_trace("h2", "HR-44107", "please refund the hire, the bike never unlocked")]
    return Task(id="task_hire", run_ids=["h1", "h2"]), traces


def hire_schema() -> EntitySchema:
    """The station is exempt to the compare: two desks may name it differently and mean one place."""
    return EntitySchema(
        tables=["hires"],
        columns=[Column(table="hires", name="hire_ref", **{"class": "hard"}),
                 Column(table="hires", name="station", **{"class": "exempt"}),
                 Column(table="hires", name="started_on", **{"class": "hard"})],
    )


def test_a_value_only_the_tools_held_is_taken_out_and_the_record_names_its_column_and_class():
    _, traces = hire_task()
    text, stripped = strip_intent("refund the hire HR-88231 at Riverside", traces, schema=hire_schema())
    assert "HR-88231" not in text and "Riverside" not in text
    by_column = {value.column: value for value in stripped}
    assert by_column["hire_ref"].class_ == "hard" and by_column["hire_ref"].source == "tool_arg"
    assert by_column["station"].class_ == "exempt" and by_column["station"].shape == "removed"
    dumped = json.dumps([value.model_dump(mode="json", by_alias=True) for value in stripped])
    assert "HR-88231" not in dumped and "Riverside" not in dumped, "a record of a strip holds no value"


def test_a_value_the_user_said_stays_in_the_intent():
    _, traces = hire_task()
    traces[0].turns[0].content = "hire HR-88231 at Riverside never unlocked, please refund me"
    text, stripped = strip_intent("refund the hire HR-88231 at Riverside", traces, schema=hire_schema())
    assert text == "refund the hire HR-88231 at Riverside"
    assert stripped == []


def test_a_code_date_or_amount_the_user_never_said_is_reduced_to_a_shape_that_reads():
    _, traces = hire_task()
    text, stripped = strip_intent("refund the hire HR-88231", traces, schema=hire_schema())
    assert text == "refund the hire ending 8231"
    assert [value.shape for value in stripped] == ["last4"]
    _, traces = hire_task()
    text, _ = strip_intent("refund the hire started on 2026-03-04", traces, schema=hire_schema())
    assert text == "refund the hire started on march"
    _, traces = hire_task()
    text, stripped = strip_intent("refund the 45.0 deposit on the hire", traces, schema=hire_schema())
    assert text == "refund the deposit on the hire"
    assert [value.column for value in stripped] == ["deposit"]


# --- D245 stable heads ---

