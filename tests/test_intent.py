"""Tests for builder/intent.py: the one-line Intent, every noun phrase tied to a span (D47)."""

from __future__ import annotations

import pytest

from harness.builder.intent import (
    MAX_PROMPT_RUNS,
    Intent,
    apply_intent,
    ground_phrases,
    noun_phrases,
    span_candidates,
    write_intent,
)
from harness.shared.records import Task, ToolCall, Trace, Turn

WRITES = {"cancel_order"}


def make_trace(trace_id: str, user_turns: list[str], calls: list[dict]) -> Trace:
    return Trace(
        trace_id=trace_id,
        raw_hash="raw",
        ingest_version="0",
        source="tau2",
        turns=[Turn(idx=i, role="user", content=c) for i, c in enumerate(user_turns)],
        tool_calls=[
            ToolCall(id=f"{trace_id}-{i}", name=c["name"], args=c.get("args", {}), result=c.get("result"))
            for i, c in enumerate(calls)
        ],
    )


def cancel_trace(trace_id: str, order: str) -> Trace:
    return make_trace(
        trace_id,
        [f"i want to cancel order {order} because the delivery was late"],
        [{"name": "cancel_order", "args": {"order_id": order, "reason": "late delivery"}, "result": {"status": "cancelled"}}],
    )


def two_run_task() -> tuple[Task, list[Trace]]:
    traces = [cancel_trace("t1", "W1"), cancel_trace("t2", "W2")]
    return Task(id="task_1", category_id="cat_1", run_ids=["t1", "t2"]), traces


# --- noun phrases ---


def test_noun_phrases_are_the_runs_of_non_stopword_tokens():
    assert noun_phrases("cancel the order because the delivery was late") == ["cancel", "order", "delivery", "late"]


def test_noun_phrases_keep_multi_word_phrases_and_drop_duplicates():
    phrases = noun_phrases("cancel a late order, cancel a late order")
    assert phrases == ["cancel", "late order"]


def test_noun_phrases_of_empty_text_is_empty():
    assert noun_phrases("") == []


# --- spans ---


def test_span_candidates_cover_user_words_tool_arguments_and_written_values():
    trace = cancel_trace("t1", "W1")
    sources = {s.source for s in span_candidates(trace, WRITES)}
    assert sources == {"user_utterance", "tool_arg", "written_value"}


def test_a_result_of_a_tool_that_is_not_a_write_is_not_a_span():
    trace = make_trace("t1", ["where is it"], [{"name": "get_order", "args": {"order_id": "W1"}, "result": {"status": "delivered"}}])
    texts = [s.text for s in span_candidates(trace, WRITES)]
    assert not any("delivered" in t for t in texts)
    assert any("W1" in t for t in texts)


def test_ground_phrases_reports_a_phrase_with_no_span():
    _, traces = two_run_task()
    spans, ungrounded = ground_phrases(["cancel order", "gift card"], traces, WRITES)
    assert ungrounded == ["gift card"]
    assert [s.phrase for s in spans] == ["cancel order"]


def test_ground_phrases_spread_across_runs_when_both_runs_evidence_them():
    _, traces = two_run_task()
    spans, ungrounded = ground_phrases(["cancel order", "late delivery"], traces, WRITES)
    assert ungrounded == []
    assert {s.trace_id for s in spans} == {"t1", "t2"}


def test_the_tool_name_is_not_part_of_the_evidence():
    """'delivered items' must not be grounded by the tool called exchange_delivered_order_items (D47)."""
    trace = make_trace(
        "t1",
        ["i would like to swap the shirt for a larger one"],
        [{"name": "exchange_delivered_order_items", "args": {"item_ids": ["4983901480"]}, "result": {"status": "exchange requested"}}],
    )
    texts = [s.text for s in span_candidates(trace, {"exchange_delivered_order_items"})]
    assert not any("delivered" in t for t in texts), texts
    assert any("4983901480" in t for t in texts)
    _, ungrounded = ground_phrases(["delivered items"], [trace], {"exchange_delivered_order_items"})
    assert ungrounded == ["delivered items"]


def test_a_written_value_span_is_what_the_write_changed():
    """The whole result JSON grounds phrases the write never touched, so the span is the diff (D47)."""
    trace = make_trace(
        "t1",
        ["cancel it please"],
        [
            {"name": "get_order", "args": {"order_id": "W1"}, "result": {"order_id": "W1", "status": "pending", "address": "elm street"}},
            {"name": "cancel_order", "args": {"order_id": "W1"}, "result": {"order_id": "W1", "status": "cancelled", "address": "elm street"}},
        ],
    )
    written = [s for s in span_candidates(trace, WRITES) if s.source == "written_value"]
    assert len(written) == 1
    assert "cancelled" in written[0].text
    assert "elm" not in written[0].text, written[0].text


def test_a_write_with_no_earlier_read_keeps_its_whole_result():
    trace = make_trace("t1", ["cancel it"], [{"name": "cancel_order", "args": {}, "result": {"status": "cancelled"}}])
    written = [s for s in span_candidates(trace, WRITES) if s.source == "written_value"]
    assert [s.text for s in written] == ['{"status": "cancelled"}']


def test_a_phrase_whose_words_are_scattered_is_not_grounded():
    """'late delivery' is not evidenced by 'the delivery was fine but the card was late' (D47)."""
    trace = make_trace("t1", ["the delivery was fine but the card was late"], [])
    spans, ungrounded = ground_phrases(["late delivery", "delivery card"], [trace], set())
    assert ungrounded == ["late delivery", "delivery card"]
    assert spans == []
    assert ground_phrases(["delivery"], [trace], set())[1] == []


def test_a_phrase_the_user_ruled_out_is_not_evidence():
    """'i do not want a full refund' is not evidence that the user wanted a full refund (D47)."""
    trace = make_trace("t1", ["i do not want a full refund, just cancel order W1"], [])
    spans, ungrounded = ground_phrases(["full refund", "cancel order"], [trace], set())
    assert ungrounded == ["full refund"]
    assert [s.phrase for s in spans] == ["cancel order"]


def test_a_span_names_the_run_and_the_text_it_points_at():
    _, traces = two_run_task()
    spans, _ = ground_phrases(["cancelled"], traces, WRITES)
    assert spans[0].trace_id == "t1"
    assert spans[0].source == "written_value"
    assert "cancelled" in spans[0].text


# --- write_intent ---


def test_write_intent_is_one_grounded_line(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["cancel the order because the delivery was late\nextra line the model added"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert isinstance(intent, Intent)
    assert intent.text == "cancel the order because the delivery was late"
    assert intent.grounded is True
    assert intent.ungrounded_phrases == []
    assert intent.task_id == "task_1"
    assert len(model.calls) == 1


def test_the_prompt_shows_the_member_runs_evidence(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["cancel the late order"])
    write_intent(model, task, traces, write_tools=WRITES)
    prompt = model.calls[0]["messages"][-1]["content"]
    assert "t1" in prompt and "t2" in prompt
    assert "cancel order W1" in prompt


def test_an_ungrounded_noun_phrase_refuses_the_intent(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["refund the order to a gift card"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.grounded is False
    assert "gift card" in intent.ungrounded_phrases
    assert intent.reason and "span" in intent.reason


def test_spans_from_one_run_only_refuse_a_multi_run_intent(make_test_model):
    traces = [cancel_trace("t1", "W1"), make_trace("t2", ["hello there"], [])]
    task = Task(id="task_1", run_ids=["t1", "t2"])
    model = make_test_model(["cancel order W1 because of the late delivery"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.grounded is False
    assert intent.reason and "cancel order w1" in intent.reason and "t2" in intent.reason
    assert {s.trace_id for s in intent.spans} == {"t1"}


def test_a_one_run_task_skips_the_cross_run_check_and_is_unguarded(make_test_model):
    traces = [cancel_trace("t1", "W1")]
    task = Task(id="task_1", run_ids=["t1"])
    model = make_test_model(["cancel order W1 because of the late delivery"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.grounded is True
    assert intent.unguarded is True
    assert {s.trace_id for s in intent.spans} == {"t1"}


def test_one_phrase_that_every_run_evidences_is_grounded(make_test_model):
    """The check is about the evidence, not about how many spans the chooser happened to pick."""
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["cancel"]), task, traces, write_tools=WRITES)
    assert intent.grounded is True, (intent.reason, [(s.phrase, s.trace_id) for s in intent.spans])
    assert intent.run_coverage == {"cancel": ["t1", "t2"]}


def test_an_intent_that_is_the_union_of_two_runs_is_refused(make_test_model):
    """t1 cancels, t2 changes an address: neither half is the Task's shared intent (D83)."""
    traces = [cancel_trace("t1", "W1"), make_trace("t2", ["please change shipping address on order W2"], [])]
    task = Task(id="task_1", run_ids=["t1", "t2"])
    intent = write_intent(make_test_model(["cancel order and change shipping address"]), task, traces, write_tools=WRITES)
    assert intent.ungrounded_phrases == []  # each half is evidenced, in one Run each
    assert intent.run_coverage == {"cancel order": ["t1"], "change shipping address": ["t2"]}
    assert intent.grounded is False, [(s.phrase, s.trace_id) for s in intent.spans]
    assert intent.reason and "cancel order (not in t2)" in intent.reason


def test_a_task_whose_member_traces_are_missing_is_an_error(make_test_model):
    """Two Run ids, one Trace handed in: not a single-Run Task, an incomplete call (D97, D81)."""
    task = Task(id="task_1", run_ids=["t1", "t2"])
    with pytest.raises(ValueError) as excinfo:
        write_intent(make_test_model(["cancel order W1 because of the late delivery"]), task, [cancel_trace("t1", "W1")], write_tools=WRITES)
    assert "t2" in str(excinfo.value)


def test_the_prompt_is_bounded_while_the_grounding_reads_every_run(make_test_model):
    """The Builder's calls obey the 40% context cap (D65), so the prompt samples the member Runs."""
    traces = [cancel_trace(f"t{i}", f"W{i}") for i in range(12)]
    task = Task(id="task_1", run_ids=[t.trace_id for t in traces])
    model = make_test_model(["cancel the order because the delivery was late"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    prompt = model.calls[0]["messages"][-1]["content"]
    named = {t.trace_id for t in traces if f"\n{t.trace_id} " in prompt}
    assert len(named) <= MAX_PROMPT_RUNS < len(traces)
    assert len(prompt) < 8000, len(prompt)
    assert intent.grounded is True
    assert intent.run_coverage["cancel"] == sorted(t.trace_id for t in traces)


def test_an_empty_model_reply_is_not_grounded(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["  "]), task, traces, write_tools=WRITES)
    assert intent.text == ""
    assert intent.grounded is False
    assert intent.reason == "the model returned no intent"


def test_the_model_name_travels_with_the_intent(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["cancel the order because the delivery was late"], name="scripted"), task, traces, write_tools=WRITES)
    assert intent.model == "scripted"


def test_a_task_with_no_member_traces_is_an_error(make_test_model):
    task = Task(id="task_1", run_ids=["missing"])
    with pytest.raises(ValueError):
        write_intent(make_test_model(["anything"]), task, [cancel_trace("t1", "W1")], write_tools=WRITES)


def test_write_intent_round_trips_through_json(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES)
    assert Intent.model_validate(intent.model_dump(mode="json", by_alias=True)) == intent


# --- applying the Intent to the Task ---


def test_apply_intent_sets_the_task_name_and_intent(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES)
    updated = apply_intent(task, intent)
    assert updated.intent == "cancel the order because the delivery was late"
    assert updated.name == "cancel the order because the delivery was late"
    assert task.intent is None


def test_apply_intent_leaves_an_ungrounded_intent_off_the_task(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["refund to a gift card"]), task, traces, write_tools=WRITES)
    updated = apply_intent(task, intent)
    assert updated.intent is None
    assert updated.name is None


def test_apply_intent_carries_the_unguarded_mark(make_test_model):
    traces = [cancel_trace("t1", "W1")]
    task = Task(id="task_1", run_ids=["t1"])
    intent = write_intent(make_test_model(["cancel order W1 because of the late delivery"]), task, traces, write_tools=WRITES)
    assert apply_intent(task, intent).unguarded is True
