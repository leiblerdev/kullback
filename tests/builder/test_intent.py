"""Tests for builder/intent.py: the one-line Intent, every noun phrase tied to a span (D47)."""

from __future__ import annotations

import pytest

from conftest import PTR
from kullback.builder.intent import (
    MAX_INTENT_ATTEMPTS,
    MAX_PROMPT_RUNS,
    Intent,
    apply_intent,
    ground_phrases,
    normalise,
    noun_phrases,
    span_candidates,
    span_mode,
    still_grounds,
    write_intent,
)
from kullback.runner.records import Task, ToolCall, Trace, Turn

WRITES = {"cancel_order"}


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


def test_noun_phrases_are_the_runs_of_non_stopword_tokens():
    assert noun_phrases("cancel the order because the delivery was late") == ["cancel", "order", "delivery", "late"]


def test_noun_phrases_keep_multi_word_phrases_and_drop_duplicates():
    phrases = noun_phrases("cancel a late order, cancel a late order")
    assert phrases == ["cancel", "late order"]


def test_noun_phrases_of_empty_text_is_empty():
    assert noun_phrases("") == []


# --- tokenising ---


def test_a_price_with_a_decimal_point_is_one_token():
    """A live build asked the evidence for "99 price difference": the splitter had cut "$17." away."""
    assert noun_phrases("credit the $17.99 price difference") == ["credit", "17.99 price difference"]
    trace = make_trace("t1", ["please credit the $17.99 price difference to me"], [])
    assert ground_phrases(["17.99 price difference"], [trace], set())[1] == []


def test_a_time_and_a_thousands_separator_survive_the_clause_splitter():
    trace = make_trace("t1", ["the 1,200 point voucher expired at 09:30 today"], [])
    assert ground_phrases(["1,200 point voucher"], [trace], set())[1] == []
    assert ground_phrases(["09:30"], [trace], set())[1] == []


def test_an_order_code_is_one_token():
    trace = make_trace("t1", ["can you look at order #G4471902 and item 9844888101 for me"], [])
    assert noun_phrases("check order #G4471902") == ["check", "order", "g4471902"]
    assert ground_phrases(["order g4471902", "item 9844888101"], [trace], set())[1] == []


def test_a_verb_and_its_object_are_separate_phrases():
    """A customer writes "order change" and "trellis planter" a sentence apart, so asking the
    evidence for "change trellis planter" in that order in one clause asks for a sentence nobody wrote."""
    assert noun_phrases("change trellis planter to brass on order W4471902") == [
        "change", "trellis planter", "brass", "order w4471902"]


def test_an_ing_word_is_a_modifier_and_does_not_split_its_phrase():
    assert noun_phrases("change the shipping address") == ["change", "shipping address"]


def test_a_hyphen_or_a_slash_joins_the_words_it_sits_between():
    assert noun_phrases("issue a gift-card/voucher") == ["issue", "gift card voucher"]
    trace = make_trace("t1", ["I want a gift-card, not store credit"], [])
    assert ground_phrases(["gift card"], [trace], set())[1] == []


def test_a_possessive_and_a_contraction_lose_their_apostrophe():
    assert noun_phrases("update the customer's default address") == ["update", "customers default address"]
    trace = make_trace("t1", ["I'd like the customer's default address updated"], [])
    assert ground_phrases(["customers default address"], [trace], set())[1] == []


def test_a_plural_and_its_singular_are_the_same_word():
    assert normalise("earbuds") == normalise("earbud")
    assert normalise("addresses") == normalise("address")
    assert normalise("policies") == normalise("policy")
    assert normalise("shipped") == normalise("shipping") == normalise("ship")
    assert normalise("w4471902") == "w4471902", "an id carries digits and is never stemmed"
    trace = make_trace("t1", ["two trellis planters arrived cracked"], [])
    assert ground_phrases(["trellis planter"], [trace], set())[1] == []


def test_an_underscored_id_is_reached_by_the_words_it_spells():
    trace = make_trace("t1", ["cancel it"],
                       [{"name": "cancel_order", "args": {"row": "gift_card_4471"}}])
    assert ground_phrases(["gift card"], [trace], WRITES)[1] == []
    assert ground_phrases(["gift_card_4471"], [trace], WRITES)[1] == []


# --- spans ---


def test_span_candidates_cover_user_words_tool_arguments_and_written_values():
    trace = cancel_trace("t1", "W1")
    sources = {s.source for s in span_candidates(trace, WRITES)}
    assert sources == {"user_utterance", "tool_arg", "written_value"}


def test_a_result_of_a_tool_that_is_not_a_write_is_not_a_span():
    trace = make_trace(
        "t1", ["where is it"], [{"name": "get_order", "args": {"order_id": "W1"}, "result": {"status": "delivered"}}]
    )
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
        [
            {
                "name": "exchange_delivered_order_items",
                "args": {"item_ids": ["4983901480"]},
                "result": {"status": "exchange requested"},
            }
        ],
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
            {
                "name": "get_order",
                "args": {"order_id": "W1"},
                "result": {"order_id": "W1", "status": "pending", "address": "elm street"},
            },
            {
                "name": "cancel_order",
                "args": {"order_id": "W1"},
                "result": {"order_id": "W1", "status": "cancelled", "address": "elm street"},
            },
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


def test_a_phrase_whose_words_the_user_said_in_another_order_is_grounded_by_its_tokens():
    """The words are the customer's; only the arrangement is the model's, and that is not an invention."""
    trace = make_trace("t1", ["the delivery was fine but the card was late"], [])
    spans, ungrounded = ground_phrases(["late delivery"], [trace], set())
    assert ungrounded == []
    assert [span_mode(s) for s in spans] == ["tokens"]


def test_a_phrase_with_a_word_no_run_says_stays_ungrounded():
    """Token grounding relaxes the order of the words, never the requirement that they be said (D47)."""
    trace = make_trace("t1", ["the delivery was fine but the card was late"], [])
    spans, ungrounded = ground_phrases(["late hamper delivery"], [trace], set())
    assert ungrounded == ["late hamper delivery"]
    assert spans == []


def test_a_span_that_said_the_whole_phrase_is_marked_as_a_phrase():
    trace = make_trace("t1", ["the late delivery ruined it"], [])
    spans, _ = ground_phrases(["late delivery"], [trace], set())
    assert [span_mode(s) for s in spans] == ["phrase"]


def test_a_word_the_user_ruled_out_does_not_ground_a_phrase_by_its_tokens():
    """"I do not want a voucher" is not evidence that the user wanted one, whichever way it is read."""
    trace = make_trace("t1", ["I do not want a voucher, just replace the cracked planter"], [])
    spans, ungrounded = ground_phrases(["voucher planter"], [trace], set())
    assert ungrounded == ["voucher planter"]
    assert spans == []


def test_the_words_of_a_phrase_have_to_be_evidenced_in_one_run_not_across_two():
    traces = [make_trace("t1", ["the trellis planter is cracked"], []),
              make_trace("t2", ["my brass watering can leaks"], [])]
    spans, ungrounded = ground_phrases(["brass planter"], traces, set())
    assert ungrounded == ["brass planter"]
    assert spans == []


def test_a_phrase_the_user_ruled_out_is_not_evidence():
    """'i do not want a full refund' is not evidence that the user wanted a full refund (D47)."""
    trace = make_trace("t1", ["i do not want a full refund, just cancel order W1"], [])
    spans, ungrounded = ground_phrases(["full refund", "cancel order"], [trace], set())
    assert ungrounded == ["full refund"]
    assert [s.phrase for s in spans] == ["cancel order"]


def test_a_span_names_the_run_and_the_text_it_points_at():
    _, traces = two_run_task()
    spans, _ = ground_phrases(["status"], traces, WRITES)
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
    assert len(model.calls) == 1, "a grounded line is the answer; nothing is asked twice"


# --- the rewrite loop ---


def test_an_intent_with_an_ungrounded_phrase_is_rewritten_with_the_phrases_named(make_test_model):
    """The first live builds refused 160 of 205 Tasks and never told the model which words were the
    problem. The second prompt names them and asks for a line the evidence supports."""
    task, traces = two_run_task()
    model = make_test_model(["refund the order to a gift card",
                             "cancel the order because the delivery was late"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    second = model.calls[1]["messages"][-1]["content"]
    assert "gift card" in second and "no span in the evidence" in second
    assert 'Your last line was: "refund the order to a gift card"' in second
    assert "only words that appear in the evidence" in second
    assert intent.grounded is True and intent.text == "cancel the order because the delivery was late"
    assert len(model.calls) == 2, "the rewrite grounded, so no third attempt"


def test_a_phrase_missing_from_one_run_is_named_with_the_run_that_lacks_it(make_test_model):
    traces = [cancel_trace("t1", "W1"), make_trace("t2", ["please change shipping address on order W2"], [])]
    task = Task(id="task_1", run_ids=["t1", "t2"])
    model = make_test_model(["cancel order and change shipping address"], loop=True)
    write_intent(model, task, traces, write_tools=WRITES)
    second = model.calls[1]["messages"][-1]["content"]
    assert "not evidenced in every run" in second
    assert "cancel (not in t2)" in second


def test_token_grounding_still_has_to_hold_in_every_run(make_test_model):
    """A Task's Intent is what all its Runs show (D83). Reading a phrase word by word relaxes the
    order of the words, not the Run that says none of them."""
    traces = [make_trace("t1", ["the planter arrived cracked and I want it replaced"], []),
              make_trace("t2", ["I want the cracked planter replaced please"], []),
              make_trace("t3", ["my brass watering can leaks"], [])]
    task = Task(id="task_1", run_ids=["t1", "t2", "t3"])
    intent = write_intent(make_test_model(["replaced cracked planter"], loop=True), task, traces, write_tools=set())
    assert intent.ungrounded_phrases == []
    assert intent.run_coverage == {"replaced": ["t1", "t2"], "cracked planter": ["t1", "t2"]}
    assert intent.grounded is False
    assert intent.reason and "t3" in intent.reason


def test_the_rewrite_prompt_names_the_words_found_and_the_words_missing(make_test_model):
    """Naming the phrase alone leaves the model guessing at a whole line; naming its words does not."""
    task, traces = two_run_task()
    model = make_test_model(["cancel the late hamper"], loop=True)
    write_intent(model, task, traces, write_tools=WRITES)
    second = model.calls[1]["messages"][-1]["content"]
    assert 'In "late hamper": the evidence shows late; no run says hamper.' in second


def test_the_best_of_three_attempts_is_kept(make_test_model):
    """Three refused lines: the one with the fewest phrases the evidence cannot show is the Intent."""
    task, traces = two_run_task()
    model = make_test_model(["refund the order to a gift card by paypal",  # gift card, paypal
                             "cancel the order to a gift card",           # gift card
                             "refund the order to a gift card by paypal"])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert len(model.calls) == MAX_INTENT_ATTEMPTS
    assert intent.grounded is False
    assert intent.text == "cancel the order to a gift card"
    assert intent.ungrounded_phrases == ["gift card"]


def test_a_line_at_all_beats_an_empty_reply(make_test_model):
    """An empty reply grounds nothing and has no phrase to be ungrounded, so it must not win on count."""
    task, traces = two_run_task()
    model = make_test_model(["refund to a gift card", "   ", "  "])
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.text == "refund to a gift card"
    assert intent.reason and "gift card" in intent.reason


def test_a_repair_hint_is_in_every_attempt_of_the_prompt(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["refund to a gift card"], loop=True)
    write_intent(model, task, traces, write_tools=WRITES, hint="say what the user asked for, not the refund method")
    prompts = [call["messages"][-1]["content"] for call in model.calls]
    assert len(prompts) == MAX_INTENT_ATTEMPTS
    assert all("A repair asks for this: say what the user asked for, not the refund method" in p for p in prompts)


def test_the_prompt_shows_the_member_runs_evidence(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["cancel the late order"], loop=True)
    write_intent(model, task, traces, write_tools=WRITES)
    prompt = model.calls[0]["messages"][-1]["content"]
    assert "t1" in prompt and "t2" in prompt
    assert "cancel order W1" in prompt


def test_an_ungrounded_noun_phrase_refuses_the_intent(make_test_model):
    task, traces = two_run_task()
    model = make_test_model(["refund the order to a gift card"], loop=True)
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.grounded is False
    assert "gift card" in intent.ungrounded_phrases
    assert intent.reason and "span" in intent.reason


def test_spans_from_one_run_only_refuse_a_multi_run_intent(make_test_model):
    traces = [cancel_trace("t1", "W1"), make_trace("t2", ["hello there"], [])]
    task = Task(id="task_1", run_ids=["t1", "t2"])
    model = make_test_model(["cancel order W1 because of the late delivery"], loop=True)
    intent = write_intent(model, task, traces, write_tools=WRITES)
    assert intent.grounded is False
    assert intent.reason and "order w1" in intent.reason and "t2" in intent.reason
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
    intent = write_intent(
        make_test_model(["cancel order and change shipping address"], loop=True), task, traces, write_tools=WRITES
    )
    assert intent.ungrounded_phrases == []  # each half is evidenced, in one Run each
    assert intent.run_coverage == {"cancel": ["t1"], "order": ["t1", "t2"],
                                   "change": ["t2"], "shipping address": ["t2"]}
    assert intent.grounded is False, [(s.phrase, s.trace_id) for s in intent.spans]
    assert intent.reason and "cancel (not in t2)" in intent.reason


def test_a_task_whose_member_traces_are_missing_is_an_error(make_test_model):
    """Two Run ids, one Trace handed in: not a single-Run Task, an incomplete call (D97, D81)."""
    task = Task(id="task_1", run_ids=["t1", "t2"])
    with pytest.raises(ValueError) as excinfo:
        write_intent(
            make_test_model(["cancel order W1 because of the late delivery"]),
            task,
            [cancel_trace("t1", "W1")],
            write_tools=WRITES,
        )
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
    intent = write_intent(make_test_model(["  "], loop=True), task, traces, write_tools=WRITES)
    assert intent.text == ""
    assert intent.grounded is False
    assert intent.reason == "the model returned no intent"


def test_the_model_name_travels_with_the_intent(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(
        make_test_model(["cancel the order because the delivery was late"], name="scripted"),
        task,
        traces,
        write_tools=WRITES,
    )
    assert intent.model == "scripted"


def test_a_task_with_no_member_traces_is_an_error(make_test_model):
    task = Task(id="task_1", run_ids=["missing"])
    with pytest.raises(ValueError):
        write_intent(make_test_model(["anything"]), task, [cancel_trace("t1", "W1")], write_tools=WRITES)


def test_write_intent_round_trips_through_json(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(
        make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES
    )
    assert Intent.model_validate(intent.model_dump(mode="json", by_alias=True)) == intent


# --- whether a recorded Intent still holds for its Task ---


def test_a_grounded_intent_still_holds_for_the_runs_it_was_written_over(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(
        make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES
    )
    assert intent.grounded
    assert still_grounds(intent, task.run_ids) is True


def test_an_intent_stops_holding_when_its_task_gains_or_loses_a_run(make_test_model):
    """A grounded line says what every member Run showed. A Run that has joined the Task since was
    never read for it, and a Run that has left took its share of the evidence with it, so either
    way the line has to be written again rather than kept."""
    task, traces = two_run_task()
    intent = write_intent(
        make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES
    )
    assert still_grounds(intent, ["t1", "t2", "t3"]) is False
    assert still_grounds(intent, ["t1"]) is False


def test_an_ungrounded_intent_never_holds_however_wide_its_coverage():
    """Coverage alone is not the answer: a refused line can still name every Run for the phrases it
    did place, and keeping it would be a build ratcheting onto a Task with no Verdict."""
    refused = Intent(task_id="task_1", text="refund to a gift card", grounded=False,
                     run_coverage={"a refund": ["t1", "t2"]},
                     ungrounded_phrases=["a gift card"])
    assert still_grounds(refused, ["t1", "t2"]) is False


# --- applying the Intent to the Task ---


def test_apply_intent_sets_the_task_name_and_intent(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(
        make_test_model(["cancel the order because the delivery was late"]), task, traces, write_tools=WRITES
    )
    updated = apply_intent(task, intent)
    assert updated.intent == "cancel the order because the delivery was late"
    assert updated.name == "cancel the order because the delivery was late"
    assert task.intent is None


def test_apply_intent_leaves_an_ungrounded_intent_off_the_task(make_test_model):
    task, traces = two_run_task()
    intent = write_intent(make_test_model(["refund to a gift card"], loop=True), task, traces, write_tools=WRITES)
    updated = apply_intent(task, intent)
    assert updated.intent is None
    assert updated.name is None


def test_apply_intent_carries_the_unguarded_mark(make_test_model):
    traces = [cancel_trace("t1", "W1")]
    task = Task(id="task_1", run_ids=["t1"])
    intent = write_intent(
        make_test_model(["cancel order W1 because of the late delivery"]), task, traces, write_tools=WRITES
    )
    assert apply_intent(task, intent).unguarded is True


def test_the_user_wanted_frame_is_stripped_before_grounding():
    from kullback.builder.intent import strip_frame

    assert strip_frame("The user wanted help to cancel order #W1.") == "cancel order #W1."
    assert strip_frame("Wanted to fix sending picture messages") == "fix sending picture messages"
    assert strip_frame("The customer asked for a refund on #W2") == "a refund on #W2"
    assert strip_frame("cancel order #W1") == "cancel order #W1"
    assert strip_frame("The user wanted") == "The user wanted"
