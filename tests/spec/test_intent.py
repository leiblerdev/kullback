"""The intent miner reads user turns only, keeps facts in the user's words, and writes the Spec."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.spec import intent_tools as T
from kullback.spec.intent import intent_system_prompt, mine_intent
from kullback.spec.schema import Spec, load_spec, save_spec
from tests.helpers.scripted import tool_reply_as
from tests.spec.fixtures import check, spec

SECRET_RESULT = "row value only the tool knew"
SECRET_ARG = "argument only the agent passed"


def _turn(idx, role, content=None, calls=()):
    return {"idx": idx, "role": role, "content": content, "tool_call_ids": list(calls),
            "raw_ptr": {"file_hash": "h"}}


def _trace(trace_id, turns):
    return {"trace_id": trace_id, "raw_hash": "h", "ingest_version": "v", "source": "s", "turns": turns,
            "tool_calls": [{"id": "c1", "trace_id": trace_id, "name": "lookup", "args": {"q": SECRET_ARG},
                            "result": SECRET_RESULT, "requestor": "assistant",
                            "raw_ptr": {"file_hash": "h"}}],
            "raw_ptr": {"file_hash": "h"}}


RECORDINGS = {
    "rec-a": [_turn(0, "assistant", "Hello, how can I help?"),
              _turn(1, "user", "Please move my blue lamp order to the Oak street address."),
              _turn(2, "assistant", None, calls=["c1"]),
              _turn(3, "tool", SECRET_RESULT, calls=["c1"]),
              _turn(4, "assistant", "Shall I also refund the shipping fee to your card?"),
              _turn(5, "user", "Yes please.")],
    "rec-b": [_turn(0, "user", "Hi, I need the blue lamp order sent to Oak street instead."),
              _turn(1, "assistant", "Do you want the shipping fee refunded to your card?"),
              _turn(2, "user", "Sure, refund the shipping fee to my card.")],
    "rec-c": [_turn(0, "user", "Also refund the shipping fee to my card, and move the blue lamp to Oak street.")],
}


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "traces").mkdir()
    task = {"id": "t1", "run_ids": list(RECORDINGS)}
    (tmp_path / "tasks" / "t1.json").write_text(json.dumps(task))
    for trace_id, turns in RECORDINGS.items():
        (tmp_path / "traces" / f"{trace_id}.json").write_text(json.dumps(_trace(trace_id, turns)))
    return tmp_path


def _state(workdir):
    return T.MinerState(found=T.recordings_of(workdir, "t1"))


def _run(tool, **args):
    return asyncio.run(tool.run(args))


def _tools(state):
    return {tool.name: tool for tool in T.miner_tools(state)}


# --- what the miner is shown -----------------------------------------------------------------------

def test_user_turns_show_the_user_text_and_the_agent_line_before_and_never_a_tool_call_or_result(workdir):
    result = _run(_tools(_state(workdir))["user_turns"], recording="rec-a")
    assert not result.is_error
    assert SECRET_RESULT not in result.content and SECRET_ARG not in result.content
    rows = result.details["turns"]
    assert [row["turn"] for row in rows] == [1, 5]
    assert rows[0]["agent_before"] == "Hello, how can I help?"
    assert rows[1]["agent_before"] == "Shall I also refund the shipping fee to your card?"


def test_list_recordings_names_every_recording_of_the_task_with_its_user_turn_count(workdir):
    result = _run(_tools(_state(workdir))["list_recordings"], task_id="t1")
    assert {row["recording"]: row["user_turns"] for row in result.details["recordings"]} == \
        {"rec-a": 2, "rec-b": 2, "rec-c": 1}


def test_a_task_that_is_not_under_tasks_is_refused_by_name(workdir):
    with pytest.raises(ValueError, match="unknown Task"):
        T.recordings_of(workdir, "../t1")


# --- what a fact must carry ------------------------------------------------------------------------

def _add(state, **fact):
    return _run(_tools(state)["add_facts"], facts=[{"kind": "do", **fact}])


def test_a_fact_whose_words_the_named_turn_does_not_say_is_refused_with_the_missing_words(workdir):
    state = _state(workdir)
    result = _add(state, text="wants a refund of the green chair", recording="rec-a", turn=1, stance="volunteered")
    refusal = result.details["rows"][0]["refused"]
    assert "green" in refusal and "chair" in refusal and state.facts == []


def test_a_volunteered_fact_is_kept_with_its_source_and_the_recordings_that_volunteer_it_too(workdir):
    state = _state(workdir)
    result = _add(state, text="move the blue lamp order to Oak street", recording="rec-a", turn=1,
                  stance="volunteered")
    assert result.details["rows"][0]["fact_id"] == "f1"
    fact = state.facts[0]
    assert (fact.source.recording, fact.source.turn, fact.stance) == ("rec-a", 1, "volunteered")
    assert fact.witnesses == ["rec-b", "rec-c"]
    assert T.counts(fact)


def test_a_yes_to_the_agent_offer_cannot_be_kept_as_volunteered(workdir):
    state = _state(workdir)
    result = _add(state, text="refund the shipping fee to the card", recording="rec-a", turn=5, stance="volunteered")
    assert "mark it accepted" in result.details["rows"][0]["refused"] and state.facts == []


OFFER = {"turn": 5, "quote": "refund the shipping fee to your card"}


def test_an_accepted_fact_rests_on_the_offer_and_counts_only_when_another_user_volunteers_it(workdir):
    state = _state(workdir)
    _add(state, text="refund the shipping fee to the card", recording="rec-a", turn=5, stance="accepted",
         in_reply_to=OFFER)
    fact = state.facts[0]
    # rec-b only echoes the agent's offer; rec-c raises it unasked.
    assert fact.witnesses == ["rec-c"]
    assert T.counts(fact)


def test_an_accepted_fact_no_other_user_volunteers_is_kept_but_does_not_count(workdir):
    state = _state(workdir)
    state.found.recordings.pop("rec-c")
    result = _add(state, text="refund the shipping fee to the card", recording="rec-a", turn=5, stance="accepted",
                  in_reply_to=OFFER)
    assert "1 accepted with no volunteering witness, kept and marked" in result.content
    assert state.facts[0].witnesses == [] and not T.counts(state.facts[0])


def test_one_call_keeps_the_good_facts_and_refuses_a_repeat_and_a_turn_that_is_no_user_turn(workdir):
    state = _state(workdir)
    first = dict(text="move the blue lamp order to Oak street", recording="rec-a", turn=1, stance="volunteered",
                 kind="do")
    result = _run(_tools(state)["add_facts"], facts=[first, first, {**first, "turn": 3}])
    rows = result.details["rows"]
    assert rows[0]["fact_id"] == "f1"
    assert "already kept" in rows[1]["refused"] and "not a user turn" in rows[2]["refused"]
    assert len(state.facts) == 1 and "2 refused" in result.content


# --- the session -----------------------------------------------------------------------------------

def test_the_miner_reads_the_recordings_keeps_facts_and_writes_them_to_the_spec_keeping_its_checks(workdir):
    save_spec(workdir, spec(checks=[check()]).model_copy(update={"task_id": "t1"}))
    model = TestModel([
        tool_reply_as("a", "list_recordings", {"task_id": "t1"}),
        tool_reply_as("b", "user_turns", {"recording": "rec-a"}),
        tool_reply_as("c", "add_facts", {"facts": [{"text": "move the blue lamp order to Oak street",
                                                    "recording": "rec-a", "turn": 1, "stance": "volunteered",
                                                    "kind": "do"}]}),
        ModelReply(content="1 fact kept."),
    ])
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    assert out.tool_calls == {"list_recordings": 1, "user_turns": 1, "add_facts": 1}
    assert out.stopped is None
    written = load_spec(workdir, "t1")
    assert isinstance(written, Spec) and len(written.checks) == 1 and written.version == 1
    assert [f.text for f in written.intent.facts] == ["move the blue lamp order to Oak street"]
    assert written.intent.text == "move the blue lamp order to Oak street"
    assert SECRET_RESULT not in json.dumps(model.calls[-1]["messages"])


def test_the_call_past_the_cap_is_blocked_and_the_session_stops_saying_so(workdir):
    model = TestModel([ModelReply(content=None, tool_calls=[
        ToolCallRequest(id="a", name="list_recordings", arguments={"task_id": "t1"}),
        ToolCallRequest(id="b", name="user_turns", arguments={"recording": "rec-a"})]),
        ModelReply(content="done")], loop=True)
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0, max_calls=1)
    assert out.tool_calls == {"list_recordings": 1, "user_turns": 1}
    assert out.stopped == "call cap 1 reached"


def test_a_spent_ceiling_stops_the_session_after_the_call_that_reached_it(workdir):
    model = TestModel([tool_reply_as("a", "list_recordings", {"task_id": "t1"})], loop=True)
    out = mine_intent(workdir, "t1", model, ceiling_usd=0.0)
    assert out.stopped == "ceiling 0.00 USD reached"
    assert len(model.calls) == 1
    assert load_spec(workdir, "t1").intent.facts == []


def test_the_prompt_names_each_tool_with_an_example_and_ends_on_the_stop_rule():
    prompt = intent_system_prompt()
    for name in T.tool_names():
        assert f"{name}({{" in prompt
    assert prompt.split("\n\n")[-1].startswith("# When to stop")


def test_the_cap_stops_the_session_and_the_facts_kept_before_it_are_saved(workdir):
    kept = {"text": "move the blue lamp order to Oak street", "recording": "rec-a", "turn": 1,
            "stance": "volunteered", "kind": "do"}
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [kept]}),
                       tool_reply_as("b", "user_turns", {"recording": "rec-b"})], loop=True)
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0, max_calls=1)
    assert out.stopped == "call cap 1 reached" and out.tool_calls == {"add_facts": 1, "user_turns": 1}
    assert [f.text for f in load_spec(workdir, "t1").intent.facts] == [kept["text"]]


def test_refused_facts_are_counted_on_the_result_and_in_the_spec_file_as_a_number(workdir):
    bad = {"text": "wants a green chair", "recording": "rec-a", "turn": 1, "stance": "volunteered", "kind": "do"}
    good = {**bad, "text": "move the blue lamp order to Oak street"}
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [bad, good, good]}), ModelReply(content="1 kept.")])
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    written = load_spec(workdir, "t1")
    assert out.refusals == 2 and written.intent.refusals == 2
    assert "green chair" not in (workdir / "spec" / "t1.json").read_text()


def _pushback_state():
    """Two user turns: the ask, then a fallback after the agent says it cannot do the ask."""
    return T.MinerState(found=T.TaskRecordings(task_id="t", recordings={"rec": [
        T.UserTurn(index=0, text="Please move my blue lamp order to Oak street.", agent_before="Hello."),
        T.UserTurn(index=2, text="Then cancel the blue lamp order.",
                   agent_before="I cannot move the lamp to that street.")]}))


REFUSAL = {"turn": 2, "quote": "I cannot  move the lamp"}


def _fallback(state, condition="agent", **fact):
    return _add(state, **{"text": "Then cancel the blue lamp order.", "recording": "rec", "turn": 2,
                          "stance": "fallback", "condition": condition, "in_reply_to": REFUSAL, **fact})


def test_a_fact_the_miner_marks_after_pushback_keeps_the_flag():
    state = _pushback_state()
    result = _fallback(state, after_pushback=True)
    assert result.details["rows"][0]["fact_id"] == "f1" and state.facts[0].after_pushback is True


# --- position and condition (H1, D366) ------------------------------------------------------------

@pytest.mark.parametrize("condition", ["world_or_policy", "agent"])
def test_a_fact_after_a_quoted_agent_line_is_kept_as_fallback_with_its_quote_and_condition(condition):
    state = _pushback_state()
    _fallback(state, condition=condition)
    fact = state.facts[0]
    assert (fact.stance, fact.condition, state.fallback_unquoted) == ("fallback", condition, 0)
    assert fact.in_reply_to.turn == 2 and fact.in_reply_to.quote == "I cannot move the lamp"


@pytest.mark.parametrize("kind", ["keep", "told"])
def test_a_keep_or_told_fact_in_reply_to_a_decline_is_a_fallback_too(kind):
    state = _pushback_state()
    _fallback(state, kind=kind)
    assert (state.facts[0].kind, state.facts[0].stance) == (kind, "fallback")


@pytest.mark.parametrize("stance, condition", [("fallback", "none"), ("fallback", "asked"),
                                               ("volunteered", "agent"), ("accepted", "world_or_policy")])
def test_a_fallback_without_a_fallback_condition_or_the_reverse_is_refused(stance, condition):
    state = _pushback_state()
    result = _fallback(state, stance=stance, condition=condition)
    assert "is stance fallback" in result.details["rows"][0]["refused"] and state.facts == []


@pytest.mark.parametrize("reply", [None, {"turn": 2, "quote": "we never sell lamps"},
                                   {"turn": 0, "quote": "I cannot move the lamp"}],
                         ids=["no quote", "quote in no agent line", "quote from another turn's line"])
def test_a_late_fact_whose_quote_is_not_found_is_sent_back_once_then_kept_as_a_reaction(reply):
    state = _pushback_state()
    first = _fallback(state, condition="world_or_policy", in_reply_to=reply)
    assert "not the user's first turn" in first.details["rows"][0]["refused"] and state.facts == []
    _fallback(state, condition="world_or_policy", in_reply_to=reply)
    fact = state.facts[0]
    assert (fact.condition, fact.in_reply_to) == ("agent", None) and T.is_reaction(fact)
    assert len(state.sent_back) == 1 and state.late_unplaced == 1


def test_a_late_volunteered_fact_with_its_quote_is_kept_as_asked():
    state = _pushback_state()
    _add(state, text="Then cancel the blue lamp order.", recording="rec", turn=2, stance="volunteered",
         in_reply_to=REFUSAL)
    assert (state.facts[0].stance, state.facts[0].condition) == ("volunteered", "asked")


def test_a_fallback_in_the_first_turn_with_no_quote_is_kept_volunteered_and_counted():
    state = _pushback_state()
    _fallback(state, text="Please move my blue lamp order to Oak street.", turn=0,
              in_reply_to={"turn": 2, "quote": "I cannot move the lamp"})
    fact = state.facts[0]
    assert (fact.stance, fact.condition, state.fallback_unquoted) == ("volunteered", "none", 1)


def test_a_reaction_another_recording_volunteers_before_any_tool_call_is_promoted():
    turns = [T.UserTurn(index=0, text="Please move my blue lamp order to Oak street.", agent_before="Hello."),
             T.UserTurn(index=2, text="Then cancel the blue lamp order.", agent_before="I cannot move the lamp.")]
    found = T.TaskRecordings(task_id="t", recordings={
        "rec": turns,
        "opens": [T.UserTurn(index=0, text="Cancel the blue lamp order please.", agent_before="Hi.")],
        "late": [T.UserTurn(index=4, text="Cancel the blue lamp order please.", agent_before="Done.",
                            opening=False)]})
    state = T.MinerState(found=found)
    _add(state, text="Then cancel the blue lamp order.", recording="rec", turn=2, stance="fallback",
         condition="agent", in_reply_to=REFUSAL)
    promoted, count = T.promote(state.facts, found)
    assert count == 1 and (promoted[0].condition, promoted[0].stance) == ("none", "volunteered")
    found.recordings.pop("opens")
    assert T.promote(state.facts, found)[1] == 0


def test_user_turns_mark_a_turn_after_an_agent_tool_call_as_not_opening(workdir):
    turns = T.recordings_of(workdir, "t1").recordings["rec-a"]
    assert [(turn.index, turn.opening) for turn in turns] == [(1, True), (5, False)]


def test_the_saved_intent_counts_unquoted_fallbacks_and_a_task_with_none_saves_neither_key(workdir):
    fact = {"text": "Please move my blue lamp order to the Oak street address.", "recording": "rec-a",
            "turn": 1, "stance": "fallback", "condition": "agent", "kind": "do",
            "in_reply_to": {"turn": 1, "quote": "no such words"}}
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [fact]}), ModelReply(content="1 kept.")])
    mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    saved = json.loads((workdir / "spec" / "t1.json").read_text())["intent"]
    assert saved["fallback_unquoted"] == 1 and saved["facts"][0]["stance"] == "volunteered"
    assert "in_reply_to" not in saved["facts"][0] and "condition" not in saved["facts"][0]
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [dict(fact, stance="volunteered", condition="none")]}),
                       ModelReply(content="1 kept.")])
    (workdir / "spec" / "t1.json").unlink()
    mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    saved = json.loads((workdir / "spec" / "t1.json").read_text())["intent"]
    assert not {"fallback_unquoted", "late_sent_back", "late_unplaced", "promoted"} & set(saved)


def test_the_saved_intent_counts_late_facts_sent_back_kept_unplaced_and_promoted(workdir):
    late = {"text": "refund the shipping fee to the card", "recording": "rec-a", "turn": 5,
            "stance": "accepted", "kind": "do"}
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [late]}), tool_reply_as("b", "add_facts", {"facts": [late]}),
                       ModelReply(content="1 kept.")])
    mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    saved = json.loads((workdir / "spec" / "t1.json").read_text())["intent"]
    # rec-c opens with the same ask, so the reaction is promoted to the ask it is there.
    assert (saved["late_sent_back"], saved["late_unplaced"], saved["promoted"]) == (1, 1, 1)
    assert "condition" not in saved["facts"][0] and saved["facts"][0]["stance"] == "volunteered"


def test_the_prompt_names_each_condition_and_says_code_checks_the_quote():
    flat = " ".join(intent_system_prompt().split())
    for condition in ("asked:", "world_or_policy:", "agent:"):
        assert condition in flat
    assert "has the stance fallback, whatever its kind" in flat
    assert "a later fact without one is sent back once" in flat


def test_the_prompt_marks_a_fact_after_pushback_in_one_general_sentence():
    sentence = ("Mark a fact after_pushback too when the user says it in reply to the agent declining, failing, "
                "or offering another way, before what the user first asked for was met.")
    flat = " ".join(intent_system_prompt().split())
    assert sentence in flat and flat.count("after_pushback") == 1


# --- side facts (D361) -----------------------------------------------------------------------------

def _fact(fact_id, kind, turn=0):
    return T.IntentFact(id=fact_id, text=f"fact {fact_id}", stance="volunteered", kind=kind,
                        source=T.FactSource(recording="rec", turn=turn))


def test_a_told_fact_in_the_opening_turn_beside_a_later_write_stays_core():
    facts = T.mark_side([_fact("f1", "told", turn=0), _fact("f2", "detail", turn=1), _fact("f3", "do", turn=2)])
    assert not any(fact.side for fact in facts)


def test_a_told_fact_after_the_opening_turn_beside_a_write_is_side():
    facts = T.mark_side([_fact("f1", "do", turn=0), _fact("f2", "told", turn=0), _fact("f3", "told", turn=3)])
    assert [fact.side for fact in facts] == [False, False, True]


def test_a_greeting_turn_before_the_ask_does_not_make_the_asks_told_facts_side():
    facts = T.mark_side([_fact("f1", "detail", turn=0), _fact("f2", "told", turn=2), _fact("f3", "do", turn=4)])
    assert T.opening_turns(facts) == {"rec": 2} and not facts[1].side


def test_a_told_fact_after_the_opening_turn_beside_an_ask_to_leave_something_undone_is_side():
    facts = T.mark_side([_fact("f1", "keep", turn=0), _fact("f2", "told", turn=1)])
    assert [fact.side for fact in facts] == [False, True]


def test_each_recording_has_its_own_opening_turn():
    later = _fact("f3", "told", turn=0).model_copy(update={"source": T.FactSource(recording="rec-b", turn=5)})
    facts = T.mark_side([_fact("f1", "do", turn=0), _fact("f2", "told", turn=3), later])
    assert [fact.side for fact in facts] == [False, True, False]


def test_a_task_whose_only_asks_are_told_facts_has_no_side_facts():
    facts = T.mark_side([_fact("f1", "detail"), _fact("f2", "told"), _fact("f3", "told", turn=4)])
    assert not any(fact.side for fact in facts)


def test_a_fact_mined_before_kinds_were_read_is_never_side_and_saves_without_kind_or_side():
    facts = T.mark_side([_fact("f1", None), _fact("f2", "do")])
    assert not any(fact.side for fact in facts)
    assert set(facts[0].model_dump(mode="json")) == {"id", "text", "source", "stance", "witnesses", "echo",
                                                     "after_pushback", "unfounded"}


def test_the_miner_keeps_each_facts_kind_and_code_sets_side_on_the_saved_intent(workdir):
    move = {"text": "move the blue lamp order to Oak street", "recording": "rec-a", "turn": 1,
            "stance": "volunteered", "kind": "do"}
    sent = {"text": "need the blue lamp order sent to Oak street instead", "recording": "rec-b", "turn": 0,
            "stance": "volunteered", "kind": "do"}
    fee = {"text": "refund the shipping fee to my card", "recording": "rec-b", "turn": 2,
           "stance": "accepted", "kind": "told", "in_reply_to": {"turn": 2, "quote": "shipping fee refunded"}}
    model = TestModel([tool_reply_as("a", "add_facts", {"facts": [move, sent, fee]}), ModelReply(content="3 kept.")])
    mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    saved = load_spec(workdir, "t1").intent.facts
    assert [(fact.kind, fact.side) for fact in saved] == [("do", False), ("do", False), ("told", True)]


def test_the_miner_may_not_set_side_itself(workdir):
    state = _state(workdir)
    result = _add(state, text="move the blue lamp order to Oak street", recording="rec-a", turn=1,
                  stance="volunteered", side=True)
    assert result.is_error and state.facts == []
