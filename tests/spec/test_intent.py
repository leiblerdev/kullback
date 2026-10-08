"""The intent miner reads user turns only, keeps facts in the user's words, and writes the Spec."""

from __future__ import annotations

import asyncio
import json

import pytest

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.spec import intent_tools as T
from kullback.spec.intent import intent_system_prompt, mine_intent
from kullback.spec.schema import Spec, load_spec, save_spec
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


def _call(call_id, name, arguments):
    return ModelReply(content=None, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])


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
    return _run(_tools(state)["add_facts"], facts=[fact])


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


def test_an_accepted_fact_rests_on_the_offer_and_counts_only_when_another_user_volunteers_it(workdir):
    state = _state(workdir)
    _add(state, text="refund the shipping fee to the card", recording="rec-a", turn=5, stance="accepted")
    fact = state.facts[0]
    # rec-b only echoes the agent's offer; rec-c raises it unasked.
    assert fact.witnesses == ["rec-c"]
    assert T.counts(fact)


def test_an_accepted_fact_no_other_user_volunteers_is_kept_but_does_not_count(workdir):
    state = _state(workdir)
    state.found.recordings.pop("rec-c")
    result = _add(state, text="refund the shipping fee to the card", recording="rec-a", turn=5, stance="accepted")
    assert "1 accepted with no volunteering witness, kept and marked" in result.content
    assert state.facts[0].witnesses == [] and not T.counts(state.facts[0])


def test_one_call_keeps_the_good_facts_and_refuses_a_repeat_and_a_turn_that_is_no_user_turn(workdir):
    state = _state(workdir)
    first = dict(text="move the blue lamp order to Oak street", recording="rec-a", turn=1, stance="volunteered")
    result = _run(_tools(state)["add_facts"], facts=[first, first, {**first, "turn": 3}])
    rows = result.details["rows"]
    assert rows[0]["fact_id"] == "f1"
    assert "already kept" in rows[1]["refused"] and "not a user turn" in rows[2]["refused"]
    assert len(state.facts) == 1 and "2 refused" in result.content


# --- the session -----------------------------------------------------------------------------------

def test_the_miner_reads_the_recordings_keeps_facts_and_writes_them_to_the_spec_keeping_its_checks(workdir):
    save_spec(workdir, spec(checks=[check()]).model_copy(update={"task_id": "t1"}))
    model = TestModel([
        _call("a", "list_recordings", {"task_id": "t1"}),
        _call("b", "user_turns", {"recording": "rec-a"}),
        _call("c", "add_facts", {"facts": [{"text": "move the blue lamp order to Oak street",
                                            "recording": "rec-a", "turn": 1, "stance": "volunteered"}]}),
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
    model = TestModel([_call("a", "list_recordings", {"task_id": "t1"})], loop=True)
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
            "stance": "volunteered"}
    model = TestModel([_call("a", "add_facts", {"facts": [kept]}),
                       _call("b", "user_turns", {"recording": "rec-b"})], loop=True)
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0, max_calls=1)
    assert out.stopped == "call cap 1 reached" and out.tool_calls == {"add_facts": 1, "user_turns": 1}
    assert [f.text for f in load_spec(workdir, "t1").intent.facts] == [kept["text"]]


def test_refused_facts_are_counted_on_the_result_and_in_the_spec_file_as_a_number(workdir):
    bad = {"text": "wants a green chair", "recording": "rec-a", "turn": 1, "stance": "volunteered"}
    good = {**bad, "text": "move the blue lamp order to Oak street"}
    model = TestModel([_call("a", "add_facts", {"facts": [bad, good, good]}), ModelReply(content="1 kept.")])
    out = mine_intent(workdir, "t1", model, ceiling_usd=1.0)
    written = load_spec(workdir, "t1")
    assert out.refusals == 2 and written.intent.refusals == 2
    assert "green chair" not in (workdir / "spec" / "t1.json").read_text()
