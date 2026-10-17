"""The coherence check: the miner says what became of each opening ask; a user ending sets the Task aside."""

from __future__ import annotations

import json

from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import Trace
from kullback.spec import coherence as C
from kullback.spec import stage
from kullback.spec import trust as T
from kullback.spec.intent import intent_system_prompt, mine_intent
from kullback.spec.router import INTENT_STRAYED, no_verifier_why
from kullback.spec.schema import FactSource, IntentFact, ReplyQuote, Spec, SpecIntent, load_spec
from kullback.spec.writer import runner_verifier_path, spec_verifier_path
from tests.spec.test_stage import FACT, ITEM, _events, _items, _mined, _workdir

PTR = {"file_hash": "h"}
OPENING = IntentFact(id="f1", text="move item A1 to slot seven", stance="volunteered", kind="do",
                     source=FactSource(recording="rec-a", turn=1))


def _intent(*facts: IntentFact) -> SpecIntent:
    return SpecIntent(task_id="t1", facts=list(facts))


def _trace_with_call(root) -> None:
    """The one recording, with the agent's write call and its result after the user's ask."""
    turns = [{"idx": 0, "role": "assistant", "content": "How can I help?", "raw_ptr": PTR},
             {"idx": 1, "role": "user", "content": "Please move item A1 to slot seven today.", "raw_ptr": PTR},
             {"idx": 2, "role": "assistant", "content": "", "tool_call_ids": ["c1"], "raw_ptr": PTR},
             {"idx": 3, "role": "tool", "content": "{\"slot\": \"seven\"}", "raw_ptr": PTR},
             {"idx": 4, "role": "assistant", "content": "Done, it sits in slot seven now.", "raw_ptr": PTR},
             {"idx": 5, "role": "user", "content": "Actually forget it, I will call back later.", "raw_ptr": PTR},
             {"idx": 6, "role": "assistant", "content": "I cannot finish this, passing you on.", "raw_ptr": PTR},
             {"idx": 7, "role": "user", "content": "###STOP###", "raw_ptr": PTR}]
    calls = [{"id": "c1", "name": "update_item", "args": {"item_id": "A1", "slot": "seven"},
              "result": {"slot": "seven"}, "raw_ptr": PTR}]
    (root / "traces" / "rec-a.json").write_text(json.dumps(
        {"trace_id": "rec-a", "raw_hash": "h", "ingest_version": "v", "source": "s", "turns": turns,
         "tool_calls": calls, "raw_ptr": PTR}))


def _row(answer: str, quote: str, recording: str = "rec-a", ender: str = "") -> dict:
    return {"answer": answer, "ender": ender or None, "recording": recording, "quote": quote,
            "why": "shown in that line"}


def _ended(quote: str, ender: str = C.USER, recording: str = "rec-a") -> dict:
    return _row(C.ENDED_BY, quote, recording, ender)


def _checked(root, row: dict) -> dict:
    """The coherence record code makes of the miner's outcome row on the one opening ask."""
    return C.coherence_of(root, "t1", _intent(OPENING), {"f1": row})


def _with_outcome(row: dict) -> dict:
    """The stage's mined fact, carrying the miner's outcome on it in the same add_facts call."""
    return FACT | {"outcome": {key: value for key, value in row.items() if value is not None}}


def test_only_volunteered_do_or_told_facts_the_user_raised_unprompted_are_opening_asks():
    told = OPENING.model_copy(update={"id": "f2", "kind": "told"})
    keep = OPENING.model_copy(update={"id": "f3", "kind": "keep"})
    detail = OPENING.model_copy(update={"id": "f4", "kind": "detail"})
    accepted = OPENING.model_copy(update={"id": "f5", "stance": "accepted"})
    asked = OPENING.model_copy(update={"id": "f6", "condition": "asked",
                                       "in_reply_to": ReplyQuote(turn=1, quote="How can I help?")})
    intent = _intent(OPENING, told, keep, detail, accepted, asked)
    assert [fact.id for fact in C.opening_asks(intent)] == ["f1", "f2"]


def test_a_recording_shows_each_call_with_its_arguments_and_result_and_never_the_tool_turn_twice(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    trace = Trace.model_validate_json((root / "traces" / "rec-a.json").read_text())
    lines = C.recording_lines(trace)
    assert lines[1] == "user: Please move item A1 to slot seven today."
    assert lines[2].startswith("agent calls update_item {") and lines[2].endswith("-> {'slot': 'seven'}")
    assert len(lines) == 7 and lines[-1] == "user: ###STOP###"


def test_a_quote_counts_only_when_it_sits_word_for_word_inside_one_line_and_carries_content_words():
    lines = ["user: Please move item A1 to slot seven today.", "agent: Done."]
    assert C.quote_in("move  item a1 to slot SEVEN", lines)
    assert not C.quote_in("slot seven today. agent: Done", lines)
    assert not C.quote_in("to the", lines) and not C.quote_in("", lines)


def test_an_ask_the_agent_carried_out_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    record = _checked(root, _row(C.CARRIED_OUT, "Done, it sits in slot seven now."))
    assert record["decided_by"] == C.MINER and record["strayed"] == []
    assert record["facts"]["f1"] == {"answer": "carried_out", "ender": None, "recording": "rec-a",
                                     "quote": "Done, it sits in slot seven now.", "quote_ok": True,
                                     "why": "shown in that line"}


def test_an_ask_the_user_withdrew_in_their_own_words_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    record = _checked(root, _row(C.REFUSED_OR_WITHDRAWN, "Actually forget it"))
    assert record["strayed"] == [] and record["facts"]["f1"]["quote_ok"] is True


def test_an_ask_the_user_ended_in_their_own_words_is_named_strayed(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    record = _checked(root, _ended("I will call back later"))
    assert record["strayed"] == ["f1"] and record["unquoted"] == 0
    assert (record["facts"]["f1"]["answer"], record["facts"]["f1"]["ender"]) == (C.ENDED_BY, C.USER)


def test_an_ask_the_agent_or_the_world_ended_sets_nothing_aside_and_keeps_the_ender(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for ender, quote in ((C.AGENT, "I cannot finish this, passing you on."), (C.WORLD, "update_item")):
        record = _checked(root, _ended(quote, ender))
        assert record["strayed"] == [] and record["unquoted"] == 0
        assert (record["facts"]["f1"]["ender"], record["facts"]["f1"]["quote_ok"]) == (ender, True)


def test_a_user_ending_quoted_from_the_agent_or_from_a_marker_alone_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for quote in ("I cannot finish this, passing you on.", "###STOP###"):
        record = _checked(root, _ended(quote))
        assert record["strayed"] == [] and record["unquoted"] == 1 and record["facts"]["f1"]["quote_ok"] is False


def test_an_ender_on_another_answer_or_outside_the_three_is_not_kept(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for row in (_row(C.CARRIED_OUT, "update_item", ender=C.USER), _ended("call back later", "someone")):
        record = _checked(root, row)
        assert record["strayed"] == [] and record["facts"]["f1"]["ender"] is None


def test_a_user_ending_whose_quote_is_not_in_the_named_recording_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for row in (_ended("the user never gave the code"), _ended("call back later", recording="rec-z")):
        record = _checked(root, row)
        assert record["strayed"] == [] and record["unquoted"] == 1 and record["facts"]["f1"]["quote_ok"] is False


def test_a_task_with_no_opening_ask_is_decided_by_code(tmp_path):
    root = _workdir(tmp_path)
    detail = OPENING.model_copy(update={"kind": "detail"})
    record = C.coherence_of(root, "t1", _intent(detail), {})
    assert (record["decided_by"], record["strayed"], record["why"]) == (C.CODE, [], "no opening ask")


def test_an_opening_ask_the_miner_gave_no_outcome_sets_nothing_aside_and_is_counted(tmp_path):
    root = _workdir(tmp_path)
    record = C.coherence_of(root, "t1", _intent(OPENING), {})
    assert record["strayed"] == [] and record["unanswered"] == 1 and record["facts"]["f1"]["answer"] is None


def test_the_miners_prompt_asks_the_outcome_and_reads_a_withdrawal_after_a_price_as_no_user_ending():
    prompt = " ".join(intent_system_prompt().split())
    assert " ".join(C.OUTCOME_RULE.split()) in prompt
    assert ("When the user drops or declines an ask in their own words after hearing a price, a condition or a "
            "refusal, that is refused_or_withdrawn, not a user ending.") in prompt
    assert prompt.index("The outcome of each opening ask") < prompt.index("When to stop")


def test_the_miner_answers_the_outcome_in_its_own_add_facts_call_with_no_further_model_call(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel(_mined(_with_outcome(_row(C.CARRIED_OUT, "move item A1 to slot seven"))))
    mined = mine_intent(root, "t1", model, ceiling_usd=1.0)
    assert len(model.calls) == 3 and mined.coherence["facts"]["f1"]["answer"] == C.CARRIED_OUT
    assert load_spec(root, "t1").coherence == mined.coherence


def test_a_task_whose_user_strayed_is_set_aside_with_no_verifier_and_the_bus_is_told(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel(_mined(_with_outcome(_ended("Please move item A1 to slot seven today"))))
    counts = stage.write_specs(root, ["t1"], model)
    assert (counts["written"], counts["set_aside_strayed"], len(model.calls)) == (0, 1, 3)
    spec = load_spec(root, "t1")
    assert spec.set_aside == INTENT_STRAYED and spec.coherence["strayed"] == ["f1"]
    assert not spec_verifier_path(root, "t1").exists() and not runner_verifier_path(root, "t1").exists()
    (line,) = (root / "bus.jsonl").read_text().splitlines()
    assert json.loads(line)["event"]["payload"] == {"task_id": "t1", "reason": INTENT_STRAYED, "rounds": 0}
    again = stage.write_specs(root, ["t1"], TestModel([ModelReply(content="unused")]))
    assert again["skipped_existing"] == 1


def test_a_strayed_task_reads_as_no_verifier_why_intent_strayed_in_the_tiers(tmp_path):
    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_mined(_with_outcome(_ended("slot seven today")))))
    spec = load_spec(root, "t1")
    assert no_verifier_why(spec) == INTENT_STRAYED
    row = T.intent_tiers(root)["t1"].row
    assert (row["tier"], row["reason"], row["why"]) == (T.UNTRUSTED, T.NO_VERIFIER, INTENT_STRAYED)


def test_a_task_whose_asks_held_is_written_and_its_spec_keeps_the_coherence_record(tmp_path):
    root = _workdir(tmp_path)
    fact = _with_outcome(_row(C.CARRIED_OUT, "move item A1 to slot seven"))
    stage.write_specs(root, ["t1"], TestModel(_mined(fact) + _items(ITEM)))
    spec = load_spec(root, "t1")
    assert spec.set_aside is None and spec.coherence["facts"]["f1"]["answer"] == C.CARRIED_OUT
    assert spec_verifier_path(root, "t1").is_file() and _events(root) == ["spec.written"]


def test_a_spec_with_no_coherence_record_dumps_without_the_field():
    spec = Spec(task_id="t1", intent=_intent(OPENING))
    assert "coherence" not in spec.model_dump(mode="json")
    held = spec.model_copy(update={"coherence": {"decided_by": C.CODE}})
    assert held.model_dump(mode="json")["coherence"] == {"decided_by": C.CODE}
