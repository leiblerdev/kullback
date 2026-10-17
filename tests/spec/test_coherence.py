"""The coherence check: a Task whose recorded user strayed from an opening ask is set aside, no Verifier."""

from __future__ import annotations

import json

from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import Trace, Usage
from kullback.spec import coherence as C
from kullback.spec import stage
from kullback.spec import trust as T
from kullback.spec.router import INTENT_STRAYED, no_verifier_why
from kullback.spec.schema import FactSource, IntentFact, ReplyQuote, Spec, SpecIntent, load_spec
from kullback.spec.writer import runner_verifier_path, spec_verifier_path
from tests.spec.test_stage import FACT, ITEM, _call, _events, _held, _items, _workdir

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


def _reply(**rows: dict) -> ModelReply:
    return ModelReply(content=json.dumps({"facts": rows}), usage=Usage(input=200, output=30))


def _row(answer: str, quote: str, recording: str = "r1", ender: str = "") -> dict:
    return {"answer": answer, "ender": ender or None, "recording": recording, "quote": quote,
            "why": "shown in that line"}


def _ended(quote: str, ender: str = C.USER, recording: str = "r1") -> dict:
    return _row(C.ENDED_BY, quote, recording, ender)


def _mining() -> list[ModelReply]:
    return [_call("user_turns", {"recording": "rec-a"}), _call("add_facts", {"facts": [FACT]}),
            ModelReply(content="1 fact kept.", usage=Usage(input=100, output=5))]


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
    model = TestModel([_reply(f1=_row("carried_out", "Done, it sits in slot seven now."))])
    record, _ = C.check_coherence(root, "t1", _intent(OPENING), model)
    assert record["decided_by"] == C.JUDGE and record["strayed"] == []
    assert record["facts"]["f1"] == {"answer": "carried_out", "ender": None, "recording": "rec-a",
                                     "quote": "Done, it sits in slot seven now.", "quote_ok": True,
                                     "why": "shown in that line"}


def test_an_ask_the_user_withdrew_in_their_own_words_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    model = TestModel([_reply(f1=_row("refused_or_withdrawn", "Actually forget it"))])
    record, _ = C.check_coherence(root, "t1", _intent(OPENING), model)
    assert record["strayed"] == [] and record["facts"]["f1"]["quote_ok"] is True


def test_an_ask_the_user_ended_in_their_own_words_is_named_strayed(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    model = TestModel([_reply(f1=_ended("I will call back later"))])
    record, _ = C.check_coherence(root, "t1", _intent(OPENING), model)
    assert record["strayed"] == ["f1"] and record["unquoted"] == 0
    assert (record["facts"]["f1"]["answer"], record["facts"]["f1"]["ender"]) == (C.ENDED_BY, C.USER)


def test_an_ask_the_agent_or_the_world_ended_sets_nothing_aside_and_keeps_the_ender(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for ender, quote in ((C.AGENT, "I cannot finish this, passing you on."), (C.WORLD, "update_item")):
        record, _ = C.check_coherence(root, "t1", _intent(OPENING), TestModel([_reply(f1=_ended(quote, ender))]))
        assert record["strayed"] == [] and record["unquoted"] == 0
        assert (record["facts"]["f1"]["ender"], record["facts"]["f1"]["quote_ok"]) == (ender, True)


def test_a_user_ending_quoted_from_the_agent_or_from_a_marker_alone_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for quote in ("I cannot finish this, passing you on.", "###STOP###"):
        record, _ = C.check_coherence(root, "t1", _intent(OPENING), TestModel([_reply(f1=_ended(quote))]))
        assert record["strayed"] == [] and record["unquoted"] == 1 and record["facts"]["f1"]["quote_ok"] is False


def test_an_ender_on_another_answer_or_outside_the_three_is_not_kept(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for row in (_row(C.CARRIED_OUT, "update_item", ender=C.USER), _ended("call back later", "someone")):
        record, _ = C.check_coherence(root, "t1", _intent(OPENING), TestModel([_reply(f1=row)]))
        assert record["strayed"] == [] and record["facts"]["f1"]["ender"] is None


def test_a_user_ending_whose_quote_is_not_in_the_named_recording_sets_nothing_aside(tmp_path):
    root = _workdir(tmp_path)
    _trace_with_call(root)
    for row in (_ended("the user never gave the code"), _ended("call back later", recording="r9")):
        record, _ = C.check_coherence(root, "t1", _intent(OPENING), TestModel([_reply(f1=row)]))
        assert record["strayed"] == [] and record["unquoted"] == 1 and record["facts"]["f1"]["quote_ok"] is False


def test_a_task_with_no_opening_ask_is_decided_by_code_with_no_model_call(tmp_path):
    root = _workdir(tmp_path)
    detail = OPENING.model_copy(update={"kind": "detail"})
    model = TestModel([])
    record, spent = C.check_coherence(root, "t1", _intent(detail), model)
    assert (record["decided_by"], record["strayed"], spent, model.calls) == (C.CODE, [], 0.0, [])


def test_an_unreadable_reply_or_a_failed_call_sets_nothing_aside_and_says_why(tmp_path):
    root = _workdir(tmp_path)

    class Down:
        name = "down"

        def query(self, *args, **kwargs):
            raise ConnectionError("no route")

    unread, _ = C.check_coherence(root, "t1", _intent(OPENING), TestModel([ModelReply(content="no json")]))
    failed, _ = C.check_coherence(root, "t1", _intent(OPENING), Down())
    assert unread["strayed"] == [] and unread["error"] == "unreadable reply"
    assert failed["strayed"] == [] and failed["error"].startswith("ConnectionError")


def test_a_task_whose_user_strayed_is_set_aside_with_no_verifier_and_the_bus_is_told(tmp_path):
    root = _workdir(tmp_path)
    model = TestModel(_mining() + [_held(C.ENDED_BY, "Please move item A1 to slot seven today", C.USER)])
    counts = stage.write_specs(root, ["t1"], model)
    assert (counts["written"], counts["set_aside_strayed"], len(model.calls)) == (0, 1, 4)
    spec = load_spec(root, "t1")
    assert spec.set_aside == INTENT_STRAYED and spec.coherence["strayed"] == ["f1"]
    assert not spec_verifier_path(root, "t1").exists() and not runner_verifier_path(root, "t1").exists()
    (line,) = (root / "bus.jsonl").read_text().splitlines()
    assert json.loads(line)["event"]["payload"] == {"task_id": "t1", "reason": INTENT_STRAYED, "rounds": 0}
    again = stage.write_specs(root, ["t1"], TestModel([ModelReply(content="unused")]))
    assert again["skipped_existing"] == 1


def test_a_strayed_task_reads_as_no_verifier_why_intent_strayed_in_the_tiers(tmp_path):
    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_mining() + [_held(C.ENDED_BY, "slot seven today", C.USER)]))
    spec = load_spec(root, "t1")
    assert no_verifier_why(spec) == INTENT_STRAYED
    row = T.intent_tiers(root)["t1"].row
    assert (row["tier"], row["reason"], row["why"]) == (T.UNTRUSTED, T.NO_VERIFIER, INTENT_STRAYED)


def test_a_task_whose_asks_held_is_written_and_its_spec_keeps_the_coherence_record(tmp_path):
    root = _workdir(tmp_path)
    stage.write_specs(root, ["t1"], TestModel(_mining() + [_held()] + _items(ITEM)))
    spec = load_spec(root, "t1")
    assert spec.set_aside is None and spec.coherence["facts"]["f1"]["answer"] == C.CARRIED_OUT
    assert spec_verifier_path(root, "t1").is_file() and _events(root) == ["spec.written"]


def test_a_spec_with_no_coherence_record_dumps_without_the_field():
    spec = Spec(task_id="t1", intent=_intent(OPENING))
    assert "coherence" not in spec.model_dump(mode="json")
    held = spec.model_copy(update={"coherence": {"decided_by": C.CODE}})
    assert held.model_dump(mode="json")["coherence"] == {"decided_by": C.CODE}
