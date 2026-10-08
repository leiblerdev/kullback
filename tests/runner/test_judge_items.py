"""Judge items (D328): one call per item, 0 or 1, evidence from the event log, recorded on the Verdict."""

from __future__ import annotations

import json
from pathlib import Path

from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from kullback.judge import items
from kullback.judge.items import (
    CUT_MARK,
    build_evidence,
    checkpoint_line,
    judge_item,
    judge_items,
    tally,
)
from kullback.judge.shape import evidence_refs
from kullback.runner.records import Atom, Run, Verifier
from kullback.runner.verdict import verdict

ROOT = Path(__file__).resolve().parents[2]
START = {"rows": {"R1": {"state": "open"}}}


def _run(assistant, end=START, tool_after=None):
    events = [{"idx": 0, "type": "user_turn", "payload": {"text": "What is the total for R1?"}}]
    for text in assistant:
        if tool_after and text == tool_after[1]:
            n = len(events)
            events.append({"idx": n, "type": "tool_call", "payload": {"id": "c", "name": tool_after[0], "args": {}}})
            events.append({"idx": n + 1, "type": "tool_result", "payload": {"id": "c", "name": tool_after[0],
                                                                            "result": {}}})
        events.append({"idx": len(events), "type": "model_call", "payload": {"reply": {"content": text}}})
        events.append({"idx": len(events), "type": "user_turn", "payload": {"text": "ok"}})
    events.append({"idx": len(events), "type": "stop", "payload": {"start_state": START, "end_state": end}})
    return Run.model_validate({"run_id": "r", "events": events})


def _item(item_id="j1", gate=False, evidence=None, anchor=None):
    target = {"question": "Did the agent tell the user the total?",
              "anchor": anchor or {"answer": "42.50", "accepted": ["$42.5"], "reject": ["40.00"]}}
    if evidence is not None:
        target["evidence"] = evidence
    return Atom(id=item_id, kind="allowed", judge=True, gate=gate, target=target)


def _says(score, why="ok", item_id="j1"):
    return ModelReply(content=json.dumps({"results": {item_id: {"score": score, "why": why}}}),
                      usage=Usage(input=1000, output=20))


def test_one_item_is_one_call_at_temperature_zero_and_the_anchor_reaches_only_the_judge(monkeypatch):
    monkeypatch.setattr(items, "call_cost", lambda usage, name: usage.input / 1e6)
    model = TestModel([_says(1)], name="m")
    result = judge_item(model, _item(), _run(["Your total is $42.50."]))
    assert (result.score, result.why) == (1, "ok")
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["config"].temperature == 0 and call["tools"] is None
    prompt = call["messages"][1]["content"]
    assert "[j1]" in prompt and "met = 1: 42.50" in prompt and "reject (score 0): 40.00" in prompt
    assert "<assistant turn 1>\nYour total is $42.50.\n</assistant turn 1>" in prompt
    assert result.cost_usd > 0


def test_a_failed_call_or_an_unreadable_reply_gives_no_score_never_a_zero():
    broken = TestModel([ModelReply(content="I think it is met")])
    assert judge_item(broken, _item(), _run(["hi"])).score is None
    silent = TestModel([])
    result = judge_item(silent, _item(), _run(["hi"]))
    assert result.score is None and result.error.startswith("IndexError")
    assert judge_item(TestModel([_says(0.5)]), _item(), _run(["hi"])).score is None


def test_empty_evidence_scores_zero_by_code_with_no_model_call_also_after_a_tool_never_called():
    silent = TestModel([])
    for item, run in ((_item(), _run([])), (_item(evidence=["after_call:lookup"]), _run(["Your total is 42.50."]))):
        result = judge_item(silent, item, run)
        assert (result.score, result.why) == (0, "no evidence")
    assert silent.calls == []
    gated = Verifier(task_id="t", atoms=[_item(gate=True)])
    unjudged = verdict(_run([]), gated)
    assert unjudged.passed is False and unjudged.failing_atom == "j1" and _row(unjudged).score == 0


def test_the_final_answer_is_the_default_evidence_and_named_turns_replace_it():
    run = _run(["The total is 42.50.", "Anything else?"])
    text, cut, missing = build_evidence(run, ["final_answer"])
    assert "Anything else?" in text and "42.50" not in text and not cut and not missing
    text, _, _ = build_evidence(run, ["assistant:1"])
    assert "42.50" in text and "Anything else?" not in text
    text, _, _ = build_evidence(run, ["assistant_turns", "final_answer"])
    assert text.count("Anything else?") == 1


def test_a_turn_after_a_named_call_is_found_without_an_event_index():
    """after_call reads every agent turn after the call's result, past the next user turn."""
    run = _run(["Let me look.", "It shipped on Monday.", "Bye."], tool_after=("read_row", "It shipped on Monday."))
    text, _, missing = build_evidence(run, ["after_call:read_row"])
    assert "It shipped on Monday." in text and "Bye." in text and "Let me look." not in text and not missing


def test_the_default_evidence_reads_an_answer_given_two_turns_before_the_end_and_nothing_after_a_transfer():
    run = _run(["Your total is $42.50.", "Anything else?", "Goodbye."])
    text, _, missing = build_evidence(run, evidence_refs({}))
    assert "Your total is $42.50." in text and not missing
    held = _run(["Let me check.", "Your total is $42.50.", "Please hold."], tool_after=("transfer_to_desk", "Please hold."))
    text, _, _ = build_evidence(held, evidence_refs({}))
    assert "Your total is $42.50." in text and "Please hold." not in text
    assert "Please hold." not in build_evidence(held, ["final_answer"])[0]
    _, _, missing = build_evidence(run, ["after_call:never_called"])
    assert missing == ["after_call:never_called"]


def test_long_evidence_is_clamped_marked_and_flagged_truncated():
    run = _run(["x" * 50, "y" * 50])
    text, cut, _ = build_evidence(run, ["assistant_turns"], turn_chars=20)
    assert cut and text.count(CUT_MARK) == 2
    text, cut, _ = build_evidence(run, ["assistant_turns"], total_chars=90)
    assert cut and "y" * 50 not in text and text.endswith("(further turns not shown)")
    result = judge_item(TestModel([_says(0, "evidence truncated")]), _item(evidence=["assistant_turns"]), run)
    assert tally([result])["said_truncated"] == 1


def test_a_plain_string_anchor_is_the_answer_alone():
    assert checkpoint_line("j", {"question": "Q?", "anchor": "yes"}) == "- [j] Q? (met = 1: yes)"


def _row(graded, item_id="j1"):
    return next(item for item in graded.items if item.id == item_id)


def test_the_verdict_records_judge_items_and_a_scored_item_never_moves_pass():
    verifier = Verifier(task_id="t", atoms=[_item()])
    run = _run(["Your total is 40.00."])
    results = judge_items(TestModel([_says(0, "wrong total")]), verifier.atoms, run)
    graded = verdict(run, verifier, judge_results=results)
    assert graded.passed is True and graded.class_ == "pass"
    assert _row(graded).score == 0 and _row(graded).why == "wrong total"
    assert _row(graded).gate is False and graded.judge_used


def test_a_gate_item_scored_zero_fails_the_run_and_unjudged_leaves_it_not_verdicted():
    verifier = Verifier(task_id="t", atoms=[_item(gate=True)])
    run = _run(["Your total is 40.00."])
    failed = verdict(run, verifier, judge_results=judge_items(TestModel([_says(0)]), verifier.atoms, run))
    assert failed.passed is False and failed.failing_atom == "j1"
    unjudged = verdict(run, verifier)
    assert unjudged.class_ == "not_verdicted" and _row(unjudged).score is None


def test_an_unjudged_scored_item_does_not_block_a_verdict_judged_by_code_only():
    unjudged = verdict(_run(["hi"]), Verifier(task_id="t", atoms=[_item()]))
    assert unjudged.class_ == "pass" and _row(unjudged).score is None


def test_the_candidate_side_never_reads_an_anchor():
    for folder in ("kullback/agent", "kullback/user"):
        for path in (ROOT / folder).rglob("*.py"):
            assert "anchor" not in path.read_text(encoding="utf-8"), path
    loop = (ROOT / "kullback/runner/loop.py").read_text(encoding="utf-8")
    assert "anchor" not in loop and "kullback.judge" not in loop


def test_an_atom_dumps_gate_and_weight_only_when_they_differ_from_the_default():
    plain = Atom(id="a", kind="required")
    assert "gate" not in plain.model_dump() and "weight" not in plain.model_dump()
    shaped = Atom(id="a", kind="required", gate=True, weight=2.0)
    assert Atom.model_validate(shaped.model_dump()) == shaped
