"""The Reference never repairs a Verifier (D367): no fails_reference ruling, no reference_wrong set-aside."""

import asyncio

import pytest

from kullback.agent.bus import Bus
from kullback.ai.provider import TestModel
from kullback.examiner import prompt as P
from kullback.examiner import rule_tool as R
from kullback.examiner.exam_files import ExamRoot
from kullback.runner.records import write_json
from kullback.spec import review as RV
from kullback.spec import router as ROUTER
from kullback.spec import rulings as RL
from kullback.spec import trust as T
from kullback.spec import writer as WR
from kullback.spec.schema import Spec, load_spec, save_spec
from tests.spec.test_roles import _both_copies, _reference, _told

REASON = "The agent moved the item but never said the new slot the user asked to hear."
FIX = "Keep the check as written because the Reference left the told value unsaid."


def _live_root(tmp_path):
    root = _told(tmp_path)
    write_json(root / "replays.json", {"t1": {"rec1": {"task_id": "t1", "run_id": "ref", "trace_id": "rec1",
                                                       "confirmed": True, "fidelity": 1}}})
    return ExamRoot(workdir=root, round=1, bus=Bus(root / "bus.jsonl", agent="examiner"))


def _rule(root, **kwargs):
    args = {"task_id": "t1", "item": "s1", "kind": "code", "code": "fails_reference",
            "reason": REASON, "fix": FIX, "blocking": True}
    args.update(kwargs)
    return asyncio.run(R._rule(root)(R.RuleArgs(**args)))


def test_verify_rows_carry_no_reference_step_or_judge_row(tmp_path):
    root = _told(tmp_path)
    rows = R.verify_rows(root, "t1", _both_copies(root), [_reference()])
    assert not [row for row in rows if str(row.get("run")).startswith("reference")]
    assert "reference" not in R.verify_line(rows)


def test_a_fails_reference_ruling_filed_before_d367_asks_nothing_of_the_writer(tmp_path):
    root = _told(tmp_path)
    RL.save_ruling(root, RL.Ruling(task_id="t1", number=1, round=1, item="s1", kind="code",
                                   code="fails_reference", reason=REASON, fix=FIX, blocking=False,
                                   wrong_side="reference", ref_step=1))
    out = RV.answer_rulings(root, "t1", TestModel([]), 2)
    assert out["answered"] == 0 and out["skipped"] == 1


def test_not_enough_truth_kept_open_after_the_answer_ends_as_intent_unclear(tmp_path):
    root = _live_root(tmp_path)
    RL.save_ruling(tmp_path, RL.Ruling(task_id="t1", number=1, round=1, item="task", kind="scope",
                                       code=RL.NO_TRUTH_CODE, reason=REASON, fix=RL.NO_TRUTH_FIX, blocking=True))
    RL.answer(tmp_path, "t1", 1, "rebutted", "rebut: the user named the slot.", 1)
    root.round = 2
    asyncio.run(R._close(root)(R.CloseArgs(task_id="t1", number=1, keep_open=True,
                                            why="The turns never say which slot the user wants.")))
    assert load_spec(tmp_path, "t1").set_aside == ROUTER.INTENT_UNCLEAR


def test_an_open_not_enough_truth_at_the_round_cap_reads_as_intent_unclear():
    spec = Spec.model_construct(set_aside=None, truth_open=1, rulings_open=1, round=ROUTER.ROUNDS_CAP)
    assert ROUTER.no_verifier_why(spec) == ROUTER.INTENT_UNCLEAR
    assert ROUTER.no_verifier_why(spec.model_copy(update={"round": 1})) is None


def test_the_writer_is_never_asked_about_a_task_with_no_verifier(tmp_path):
    root = _told(tmp_path)
    RL.save_ruling(root, RL.Ruling(task_id="t1", number=1, round=1, item="s1", kind="code", code="reads_wrong",
                                   reason=REASON, fix="Read the field the user named.", blocking=True))
    save_spec(root, load_spec(root, "t1").model_copy(update={"set_aside": ROUTER.INTENT_UNCLEAR}))
    out = RV.answer_rulings(root, "t1", TestModel([]), 2)
    assert out["answered"] == 0 and out["skipped"] == 1


def test_the_repair_prompt_says_cannot_verify_instead_of_loosening_a_check():
    assert "cannot verify: <why>" in WR.REPAIR_PROMPT
    assert "loosen a check" in WR.REPAIR_PROMPT


def test_a_fails_reference_ruling_is_refused_by_code_and_the_rule_tool_takes_no_side_or_step(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    with pytest.raises(ValueError, match="code ruling's code is one of"):
        _rule(root)
    assert not {"wrong_side", "ref_step"} & set(R.RuleArgs.model_fields)
    assert "fails_reference" not in RL.VERIFY_CODES and RL.load_rulings(tmp_path, "t1") == []


def test_the_choice_rule_asks_whether_a_right_solve_would_pass_and_never_moves_a_check_on_the_reference():
    rule = " ".join(P.choice_section().split())
    assert "a right solve of the user's own requests, under the policy and the world, would pass it" in rule
    assert "A Reference passing or failing is never a reason to change a check" in rule
    assert "fails_reference" not in rule and "ref_step" not in P.tools_section()


def test_no_verifier_reasons_add_the_stage_end_veto_but_never_a_reference_wrong_set_aside(tmp_path):
    assert ROUTER.NO_VERIFIER_WHYS == (ROUTER.INTENT_UNCLEAR, ROUTER.INTENT_STRAYED, ROUTER.UNCHECKED_END,
                                       "precondition_unrendered", "no_reference_holds")
    assert not hasattr(ROUTER, "REFERENCE_WRONG")
    root = _told(tmp_path)
    spec = load_spec(root, "t1")
    save_spec(root, spec.model_copy(update={"set_aside": ROUTER.INTENT_UNCLEAR, "examined_rounds": [1]}))
    row = T.intent_tiers(root)["t1"].row
    assert (row["tier"], row["reason"], row["why"]) == (T.UNTRUSTED, T.NO_VERIFIER, ROUTER.INTENT_UNCLEAR)


def test_a_spec_set_aside_reference_wrong_before_d367_stays_held_not_no_verifier(tmp_path):
    root = _told(tmp_path)
    spec = load_spec(root, "t1")
    save_spec(root, spec.model_copy(update={"set_aside": "reference_wrong", "examined_rounds": [1]}))
    row = T.intent_tiers(root)["t1"].row
    assert row["tier"] == T.UNTRUSTED and row["reason"] != T.NO_VERIFIER and row["why"] is None


def test_a_review_session_refuses_reading_a_run_or_searching_the_whole_root_and_reads_its_view():
    from kullback.ai.messages import ToolCall
    from kullback.examiner.session import _refuse_recordings

    refuse = _refuse_recordings()
    for name, args in (("read", {"path": "runs/t1/fresh-1.jsonl"}), ("grep", {"pattern": "x"}),
                       ("grep", {"pattern": "x", "path": "."}), ("ls", {"path": "./judged"}),
                       ("read", {"path": "references.json"})):
        with pytest.raises(PermissionError, match="never reads a Run"):
            refuse(ToolCall(id="c", name=name, arguments=args))
    for name, args in (("read", {"path": "review/views/t1.json"}), ("grep", {"pattern": "x", "path": "review"}),
                       ("ls", {"path": "."}), ("verify", {"task_id": "t1"})):
        assert refuse(ToolCall(id="c", name=name, arguments=args)) is None
