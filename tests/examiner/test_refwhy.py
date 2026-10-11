"""When the Reference fails a check the Examiner investigates its steps before naming a side."""

import asyncio
from types import SimpleNamespace

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
from tests.examiner.test_rule_tool import _root as _exam_root
from tests.spec.fixtures import reference_run
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


def test_steps_list_user_turns_tool_calls_and_replies_in_order():
    steps = R.reference_steps(reference_run())
    assert [(row["step"], row["kind"]) for row in steps] == [
        (0, "user_turn"), (1, "tool_call"), (3, "tool_call"), (5, "model_call")]
    assert steps[0]["text"].startswith("Please move")
    assert "update_item" in steps[2]["text"]
    assert steps[3]["text"].startswith("Item A1 is now")
    assert all(row["run"] == "reference_step" for row in steps)


def test_verify_rows_carry_steps_and_the_line_counts_them(tmp_path):
    root = _told(tmp_path)
    rows = R.verify_rows(root, "t1", _both_copies(root), _reference())
    assert [row["step"] for row in rows if row.get("run") == "reference_step"] == [0, 1, 3]
    assert "reference steps: 3" in R.verify_line(rows)


def test_a_fails_reference_ruling_without_a_side_or_a_step_is_refused(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    with pytest.raises(ValueError, match="wrong_side"):
        _rule(root)
    with pytest.raises(ValueError, match="ref_step"):
        _rule(root, wrong_side="verifier")


def test_a_fails_reference_ruling_on_a_step_the_verify_answer_does_not_show_is_refused(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    with pytest.raises(ValueError, match="not one the verify answer shows"):
        _rule(root, wrong_side="verifier", ref_step=99)
    with pytest.raises(ValueError, match="not one the verify answer shows"):
        _rule(root, wrong_side="verifier", ref_step=2)  # the tool result, which verify leaves out


@pytest.mark.parametrize("step", [None, 99])
def test_a_refused_step_says_the_ruling_stands_and_how_to_refile(tmp_path, step):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    with pytest.raises(ValueError) as refused:
        _rule(root, wrong_side="reference", ref_step=step)
    assert "still stands" in str(refused.value) and "never" in str(refused.value)
    assert "a user turn, a tool call or a reply" in str(refused.value)


def test_never_names_an_omission_on_a_state_cell(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "state:items.A1.label"}]
    _rule(root, wrong_side="reference", ref_step="never")
    [ruling] = RL.load_rulings(tmp_path, "t1")
    assert ruling.ref_step == RL.NEVER_STEP


def test_never_is_refused_when_the_failing_check_cannot_fail_by_omission(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "sanity"}]
    with pytest.raises(ValueError, match="never fits only"):
        _rule(root, wrong_side="reference", ref_step="never")


def test_a_called_check_can_fail_by_omission_and_a_said_one_cannot():
    verifier = SimpleNamespace(atoms=[SimpleNamespace(id="c1", target={"kind": "called"}),
                                      SimpleNamespace(id="c2", target={"kind": "say"})])
    assert R._omission_check(verifier, "c1") and not R._omission_check(verifier, "c2")
    assert R._omission_check(None, "state:items.A1.slot") and not R._omission_check(verifier, None)


def test_a_fails_reference_ruling_without_a_reference_on_disk_is_refused(tmp_path):
    root = _exam_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "c1"}]
    with pytest.raises(ValueError, match="no faithful Reference"):
        asyncio.run(R._rule(root)(R.RuleArgs(task_id="t1", item="c1", kind="code", code="fails_reference",
                                             reason=REASON, fix=FIX, blocking=True,
                                             wrong_side="verifier", ref_step=1)))


def test_a_fails_reference_ruling_carries_its_step_but_not_the_steps_or_judge_rows(tmp_path):
    root = _live_root(tmp_path)
    reference = {"run": "reference", "verifier_passes_it": False, "failing": "s1"}
    root.verified["t1"] = [reference,
                           {"run": "reference_step", "step": 1, "kind": "tool_call", "text": "update_item {}"},
                           {"run": "reference_judge", "item": "j1", "holds": False,
                            "why": "never said", "lines": ["an agent line"]}]
    _rule(root, wrong_side="reference", ref_step=1)
    [ruling] = RL.load_rulings(tmp_path, "t1")
    assert (ruling.wrong_side, ruling.ref_step) == ("reference", 1)
    assert ruling.verified == [reference]
    assert ruling.blocking is False
    assert load_spec(tmp_path, "t1").rulings_open == 0


def test_a_verifier_side_ruling_with_a_step_blocks(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    _rule(root, wrong_side="verifier", ref_step=3)
    [ruling] = RL.load_rulings(tmp_path, "t1")
    assert (ruling.wrong_side, ruling.ref_step, ruling.blocking) == ("verifier", 3, True)


def test_a_reference_side_ruling_asks_nothing_of_the_writer(tmp_path):
    root = _told(tmp_path)
    RL.save_ruling(root, RL.Ruling(task_id="t1", number=1, round=1, item="s1", kind="code",
                                   code="fails_reference", reason=REASON, fix=FIX, blocking=False,
                                   wrong_side="reference", ref_step=1))
    out = RV.answer_rulings(root, "t1", TestModel([]), 2)
    assert out["answered"] == 0 and out["skipped"] == 1


def test_the_choice_rule_investigates_before_ruling_and_never_loosens_on_a_pass():
    assert "investigate before ruling" in P.choice_section()
    assert "never a reason to change a check" in P.choice_section()
    assert "ref_step" in P.tools_section()


def test_the_choice_rule_asks_whether_a_right_solve_would_pass_before_naming_a_side():
    rule = P.choice_section()
    assert "a right solve of the user's own requests, under the policy, would pass this check" in rule
    assert rule.index("would pass this check") < rule.index("A Reference passing is never")


def test_a_reference_side_ruling_ends_the_task_with_no_verifier_once(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    _rule(root, wrong_side="reference", ref_step=1)
    assert load_spec(tmp_path, "t1").set_aside == ROUTER.REFERENCE_WRONG
    R.no_verifier(tmp_path, "t1", ROUTER.INTENT_UNCLEAR)
    assert load_spec(tmp_path, "t1").set_aside == ROUTER.REFERENCE_WRONG


def test_a_verifier_side_ruling_leaves_the_task_open_to_repair(tmp_path):
    root = _live_root(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    _rule(root, wrong_side="verifier", ref_step=3)
    assert load_spec(tmp_path, "t1").set_aside is None


def test_not_enough_truth_kept_open_after_the_answer_ends_as_intent_unclear(tmp_path):
    root = _live_root(tmp_path)
    RL.save_ruling(tmp_path, RL.Ruling(task_id="t1", number=1, round=1, item="task", kind="scope",
                                       code=RL.NO_TRUTH_CODE, reason=REASON, fix=RL.NO_TRUTH_FIX, blocking=True))
    RL.answer(tmp_path, "t1", 1, "rebutted", "rebut: the user named the slot.", 1)
    root.round = 2
    asyncio.run(R._close(root)(R.CloseArgs(task_id="t1", number=1, keep_open=True,
                                            why="The turns never say which slot the user wants.")))
    assert load_spec(tmp_path, "t1").set_aside == ROUTER.INTENT_UNCLEAR


def test_a_task_with_no_verifier_is_untrusted_for_that_reason_and_not_an_error(tmp_path):
    root = _told(tmp_path)
    spec = load_spec(root, "t1")
    save_spec(root, spec.model_copy(update={"set_aside": ROUTER.REFERENCE_WRONG, "examined_rounds": [1]}))
    row = T.intent_tiers(root)["t1"].row
    assert (row["tier"], row["reason"], row["why"]) == (T.UNTRUSTED, T.NO_VERIFIER, ROUTER.REFERENCE_WRONG)


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


def _routed(root):
    RL.save_ruling(root, RL.Ruling(task_id="t1", number=1, round=0, item="s1", kind="code", code="fails_reference",
                                   reason="The kept Reference fails this Verifier, found by code at the trust step.",
                                   fix="File a fails_reference ruling naming wrong_side and ref_step.", blocking=True))
    RL.sync_spec(root, "t1")


def test_a_routed_reference_failure_cannot_be_closed_without_a_side(tmp_path):
    root = _live_root(tmp_path)
    _routed(tmp_path)
    with pytest.raises(ValueError, match="rule it"):
        asyncio.run(R._close(root)(R.CloseArgs(task_id="t1", number=1, keep_open=False,
                                                why="The Reference is the wrong side here.")))
    assert RL.load_rulings(tmp_path, "t1")[0].status == "open"


def test_a_sided_ruling_closes_the_routed_row_and_ends_the_task_with_no_verifier(tmp_path):
    root = _live_root(tmp_path)
    _routed(tmp_path)
    root.verified["t1"] = [{"run": "reference", "verifier_passes_it": False, "failing": "s1"}]
    _rule(root, wrong_side="reference", ref_step=1)
    routed, sided = RL.load_rulings(tmp_path, "t1")
    assert routed.status == "closed" and sided.ref_step == 1
    spec = load_spec(tmp_path, "t1")
    assert spec.rulings_open == 0 and spec.set_aside == ROUTER.REFERENCE_WRONG
