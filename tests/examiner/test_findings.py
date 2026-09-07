"""The findings the round's records file by themselves: what each rule reads, how they are ranked and
cut, and what makes two of them the same finding (D170).

The world here is a lending library: two tools, three Tasks, and records in the shapes the derivation
and the compile_tools gates actually write.
"""

from __future__ import annotations

import json
from pathlib import Path

from gates.examiner_fixtures import SIGS, TASK, base, tighten
from gates.verifier_fixtures import other_reason_run, reference_run
from kullback.examiner import findings as F
from kullback.examiner.plan import ExaminerPlan

RENEW, HOLD, FINE = "task_renew", "task_hold", "task_fine"
DUE_DATE = 'hard columns differ: due_date: ours "2026-03-01", recorded "2026-03-15"'


def _ruling(stage: str, passed: bool, failures: tuple = (), **metrics) -> dict:
    return {"stage": stage, "pass": passed, "failures": list(failures), "metrics": metrics}


def _status() -> dict:
    """Two Tasks an assisted tool blocks, one with a Reference whose Verifier failed the suite."""
    return {
        RENEW: {"reference_confirmed": False, "assisted_tools": ["renew_loan"], "reason": "the seed Trace calls"},
        HOLD: {"reference_confirmed": False, "assisted_tools": ["renew_loan", "place_hold"], "reason": "x"},
        FINE: {"reference_confirmed": True, "verifier_passed": False, "not_run": ["verifier_alt_path"],
               "checks": {"mutation_flips": False, "second_path_passes": False, "empty_fails": True}},
    }


def _rulings() -> list[dict]:
    """What the compile_tools gates wrote: one row per tool per split, the tool only in the failures."""
    return [
        _ruling("replay_fidelity", False, (f'renew_loan({{"loan_id": "L1"}}): {DUE_DATE}',),
                split="shown", success_calls=12, success_matches=11, error_calls=0, error_matches=0),
        _ruling("replay_fidelity", False, ('renew_loan({"loan_id": "L2"}): expected a result, got KeyError: L2',),
                split="held_out", success_calls=4, success_matches=3, error_calls=0, error_matches=0),
        _ruling("replay_fidelity", True, (), split="shown", success_calls=9, success_matches=9),
        _ruling("non_trivial", False, ("the body answers every call the same way",), arg_sets=3),
    ]


def _plan(tmp_path: Path, *, status: dict, rulings: list, references: dict) -> ExaminerPlan:
    """A plan over a workdir holding the three files the rules read and nothing else."""
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    for name, body in (("task_status.json", status), ("gates.json", rulings), ("references.json", references)):
        (workdir / name).write_text(json.dumps(body), encoding="utf-8")
    return ExaminerPlan(workdir=workdir, inputs={})


def _filed(plan: ExaminerPlan) -> list[tuple[str, str, int, str]]:
    return [(f.kind, f.tool or f.task_id or "", f.cost, f.suggested) for f in plan.open_findings()]


# --- reading a gate's failure line ------------------------------------------------------

def test_a_gate_failure_line_gives_up_the_tool_it_names_and_what_the_gate_says_about_it():
    """gates.json holds one row per tool per check and no row names its tool: the name is only in the
    failure line, so attributing a ruling at all depends on reading it back off that line."""
    assert F.failure_subject(f'renew_loan({{"loan_id": "L1"}}): {DUE_DATE}') == ("renew_loan", DUE_DATE)
    assert F.failure_subject("renew_loan: a recorded success call replays differently") == (
        "renew_loan", "a recorded success call replays differently")
    assert F.failure_subject("the body answers every call the same way") == (
        "", "the body answers every call the same way")


def test_a_tools_replay_counts_come_from_its_own_failing_rulings_and_a_split_is_counted_once():
    """The replay_fidelity metrics are that one tool's own split. A tool named by no failing ruling
    has no counts here, and is reported without them rather than with another tool's."""
    seen = F.tool_evidence(_rulings())
    assert seen["renew_loan"]["calls"] == 16 and seen["renew_loan"]["replayed"] == 14
    assert seen["renew_loan"]["detail"] == DUE_DATE, "the first failure is the one the hint quotes"
    assert "place_hold" not in seen


# --- the four rules ---------------------------------------------------------------------

def test_an_assisted_tool_is_filed_as_a_finding_naming_the_tasks_it_blocks_and_the_column_that_differs():
    """D49: a Task whose seed Trace calls an assisted tool has no Reference. The finding is about the
    tool, not the Task, so one tool blocking two Tasks is one finding with a count and both ids."""
    rows = F.assisted_tool_rows(_status(), _rulings())
    renew = next(r for r in rows if r["tool"] == "renew_loan")
    assert renew["task_ids"] == [HOLD, RENEW] and renew["suggested"] == "repair_recompile"
    assert renew["kind"] == "assisted_tool" and renew["key"] == "assisted_tool:renew_loan:"
    assert "2 Tasks have no Reference" in renew["text"]
    assert "replays 14 of 16 recorded calls" in renew["text"]
    assert DUE_DATE in renew["text"] and renew["hint"] == DUE_DATE


def test_a_tool_no_failing_ruling_names_is_filed_without_replay_counts_rather_than_with_another_tools():
    rows = F.assisted_tool_rows(_status(), _rulings())
    held = next(r for r in rows if r["tool"] == "place_hold")
    assert held["task_ids"] == [HOLD] and "No ruling names how its recorded calls replay." in held["text"]
    assert held["hint"] and "compile_tools" in held["hint"]


def test_a_d79_check_that_failed_and_the_same_check_never_run_are_two_findings_with_two_verbs():
    """A check whose gate never ran is a Task short of an input, not a wrong Verifier: it asks for a
    Run (`reroll`), where a check that ran and failed asks the Examiner to repair the Verifier."""
    rows = {row["key"]: row for row in F.suite_rows(_status())}
    assert set(rows) == {"suite:mutation_flips:", "suite:second_path_passes:not_run"}
    failed = rows["suite:mutation_flips:"]
    assert failed["task_ids"] == [FINE] and failed["suggested"] == "repair" and failed["kind"] == "suite"
    assert "failed on 1 Tasks" in failed["text"]
    missing = rows["suite:second_path_passes:not_run"]
    assert missing["suggested"] == "reroll" and "missing Run, not a wrong Verifier" in missing["text"]


def test_a_check_is_counted_only_against_a_task_that_reached_the_suite():
    """A Task with no Reference never ran the suite, and a Task whose Verifier passed lost nothing;
    counting either would rank a check above the tool that is actually blocking the build."""
    status = dict(_status(), task_paid={"reference_confirmed": True, "verifier_passed": True,
                                        "checks": {"mutation_flips": False}})
    assert [row["task_ids"] for row in F.suite_rows(status) if row["key"] == "suite:mutation_flips:"] == [[FINE]]
    assert F.suite_rows({RENEW: _status()[RENEW]}) == []


def test_a_verifier_that_rejects_every_held_out_run_is_filed_against_its_task_and_names_the_atom(tmp_path):
    """D133: the required atoms recognise no path but their seeds. The number is the loosening gate's;
    the atom is not in the ruling, so it is recomputed against the Run the gate rejected."""
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    store = {"verifiers": [strict], "task_runs": {TASK: [reference_run(), other_reason_run()]}, "sigs": SIGS,
             "replays": {TASK: {"tr1": {"trace_id": "tr1", "run_id": "ref", "confirmed": True, "path": ""}}},
             "rerolls": {TASK: [{"run_id": "rr2", "termination_reason": "success", "path": ""}]},
             "canon_rules": None}
    row = F.false_rejection_rows(store)[0]
    assert row["kind"] == "false_rejection" and row["task_id"] == TASK and row["run_id"] == "rr2"
    assert row["suggested"] == "repair" and "reject all 1 held-out" in row["text"]
    assert "w0.reason" in row["hint"] and "w0.reason" in row["text"], "the atom that rejected the Run"


def test_a_verifier_that_accepts_a_held_out_run_is_not_filed(tmp_path):
    store = {"verifiers": [base(tmp_path).model_copy(update={"seed_run_ids": ["ref"]})],
             "task_runs": {TASK: [reference_run(), other_reason_run()]}, "sigs": SIGS,
             "replays": {TASK: {"tr1": {"trace_id": "tr1", "run_id": "ref", "confirmed": True, "path": ""}}},
             "rerolls": {TASK: [{"run_id": "rr2", "termination_reason": "success", "path": ""}]},
             "canon_rules": None}
    assert F.false_rejection_rows(store) == []


def test_a_task_whose_recordings_disagree_is_filed_for_refusal_and_one_an_assisted_tool_blocks_is_not():
    """The corpus not settling on an End state is the Task's own loss and `repair_refuse_task` answers
    it. A Task whose recordings disagree because its tool replays differently is the tool's loss, and
    refusing it would give up a Task the Builder can still fix."""
    references = {
        FINE: {"references": [], "recordings": ["r1", "r2"], "failed": {},
               "groups": [{"label": "A", "state": "renew_loan on L1"}, {"label": "B", "state": "no writes"}],
               "reason": "recordings disagree on the End state (2 states)"},
        RENEW: {"references": [], "recordings": ["r1"], "failed": {},
                "groups": [{"label": "A", "state": "renew_loan on L1"}, {"label": "B", "state": "no writes"}],
                "reason": "the seed Trace calls renew_loan"},
        HOLD: {"references": ["r1"], "recordings": ["r1"], "failed": {}, "groups": [], "reason": None},
    }
    rows = F.disagreement_rows(_status(), references)
    assert [row["task_id"] for row in rows] == [FINE], "only the Task no assisted tool explains"
    assert rows[0]["kind"] == "reference_disagreement" and rows[0]["suggested"] == "repair_refuse_task"
    assert "A: renew_loan on L1; B: no writes" in rows[0]["text"]


def test_a_task_the_judge_failed_on_every_recording_is_filed_as_a_disagreement_too():
    references = {FINE: {"references": [], "recordings": ["r1", "r2"], "groups": [],
                         "failed": {"r1": "judge: neither state does it", "r2": "judge: neither state does it"},
                         "reason": "the judge failed every recording"}}
    row = F.disagreement_rows({}, references)[0]
    assert row["suggested"] == "repair_refuse_task" and "the judge failed all 2 recordings" in row["text"]
    assert F.disagreement_rows({}, {FINE: dict(references[FINE], failed={"r1": "judge: no"})}) == [], \
        "a judge that failed one recording of two has not failed the Task"


# --- ranking, dedup and the cut ---------------------------------------------------------

def test_findings_are_filed_most_costly_first_so_the_builder_opens_on_the_tool_and_not_on_an_intent(tmp_path):
    plan = _plan(tmp_path, status=_status(), rulings=_rulings(), references={})
    filed = F.file_rule_findings(plan)
    assert [(f.kind, f.tool or f.task_id, f.cost) for f in filed][0] == ("assisted_tool", "renew_loan", 2)
    assert [f.cost for f in filed] == sorted((f.cost for f in filed), reverse=True)
    assert [f.finding_id for f in filed] == [f"finding-{n}" for n in range(1, len(filed) + 1)]
    assert json.loads((plan.workdir / "examiner" / "findings.json").read_text(encoding="utf-8"))


def test_a_key_already_open_is_not_filed_again_and_nor_is_one_answered_that_costs_the_same_tasks(tmp_path):
    """A round that files the same loss twice buries the ranking, and a round that files an answered
    loss again over the very same Tasks never lets the loop exit: a round with a finding pending
    runs another round (D126), and a code-driven build reached round 454 on three Tasks that way."""
    plan = _plan(tmp_path, status=_status(), rulings=_rulings(), references={})
    first = F.file_rule_findings(plan)
    assert F.file_rule_findings(plan) == [], "nothing new while every key is open"
    plan.close_findings([f.finding_id for f in first])
    assert F.file_rule_findings(plan) == [], "the Builder has been told, and the Tasks have not moved"


def test_a_loss_the_builder_answered_is_filed_again_once_the_tasks_it_costs_are_a_different_set(tmp_path):
    """Which is how a repair that half worked, or made things worse, stays visible."""
    plan = _plan(tmp_path, status=_status(), rulings=_rulings(), references={})
    plan.close_findings([f.finding_id for f in F.file_rule_findings(plan)])
    status = _status()
    status[HOLD]["assisted_tools"] = ["place_hold"]
    plan.store["task_status"] = status
    again = F.file_rule_findings(plan)
    assert [(f.kind, f.tool, f.cost) for f in again] == [("assisted_tool", "renew_loan", 1)]


def test_the_rule_findings_of_a_round_are_cut_at_the_limit_from_the_bottom_of_the_ranking(tmp_path):
    plan = _plan(tmp_path, status=_status(), rulings=_rulings(), references={})
    filed = F.file_rule_findings(plan, limit=2)
    assert [(f.kind, f.tool or f.task_id) for f in filed] == [("assisted_tool", "renew_loan"),
                                                              ("assisted_tool", "place_hold")]


def test_two_findings_are_the_same_finding_when_the_kind_and_the_thing_they_are_about_agree():
    assert F.finding_key("assisted_tool", "renew_loan") == F.finding_key("assisted_tool", "renew_loan", "")
    assert F.finding_key("suite", "mutation_flips") != F.finding_key("suite", "mutation_flips", "not_run")
    assert F.finding_key("fidelity", "", RENEW) != F.finding_key("environment", "", RENEW)
