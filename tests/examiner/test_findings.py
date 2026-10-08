"""The findings the records file by themselves: what each rule reads and what makes two of them the
same finding (D170).

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
from kullback.runner.canon import CanonRules

RENEW, HOLD, FINE, CARD = "task_renew", "task_hold", "task_fine", "task_card"
DUE_DATE = 'hard columns differ: due_date: ours "2026-03-01", recorded "2026-03-15"'
NO_SHELF = "expected a result, got KeyError: shelf"


def _status() -> dict:
    """Four Tasks: two an assisted tool blocks by their own calls, one that calls it and is not
    blocked (D171), one with a Reference whose Verifier failed the suite."""
    return {
        RENEW: {"reference_confirmed": False, "assisted_tools": ["renew_loan"],
                "blocking_tools": ["renew_loan"], "reason": "this Task's own recorded calls"},
        HOLD: {"reference_confirmed": False, "assisted_tools": ["renew_loan", "place_hold"],
               "blocking_tools": ["place_hold", "renew_loan"], "reason": "x"},
        CARD: {"reference_confirmed": False, "assisted_tools": ["renew_loan"],
               "blocking_tools": [], "reason": "recordings disagree on the End state"},
        FINE: {"reference_confirmed": True, "verifier_passed": False, "not_run": ["verifier_alt_path"],
               "checks": {"mutation_flips": False, "second_path_passes": False, "empty_fails": True}},
    }


def _fidelity() -> dict:
    """`tool_fidelity.json` in the shape compile_tools writes it (D171): the corpus ruling per tool,
    and per Task per tool how many of that Task's own recorded calls replayed and how many differed."""
    return {
        "tools": {"renew_loan": {"calls": 16, "replayed": 14}, "place_hold": {"calls": 4, "replayed": 3}},
        "tasks": {
            RENEW: {"renew_loan": {"replayed": 2, "differing": 1, "reasons": [DUE_DATE]}},
            HOLD: {"renew_loan": {"replayed": 1, "differing": 1, "reasons": [DUE_DATE]},
                   "place_hold": {"replayed": 0, "differing": 1, "reasons": [NO_SHELF]}},
            CARD: {"renew_loan": {"replayed": 3, "differing": 0, "reasons": []}},
        },
    }


def _replays() -> dict:
    """`replays.json` in the shape the reference replay gate reads: per Task, per Trace, whether
    the Trace replayed to its End state and the gate's own reasons where it did not."""
    return {
        RENEW: {"trace-1": {"confirmed": False, "reasons": [DUE_DATE]}},
        HOLD: {"trace-2": {"confirmed": False, "reasons": [NO_SHELF]}},
        CARD: {"trace-3": {"confirmed": False, "reasons": ["the End state was never reached"]}},
        FINE: {"trace-4": {"confirmed": True, "reasons": []}},
    }


def _plan(tmp_path: Path, *, status: dict, fidelity: dict, references: dict,
          replays: dict | None = None) -> ExaminerPlan:
    """A plan over a workdir holding the files the rules read and nothing else."""
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    for name, body in (("task_status.json", status), ("tool_fidelity.json", fidelity),
                       ("references.json", references)):
        (workdir / name).write_text(json.dumps(body), encoding="utf-8")
    return ExaminerPlan(workdir=workdir,
                        inputs={"tool_fidelity": fidelity, "replays": replays or {}})


# --- the five rules ---------------------------------------------------------------------

def test_an_assisted_tool_is_filed_as_a_finding_naming_the_tasks_it_blocks_and_the_column_that_differs():
    """The finding is about the tool, not the Task, so one tool blocking two Tasks is one finding with
    a count and both ids; the change line is the corpus gate's own words for the first differing call."""
    rows = F.assisted_tool_rows(_status(), _fidelity())
    renew = next(r for r in rows if r["tool"] == "renew_loan")
    assert renew["task_ids"] == [HOLD, RENEW] and "suggested" not in renew
    assert renew["kind"] == "assisted_tool" and renew["key"] == "assisted_tool:renew_loan:"
    assert "2 Tasks' own recorded calls differently" in renew["text"]
    assert "replays 14 of 16 recorded calls of the corpus and 3 Tasks call it" in renew["text"]
    assert DUE_DATE in renew["text"] and renew["change"] == DUE_DATE


def test_a_task_that_calls_an_assisted_tool_its_own_calls_replay_is_not_counted_against_it():
    """D171: assisted is a corpus ruling, and reading it as every Task that calls the tool put one
    tool's name on 51 Tasks of a live build that make no call it answers differently."""
    rows = F.assisted_tool_rows(_status(), _fidelity())
    assert CARD not in next(r for r in rows if r["tool"] == "renew_loan")["task_ids"]
    assert [r["task_ids"] for r in rows if r["tool"] == "place_hold"] == [[HOLD]]
    assert next(r for r in rows if r["tool"] == "place_hold")["change"] == NO_SHELF


def test_a_tool_that_is_assisted_and_blocks_no_task_is_not_a_finding():
    """The red light still stands in the Builder's status; a recompile of it buys no Task."""
    status = {CARD: _status()[CARD]}
    assert F.assisted_tool_rows(status, _fidelity()) == []


def test_a_d79_check_that_failed_and_the_same_check_never_run_are_two_findings():
    """A check whose gate never ran is a Task short of an input, not a wrong Verifier: it asks for a
    Run, where a check that ran and failed asks the Examiner to repair the Verifier."""
    rows = {row["key"]: row for row in F.suite_rows(_status())}
    assert set(rows) == {"suite:mutation_flips:", "suite:second_path_passes:not_run"}
    failed = rows["suite:mutation_flips:"]
    assert failed["task_ids"] == [FINE] and failed["kind"] == "suite"
    assert "failed on 1 Tasks" in failed["text"]
    missing = rows["suite:second_path_passes:not_run"]
    assert "one Reference, so there is no second path to score" in missing["text"]
    assert "missing Run, not a wrong Verifier" in missing["text"]


def test_a_leak_the_strip_missed_is_filed_per_task_and_column():
    """D196: the strip has already run, so what the leak check reports is a column it does not cover.
    The key is the Task and the column, so a second column on one Task is a second finding while the
    same column found again in the next round is the same one."""
    status = dict(_status(), task_late={
        "reference_confirmed": True, "verifier_passed": False,
        "checks": {"leak_check_clean": False}, "leak_columns": ["renew_loan.due_date", "loans.fine"]})
    rows = {row["key"]: row for row in F.suite_rows(status)}
    assert "intent_leak:renew_loan.due_date:task_late" in rows
    assert "intent_leak:loans.fine:task_late" in rows
    assert "suite:leak_check_clean:" not in rows, "the grouped row is the fallback, not the answer"
    row = rows["intent_leak:renew_loan.due_date:task_late"]
    assert row["kind"] == "intent_leak"
    assert row["task_ids"] == ["task_late"] and "renew_loan.due_date" in row["change"]


def test_a_leak_on_a_task_that_names_no_column_is_still_filed_as_the_one_grouped_finding():
    """A status row written before D196 names no column, and the loss is not dropped for that."""
    status = dict(_status(), task_late={"reference_confirmed": True, "verifier_passed": False,
                                        "checks": {"leak_check_clean": False}})
    rows = {row["key"]: row for row in F.suite_rows(status)}
    assert rows["suite:leak_check_clean:"]["task_ids"] == ["task_late"]
    assert not any(key.startswith("intent_leak:") for key in rows)


def test_a_check_is_counted_only_against_a_task_that_reached_the_suite():
    """A Task with no Reference never ran the suite, and a Task whose Verifier passed lost nothing;
    counting either would rank a check above the tool that is actually blocking the build."""
    status = dict(_status(), task_paid={"reference_confirmed": True, "verifier_passed": True,
                                        "checks": {"mutation_flips": False}})
    assert [row["task_ids"] for row in F.suite_rows(status) if row["key"] == "suite:mutation_flips:"] == [[FINE]]
    assert F.suite_rows({RENEW: _status()[RENEW]}) == []


def test_a_verifier_is_filed_against_its_task_only_when_it_rejects_every_held_out_run_and_names_the_atom(tmp_path):
    """D133: the required atoms recognise no path but their seeds. The number is the loosening gate's;
    the atom is not in the ruling, so it is recomputed against the Run the gate rejected."""
    strict = tighten(base(tmp_path)).model_copy(update={"seed_run_ids": ["ref"]})
    store = {"verifiers": [strict], "task_runs": {TASK: [reference_run(), other_reason_run()]}, "sigs": SIGS,
             "replays": {TASK: {"tr1": {"trace_id": "tr1", "run_id": "ref", "confirmed": True, "path": ""}}},
             "rerolls": {TASK: [{"run_id": "rr2", "termination_reason": "success", "path": ""}]},
             "canon_rules": CanonRules()}
    row = F.false_rejection_rows(store)[0]
    assert row["kind"] == "false_rejection" and row["task_id"] == TASK and row["run_id"] == "rr2"
    assert "reject all 1 held-out" in row["text"]
    assert "w0.reason" in row["change"] and "w0.reason" in row["text"], "the atom that rejected the Run"
    accepting = dict(store, verifiers=[base(tmp_path).model_copy(update={"seed_run_ids": ["ref"]})])
    assert F.false_rejection_rows(accepting) == [], "a Verifier that accepts a held-out Run is not filed"


def test_a_task_whose_recordings_disagree_is_filed_for_refusal_and_one_a_tool_blocks_is_not():
    """The corpus not settling on an End state is the Task's own loss and refusing it answers it. A Task whose recordings disagree because its tool replays differently is the tool's loss, and
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
    assert rows[0]["kind"] == "reference_disagreement"
    assert "A: renew_loan on L1; B: no writes" in rows[0]["text"]


def test_a_task_the_judge_failed_on_every_recording_is_filed_as_a_disagreement_too():
    references = {FINE: {"references": [], "recordings": ["r1", "r2"], "groups": [],
                         "failed": {"r1": "judge: neither state does it", "r2": "judge: neither state does it"},
                         "reason": "the judge failed every recording"}}
    row = F.disagreement_rows({}, references)[0]
    assert "the judge failed all 2 recordings" in row["text"]
    assert F.disagreement_rows({}, {FINE: dict(references[FINE], failed={"r1": "judge: no"})}) == [], \
        "a judge that failed one recording of two has not failed the Task"


# --- keys, fidelity and the kinds a finding may take ------------------------------------


def test_two_findings_are_the_same_finding_when_the_kind_and_the_thing_they_are_about_agree():
    assert F.finding_key("assisted_tool", "renew_loan") == F.finding_key("assisted_tool", "renew_loan", "")
    assert F.finding_key("suite", "mutation_flips") != F.finding_key("suite", "mutation_flips", "not_run")
    assert F.finding_key("fidelity", "", RENEW) != F.finding_key("environment", "", RENEW)


def test_a_task_whose_replay_never_reached_its_end_state_is_filed_as_the_fidelity_loss_it_is():
    """One live build labelled 1 of 46 findings fidelity while replay_reference was what blocked 58
    Tasks: the loss reached the Builder under the name of whichever grading layer the Examiner had
    read, so the Builder worked Verifiers of Tasks that had no Reference to derive one from."""
    rows = F.fidelity_rows(_status(), _fidelity(), _replays())
    assert [r["task_id"] for r in rows] == [CARD, HOLD, RENEW], "one per Task, none for the confirmed one"
    renew = next(r for r in rows if r["task_id"] == RENEW)
    assert renew["kind"] == "fidelity" and renew["tool"] == "renew_loan"
    assert renew["change"] == DUE_DATE
    assert DUE_DATE in renew["text"] and "renew_loan" in renew["text"]
    assert next(r for r in rows if r["task_id"] == HOLD)["tool"] == "place_hold", "its own blocker"
    card = next(r for r in rows if r["task_id"] == CARD)
    assert card["tool"] is None and card["edits"] == [], "no tool is named, so no body edit is"
    assert card["change"] == "the End state was never reached"


def test_a_runs_disagree_finding_validates_and_is_filed(tmp_path):
    """The kind D213 files under was not one FindingKind carried, so every such filing raised on
    validation and the finding never reached the Builder (D227)."""
    from kullback.runner.records import Finding

    assert Finding(finding_id="finding-1", kind="runs_disagree", text="two versions of one row",
                   key="k").kind == "runs_disagree"
    pins = {"runs_disagree": [
        {"task_id": RENEW, "table": "loans", "key_class": "own", "column_classes": ["hard"],
         "columns": 1, "run_ids": ["trace-1", "trace-9"], "split_candidate": True},
    ]}
    row = F.runs_disagree_rows(pins)[0]
    plan = _plan(tmp_path, status=_status(), fidelity=_fidelity(), references={})
    filed = F.file_finding(plan, kind=row["kind"], text=row["text"], key=row["key"],
                           change=row["change"], task_id=row["task_id"],
                           task_ids=row["task_ids"])
    assert filed.kind == "runs_disagree" and filed.edits == []
    assert plan.store["findings"][-1]["kind"] == "runs_disagree"


# --- one finding shape: a diff with the why and the rows (D317) --------------------------

def test_a_differing_call_becomes_a_body_edit_naming_call_column_and_both_values_and_no_verb():
    worded = "call c7: due_date recorded 2026-03-15 ours 2026-03-01"
    calls = [{"call_id": "c7", "column": "due_date", "recorded": "2026-03-15", "ours": "2026-03-01"}]
    fidelity = {"tools": {"renew_loan": {"calls": 2, "replayed": 1}},
                "tasks": {RENEW: {"renew_loan": {"replayed": 1, "differing": 1, "reasons": [worded],
                                                 "differing_calls": calls}}}}
    status = {RENEW: {"reference_confirmed": False, "blocking_tools": ["renew_loan"]}}
    for row in (F.assisted_tool_rows(status, fidelity)[0],
                F.fidelity_rows(status, fidelity, {RENEW: {"trace-1": {"confirmed": False, "reasons": [worded]}}})[0]):
        assert "suggested" not in row and "hint" not in row
        [edit] = row["edits"]
        assert edit == {"kind": "body", "path": "env/tools/renew_loan.py", "call_id": "c7", "column": "due_date",
                        "recorded": "2026-03-15", "replayed": "2026-03-01", "why": edit["why"]}
        assert edit["why"]


def test_a_fidelity_file_written_before_differing_calls_gives_no_edit_and_keeps_the_change_line():
    worded = "call c7: due_date recorded 2026-03-15 ours 2026-03-01"
    fidelity = {"tools": {"renew_loan": {"calls": 2, "replayed": 1}},
                "tasks": {RENEW: {"renew_loan": {"replayed": 1, "differing": 1, "reasons": [worded]}}}}
    status = {RENEW: {"reference_confirmed": False, "blocking_tools": ["renew_loan"]}}
    [row] = F.assisted_tool_rows(status, fidelity)
    assert row["edits"] == [] and row["change"] == worded


def test_a_rule_finding_files_its_edits_into_the_store(tmp_path):
    plan = _plan(tmp_path, status=_status(), fidelity=_fidelity(), references={})
    edit = {"kind": "body", "path": "env/tools/renew_loan.py", "call_id": "c7", "column": "due_date",
            "recorded": "a", "replayed": "b", "why": "the recording is the standard"}
    filed = F.file_finding(plan, kind="fidelity", text="x", key="k", change="due_date differs",
                           edits=[edit], task_ids=[RENEW])
    assert filed.edits == [edit] and plan.store["findings"][-1]["edits"] == [edit]


def test_a_findings_file_written_with_a_verb_and_a_hint_still_loads():
    from kullback.runner.records import Finding

    old = {"finding_id": "finding-1", "task_id": RENEW, "kind": "assisted_tool", "text": "x",
           "tool": "renew_loan", "suggested": "compile_tool", "hint": DUE_DATE, "round": 1,
           "status": "open", "task_ids": [RENEW], "key": "assisted_tool:renew_loan:"}
    loaded = Finding.model_validate(old)
    assert loaded.change == DUE_DATE and loaded.edits == []
    assert "suggested" not in loaded.model_dump() and "hint" not in loaded.model_dump()
