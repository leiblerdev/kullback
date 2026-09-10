"""The eight Examiner tools, driven through the harness's registry and hooks with no model turn: what each
writes, what the gates say about it, and what is refused (D120, D123, D127, D128, D133)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examiner.worlds import WORLD_TASK as T
from examiner.worlds import World, drive, events_of, make_world, probe_runner_over
from gates import verifier_fixtures as VF
from kullback.examiner import agent as examiner_agent
from kullback.examiner import tools as tools_mod
from kullback.examiner.plan import ExaminerPlan
from kullback.gates.probes import version_hash
from kullback.runner.records import Intent, Verifier, VerifierHistory, as_dict, content_hash

TOOL_NAMES = ["read", "search", "derive", "probe", "repair", "refuse", "reroll", "finding"]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _history(world: World) -> VerifierHistory:
    return VerifierHistory.model_validate(_read(world.workdir / "examiner" / "history" / f"{T}.json"))


def _harness(world: World, **kwargs):
    plan = world.plan(probe_model=object(), run_probe=probe_runner_over(), **kwargs)
    return plan, examiner_agent.examiner_harness(plan)


def _probe(harness, run, bug_class="other", call_id="p"):
    return drive(harness, "probe", {"task_id": T, "bug_class": bug_class, "events": events_of(run),
                                    "termination_reason": run.termination_reason}, call_id=call_id)


def _atom_row(verifier: Verifier, atom_id: str, kind: str) -> dict:
    atom = next(a for a in verifier.atoms if a.id == atom_id)
    return {"id": atom.id, "kind": kind, "payload": dict(atom.target or {}), "description": atom.description}


def _reason_repair(harness, plan, kind: str, reason: str, call_id: str = "r"):
    """The cancel reason required (`kind` required: a tightening) or allowed again (a loosening)."""
    row = _atom_row(plan.current(T), "w0.reason", kind)
    return drive(harness, "repair", {"task_id": T, "reason": reason, "drop": ["w0.reason"], "add": [row]},
                 call_id=call_id)


def test_derive_writes_verifiers_task_status_references_and_one_history_row_per_task(world):
    plan, harness = _harness(world)
    result = drive(harness, "derive", {"target": "all"})
    assert result.is_error is False and result.details["verifiers"] == [T] and result.details["passed"] == 1
    assert [r["stage"] for r in result.details["rulings"]] == ["compile_policy", "derive_verifier"]
    verifier = Verifier.model_validate(_read(world.workdir / "verifiers" / f"{T}.json"))
    status = _read(world.workdir / "task_status.json")[T]
    assert verifier.task_id == T and status["reference_confirmed"] and status["verifier_passed"]
    assert _read(world.workdir / "references.json")[T]["references"] == [
        {"run_id": "ref", "trace_id": "ref", "kind": "recording"}, {"run_id": "alt", "trace_id": None, "kind": "reroll"}]
    history = _history(world)
    assert len(history.versions) == 1 and history.versions[0].by == "derive" and history.versions[0].accepted
    assert history.versions[0].content_hash == version_hash(verifier) and history.versions[0].verifier_version == "1"
    assert plan.current(T) is not None and version_hash(plan.current(T)) == version_hash(verifier)


def test_deriving_again_with_nothing_changed_adds_no_history_row(derived):
    plan, harness = _harness(derived)
    before = _history(derived)
    result = drive(harness, "derive", {"target": T})
    assert result.is_error is False and result.details["verifiers"] == [T]
    after = _history(derived)
    assert len(after.versions) == len(before.versions) == 1
    assert after.versions[0].content_hash == before.versions[0].content_hash


def test_probe_scores_the_run_against_the_current_verifier_and_keeps_it_in_the_pool_either_way(derived):
    plan, harness = _harness(derived)
    passed = _probe(harness, VF.alt_path_run(), "visible-test overfitting")
    failed = _probe(harness, VF.wrong_run(), "schema-only validation")
    assert passed.details["scored_pass"] is True and passed.details["failing_atom"] is None
    assert failed.details["scored_pass"] is False and failed.details["failing_atom"] == "w0"
    pool = _read(derived.workdir / "probes" / T / "pool.json")
    assert [p["probe_id"] for p in pool["probes"]] == [f"probe-{T}-1", f"probe-{T}-2"]
    assert [p["scored_pass"] for p in pool["probes"]] == [True, False]
    assert {p["verifier_hash"] for p in pool["probes"]} == {version_hash(plan.current(T))}
    assert [p["bug_class"] for p in pool["probes"]] == ["visible-test overfitting", "schema-only validation"]
    assert pool["probes"][1]["run"]["model"] == "probe:examiner" and failed.details["pool_size"] == 2


def test_a_probe_that_scores_a_pass_fails_the_pool_ruling_on_the_result(derived):
    plan, harness = _harness(derived)
    rejected = _probe(harness, VF.wrong_run())
    assert {r["stage"]: r["passed"] for r in rejected.details["rulings"]} == {"probe_pool": True, "probe_admission": True}
    passed = _probe(harness, VF.alt_path_run())
    rulings = {r["stage"]: r for r in passed.details["rulings"]}
    assert rulings["probe_pool"]["passed"] is False and f"probe-{T}-2" in rulings["probe_pool"]["failures"][0]
    assert "probe_pool fail" in passed.content
    recorded = [row for row in _read(derived.workdir / "gates.json") if row["stage"] == "probe_pool"]
    assert len(recorded) == 1 and recorded[0]["pass"] is False


def test_a_fourth_probe_after_three_rejected_ones_is_refused_by_the_admission_gate(derived):
    plan, harness = _harness(derived)
    for run in (VF.wrong_run(), VF.failed_run(), VF.empty_run()):
        result = _probe(harness, run)
        assert result.is_error is False and result.details["scored_pass"] is False
    assert result.details["consecutive_failed"] == 3
    assert {r["stage"]: r["passed"] for r in result.details["rulings"]} == {"probe_pool": True, "probe_admission": False}
    fourth = _probe(harness, VF.extra_write_run())
    assert fourth.is_error and "probe refused" in fourth.content and T in fourth.content
    assert len(_read(derived.workdir / "probes" / T / "pool.json")["probes"]) == 3


def test_a_probe_of_a_task_with_no_verifier_is_an_error_result(world):
    plan, harness = _harness(world)
    result = _probe(harness, VF.wrong_run())
    assert result.is_error and "no live Verifier" in result.content and T in result.content
    assert not (world.workdir / "probes").exists()
    missing = drive(harness, "probe", {"task_id": "nobody", "bug_class": "other", "events": []})
    assert missing.is_error and "no Task is named nobody" in missing.content


def test_repair_that_tightens_is_accepted_and_becomes_the_current_version(derived):
    plan, harness = _harness(derived)
    current = plan.current(T)
    result = drive(harness, "repair", {"task_id": T, "reason": "the reason is what the user gave; require it",
                                       "drop": ["w0.reason"], "add": [_atom_row(current, "w0.reason", "required")]})
    assert result.is_error is False and result.details["accepted"] is True and result.details["rejected_by"] == []
    assert {r["stage"]: r["passed"] for r in result.details["rulings"]} == {
        "derive_verifier": True, "probe_pool": True, "loosening": True}
    on_disk = Verifier.model_validate(_read(derived.workdir / "verifiers" / f"{T}.json"))
    assert version_hash(on_disk) == result.details["content_hash"] != version_hash(current)
    assert next(a.kind for a in on_disk.atoms if a.id == "w0.reason") == "required"
    assert on_disk.verifier_version == "2" == result.details["verifier_version"]
    status = _read(derived.workdir / "task_status.json")[T]
    assert status["verifier_passed"] and all(status["checks"].values())
    # The tightened version fails the Run that gave another reason, which the derived one passed.
    assert plan.current(T) is not None and version_hash(plan.current(T)) == version_hash(on_disk)


def test_repair_that_newly_passes_a_probe_is_rejected_and_the_current_version_stays(derived):
    plan, harness = _harness(derived)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    tightened = plan.current(T)
    caught = _probe(harness, VF.other_reason_run(), "loose answer extraction")
    assert caught.details["scored_pass"] is False and caught.details["failing_atom"] == "w0.reason"
    result = _reason_repair(harness, plan, "allowed", "any reason will do")
    assert result.is_error is False and result.details["accepted"] is False
    assert result.details["rejected_by"] == ["probe_pool"]
    pool = next(r for r in result.details["rulings"] if r["stage"] == "probe_pool")
    assert f"probe-{T}-1" in pool["failures"][0] and result.details["content_hash"][:12] in pool["failures"][0]
    on_disk = Verifier.model_validate(_read(derived.workdir / "verifiers" / f"{T}.json"))
    assert version_hash(on_disk) == version_hash(tightened) == version_hash(plan.current(T))
    assert _read(derived.workdir / "task_status.json")[T]["verifier_passed"] is True


def test_repair_that_newly_passes_a_run_outside_the_legitimate_pool_is_rejected(tmp_path):
    # rr2 is on disk as a re-roll that died on max_steps: not a Reference and not legitimate.
    world = make_world(tmp_path, rerolls=("alt", "rr2"), terminations={"rr2": "max_steps"})
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(), run_probe=probe_runner_over())
    plan, harness = _harness(world)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    result = _reason_repair(harness, plan, "allowed", "any reason will do")
    assert result.is_error is False and result.details["accepted"] is False
    assert result.details["rejected_by"] == ["loosening"]
    loosening = next(r for r in result.details["rulings"] if r["stage"] == "loosening")
    assert "rr2" in loosening["failures"][0]
    assert [v.accepted for v in _history(world).versions] == [True, True, False]


def test_a_derive_after_an_accepted_tightening_that_would_newly_pass_a_run_outside_the_legitimate_pool_is_recorded_rejected_and_the_tightened_version_stays_current(tmp_path):
    """A derivation is a new version like a repair: the loosening gate rules on it against the last
    accepted version, so re-deriving cannot undo a tightening the gates accepted (D127)."""
    world = make_world(tmp_path, rerolls=("alt", "rr2"), terminations={"rr2": "max_steps"})
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(), run_probe=probe_runner_over())
    plan, harness = _harness(world, round=2)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    tightened = plan.current(T)
    status_before = _read(world.workdir / "task_status.json")[T]
    result = drive(harness, "derive", {"target": T})
    assert result.is_error is False and result.details["verifiers"] == [T]
    assert [(r["stage"], r["passed"]) for r in result.details["rulings"]] == [
        ("compile_policy", True), ("derive_verifier", True), ("loosening", False)]
    assert "loosening" in result.content and "rr2" in next(r["failures"][0] for r in result.details["rulings"]
                                                              if r["stage"] == "loosening")
    history = _history(world)
    assert [(v.by, v.accepted) for v in history.versions] == [("derive", True), ("repair", True), ("derive", False)]
    assert history.versions[-1].rejected_by == ["loosening"] and history.versions[-1].round == 2
    assert history.versions[-1].parent_hash == version_hash(tightened)
    on_disk = Verifier.model_validate(_read(world.workdir / "verifiers" / f"{T}.json"))
    assert version_hash(on_disk) == version_hash(tightened) == version_hash(plan.current(T))
    assert _read(world.workdir / "task_status.json")[T]["verifier_passed"] is True
    assert _read(world.workdir / "task_status.json")[T] == status_before
    recorded = [row for row in _read(world.workdir / "gates.json") if row["stage"] == "loosening"]
    assert len(recorded) == 1 and recorded[0]["pass"] is False
    # Deriving again finds the same rejected derivation as the last row and adds nothing.
    again = drive(harness, "derive", {"target": T})
    assert again.is_error is False and len(_history(world).versions) == 3


def test_repair_that_newly_passes_a_frontier_reroll_of_a_later_round_is_accepted(derived):
    plan, harness = _harness(derived, round=2)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    plan.extra_rerolls[T] = [{"run_id": "rr2", "path": derived.paths["rr2"], "termination_reason": "success"}]
    plan.write_state()
    plan.load_state()
    assert [row["run_id"] for row in plan.store["rerolls"][T]] == ["alt", "rr2"]
    result = _reason_repair(harness, plan, "allowed", "the frontier gives its own reasons")
    assert result.is_error is False and result.details["accepted"] is True, result.content
    assert result.details["verifier_version"] == "3"
    history = _history(derived)
    assert [v.accepted for v in history.versions] == [True, True, True] and history.versions[-1].round == 2
    assert version_hash(plan.current(T)) == history.versions[-1].content_hash


def test_a_rejected_repair_is_kept_in_the_history_with_the_gate_that_rejected_it(derived):
    plan, harness = _harness(derived)
    _reason_repair(harness, plan, "required", "require the reason")
    _probe(harness, VF.other_reason_run())
    result = _reason_repair(harness, plan, "allowed", "any reason will do")
    assert result.details["accepted"] is False
    history = _history(derived)
    assert [(v.by, v.accepted) for v in history.versions] == [("derive", True), ("repair", True), ("repair", False)]
    rejected = history.versions[-1]
    assert rejected.rejected_by == ["probe_pool"] and rejected.reason == "any reason will do"
    assert rejected.content_hash == result.details["content_hash"] == version_hash(rejected.verifier)
    assert rejected.parent_hash == history.versions[1].content_hash
    # A repair that leaves no atom is refused before any gate sees it.
    empty = drive(harness, "repair", {"task_id": T, "reason": "drop all",
                                      "drop": [a.id for a in plan.current(T).atoms]})
    assert empty.is_error and "without atoms" in empty.content and len(_history(derived).versions) == 3


def test_every_version_is_named_by_its_content_hash_and_the_file_matches_the_last_accepted_row(derived):
    plan, harness = _harness(derived)
    current = plan.current(T)
    accepted = _reason_repair(harness, plan, "required", "require the reason")
    _probe(harness, VF.other_reason_run())
    rejected = _reason_repair(harness, plan, "allowed", "any reason will do")
    assert accepted.details["accepted"] and not rejected.details["accepted"]
    assert version_hash(current) == _history(derived).versions[0].content_hash
    history = _history(derived)
    for row in history.versions:
        assert row.content_hash == version_hash(row.verifier) == content_hash(as_dict(row.verifier))
    on_disk = Verifier.model_validate(_read(derived.workdir / "verifiers" / f"{T}.json"))
    last_accepted = [v for v in history.versions if v.accepted][-1]
    assert version_hash(on_disk) == last_accepted.content_hash == accepted.details["content_hash"]
    assert history.versions[-1].content_hash == rejected.details["content_hash"] != version_hash(on_disk)


def test_refuse_is_admitted_when_no_frontier_run_finished_and_marks_the_task_status(tmp_path):
    world = make_world(tmp_path, rerolls=("bad",), confirmed=False)
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs)
    assert _read(world.workdir / "task_status.json")[T]["reference_confirmed"] is False
    plan, harness = _harness(world, round=1)
    result = drive(harness, "refuse", {"task_id": T, "reason": "no Run of it ever finished"})
    assert result.is_error is False and result.details["admitted"] is True and result.details["finished_runs"] == []
    assert [(r["stage"], r["passed"]) for r in result.details["rulings"]] == [("refuse", True)]
    refusals = _read(world.workdir / "examiner" / "refusals.json")
    assert refusals[T]["reason"] == "no Run of it ever finished" and refusals[T]["admitted"] and refusals[T]["round"] == 1
    assert _read(world.workdir / "task_status.json")[T]["refused"] == {"reason": "no Run of it ever finished", "round": 1}
    assert plan.refusal(T) is not None and plan.refusal(T).task_id == T


def test_refuse_is_rejected_when_a_reroll_finished_and_the_error_names_it(derived):
    plan, harness = _harness(derived)
    result = drive(harness, "refuse", {"task_id": T, "reason": "too hard"})
    assert result.is_error and "rejected" in result.content
    assert "alt" in result.content and "ref" in result.content
    assert not (derived.workdir / "examiner" / "refusals.json").is_file() or \
        _read(derived.workdir / "examiner" / "refusals.json") == {}
    assert "refused" not in _read(derived.workdir / "task_status.json")[T]


def test_reroll_runs_count_more_runs_through_the_runner_callable_and_records_them_beside_the_builders_rows(derived):
    asked = []

    def run_rerolls(task_id: str, count: int, prefix: str) -> list[dict]:
        asked.append((task_id, count, prefix))
        rows = []
        for number in range(count):
            run = VF.make_run(f"{prefix}-{task_id}-{number}", VF.reference_events())
            path = VF.write_events_jsonl(run, derived.workdir / "runs" / task_id / f"{run.run_id}.jsonl")
            rows.append({"run_id": run.run_id, "path": path, "termination_reason": "success"})
        return rows

    plan, harness = _harness(derived, run_rerolls=run_rerolls, round=3)
    result = drive(harness, "reroll", {"task_id": T, "count": 2})
    assert result.is_error is False and asked == [(T, 2, "reroll-r3")]
    assert result.details["runs"] == [f"reroll-r3-{T}-0", f"reroll-r3-{T}-1"] and result.details["finished"] == 2
    rows = _read(derived.workdir / "examiner" / "rerolls.json")[T]
    assert [r["run_id"] for r in rows] == result.details["runs"]
    assert [r["run_id"] for r in plan.store["rerolls"][T]] == ["alt", *result.details["runs"]]
    assert {r.run_id for r in plan.store["task_runs"][T]} >= {"ref", "alt", *result.details["runs"]}
    assert [r["run_id"] for r in derived.inputs["rerolls"][T]] == ["alt"], "the Builder's rows are not rewritten"
    again = drive(harness, "reroll", {"task_id": T, "count": 1})
    assert again.details["runs"] == [f"reroll-r3-1-{T}-0"], "a second call in the round takes a longer prefix"
    # D212: the attempt index is read off this Task's own rows, so a re-roll of another Task in the
    # same round starts at its own first attempt rather than at the next number nobody has used.
    plan.extra_rerolls["other"] = [{"run_id": "reroll-r3-other-0", "path": "", "termination_reason": "success"}]
    plan.store.setdefault("rerolls", {})["other"] = list(plan.extra_rerolls["other"])
    third = drive(harness, "reroll", {"task_id": T, "count": 1})
    assert third.details["runs"] == [f"reroll-r3-2-{T}-0"], \
        "another Task's re-rolls are not an input to this Task's attempt index"


def test_reroll_without_a_runner_is_an_error_result_not_a_crash(derived):
    plan, harness = _harness(derived)
    result = drive(harness, "reroll", {"task_id": T, "count": 1})
    assert result.is_error and "no Runner" in result.content
    assert not (derived.workdir / "examiner" / "rerolls.json").is_file() or \
        _read(derived.workdir / "examiner" / "rerolls.json") == {}


def test_reroll_refuses_when_the_allowance_is_spent(derived):
    called = []

    def run_rerolls(task_id: str, count: int, prefix: str) -> list[dict]:
        called.append(task_id)
        return []

    plan, harness = _harness(derived, run_rerolls=run_rerolls, allowance_remaining=0.0)
    result = drive(harness, "reroll", {"task_id": T, "count": 1})
    assert result.is_error and "allowance" in result.content and called == []
    plan.allowance_remaining = 1.0
    assert drive(harness, "reroll", {"task_id": T, "count": 1}).is_error is False and called == [T]


def test_finding_returns_the_structured_record_in_details_and_files_it(derived):
    plan, harness = _harness(derived, round=1)
    result = drive(harness, "finding", {"task_id": T, "kind": "environment", "text": "the order table lacks a row",
                                        "run_id": "ref", "tool": "cancel_pending_order", "suggested": "compile_tool"})
    assert result.is_error is False and result.details["finding_id"] == "finding-1"
    record = result.details["finding"]
    assert record["kind"] == "environment" and record["status"] == "open" and record["round"] == 1
    assert record["suggested"] == "compile_tool" and record["tool"] == "cancel_pending_order"
    assert record["about_entry_id"] is None, "no session, so no entry to point at"
    assert result.details["produced"] == ["findings"] and "finding-1" in result.content
    assert _read(derived.workdir / "examiner" / "findings.json") == [record]
    assert [f.finding_id for f in plan.open_findings()] == ["finding-1"]
    assert plan.close_findings(["finding-1"]) == ["finding-1"] and plan.open_findings() == []
    assert _read(derived.workdir / "examiner" / "findings.json")[0]["status"] == "closed"


def test_a_finding_can_suggest_repair_intent_with_a_hint(derived):
    """The Examiner names the Builder verb that answers the finding and the line that verb is given.
    Build 11 could only say `replay`, so the Builder replayed a cached result three times."""
    plan, harness = _harness(derived, round=2)
    hint = "the Runs only ever cancel one order; nothing in them mentions a gift card"
    result = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "suggested": "repair_intent",
                                        "hint": hint,
                                        "text": "the Intent says gift card and no Run of this Task says it"})
    assert result.is_error is False
    record = result.details["finding"]
    assert record["suggested"] == "repair_intent" and record["hint"] == hint
    assert "suggested repair_intent" in result.content and "with a hint" in result.content
    assert _read(derived.workdir / "examiner" / "findings.json")[-1]["hint"] == hint
    again = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "suggested": "repair_recompile",
                                       "tool": "cancel_pending_order", "hint": "  the status column differs  ",
                                       "text": "the body writes a different status on replay"},
                  call_id="f2")
    assert again.details["finding"]["hint"] == "the status column differs", "the hint is stored trimmed"
    assert again.details["finding"]["suggested"] == "repair_recompile"
    unknown = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "text": "x",
                                         "suggested": "repair_everything"}, call_id="f3")
    assert unknown.is_error, "a verb the Builder has no tool for is refused by the schema"


def test_a_second_finding_of_the_same_kind_on_the_same_subject_is_refused_with_the_open_findings_id(derived):
    """D170: one loss is one finding. Build 12's Examiner made 24 finding calls to put 7 on the list;
    a model filing the same thing again is told which finding already holds it, and the answer names
    the id so it can act on that one instead of adding a second row for one loss."""
    plan, harness = _harness(derived, round=1)
    first = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "text": "the body differs on replay",
                                       "tool": "cancel_pending_order", "suggested": "repair_recompile",
                                       "hint": "the status column differs"})
    again = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "text": "said again, other words",
                                       "tool": "cancel_pending_order", "suggested": "repair_recompile"},
                  call_id="f2")
    assert again.is_error and first.details["finding_id"] in again.content
    assert "cancel_pending_order" in again.content and T in again.content
    assert len(_read(derived.workdir / "examiner" / "findings.json")) == 1, "one loss, one row"
    plan.close_findings([first.details["finding_id"]])
    reopened = drive(harness, "finding", {"task_id": T, "kind": "fidelity", "text": "still differs",
                                          "tool": "cancel_pending_order", "suggested": "repair_recompile"},
                     call_id="f3")
    assert reopened.is_error is False, "the loss is still there after the Builder answered: a new finding"


def test_derive_files_the_losses_its_own_records_show_before_the_model_chooses_anything(tmp_path):
    """D170: a finding the model chooses to write is a finding the model may not write. The derivation
    files what the records say on its way out, so the list exists on a code-driven beat that calls
    nothing else, and the round driver reads it off the derive result."""
    world = make_world(tmp_path, rerolls=("rr2",))
    plan = world.plan()
    harness = examiner_agent.examiner_harness(plan)
    result = drive(harness, "derive", {"target": "all"})
    filed = result.details["findings"]
    assert [(f["kind"], f["task_id"], f["suggested"]) for f in filed] == [
        ("reference_disagreement", T, "repair_refuse_task")]
    assert filed[0]["task_ids"] == [T] and filed[0]["key"] == f"reference_disagreement::{T}"
    assert "findings filed from the records, most costly first" in result.details["summary"]
    assert [f.finding_id for f in plan.open_findings()] == [filed[0]["finding_id"]]
    assert _read(world.workdir / "examiner" / "findings.json")[0]["round"] == plan.round


def test_an_examiner_reading_an_intent_record_sees_the_refused_phrases(derived):
    """`read` with kind `intent` is where the Examiner learns what the intent gate refused: the
    ungrounded phrases and, for the phrases that are grounded, the Runs that evidence them. Without
    them a repair_intent hint would be a guess."""
    derived.inputs["intents"] = {T: as_dict(Intent(
        task_id=T, text="cancel the order and refund the gift card", grounded=False,
        ungrounded_phrases=["gift card", "refund"],
        run_coverage={"the order": ["ref", "alt"]},
        reason="two noun phrases no Run of this Task says"))}
    plan, harness = _harness(derived)
    result = drive(harness, "read", {"kind": "intent", "id": T})
    assert result.is_error is False
    intent = json.loads(result.details["text"])
    assert intent["ungrounded_phrases"] == ["gift card", "refund"]
    assert intent["run_coverage"] == {"the order": ["ref", "alt"]}
    assert intent["grounded"] is False and "gift card" in intent["text"]
    schema = harness.registry.get("read").schema()["input_schema"]
    assert "ungrounded_phrases" in schema["properties"]["kind"]["description"]


def test_read_returns_a_run_a_trace_an_intent_a_verifier_and_the_pool_as_json(derived, fixture_build):
    plan, harness = _harness(derived)
    _probe(harness, VF.wrong_run())
    run = json.loads(drive(harness, "read", {"kind": "run", "id": "alt"}).details["text"])
    assert run["run_id"] == "alt" and len(run["events"]) == len(VF.alt_path_run().events)
    verifier = json.loads(drive(harness, "read", {"kind": "verifier", "id": T}).details["text"])
    assert verifier["task_id"] == T and {a["id"] for a in verifier["atoms"]} >= {"w0", "entity_count"}
    pool = json.loads(drive(harness, "read", {"kind": "probes", "id": T}).details["text"])
    assert [p["probe_id"] for p in pool["probes"]] == [f"probe-{T}-1"]
    probe_run = json.loads(drive(harness, "read", {"kind": "run", "id": f"probe-{T}-1"}).details["text"])
    assert probe_run["model"] == "probe:examiner"
    status = json.loads(drive(harness, "read", {"kind": "task_status"}).details["text"])
    assert set(status["rows"]) == {T} and status["tasks"] == 1, "a read with no id is an index (D175)"
    whole = json.loads(drive(harness, "read", {"kind": "task_status", "id": T}).details["text"])
    assert set(whole[T]) >= {"reference_confirmed", "verifier_passed"}
    missing = drive(harness, "read", {"kind": "run", "id": "nowhere"})
    assert missing.is_error is False and "nowhere" in missing.content, "an id nothing carries is answered"
    # Traces and Intents come from a Builder store: the fixture build's.
    built = ExaminerPlan(workdir=fixture_build.workdir, inputs=fixture_build.inputs)
    reader = examiner_agent.examiner_harness(built)
    task = fixture_build.inputs["tasks"][0]
    trace = json.loads(drive(reader, "read", {"kind": "trace", "id": task.run_ids[0]}).details["text"])
    assert trace["trace_id"] == task.run_ids[0] and trace["turns"]
    intent_id = next(iter(fixture_build.inputs["intents"]))
    intent = json.loads(drive(reader, "read", {"kind": "intent", "id": intent_id}).details["text"])
    assert intent["task_id"] == intent_id and "text" in intent
    listed = json.loads(drive(reader, "read", {"kind": "task", "id": task.id}).details["text"])
    assert listed["id"] == task.id


def test_the_examiner_store_never_holds_bodies_db_schema_or_the_environment(fixture_build, tmp_path):
    full = fixture_build.plan.store
    assert {"bodies", "db", "schema", "environment"} <= set(full)
    with pytest.raises(ValueError, match="bodies"):
        ExaminerPlan(workdir=tmp_path, inputs=full)
    plan = ExaminerPlan(workdir=tmp_path, inputs=fixture_build.inputs)
    assert not {"bodies", "db", "schema", "environment", "overlays", "policy_text"} & set(plan.store)
    assert not {"bodies", "db", "schema", "environment"} & set(plan.inputs)
    plan.refresh(fixture_build.inputs)
    assert not {"bodies", "db", "schema", "environment"} & set(plan.store)
    with pytest.raises(ValueError):
        plan.refresh(full)
    assert [tool.name for tool in tools_mod.examiner_tools(plan)] == TOOL_NAMES


def test_a_rejected_derivation_restores_the_prior_task_status_and_references_rows(tmp_path):
    """Greptile P1: the loosening rejection restores a version-consistent artifact set. The on-disk
    rows are marked before the rejected derive, so the test fails when the rejected derivation's
    rows are kept against the restored Verifier."""
    world = make_world(tmp_path, rerolls=("alt", "rr2"), terminations={"rr2": "max_steps"})
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(), run_probe=probe_runner_over())
    plan, harness = _harness(world, round=2)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    status_path, references_path = world.workdir / "task_status.json", world.workdir / "references.json"
    status_marked = _read(status_path)
    status_marked[T] = {"_prior": "the accepted version's own row"}
    status_path.write_text(json.dumps(status_marked), encoding="utf-8")
    references_marked = _read(references_path)
    references_marked[T] = {"_prior": "the accepted version's own row"}
    references_path.write_text(json.dumps(references_marked), encoding="utf-8")
    result = drive(harness, "derive", {"target": T})
    assert result.is_error is False and any(r["stage"] == "loosening" and not r["passed"]
                                            for r in result.details["rulings"])
    assert _read(status_path) == status_marked, "the rejected derivation's status row must not survive"
    assert _read(references_path) == references_marked, "the rejected derivation's references row must not survive"
    assert plan.store["task_status"] == status_marked
    on_disk = Verifier.model_validate(_read(world.workdir / "verifiers" / f"{T}.json"))
    assert version_hash(on_disk) == version_hash(plan.current(T))


def test_a_rejected_derivation_recomputes_the_scorecard_from_the_restored_rows(tmp_path):
    """Greptile P1 (round 2): derive_all rewrites scorecard.json from the rejected rows before the
    loosening gate rules. After a rejection the file must read as computed from the restored
    task_status.json: the marked row confirms nothing, so T is uncovered, where the rejected
    derivation's scorecard would have counted it covered."""
    from kullback.gates import scorecard as scorecard_mod

    world = make_world(tmp_path, rerolls=("alt", "rr2"), terminations={"rr2": "max_steps"})
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(), run_probe=probe_runner_over())
    plan, harness = _harness(world, round=2)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    status_path = world.workdir / "task_status.json"
    status_marked = _read(status_path)
    status_marked[T] = {"_prior": "the accepted version's own row"}
    status_path.write_text(json.dumps(status_marked), encoding="utf-8")
    # A denominator for task coverage and one clean code-routed Run, so the status row decides:
    # the marked row confirms nothing (T uncovered), the rejected row would confirm all (T covered).
    (world.workdir / "tasks.json").write_text(json.dumps({"tasks": [{"id": T, "run_ids": ["ref"]}]}),
                                                 encoding="utf-8")
    (world.workdir / "runs.json").write_text(
        json.dumps({"runs": [{"run_id": "ref", "task_id": T, "events": [], "route_counts": {}}]}),
        encoding="utf-8")
    result = drive(harness, "derive", {"target": T})
    assert result.is_error is False and any(r["stage"] == "loosening" and not r["passed"]
                                            for r in result.details["rulings"])
    on_disk = _read(world.workdir / "scorecard.json")
    assert on_disk == scorecard_mod.scorecard(world.workdir), \
        "the scorecard must be recomputed from the restored rows, not left as the rejected derive wrote it"
    uncovered_ids = {u["task_id"] for u in on_disk["task_coverage"]["uncovered"]}
    assert T in uncovered_ids, "the restored row confirms nothing for T; a stale scorecard would count it covered"


def test_a_whole_file_read_is_an_index_of_one_line_per_id_and_a_long_read_is_cut_with_the_cut_named(derived):
    """One live build's Examiner reached 899 percent of its window in round 1 reading task_status,
    gates and references whole (D175)."""
    from kullback.examiner import tools as examiner_tools

    plan, harness = _harness(derived)
    gates = json.loads(drive(harness, "read", {"kind": "gates"}).details["text"])
    assert set(gates) == {"stages", "note"} and all(set(v) == {"rulings", "failing"} for v in gates["stages"].values())
    index = examiner_tools._index("task_status", {
        "t1": {"reference_confirmed": True, "verifier_passed": False,
               "checks": {"mutation_flips": False, "empty_fails": True}},
        "t2": {"reference_confirmed": False, "blocking_tools": ["lookup_shelf"]}})
    assert index["rows"] == {"t1": "reference confirmed; verifier not passed; failed mutation_flips",
                             "t2": "no reference; verifier not passed; blocked by lookup_shelf"}
    long = examiner_tools._clamped_text({"rows": ["x" * 100] * 1000})
    assert len(long) < examiner_tools.READ_CHARS + 120 and "characters cut" in long.splitlines()[-1]
    assert examiner_tools._clamped_text({"a": 1}) == examiner_tools._text({"a": 1})


def test_a_finding_filed_under_a_ruling_name_lands_with_the_kind_that_ruling_is_about(derived):
    """One live build's Examiner filed 17 of its 28 findings under a ruling name (`replay_reference`)
    and every one was refused by the enum, 13 of them about real replay differences that were never
    retried. A ruling name is what the Examiner is reading when it files, so it is taken as the kind
    that ruling is about; a name that is neither is refused with the kinds and the mapping."""
    plan, harness = _harness(derived, round=1)
    filed = drive(harness, "finding", {"task_id": T, "kind": "replay_reference",
                                       "text": "the body answers a recorded call differently",
                                       "suggested": "repair_recompile", "hint": "the status differs"})
    assert filed.is_error is False
    assert filed.details["finding"]["kind"] == "fidelity", "the fidelity ruling maps to the fidelity kind"
    assert "(fidelity)" in filed.content
    suite = drive(harness, "finding", {"task_id": T, "kind": "mutation_flips", "text": "no atom names a value",
                                       "suggested": "repair"}, call_id="f2")
    assert suite.is_error is False and suite.details["finding"]["kind"] == "suite"
    unknown = drive(harness, "finding", {"task_id": T, "kind": "not_a_ruling", "text": "x"}, call_id="f3")
    assert unknown.is_error
    assert "assisted_tool, fidelity, reference_disagreement" in unknown.content, "the kinds are listed"
    assert "fidelity <- " in unknown.content, "and the mapping the model may use instead"
    assert tools_mod.finding_kind("ledger_rebalance", ["ledger_rebalance"]) == "assisted_tool"


def test_a_third_repair_of_one_task_against_the_check_that_rejected_the_first_two_is_refused(derived):
    """One live build's Examiner spent a session on 13 repairs, none accepted, two Tasks repaired
    four times each with the same gate failing every time. Two rejections by one check are what the
    session has to learn from; the third is refused with the check and the verbs that buy something."""
    plan, harness = _harness(derived)
    assert _reason_repair(harness, plan, "required", "require the reason").details["accepted"] is True
    _probe(harness, VF.other_reason_run())
    first = _reason_repair(harness, plan, "allowed", "any reason will do", call_id="r1")
    second = _reason_repair(harness, plan, "allowed", "said again, other words", call_id="r2")
    assert first.details["rejected_by"] == second.details["rejected_by"] == ["probe_pool"]
    third = _reason_repair(harness, plan, "allowed", "a third rationale", call_id="r3")
    assert third.is_error and "probe_pool" in third.content
    assert "any reason will do" in third.content and "said again, other words" in third.content
    assert "refuse, reroll_then_derive" in third.content
    assert len(_history(derived).versions) == 4, "the refused repair wrote no version"
    # The count is per check: a Task each of whose two rejections came from a different check is open.
    two_checks = {("t9", "verifier_mutation"): ["version 2: a"], ("t9", "loosening"): ["version 3: b"]}
    assert tools_mod.repair_lock(two_checks, "t9") is None
    one_check = {("t9", "verifier_mutation"): ["version 2: a", "version 3: b"]}
    assert "verifier_mutation" in (tools_mod.repair_lock(one_check, "t9") or "")


def test_a_repair_whose_atom_payload_is_not_an_object_is_a_validation_error_not_a_crash(derived):
    """One live build answered `repair failed: AttributeError: 'str' object has no attribute 'get'`
    three frames below the tool, and the Examiner had nothing to correct. The shape an atom takes is
    said once, in the words the tool's own schema uses."""
    plan, harness = _harness(derived)
    result = drive(harness, "repair", {"task_id": T, "reason": "name the entity",
                                       "add": [{"id": "a1", "kind": "required", "payload": "the entity"}]})
    assert result.is_error and "AttributeError" not in result.content
    assert "payload of atom a1 is str" in result.content and "`kind`" in result.content
    nameless = drive(harness, "repair", {"task_id": T, "reason": "x", "add": [{"kind": "required"}]},
                     call_id="r2")
    assert nameless.is_error and "names no id" in nameless.content
    assert len(_history(derived).versions) == 1, "no version was written for either"


def test_a_shape_refusal_hands_back_the_same_atom_written_the_way_the_tool_takes_it(derived):
    """Naming the rule was not enough: the one repair of a live build was refused cleanly for a
    payload that was a string, and no second call was ever made. The refusal now carries the atom
    the model sent, with the payload written as an object of the keys this Verifier's own atoms
    use, and the ask the loop puts in front of the model once (agent/loop.py)."""
    plan, harness = _harness(derived)
    result = drive(harness, "repair", {"task_id": T, "reason": "name the entity",
                                       "add": [{"id": "a1", "kind": "required", "payload": "the entity"}]})
    shape = json.loads(tools_mod.corrected_atom({"id": "a1", "kind": "required", "payload": "the entity"},
                                                plan.current(T).atoms))
    assert shape["id"] == "a1" and shape["kind"] == "required"
    assert set(shape["payload"]) == set(next(a.target for a in plan.current(T).atoms if a.kind == "required"))
    assert json.dumps(shape, sort_keys=True, ensure_ascii=False) in result.content
    assert result.details["retry_ask"].startswith("Your repair was refused for the shape of one atom")
    assert len(_history(derived).versions) == 1, "the refused repair wrote no version"


def test_reading_a_run_or_a_trace_id_nothing_carries_answers_the_ids_that_exist(derived):
    """One live build spent 13 of a session's 22 turns on one Task, four of them re-reading the same
    Run and Trace ids that were not there; the tool answered each with a KeyError and nothing to try."""
    plan, harness = _harness(derived)
    _probe(harness, VF.wrong_run())
    answer = json.loads(drive(harness, "read", {"kind": "run", "id": "reroll-r9-t1-0"}).details["text"])
    assert answer["found"] is False and answer["kind"] == "Run"
    assert set(answer["ids"]) == {f"{T}: ref", f"{T}: alt", f"{T}: probe-{T}-1"}
    assert "the Run ids of this build (3 of 3)" in answer["note"]
    by_task = json.loads(drive(harness, "read", {"kind": "run", "id": T}, call_id="r2").details["text"])
    assert by_task["found"] is False and f"of task {T}" in by_task["note"], "a Task id narrows the answer"
    trace = json.loads(drive(harness, "read", {"kind": "trace", "id": "nowhere"}, call_id="r3").details["text"])
    assert trace["found"] is False and trace["kind"] == "Trace" and trace["ids"] == []


def _unreadable_reference(world: World, failed: dict) -> None:
    """The Task's row left naming a Reference whose Run no replay and no re-roll of this session
    carries, which is the state one live round ended stalled in: the row was written by an earlier
    round and the Runs behind it are no longer among the session's own."""
    references = _read(world.workdir / "references.json")
    references[T] = {**references[T], "failed": failed,
                     "references": [{"kind": "reroll", "run_id": "gone-0", "trace_id": None}]}
    (world.workdir / "references.json").write_text(json.dumps(references), encoding="utf-8")


def test_a_repair_of_a_task_whose_reference_cannot_be_read_answers_a_refusal_and_not_an_error(derived):
    """D230: the tool raised here and the whole round ended stalled with its findings never
    delivered. A Task whose Reference cannot be read is a state of the Task, so the answer says so:
    how many of its Runs were turned down, under which reasons and how many times each, and the
    verbs that buy something instead of another repair."""
    _unreadable_reference(derived, {"gone-1": "the cancellation the request asked for never happened",
                                    "gone-2": "the cancellation the request asked for never happened",
                                    "gone-3": "it reached no End state: it wrote nothing"})
    plan, harness = _harness(derived)
    before = version_hash(plan.current(T))
    result = _reason_repair(harness, plan, "allowed", "any reason will do")
    assert result.is_error is False, "a Task the tool cannot serve never fails the session"
    assert result.details["accepted"] is False and result.details["rejected_by"] == ["no_reference"]
    assert result.details["produced"] == [], "a refusal wrote nothing"
    assert "has no Reference this session can read" in result.content
    assert '"the cancellation the request asked for never happened": 2' in result.content
    assert '"it reached no End state: it wrote nothing": 1' in result.content
    assert "refuse" in result.content and "file a finding" in result.content
    assert version_hash(plan.current(T)) == before, "the Verifier on disk did not move"
    assert len(_history(derived).versions) == 1, "the refused repair wrote no version"


def test_the_refusal_tells_a_task_that_never_had_a_reference_from_one_whose_reference_is_not_on_disk(derived):
    """The two states read differently, because what to do about them differs: one Task is waiting
    for a Run that finishes, the other for the Runs behind its row to be derived again."""
    plan, _ = _harness(derived)
    _unreadable_reference(derived, {})
    named = tools_mod.reference_state(plan, T)
    assert named["reference"] is False and named["references_named_not_on_disk"] == 1
    assert "none of its Runs is on disk in this session" in named["note"]
    references = _read(derived.workdir / "references.json")
    references[T] = {**references[T], "references": [], "reason": "no Run of it reached an End state"}
    (derived.workdir / "references.json").write_text(json.dumps(references), encoding="utf-8")
    never = tools_mod.reference_state(plan, T)
    assert never["reference"] is False and never["references_named_not_on_disk"] == 0
    assert "no Run of it reached an End state" in never["note"]
    assert tools_mod.reference_state(plan, "nobody")["reference"] is False, "an unknown Task is the same answer"


def test_reading_a_kind_outside_the_table_reads_the_references_file(derived):
    """A kind the table does not know is read off the References file, where the chain the table
    replaced sent it, rather than raised."""
    import asyncio

    plan, _ = _harness(derived)
    read = tools_mod._read(plan)

    async def text_of(kind, ident):
        args = tools_mod.ReadArgs.model_construct(kind=kind, id=ident)
        return (await read(args)).text

    whole = json.loads(asyncio.run(text_of("frobnicate", None)))
    assert set(whole) == {"ids", "note"} and "frobnicate" in whole["note"]
    one = json.loads(asyncio.run(text_of("frobnicate", T)))
    assert set(one) == {T}
    assert one[T] == json.loads(asyncio.run(text_of("references", T)))[T]


def test_the_read_table_names_every_read_kind_and_no_other():
    """A kind added to the schema without a handler (or a handler without a kind) must fail here,
    not in front of the model: the table keys and the ReadKind literal stay equal."""
    import typing

    assert set(tools_mod.READ_HANDLERS) == set(typing.get_args(tools_mod.ReadKind))
