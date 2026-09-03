"""The seven Examiner tools, driven through the harness's registry and hooks with no model turn: what each
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
from kullback.runner.records import Verifier, VerifierHistory, as_dict, content_hash

TOOL_NAMES = ["read", "derive", "probe", "repair", "refuse", "reroll", "finding"]


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
    assert result.is_error and "no current Verifier" in result.content and T in result.content
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
    assert set(status) == {T}
    missing = drive(harness, "read", {"kind": "run", "id": "nowhere"})
    assert missing.is_error and "nowhere" in missing.content
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
