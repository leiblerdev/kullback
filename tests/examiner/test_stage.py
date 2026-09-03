"""The derivation as the Examiner runs it: derive_all over the Builder's inputs writes what the pipeline's
derive_verifier stage wrote, one Task at a time when asked, never over a store that names a body."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from examiner.worlds import anchor_of, make_world, probe_runner_over
from gates import verifier_fixtures as VF
from kullback.builder.pipeline import Anchor
from kullback.examiner import stage
from kullback.gates.artifacts import D79_CHECKS, D79_STAGES
from kullback.gates.ledger import GateLedger
from kullback.runner.records import Task, Verifier

STATUS_KEYS = {"reference_confirmed", "verifier_passed", "reason", "recordings", "rerolls", "judged", "assisted_tools"}
REFERENCE_KEYS = {"references", "recordings", "failed", "groups", "reason", "judged", "judge_reason"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _derive(workdir: Path, inputs: dict, **kwargs) -> dict:
    ctx = stage.ExamContext(workdir, GateLedger(workdir), anchor=kwargs.pop("anchor", None))
    return stage.derive_all(ctx, inputs, **kwargs)


def test_derive_all_over_the_builder_store_writes_the_task_status_and_references_the_pipeline_stage_wrote(
        fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    out = _derive(workdir, inputs, anchor=anchor_of(workdir))
    status = _read(workdir / "task_status.json")
    references = _read(workdir / "references.json")
    task_ids = {task.id for task in inputs["tasks"]}
    assert set(status) == task_ids == set(references) and out["task_status"] == status
    # No Task of the small fixture has a Reference: the rows are the stage's no-Reference rows, each
    # with the reason the setup review reads, and the same rows land in the returned status.
    assert all(set(row) == STATUS_KEYS and row["reference_confirmed"] is False for row in status.values())
    assert all(set(row) == REFERENCE_KEYS and row["references"] == [] for row in references.values())
    assert not (workdir / "verifiers").exists() and out["verifiers"] == []
    stages = [row["stage"] for row in _read(workdir / "gates.json")]
    assert stages[-2:] == ["compile_policy", "derive_verifier"]
    assert (workdir / "constraints_check.json").is_file()


def test_derive_for_one_task_leaves_the_other_tasks_rows_untouched(fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    _derive(workdir, inputs)
    first, *others = [task.id for task in inputs["tasks"]]
    status = _read(workdir / "task_status.json")
    for task_id in others:
        status[task_id]["marker"] = "left alone"
    (workdir / "task_status.json").write_text(json.dumps(status), encoding="utf-8")
    out = _derive(workdir, inputs, only=first)
    after = _read(workdir / "task_status.json")
    assert out["task_status"] == after and set(after) == {first, *others}
    assert "marker" not in after[first] and set(after[first]) == STATUS_KEYS
    assert all(after[task_id]["marker"] == "left alone" for task_id in others)
    assert set(_read(workdir / "references.json")) == set(after)
    with pytest.raises(ValueError, match="no Task is named"):
        _derive(workdir, inputs, only="no-such-task")


def test_a_task_without_a_reference_gets_no_verifier_and_its_reason_names_the_assisted_tool(fixture_build, tmp_path):
    workdir = fixture_build.copy(tmp_path)
    inputs = fixture_build.inputs_for(workdir)
    assisted = set(inputs["assisted_tools"])
    assert assisted, "the fixture build has at least one assisted tool"
    status = _derive(workdir, inputs)["task_status"]
    named = {task_id: row for task_id, row in status.items() if row["assisted_tools"]}
    assert named, "one Task's seed Trace calls an assisted tool"
    for row in named.values():
        assert row["reference_confirmed"] is False and row["verifier_passed"] is False
        assert set(row["assisted_tools"]) <= assisted
        assert "assisted tool" in row["reason"] and all(tool in row["reason"] for tool in row["assisted_tools"])
    assert not (workdir / "verifiers").exists()


def test_the_suite_for_a_repaired_verifier_runs_every_d79_check_the_derivation_ran(derived):
    verifier = Verifier.model_validate(_read(derived.workdir / "verifiers" / "t1.json"))
    status = _read(derived.workdir / "task_status.json")["t1"]
    task = derived.inputs["tasks"][0]
    gates = stage.suite_for(task, verifier, [derived.paths["ref"], derived.paths["alt"]], canon_rules=None,
                            write_tools=VF.WRITE_TOOLS, user_rules={}, rules_trace="ref", probe_model=object(),
                            run_probe=probe_runner_over(), may_probe=True)
    assert [g.stage for g in gates] == list(D79_STAGES)
    assert set(status["checks"]) == set(D79_CHECKS) and all(status["checks"].values())
    assert all(g.passed for g in gates)


def test_the_context_seeds_from_the_anchor_when_given_and_from_every_run_without_one(fixture_build, tmp_path):
    # The fixture is too small to hold a Run out of any Task, so its anchor seeds from every Run;
    # a build large enough holds a share out, and the context seeds from the rest (D81).
    task = fixture_build.inputs["tasks"][0]
    assert set(task.run_ids) and not anchor_of(fixture_build.workdir).held_out.get(task.id)
    plain = stage.ExamContext(tmp_path, GateLedger(tmp_path))
    assert plain.seed_runs(task.id, task.run_ids) == list(task.run_ids)
    empty = stage.ExamContext(tmp_path, GateLedger(tmp_path), anchor=anchor_of(fixture_build.workdir))
    assert empty.seed_runs(task.id, task.run_ids) == list(task.run_ids)
    held = Task(id="t", run_ids=["a", "b", "c"])
    anchored = stage.ExamContext(tmp_path, GateLedger(tmp_path),
                                 anchor=Anchor(held_out={"t": ["b"]}, unguarded=[]))
    assert anchored.seed_runs(held.id, held.run_ids) == ["a", "c"]
    assert stage.seed_ids(anchored, held) == {"a", "c"} and stage.seed_ids(plain, held) == {"a", "b", "c"}


def test_the_loophole_probe_runs_through_the_runner_callable_given_and_the_inputs_never_hold_bodies(tmp_path):
    world = make_world(tmp_path)
    seen = []

    def run_probe(model, verifier):
        seen.append((model, verifier))
        return VF.wrong_run()

    marker = object()
    out = _derive(world.workdir, world.inputs, probe_model=marker, run_probe=run_probe)
    assert len(seen) == 1 and seen[0][0] is marker and seen[0][1].task_id == "t1"
    assert out["task_status"]["t1"]["checks"]["loophole_probe_fails"] is True
    assert not set(stage.FORBIDDEN_INPUTS) & set(world.inputs)
    # Without a model the callable is never run and the check is listed as not run, not as passed.
    quiet = make_world(tmp_path / "quiet")
    out = _derive(quiet.workdir, quiet.inputs, run_probe=run_probe)
    assert len(seen) == 1 and "verifier_loophole" in out["task_status"]["t1"]["not_run"]
    assert out["task_status"]["t1"]["verifier_passed"] is False


@pytest.mark.parametrize("name", stage.FORBIDDEN_INPUTS[:4])
def test_inputs_from_refuses_a_store_that_names_bodies_db_schema_or_the_environment(name):
    store = {"tasks": [Task(id="t")], "sigs": [], "constraints": [], name: {"anything": 1}}
    with pytest.raises(ValueError, match=name):
        stage.inputs_from(store)
    del store[name]
    store["policy_text"] = "the policy"
    with pytest.raises(ValueError, match="policy_text"):
        stage.inputs_from(store)
    del store["policy_text"]
    assert set(stage.inputs_from(dict(store, extra="ignored"))) == {"tasks", "sigs", "constraints"}


def test_derive_all_rewrites_the_scorecard_after_the_task_status(tmp_path):
    world = make_world(tmp_path)
    (world.workdir / "tasks.json").write_text(json.dumps({"tasks": [{"id": "t1", "run_ids": ["ref"]}]}),
                                             encoding="utf-8")
    (world.workdir / "runs.json").write_text(json.dumps({"runs": [{"run_id": "ref", "task_id": "t1"}]}),
                                            encoding="utf-8")
    (world.workdir / "scorecard.json").write_text("{}", encoding="utf-8")
    _derive(world.workdir, world.inputs, probe_model=object(), run_probe=probe_runner_over())
    status_at = os.stat(world.workdir / "task_status.json").st_mtime_ns
    card_at = os.stat(world.workdir / "scorecard.json").st_mtime_ns
    assert card_at >= status_at
    coverage = _read(world.workdir / "scorecard.json")["task_coverage"]
    assert coverage["tasks_covered"] == 1 and coverage["uncovered"] == [], \
        "the card counts the Task the status just verdicted"
