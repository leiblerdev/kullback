"""The verifiers binding's loaders read the files the Examiner saved, at the paths records names (F22)."""

from __future__ import annotations

from examiner.worlds import make_world
from gates import verifier_fixtures as VF
from kullback.examiner import exam_files as F
from kullback.gates.bindings import binding_for
from kullback.runner.records import (
    Probe,
    ProbePool,
    as_dict,
    exam_probe_path,
    exam_task_runs_path,
    probe_pool_path,
    write_json,
)


def _reads():
    return binding_for("verifiers/t1.json").reads


def _verifier(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return VF.derive(tmp_path)


def test_the_history_and_task_runs_loaders_read_what_the_examiner_saved(tmp_path):
    world = make_world(tmp_path)
    verifier = _verifier(tmp_path / "v")
    root = F.ExamRoot(workdir=world.workdir, replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                      history={"t1": F.seeded_history({}, "t1", verifier)})
    F.save_ruling_files(root)
    reads = _reads()
    history = reads["history"](world.workdir, "verifiers/t1.json")
    task_runs = reads["task_runs"](world.workdir, "verifiers/t1.json")
    assert F.history_path(root) == world.workdir / "exam" / "history.json"
    assert history["t1"].versions == root.history["t1"].versions
    assert exam_task_runs_path(world.workdir).is_file()
    assert [run.run_id for run in task_runs["t1"]] == [
        run.run_id for run in F.evidence_of(root)["task_runs"]["t1"]]


def test_the_probes_loader_reads_saved_pools_and_the_sessions_probe_runs(tmp_path):
    run = VF.reference_run()
    saved = ProbePool(task_id="t1", probes=[Probe(probe_id="p-saved", task_id="t1", bug_class="skip",
                                                  verifier_hash="h", run=run)])
    write_json(probe_pool_path(tmp_path, "t1"), as_dict(saved))
    write_json(exam_probe_path(tmp_path, "t1", 1), as_dict(run.model_copy(update={"run_id": "p-session"})))
    pools = _reads()["probes"](tmp_path, "verifiers/t1.json")
    assert [probe.probe_id for probe in pools["t1"].probes] == ["p-saved", "p-session"]


def test_the_loaders_read_nothing_as_missing_evidence(tmp_path):
    reads = _reads()
    assert [reads[name](tmp_path, "verifiers/t1.json") for name in ("history", "task_runs", "probes")] == [None] * 3
