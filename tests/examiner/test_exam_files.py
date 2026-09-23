"""The Examiner's root holds what the session may read and nothing it may not (D123)."""

from __future__ import annotations

from examiner.worlds import make_world
from gates import verifier_fixtures as VF
from kullback.examiner import exam_files as F
from kullback.gates.probes import version_hash
from kullback.runner.records import ProbePool, as_dict, read_json, write_json


def test_expose_copies_runs_json_records_and_spoken(tmp_path):
    world = make_world(tmp_path)
    workdir = world.workdir
    write_json(workdir / "replays.json", world.inputs["replays"])
    write_json(workdir / "rerolls.json", world.inputs["rerolls"])
    write_json(workdir / "references.json", {"t1": {"references": [{"run_id": "ref"}]}})
    write_json(workdir / "task_status.json", {"t1": {"reference_confirmed": True}})
    (workdir / "tasks").mkdir()
    write_json(workdir / "tasks" / "t1.json", as_dict(world.inputs["tasks"][0]))
    out = F.expose(workdir)
    assert (workdir / "exam" / "runs" / "t1" / "ref.jsonl").is_file()
    assert (workdir / "exam" / "replays.json").is_file()
    assert (workdir / "exam" / "tasks" / "t1.json").is_file()
    assert (workdir / "exam" / "spoken" / "t1.txt").is_file()
    assert (workdir / "exam" / "spoken" / "t1.txt").read_text().strip() != ""
    assert any(name.startswith("runs/") for name in out["copied"])
    assert out["spoken"] == ["spoken/t1.txt"]


def test_expose_never_copies_a_forbidden_file(tmp_path):
    world = make_world(tmp_path)
    workdir = world.workdir
    write_json(workdir / "bodies.json", {"t1": "body"})
    write_json(workdir / "db.json", {})
    write_json(workdir / "schema.json", {})
    (workdir / "env").mkdir()
    (workdir / "env" / "tools.py").write_text("x = 1")
    (workdir / "overlays").mkdir()
    (workdir / "overlays" / "o.json").write_text("{}")
    F.expose(workdir)
    exam = workdir / "exam"
    assert not (exam / "bodies.json").exists()
    assert not (exam / "db.json").exists()
    assert not (exam / "schema.json").exists()
    assert not (exam / "env").exists()
    assert not (exam / "overlays").exists()


def test_names_forbidden_path_flags_the_compiled_side():
    assert F.names_forbidden_path({"path": "env/tools/x.py"}) is not None
    assert F.names_forbidden_path({"path": "bodies.json"}) is not None
    assert F.names_forbidden_path({"path": "exam/verifiers/t1.json"}) is None
    assert F.names_forbidden_path({"path": "exam/runs/t1/ref.jsonl"}) is None


def test_finding_carries_rows_path_change_and_source():
    finding = F.Finding(task_id="t1", kind="fidelity", text="a body replays differently",
                        rows=[{"run_id": "ref", "recorded": 1, "ours": 2}],
                        path="env/tools/x.py", change="the body answers the recorded call",
                        source="derive")
    body = finding.as_dict()
    assert body["rows"] == [{"run_id": "ref", "recorded": 1, "ours": 2}]
    assert body["path"] == "env/tools/x.py"
    assert body["change"] == "the body answers the recorded call"
    assert body["source"] == "derive"
    assert body["finding_id"] == finding.as_dict()["finding_id"]


def _verifier(tmp_path):
    (tmp_path / "v").mkdir(exist_ok=True)
    return VF.derive(tmp_path / "v")


def test_seeded_history_starts_a_task_at_version_1_accepted(tmp_path):
    verifier = _verifier(tmp_path)
    hist = F.seeded_history({}, "t1", verifier)
    [row] = hist.versions
    assert (row.verifier_version, row.accepted, row.by) == ("1", True, "derive")
    assert row.reason == "the Verifier on disk before any repair"
    assert row.content_hash == version_hash(verifier)
    assert F.seeded_history({"t1": hist}, "t1", verifier).versions == hist.versions


def test_history_round_trips_per_task_through_history_json(tmp_path):
    verifier = _verifier(tmp_path)
    root = F.ExamRoot(workdir=tmp_path)
    history = {"t1": F.seeded_history({}, "t1", verifier)}
    F.save_history(root, history)
    assert F.history_path(root) == tmp_path / "exam" / "history.json"
    loaded = F.load_history(root)
    assert loaded["t1"].versions == history["t1"].versions
    assert read_json(tmp_path / "exam" / "history.json")["t1"]["versions"][0]["accepted"] is True
    assert F.load_history(F.ExamRoot(workdir=tmp_path / "missing")) == {}


def test_evidence_of_carries_the_binding_reads_and_task_status(tmp_path):
    world = make_world(tmp_path)
    verifier = _verifier(tmp_path)
    root = F.ExamRoot(workdir=world.workdir, verifiers={"t1": verifier},
                      replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                      task_status={"t1": {"verifier_passed": True}},
                      probes={"t1": [VF.reference_run()]},
                      history={"t1": F.seeded_history({}, "t1", verifier)})
    evidence = F.evidence_of(root)
    assert {"replays", "rerolls", "history", "task_runs", "verifiers", "probes", "sigs",
            "rules"} <= set(evidence)
    assert evidence["task_status"] == {"t1": {"verifier_passed": True}}
    assert evidence["task_runs"]["t1"], "the loosening Runs load off the replay rows"
    assert isinstance(evidence["probes"]["t1"], ProbePool)


def test_evidence_of_a_task_hands_the_gates_that_task_only(tmp_path):
    """F47: with a task_id the gates see one Task's Verifier, probes, Runs, rows and history, so
    another Task's failure cannot refuse the write; without one (the status pass) they see all."""
    world = make_world(tmp_path)
    verifier = _verifier(tmp_path)
    other = verifier.model_copy(update={"task_id": "t2"})
    root = F.ExamRoot(workdir=world.workdir, verifiers={"t1": verifier, "t2": other},
                      replays=world.inputs["replays"], rerolls=world.inputs["rerolls"],
                      task_status={"t1": {"verifier_passed": True}, "t2": {"verifier_passed": False}},
                      probes={"t2": [VF.reference_run()]},
                      history={"t1": F.seeded_history({}, "t1", verifier), "t2": F.seeded_history({}, "t2", other)})
    own = F.evidence_of(root, "t1")
    assert own["verifiers"] == [verifier]
    assert own["probes"] == {}, "an empty mapping, so no loader fills in the other Tasks' pools"
    assert set(own["history"]) == {"t1"} and set(own["task_status"]) == {"t1"}
    assert set(own["task_runs"]) <= {"t1"} and set(own["replays"]) <= {"t1"} and set(own["rerolls"]) <= {"t1"}
    every = F.evidence_of(root)
    assert {v.task_id for v in every["verifiers"]} == {"t1", "t2"}
    assert set(every["probes"]) == {"t2"} and set(every["history"]) == {"t1", "t2"}


def test_evidence_of_a_referenced_task_hands_the_probe_model_and_runner_to_the_suite(tmp_path):
    world = make_world(tmp_path)
    refs = [{"run_id": row["run_id"]} for row in world.inputs["replays"]["t1"].values()]
    write_json(world.workdir / "references.json", {"t1": {"references": refs}})
    model, runner = object(), lambda model, verifier: VF.wrong_run()
    keyword = dict(workdir=world.workdir, verifiers={"t1": _verifier(tmp_path)},
                   replays=world.inputs["replays"], rerolls=world.inputs["rerolls"])
    evidence = F.evidence_of(F.ExamRoot(probe_model=model, run_probe=runner, **keyword), "t1")
    assert evidence["reference"] is not None
    assert evidence["probe_model"] is model and evidence["run_probe"] is runner


def test_expose_writes_the_derived_verifier_under_derived_and_leaves_verifiers_alone(tmp_path):
    world = make_world(tmp_path)
    workdir = world.workdir
    (tmp_path / "v").mkdir()
    verifier = VF.derive(tmp_path / "v")
    write_json(workdir / "verifiers" / "t1.json", as_dict(verifier))
    out = F.expose(workdir)
    assert read_json(workdir / "exam" / "derived" / "t1.json") == as_dict(verifier)
    assert "derived/t1.json" in out["copied"]
    assert not (workdir / "exam" / "verifiers").exists(), "the proposal path stays the Examiner's own"
