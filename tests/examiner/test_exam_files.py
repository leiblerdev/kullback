"""The Examiner's root holds what the session may read and nothing it may not (D123)."""

from __future__ import annotations

from examiner.worlds import make_world
from kullback.examiner import exam_files as F
from kullback.runner.records import as_dict, write_json


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
