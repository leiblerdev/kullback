"""world_tools runs the first pass without the scheduler; this checks parity, cache and anchor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tau2_retail_small.json"

STABLE_ARTIFACTS = [
    "schema.json",
    "tool_sigs.json",
    "unknown_tools.json",
    "row_homes.json",
    "world_constants.json",
    "readers.json",
    "tasks.json",
    "grouping.json",
    "tasks_frozen.json",
    "canon-rules.json",
    "db.json",
    "assumptions.json",
    "world_provenance.json",
    "anchor.json",
    "evidence_counts.json",
    "overlay_pins.json",
]


def _first_pass(workdir: Path):
    from kullback.builder import world_tools

    report = world_tools.ingest_files([FIXTURE], workdir)
    assert report.gate.get("pass")
    return world_tools.derive_world(workdir)


def test_derive_world_writes_what_the_pipeline_writes_byte_for_byte(tmp_path):
    """The world tools carry the stages' own run logic, so their artifacts match the scheduler's."""
    from kullback.builder import world_tools
    from kullback.builder.build import BuildPlan, execute

    ours = tmp_path / "ours"
    ours.mkdir()
    report = _first_pass(ours)
    assert not report.from_cache
    assert report.tables and report.tools and report.tasks
    assert set(report.gates) == {"mine", "readers", "cluster"}

    theirs = tmp_path / "theirs"
    shutil.copytree(ours, theirs)
    result = execute(BuildPlan(workdir=theirs), "starting_state")
    assert result.status == "complete"

    for name in STABLE_ARTIFACTS:
        assert (theirs / name).is_file(), name
        assert (ours / name).read_bytes() == (theirs / name).read_bytes(), name
    for path in sorted((ours / "tasks").glob("*.json")):
        other = theirs / "tasks" / path.name
        assert other.is_file(), path.name
        assert path.read_bytes() == other.read_bytes(), path.name
    for path in sorted((ours / "overlays").glob("*.json")):
        other = theirs / "overlays" / path.name
        assert other.is_file(), path.name
        assert path.read_bytes() == other.read_bytes(), path.name
    # The split records its own history: a fresh derive freezes three Tasks, the
    # resumed pipeline run finds them frozen. The grouping it was taken over is
    # what must agree, and it does.
    split = json.loads((theirs / "task_split.json").read_text(encoding="utf-8"))
    regrouped = [world_tools.Task.model_validate(json.loads(p.read_text(encoding="utf-8")))
                 for p in sorted((theirs / "tasks").glob("*.json"))]
    assert split["grouping"] == world_tools.cluster.grouping_fingerprint(regrouped)
    assert split["frozen"] == 3 and split["added"] == []


def test_derive_world_serves_an_unchanged_world_from_the_cache(tmp_path):
    from kullback.builder import world_tools

    workdir = tmp_path / "work"
    workdir.mkdir()
    first = _first_pass(workdir)
    assert not first.from_cache
    stamped = {name: (workdir / name).stat().st_mtime for name in STABLE_ARTIFACTS
               if (workdir / name).is_file()}

    second = world_tools.derive_world(workdir)

    assert second.from_cache
    assert second.tables == first.tables
    assert second.tools == first.tools
    assert second.tasks == first.tasks
    for name, mtime in stamped.items():
        assert (workdir / name).stat().st_mtime == mtime, name


def test_the_held_out_anchor_is_absent_from_the_seed(tmp_path):
    """D81: every Task's anchor Runs are stored, counted apart, and never seed Runs."""
    from kullback.builder import world_tools

    workdir = tmp_path / "work"
    workdir.mkdir()
    report = _first_pass(workdir)
    anchor = json.loads((workdir / "anchor.json").read_text(encoding="utf-8"))
    counts = json.loads((workdir / "evidence_counts.json").read_text(encoding="utf-8"))
    assert set(anchor["held_out"]) == set(report.tasks)
    for task_id, runs in report.tasks.items():
        held = anchor["held_out"][task_id]
        assert world_tools.seed_runs(anchor, task_id, [f"run-{i}" for i in range(runs)]) == [
            f"run-{i}" for i in range(runs) if f"run-{i}" not in held]
        assert counts[task_id] == {"evidence_traces": runs - len(held), "anchor_traces": len(held)}
    assert (workdir / "world_provenance.json").is_file()

    room = tmp_path / "room"
    room.mkdir()
    tasks = [{"id": "task-big", "run_ids": [f"run-{i}" for i in range(5)]},
             {"id": "task-small", "run_ids": ["only"]}]
    drawn = world_tools.draw_anchor(tasks, room)
    assert len(drawn["held_out"]["task-big"]) == 1
    assert drawn["held_out"]["task-small"] == []
    assert drawn["unguarded"] == ["task-small"]
    seed = world_tools.seed_runs(drawn, "task-big", [f"run-{i}" for i in range(5)])
    assert len(seed) == 4
    assert not set(seed) & set(drawn["held_out"]["task-big"])
