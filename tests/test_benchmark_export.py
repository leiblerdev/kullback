"""The travel-agency benchmark exports to Harbor tasks from one environment and its task files."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "travel-agency"
HARBOR_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/runtime.py",
    "environment/world/db.json",
    "environment/world/grade.json",
    "environment/world/overlay.json",
    "environment/world/secret",
    "environment/world/toolkit.src",
    "solution/solve.sh",
    "solution/walk.json",
    "tests/grade.json",
    "tests/runtime.py",
    "tests/test.sh",
)


def run_export(out: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(BENCHMARK / "export.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    out = tmp_path_factory.mktemp("harbor")
    return out, run_export(out).stdout.strip()


def test_export_writes_the_recorded_bytes(export):
    _, digest = export
    assert digest == (BENCHMARK / "harbor.sha256").read_text().strip()


def test_every_task_file_becomes_a_complete_harbor_task(export):
    out, _ = export
    tasks = sorted(path.stem for path in (BENCHMARK / "tasks").glob("*.json"))
    assert len(tasks) == 65
    missing = [f"{task}/{name}" for task in tasks for name in HARBOR_FILES if not (out / task / name).is_file()]
    assert missing == []


def test_exported_grade_holds_the_shared_guards_and_the_tasks_own_end_state(export):
    out, _ = export
    source = next((BENCHMARK / "tasks").glob("*.json"))
    task = json.loads(source.read_text(encoding="utf-8"))
    grade = json.loads((out / source.stem / "tests" / "grade.json").read_text(encoding="utf-8"))
    assert grade["expected_diff"] == task["grade"]["expected_diff"]
    assert len(grade["code_guards"]) == 104


def test_the_served_grade_file_equals_the_verifiers(export):
    out, _ = export
    tasks = sorted(path.stem for path in (BENCHMARK / "tasks").glob("*.json"))
    differ = [
        task
        for task in tasks
        if (out / task / "environment/world/grade.json").read_bytes() != (out / task / "tests/grade.json").read_bytes()
    ]
    assert differ == []


def test_exporting_over_an_earlier_export_removes_files_it_no_longer_writes(export, tmp_path):
    _, fresh = export
    out = tmp_path / "harbor"
    run_export(out)
    (out / "RESULTS.md").write_text("stale\n", encoding="utf-8")
    again = run_export(out).stdout.strip()
    assert not (out / "RESULTS.md").exists()
    assert again == fresh


def test_export_refuses_to_clear_a_folder_that_holds_no_earlier_export(tmp_path):
    (tmp_path / "keep.txt").write_text("mine\n", encoding="utf-8")
    done = run_export(tmp_path, check=False)
    assert done.returncode != 0
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "mine\n"
