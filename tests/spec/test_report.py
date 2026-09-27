import json

from kullback.spec import report as R
from kullback.spec.trust import TaskTier


def _tiers(**tiers):
    return {task_id: TaskTier(tier, {"task_id": task_id, "tier": tier}) for task_id, tier in tiers.items()}


def test_the_report_counts_every_tier_and_splits_the_trusted_tasks_by_reference_class():
    tiers = _tiers(a="trusted", b="trusted", c="replay_only", d="untrusted", e="set_aside", f="trusted")
    report = R.build_report(tiers, verifier_from="intent", corpus="c1", build="b1",
                            classes={"a": "right", "b": "wrong"})
    assert report["tiers"] == {"trusted": 3, "replay_only": 1, "untrusted": 1, "set_aside": 1}
    assert report["trusted_by_reference"] == {"right": 1, "wrong": 1, "unknown": 1}
    assert [row["task_id"] for row in report["rows"]] == ["a", "b", "c", "d", "e", "f"]
    assert R.table_line(report) == "| b1 | c1 | intent | 6 | 3 | 1 | 1 | 1 | 1, 1, 0, 1 |"
    assert R.table_line(report).count("|") == R.TABLE_HEADER.count("|")


def test_without_a_sidecar_the_split_is_absent_rather_than_zero(tmp_path):
    report = R.build_report(_tiers(a="trusted"), verifier_from="runs")
    assert report["trusted_by_reference"] is None and R.table_line(report).endswith("| n/a |")
    assert R.reference_classes(tmp_path) is None


def test_the_class_of_a_task_is_trust_splits_class_of_its_references_rewards(tmp_path):
    grader = tmp_path / "grader"
    grader.mkdir()
    for trace, reward in (("r1", 1.0), ("r2", 0.0), ("r3", 1.0)):
        (grader / f"{trace}.json").write_text(json.dumps({"trace_id": trace, "fields": {"reward": reward}}))
    (tmp_path / "references.json").write_text(json.dumps({
        "t1": {"references": [{"trace_id": "r1"}, {"trace_id": "r3"}]},
        "t2": {"references": [{"trace_id": "r2"}]},
        "t3": {"references": [{"trace_id": "r1"}, {"trace_id": "r2"}]},
        "t4": {"references": [{"run_id": "reroll-1"}]}}))
    assert R.reference_classes(tmp_path) == {"t1": "right", "t2": "wrong", "t3": "mixed", "t4": "unknown"}


def test_the_report_is_written_once_to_the_workdir_under_the_stored_foundation(tmp_path):
    report = R.write_report(tmp_path, corpus="c1")
    assert report["verifier_from"] == "runs" and report["tasks"] == 0
    assert json.loads((tmp_path / R.REPORT_FILE).read_text()) == report


def test_the_builds_readme_carries_the_tier_table_header_the_report_writes_lines_for():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[2] / "docs" / "builds" / "README.md"
    assert R.TABLE_HEADER in readme.read_text(encoding="utf-8").splitlines()
