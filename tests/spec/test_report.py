import json

from kullback.spec import report as R
from kullback.spec.trust import TaskTier


def _tiers(code="c0de" * 16, **tiers):
    """task id -> tier, or (tier, extra row fields)."""
    out = {}
    for task_id, value in tiers.items():
        tier, extra = value if isinstance(value, tuple) else (value, {})
        out[task_id] = TaskTier(tier, {"task_id": task_id, "tier": tier, "code_hash": code, **extra})
    return out


def test_the_report_counts_tiers_reasons_and_flags_and_splits_the_trusted_tasks_by_reference_class():
    tiers = _tiers(a=("trusted", {"reference_passes": True, "solvable": True}),
                   b=("trusted", {"reference_passes": False, "solvable": None}),
                   c=("untrusted", {"reason": "open_ruling"}), d=("untrusted", {"reason": "no_intent"}),
                   e=("untrusted", {"reason": "constructed_run_passed", "solvable": False}), f="trusted")
    report = R.build_report(tiers, corpus="c1", build="b1",
                            classes={"a": "right", "b": "wrong"})
    assert report["tiers"] == {"trusted": 3, "untrusted": 3}
    assert report["reasons"] == {"no_intent": 1, "open_ruling": 1, "constructed_run_passed": 1}
    assert report["flags"] == {"reference_passes": 1, "reference_passes_false": 1, "solvable": 1, "solvable_false": 1}
    assert report["trusted_by_reference"] == {"right": 1, "wrong": 1, "unknown": 1}
    assert [row["task_id"] for row in report["rows"]] == ["a", "b", "c", "d", "e", "f"]
    assert R.table_line(report) == "| b1 | c1 | spec | 6 | 3 | 3 | 1 | 1 | 1 | 1 of 2 | 1 of 2 | 1, 1, 0, 1 | c0dec0dec0de |"
    assert R.table_line(report).count("|") == R.TABLE_HEADER.count("|")


def test_without_a_sidecar_the_split_is_absent_rather_than_zero(tmp_path):
    report = R.build_report(_tiers(a="trusted"))
    assert report["trusted_by_reference"] is None and R.table_line(report).endswith("| n/a | c0dec0dec0de |")
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


def test_the_report_is_written_once_to_the_workdir(tmp_path):
    report = R.write_report(tmp_path, corpus="c1")
    assert report["tasks"] == 0
    assert json.loads((tmp_path / R.REPORT_FILE).read_text()) == report


def test_the_builds_readme_carries_the_tier_table_header_the_report_writes_lines_for():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[2] / "docs" / "builds" / "README.md"
    assert R.TABLE_HEADER in readme.read_text(encoding="utf-8").splitlines()


def test_the_report_heads_with_the_one_code_hash_and_warns_when_its_rows_carry_two():
    one = R.build_report(_tiers(a="trusted", b="untrusted"))
    assert one["code_hash"] == "c0de" * 16 and one["warnings"] == []
    assert R.header_line(one) == "tier report, verifier from the Spec, code c0dec0dec0de"
    rows = {**_tiers(a="trusted"), **_tiers(code="beef" * 16, b="untrusted")}
    mixed = R.build_report(rows)
    assert mixed["code_hash"] is None and R.header_line(mixed).endswith("code mixed")
    assert mixed["warnings"] and "2 code hashes" in mixed["warnings"][0]
    assert R.table_line(mixed).endswith("| mixed |")


def test_every_row_carries_the_live_code_hash_and_a_report_over_other_code_is_flagged(tmp_path, monkeypatch):
    from kullback.runner.code_hash import CODE_HASH
    from kullback.spec import trust as T

    tiers = {"a": T.missing_spec_row("a")}
    monkeypatch.setattr(R, "workdir_tiers", lambda workdir: tiers)
    first = R.write_report(tmp_path)
    assert [row["code_hash"] for row in first["rows"]] == [CODE_HASH]
    assert first["code_hash"] == CODE_HASH and first["warnings"] == []
    stale = dict(first, code_hash="beef" * 16, rows=[dict(first["rows"][0], code_hash="beef" * 16)])
    (tmp_path / R.REPORT_FILE).write_text(json.dumps(stale))
    second = R.write_report(tmp_path)
    assert second["warnings"] and "beefbeefbeef" in second["warnings"][0]
    assert json.loads((tmp_path / R.REPORT_FILE).read_text()) == second
