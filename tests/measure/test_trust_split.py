"""`scripts/measure/trust_split.py`: the trusted-Task Reference split, on a tiny
fixture workdir built here. No model, no corpus, no mocks of the script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "measure"))

import trust_split as T  # noqa: E402


def _write(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _sidecar(trace_id: str, reward: float) -> dict:
    return {"trace_id": trace_id, "fields": {"reward_info": {"reward": reward}}}


def _ref(run_id: str, trace_id: str | None) -> dict:
    return {"run_id": run_id, "trace_id": trace_id}


def fixture(root: Path, *, rounds: bool = True, sidecars: bool = True) -> Path:
    wd = root / "wd"
    status = {
        "t_right": {"reference_run_ids": ["run-r1"]},
        "t_wrong": {"reference_run_ids": ["run-w1"]},
        "t_mixed": {"reference_run_ids": ["run-m1", "run-m2"]},
        "t_partial": {"reference_run_ids": ["run-p1", "run-u3"]},
        "t_no_trace": {"reference_run_ids": ["run-u1"]},
        "t_no_sidecar": {"reference_run_ids": ["run-u2"]},
        "t_extra": {"reference_run_ids": ["run-e1"]},
    }
    refs = {
        "t_right": {"references": [_ref("run-r1", "trace-r1")],
                    "recordings": [_ref("run-r1", "trace-r1")]},
        "t_wrong": {"references": [_ref("run-w1", "trace-w1")],
                    "recordings": [_ref("run-w1", "trace-w1")]},
        "t_mixed": {"references": [_ref("run-m1", "trace-m1"),
                                   _ref("run-m2", "trace-m2")],
                    "recordings": [_ref("run-m1", "trace-m1"),
                                   _ref("run-m2", "trace-m2")]},
        "t_partial": {"references": [_ref("run-p1", "trace-p1"), _ref("run-u3", None)],
                      "recordings": [_ref("run-p1", "trace-p1"), _ref("run-u3", None)]},
        "t_no_trace": {"references": [_ref("run-u1", None)],
                       "recordings": [_ref("run-u1", None)]},
        "t_no_sidecar": {"references": [_ref("run-u2", "trace-u9")],
                         "recordings": [_ref("run-u2", "trace-u9")]},
        "t_extra": {"references": [_ref("run-e1", "trace-e1")],
                    "recordings": [_ref("run-e1", "trace-e1")]},
    }
    # Round 1 trusts the six; round 2 additionally trusts t_extra and refuses t_out.
    trusted_1 = ["t_right", "t_wrong", "t_mixed", "t_partial", "t_no_trace", "t_no_sidecar"]
    if rounds:
        for name, trusted in (("1", trusted_1), ("2", [*trusted_1, "t_extra"])):
            rows = [{"task_id": tid, "trusted": tid in trusted, "refused": False}
                    for tid in [*status, "t_out"]]
            rows.append({"task_id": "t_out", "trusted": False, "refused": name == "2"})
            _write(wd / "rounds" / name / "tasks.json",
                   {"counts": {}, "rows": rows})
    _write(wd / "task_status.json", status)
    _write(wd / "references.json", refs)
    if sidecars:
        for trace_id, reward in (("trace-r1", 1.0), ("trace-w1", 0.0), ("trace-m1", 1.0),
                                 ("trace-m2", 0.0), ("trace-p1", 1.0), ("trace-e1", 1.0)):
            _write(wd / "grader" / f"{trace_id}.json", _sidecar(trace_id, reward))
    return wd


ALL_SEVEN = ["t_extra", "t_mixed", "t_no_sidecar", "t_no_trace", "t_partial", "t_right", "t_wrong"]


# --- the sidecar split ------------------------------------------------------


def test_every_reference_passing_is_right_and_every_failing_is_wrong(tmp_path):
    result = T.measure(fixture(tmp_path))
    assert result["classes"]["right"] == ["t_extra", "t_right"]
    assert result["classes"]["wrong"] == ["t_wrong"]
    assert result["wrong_share_scored"] == 1 / 4
    assert result["wrong_share_trusted"] == 1 / 7


def test_a_task_split_across_pass_and_fail_is_mixed(tmp_path):
    result = T.measure(fixture(tmp_path))
    assert result["classes"]["mixed"] == ["t_mixed"]
    assert sorted(result["classes"]["unknown"]) == ["t_no_sidecar", "t_no_trace", "t_partial"]


def test_a_task_with_an_unscored_reference_is_unknown_even_when_the_rest_pass(tmp_path):
    by_id = {r["task_id"]: r for r in T.measure(fixture(tmp_path))["rows"]}
    assert by_id["t_partial"]["unscored_references"] == 1
    assert by_id["t_partial"]["class"] == "unknown"
    assert by_id["t_no_trace"]["unscored_references"] == 1
    assert by_id["t_no_sidecar"]["unscored_references"] == 1


def test_the_text_table_names_the_wrong_share_and_every_task_id(tmp_path):
    text = T.report(T.measure(fixture(tmp_path)))
    assert "25.0% (1/4) of scored" in text
    for tid in ALL_SEVEN:
        assert tid in text


# --- latest round and refusals ----------------------------------------------


def test_only_the_latest_round_is_measured(tmp_path):
    result = T.measure(fixture(tmp_path))
    assert result["round"] == "2"
    assert "t_extra" in result["trusted"]


def test_refusals_come_off_the_round_file(tmp_path):
    assert T.measure(fixture(tmp_path))["refused"] == ["t_out"]


def test_a_workdir_without_a_round_file_is_an_error(tmp_path, capsys):
    assert T.main([str(fixture(tmp_path, rounds=False))]) == 2
    assert "no rounds" in capsys.readouterr().out


# --- corrupt snapshots and confined output ----------------------------------


def test_a_corrupt_round_file_is_an_error(tmp_path, capsys):
    wd = fixture(tmp_path)
    (wd / "rounds" / "2" / "tasks.json").write_text('{"rows": [{"task_id":', encoding="utf-8")
    assert T.main([str(wd)]) == 2
    assert "unreadable round file" in capsys.readouterr().out


def test_a_round_file_with_the_wrong_shape_is_an_error(tmp_path, capsys):
    wd = fixture(tmp_path)
    (wd / "rounds" / "2" / "tasks.json").write_text("[1, 2]", encoding="utf-8")
    assert T.main([str(wd)]) == 2
    assert "malformed round file" in capsys.readouterr().out


def test_a_json_output_inside_the_workdir_is_refused(tmp_path, capsys):
    wd = fixture(tmp_path)
    out = wd / "out.json"
    assert T.main([str(wd), "--json", str(out)]) == 2
    assert "inside the workdir" in capsys.readouterr().out
    assert not out.exists()


# --- no sidecar, read-only --------------------------------------------------


def test_a_missing_sidecar_dir_marks_all_unknown(tmp_path, capsys):
    wd = fixture(tmp_path, sidecars=False)
    assert T.main([str(wd)]) == 0
    text = capsys.readouterr().out
    assert "no sidecar" in text
    assert T.measure(wd)["classes"]["unknown"] == sorted(ALL_SEVEN)
    assert "t_right" in text


def test_the_run_writes_nothing_inside_the_workdir(tmp_path):
    wd = fixture(tmp_path)
    before = sorted(p.relative_to(wd).as_posix() for p in wd.rglob("*") if p.is_file())
    out = tmp_path / "out.json"
    assert T.main([str(wd), "--json", str(out)]) == 0
    after = sorted(p.relative_to(wd).as_posix() for p in wd.rglob("*") if p.is_file())
    assert before == after
    assert json.loads(out.read_text(encoding="utf-8"))["trusted_count"] == 7
