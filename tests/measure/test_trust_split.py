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


def _ref(run_id: str, trace_id: str | None, kind: str) -> dict:
    return {"run_id": run_id, "trace_id": trace_id, "kind": kind}


def fixture(root: Path, *, pool: bool = True, sidecars: bool = True, rounds: bool = True) -> Path:
    wd = root / "wd"
    status = {
        "t_right": {"reference_run_ids": ["run-r1", "re-a"]},
        "t_wrong": {"reference_run_ids": ["run-w1"]},
        "t_mixed": {"reference_run_ids": ["run-m1", "run-m2", "re-c"]},
        "t_reroll_only": {"reference_run_ids": ["run-u1"]},
        "t_no_sidecar": {"reference_run_ids": ["run-u2"]},
        "t_extra": {"reference_run_ids": ["run-e1", "re-e"]},
    }
    refs = {
        "t_right": {"references": [_ref("run-r1", "trace-r1", "recording"),
                                  _ref("re-a", None, "reroll")],
                    "recordings": [_ref("run-r1", "trace-r1", "recording"),
                                   _ref("re-a", None, "reroll")],
                    "failed": {}},
        "t_wrong": {"references": [_ref("run-w1", "trace-w1", "recording")],
                    "recordings": [_ref("run-w1", "trace-w1", "recording"),
                                   _ref("re-b", None, "reroll")],
                    "failed": {"re-b": "did not reach the End state"}},
        "t_mixed": {"references": [_ref("run-m1", "trace-m1", "recording"),
                                   _ref("run-m2", "trace-m2", "recording"),
                                   _ref("re-c", None, "reroll")],
                    "recordings": [_ref("run-m1", "trace-m1", "recording"),
                                   _ref("run-m2", "trace-m2", "recording"),
                                   _ref("re-c", None, "reroll"),
                                   _ref("re-d", None, "reroll")],
                    "failed": {"re-d": "did not reach the End state"}},
        "t_reroll_only": {"references": [_ref("run-u1", None, "reroll")],
                          "recordings": [_ref("run-u1", None, "reroll")],
                          "failed": {}},
        "t_no_sidecar": {"references": [_ref("run-u2", "trace-u9", "recording")],
                         "recordings": [_ref("run-u2", "trace-u9", "recording")],
                         "failed": {}},
        "t_extra": {"references": [_ref("run-e1", "trace-e1", "recording"),
                                   _ref("re-e", None, "reroll")],
                    "recordings": [_ref("run-e1", "trace-e1", "recording"),
                                   _ref("re-e", None, "reroll")],
                    "failed": {}},
    }
    # Round 1 trusts the five; round 2 additionally trusts t_extra and refuses t_out.
    if rounds:
        for name, trusted in (("1", ["t_right", "t_wrong", "t_mixed", "t_reroll_only", "t_no_sidecar"]),
                              ("2", ["t_right", "t_wrong", "t_mixed", "t_reroll_only",
                                     "t_no_sidecar", "t_extra"])):
            rows = [{"task_id": tid, "trusted": tid in trusted, "refused": False}
                    for tid in [*status, "t_out"]]
            rows.append({"task_id": "t_out", "trusted": False, "refused": name == "2"})
            _write(wd / "rounds" / name / "tasks.json",
                   {"counts": {}, "rows": rows})
    _write(wd / "task_status.json", status)
    _write(wd / "references.json", refs)
    if sidecars:
        for trace_id, reward in (("trace-r1", 1.0), ("trace-w1", 0.0), ("trace-m1", 1.0),
                                 ("trace-m2", 0.0), ("trace-e1", 1.0)):
            _write(wd / "grader" / f"{trace_id}.json", _sidecar(trace_id, reward))
    if pool:
        _write(wd / "exam" / "reference_pool.json", {"t_waiting": {}})
    return wd


# --- the sidecar split ------------------------------------------------------


def test_every_reference_passing_is_right_and_every_failing_is_wrong(tmp_path):
    result = T.measure(fixture(tmp_path), round_name="1")
    assert result["classes"]["right"] == ["t_right"]
    assert result["classes"]["wrong"] == ["t_wrong"]
    assert result["wrong_share_scored"] == 1 / 3
    assert result["wrong_share_trusted"] == 1 / 5


def test_a_task_split_across_pass_and_fail_is_mixed_and_unscored_is_unknown(tmp_path):
    result = T.measure(fixture(tmp_path), round_name="1")
    assert result["classes"]["mixed"] == ["t_mixed"]
    assert sorted(result["classes"]["unknown"]) == ["t_no_sidecar", "t_reroll_only"]
    by_id = {r["task_id"]: r for r in result["rows"]}
    assert by_id["t_reroll_only"]["unscored_references"] == 1
    assert by_id["t_no_sidecar"]["unscored_references"] == 1


def test_the_text_table_names_the_wrong_share_and_every_task_id(tmp_path):
    text = T.report(T.measure(fixture(tmp_path), round_name="1"))
    assert "33.3% (1/3) of scored" in text
    for tid in ("t_right", "t_wrong", "t_mixed", "t_reroll_only", "t_no_sidecar"):
        assert tid in text


# --- the harness-only proxy -------------------------------------------------


def test_one_reach_corroborates_and_an_unreproduced_end_state_flags(tmp_path):
    result = T.measure(fixture(tmp_path), round_name="1")
    by_id = {r["task_id"]: r["proxy"] for r in result["rows"]}
    assert by_id["t_right"] == T.AGREE
    assert by_id["t_wrong"] == T.DISAGREE
    assert by_id["t_reroll_only"] == T.AGREE
    assert by_id["t_no_sidecar"] == T.NONE


def test_a_reach_beats_a_miss_on_the_same_task(tmp_path):
    by_id = {r["task_id"]: r for r in T.measure(fixture(tmp_path), round_name="1")["rows"]}
    assert by_id["t_mixed"]["reruns_reached"] == 1
    assert by_id["t_mixed"]["reruns_failed"] == 1
    assert by_id["t_mixed"]["proxy"] == T.AGREE


def test_the_cross_table_counts_scored_tasks_only(tmp_path):
    result = T.measure(fixture(tmp_path), round_name="1")
    assert result["cross"] == {"right": {"agree": 1, "disagree": 0, "none": 0},
                               "wrong": {"agree": 0, "disagree": 1, "none": 0}}
    assert result["cross_excluded"] == {"mixed": 1, "unknown": 2}


# --- rounds, pool, refusals -------------------------------------------------


def test_the_latest_round_is_default_and_a_named_round_selects(tmp_path):
    assert "t_extra" in T.measure(fixture(tmp_path))["trusted"]
    assert "t_extra" not in T.measure(fixture(tmp_path), round_name="1")["trusted"]


def test_pool_size_and_refusals_come_off_the_build_records(tmp_path):
    late = T.measure(fixture(tmp_path / "a"))
    assert (late["pool_size"], late["pool_found"]) == (1, True)
    assert late["refused"] == ["t_out"]
    missing = T.measure(fixture(tmp_path / "b", pool=False))
    assert (missing["pool_size"], missing["pool_found"]) == (0, False)


# --- no sidecar, read-only --------------------------------------------------


def test_a_missing_sidecar_dir_leaves_the_proxy_only_and_marks_all_unknown(tmp_path, capsys):
    wd = fixture(tmp_path, sidecars=False)
    assert T.main([str(wd), "--round", "1"]) == 0
    text = capsys.readouterr().out
    assert "no sidecar" in text
    assert T.measure(wd, round_name="1")["classes"]["unknown"] == sorted(
        ["t_right", "t_wrong", "t_mixed", "t_reroll_only", "t_no_sidecar"])
    assert "agree" in text and "t_right" in text


def test_the_run_writes_nothing_inside_the_workdir(tmp_path):
    wd = fixture(tmp_path)
    before = sorted(p.relative_to(wd).as_posix() for p in wd.rglob("*") if p.is_file())
    out = tmp_path / "out.json"
    assert T.main([str(wd), "--round", "1", "--json", str(out)]) == 0
    after = sorted(p.relative_to(wd).as_posix() for p in wd.rglob("*") if p.is_file())
    assert before == after
    assert json.loads(out.read_text(encoding="utf-8"))["trusted_count"] == 5


def test_a_workdir_without_a_round_file_is_an_error(tmp_path, capsys):
    assert T.main([str(fixture(tmp_path, rounds=False))]) == 2
    assert "no rounds" in capsys.readouterr().out


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
    assert T.main([str(wd), "--round", "1", "--json", str(out)]) == 2
    assert "inside the workdir" in capsys.readouterr().out
    assert not out.exists()
