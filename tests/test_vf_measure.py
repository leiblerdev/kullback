"""`scripts/experiments/vf_measure.py`: the honesty harness. No model, no live build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gates.verifier_fixtures import WRITE_TOOLS, alt_path_run, derive, reference_run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))

from kullback.runner.records import as_dict  # noqa: E402
import vf_lib as L  # noqa: E402
import vf_measure as M  # noqa: E402

TASK = "t1"
OTHER = "t2"


def write_workdir(root: Path, rewards: dict[str, float], trusted: dict[str, bool]) -> Path:
    """A workdir where each task's kept Reference is the hand-built one, scored as told."""
    workdir = root / "work"
    (workdir / "rounds" / "5").mkdir(parents=True)
    (workdir / "rounds" / "5" / "tasks.json").write_text(json.dumps(
        {"rows": [{"task_id": t, "trusted": trusted[t]} for t in rewards]}), encoding="utf-8")
    references, runs = {}, {}
    for task_id, reward in rewards.items():
        trace = f"trace-{task_id}"
        run = reference_run().model_copy(update={"run_id": f"ref-{task_id}", "task_id": task_id})
        refs = [{"run_id": run.run_id, "trace_id": trace}]
        references[task_id] = {"references": refs, "recordings": refs}
        runs[task_id] = [as_dict(run)]
        (workdir / "grader").mkdir(exist_ok=True)
        (workdir / "grader" / f"{trace}.json").write_text(json.dumps(
            {"trace_id": trace, "fields": {"reward_info": {"reward": reward}}}), encoding="utf-8")
    (workdir / "references.json").write_text(json.dumps(references), encoding="utf-8")
    (workdir / "exam").mkdir(exist_ok=True)
    (workdir / "exam" / "task_runs.json").write_text(json.dumps(runs), encoding="utf-8")
    (workdir / "canon-rules.json").write_text("{}", encoding="utf-8")
    (workdir / "tool_sigs.json").write_text(json.dumps(
        [{"name": name, "kind": "write"} for name in sorted(WRITE_TOOLS)]), encoding="utf-8")
    (workdir / "tasks.json").write_text(json.dumps(
        {"tasks": [{"id": t, "intent": None, "name": None} for t in rewards]}), encoding="utf-8")
    (workdir / "constraints_check.json").write_text(json.dumps({"constraints": []}), encoding="utf-8")
    (workdir / "budget.json").write_text(json.dumps({"total": {"usd": 10.0}}), encoding="utf-8")
    (workdir / "verifiers").mkdir(exist_ok=True)
    for task_id in rewards:
        scratch = workdir / f"derive-{task_id}"
        scratch.mkdir(exist_ok=True)
        verifier = derive(scratch, reruns=[alt_path_run()])
        verifier = verifier.model_copy(update={"task_id": task_id})
        (workdir / "verifiers" / f"{task_id}.json").write_text(
            json.dumps(as_dict(verifier)), encoding="utf-8")
    return workdir

def test_a_reward_zero_reference_is_right_only_when_every_kept_reference_scores_one():
    rewards = {"tr1": 1.0, "tr2": 0.0}
    assert L.reference_class({"references": [{"trace_id": "tr1"}]}, rewards) == L.RIGHT
    assert L.reference_class({"references": [{"trace_id": "tr2"}]}, rewards) == L.WRONG
    assert L.reference_class({"references": [{"trace_id": "tr1"}, {"trace_id": "tr2"}]},
                             rewards) == L.MIXED
    assert L.reference_class({"references": [{"run_id": "reroll-only"}]}, rewards) == L.UNKNOWN
    assert L.reference_class({"references": []}, rewards) == L.UNKNOWN


def test_a_because_holds_only_on_an_intent_quote_or_a_policy_rule():
    atom = derive(Path("/tmp"), reruns=[alt_path_run()]).atoms[0]
    quoted = dict(atom.target, because={"quote": "cancel the order"})
    ruled = dict(atom.target, because={"rule": "r1"})
    assert L.because_ok(atom.model_copy(update={"target": quoted}), "please cancel the order", set())
    assert L.because_ok(atom.model_copy(update={"target": ruled}), None, {"r1"})
    assert not L.because_ok(atom, "please cancel the order", {"r1"})
    assert not L.because_ok(atom.model_copy(update={"target": quoted}), "unrelated words", set())
    assert not L.because_ok(atom.model_copy(update={"target": ruled}), None, {"other"})


def test_wilson_rates_zero_of_nothing_as_unmeasurable():
    assert L.wilson(0, 0) is None and L.rate(3, 0) is None
    lo, hi = L.wilson(0, 10)
    assert lo == 0.0 and 0.24 < hi < 0.30
    lo, hi = L.wilson(10, 10)
    assert hi == 1.0 and 0.70 < lo < 0.75


def test_a_trusted_task_on_a_reward_zero_reference_counts_against_the_share(tmp_path):
    workdir = write_workdir(tmp_path, {TASK: 0.0, OTHER: 1.0}, {TASK: True, OTHER: True})
    result = M.with_intervals(M.measure(workdir, workdir / "grader", [TASK, OTHER], 7, 20))
    assert (result["trusted_wrong"]["k"], result["trusted_wrong"]["n"]) == (1, 2)
    assert result["caught"] == {"k": 0, "n": 1, "ids": [], "rejected": 0,
                                "rate": 0.0, "wilson95": result["caught"]["wilson95"]}
    assert result["false_rejects"]["n"] == 1 and result["false_rejects"]["k"] == 0
    assert result["sidecar_loopholes"]["k"] == 1 and result["sidecar_loopholes"]["n"] == 1
    assert result["because"]["k"] == result["because"]["n"] > 0
    assert result["cost_usd"] == 10.0


def test_an_untrusted_task_on_a_reward_zero_reference_counts_as_caught(tmp_path):
    workdir = write_workdir(tmp_path, {TASK: 0.0}, {TASK: False})
    result = M.with_intervals(M.measure(workdir, workdir / "grader", [TASK], 7, 20))
    assert (result["trusted_wrong"]["k"], result["trusted_wrong"]["n"]) == (0, 0)
    assert result["trusted_wrong"]["rate"] is None
    assert (result["caught"]["k"], result["caught"]["n"]) == (1, 1)


def test_test_numbers_stay_sealed_until_final(tmp_path, capsys):
    workdir = write_workdir(tmp_path, {TASK: 0.0, OTHER: 1.0}, {TASK: True, OTHER: True})
    split = {"dev": {"c": [TASK]}, "test": {"c": [OTHER]}}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    base = [str(workdir), "--corpus", "c", "--split", str(split_path),
            "--out", str(tmp_path / "r.json")]
    assert M.main(base) == 0
    shown = capsys.readouterr().out
    assert "(dev)" in shown and "(test)" not in shown
    assert M.main(base + ["--final", "--final-log", str(tmp_path / "finals.log")]) == 0
    shown = capsys.readouterr().out
    assert "(test)" in shown
    assert len((tmp_path / "finals.log").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_an_unknown_corpus_and_unequal_runs_both_refuse(tmp_path):
    workdir = write_workdir(tmp_path, {TASK: 1.0}, {TASK: True})
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"dev": {"c": [TASK]}, "test": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.main([str(workdir), "--corpus", "missing", "--split", str(split_path)])
    other = write_workdir(tmp_path / "o", {TASK: 1.0}, {TASK: True})
    runs = json.loads((other / "exam" / "task_runs.json").read_text(encoding="utf-8"))
    runs[TASK][0]["run_id"] = "different-run"
    (other / "exam" / "task_runs.json").write_text(json.dumps(runs), encoding="utf-8")
    assert M.main([str(workdir), "--corpus", "c", "--split", str(split_path),
                   "--against", str(other)]) == 2
