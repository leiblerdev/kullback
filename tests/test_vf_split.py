"""`scripts/experiments/vf_split.py`: the sealed dev/test split. No model, no live build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))
import vf_lib as L  # noqa: E402
import vf_split as P  # noqa: E402


def task_row(task_id: str, trusted: bool = True) -> dict:
    return {"task_id": task_id, "trusted": trusted}


def write_corpus(root: Path, label: str, classes: dict[str, str]) -> Path:
    """A workdir with one trusted task per class entry, each scored by one trace."""
    workdir = root / label
    (workdir / "rounds" / "3").mkdir(parents=True)
    (workdir / "rounds" / "3" / "tasks.json").write_text(json.dumps(
        {"rows": [task_row(t) for t in classes]}), encoding="utf-8")
    references, n = {}, 0
    for task_id, cls in classes.items():
        reward = {"right": 1.0, "wrong": 0.0, "mixed": None}.get(cls)
        trace = f"trace-{label}-{n}"
        n += 1
        refs = [{"run_id": f"run-{task_id}", "trace_id": trace}]
        if cls == "mixed":
            trace2 = f"trace-{label}-{n}-b"
            n += 1
            refs.append({"run_id": f"run-{task_id}-b", "trace_id": trace2})
            (workdir / "grader").mkdir(exist_ok=True)
            (workdir / "grader" / f"{trace2}.json").write_text(json.dumps(
                {"trace_id": trace2, "fields": {"reward_info": {"reward": 0.0}}}), encoding="utf-8")
            reward = 1.0
        references[task_id] = {"references": refs, "recordings": refs}
        if reward is not None:
            (workdir / "grader").mkdir(exist_ok=True)
            (workdir / "grader" / f"{trace}.json").write_text(json.dumps(
                {"trace_id": trace, "fields": {"reward_info": {"reward": reward}}}), encoding="utf-8")
    (workdir / "references.json").write_text(json.dumps(references), encoding="utf-8")
    return workdir


def test_the_split_keeps_each_class_share_across_dev_and_test(tmp_path):
    corpus = {f"t{i:02d}": ("right" if i % 2 else "wrong") for i in range(20)}
    workdir = write_corpus(tmp_path, "c1", corpus)
    body = P.build({"c1": workdir}, seed=7, dev_fraction=0.3)
    assert sorted(body["dev"]["c1"] + body["test"]["c1"]) == sorted(corpus)
    assert set(body["dev"]["c1"]) & set(body["test"]["c1"]) == set()
    assert len(body["dev"]["c1"]) == 6
    class_of = L.classes(workdir, workdir / "grader")
    assert {L.WRONG, L.RIGHT} <= {class_of[t] for t in body["dev"]["c1"]}


def test_a_lone_stratum_member_tunes_but_never_tests_alone(tmp_path):
    workdir = write_corpus(tmp_path, "c1", {"t1": "wrong", "t2": "right", "t3": "right"})
    body = P.build({"c1": workdir}, seed=7, dev_fraction=0.3)
    assert "t1" in body["dev"]["c1"] and "t1" not in body["test"]["c1"]


def test_the_split_is_written_once_and_deterministic(tmp_path):
    workdir = write_corpus(tmp_path, "c1", {f"t{i}": "right" for i in range(4)})
    out = tmp_path / "split.json"
    assert P.main([str(out), "--workdir", f"c1={workdir}", "--seed", "7"]) == 0
    first = out.read_text(encoding="utf-8")
    assert P.main([str(out), "--workdir", f"c1={workdir}", "--seed", "7"]) == 1
    out.unlink()
    assert P.main([str(out), "--workdir", f"c1={workdir}", "--seed", "7"]) == 0
    assert out.read_text(encoding="utf-8") == first


def test_two_corpora_split_independently(tmp_path):
    first = write_corpus(tmp_path, "c1", {f"a{i}": "right" for i in range(4)})
    second = write_corpus(tmp_path, "c2", {f"b{i}": "wrong" for i in range(4)})
    body = P.build({"c1": first, "c2": second}, seed=7, dev_fraction=0.5)
    assert set(body["dev"]) == {"c1", "c2"} == set(body["test"])
    assert all(t.startswith("a") for t in body["dev"]["c1"] + body["test"]["c1"])
    assert all(t.startswith("b") for t in body["dev"]["c2"] + body["test"]["c2"])
