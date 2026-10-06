"""Offline G56 measure: re-derive every Task, score held-out Runs, name blocking atoms.

Re-uses scripts/experiments/rungs_loophole.py's Build (same derive + suite the
build runs, read-only over the smoke9 workdirs). Per Task it records: the suite
verdict, the false-rejection row, the atom that blocks each rejected held-out
Run, the full D79 check vector (so a flip that lets a wrong Run pass is
visible), and the tau2 reward of the Reference traces (measurement only).

Usage: PYTHONPATH=<repo> python scripts/experiments/g56_measure.py <workdir> <out.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from kullback import derive as derive_mod
from kullback.runner import target as T
from scripts.experiments import rungs_loophole as R


def _grader_rewards(workdir: Path) -> dict[str, float]:
    """tau2 reward by trace_id, from the grader sidecars (measurement only)."""
    out: dict[str, float] = {}
    grader = workdir / "grader"
    if not grader.is_dir():
        return out
    for path in sorted(grader.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fields = doc.get("fields") or {}
        reward = (fields.get("reward_info") or {}).get("reward")
        trace_id = doc.get("trace_id")
        if trace_id is not None and reward is not None:
            out.setdefault(str(trace_id), float(reward))
    return out


def _blocking_atoms(build: R.Build, task_id: str, verifier) -> dict[str, dict]:
    """Per held-out legitimate non-seed Run: pass, or the atom check_run names."""
    required = verifier.model_copy(update={"atoms": [a for a in verifier.atoms if a.kind != "hard"]})
    seeds = {r.run_id for r in build.confirmation(task_id).references}
    legitimate = build.legitimate.get(task_id, set())
    atoms = {a.id: a for a in verifier.atoms}
    out: dict[str, dict] = {}
    for run in build.held_out(task_id):
        if run.run_id in seeds or run.run_id not in legitimate:
            continue
        passed, atom_id = T.check_run(required, run, build.canon, write_tools=build.write_tools)
        if passed:
            out[run.run_id] = {"atom": None}
            continue
        atom = atoms.get(str(atom_id))
        payload = dict(atom.target) if atom is not None else {}
        out[run.run_id] = {"atom": str(atom_id), "kind": atom.kind if atom else None,
                           "payload_kind": payload.get("kind"), "key": payload.get("key"),
                           "field": payload.get("field"), "tool": payload.get("tool"),
                           "value": payload.get("value"), "text": payload.get("text")}
    return out


def measure_task(build: R.Build, rewards: dict[str, float], task_id: str) -> dict:
    row = R.measure(build, task_id, "tightest_first")
    confirmation = build.confirmation(task_id)
    ref_rewards = {}
    for ref in confirmation.references:
        trace_id = ref.trace_id
        ref_rewards[ref.run_id] = {"trace_id": trace_id, "reward": rewards.get(str(trace_id))}
    held = build.held_out(task_id)
    legitimate = build.legitimate.get(task_id, set())
    held_ids = sorted(run.run_id for run in held if run.run_id in legitimate)
    seeds = {r.run_id for r in confirmation.references}
    # Re-derive the verifier the same way R.measure did, to score blocking atoms.
    task = build.tasks[task_id]
    from kullback.runner.records import apply_intent
    task_for = apply_intent(task, build.intents[task_id]) if task_id in build.intents else task
    verifier = build_verifier(build, task_for, confirmation, task_id)
    blocking = _blocking_atoms(build, task_id, verifier)
    question_atoms = [a.id for a in verifier.atoms
                      if isinstance(a.target, dict) and a.target.get("kind") == "question"]
    demanded_facts = [a.id for a in verifier.atoms
                      if isinstance(a.target, dict) and a.target.get("kind") == "communicate"]
    return {"suite_passed": row.get("with_b"), "failed_checks": row.get("failed_with_b"),
            "held_out": row.get("held_out"), "held_out_ids": held_ids,
            "rejected": row.get("rejected"), "over_strict": row.get("over_strict"),
            "blocking": blocking, "question_atoms": question_atoms,
            "demanded_facts": demanded_facts, "reference_rewards": ref_rewards,
            "seed_run_ids": sorted(seeds), "hash": row.get("hash")}


def build_verifier(build: R.Build, task_for, confirmation, task_id):
    from kullback.examiner import stage

    return stage.derive_for(task_for, confirmation, canon_rules=build.canon,
                            write_tools=build.write_tools, constraints=build.constraints,
                            intent=build.intents.get(task_id))


def _wanted() -> set[str] | None:
    """A task subset from G56_TASKS (comma separated), for sharded runs."""
    raw = os.environ.get("G56_TASKS", "")
    return set(raw.split(",")) if raw.strip() else None


def run(workdir: Path) -> dict:
    build = R.Build(workdir)
    rewards = _grader_rewards(workdir)
    task_ids = sorted(t for t in build.status if build.references.get(t, {}).get("references"))
    wanted = _wanted()
    if wanted is not None:
        task_ids = [t for t in task_ids if t in wanted]
    rows: dict[str, dict] = {}
    for task_id in task_ids:
        try:
            rows[task_id] = measure_task(build, rewards, task_id)
        except Exception as error:  # noqa: BLE001, one Task that cannot be derived is counted, not fatal
            rows[task_id] = {"error": type(error).__name__}
    derive_mod.SHAPE_RUNG_ORDER = R.BASELINE
    suite_passed = sum(1 for r in rows.values() if r.get("suite_passed"))
    rejected = sum(1 for r in rows.values() if r.get("over_strict"))
    clean = sum(1 for r in rows.values() if r.get("suite_passed") and not r.get("over_strict"))
    return {"workdir": str(workdir), "tasks": len(task_ids), "suite_passed": suite_passed,
            "false_rejections": rejected, "passed_no_false_rejection": clean, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    out = run(args.workdir)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{out['workdir']}: tasks={out['tasks']} suite_passed={out['suite_passed']} "
          f"false_rejections={out['false_rejections']} clean={out['passed_no_false_rejection']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
