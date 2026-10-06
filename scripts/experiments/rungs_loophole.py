"""Offline measure of the shape rung orders and the skipped-step loophole rule over a stored build.

Re-derives every Task's Verifier from the Reference and seeds the build kept, under each shape rung
order (`derive.SHAPE_RUNG_ORDERS`), runs the D79 suite with no model (check 6 re-scores the probe Run
the build stored), and the false rejection rule of the trusted gate over the held-out Runs. Each rung
order is read twice: check 6 under the skipped-step rule, and under the old rule where any passing
probe fails it. Nothing is written: the table goes to stdout, `--json` adds the per-Task rows.

Usage:
    PYTHONPATH=<worktree> python scripts/experiments/rungs_loophole.py --workdir <build workdir> [--json]

Prints counts and Task ids only, never a value out of the Runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from kullback import derive as derive_mod
from kullback.examiner import reference as reference_mod
from kullback.examiner import stage
from kullback.examiner.session import load_store
from kullback.gates import artifacts, loosening, trust
from kullback.gates import verifier_suite as S
from kullback.gates.probes import version_hash
from kullback.runner.records import Constraint, Verifier, apply_intent, load_task_run, read_json

BASELINE = "row_first"
LOOPHOLE = "loophole_probe_fails"


def _paths(workdir: Path, replays: dict) -> dict[str, dict[str, Path]]:
    """Per Task, the file each run id names, the way the trusted gate resolves seeds."""
    return trust.seed_files(workdir, replays, read_json(workdir / "rerolls.json", {}) or {})


def _path_of(workdir: Path, files: dict, task_id: str, run_id: str) -> Path:
    named = (files.get(task_id) or {}).get(run_id)
    return named or workdir / "runs" / task_id / f"{run_id}.jsonl"


def _recording(workdir: Path, files: dict, task_id: str, row: dict, write_tools: set, fn: Any,
               atoms: list) -> reference_mod.Recording:
    path = _path_of(workdir, files, task_id, str(row["run_id"]))
    return reference_mod.load(str(path), row.get("kind") or reference_mod.RECORDING, run_id=row["run_id"],
                              trace_id=row.get("trace_id"), write_tools=write_tools, fn=fn, atoms=atoms,
                              task_id=task_id)


class Build:
    """The build's inputs, read once, and nothing written back."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        store = load_store(workdir)
        self.canon = stage.rules_of(store)
        self.fn = S.canon_fn(self.canon)
        self.write_tools, self.read_tools = stage._tool_names(store["sigs"])
        self.intents = stage._validated_intents(store)
        self.user_rules = store["user_rules"]
        self.tasks = {task.id: task for task in store["tasks"]}
        checked = read_json(workdir / "constraints_check.json", {}) or {}
        final = [Constraint.model_validate(c) for c in checked.get("constraints") or []]
        self.constraints = [c for c in final if c.compiled or c.judge_atom]
        self.atoms = reference_mod.hard_atoms(self.constraints, self.write_tools, self.read_tools)
        self.status = read_json(workdir / "task_status.json", {}) or {}
        self.references = read_json(workdir / "references.json", {}) or {}
        replays = read_json(workdir / "replays.json", {}) or {}
        rerolls = read_json(workdir / "rerolls.json", {}) or {}
        self.files = _paths(workdir, replays)
        self.legitimate = loosening.legitimate_runs(replays, rerolls, loosening.discarded_runs(self.status))
        self._held: dict[str, list] = {}

    def held_out(self, task_id: str) -> list:
        """The Task's legitimate Runs, loaded once, for the false rejection rule."""
        if task_id not in self._held:
            runs = []
            for run_id in sorted(self.legitimate.get(task_id) or ()):
                try:
                    runs.append(load_task_run(_path_of(self.workdir, self.files, task_id, run_id), task_id))
                except (OSError, ValueError):
                    continue
            self._held[task_id] = runs
        return self._held[task_id]

    def confirmation(self, task_id: str) -> Optional[reference_mod.Confirmation]:
        row = self.references.get(task_id) or {}
        if not row.get("references"):
            return None

        def load(r: dict) -> reference_mod.Recording:
            return _recording(self.workdir, self.files, task_id, r, self.write_tools, self.fn, self.atoms)

        return reference_mod.Confirmation(references=[load(r) for r in row["references"]],
                                          recordings=[load(r) for r in row.get("recordings") or []],
                                          failed=dict(row.get("failed") or {}))

    def probe(self, task_id: str):
        path = self.workdir / "probes" / f"probe-{task_id}.jsonl"
        return (lambda model, verifier: load_task_run(path, task_id)) if path.is_file() else None


def measure(build: Build, task_id: str, order: str) -> dict:
    """One Task under one rung order: the suite both ways, and the false rejection rule."""
    derive_mod.SHAPE_RUNG_ORDER = order
    task = build.tasks[task_id]
    confirmation = build.confirmation(task_id)
    row = build.status.get(task_id) or {}
    task_for = apply_intent(task, build.intents[task_id]) if task_id in build.intents else task
    verifier = stage.derive_for(task_for, confirmation, canon_rules=build.canon, write_tools=build.write_tools,
                                constraints=build.constraints, intent=build.intents.get(task_id))
    paths = [r.path for r in confirmation.references]
    probe = build.probe(task_id)
    may_probe = probe is not None and "verifier_loophole" not in (row.get("not_run") or [])
    first = confirmation.references[0]
    rules_trace = first.trace_id or next((r.trace_id for r in confirmation.references if r.trace_id), None)
    gates = stage.suite_for(task_for, verifier, paths, canon_rules=build.canon, write_tools=build.write_tools,
                            user_rules=build.user_rules, rules_trace=rules_trace,
                            probe_model=object() if may_probe else None, run_probe=probe,
                            may_probe=may_probe, intent=build.intents.get(task_id))
    results = S.d79_results(gates)
    loophole = next(g for g in gates if g.stage == "verifier_loophole")
    old = dict(results, **{LOOPHOLE: bool(may_probe and not loophole.metrics.get("probe_passed"))})
    held = loosening.false_rejection(verifier, build.held_out(task_id), build.legitimate.get(task_id, set()),
                                     build.canon, build.write_tools)
    return {"hash": version_hash(verifier), "with_b": _passed(results, row), "without_b": _passed(old, row),
            "failed_with_b": sorted(k for k, v in results.items() if not v),
            "failed_without_b": sorted(k for k, v in old.items() if not v),
            "skipped_calls": loophole.metrics.get("skipped_calls") or [],
            "over_strict": loosening.over_strict(held), "rejected": held["rejected"], "held_out": held["held_out"]}


def _passed(results: dict, row: dict) -> bool:
    """The suite verdict with the D199 single-path waiver the build applies off the same row."""
    if artifacts.verifier_gate(results).passed:
        return True
    return bool((row.get("second_path") or {}).get("reason") == stage.SINGLE_PATH_BY_STRUCTURE
                and row.get("pool_at_reference")
                and all(v for k, v in results.items() if k != "second_path_passes"))


def run(workdir: Path) -> dict:
    build = Build(workdir)
    task_ids = sorted(t for t in build.status if build.references.get(t, {}).get("references"))
    stored = {path.stem: version_hash(Verifier.model_validate(read_json(path)))
              for path in sorted((workdir / "verifiers").glob("*.json"))}
    rows: dict[str, dict] = {}
    for order in derive_mod.SHAPE_RUNG_ORDERS:
        rows[order] = {}
        for task_id in task_ids:
            try:
                rows[order][task_id] = measure(build, task_id, order)
            except Exception as error:  # noqa: BLE001, one Task that cannot be derived is counted, not fatal
                rows[order][task_id] = {"error": type(error).__name__}
    derive_mod.SHAPE_RUNG_ORDER = BASELINE
    base = rows[BASELINE]
    return {"workdir": str(workdir), "tasks": len(build.status),
            "build_suite_passed": sum(bool(r.get("verifier_passed")) for r in build.status.values()),
            "baseline_same_verifier": sum(base[t].get("hash") == stored.get(t) for t in task_ids),
            "baseline_suite_agrees": sum(base[t].get("without_b") ==
                                         bool((build.status[t] or {}).get("verifier_passed")) for t in task_ids),
            "baseline_disagrees": sorted(t for t in task_ids if base[t].get("without_b") !=
                                         bool((build.status[t] or {}).get("verifier_passed"))),
            "rows": rows, "task_ids": task_ids}


def table(out: dict) -> str:
    task_ids, rows = out["task_ids"], out["rows"]
    base = {t: rows[BASELINE][t].get("without_b") for t in task_ids}
    lines = [f"{out['workdir']}",
             f"Tasks {out['tasks']}, with a Reference {len(task_ids)}, build suite passed {out['build_suite_passed']}",
             f"baseline re-derivation: same Verifier as stored {out['baseline_same_verifier']} of {len(task_ids)}, "
             f"suite verdict as stored {out['baseline_suite_agrees']} of {len(task_ids)}",
             "",
             "| variant | derived | suite passed | false rejections | suite passed, no false rejection | changed vs baseline |",
             "|---|---|---|---|---|---|"]
    for order in rows:
        for b in ("without_b", "with_b"):
            got = rows[order]
            derived = [t for t in task_ids if "error" not in got[t]]
            passed = [t for t in derived if got[t][b]]
            over = [t for t in derived if got[t]["over_strict"]]
            kept = [t for t in passed if not got[t]["over_strict"]]
            changed = sorted(t for t in task_ids if got[t].get(b) != base[t])
            name = f"{order}{' + B' if b == 'with_b' else ''}"
            lines.append(f"| {name} | {len(derived)} | {len(passed)} | {len(over)} | {len(kept)} | {len(changed)} |")
    lines.append("")
    for order in rows:
        for b in ("without_b", "with_b"):
            got = rows[order]
            gained = sorted(t for t in task_ids if got[t].get(b) and not base[t])
            lost = sorted(t for t in task_ids if base[t] and not got[t].get(b))
            if gained or lost:
                name = f"{order}{' + B' if b == 'with_b' else ''}"
                lines.append(f"{name}: suite gained {', '.join(gained) or 'none'}; lost {', '.join(lost) or 'none'}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="print the per-Task rows as JSON after the table")
    args = parser.parse_args(argv)
    out = run(args.workdir.resolve())
    print(table(out))
    if args.json:
        print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
