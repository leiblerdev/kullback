"""The vf test harness: what the foundation costs and what it gets wrong, honestly.

Per corpus, for one workdir (or two, where only the Verifier source differs):

1. trusted tasks on a wrong Reference, as a share of trusted;
2. wrong References caught (the task left untrusted or its Reference rejected);
3. false rejections: runs the sidecar marks right that the Verifier fails;
4. loopholes: runs the sidecar marks wrong that pass, plus mutated runs that pass;
5. checks with no valid because;
6. cost per task and per trusted task.

Every rate with its denominator and a Wilson 95% interval. The sidecar is the
benchmark's own reward per recording: measurement only, never read by a build.

    uv run python scripts/experiments/vf_measure.py WORKDIR --corpus LABEL --split SPLIT \
        [--against OTHER] [--sidecar DIR] [--final] [--out report.json]

Without --final only the dev slice is reported. With --final the test slice is
reported too, and the run is appended to the finals log. Two workdirs must hold
the same tasks and the same recorded runs; anything else is refused.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vf_lib as L  # noqa: E402

from kullback.gates import verifier_suite as S  # noqa: E402

FINALS_LOG = "vf_finals.log"


def audited_subset(split: dict, corpus: str, final: bool) -> tuple[list[str], list[str]]:
    """The task ids this run may report: dev always, test only with --final."""
    try:
        dev = list(split["dev"][corpus])
    except KeyError:
        raise ValueError(f"split has no dev slice for corpus {corpus!r}") from None
    if not final:
        return dev, []
    return dev, list(split.get("test", {}).get(corpus, []))


def score_runs(workdir: Path, sidecar: Path, task_ids: list[str]) -> dict[str, Any]:
    """Every recorded run with a sidecar reward, scored against its Verifier."""
    canon = L.load_canon(workdir)
    tools = L.write_tools_of(workdir)
    rewards = L.sidecar_rewards(sidecar)
    traces = L.run_trace_of(workdir)
    runs = L.task_runs(workdir)
    false_rejects, loopholes = [], []
    scored_right = scored_wrong = 0
    for task_id in task_ids:
        verifier = L.load_verifier(workdir, task_id)
        if verifier is None:
            continue
        for run in runs.get(task_id, []):
            reward = rewards.get(traces.get(task_id, {}).get(run.run_id) or "")
            if reward is None:
                continue
            try:
                passed, _ = S.check_run(verifier, run, canon, write_tools=tools)
            except Exception:
                continue
            if reward == 1.0:
                scored_right += 1
                if not passed:
                    false_rejects.append(f"{task_id}/{run.run_id}")
            elif reward == 0.0:
                scored_wrong += 1
                if passed:
                    loopholes.append(f"{task_id}/{run.run_id}")
    return {"scored_right": scored_right, "false_rejects": false_rejects,
            "scored_wrong": scored_wrong, "loopholes": loopholes}


def mutated_passes(workdir: Path, task_ids: list[str]) -> dict[str, Any]:
    """Mutated runs that pass, reusing the suite's own wrong/swap/unfinished shapes."""
    canon = L.load_canon(workdir)
    fn = L.canon_fn_of(workdir)
    tools = L.write_tools_of(workdir)
    references = L._read(workdir / "references.json", {}) or {}
    runs = L.task_runs(workdir)
    generated = passing = tasks_with = tasks_scored = 0
    passing_ids: list[str] = []
    for task_id in task_ids:
        verifier = L.load_verifier(workdir, task_id)
        if verifier is None:
            continue
        by_id = {run.run_id: run for run in runs.get(task_id, [])}
        kept = [r.get("run_id") for r in (references.get(task_id) or {}).get("references", [])]
        reference = next((by_id[r] for r in kept if r in by_id), None)
        if reference is None:
            continue
        tasks_scored += 1
        rows = S.world_rows(by_id.values())
        mutants: list[tuple[str, Any]] = [("swap", run) for _, _, run in S.swapped_runs(verifier, reference, fn, rows)]
        for name, mutant in (("wrong", S.wrong_run(verifier, reference, canon)),
                             ("unfinished", S.unfinished_run(verifier, reference, canon)),
                             ("empty", S._empty_run(reference))):
            if mutant is not None:
                mutants.append((name, mutant))
        task_passed = False
        for name, mutant in mutants:
            generated += 1
            try:
                passed, _ = S.check_run(verifier, mutant, canon, write_tools=tools)
            except Exception:
                continue
            if passed:
                passing += 1
                task_passed = True
                passing_ids.append(f"{task_id}/{name}/{mutant.run_id}")
        tasks_with += int(task_passed)
    return {"generated": generated, "passing": passing, "passing_ids": passing_ids,
            "tasks_with": tasks_with, "tasks_scored": tasks_scored}


def becauseless(workdir: Path, task_ids: list[str]) -> dict[str, int]:
    """Atoms with no valid because, over tasks carrying a Verifier."""
    intents = L.intent_text_of(workdir)
    rules = L.policy_rules_of(workdir)
    atoms = missing = 0
    for task_id in task_ids:
        verifier = L.load_verifier(workdir, task_id)
        if verifier is None:
            continue
        for atom in verifier.atoms:
            atoms += 1
            missing += not L.because_ok(atom, intents.get(task_id), rules)
    return {"atoms": atoms, "missing": missing}


def pooled_ids(workdir: Path) -> set[str]:
    body = L._read(workdir / "exam" / "reference_pool.json", None)
    return set(body) if isinstance(body, dict) else set()


def tier_rows(workdir: Path) -> dict[str, dict]:
    """task_id -> {"trusted": ...} off the build's tier report (tiers.json), else the latest round's rows."""
    body = L._read(workdir / "tiers.json", None)
    if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
        return L.latest_round_rows(workdir)
    return {row["task_id"]: {"trusted": row.get("tier") == "trusted", "tier": row.get("tier")}
            for row in body["rows"] if isinstance(row, dict) and row.get("task_id")}


def measure(workdir: Path, sidecar: Path, task_ids: list[str], seed: int, audit_n: int) -> dict[str, Any]:
    """All six numbers over one task slice of one workdir; trusted is the tier report's where it has one."""
    rows = tier_rows(workdir)
    trusted = {t for t in task_ids if rows.get(t, {}).get("trusted")}
    references = L._read(workdir / "references.json", {}) or {}
    rewards = L.sidecar_rewards(sidecar)
    class_of = {t: L.reference_class(references.get(t, {}), rewards) for t in task_ids}
    scored = [t for t in task_ids if class_of[t] != L.UNKNOWN]
    trusted_scored = [t for t in scored if t in trusted]
    wrong_trusted = [t for t in trusted_scored if class_of[t] == L.WRONG]
    wrong = [t for t in scored if class_of[t] == L.WRONG]
    pooled = pooled_ids(workdir)
    caught = [t for t in wrong if t not in trusted]
    rejected = [t for t in caught if not (references.get(t) or {}).get("references") or t in pooled]
    runs = score_runs(workdir, sidecar, task_ids)
    mutated = mutated_passes(workdir, task_ids)
    because = becauseless(workdir, task_ids)
    rng = random.Random(seed)
    disagreements = sorted(set(runs["false_rejects"]) | set(runs["loopholes"]))
    rng.shuffle(disagreements)
    cost = L.build_cost_usd(workdir)
    n_trusted_all = sum(1 for row in rows.values() if row.get("trusted"))
    return {
        "tasks": len(task_ids), "trusted": len(trusted),
        "trusted_wrong": {"k": len(wrong_trusted), "n": len(trusted_scored), "ids": sorted(wrong_trusted)},
        "caught": {"k": len(caught), "n": len(wrong), "ids": sorted(caught),
                   "rejected": len(rejected)},
        "false_rejects": {"k": len(runs["false_rejects"]), "n": runs["scored_right"],
                          "ids": sorted(runs["false_rejects"])},
        "sidecar_loopholes": {"k": len(runs["loopholes"]), "n": runs["scored_wrong"],
                              "ids": sorted(runs["loopholes"])},
        "mutated": {"k": mutated["passing"], "n": mutated["generated"],
                    "ids": sorted(mutated["passing_ids"]),
                    "tasks": {"k": mutated["tasks_with"], "n": mutated["tasks_scored"]}},
        "because": {"k": because["missing"], "n": because["atoms"]},
        "cost_usd": cost, "cost_tasks": len(rows), "cost_trusted": n_trusted_all,
        "audit": disagreements[:audit_n],
    }


def with_intervals(result: dict[str, Any]) -> dict[str, Any]:
    """Every rate annotated with its Wilson 95% interval, in place."""
    for key in ("trusted_wrong", "caught", "false_rejects", "sidecar_loopholes", "mutated"):
        cell = result[key]
        cell["rate"] = L.rate(cell["k"], cell["n"])
        interval = L.wilson(cell["k"], cell["n"])
        cell["wilson95"] = list(interval) if interval else None
    for key in ("tasks", "because"):
        cell = result[key] if isinstance(result[key], dict) else None
        if cell is not None and "k" in cell:
            cell["rate"] = L.rate(cell["k"], cell["n"])
            interval = L.wilson(cell["k"], cell["n"])
            cell["wilson95"] = list(interval) if interval else None
    return result


def check_same_runs(first: Path, second: Path) -> list[str]:
    """The one-variable rule: same tasks, same recorded runs, or refuse."""
    problems = []
    a, b = L.run_signature(first), L.run_signature(second)
    for task_id in sorted(set(a) | set(b)):
        if task_id not in a:
            problems.append(f"{task_id}: missing from {first}")
        elif task_id not in b:
            problems.append(f"{task_id}: missing from {second}")
        elif a[task_id] != b[task_id]:
            problems.append(f"{task_id}: runs differ: {a[task_id]} vs {b[task_id]}")
    return problems


def fmt(result: dict[str, Any]) -> str:
    def line(name: str, cell: dict) -> str:
        interval = cell.get("wilson95")
        span = f" [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else " [n/a]"
        rate = cell.get("rate")
        return f"{name}: {rate:.3f} ({cell['k']}/{cell['n']}){span}" if rate is not None else \
            f"{name}: n/a ({cell['k']}/{cell['n']})"
    cost = result["cost_usd"]
    per_task = f"{cost / result['cost_tasks']:.4f}" if cost is not None and result["cost_tasks"] else "n/a"
    per_trusted = f"{cost / result['cost_trusted']:.4f}" if cost is not None and result["cost_trusted"] else "n/a"
    return "\n".join([
        f"tasks: {result['tasks']}, trusted: {result['trusted']}",
        line("trusted on wrong Reference", result["trusted_wrong"]),
        line("wrong References caught    ", result["caught"])
        + f" (rejected: {result['caught']['rejected']})",
        line("false rejections           ", result["false_rejects"]),
        line("sidecar loopholes          ", result["sidecar_loopholes"]),
        line("mutated runs passing       ", result["mutated"])
        + f"; tasks with one: {result['mutated']['tasks']['k']}/{result['mutated']['tasks']['n']}",
        line("checks with no because     ", result["because"]),
        f"cost per task: {per_task}, per trusted task: {per_trusted} (usd {cost})",
        f"audit: {len(result['audit'])} disagreement ids (JSON report only)",
    ])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--against", default=None,
                        help="second workdir; only its Verifiers may differ")
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--final-log", default=FINALS_LOG)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit-n", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        split = json.loads(Path(args.split).read_text(encoding="utf-8"))
        dev, test = audited_subset(split, args.corpus, args.final)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    workdirs = [Path(args.workdir)]
    if args.against:
        workdirs.append(Path(args.against))
        problems = check_same_runs(workdirs[0], workdirs[1])
        if problems:
            print("refusing: the recorded runs differ:", file=sys.stderr)
            for problem in problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            return 2
    report: dict[str, Any] = {"corpus": args.corpus, "final": args.final, "seed": args.seed, "arms": []}
    for workdir in workdirs:
        sidecar = Path(args.sidecar) if args.sidecar else workdir / "grader"
        arm = {"workdir": str(workdir)}
        if not args.final:
            arm["dev"] = with_intervals(measure(workdir, sidecar, dev, args.seed, args.audit_n))
        else:
            arm["dev"] = with_intervals(measure(workdir, sidecar, dev, args.seed, args.audit_n))
            arm["test"] = with_intervals(measure(workdir, sidecar, test, args.seed, args.audit_n))
        report["arms"].append(arm)
    for arm in report["arms"]:
        print(f"== {arm['workdir']} (dev) ==")
        print(fmt(arm["dev"]))
        if "test" in arm:
            print(f"== {arm['workdir']} (test) ==")
            print(fmt(arm["test"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.final:
        summary = {str(w): {s: {"tasks": arm[s]["tasks"], "trusted": arm[s]["trusted"]}
                            for s in arm if s in ("dev", "test")}
                   for w, arm in zip(workdirs, report["arms"], strict=True)}
        with open(args.final_log, "a", encoding="utf-8") as log:
            log.write(json.dumps({"at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          "corpus": args.corpus, "seed": args.seed, "summary": summary},
                                         sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
