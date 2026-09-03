"""One-directional loosening over an expandable legitimate pool (D127, D133), and the false-rejection number.

Tightening is free. A new Verifier version may newly pass a Run the last accepted version failed
only when that Run is in the legitimate pool: a confirmed replay of a Trace (the Reference and every
other production Run the Environment reproduced) or a re-roll of any round whose termination is a
success termination. The pool is built by `legitimate_runs` from the records the Runner wrote,
replays.json and the merged re-roll rows, never from a model's opinion, and it grows as re-rolls and
traces arrive. `false_rejection` is D133's per-Task number: over the legitimate Runs the Verifier
was not derived from, the fraction its atoms wrongly fail; the gate on it has no tuned threshold and
fails only the one case D133 warns about, a Verifier that recognises no frontier path but its seeds.
"""

from __future__ import annotations

from typing import Any, Optional

from kullback.gates.probes import as_verifier, version_hash, write_tools_of
from kullback.gates.verifier_suite import SUCCESS_TERMINATIONS, check_run
from kullback.runner.gate_support import gate
from kullback.runner.records import GateResult, Run, Verifier, VerifierHistory, VerifierVersion


def finished_run_ids(task_id: str, replays: dict, rerolls: dict) -> list[str]:
    """The Runs of one Task the frontier finished, in the order the records hold them: the confirmed
    replays (replays.json rows with `confirmed`) then the re-roll rows with a success termination."""
    out: list[str] = []
    for _, row in sorted(((replays or {}).get(task_id) or {}).items()):
        if row.get("confirmed") and row.get("run_id") and row["run_id"] not in out:
            out.append(str(row["run_id"]))
    for row in (rerolls or {}).get(task_id) or []:
        if (row.get("termination_reason") or "") in SUCCESS_TERMINATIONS and row.get("run_id") \
                and row["run_id"] not in out:
            out.append(str(row["run_id"]))
    return out


def legitimate_runs(replays: dict, rerolls: dict) -> dict[str, set[str]]:
    """Per Task, the run_ids a new version may newly pass: confirmed replays plus finished re-rolls.

    `rerolls` is the merged dict (the Builder's rows plus the Examiner's examiner/rerolls.json
    rows), so the pool grows with any round (D133).
    """
    tasks = sorted(set(replays or {}) | set(rerolls or {}))
    return {task_id: set(finished_run_ids(task_id, replays, rerolls)) for task_id in tasks}


def as_history(obj: Any) -> VerifierHistory:
    return obj if isinstance(obj, VerifierHistory) else VerifierHistory.model_validate(obj)


def accepted_versions(history: VerifierHistory) -> list[VerifierVersion]:
    """The accepted rows in order; the last one is the current version."""
    return [v for v in as_history(history).versions if v.accepted]


def newly_passed(previous: Verifier, current: Verifier, runs: list[Run], canon_rules: Any,
                 write_tools: set[str]) -> list[str]:
    """The run_ids `current` passes and `previous` failed."""
    previous, current = as_verifier(previous), as_verifier(current)
    out: list[str] = []
    for run in runs or []:
        if check_run(current, run, canon_rules, write_tools=write_tools)[0] \
                and not check_run(previous, run, canon_rules, write_tools=write_tools)[0]:
            out.append(run.run_id)
    return out


def loosening_gate(history: dict[str, VerifierHistory], task_runs: dict[str, list[Run]], replays: dict,
                   rerolls: dict, canon_rules: Any, sigs: list) -> GateResult:
    """Per Task, the newest version against the last accepted version before it: a failure per Run it
    newly passes that is not in the legitimate pool. A Task with no earlier accepted version passes."""
    write_tools = write_tools_of(sigs)
    legitimate = legitimate_runs(replays, rerolls)
    failures: list[str] = []
    loosened: dict[str, list[str]] = {}
    compared = 0
    for task_id, hist in sorted((history or {}).items()):
        versions = as_history(hist).versions
        if not versions:
            continue
        newest = versions[-1]
        earlier = [v for v in versions[:-1] if v.accepted]
        if not earlier:
            continue
        compared += 1
        for run_id in newly_passed(earlier[-1].verifier, newest.verifier, (task_runs or {}).get(task_id, []),
                                   canon_rules, write_tools):
            if run_id in legitimate.get(task_id, set()):
                continue
            failures.append(f"task {task_id}: version {newest.content_hash} newly passes {run_id}, which is "
                            "not the Reference, a frontier re-roll or a production Run")
            loosened.setdefault(task_id, []).append(run_id)
    return gate("loosening", failures, tasks=len(history or {}), compared=compared, loosened=loosened,
                legitimate={task_id: len(ids) for task_id, ids in legitimate.items()})


def false_rejection(verifier: Verifier, runs: list[Run], legitimate: set[str], canon_rules: Any,
                    write_tools: set[str]) -> dict:
    """Over the legitimate Runs of the Task not among the Verifier's seeds: how many it rejects.

    `fraction` is None when nothing is held out; an empty sample is not a zero rate. The Hard atoms
    are left out of the scoring: a Hard constraint is the policy over the Run, and a frontier Run
    that broke it is rightly rejected. Over-strictness is the required atoms recognising no path
    but their seeds, which is what D133 measures.
    """
    verifier = as_verifier(verifier)
    seeds = set(verifier.seed_run_ids)
    held = [run for run in runs or [] if run.run_id in legitimate and run.run_id not in seeds]
    required = verifier.model_copy(update={"atoms": [a for a in verifier.atoms if a.kind != "hard"]})
    rejected = [run.run_id for run in held
                if not check_run(required, run, canon_rules, write_tools=write_tools)[0]]
    fraction: Optional[float] = (len(rejected) / len(held)) if held else None
    return {"held_out": len(held), "rejected": len(rejected), "fraction": fraction, "rejected_ids": rejected}


def false_rejection_gate(verifiers: list[Verifier], task_runs: dict[str, list[Run]], replays: dict, rerolls: dict,
                         canon_rules: Any, sigs: list) -> GateResult:
    """The per-Task false-rejection number (D133); a failure only for a Verifier that rejects every
    held-out frontier Run, the single-path Verifier, and no tuned threshold otherwise."""
    write_tools = write_tools_of(sigs)
    legitimate = legitimate_runs(replays, rerolls)
    per_task: dict[str, dict] = {}
    failures: list[str] = []
    for verifier in sorted(map(as_verifier, verifiers or ()), key=lambda v: v.task_id):
        task_id = verifier.task_id
        row = false_rejection(verifier, (task_runs or {}).get(task_id, []), legitimate.get(task_id, set()),
                              canon_rules, write_tools)
        row["version"] = version_hash(verifier)
        per_task[task_id] = row
        if row["held_out"] >= 1 and row["fraction"] == 1.0:
            failures.append(f"task {task_id}: the required atoms reject every held-out frontier Run")
    return gate("false_rejection", failures, per_task=per_task)
