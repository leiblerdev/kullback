"""The refuse ruling (the solvability judge as code, D128) and what a trusted Verifier is (D126).

A Task may be refused only when no frontier Run of it finished: no confirmed replay and no re-roll
of any round with a success termination. The rule reads re-rolls already paid for, costs no Run and
never awards a pass. A trusted Verifier passed the D79 suite (`task_status` says `verifier_passed`),
scores no pass on every probe in its pool, is the last accepted version of its history, loosens past
no frontier Run (the loosening gate is run again here over the history cut at that version, so an
accepted flag written without the gate is not trust, D127), its false rejection over the held-out
pool is under D133's threshold, and its Task is not refused; round_end's "Tasks with a trusted
Verifier" is this ruling's count.

D194 made the false-rejection number a step in the chain rather than a number riding beside it. It
was measured here from the start and never read: a Verifier whose required atoms reject every
held-out Run that reached the Reference recognises no path but its own seeds, so what it checks is
one path and not the Task, and calling that trusted overstated the count. A Task with nothing held
out is not evidence either way, so it keeps `no_pool` in `false_rejection_ruling` and is decided by
the other steps. Every ruling carries the fraction (`false_rejection`), the pool size
(`false_rejection_pool`) and that per-Task word, so a reader sees the denominator behind the rate.
"""

from __future__ import annotations

from typing import Any, Optional

from kullback.gates.loosening import (
    accepted_versions,
    as_history,
    discarded_runs,
    false_rejection,
    finished_run_ids,
    legitimate_runs,
    loosening_gate,
    over_strict,
)
from kullback.gates.probes import (
    as_pool,
    as_verifier,
    probe_scores,
    version_hash,
    write_tools_of,
)
from kullback.gates.verifier_suite import ALT_PATH_NOT_RUN, D79_STAGES
from kullback.runner.gate_support import _get, gate
from kullback.runner.records import GateResult, ProbePool, Run, Verifier, VerifierHistory

# A check the suite could not run, and why it had no input. The stage names the status row carries
# are the suite's; the trusted ruling speaks the D79 check names a person reads (D173).
NOT_RUN_REASON = {"second_path_passes": ALT_PATH_NOT_RUN}
# What the false-rejection step says about a Task that held nothing out: no held-out Run reached the
# Reference, so the number is not a zero rate and not a failure, it is an absent measurement (D194).
NO_POOL = "no_pool"


def finished_runs(task_id: str, replays: dict, rerolls: dict) -> list[str]:
    """The run_ids of the confirmed replays and the finished re-rolls of one Task."""
    return finished_run_ids(task_id, replays, rerolls)


def refuse_gate(refusals: dict[str, dict], replays: dict, rerolls: dict) -> GateResult:
    """A refusal is admitted only when no frontier Run of the Task finished; a failure names the Runs."""
    refused: list[str] = []
    rejected: list[str] = []
    failures: list[str] = []
    for task_id in sorted(refusals or {}):
        finished = finished_runs(task_id, replays, rerolls)
        if finished:
            rejected.append(task_id)
            failures.append(f"task {task_id}: the frontier finished it: {', '.join(finished)}")
        else:
            refused.append(task_id)
    return gate("refuse", failures, refused=refused, rejected=rejected)


def _reason_of(refusal: Any) -> str:
    return str(_get(refusal, "reason", "") or "")


def _suite_reason(row: Any, skipped: list[str]) -> str:
    """Why the suite did not pass, saying which checks had no input rather than treating a check
    nobody could run as a check the Verifier failed (D173).

    A single-Reference Task fails `second_path_passes` because there is no second path to score, not
    because the Verifier turned one away, and the two ask for different repairs: more Runs of the
    Task (the Examiner's `reroll`, then `derive`) against a looser Verifier. The rule is unchanged
    either way, so a single-path Task is still untrusted; what changes is what the reason says and
    what `checks_not_run` in the metrics lets a reader act on.
    """
    checks = _get(row, "checks", None) or {}
    failed = sorted(name for name, ok in checks.items() if not ok and name not in skipped)
    if not skipped:
        return "the D79 suite did not pass"
    named = ", ".join(f"{name} not run" + (f" ({NOT_RUN_REASON[name]})" if name in NOT_RUN_REASON else "")
                      for name in skipped)
    line = f"the D79 suite did not pass: {named}"
    return f"{line}; {', '.join(failed)} failed" if failed else line


def _is_accepted_version(verifier: Verifier, history: Optional[Any]) -> bool:
    """The current file is the last accepted row of the Task's history, hash for hash."""
    if history is None:
        return False
    accepted = accepted_versions(history)
    return bool(accepted) and accepted[-1].content_hash == version_hash(verifier)


def _loosens_past_the_frontier(task_id: str, history: Any, task_runs: dict, replays: dict, rerolls: dict,
                               canon_rules: Any, sigs: list) -> Optional[str]:
    """The loosening gate's first failure over the Task's history cut at its last accepted row, or None.

    The gate compares the newest row against the accepted one before it, so the history is cut at
    the current version: a rejected attempt after it is not what the file holds.
    """
    hist = as_history(history)
    last = max(i for i, row in enumerate(hist.versions) if row.accepted)
    cut = hist.model_copy(update={"versions": hist.versions[:last + 1]})
    ruling = loosening_gate({task_id: cut}, task_runs, replays, rerolls, canon_rules, sigs)
    return None if ruling.passed else ruling.failures[0]


def trusted_gate(task_status: dict, verifiers: list[Verifier], probes: dict[str, ProbePool],
                 history: dict[str, VerifierHistory], refusals: dict[str, dict], task_runs: dict[str, list[Run]],
                 replays: dict, rerolls: dict, canon_rules: Any, sigs: list) -> GateResult:
    """A failure per Task with a Verifier that is not trusted, with the first reason that holds."""
    write_tools = write_tools_of(sigs)
    # D133's number is over the Runs that did the job, so the discarded recordings are out of the
    # pool (D173); the loosening rule below reads the whole pool, which is a different question.
    legitimate = legitimate_runs(replays, rerolls, discarded_runs(task_status))
    admitted = refuse_gate(refusals, replays, rerolls).metrics["refused"]
    refused = {task_id: _reason_of((refusals or {})[task_id]) for task_id in admitted}
    trusted: list[str] = []
    untrusted: dict[str, str] = {}
    fractions: dict[str, Optional[float]] = {}
    pool_sizes: dict[str, int] = {}
    pool_says: dict[str, str] = {}
    not_run: dict[str, list[str]] = {}
    probes_passing = 0
    failures: list[str] = []
    for verifier in sorted(map(as_verifier, verifiers or ()), key=lambda v: v.task_id):
        task_id = verifier.task_id
        pool = (probes or {}).get(task_id)
        scores = probe_scores(verifier, as_pool(pool), canon_rules, write_tools) if pool is not None else {}
        passing = [probe_id for probe_id, ok in scores.items() if ok]
        probes_passing += len(passing)
        held = false_rejection(verifier, (task_runs or {}).get(task_id, []), legitimate.get(task_id, set()),
                               canon_rules, write_tools)
        fractions[task_id] = held["fraction"]
        pool_sizes[task_id] = held["held_out"]
        pool_says[task_id] = NO_POOL if not held["held_out"] else \
            f"{held['fraction']:.2f} of {held['held_out']} held-out Runs"
        row = (task_status or {}).get(task_id) or {}
        skipped = [D79_STAGES.get(stage, stage) for stage in (_get(row, "not_run", None) or [])]
        if skipped:
            not_run[task_id] = skipped
        if not _get(row, "verifier_passed", False):
            reason = _suite_reason(row, skipped)
        elif passing:
            reason = f"probe {passing[0]} scores a pass"
        elif not _is_accepted_version(verifier, (history or {}).get(task_id)):
            reason = f"version {version_hash(verifier)} is not an accepted version"
        elif (loosened := _loosens_past_the_frontier(task_id, history[task_id], task_runs or {}, replays or {},
                                                     rerolls or {}, canon_rules, sigs)) is not None:
            reason = f"version {version_hash(verifier)} loosens past the frontier: {loosened}"
        elif task_id in refused:
            reason = "the Task is refused"
        elif over_strict(held):
            reason = (f"false_rejection {pool_says[task_id]}: the required atoms reject every held-out Run that "
                      "reached the Reference, so the Verifier checks one path and not the Task")
        else:
            trusted.append(task_id)
            continue
        untrusted[task_id] = reason
        failures.append(f"task {task_id}: {reason}")
    return gate("trusted", failures, trusted=trusted, untrusted=untrusted, probes_passing=probes_passing,
                false_rejection=fractions, false_rejection_pool=pool_sizes, false_rejection_ruling=pool_says,
                refused=refused, checks_not_run=not_run)
