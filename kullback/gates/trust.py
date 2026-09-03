"""The refuse ruling (the solvability judge as code, D128) and what a trusted Verifier is (D126).

A Task may be refused only when no frontier Run of it finished: no confirmed replay and no re-roll
of any round with a success termination. The rule reads re-rolls already paid for, costs no Run and
never awards a pass. A trusted Verifier passed the D79 suite (`task_status` says `verifier_passed`),
scores no pass on every probe in its pool, is the last accepted version of its history, loosens past
no frontier Run (the loosening gate is run again here over the history cut at that version, so an
accepted flag written without the gate is not trust, D127), and its Task is not refused; round_end's
"Tasks with a trusted Verifier" is this ruling's count, and the per-Task false-rejection number rides
in its metrics so the report can put it next to that count (D133).
"""

from __future__ import annotations

from typing import Any, Optional

from kullback.gates.loosening import (
    accepted_versions,
    as_history,
    false_rejection,
    finished_run_ids,
    legitimate_runs,
    loosening_gate,
)
from kullback.gates.probes import (
    as_pool,
    as_verifier,
    probe_scores,
    version_hash,
    write_tools_of,
)
from kullback.runner.gate_support import _get, gate
from kullback.runner.records import GateResult, ProbePool, Run, Verifier, VerifierHistory


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
    legitimate = legitimate_runs(replays, rerolls)
    admitted = refuse_gate(refusals, replays, rerolls).metrics["refused"]
    refused = {task_id: _reason_of((refusals or {})[task_id]) for task_id in admitted}
    trusted: list[str] = []
    untrusted: dict[str, str] = {}
    fractions: dict[str, Optional[float]] = {}
    probes_passing = 0
    failures: list[str] = []
    for verifier in sorted(map(as_verifier, verifiers or ()), key=lambda v: v.task_id):
        task_id = verifier.task_id
        pool = (probes or {}).get(task_id)
        scores = probe_scores(verifier, as_pool(pool), canon_rules, write_tools) if pool is not None else {}
        passing = [probe_id for probe_id, ok in scores.items() if ok]
        probes_passing += len(passing)
        fractions[task_id] = false_rejection(verifier, (task_runs or {}).get(task_id, []),
                                             legitimate.get(task_id, set()), canon_rules, write_tools)["fraction"]
        row = (task_status or {}).get(task_id) or {}
        if not _get(row, "verifier_passed", False):
            reason = "the D79 suite did not pass"
        elif passing:
            reason = f"probe {passing[0]} scores a pass"
        elif not _is_accepted_version(verifier, (history or {}).get(task_id)):
            reason = f"version {version_hash(verifier)} is not an accepted version"
        elif (loosened := _loosens_past_the_frontier(task_id, history[task_id], task_runs or {}, replays or {},
                                                     rerolls or {}, canon_rules, sigs)) is not None:
            reason = f"version {version_hash(verifier)} loosens past the frontier: {loosened}"
        elif task_id in refused:
            reason = "the Task is refused"
        else:
            trusted.append(task_id)
            continue
        untrusted[task_id] = reason
        failures.append(f"task {task_id}: {reason}")
    return gate("trusted", failures, trusted=trusted, untrusted=untrusted, probes_passing=probes_passing,
                false_rejection=fractions, refused=refused)
