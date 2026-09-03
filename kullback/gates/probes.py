"""The monotone probe pool (D127) and the stop rule (D133), as code over a Task's pool and its Verifier.

Every probe ever written for a Task stays in its pool, and every version of the Task's Verifier,
derived or repaired, must score no pass on all of them: `probe_pool_gate` rescores the whole pool
against the current version, so a repair cannot buy a pass by dropping the attack that found it.
`probe_admission_gate` is the fuzzing paper's stopping rule: a Task closes to new probes once the
last `PROBE_STOP` probes written against the current version were already rejected by it, and an
accepted repair, being a new version, opens a new tail. Nothing here executes a Run: a probe is a
Run the Examiner wrote, scored with the suite's `check_run` under the Task's write tools.
"""

from __future__ import annotations

from typing import Any, Iterable

from kullback.gates.verifier_suite import check_run
from kullback.runner.gate_support import _get, gate
from kullback.runner.records import GateResult, ProbePool, Verifier, as_dict, content_hash

PROBE_STOP = 3


def version_hash(verifier: Any) -> str:
    """The content hash that names a Verifier version everywhere (the history, the pool, the rulings)."""
    return content_hash(as_dict(as_verifier(verifier)))


def as_verifier(obj: Any) -> Verifier:
    return obj if isinstance(obj, Verifier) else Verifier.model_validate(obj)


def as_pool(obj: Any) -> ProbePool:
    return obj if isinstance(obj, ProbePool) else ProbePool.model_validate(obj)


def write_tools_of(sigs: Iterable[Any]) -> set[str]:
    """The mined write tools, the authority `check_run` scores writes under."""
    return {str(_get(s, "name")) for s in sigs or () if _get(s, "kind") == "write"}


def current_verifiers(verifiers: Iterable[Any]) -> dict[str, Verifier]:
    """The current Verifier per Task; a later entry for the same Task replaces an earlier one."""
    return {v.task_id: v for v in map(as_verifier, verifiers or ())}


def probe_scores(verifier: Verifier, pool: ProbePool, canon_rules: Any, write_tools: set[str]) -> dict[str, bool]:
    """probe_id -> whether `check_run` passes the probe's Run under this Verifier."""
    verifier, pool = as_verifier(verifier), as_pool(pool)
    return {probe.probe_id: check_run(verifier, probe.run, canon_rules, write_tools=write_tools)[0]
            for probe in pool.probes}


def probe_pool_gate(verifiers: list[Verifier], probes: dict[str, ProbePool], canon_rules: Any,
                    sigs: list) -> GateResult:
    """One failure per probe the Task's current Verifier passes; every probe in the pool is scored, always."""
    write_tools = write_tools_of(sigs)
    current = current_verifiers(verifiers)
    failures: list[str] = []
    passing_ids: dict[str, list[str]] = {}
    total = probed = 0
    for task_id, pool in sorted((probes or {}).items()):
        pool = as_pool(pool)
        total += len(pool.probes)
        probed += int(bool(pool.probes))
        verifier = current.get(task_id)
        if verifier is None:
            continue
        scores = probe_scores(verifier, pool, canon_rules, write_tools)
        version = version_hash(verifier)
        for probe in pool.probes:
            if scores[probe.probe_id]:
                failures.append(f"task {task_id}: probe {probe.probe_id} ({probe.bug_class}) scores a pass "
                                f"on version {version}")
                passing_ids.setdefault(task_id, []).append(probe.probe_id)
    return gate("probe_pool", failures, probes=total, tasks_probed=probed, passing=len(failures),
                passing_ids=passing_ids)


def consecutive_failed(pool: ProbePool, verifier_hash: str) -> int:
    """The length of the tail of probes written against this version that it already rejected.

    A probe that scored a pass ends the tail, and so does a probe written against another version:
    the count is per version, so an accepted repair starts from zero.
    """
    count = 0
    for probe in reversed(as_pool(pool).probes):
        if probe.verifier_hash != verifier_hash or probe.scored_pass:
            break
        count += 1
    return count


def probe_admission_gate(probes: dict[str, ProbePool], verifiers: list[Verifier]) -> GateResult:
    """A Task is open to a new probe until `PROBE_STOP` consecutive probes were already rejected (D133)."""
    current = current_verifiers(verifiers)
    closed: list[str] = []
    failures: list[str] = []
    for task_id, pool in sorted((probes or {}).items()):
        verifier = current.get(task_id)
        if verifier is None:
            continue
        version = version_hash(verifier)
        if consecutive_failed(pool, version) >= PROBE_STOP:
            closed.append(task_id)
            failures.append(f"task {task_id}: probing stopped, the last {PROBE_STOP} probes were already "
                            f"rejected by version {version}")
    return gate("probe_admission", failures, closed=closed, stop=PROBE_STOP)
