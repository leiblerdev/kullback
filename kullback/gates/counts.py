"""The workdir counts as code: every count comes from a gate ruling.

Carried verbatim from kullback/gates/round_end.py (the round driver is gone): `round_counts`
is the state of a workdir read off the rulings, and `trusted_share`, `fidelity_rate` and
`goal_met` read the Builder stop rule ("every Task is trusted or refused") off those counts.
The round-window functions (`stalled`, `fidelity_flat`, `exit_for`, `done`) died with the rounds.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from kullback.gates.fidelity import reference_replay_gate
from kullback.gates.trust import trusted_gate
from kullback.runner.gate_support import _get
from kullback.runner.records import GateResult, ProbePool, Run, Verifier, VerifierHistory

GATE_COUNTS: tuple[str, ...] = ("fidelity", "trusted", "refused_count", "assisted_runs", "probes_passing")

# G31: the round goal as rates over the Task list of that round. Both are invented and
# await measurement; 1.0 says every Task of the round is trusted and clears fidelity.
FIDELITY_RATE_GOAL = 1.0
TRUSTED_SHARE_GOAL = 1.0

def _goal_rates(task_status: Any, clearing: set, trusted_ids: list) -> dict:
    """G31 rates over this round's Task list, not absolute counts.

    The numerators only count Tasks of this round, so a growing list cannot read as progress
    on its own. A round with no Tasks reads 1.0, since no Task is then left unmet."""
    task_ids = list((task_status or {}).keys())
    fidelity_tasks = len([task_id for task_id in task_ids if task_id in clearing])
    trusted_tasks = len([task_id for task_id in trusted_ids if task_id in (task_status or {})])
    total = len(task_ids)
    return {"fidelity_tasks": fidelity_tasks, "fidelity_rate": (fidelity_tasks / total) if total else 1.0,
            "trusted_tasks": trusted_tasks, "trusted_share": (trusted_tasks / total) if total else 1.0}

def round_counts(task_status: dict, verifiers: list[Verifier], probes: dict[str, ProbePool],
                 history: dict[str, VerifierHistory], refusals: dict[str, dict], task_runs: dict[str, list[Run]],
                 replays: dict, rerolls: dict, canon_rules: Any, sigs: list, *,
                 record: Optional[Callable[[GateResult], Any]] = None,
                 intents: Optional[dict] = None, workdir: Any = None) -> dict:
    """D126's counts for one round, each read off a ruling; `record`, when given, receives the two
    rulings computed here (replay_reference, trusted) so a driver can land them in its ledger.

    `intents` is what the Builder's Intent stage left, and the three D196 counts are read straight
    off it and off the status rows: how many Intents the strip touched, how many values it took out,
    and how many Task and column pairs the leak check found it had missed. The three say whether the
    strip is doing the work or the check still is, which is the only way to tell a strip that covers
    a corpus from one that covers the two lines someone looked at.

    `workdir`, when the caller has the path, turns on the trusted gate's seed provenance step (D281)."""
    fidelity_ruling = reference_replay_gate(replays or {})
    trusted_ruling = trusted_gate(task_status, verifiers, probes, history, refusals, task_runs, replays, rerolls,
                                  canon_rules, sigs, workdir=workdir)
    for ruling in (fidelity_ruling, trusted_ruling):
        if record is not None:
            record(ruling)
    clearing = {task_id for task_id, per_task in (replays or {}).items()
                if any(_get(r, "confirmed", False) for r in per_task.values())}
    trusted_ids = list(trusted_ruling.metrics["trusted"])
    refused = dict(trusted_ruling.metrics["refused"])
    with_reference = [task_id for task_id, row in (task_status or {}).items()
                      if _get(row, "reference_confirmed", False)]
    # D172: every Task the corpus gave is unfinished until it is trusted and clears fidelity, or is
    # refused; a Task with no Reference yet (a seed tool assisted, a disagreement not yet refused)
    # is the loop's remaining work, not a Task outside it. Reading `unfinished` over the Tasks with
    # a Reference alone closed a build on "done" at fidelity 0 of 183 with every tool assisted,
    # because no Task had a Reference to be unfinished.
    unfinished = [task_id for task_id in (task_status or {})
                  if not ((task_id in trusted_ids and task_id in clearing) or task_id in refused)]
    stripped = [list(_get(record_of, "stripped", []) or []) for record_of in (intents or {}).values()]
    counts = {
        "fidelity": len(replays or {}) - len(fidelity_ruling.failures),
        "tasks": len(task_status or {}),
        "tasks_with_reference": len(with_reference),
        "trusted": len(trusted_ids),
        "trusted_ids": trusted_ids,
        "refused": refused,
        "refused_count": len(refused),
        "assisted_runs": sum(1 for runs in (task_runs or {}).values() for run in runs if _get(run, "assisted", False)),
        "probes_passing": int(trusted_ruling.metrics["probes_passing"]),
        "false_rejection": dict(trusted_ruling.metrics["false_rejection"]),
        "unfinished": unfinished,
        # G31: rates over the Task list of this round (1.0 when there are no Tasks, since no
        # Task is then left unmet). New fields only; every count above reads as before.
        **_goal_rates(task_status, clearing, trusted_ids),
        # D196: the strip, and what it missed.
        "intents_stripped": sum(1 for values in stripped if values),
        "values_stripped": sum(len(values) for values in stripped),
        "leak_misses": sum(len(_get(row, "leak_columns", []) or []) for row in (task_status or {}).values()),
    }
    return counts

def _counts(entry: Any) -> dict:
    """The counts of one round, whether handed a bare counts dict, a RoundRecord or its dict form."""
    counts = getattr(entry, "counts", None)
    if counts is not None:
        return counts
    if isinstance(entry, dict) and "counts" in entry and "round" in entry:
        return entry["counts"]
    return entry

def fidelity_rate(counts: Any) -> Optional[float]:
    """This round's fidelity as a rate over its Task list, or None where no rate was recorded.

    Only counts recorded by the new `round_counts` carry a rate; anything older has no rate to
    read and the caller falls back to the absolute count. Where the numerators are present the
    rate is recomputed against the live Task count, because the driver can widen the count
    afterwards (D231 reads a round with no Examiner plan off the Builder's handover, so stored
    rates must never outlive their denominator). A round with no Tasks at all reads 1.0, since
    no Task is then left unmet."""
    counts = _counts(counts)
    tasks = counts.get("tasks")
    numerator = counts.get("fidelity_tasks")
    if isinstance(numerator, int) and tasks:
        return float(numerator) / int(tasks)
    if isinstance(counts.get("fidelity_rate"), (int, float)):
        return float(counts["fidelity_rate"])
    return None

def trusted_share(counts: Any) -> Optional[float]:
    """This round's trusted share over its Task list, or None where no share was recorded.

    Like `fidelity_rate`, recomputed against the live Task count where the numerators are
    present, so a widened D231 denominator cannot leave a perfect stored share behind."""
    counts = _counts(counts)
    tasks = counts.get("tasks")
    numerator = counts.get("trusted_tasks")
    if isinstance(numerator, int) and tasks:
        return float(numerator) / int(tasks)
    if isinstance(counts.get("trusted_share"), (int, float)):
        return float(counts["trusted_share"])
    return None

def goal_met(counts: Any, *, fidelity_rate_goal: float = FIDELITY_RATE_GOAL,
             trusted_share_goal: float = TRUSTED_SHARE_GOAL) -> bool:
    """True when the round met the goal: fidelity rate and trusted share at their goals.

    The goals ride as arguments (defaulting to the named constants) so a plan can pass the
    founder's measured targets later without another frozen change. Counts with no recorded
    rates read as before (True), so every caller that never saw a list is unaffected; a round
    with no Tasks is vacuously met."""
    counts = _counts(counts)
    rate, share = fidelity_rate(counts), trusted_share(counts)
    if rate is None or share is None:
        return True
    if not counts.get("tasks"):
        return True
    return rate >= fidelity_rate_goal and share >= trusted_share_goal


__all__ = ["FIDELITY_RATE_GOAL", "GATE_COUNTS", "TRUSTED_SHARE_GOAL", "fidelity_rate", "goal_met", "round_counts", "trusted_share"]
