"""The round_end counts and the three exits as code (D126): every count comes from a gate ruling.

`round_counts` is the state of a workdir at the end of a round, read off the rulings: Tasks clearing
fidelity from the replay_reference ruling, Tasks with a trusted Verifier and the refusals from the
trusted ruling, assisted Runs from the Run records, probes that scored a pass from the pool. The
driver adds what only it knows (fallback compactions per agent, spend, findings). `done` is D126's
state taken literally, `stalled` is `stall_rounds` consecutive rounds that moved no gate count in
either direction, and `exit_for` applies the three exits in the order ceiling, done, stalled.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from kullback.gates.fidelity import reference_replay_gate
from kullback.gates.trust import trusted_gate
from kullback.runner.gate_support import _get
from kullback.runner.records import GateResult, ProbePool, Run, Verifier, VerifierHistory

GATE_COUNTS: tuple[str, ...] = ("fidelity", "trusted", "refused_count", "assisted_runs", "probes_passing")


def round_counts(task_status: dict, verifiers: list[Verifier], probes: dict[str, ProbePool],
                 history: dict[str, VerifierHistory], refusals: dict[str, dict], task_runs: dict[str, list[Run]],
                 replays: dict, rerolls: dict, canon_rules: Any, sigs: list, *,
                 record: Optional[Callable[[GateResult], Any]] = None) -> dict:
    """D126's counts for one round, each read off a ruling; `record`, when given, receives the two
    rulings computed here (replay_reference, trusted) so a driver can land them in its ledger."""
    fidelity_ruling = reference_replay_gate(replays or {})
    trusted_ruling = trusted_gate(task_status, verifiers, probes, history, refusals, task_runs, replays, rerolls,
                                  canon_rules, sigs)
    for ruling in (fidelity_ruling, trusted_ruling):
        if record is not None:
            record(ruling)
    clearing = {task_id for task_id, per_task in (replays or {}).items()
                if any(_get(r, "confirmed", False) for r in per_task.values())}
    trusted_ids = list(trusted_ruling.metrics["trusted"])
    refused = dict(trusted_ruling.metrics["refused"])
    with_reference = [task_id for task_id, row in (task_status or {}).items()
                      if _get(row, "reference_confirmed", False)]
    unfinished = [task_id for task_id in with_reference
                  if not ((task_id in trusted_ids and task_id in clearing) or task_id in refused)]
    return {
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
    }


def _counts(entry: Any) -> dict:
    """The counts of one round, whether handed a bare counts dict, a RoundRecord or its dict form."""
    counts = getattr(entry, "counts", None)
    if counts is not None:
        return counts
    if isinstance(entry, dict) and "counts" in entry and "round" in entry:
        return entry["counts"]
    return entry


def done(counts: dict) -> bool:
    """D126's state, literal: every Task with a Reference is trusted and clears fidelity or is refused,
    and no probe passes."""
    counts = _counts(counts)
    return not counts.get("unfinished") and int(counts.get("probes_passing", 0)) == 0


def stalled(rounds: list[dict], stall_rounds: int) -> bool:
    """True when the last `stall_rounds + 1` rounds moved none of GATE_COUNTS in either direction;
    a first round is never stalled."""
    stall_rounds = max(1, int(stall_rounds))
    if len(rounds) <= stall_rounds:
        return False
    window = [_counts(r) for r in rounds[-(stall_rounds + 1):]]
    return all(len({counts.get(key) for counts in window}) == 1 for key in GATE_COUNTS)


def exit_for(rounds: list[dict], stall_rounds: int, *, ceiling_reached: bool,
             exhausted: list[bool]) -> Optional[str]:
    """ceiling when the build ceiling was reached or the allowance was exhausted two rounds in a row,
    else done, else stalled, else None."""
    exhausted = list(exhausted or [])
    if ceiling_reached or (len(exhausted) >= 2 and exhausted[-1] and exhausted[-2]):
        return "ceiling"
    if rounds and done(rounds[-1]):
        return "done"
    if stalled(rounds, stall_rounds):
        return "stalled"
    return None
