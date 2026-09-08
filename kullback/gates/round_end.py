"""The round_end counts and the three exits as code (D126): every count comes from a gate ruling.

`round_counts` is the state of a workdir at the end of a round, read off the rulings: Tasks clearing
fidelity from the replay_reference ruling, Tasks with a trusted Verifier and the refusals from the
trusted ruling, assisted Runs from the Run records, probes that scored a pass from the pool. The
driver adds what only it knows (fallback compactions per agent, spend, findings, and whether the
target was built at all). `done` is D126's state taken literally over a round that built its target
(D166), `stalled` is `stall_rounds` consecutive rounds that moved no gate count in either direction,
and `exit_for` applies the exits in the order ceiling, done, stalled (gate counts still, or
fidelity flat for `fidelity_stall` rounds, D169), max_rounds. A Task with no Reference is
unfinished until it is refused (D172).
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
    # D172: every Task the corpus gave is unfinished until it is trusted and clears fidelity, or is
    # refused; a Task with no Reference yet (a seed tool assisted, a disagreement not yet refused)
    # is the loop's remaining work, not a Task outside it. Reading `unfinished` over the Tasks with
    # a Reference alone closed a build on "done" at fidelity 0 of 183 with every tool assisted,
    # because no Task had a Reference to be unfinished.
    unfinished = [task_id for task_id in (task_status or {})
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
    """D126's state, literal, over a round that built its target (D166): every Task is trusted and
    clears fidelity or is refused, and no probe passes.

    A round whose build failed a stage holds no Task with a Reference at all, so `unfinished` was
    empty and the literal reading called it done. That closed a build whose compile_tools stage had
    failed three times, on the exit "done", with fidelity 0 of 183. `built` False is never done,
    whatever the rest of the counts say; a round that does not carry the count reads as before.
    The same build, rebuilt, closed on "done" again at fidelity 0 with every tool assisted and no
    Task holding a Reference: `unfinished` now counts every Task without a ruling (D172), so a
    round is done only when the number is final, and a loop that cannot move ends on stalled.
    """
    counts = _counts(counts)
    if counts.get("built") is False:
        return False
    return not counts.get("unfinished") and int(counts.get("probes_passing", 0)) == 0


def stalled(rounds: list[dict], stall_rounds: int) -> bool:
    """True when the last `stall_rounds + 1` rounds moved none of GATE_COUNTS in either direction;
    a first round is never stalled."""
    stall_rounds = max(1, int(stall_rounds))
    if len(rounds) <= stall_rounds:
        return False
    window = [_counts(r) for r in rounds[-(stall_rounds + 1):]]
    return all(len({counts.get(key) for counts in window}) == 1 for key in GATE_COUNTS)


def fidelity_flat(rounds: list[dict], flat_rounds: int) -> bool:
    """True when the last `flat_rounds` rounds lifted fidelity above nothing the rounds before them
    had reached (D169). Fidelity is the count the Builder's repairs move first, and a build whose
    fidelity has not risen in that many rounds is spending on repairs that land nowhere, whatever the
    other counts do: build 12's model arm sat at fidelity 153 for six rounds while trusted wobbled
    between 48 and 50, which kept every count from being still and the stalled exit from firing."""
    flat_rounds = max(1, int(flat_rounds))
    if len(rounds) <= flat_rounds:
        return False
    counts = [_counts(r) for r in rounds]
    before = max(int(c.get("fidelity") or 0) for c in counts[:-flat_rounds])
    recent = max(int(c.get("fidelity") or 0) for c in counts[-flat_rounds:])
    return recent <= before


def exit_for(rounds: list[dict], stall_rounds: int, *, ceiling_reached: bool,
             exhausted: list[bool], all_rounds: Optional[list[dict]] = None,
             fidelity_stall: Optional[int] = None, max_rounds: Optional[int] = None) -> Optional[str]:
    """ceiling when the build ceiling was reached or the allowance was exhausted two rounds in a row,
    else done, else stalled (no gate count moved, or fidelity flat for `fidelity_stall` rounds), else
    max_rounds when `all_rounds` holds that many, else None.

    `rounds` is the tail since the last round that moved, which is what the gate-count stall reads;
    `all_rounds` is every round so far, which is what the fidelity window and the round cap read,
    since a round that moved a count sideways still counts against both (D169)."""
    exhausted = list(exhausted or [])
    history = list(all_rounds) if all_rounds is not None else list(rounds)
    if ceiling_reached or (len(exhausted) >= 2 and exhausted[-1] and exhausted[-2]):
        return "ceiling"
    if rounds and done(rounds[-1]):
        return "done"
    if stalled(rounds, stall_rounds):
        return "stalled"
    if fidelity_stall and fidelity_flat(history, fidelity_stall):
        return "stalled"
    if max_rounds and len(history) >= max_rounds:
        return "max_rounds"
    return None
