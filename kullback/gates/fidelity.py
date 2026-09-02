"""The replay fidelity bar, at the three grains it is held to (D39, D51, D80, D108).

A tool body clears fidelity when every recorded call replays the way the recording did, success
and error fidelity counted apart and held to 100% on recorded calls (`replay_fidelity_gate`). A
Task clears fidelity when at least one of its Traces, replayed through the built tools with the
Trace's own turns as the model and the user, reaches its End state: every write agreeing after
canonicalization and no read differing in substance (`reference_replay_gate`; the per-Run
`confirmed` verdict is the Runner's, written into `replays.json` by `runner/replay.py`, and this
gate rules over those records). Gate A of design section 6 (`oracle_replay_gate`) is the same bar
over the seed and held-out split, not wired into a stage yet.

Every function here is pure over the records it is given: no Run is executed, no file is read,
no model is called (D110, D122).
"""

from __future__ import annotations

from typing import Any

from kullback.runner.gate_support import MISS_REASONS, _get, _rate, _same, gate
from kullback.runner.records import GateResult

# --- the per-tool bar (compile_tools gate 5, D80) ---

def replay_match(call: Any, canon_rules: Any = None) -> bool:
    """Does the rebuilt tool answer a recorded call the way the recording did (errors by shape, D51)."""
    expected_error, actual_error = _get(call, "expected_error"), _get(call, "actual_error")
    if expected_error is not None:
        return actual_error is not None and _get(expected_error, "class_") == _get(actual_error, "class_")
    return actual_error is None and _same(_get(call, "expected"), _get(call, "actual"),
                                          canon_rules=canon_rules)


def replay_fidelity_gate(calls, canon_rules: Any = None) -> GateResult:
    """Gate 5: replay of recorded calls, success and error fidelity reported separately (R22 item 8, D80).

    `canon_rules` is the customer's canonicalization: a gate given none compares under the module
    defaults and can call two values different that the Verdict calls the same (gate_support.py).
    """
    failures, misses, per_tool = [], [], {}
    totals = {"success": [0, 0], "error": [0, 0]}
    for call in calls or ():
        kind = "error" if _get(call, "expected_error") is not None else "success"
        tool = _get(call, "tool", "?")
        slot = per_tool.setdefault(tool, {"success": {"total": 0, "matched": 0}, "error": {"total": 0, "matched": 0}})
        totals[kind][0] += 1
        slot[kind]["total"] += 1
        if replay_match(call, canon_rules):
            totals[kind][1] += 1
            slot[kind]["matched"] += 1
            continue
        held_out = bool(_get(call, "held_out", False))
        reason = _get(call, "reason")
        misses.append({"tool": tool, "kind": kind, "reason": reason, "held_out": held_out})
        if not held_out:
            failures.append(f"{tool}: a recorded {kind} call replays differently; D80 wants 100% on recorded calls")
        elif reason not in MISS_REASONS:
            failures.append(f"{tool}: a held-out {kind} call misses with no reason ({reason!r})")
    metrics = {kind: _rate(totals[kind][0], totals[kind][1], [m for m in misses if m["kind"] == kind])
               for kind in ("success", "error")}
    metrics["per_tool"] = per_tool
    return gate("compile_tools.replay_fidelity", failures, **metrics)


# --- the per-Task bar (the replay_reference stage, D108) ---

def summarize(replays: dict[str, dict[str, dict]]) -> dict:
    """The stage's numbers over every replay: Traces, confirmed, per Task, writes and reads."""
    rows = [r for per_task in replays.values() for r in per_task.values()]
    tasks_confirmed = sum(any(r["confirmed"] for r in per_task.values()) for per_task in replays.values())
    total = lambda key: sum(int((r.get("counts") or {}).get(key) or 0) for r in rows)  # noqa: E731
    return {"traces": len(rows), "confirmed": sum(bool(r["confirmed"]) for r in rows),
            "tasks": len(replays), "tasks_confirmed": tasks_confirmed,
            "writes": total("writes"), "writes_matched": total("writes_matched"),
            "reads": total("reads"), "reads_semantic": total("reads_semantic"),
            "reads_cosmetic": total("reads_cosmetic"), "unmade": total("unmade")}


def unconfirmed_reason(per_task: dict[str, dict]) -> str:
    """Why a Task has no Reference: the most common first reason across its replays."""
    firsts = [r["reasons"][0] for r in per_task.values() if r.get("reasons")]
    if not firsts:
        return "no Trace of the Task was replayed"
    return max(sorted(set(firsts)), key=firsts.count)


def reference_replay_gate(replays: dict[str, dict[str, dict]]) -> GateResult:
    """A Task clears fidelity when some Trace of it replays to its End state; the rest are named.

    `replays` is `replays.json`: per Task id, per Trace id, the `Replay` record `runner/replay.py`
    wrote. Design section 6: a Task none of whose Traces replay to their End state is rejected for
    that Task, which the Verifier stage turns into "not verdicted"; the build itself goes on, so
    the failures name Tasks and the metrics carry the stage's totals.
    """
    failures = [f"task {task_id}: {unconfirmed_reason(per_task)}"
                for task_id, per_task in sorted(replays.items())
                if not any(r["confirmed"] for r in per_task.values())]
    return gate("replay_reference", failures, **summarize(replays))


# --- Gate A over the seed and held-out split (design section 6, not wired yet) ---

def oracle_replay_gate(replays, canon_rules: Any = None, equivalence: Any = None) -> GateResult:
    """Replaying a Reference's own calls reaches its End state, seed and held-out counted apart (D39, D51).

    Not wired into any pipeline stage yet; belongs in build.py's Reference-derivation stage, next to
    where the replay records themselves are produced.
    """
    splits = {name: {"runs": 0, "writes": 0, "matched": 0, "semantic_mismatches": 0} for name in ("seed", "held_out")}
    failures = []
    for replay in replays or ():
        split = splits["held_out" if _get(replay, "held_out", False) else "seed"]
        run_id = _get(replay, "run_id", "?")
        split["runs"] += 1
        for write in _get(replay, "writes", []) or ():
            split["writes"] += 1
            if _same(_get(write, "expected"), _get(write, "actual"), canon_rules=canon_rules):
                split["matched"] += 1
            else:
                failures.append(f"{run_id}: a write does not match the Reference after canonicalization")
        for read in _get(replay, "semantic_reads", []) or ():
            if not _same(_get(read, "expected"), _get(read, "actual"), "semantic",
                         canon_rules=canon_rules, equivalence=equivalence,
                         column=_get(read, "column", "")):
                split["semantic_mismatches"] += 1
                failures.append(f"{run_id}: a semantic read does not match the Reference")
    return gate("gate_a_oracle_replay", failures, **splits)
