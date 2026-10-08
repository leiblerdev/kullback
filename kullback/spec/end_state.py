"""A Task's end-state gates, written by the Spec from its Reference Run by code (D315, D320).

The expected end states, the forbidden list and the conduct rules of a Verifier come from here: the
end states and the forbidden list off the Reference Run (`runner/expected.py`), the conduct off the
demands the writer grounded. No model: the writer never sees the Run, and nothing here asks one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Optional

from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import alternatives_from_turns, expected_from_run, forbidden_from, refusals_of
from kullback.runner.records import (
    Conduct,
    EndState,
    Forbidden,
    Run,
    ValueSource,
    load_run_jsonl,
    read_json,
    run_path,
)
from kullback.spec.schema import Check, Spec

# The demands that become conduct rules, not atoms: a confirmation, a refusal, a hand-off.
CONDUCT_DEMANDS = ("refuse", "handoff")
REFERENCE_KINDS = ("recording", "reroll")


class Gates(NamedTuple):
    """What the Verifier gates on, and what could not be sourced (counted, never filled in)."""
    expected: list[EndState]
    forbidden: list[Forbidden]
    conduct: list[Conduct]
    reference: Optional[str]
    unsupported: int


def _rows(index: Any, task_id: str) -> list[dict]:
    rows = (index or {}).get(task_id) if isinstance(index, dict) else None
    rows = rows.values() if isinstance(rows, dict) else rows or []
    return [row for row in rows if isinstance(row, dict)]


def reference_run(workdir: Path, task_id: str) -> Optional[Run]:
    """The Task's first kept Reference as a Run (references.json), a recording before a re-roll; None if none."""
    root = Path(workdir)
    row = (read_json(root / "references.json", {}) or {}).get(task_id) or {}
    refs = [ref for ref in row.get("references") or [] if isinstance(ref, dict) and ref.get("run_id")]
    refs.sort(key=lambda ref: REFERENCE_KINDS.index(ref.get("kind")) if ref.get("kind") in REFERENCE_KINDS else 2)
    paths = {str(r["run_id"]): r["path"] for index in ("replays.json", "rerolls.json", "examiner/rerolls.json")
             for r in _rows(read_json(root / index, {}), task_id) if r.get("run_id") and r.get("path")}
    for ref in refs:
        path = paths.get(str(ref["run_id"]))
        if path and run_path(root, path).is_file():
            return load_run_jsonl(run_path(root, path))
    return None


def _fact_source(check: Check, spec: Spec) -> ValueSource:
    """A conduct's source: the user turn of the first fact the check answers, else the policy its because names."""
    facts = {fact.id: fact for fact in spec.intent.facts}
    fact = next((facts[f] for f in check.fact_ids if f in facts), None)
    if fact is not None:
        return ValueSource(kind="user_turn", ptr={"recording": fact.source.recording, "turn": fact.source.turn})
    return ValueSource(kind="policy", ptr={"clause": check.because})


def conduct_of(spec: Spec, refusals: list[dict]) -> tuple[list[Conduct], int]:
    """The conduct rules the grounded demands ask for, and how many could not be sourced.

    An ask demand with a confirm_tool is a confirmation before that tool's write; a refuse demand
    is the Reference's recorded refusal shown again (its message is the source); a handoff demand
    is the Run-ending call to its tool. A refuse demand with no recorded refusal is unsupported.
    """
    out: list[Conduct] = []
    unsupported = 0
    for check in spec.checks:
        demand = check.demand.get("demand")
        if demand == "ask" and check.demand.get("confirm_tool"):
            out.append(Conduct(kind="confirm_before_write", tool=check.demand["confirm_tool"],
                               source=_fact_source(check, spec)))
        elif demand == "handoff" and check.demand.get("tool"):
            out.append(Conduct(kind="handoff", tool=check.demand["tool"], source=_fact_source(check, spec)))
        elif demand == "refuse":
            wanted = check.demand.get("tool")
            refusal = next((r for r in refusals if wanted in (None, r.get("tool"))), None)
            if refusal is None:
                unsupported += 1
                continue
            out.append(Conduct(kind="refusal", tool=refusal.get("tool"), source=ValueSource(
                kind="tool_result", ptr={"event_idx": refusal.get("event_idx"), "tool": refusal.get("tool"),
                                         "message": refusal.get("message")})))
    return out, unsupported


def _unsupported(state: EndState) -> int:
    return sum(1 for cell in state.cells if cell.source is None or cell.row_source is None)


def gates_of(workdir: Path, spec: Spec, policy_text: str = "", canon: Any = None,
             reference: Optional[Run] = None) -> Gates:
    """The Verifier's gates for one Task from its Reference Run: none (empty lists) when it has no Reference."""
    run = reference if reference is not None else reference_run(workdir, spec.task_id)
    if run is None:
        conduct, unsupported = conduct_of(spec, [])
        return Gates([], [], conduct, None, unsupported)
    context = AtomContext(run, canon)
    turns = [text for _, text in context.user]
    recording = run.trace_id or run.run_id
    end = expected_from_run(run, turns, policy_text, canon, recording=recording)
    refusals = refusals_of(run)
    forbidden = forbidden_from(end, turns, refusals, start_state=context.start_state, recording=recording,
                               canon=canon)
    conduct, unsupported = conduct_of(spec, refusals)
    return Gates(alternatives_from_turns(end, turns, canon), forbidden, conduct, run.run_id,
                 _unsupported(end) + unsupported)


__all__ = ["CONDUCT_DEMANDS", "Gates", "conduct_of", "gates_of", "reference_run"]
