"""A Task's end-state gates, written by the Spec from its Reference Run by code (D315, D320).

The expected end states, the forbidden list and the conduct rules of a Verifier come from here: the
end states and the forbidden list off the Reference Run (`runner/expected.py`), the conduct off the
demands the writer grounded. No model: the writer never sees the Run, and nothing here asks one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

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
    # Hand-off and confirmation demands on a tool the Reference never called: not emitted, counted.
    unseen: int = 0


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


def called_tools(run: Optional[Run]) -> Optional[set[str]]:
    """The tools the Reference called without an error, or None when there is no Reference."""
    if run is None:
        return None
    return {call["name"] for call in AtomContext(run).calls if not call["error"]}


def unseen_demand(check: Check, called: Optional[Iterable[str]]) -> bool:
    """A hand-off or confirmation demand on a tool the Reference never called (None: no Reference, never unseen)."""
    demand = check.demand
    tool = demand.get("tool") if demand.get("demand") == "handoff" else (
        demand.get("confirm_tool") if demand.get("demand") == "ask" else None)
    return bool(tool) and called is not None and tool not in called


def conduct_of(spec: Spec, refusals: list[dict], called: Optional[set[str]] = None) -> tuple[list[Conduct], int]:
    """The conduct rules the grounded demands ask for, and how many are unseen.

    An ask demand with a confirm_tool is a confirmation before that tool's write; a refuse demand
    with a recorded refusal is that refusal shown again (its message is the source); a handoff demand
    is the Run-ending call to its tool. A refuse demand with no recorded refusal is a forbidden write
    (`forbidden_of`), not conduct. A hand-off or confirmation on a tool the Reference (`called`) never
    called is not emitted and counted unseen.
    """
    out: list[Conduct] = []
    unseen = 0
    for check in spec.checks:
        demand = check.demand.get("demand")
        if unseen_demand(check, called):
            unseen += 1
        elif demand == "ask" and check.demand.get("confirm_tool"):
            out.append(Conduct(kind="confirm_before_write", tool=check.demand["confirm_tool"],
                               source=_fact_source(check, spec)))
        elif demand == "handoff" and check.demand.get("tool"):
            out.append(Conduct(kind="handoff", tool=check.demand["tool"], source=_fact_source(check, spec)))
        elif demand == "refuse":
            refusal = _refusal_for(check, refusals)
            if refusal is not None:
                out.append(Conduct(kind="refusal", tool=refusal.get("tool"), source=ValueSource(
                    kind="tool_result", ptr={"event_idx": refusal.get("event_idx"), "tool": refusal.get("tool"),
                                             "message": refusal.get("message")})))
    return out, unseen


def _refusal_for(check: Check, refusals: list[dict]) -> Optional[dict]:
    wanted = check.demand.get("tool")
    return next((r for r in refusals if wanted in (None, r.get("tool"))), None)


def forbidden_of(spec: Spec, refusals: list[dict], write_tools: Iterable[str],
                 called: Optional[set[str]] = None) -> list[Forbidden]:
    """A refuse demand with no recorded refusal, and a no_write demand, as forbidden writes.

    One entry per tool: the demand's tool, else each write tool. A tool the Reference called is never
    forbidden, so the Reference passes its own forbidden list. The row is the demand's entity, if any.
    """
    out: list[Forbidden] = []
    for check in spec.checks:
        demand = check.demand.get("demand")
        if demand not in ("refuse", "no_write") or (demand == "refuse" and _refusal_for(check, refusals)):
            continue
        tools = [check.demand["tool"]] if check.demand.get("tool") else sorted(write_tools)
        entity = check.demand.get("entity")
        row = str(entity) if isinstance(entity, (str, int)) and entity != "" else None
        out += [Forbidden(kind="write", tool=tool, row_id=row, source=_fact_source(check, spec))
                for tool in tools if tool not in (called or set())]
    return out


def _unsupported(state: EndState) -> int:
    return sum(1 for cell in state.cells if cell.source is None or cell.row_source is None)


def gates_of(workdir: Path, spec: Spec, policy_text: str = "", canon: Any = None,
             reference: Optional[Run] = None, write_tools: Iterable[str] = ()) -> Gates:
    """The Verifier's gates for one Task from its Reference Run: no end state when it has no Reference.

    A refuse demand with no recorded refusal and a no_write demand add forbidden writes over
    `write_tools` (`forbidden_of`), with or without a Reference.
    """
    run = reference if reference is not None else reference_run(workdir, spec.task_id)
    if run is None:
        conduct, unseen = conduct_of(spec, [])
        return Gates([], forbidden_of(spec, [], write_tools), conduct, None, 0, unseen)
    context = AtomContext(run, canon)
    turns = [text for _, text in context.user]
    recording = run.trace_id or run.run_id
    schema = read_json(Path(workdir) / "schema.json", None)
    end = expected_from_run(run, turns, policy_text, canon, recording=recording, schema=schema)
    refusals = refusals_of(run)
    forbidden = forbidden_from(end, turns, refusals, start_state=context.start_state, recording=recording,
                               canon=canon)
    called = {call["name"] for call in context.calls if not call["error"]}
    conduct, unseen = conduct_of(spec, refusals, called)
    forbidden += forbidden_of(spec, refusals, write_tools, called)
    return Gates(alternatives_from_turns(end, turns, canon), forbidden, conduct, run.run_id, _unsupported(end),
                 unseen)


__all__ = ["CONDUCT_DEMANDS", "Gates", "called_tools", "conduct_of", "forbidden_of", "gates_of", "reference_run",
           "unseen_demand"]
