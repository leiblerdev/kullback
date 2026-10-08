"""Trust gate 2: the Spec's Verifier must fail Runs that are wrong (DESIGN.md, the trust bar).

The mutated Runs are the D79 suite's own: the empty Run, the Reference with its required write aimed
at another entity, and one Run per demanded value swapped for another value of the same field. The
suite is called with the Spec's Verifier in place of the derived one; nothing here copies it.

A Verifier with gates (expected end states, forbidden, conduct, D316) is judged by the Verdict, which
reads those gates first; its mutations are the empty Run and, per expected cell, the Reference with
that cell left at its start value, and with a forbidden list the empty Run plus one forbidden call.
Each must fail. A refusal or no-write Verifier (no cell, something forbidden) drops the empty Run,
which rightly passes it.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates import verifier_suite
from kullback.runner.records import Run, Verifier
from kullback.runner.target import check_run
from kullback.runner.verdict import verdict

# The suite's checks that are mutations of the Reference, each expected to fail.
CAN_FAIL_STAGES = ("verifier_empty_run", "verifier_wrong_run", "verifier_forbidden_run")


class CanFailResult(NamedTuple):
    passed: bool
    rows: list[dict]


def can_fail(verifier: Verifier, reference_run: Any, world: Optional[Iterable[Any]] = None, *,
             canon: Any = None, write_tools: Optional[Iterable[str]] = None) -> CanFailResult:
    """Does every mutated Run fail? `world` is other Runs of the Task, which lend swap values.

    A check with no mutated Run to score is not a pass: a Verifier that demands nothing has no
    wrong Run, and the empty Run passes it, so it cannot fail.
    """
    if has_gates(verifier):
        return _gated_can_fail(verifier, reference_run, canon, write_tools)
    wrong = verifier_suite.wrong_run(verifier, reference_run, canon)
    gates = verifier_suite.validate_verifier(verifier, reference_run, wrong_run=wrong, canon=canon,
                                             write_tools=write_tools, seed_runs=list(world or ()))
    rows = [{"stage": gate.stage, "passed": bool(gate.passed), "failures": list(gate.failures),
             "swaps": gate.metrics.get("swaps", 0)}
            for gate in gates if gate.stage in CAN_FAIL_STAGES]
    return CanFailResult(passed=bool(rows) and all(row["passed"] for row in rows), rows=rows)


def has_gates(verifier: Verifier) -> bool:
    return bool(verifier.expected or verifier.forbidden or verifier.conduct)


def judge_run(verifier: Verifier, run: Any, canon: Any = None,
              write_tools: Optional[Iterable[str]] = None) -> tuple[bool, Any]:
    """(passed, the failing atom or gate): the Verdict for a gated Verifier, else the atoms alone."""
    if not has_gates(verifier):
        passed, atom = check_run(verifier, run, canon, write_tools=write_tools)
        return bool(passed), atom
    result = verdict(run, verifier, canon, write_tools=write_tools)
    return bool(result.passed), result.failing_atom or (None if result.passed else "not verdicted")


def _cell_mutations(verifier: Verifier, reference: Run) -> list[tuple[str, Run]]:
    """The Reference once per expected cell, that cell put back to its start value."""
    stops = [i for i, event in enumerate(reference.events) if event.type == "stop"]
    if not stops:
        return []
    payload = reference.events[stops[-1]].payload or {}
    start, end = payload.get("start_state") or {}, payload.get("end_state") or {}
    out = []
    for state in verifier.expected:
        for cell in state.cells:
            before_row = (start.get(cell.table) or {}).get(cell.row_id)
            after_row = (end.get(cell.table) or {}).get(cell.row_id)
            mutated = copy.deepcopy(end)
            rows = mutated.setdefault(cell.table, {})
            if cell.field is None:
                if before_row == after_row:
                    continue
                if before_row is None:
                    rows.pop(cell.row_id, None)
                else:
                    rows[cell.row_id] = copy.deepcopy(before_row)
            else:
                before = (before_row or {}).get(cell.field)
                if not isinstance(after_row, dict) or before == after_row.get(cell.field):
                    continue
                rows[cell.row_id][cell.field] = copy.deepcopy(before)
            run = reference.model_copy(deep=True)
            run.run_id = f"{reference.run_id}.undo.{cell.table}.{cell.row_id}.{cell.field or 'row'}"
            run.events[stops[-1]].payload = {**payload, "end_state": mutated}
            out.append((run.run_id, run))
    return out


def _gated_can_fail(verifier: Verifier, reference_run: Any, canon: Any,
                    write_tools: Optional[Iterable[str]]) -> CanFailResult:
    reference = verifier_suite.as_run(reference_run)
    forbidden = verifier_suite.forbidden_run(verifier, reference) if verifier.forbidden else None
    mutated = [] if verifier_suite.forbids_only(verifier) else [
        ("verifier_empty_run", verifier_suite._empty_run(reference))]
    mutated += [("verifier_forbidden_run", forbidden)] if forbidden is not None else []
    mutated += [("verifier_undone_cell", run) for _, run in _cell_mutations(verifier, reference)]
    rows = []
    for stage, run in mutated:
        try:
            passed, _ = judge_run(verifier, run, canon, write_tools)
        except Exception as exc:  # a mutation the Verdict cannot read proves nothing
            rows.append({"stage": stage, "passed": False, "failures": [type(exc).__name__], "swaps": 0})
            continue
        rows.append({"stage": stage, "passed": not passed, "failures": [run.run_id] if passed else [],
                     "swaps": 0})
    return CanFailResult(passed=bool(rows) and all(row["passed"] for row in rows), rows=rows)
