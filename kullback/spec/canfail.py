"""Trust gate 2: the Spec's Verifier must fail Runs that are wrong (DESIGN.md, the trust bar).

The mutated Runs are the D79 suite's own: the empty Run, the Reference with its required write aimed
at another entity, and one Run per demanded value swapped for another value of the same field. The
suite is called with the Spec's Verifier in place of the derived one; nothing here copies it.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

from kullback.gates import verifier_suite
from kullback.runner.records import Verifier

# The suite's checks that are mutations of the Reference, each expected to fail.
CAN_FAIL_STAGES = ("verifier_empty_run", "verifier_wrong_run")


class CanFailResult(NamedTuple):
    passed: bool
    rows: list[dict]


def can_fail(verifier: Verifier, reference_run: Any, world: Optional[Iterable[Any]] = None, *,
             canon: Any = None, write_tools: Optional[Iterable[str]] = None) -> CanFailResult:
    """Does every mutated Run fail? `world` is other Runs of the Task, which lend swap values.

    A check with no mutated Run to score is not a pass: a Verifier that demands nothing has no
    wrong Run, and the empty Run passes it, so it cannot fail.
    """
    wrong = verifier_suite.wrong_run(verifier, reference_run, canon)
    gates = verifier_suite.validate_verifier(verifier, reference_run, wrong_run=wrong, canon=canon,
                                             write_tools=write_tools, seed_runs=list(world or ()))
    rows = [{"stage": gate.stage, "passed": bool(gate.passed), "failures": list(gate.failures),
             "swaps": gate.metrics.get("swaps", 0)}
            for gate in gates if gate.stage in CAN_FAIL_STAGES]
    return CanFailResult(passed=bool(rows) and all(row["passed"] for row in rows), rows=rows)
