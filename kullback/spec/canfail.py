"""How the Spec's Verifier scores one Run: the Verdict for a gated Verifier, else the atoms alone.

The D79 mutation suite that once stood here as trust's can-fail gate is retired (D333): trust now asks
two constructed Runs to fail (spec/trust.py), the do-nothing Run and the stray write.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

from kullback.runner.records import Verifier
from kullback.runner.target import check_run
from kullback.runner.verdict import verdict


def has_gates(verifier: Verifier) -> bool:
    return bool(verifier.expected or verifier.forbidden or verifier.conduct)


class Outcome(NamedTuple):
    """One Run under the Verifier: passed, the first gate item that holds False, and the sanity item's word.

    `failed` is a definite failure only: a Run left not verdicted (a gate item open) has neither
    passed nor failed. `sanity_failed` reads the sanity item off Verdict.items.
    """
    passed: bool
    failed: Optional[str]
    sanity_failed: bool


def outcome_of(verifier: Verifier, run: Any, canon: Any = None,
               write_tools: Optional[Iterable[str]] = None, schema: Any = None) -> Outcome:
    if not has_gates(verifier):
        passed, atom = check_run(verifier, run, canon, write_tools=write_tools)
        return Outcome(bool(passed), None if passed else str(getattr(atom, "id", atom) or "atom"), False)
    result = verdict(run, verifier, canon, write_tools=write_tools, schema=schema)
    definite = any(item.gate and item.holds is False for item in result.items)
    sanity = any(item.id == "sanity" and item.holds is False for item in result.items)
    return Outcome(bool(result.passed), result.failing_atom if definite else None, sanity)


def judge_run(verifier: Verifier, run: Any, canon: Any = None,
              write_tools: Optional[Iterable[str]] = None, schema: Any = None) -> tuple[bool, Any]:
    """(passed, the failing atom or gate): the Verdict for a gated Verifier, else the atoms alone."""
    if not has_gates(verifier):
        passed, atom = check_run(verifier, run, canon, write_tools=write_tools)
        return bool(passed), atom
    result = verdict(run, verifier, canon, write_tools=write_tools, schema=schema)
    return bool(result.passed), result.failing_atom or (None if result.passed else "not verdicted")
