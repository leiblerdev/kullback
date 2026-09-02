"""The small helpers every gate in kullback/gates/ is built from: one GateResult shape, field
access that works on a record or its JSON dict, and equality after canonicalization (D39, D80)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from kullback.runner import canon
from kullback.runner.records import GateResult

MISS_REASONS = ("our_bug", "reference_bug", "ambiguous")


def gate(stage: str, failures, **metrics) -> GateResult:
    """One gate result: it passes when nothing failed."""
    failures = list(failures)
    return GateResult(stage=stage, passed=not failures, metrics=metrics, failures=failures)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off a record or off the plain dict the same record becomes in JSON."""
    if isinstance(obj, dict):
        return obj[name] if name in obj else obj.get(name.rstrip("_"), default)
    return getattr(obj, name, default)


def _passed(obj: Any) -> bool:
    """The `pass` field of a Verdict or a reference entry, under either spelling."""
    if isinstance(obj, dict):
        return bool(obj.get("pass", obj.get("passed")))
    return bool(getattr(obj, "passed", False))


def _same(a: Any, b: Any, column_class: str = "hard", canon_rules: Any = None,
          equivalence: Any = None, column: str = "") -> bool:
    """Equality after canonicalization (D39), so a gate and a Verdict agree by construction.

    The customer's own CanonRules and EquivalenceTable are what the Verdict compares under, so a
    gate given neither compares under the module defaults and can call two values different that
    the Verdict calls the same. No judge is ever passed: a gate calls no model (D84, D91).
    """
    return canon.equal(a, b, column_class, rules=canon_rules, table=equivalence, column=column)


def _share(part: int, total: int) -> Optional[float]:
    """A rate, or None when there is nothing to rate; an empty sample is not 100%."""
    return None if not total else round(part / total, 4)


def _rate(total: int, matched: int, misses: list) -> dict:
    """Raw and explained side by side, with every miss and its reason (D80)."""
    explained = [m for m in misses if m.get("reason") in MISS_REASONS]
    return {
        "total": total, "matched": matched,
        "raw": _share(matched, total),
        "explained": _share(matched + len(explained), total),
        "explained_misses": len(explained),
        "unexplained": len(misses) - len(explained),
        "by_reason": dict(Counter(m["reason"] for m in explained)),
        "misses": misses,
    }


def _checks_gate(stage: str, required: tuple, results: Optional[dict]) -> GateResult:
    """A gate whose evidence is a set of named checks run elsewhere; a check not run is a failure."""
    results = results or {}
    failures = [f"{name}: {'failed' if name in results else 'not run'}" for name in required if not results.get(name)]
    return gate(stage, failures, checks={name: bool(results.get(name)) for name in required})


def _n(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, set, dict)) else int(value or 0)
