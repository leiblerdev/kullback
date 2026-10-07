"""How the Spec's Verifier scores one Run: the Verdict for a gated Verifier, else the atoms alone.

The D79 mutation suite that once stood here as trust's can-fail gate is retired (D333): trust now asks
two constructed Runs to fail (spec/trust.py), the do-nothing Run and the stray write.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.runner.records import Verifier
from kullback.runner.target import check_run
from kullback.runner.verdict import verdict


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
