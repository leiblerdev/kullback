"""Tests for atom_context.py: the environment one atom predicate is evaluated against."""

from __future__ import annotations

from kullback.runner.atom_context import AtomContext
from kullback.runner.confinement import SAFE_BUILTIN_NAMES, SAFE_BUILTINS
from kullback.runner.records import Run


def test_the_verdict_builtins_cover_everything_policy_certifies_at_build_time():
    """A predicate that passed its build-time test must not NameError here and be skipped."""
    import re

    from kullback.builder import policy

    block = re.search(r"_ALLOWED = \(\n(.*?)\n\)", policy._RUNNER_SRC, re.S).group(1)
    allowed = set(re.findall(r'"([A-Za-z_]+)"', block))
    assert allowed
    assert allowed <= set(SAFE_BUILTIN_NAMES)


def test_an_atom_cannot_change_the_builtins_the_next_atom_sees():
    """_evaluate copies the env one level deep, so handing out the module-level SAFE_BUILTINS would
    let one atom pop a name that every later atom, and every later Verdict in this process, misses.
    """
    context = AtomContext(Run(run_id="r1"))
    context.env()["__builtins__"].pop("len")
    assert "len" in SAFE_BUILTINS
    assert "len" in context.env()["__builtins__"]
