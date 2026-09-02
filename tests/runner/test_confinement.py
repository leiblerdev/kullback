"""Tests for confinement.py: the AST check that refuses a predicate reaching outside its own case."""

from __future__ import annotations

from kullback.runner.confinement import confine


def test_a_predicate_that_names_the_builtins_mapping_is_refused():
    """The mapping is the allowed builtins as data: naming it is enough to edit or empty them."""
    assert confine('__builtins__.pop("len") or True') == ["uses __builtins__"]
    assert confine('len(calls) == 1') == []
