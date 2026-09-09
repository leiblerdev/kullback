"""Tests for atom_context.py: the environment one atom predicate is evaluated against."""

from __future__ import annotations

import pytest

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


# --- D219: a comparison of one column has three answers, and a boolean carries two ---

def _well_context(before: str, after: str) -> AtomContext:
    """An invented well whose note is prose the schema classes semantic."""
    from kullback.runner.records import Column, EntitySchema

    schema = EntitySchema(tables=["wells"], columns=[
        Column(table="wells", name="well_id", **{"class": "hard"}),
        Column(table="wells", name="well_note", **{"class": "semantic"})])
    run = Run(run_id="r1", events=[{"idx": 0, "type": "stop", "payload": {
        "start_state": {"wells": {"W1": {"well_id": "W1", "well_note": before}}},
        "end_state": {"wells": {"W1": {"well_id": "W1", "well_note": after}}}}}])
    return AtomContext(run, schema=schema)


def test_resolution_answers_a_semantic_column_with_one_of_three_words():
    from kullback.runner import canon as canon_module

    context = _well_context("the pump is primed", "a primed pump")
    assert context.resolution("wells.well_note", "the pump is primed", "a primed pump") == "unresolved"
    assert context.resolution("wells.well_note", "primed", "primed") == "equal"
    assert context.resolution("wells.well_id", "W1", "W2") == "different"
    table = canon_module.EquivalenceTable()
    canon_module.put(table, "wells.well_note", canon_module.canon_value("the pump is primed"),
                     canon_module.canon_value("a primed pump"), True, classified_by="human")
    context.equivalence = table
    assert context.resolution("wells.well_note", "the pump is primed", "a primed pump") == "equal"


def test_same_stops_on_a_pair_nobody_settled_rather_than_answering_for_it():
    """The boolean has no room for the third answer, so inventing one is what let a forbidden-state
    atom written as same(column, forbidden, actual) never fire on an unsettled pair (D219)."""
    from kullback.runner.canon import Unresolved

    context = _well_context("the pump is primed", "a primed pump")
    with pytest.raises(Unresolved):
        context.same("wells.well_note", "the pump is primed", "a primed pump")
    with pytest.raises(Unresolved):
        context.changed("wells", "W1", "well_note")


def test_a_policy_says_which_way_an_unsettled_pair_counts_for_the_caller_that_states_one():
    context = _well_context("the pump is primed", "a primed pump")
    assert context.same("wells.well_note", "the pump is primed", "a primed pump", "equal") is True
    assert context.same("wells.well_note", "the pump is primed", "a primed pump", "different") is False
    assert context.changed("wells", "W1", "well_note", "equal") is False
    with pytest.raises(ValueError):
        context.same("wells.well_note", "the pump is primed", "a primed pump", "maybe")


def test_the_diff_shows_a_pair_nobody_settled_and_says_that_is_what_it_is():
    context = _well_context("the pump is primed", "a primed pump")
    fields = context.diff()["wells.W1"]["fields"]
    assert fields["well_note"]["unresolved"] is True
