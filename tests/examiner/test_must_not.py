"""Tests for examiner/must_not.py: the must-not side of an Intent-written Verifier.

The tools here are invented (`etch` is the covered write tool, `shred` the
uncovered one); no corpus tool, table or field appears. Runs are hand-built
with the shared event helpers; scoring goes through the real check_run, never
a mock of our own code.
"""

from __future__ import annotations

import inspect

from gates.verifier_fixtures import assistant, call, make_run, result, user
from kullback.examiner import must_not as M
from kullback.gates.verifier_suite import atom_payload, make_atom
from kullback.runner.records import Verifier
from kullback.runner.target import check_run

ETCH = "etch"
SHRED = "shred"
TOOLS = [ETCH, SHRED]
INTENT = "Etch the tablet for the archive. Do not shred anything."


def _must_etch() -> list:
    return [make_atom("w0", "required", {"kind": "write", "tool": ETCH})]


def _policy_rule(**overrides) -> dict:
    rule = {"id": "r1", "text": "never shred", "compiled": True,
            "predicate_src": "def check():\n    return True\n", "judge_atom": False}
    rule.update(overrides)
    return rule


def _run_writing(tool: str, run_id: str = "r1"):
    """A Run that does the required etch, then one more write through `tool`."""
    return make_run(run_id, [
        user("please etch the tablet"),
        call(ETCH, {"tablet": "t1"}),
        result({"tablet": "t1"}),
        call(tool, {"tablet": "t1"}, cid="c2"),
        result({"tablet": "t1"}, cid="c2"),
        assistant("done"),
    ])

def _because_of(atom) -> dict:
    return atom_payload(atom).get("because") or {}

def test_tier_of_maps_required_writes_to_critical():
    atom = make_atom("w0", "required", {"kind": "write", "tool": ETCH})
    assert M.tier_of(atom) == "critical"


def test_tier_of_maps_forbidden_and_hard_to_sanity():
    assert M.tier_of(make_atom("f0", "forbidden", {"kind": "write", "tool": SHRED})) == "sanity"
    assert M.tier_of(make_atom("h0", "hard", {"kind": "hard", "constraint_id": "r1"})) == "sanity"


def test_tier_of_maps_questions_facts_and_maybes_to_important():
    assert M.tier_of(make_atom("q0", "question", {"kind": "question", "key": "k"})) == "important"
    assert M.tier_of(make_atom("c0", "communicate",
                               {"kind": "communicate", "value": "v"})) == "important"
    assert M.tier_of(make_atom("a0", "allowed", {"kind": "write", "tool": ETCH})) == "important"


def test_every_emitted_atom_maps_to_sanity():
    atoms = M.must_not(_must_etch(), INTENT, [_policy_rule()], TOOLS)
    assert atoms, "scope and policy each owe at least one atom here"
    assert {M.tier_of(atom) for atom in atoms} == {"sanity"}


# --- scope -----------------------------------------------------------------


def test_a_write_through_an_uncovered_tool_fails():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    verifier = Verifier(task_id="t9", atoms=_must_etch() + atoms)
    passed, failure = check_run(verifier, _run_writing(SHRED),
                                canon=lambda value: value, write_tools=set(TOOLS))
    assert (passed, failure) == (False, "forbidden.shred")


def test_a_write_through_the_covered_tool_passes():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    verifier = Verifier(task_id="t9", atoms=_must_etch() + atoms)
    assert check_run(verifier, _run_writing(ETCH),
                     canon=lambda value: value, write_tools=set(TOOLS)) == (True, None)


def test_a_tool_the_must_checks_cover_gets_no_forbidden_atom():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    assert [atom.id for atom in atoms] == ["forbidden.shred"]


def test_allowed_checks_cover_a_tool_too():
    allowed = [make_atom("a0", "allowed", {"kind": "write", "tool": SHRED})]
    atoms = M.must_not(_must_etch() + allowed, INTENT, [], TOOLS)
    assert atoms == []


def test_a_partly_covered_tool_gets_no_whole_tool_atom():
    partly = [make_atom("w0", "required",
                        {"kind": "write", "tool": SHRED, "entity": "tablet-1"})]
    atoms = M.must_not(partly, INTENT, [], [SHRED])
    assert atoms == [], "a whole-tool ban would also fail the required row"


def test_no_intent_text_means_no_scope_atoms():
    assert M.must_not(_must_etch(), "", [], TOOLS) == []
    assert M.must_not(_must_etch(), None, [], TOOLS) == []


# --- one to one and find, do not invent ------------------------------------


def test_every_do_not_sentence_is_named_in_a_because():
    intent = "Etch the tablet for the archive. Do not shred anything. Never purge the stacks."
    atoms = M.must_not(_must_etch(), intent, [], TOOLS)
    named = [sentence for atom in atoms for sentence in _because_of(atom).get("intent", [])]
    assert "Do not shred anything." in named
    assert "Never purge the stacks." in named


def test_a_missing_record_sentence_is_the_because_for_creating():
    intent = "The tablet record is missing from the archive. Etch the index."
    atoms = M.must_not([], intent, [], TOOLS)
    assert atoms, "with no must coverage every tool is out of scope"
    for atom in atoms:
        assert _because_of(atom) == {"intent": ["The tablet record is missing from the archive."]}


def test_because_quotes_come_from_the_intent_line():
    atoms = M.must_not([], INTENT, [_policy_rule()], TOOLS)
    for atom in atoms:
        for sentence in _because_of(atom).get("intent", []):
            assert sentence in INTENT
        rule = _because_of(atom).get("policy_rule")
        if rule is not None:
            assert rule == "r1"


def test_every_forbidden_atom_names_a_sentence_or_a_rule():
    atoms = M.must_not(_must_etch(), INTENT, [_policy_rule()], TOOLS)
    assert atoms
    for atom in atoms:
        assert _because_of(atom), f"{atom.id} names neither a sentence nor a rule"


# --- policy prohibitions ----------------------------------------------------


def test_compiled_policy_rules_are_reused_verbatim():
    rule = _policy_rule()
    (atom,) = M.must_not([], INTENT, [rule], [])
    assert atom.id == "hard.r1" and atom.kind == "hard"
    assert atom_payload(atom)["predicate_src"] == rule["predicate_src"]
    assert _because_of(atom) == {"policy_rule": "r1"}


def test_residual_policy_rules_are_skipped():
    residual = _policy_rule(compiled=False, judge_atom=False)
    assert M.must_not([], INTENT, [residual], []) == []


def test_judge_policy_rules_stay_judge_atoms():
    (atom,) = M.must_not([], INTENT, [_policy_rule(compiled=False, judge_atom=True)], [])
    assert atom.judge is True


def test_no_rules_added_by_hand():
    rules = [_policy_rule(id="r1"), _policy_rule(id="r2")]
    atoms = M.must_not([], INTENT, rules, [])
    assert sorted(atom.id for atom in atoms) == ["hard.r1", "hard.r2"]


# --- claims and the contract -------------------------------------------------


def test_claims_add_no_atoms():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    assert {atom.kind for atom in atoms} == {"forbidden"}


def test_must_not_takes_checks_not_runs():
    assert list(inspect.signature(M.must_not).parameters) == [
        "verifier_checks", "intent", "policy", "tools"]
