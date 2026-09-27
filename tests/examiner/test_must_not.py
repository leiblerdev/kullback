"""Tests for examiner/must_not.py: the must-not side of an Intent-written Verifier.

The tools here are invented (`etch` is the covered write tool, `mend` the
uncovered one); no corpus tool, table or field appears. Runs are hand-built
with the shared event helpers; scoring goes through the real check_run, never
a mock of our own code.
"""

from __future__ import annotations

import inspect

from gates.verifier_fixtures import _ev, assistant, call, make_run, result, user
from kullback import claims
from kullback.examiner import must_not as M
from kullback.gates.verifier_suite import atom_payload, make_atom
from kullback.runner.records import Verifier
from kullback.runner.target import check_run

ETCH = "etch"
MEND = "mend"
TOOLS = [ETCH, MEND]
INTENT = "Etch the tablet for the archive. Do not mend anything."


def _must_etch() -> list:
    return [make_atom("w0", "required", {"kind": "write", "tool": ETCH})]


def _policy_rule(**overrides) -> dict:
    rule = {"id": "r1", "text": "never mend", "compiled": True,
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


def _claim_atoms():
    """The claim gate alone: no intent needed, the transcript is the evidence."""
    return [atom for atom in M.must_not([], None, [], TOOLS) if atom.id.startswith("claim.")]


def _score_only(run, tool):
    """One claim atom scored alone: a failure here names the claim atom itself."""
    (atom,) = [atom for atom in _claim_atoms() if atom.id == f"claim.{tool}"]
    return check_run(Verifier(task_id="t9", atoms=[atom]), run,
                     canon=lambda value: value, write_tools=set(TOOLS))


def _said(*lines, run_id="rc"):
    return make_run(run_id, [user("please etch the tablet"),
                             assistant(" ".join(lines))])


def _refused(tool, cid="c9"):
    return [call(tool, {"tablet": "t1"}, cid=cid),
            _ev("tool_result", id=cid, error={"message": "refused"})]
def test_tier_of_maps_required_writes_to_critical():
    atom = make_atom("w0", "required", {"kind": "write", "tool": ETCH})
    assert M.tier_of(atom) == "critical"


def test_tier_of_maps_forbidden_and_hard_to_sanity():
    assert M.tier_of(make_atom("f0", "forbidden", {"kind": "write", "tool": MEND})) == "sanity"
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
    passed, failure = check_run(verifier, _run_writing(MEND),
                                canon=lambda value: value, write_tools=set(TOOLS))
    assert (passed, failure) == (False, "forbidden.mend")


def test_a_write_through_the_covered_tool_passes():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    verifier = Verifier(task_id="t9", atoms=_must_etch() + atoms)
    assert check_run(verifier, _run_writing(ETCH),
                     canon=lambda value: value, write_tools=set(TOOLS)) == (True, None)


def test_a_tool_the_must_checks_cover_gets_no_forbidden_atom():
    atoms = M.must_not(_must_etch(), INTENT, [], TOOLS)
    assert [atom.id for atom in atoms if atom.kind == "forbidden"] == ["forbidden.mend"]


def test_allowed_checks_cover_a_tool_too():
    allowed = [make_atom("a0", "allowed", {"kind": "write", "tool": MEND})]
    atoms = M.must_not(_must_etch() + allowed, INTENT, [], TOOLS)
    assert [atom for atom in atoms if atom.kind == "forbidden"] == []


def test_a_partly_covered_tool_gets_no_whole_tool_atom():
    partly = [make_atom("w0", "required",
                        {"kind": "write", "tool": MEND, "entity": "tablet-1"})]
    atoms = M.must_not(partly, INTENT, [], [MEND])
    assert [atom for atom in atoms if atom.kind == "forbidden"] == [], \
        "a whole-tool ban would also fail the required row"


def test_no_intent_text_means_no_scope_atoms():
    for intent in ("", None):
        atoms = M.must_not(_must_etch(), intent, [], TOOLS)
        assert [atom for atom in atoms if atom.kind == "forbidden"] == []
        assert [atom.id for atom in atoms if atom.kind == "hard"] == ["claim.etch", "claim.mend"]


# --- one to one and find, do not invent ------------------------------------


def test_every_do_not_sentence_is_named_in_a_because():
    intent = "Etch the tablet for the archive. Do not mend anything. Never purge the stacks."
    atoms = M.must_not(_must_etch(), intent, [], TOOLS)
    named = [sentence for atom in atoms for sentence in _because_of(atom).get("intent", [])]
    assert "Do not mend anything." in named
def test_a_missing_record_sentence_is_the_because_for_creating():
    intent = "The tablet record is missing from the archive. Etch the index."
    atoms = M.must_not([], intent, [], TOOLS)
    scope = [atom for atom in atoms if atom.kind == "forbidden"]
    assert scope, "with no must coverage every tool is out of scope"
    for atom in scope:
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


def test_claim_gate_adds_only_hard_atoms():
    atoms = M.must_not([], None, [], TOOLS)
    assert sorted(atom.id for atom in atoms) == ["claim.etch", "claim.mend"]
    assert {atom.kind for atom in atoms} == {"hard"}
    assert {M.tier_of(atom) for atom in atoms} == {"sanity"}
    for atom in atoms:
        assert _because_of(atom) == {"claim_rule": "D228"}


def test_must_not_takes_checks_not_runs():
    assert list(inspect.signature(M.must_not).parameters) == [
        "verifier_checks", "intent", "policy", "tools"]


# --- the claim gate (D228) ----------------------------------------------------


def test_claim_with_no_call_of_that_tool_fails():
    run = make_run("rc1", [
        user("please etch the tablet"),
        call(ETCH, {"tablet": "t1"}),
        result({"tablet": "t1"}),
        assistant("I etched the tablet. I mended the tablet."),
    ])
    assert _score_only(run, MEND) == (False, "claim.mend")
    verifier = Verifier(task_id="t9", atoms=_must_etch() + _claim_atoms())
    assert check_run(verifier, run, canon=lambda value: value,
                     write_tools=set(TOOLS)) == (False, "claim.mend")


def test_claim_whose_write_was_refused_fails():
    run = make_run("rc2", [user("please mend the tablet")] + _refused(MEND) +
                   [assistant("I mended the tablet.")])
    assert _score_only(run, MEND) == (False, "claim.mend")


def test_claim_whose_write_took_effect_passes():
    run = make_run("rc3", [
        user("please etch and mend the tablet"),
        call(ETCH, {"tablet": "t1"}),
        result({"tablet": "t1"}),
        call(MEND, {"tablet": "t1"}, cid="c2"),
        result({"tablet": "t1"}, cid="c2"),
        assistant("I etched the tablet. I mended the tablet."),
    ])
    verifier = Verifier(task_id="t9", atoms=[
        make_atom("w0", "required", {"kind": "write", "tool": ETCH}),
        make_atom("w1", "required", {"kind": "write", "tool": MEND}),
    ] + _claim_atoms())
    assert check_run(verifier, run, canon=lambda value: value,
                     write_tools=set(TOOLS)) == (True, None)


def test_silence_passes_the_claim_gate():
    run = _said("Noted. Thanks.")
    assert _score_only(run, ETCH) == (True, None)
    assert _score_only(run, MEND) == (True, None)


def test_false_claim_about_an_unrequired_write_fails():
    assert _score_only(_said("I mended it."), MEND) == (False, "claim.mend")


def test_general_claim_answered_by_any_write_passes():
    run = make_run("rc4", [
        user("please etch the tablet"),
        call(ETCH, {"tablet": "t1"}),
        result({"tablet": "t1"}),
        assistant("Done."),
    ])
    verifier = Verifier(task_id="t9", atoms=_must_etch() + _claim_atoms())
    assert check_run(verifier, run, canon=lambda value: value,
                     write_tools=set(TOOLS)) == (True, None)


def test_general_claim_with_no_write_at_all_fails():
    assert _score_only(_said("Done."), ETCH) == (False, "claim.etch")
    assert _score_only(_said("Done."), MEND) == (False, "claim.mend")


def test_promises_and_questions_claim_nothing():
    run = _said("I will mend it. Did I mend it?")
    assert _score_only(run, MEND) == (True, None)


def test_claim_failure_names_the_tool_never_the_sentence():
    passed, failure = _score_only(_said("I mended the tablet."), MEND)
    assert (passed, failure) == (False, "claim.mend")
    assert "mended" not in failure
    (atom,) = [atom for atom in _claim_atoms() if atom.id == "claim.mend"]
    assert "mended" not in (atom.description or "")


def test_claim_matcher_matches_claims_py():
    vocabulary = claims.tool_vocabulary([{"name": tool, "kind": "write"} for tool in TOOLS])
    general = claims.general_stems()
    assert claims.claimed_tools("I etched the tablet.", vocabulary, general) == ["etch"]
    assert claims.claimed_tools("Done.", vocabulary, general) == ["etch", "mend"]
    assert claims.claimed_tools("I will mend it.", vocabulary, general) == []
    assert claims.claimed_tools("Noted.", vocabulary, general) == []
