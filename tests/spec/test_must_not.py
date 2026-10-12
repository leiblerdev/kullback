"""Tests for spec/must_not.py: the must-not side of an Intent-written Verifier.

The tools here are invented (`etch` is the covered write tool, `mend` the
uncovered one); no corpus tool, table or field appears. Runs are hand-built
with the shared event helpers; scoring goes through the real check_run, never
a mock of our own code.
"""

from __future__ import annotations

import inspect

from gates.verifier_fixtures import _ev, assistant, call, make_run, result, user
from kullback import claims
from kullback.gates.verifier_suite import atom_payload, make_atom
from kullback.runner.records import Verifier
from kullback.runner.target import check_run
from kullback.spec import must_not as M

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


def _gate(tools=TOOLS):
    """The claim gate alone, scored as measurement: never part of must_not output."""
    (atom,) = M.claim_gate_atoms(list(tools), [{"name": t, "kind": "write"} for t in tools])
    return atom


def _score_gate(run, *extra):
    """The gate scored with covering must checks, so only it can fail the Run."""
    verifier = Verifier(task_id="t9", atoms=list(extra) + [_gate()])
    return check_run(verifier, run, canon=lambda value: value, write_tools=set(TOOLS))


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
        assert atoms == [], "scope needs a quote; the claim gate ships separately"


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


def test_claim_gate_is_one_hard_atom_measured_apart():
    assert M.must_not([], None, [], TOOLS) == []
    (atom,) = M.claim_gate_atoms(list(TOOLS), [{"name": t, "kind": "write"} for t in TOOLS])
    assert atom.id == "claim_without_effect"
    assert atom.kind == "hard" and M.tier_of(atom) == "sanity"
    assert _because_of(atom) == {"claim_rule": "D228"}


def test_must_not_takes_checks_not_runs():
    assert list(inspect.signature(M.must_not).parameters) == [
        "verifier_checks", "intent", "policy", "tools"]


# --- the claim gate (D228) ----------------------------------------------------


def test_claim_with_no_effective_write_fails():
    assert _score_gate(_said("I mended the tablet.")) == (False, "claim_without_effect")


def test_claim_whose_write_was_refused_fails():
    run = make_run("rc2", [user("please mend the tablet")] + _refused(MEND) +
                   [assistant("I mended the tablet.")])
    assert _score_gate(run) == (False, "claim_without_effect")


def test_any_effective_write_answers_every_claim():
    run = make_run("rc3", [
        user("please etch the tablet"),
        call(ETCH, {"tablet": "t1"}),
        result({"tablet": "t1"}),
        assistant("I etched the tablet. I mended the tablet."),
    ])
    assert _score_gate(run, *_must_etch()) == (True, None)


def test_general_claim_is_no_claim():
    assert _score_gate(_said("Done.")) == (True, None)


def test_silence_passes_the_claim_gate():
    assert _score_gate(_said("Noted. Thanks.")) == (True, None)


def test_promises_and_questions_claim_nothing():
    assert _score_gate(_said("I will mend it. Did I mend it?")) == (True, None)


def test_claim_failure_names_the_gate_never_the_sentence():
    passed, failure = _score_gate(_said("I mended the tablet."))
    assert (passed, failure) == (False, "claim_without_effect")
    assert "mended" not in failure
    assert "mended" not in (_gate().description or "")


def test_descriptions_mine_the_vocabulary():
    described = [{"name": ETCH, "kind": "write",
                  "description": "Carve a keepsake into the archive."},
                 {"name": MEND, "kind": "write", "description": "Fix what is broken."}]
    run = _said("I carved the tablet.")
    assert _score_gate(run) == (True, None), "carve names no bare-name tool"
    (atom,) = M.claim_gate_atoms([ETCH, MEND], described)
    verifier = Verifier(task_id="t9", atoms=[atom])
    assert check_run(verifier, run, canon=lambda value: value,
                     write_tools={ETCH, MEND}) == (False, "claim_without_effect")


def test_claim_matcher_matches_claims_py():
    vocabulary = claims.tool_vocabulary([{"name": tool, "kind": "write"} for tool in TOOLS])
    general = claims.general_stems()
    assert claims.claimed_tools("I etched the tablet.", vocabulary, general) == ["etch"]
    assert claims.claimed_tools("Done.", vocabulary, general) == ["etch", "mend"]
    assert claims.claimed_tools("I will mend it.", vocabulary, general) == []
    assert claims.claimed_tools("Noted.", vocabulary, general) == []
