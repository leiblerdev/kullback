"""Tests for kullback.gates.verifier_suite: reading Runs off disk, scoring one against a Verifier, and the D79 suite.

The derivation these Verifiers come from stays in the Builder (`derive` below reaches it); what is
tested here is the ruling side, which no agent may write and no model is consulted for (D122).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gates.verifier_fixtures import (
    ORDER,
    TASK,
    WRITE_TOOLS,
    alt_path_run,
    assistant,
    atom_by_id,
    call,
    derive,
    empty_run,
    extra_write_run,
    failed_run,
    make_run,
    other_reason_run,
    reference_events,
    reference_run,
    result,
    user,
    write_events_jsonl,
    write_run_json,
    write_run_with_footer,
    wrong_run,
)
from kullback.examiner import derive as V
from kullback.gates import artifacts
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules
from kullback.runner.confinement import confine
from kullback.runner.records import Constraint, Intent, StrippedValue, Task, UserRules, Verifier
from runner.replay_fixtures import Toolkit, do_replay

# --- reading re-runs from disk (D91) --------------------------------------

@pytest.mark.parametrize("write, make, run_id", [
    pytest.param(write_events_jsonl, reference_run, "ref", id="header_plus_event_lines"),
    pytest.param(write_run_json, alt_path_run, "alt", id="whole_run_on_one_line"),
])
def test_load_run_reads_a_run_written_as_event_lines_or_on_one_line(tmp_path, write, make, run_id):
    path = Path(write(make(), tmp_path / f"{run_id}.jsonl"))
    run = S.load_run(path)
    assert run.run_id == run_id
    assert len(run.events) == 8
    assert run.events[0].type == "user_turn"

def test_load_run_keeps_the_state_the_loop_footer_carries(tmp_path):
    """loop.py writes Start and End state on a footer line; a Hard rule reads them off the stop event."""
    path = write_run_with_footer(
        reference_run(), tmp_path / "footer.jsonl",
        run_id="delivered", task_id="t1", termination_reason="user_stop",
        start_state={"orders": {"#W123": {"status": "delivered"}}},
        end_state={"orders": {"#W123": {"status": "cancelled"}}})
    run = S.load_run(path)
    assert run.run_id == "delivered"
    assert run.termination_reason == "user_stop"
    assert S.start_state(run) == {"orders": {"#W123": {"status": "delivered"}}}
    assert [e.type for e in run.events][-1] == "stop"


def test_a_state_reading_hard_rule_fails_the_run_it_cannot_hold_on(tmp_path):
    """The Starting state comes off the footer, so 'never cancel a delivered order' can be checked.

    The gate has to let an ordinary policy predicate through as well as refuse a bad one, so the
    same source is certified here before either Run is scored.
    """
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    order = pre_state['orders'][write_call['arguments']['order_id']]\n"
                                     "    return order['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    assert confine(rule.predicate_src) == []

    def with_state(status: str) -> Path:
        return write_run_with_footer(reference_run(), tmp_path / f"{status}.jsonl", run_id=status,
                                     task_id="t1", termination_reason="user_stop",
                                     start_state={"orders": {"#W123": {"status": status}}})

    assert S.check_run(verifier, with_state("pending"), CanonRules()) == (True, None)
    assert S.check_run(verifier, with_state("delivered"), CanonRules()) == (False, "hard.k1")


def test_a_hard_rule_that_raises_is_a_defect_and_not_a_candidate_failure(tmp_path):
    """A predicate that blows up decided nothing: a Verifier defect, never a Candidate failure (G3)."""
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return pre_state['orders']['#W123']['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    assert S.hard_holds(atom_by_id(verifier, "hard.k1"), reference_run(), WRITE_TOOLS) is None
    assert S.check_run(verifier, reference_run(), CanonRules()) == (True, None)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), canon=CanonRules())}
    assert gates["verifier_oracle"].passed is False


def test_a_hard_rule_cannot_walk_dunders_out_of_the_sandbox(tmp_path):
    """A Hard constraint that never imports anything can still reach the process through
    ().__class__.__base__.__subclasses__(); _hard_holds has to refuse it before exec, the same way
    runner/verdict.py's gate() refuses it, or the walk runs for real and the Run scores a pass."""
    import subprocess  # noqa: F401  make sure Popen is a loaded subclass of object for the walk

    # the precondition the payload below rests on: the class it looks for really is reachable
    assert any(c.__name__ == "Popen" for c in ().__class__.__base__.__subclasses__())
    marker = tmp_path / "escaped.txt"
    escape = (
        "def check(pre_state, write_call, transcript):\n"
        "    for cls in ().__class__.__base__.__subclasses__():\n"
        "        if cls.__name__ == 'Popen':\n"
        f"            cls(['touch', {str(marker)!r}]).wait()\n"
        "            return True\n"
        "    return False\n"
    )
    rule = Constraint(id="k1", text="escape probe", compiled=True, predicate_src=escape)
    verifier = derive(tmp_path, constraints=[rule])
    atom = atom_by_id(verifier, "hard.k1")
    assert confine(atom.predicate_src)  # the walk is caught by the dunder-attribute gate
    assert S.hard_holds(atom, reference_run(), WRITE_TOOLS) is None
    assert not marker.exists(), "the escape payload ran outside the sandbox"
    assert S.check_run(verifier, reference_run(), CanonRules()) == (True, None)


def test_a_hard_rule_that_reaches_for_the_builtins_mapping_takes_nothing_from_the_next_rule(tmp_path):
    """Naming `__builtins__` reaches every allowed builtin as data to edit, so the source is refused
    before exec (a refused rule decided nothing, which is False here); and what does run gets its own
    copy of the mapping, so the rule scored next still has the allowlist it was certified against."""
    reaching = Constraint(id="k1", text="clears the builtins", compiled=True,
                          predicate_src=("def check(pre_state, write_call, transcript):\n"
                                         "    __builtins__.clear()\n"
                                         "    return True\n"))
    counts = Constraint(id="k2", text="never cancel without a reason", compiled=True,
                        predicate_src=("def check(pre_state, write_call, transcript):\n"
                                       "    return len(write_call['arguments'].get('reason') or '') > 0\n"))
    verifier = derive(tmp_path, constraints=[reaching, counts])
    assert S.hard_holds(atom_by_id(verifier, "hard.k1"), reference_run(), WRITE_TOOLS) is None
    assert S.hard_holds(atom_by_id(verifier, "hard.k2"), reference_run(), WRITE_TOOLS) is True


def test_a_hard_rule_is_not_asked_about_a_tool_the_seed_runs_only_read_with(tmp_path):
    """A before-write predicate judges writes and unknown tools, not the reads it was never written for."""
    rule = Constraint(id="k1", text="never touch a delivered order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['arguments'].get('order_id') != '#W123'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    reading = make_run("reading", [
        user("What is the status of #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("It is pending."),
    ])
    hard_only = Verifier(task_id="t1", atoms=[atom_by_id(verifier, "hard.k1")])
    assert S.check_run(hard_only, reading, CanonRules()) == (True, None)
    assert S.check_run(hard_only, reference_run(), CanonRules()) == (False, "hard.k1")

# --- the local scorer -----------------------------------------------------

def test_a_different_wording_of_an_allowed_value_passes(tmp_path):
    verifier = derive(tmp_path)
    reworded = make_run("reworded", reference_events(reason="I found it cheaper elsewhere"))
    assert S.check_run(verifier, reworded, CanonRules())[0] is True


def test_a_write_no_successful_run_made_still_fails_on_the_write_cap(tmp_path):
    """The stray write of a failed re-run is not an atom (D43); the cap is what stops a Candidate."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    over = make_run("over", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c2"),
        result({"status": "cancelled"}, cid="c2"),
    ])
    passed, failing = S.check_run(verifier, over, CanonRules())
    assert passed is False
    assert failing == "entity_count"


def test_an_extra_write_on_an_uncovered_tool_fails_given_write_tools(tmp_path):
    """Production always supplies the mined write_tools; no call in this file did until this test,
    so the extra-write safety net (_extra_write) had no coverage of that shape."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), extra_write_run()])
    sneaky = make_run("sneaky", reference_events() + [
        call("refund_order", {"order_id": "#W123", "amount": 25.0}, kind="read", cid="c2"),
        result({"ok": True}, cid="c2"),
    ])
    # With no write_tools, the extra call is invisible: nothing marked it as a write, so it slips
    # through as a full pass.
    assert S.check_run(verifier, sneaky, CanonRules()) == (True, None)
    # The same Run, scored the way production scores it, is caught.
    assert S.check_run(verifier, sneaky, CanonRules(), write_tools={"cancel_pending_order", "refund_order"}) == (
        False, "extra_write:refund_order")

# --- D79 validation -------------------------------------------------------

def test_the_suite_runs_the_d79_checks_and_the_specificity_check_in_order_and_passes_a_sound_verifier(
        tmp_path, test_model):
    verifier = derive(tmp_path)
    gates = S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()], canon=CanonRules())
    assert [g.stage for g in gates] == [
        "verifier_provenance_spans", "verifier_oracle", "verifier_empty_run",
        "verifier_wrong_run", "verifier_unfinished_run", "verifier_alt_path", "verifier_loophole",
        "verifier_leak", "verifier_mutation", "verifier_specific",
    ]
    assert all(g.passed for g in gates), [(g.stage, g.failures) for g in gates]


def test_the_unfinished_run_stops_just_before_the_required_write_and_must_fail(tmp_path):
    """D119: the Reference cut one step short (GLM 5.3's unsolved-state check) scores no pass."""
    verifier = derive(tmp_path)
    unfinished = S.unfinished_run(verifier, reference_run(), CanonRules())
    assert unfinished.run_id == "ref.unfinished" and unfinished.termination_reason == "max_turns"
    assert [e.type for e in unfinished.events][-2:] == ["tool_call", "tool_result"]  # the read, not the cancel
    assert not any(e.type == "tool_call" and e.payload.get("name") == "cancel_pending_order"
                   for e in unfinished.events)
    assert S.check_run(verifier, unfinished, CanonRules())[0] is False
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), canon=CanonRules())}
    assert gates["verifier_unfinished_run"].passed is True


def test_a_verifier_that_rewards_an_unfinished_run_is_caught(tmp_path):
    verifier = derive(tmp_path)
    # Only the write cap survives: a Verifier that checks the agent wrote nothing extra and never
    # that the cancellation happened, which a Run that stopped short satisfies.
    hollow = verifier.model_copy(update={"atoms": [a for a in verifier.atoms
                                                    if S.atom_payload(a).get("kind") == "entity_count"]})
    gates = {g.stage: g for g in S.validate_verifier(hollow, reference_run(), canon=CanonRules())}
    assert gates["verifier_unfinished_run"].metrics["run_passed"] is True
    assert gates["verifier_unfinished_run"].passed is False


def _answering_events(final: str = "That is all, have a good day.") -> list[dict]:
    """A Task whose outcome is the answer: the agent reads the order and states its status, twice."""
    return [
        user("What is the status of order #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Order #W123 is pending."),
        user("Thanks, anything else I should know?"),
        assistant("Nothing else on #W123."),
        user("Great, bye."),
        assistant(final),
    ]


def test_a_verifier_that_requires_no_write_is_cut_before_the_first_answer_stating_a_fact_it_asks_for():
    """The cut has to fall where the Run stops doing the job. For a Task whose outcome is the answer
    there is no write to cut before, and the facts are usually stated well before the closing turn:
    cutting before the last turn leaves the answer standing and the cut Run passes (D173)."""
    answering = make_run("answering", _answering_events())
    verifier = V.derive_verifier(TASK, answering, [], CanonRules(), write_tools=WRITE_TOOLS)
    assert [S.atom_payload(a)["text"] for a in verifier.atoms
            if S.atom_payload(a).get("kind") == "communicate"] == ["#W123"]
    unfinished = S.unfinished_run(verifier, answering, CanonRules())
    assert [e.type for e in unfinished.events] == ["user_turn", "tool_call", "tool_result"]
    assert S.check_run(verifier, unfinished, CanonRules())[0] is False
    gates = {g.stage: g for g in S.validate_verifier(verifier, answering, canon=CanonRules())}
    assert gates["verifier_unfinished_run"].passed is True


def test_a_writeless_run_is_cut_before_its_last_answer_and_an_empty_reference_has_no_cut(tmp_path):
    """No write to cut before and no fact the Verifier asks for: the final turn is all that is left
    to take away, and what the cut Run then fails on is the derivation's business, not the cut's."""
    derived = derive(tmp_path)
    assert S.unfinished_run(derived, empty_run(), CanonRules()) is None
    talk = make_run("talk", [user("hi"), assistant("hello")])
    assert [e.type for e in S.unfinished_run(derived, talk, CanonRules()).events] == ["user_turn"]

    silent = make_run("silent", [user("Are you there?"), assistant("I am here."), user("Good."),
                                 assistant("Goodbye.")])
    verifier = V.derive_verifier(TASK, silent, [], CanonRules(), write_tools=WRITE_TOOLS)
    assert not [a for a in verifier.atoms if S.atom_payload(a).get("kind") == "communicate"]
    unfinished = S.unfinished_run(verifier, silent, CanonRules())
    assert [e.type for e in unfinished.events] == ["user_turn", "model_call", "user_turn"]


def test_the_suite_reports_every_d79_check_by_the_name_the_verifier_gate_wants(tmp_path, test_model):
    """verifier_gate counts a check it was not told about as a failure, so the names have to line up;
    the mapping is the contract between the two modules and nothing imports it."""
    verifier = derive(tmp_path)
    gates = S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()], canon=CanonRules())
    results = S.d79_results(gates)
    assert sorted(results) == sorted(artifacts.D79_CHECKS)
    assert artifacts.verifier_gate(results).passed is True


def test_a_check_with_no_run_is_reported_as_not_run(tmp_path):
    """A Verifier nobody tried to break is not a validated Verifier; missing input is a failure."""
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), canon=CanonRules())}
    assert gates["verifier_wrong_run"].passed is False
    assert gates["verifier_alt_path"].passed is False
    assert gates["verifier_loophole"].passed is False
    assert all("not run" in " ".join(gates[stage].failures)
               for stage in ("verifier_wrong_run", "verifier_alt_path", "verifier_loophole"))
    assert artifacts.verifier_gate(S.d79_results(gates.values())).passed is False


def test_the_second_path_check_says_it_had_no_second_path_rather_than_that_one_failed(tmp_path):
    """A Task with one Reference has no second path to score, which is a different thing from a
    Verifier that turned one away and asks for a different repair: more Runs, not looser atoms
    (D173). The ruling still fails, and the reason it carries is the one a reader acts on."""
    verifier = derive(tmp_path)
    gate = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), canon=CanonRules())}["verifier_alt_path"]
    assert gate.passed is False
    assert gate.metrics["not_run_reason"] == S.ALT_PATH_NOT_RUN
    assert gate.failures == [f"not run: {S.ALT_PATH_NOT_RUN}"]
    given = {g.stage: g for g in S.validate_verifier(verifier, reference_run(),
                                                     alt_path_run=alt_path_run(), canon=CanonRules())}["verifier_alt_path"]
    assert given.passed is True and "not_run_reason" not in given.metrics


def test_a_check_with_no_input_says_not_run_and_a_check_that_ran_does_not(tmp_path):
    """F45: the line a check with no input carries is read as not run, never as a failure of the
    atoms, by whoever rules on a proposal; a check that ran and failed has no not-run reason."""
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), empty_run=reference_run(), canon=CanonRules())}
    assert S.not_run_reason(gates["verifier_alt_path"].failures) == S.ALT_PATH_NOT_RUN
    assert gates["verifier_empty_run"].passed is False
    assert S.not_run_reason(gates["verifier_empty_run"].failures) is None
    assert S.not_run_reason([]) is None


def test_a_required_atom_whose_payload_kind_check_run_does_not_score_fails_the_run(tmp_path):
    """F51: a required atom nothing scores used to pass every Run; it is a failure that names it."""
    verifier = derive(tmp_path)
    assert S.check_run(verifier, reference_run(), CanonRules(), write_tools=WRITE_TOOLS) == (True, None)
    for payload in ({"kind": "read", "tool": "look_up"}, {}):
        unscored = verifier.model_copy(update={"atoms": [*verifier.atoms, S.make_atom("r0", "required", payload)]})
        kind = payload.get("kind")
        assert S.check_run(unscored, reference_run(), CanonRules(), write_tools=WRITE_TOOLS) == (
            False, f"atom r0: kind {kind} is not scored")


def test_a_hard_atom_with_predicate_src_is_scored_on_every_call(tmp_path):
    verifier = derive(tmp_path)
    never = S.make_atom("h0", "hard", {"kind": "hard", "predicate_src": _NEVER_KEPT})
    ruled = verifier.model_copy(update={"atoms": [*verifier.atoms, never]})
    assert S.check_run(ruled, reference_run(), CanonRules(), write_tools=WRITE_TOOLS) == (False, "h0")


_NEVER_KEPT = "def check(pre_state, write_call, transcript):\n    return False\n"


def test_a_hollow_verifier_fails_the_suite_with_no_runs_supplied(tmp_path):
    """The empty Run needs no Runner, so check 3 always runs and an atomless Verifier cannot clear it."""
    hollow = Verifier(task_id="t1", atoms=[])
    gates = {g.stage: g for g in S.validate_verifier(hollow, reference_run(), canon=CanonRules())}
    assert gates["verifier_empty_run"].passed is False
    assert gates["verifier_empty_run"].metrics["run_passed"] is True
    assert any(not g.passed for g in gates.values())
    supplied = {g.stage: g for g in S.validate_verifier(hollow, reference_run(), empty_run(), wrong_run(),
                                                        alt_path_run(), intent_text="", user_rules=None, canon=CanonRules())}
    assert supplied["verifier_empty_run"].passed is False
    assert supplied["verifier_wrong_run"].passed is False


@pytest.mark.parametrize("break_span", [
    pytest.param(lambda atom: setattr(atom, "spans", []), id="no_span"),
    pytest.param(lambda atom: setattr(atom.spans[0], "msg_index", 3), id="wrong_turn"),
])
def test_a_user_stated_atom_whose_span_misses_the_value_fails_provenance(tmp_path, break_span):
    verifier = derive(tmp_path)
    for atom in verifier.atoms:
        if atom.provenance == "user_stated" and atom.spans:
            break_span(atom)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None, canon=CanonRules())}
    assert gates["verifier_provenance_spans"].passed is False
    assert gates["verifier_provenance_spans"].failures


def test_the_spans_gate_reads_a_rerun_span_in_the_run_it_names(tmp_path):
    """An atom only a re-run holds carries that re-run's span; resolving it elsewhere is a false hit."""
    extra = make_run("extra", [
        user("Please cancel my orders #W123 and #W888."),
        assistant("Sure. Why do you want to cancel them?"),
        user("moving abroad"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "moving abroad"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        call("cancel_pending_order", {"order_id": "#W888", "reason": "moving abroad"}, cid="c2"),
        result({"order_id": "#W888", "status": "cancelled"}, cid="c2"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    verifier = derive(tmp_path, reruns=[extra])
    w888 = [a for a in verifier.atoms
            if S.atom_payload(a)["kind"] == "write_value" and S.atom_payload(a)["field"] == "reason"
            and str(S.atom_payload(a)["entity"]).lower().endswith("w888")][0]
    assert w888.spans[0].file_hash == "extra"
    with_seed = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), seed_runs=[extra], canon=CanonRules())}
    assert with_seed["verifier_provenance_spans"].passed is True
    alone = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), canon=CanonRules())}
    assert alone["verifier_provenance_spans"].passed is False
    assert "not supplied" in " ".join(alone["verifier_provenance_spans"].failures)


def test_the_mutation_check_flags_an_atom_the_reference_cannot_fail(tmp_path):
    """D79 check 8: an atom whose mutation changes nothing is an atom that is not being checked."""
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    # A re-run wrote twice, so the cap is 2 while the Reference writes once: asking for at most 1
    # write is a smaller demand the Reference still meets, so the cap is checking nothing here.
    loose = derive(tmp_path, reruns=[alt_path_run(), extra_write_run()], constraints=[rule])
    gate = [g for g in S.validate_verifier(loose, reference_run(), canon=CanonRules()) if g.stage == "verifier_mutation"][0]
    assert gate.passed is False
    assert "entity_count" in " ".join(gate.failures)
    lively = [g for g in S.validate_verifier(derive(tmp_path, constraints=[rule]), reference_run(), canon=CanonRules())
              if g.stage == "verifier_mutation"][0]
    assert lively.passed is True
    assert lively.metrics["atoms_mutated"] >= 4


def test_a_zero_cap_has_no_mutant_and_is_not_counted_as_a_failed_flip(tmp_path):
    """A Verifier whose Reference wrote nothing caps writes at 0, and max(count - 1, 0) is that same
    cap: the Reference rightly still passes, and counting it as an atom that cannot be falsified
    failed 27 read-only Tasks of one build and 28 of another (D173)."""
    asked = make_run("asked", [
        user("What is the status of order #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Order #W123 is pending."),
        user("Thanks."),
        assistant("You are welcome."),
    ])
    verifier = V.derive_verifier(TASK, asked, [], CanonRules(), write_tools=WRITE_TOOLS)
    cap = atom_by_id(verifier, "entity_count")
    assert S.atom_payload(cap)["count"] == 0
    assert S._mutant(cap, S.canon_fn(CanonRules())) is None
    gate = [g for g in S.validate_verifier(verifier, asked, canon=CanonRules()) if g.stage == "verifier_mutation"][0]
    # The communicate atom is what carries this Verifier, and it flips, so the check passes.
    assert gate.passed is True, gate.failures
    assert gate.metrics == {"atoms_mutated": 1, "atoms_not_mutable": 1}


def test_a_hard_atom_that_judged_no_call_does_not_read_as_held(tmp_path):
    """A before-write rule over a Run with no write call was asked nothing, so it neither held nor
    failed; reading its True as "the constraint held" hid that the policy is unchecked there, and
    made the mutation check report a Verifier defect where no rule had ever run (D173)."""
    rule = Constraint(id="k1", text="never cancel a delivered order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['arguments'].get('order_id') != '#W123'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    hard = atom_by_id(verifier, "hard.k1")
    reading = make_run("reading", [
        user("What is the status of #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("It is pending."),
    ])
    assert S.hard_holds(hard, reading, WRITE_TOOLS) is None
    assert S.hard_holds(hard, reference_run(), WRITE_TOOLS) is False  # it was asked, and it failed
    # None is not a failure, so a Run the rule says nothing about is scored as it always was.
    assert S.check_run(Verifier(task_id="t1", atoms=[hard]), reading, CanonRules()) == (True, None)


def test_a_verifier_with_no_mutable_atom_fails_the_mutation_check_and_the_text_says_so(tmp_path):
    """The cap of 0 plus Hard atoms of a read-only Task: nothing in it can be made to say something
    false about its own Reference, which is the one thing check 8 is there to catch."""
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    talking = make_run("talking", [user("Hello?"), assistant("Hello, how can I help?")])
    verifier = V.derive_verifier(TASK, talking, [], CanonRules(), write_tools=WRITE_TOOLS, constraints=[rule])
    assert {S.atom_payload(a).get("kind") for a in verifier.atoms} == {"entity_count", "hard"}
    gate = [g for g in S.validate_verifier(verifier, talking, canon=CanonRules()) if g.stage == "verifier_mutation"][0]
    assert gate.passed is False
    # Three, not two: D190 attaches its no-write claim here too, and a Run with no call at all
    # gives that rule nothing to judge, so it is inert on this Reference like the other two.
    assert gate.metrics == {"atoms_mutated": 0, "atoms_not_mutable": 3}
    assert "nothing in it can be falsified" in " ".join(gate.failures)


def literal_value_atom(tool: str, entity: str, id_field: str, field: str, value,
                       provenance: str = "system_derived") -> Verifier:
    """A Verifier holding one write value as the literal it is, which is the atom check 7 rules on.

    Built by hand rather than derived: since D190 the derivation writes a required value no user of
    the Task said as a shape over its own column and keeps no literal, so it no longer produces this
    atom, while a Verifier stored before D190 and one a repair wrote by hand still do. The check is
    frozen and its behaviour over such an atom is what these tests are about.
    """
    atom = S.make_atom(f"w0.{field}", "required",
                       {"kind": "write_value", "tool": tool, "entity": entity, "entity_raw": entity,
                        "id_field": id_field, "at": 3, "field": field,
                        "value": S._key(S.canon_fn(CanonRules()), value), "raw": value},
                       provenance=provenance)
    return Verifier(task_id="t1", atoms=[atom])


def _refund_run():
    """A refund whose amount the agent read off a tool result: check 7 greps for constants only the
    Verifier should know, whether or not they are strings."""
    return make_run("ref", [
        user("Please refund my order #W123."),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("refund_order", {"order_id": "#W123", "amount": 150.0}, cid="c1"),
        result({"ok": True}, cid="c1"),
        assistant("Refunded."),
    ])


@pytest.mark.parametrize("tool, reference, amount, intent_text, user_rules, expected_pass, leaked", [
    pytest.param("cancel_pending_order", reference_run, 175.0, "cancel #W123 and refund exactly 175.0", UserRules(),
                 False, "intent leaks 175.0", id="intent"),
    pytest.param("cancel_pending_order", reference_run, 175.0, "",
                 UserRules(refusals=["do not accept less than 175.0"]), False, "user_rules leaks 175.0",
                 id="user_rules"),
    pytest.param("cancel_pending_order", reference_run, 175.0, "cancel order #W123 and record the reason",
                 UserRules(), True, None, id="own_words"),
    pytest.param("refund_order", _refund_run, 150.0, "refund exactly 150.0 on #W123", UserRules(),
                 True, None, id="world_value_read_off_a_tool_result"),
    pytest.param("cancel_pending_order", reference_run, 175.0, "refund $175 on #W123", UserRules(),
                 False, "intent leaks 175.0", id="another_spelling_of_the_value"),
])
def test_the_leak_check_finds_a_constant_only_the_verifier_holds_and_never_a_world_value_or_the_users_own_words(
        tmp_path, tool, reference, amount, intent_text, user_rules, expected_pass, leaked):
    """D287: the comparison is canonical, and a value a row of the world holds is not a secret even
    when the Intent says it; a value no row holds (the agent's own number) is."""
    verifier = literal_value_atom(tool, "#W123", "order_id", "amount", amount, provenance="agent_chosen")
    gates = {g.stage: g for g in S.validate_verifier(
        verifier, reference(), empty_run(), wrong_run(), alt_path_run(),
        intent_text=intent_text, user_rules=user_rules, canon=CanonRules())}
    assert gates["verifier_leak"].passed is expected_pass
    if leaked is not None:
        assert leaked in " ".join(gates["verifier_leak"].failures)


def test_the_leak_check_names_the_column_the_strip_missed_and_says_whether_it_audited_a_record():
    """D196 makes check 7 an audit: the strip has already run, so a leak is a column it does not
    cover yet, and the column is what a finding can be keyed on and the next strip has to read.
    A caller that holds no record still gets the check, and says on the ruling that it audited
    nothing: a strip that never ran must not read as a strip that found nothing."""
    record = Intent(task_id="t1", text="cancel #W123 and refund exactly 175.0", grounded=True,
                    stripped=[StrippedValue(column="order_id", table="orders", shape="last4",
                                            replacement="ending 0123")])
    gate = {g.stage: g for g in S.validate_verifier(
        literal_value_atom("cancel_pending_order", "#W123", "order_id", "amount", 175.0),
        reference_run(), intent_text=record.text, user_rules=UserRules(),
        intent=record, canon=CanonRules())}["verifier_leak"]
    assert gate.passed is False
    assert "column cancel_pending_order.amount" in " ".join(gate.failures)
    assert gate.metrics["columns"] == ["cancel_pending_order.amount"]
    assert gate.metrics["stripped"] == 1 and gate.metrics["audited"] is True

    gate = {g.stage: g for g in S.validate_verifier(
        literal_value_atom("cancel_pending_order", "#W123", "order_id", "amount", 175.0),
        reference_run(), intent_text="cancel #W123 and refund exactly 175.0",
        user_rules=UserRules(), canon=CanonRules())}["verifier_leak"]
    assert gate.passed is False
    assert gate.metrics["audited"] is False and gate.metrics["stripped"] == 0


def _garage_events(said: str) -> list[dict]:
    """A booking whose price the garage read off the vehicle record and never asked the driver for."""
    return [
        user(said),
        call("get_vehicle", {"plate": "KP19TRX"}, kind="read", cid="c0"),
        result({"plate": "KP19TRX", "quote": 240.0}, cid="c0"),
        call("book_service", {"plate": "KP19TRX", "quote": 264.0}, cid="c1"),
        result({"booked": True}, cid="c1"),
        assistant("Booked."),
    ]


def _garage_verifier() -> Verifier:
    """The price with tax kept as a literal: no row holds it and the driver was never told it, so check 7
    rules on it (D190, D287)."""
    return literal_value_atom("book_service", "KP19TRX", "plate", "quote", 264.0, provenance="agent_chosen")


def test_a_value_is_no_leak_only_when_a_user_said_it_in_some_seed_recording():
    """The Intent is mined from every recording of the Task, so check 7 has to read every one of them.

    Reading the user turns off the Reference alone made this check stricter than the miner it
    polices: on the last build 18 of 24 leaked values were spoken by a user in another recording of
    the same Task, and the 14 Tasks carrying them were blocked for nothing. Widening the check to
    the other seed recordings does not weaken it: a price no driver of any recording spoke is still
    the Verifier's own knowledge, and the Intent must not carry it.
    """
    gates = {g.stage: g for g in S.validate_verifier(
        _garage_verifier(), make_run("ref", _garage_events("please book my car in for its service")),
        intent_text="book the 264.0 service", user_rules=UserRules(),
        seed_runs=[make_run("seed", _garage_events("book the service, i was quoted 264.0 last month"))], canon=CanonRules())}
    assert gates["verifier_leak"].passed is True
    assert gates["verifier_leak"].failures == []

    gates = {g.stage: g for g in S.validate_verifier(
        _garage_verifier(), make_run("ref", _garage_events("please book my car in for its service")),
        intent_text="book the 264.0 service", user_rules=UserRules(),
        seed_runs=[make_run("seed", _garage_events("book my car in, the brakes are grinding"))], canon=CanonRules())}
    assert gates["verifier_leak"].passed is False
    assert "intent leaks 264.0" in " ".join(gates["verifier_leak"].failures)


def test_a_loophole_probe_that_did_not_run_is_not_a_pass(tmp_path):
    """D79 check 6 skipped is 'we do not know', which the suite has to say out loud."""
    verifier = derive(tmp_path)
    gate = S.loophole_probe(verifier, None)
    assert gate.stage == "verifier_loophole"
    assert gate.metrics["skipped"] is True
    assert gate.passed is False
    assert "not run" in " ".join(gate.failures)


def test_a_loophole_probe_run_that_passes_the_verifier_fails_check_6(tmp_path, test_model):
    verifier = derive(tmp_path)
    gate = S.loophole_probe(verifier, test_model, run_probe=lambda model, ver: empty_run(), canon=CanonRules())
    assert gate.passed is True
    gate = S.loophole_probe(verifier, test_model, run_probe=lambda model, ver: reference_run(), canon=CanonRules())
    assert gate.passed is False

def test_a_fact_stated_before_the_farewell_is_a_communicate_fact():
    run = make_run("r", [
        user("What is the status of order #W123?"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Order #W123 is pending."),
        user("Thanks, that is all."),
        assistant("You are welcome, goodbye."),
    ])
    said = S.communicate_values(run, S.canon_fn(CanonRules()))
    assert [v["text"] for v in said.values()] == ["#W123"]


# --- the D79 checks over a replayed Reference (wrong Run by code, second path) ---

def test_the_wrong_run_aims_every_required_write_at_another_entity_and_needs_a_required_write(tmp_path):
    out = do_replay(tmp_path)
    task = Task(id="t1", run_ids=["tr1"])
    verifier = V.derive_verifier(task, out.path, write_tools={"cancel_order"}, canon=CanonRules())
    wrong = S.wrong_run(verifier, out.path, CanonRules())
    assert wrong is not None and wrong.run_id.endswith(".wrong")
    call_event = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call_event.payload["args"]["order_id"] != "123"
    passed, failing = S.check_run(verifier, wrong, CanonRules(), write_tools={"cancel_order"})
    assert passed is False and failing == "w0"
    # the Reference itself is untouched
    assert S.check_run(verifier, out.path, CanonRules(), write_tools={"cancel_order"})[0] is True
    # a Verifier that requires nothing has no wrong Run
    assert S.wrong_run(Verifier(task_id="t1", atoms=[]), out.path, CanonRules()) is None


def test_the_wrong_run_prefers_an_id_the_reference_showed(tmp_path):
    class TwoOrders(Toolkit):
        def get_order_details(self, order_id):
            return {"id": order_id, "status": "delivered", "total": 25, "other": {"order_id": "456"}}

    out = do_replay(tmp_path, TwoOrders)
    verifier = V.derive_verifier(Task(id="t1", run_ids=["tr1"]), out.path, write_tools={"cancel_order"}, canon=CanonRules())
    wrong = S.wrong_run(verifier, out.path, CanonRules())
    call_event = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call_event.payload["args"]["order_id"] == "456"


def test_the_suite_runs_the_wrong_run_and_the_second_path(tmp_path):
    first = do_replay(tmp_path / "a")
    second = do_replay(tmp_path / "b")
    verifier = V.derive_verifier(Task(id="t1", run_ids=["tr1"]), first.path, [second.path],
                                 write_tools={"cancel_order"}, canon=CanonRules())
    gates = {g.stage: g for g in S.validate_verifier(
        verifier, first.path, write_tools={"cancel_order"}, wrong_run=S.wrong_run(verifier, first.path, CanonRules()),
        alt_path_run=second.path, canon=CanonRules())}
    assert gates["verifier_wrong_run"].passed and gates["verifier_alt_path"].passed
    assert gates["verifier_loophole"].metrics.get("skipped")  # no model, so not known to be tight


def test_the_transcript_helpers_are_one_text_the_derivation_and_the_policy_compiler_both_read(tmp_path):
    """HELPERS_SRC lives in the gates package since phase 5: a Hard rule is scored there, so the helpers
    it may call have to be there too, and the Builder's policy compiler reads the same text rather than
    its own copy. A Hard atom derived for a Verifier runs against exactly these helpers."""
    from kullback.builder import policy

    assert policy.predicate_source is S.predicate_source
    assert S.predicate_source("def check(a, b, c):\n    return True\n").startswith(S.HELPERS_SRC)
    assert set(S.HELPER_NAMES) >= {"user_confirmed", "called_before", "said_before"}
    assert V._helpers_src() is S.HELPERS_SRC
    namespace: dict = {}
    exec(S.HELPERS_SRC, namespace)
    for helper in ("user_confirmed", "called_before", "said_before"):
        assert callable(namespace[helper]), helper
    transcript = [{"role": "assistant", "content": "Shall I cancel the order?"}, {"role": "user", "content": "yes please"}]
    assert namespace["user_confirmed"](transcript) and namespace["said_before"](transcript, "cancel")
    assert not namespace["called_before"](transcript, "cancel_order")


# --- an expected-fail Run that passed names the atoms it passed on (F37) ---

def _allowed_only_verifier() -> Verifier:
    return Verifier(task_id="t1", atoms=[S.make_atom("cap", "allowed", {"kind": "entity_count", "count": 5}),
                                         S.make_atom("wide", "allowed", {"kind": "entity_count", "count": 9})])


def test_an_empty_run_that_passes_a_verifier_of_allowed_atoms_names_no_atom_and_says_none_is_checked(tmp_path):
    from kullback.gates.bindings import rows_for

    gates = {g.stage: g for g in S.validate_verifier(_allowed_only_verifier(), reference_run(), empty_run(), canon=CanonRules())}
    gate = gates["verifier_empty_run"]
    assert gate.passed is False
    assert gate.metrics["passing_atoms"] == []
    rows = rows_for(gate, {})
    assert {"check": "verifier_empty_run", "failure": "expected fail, got pass"} in rows
    assert {"check": "verifier_empty_run",
            "failure": "no atom of the Verifier is one a Run is checked on, so no Run can fail it"} in rows
    assert [row for row in rows if "atom" in row] == []


def test_an_expected_fail_run_that_passed_names_only_the_atoms_check_run_evaluated(tmp_path):
    from kullback.gates.bindings import rows_for

    verifier = Verifier(task_id="t1", atoms=[S.make_atom("few", "required", {"kind": "entity_count", "count": 5}),
                                             S.make_atom("wide", "allowed", {"kind": "entity_count", "count": 9})])
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), empty_run(), canon=CanonRules())}
    gate = gates["verifier_empty_run"]
    assert gate.passed is False
    assert gate.metrics["passing_atoms"] == [{"id": "few", "kind": "required"}]
    assert [row for row in rows_for(gate, {}) if "atom" in row] == [
        {"check": "verifier_empty_run", "atom": "few", "kind": "required"}]


def test_an_expected_fail_run_that_failed_or_an_expected_pass_carries_no_passing_atoms(tmp_path):
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), empty_run(), canon=CanonRules())}
    assert gates["verifier_empty_run"].passed is True
    assert "passing_atoms" not in gates["verifier_empty_run"].metrics
    assert "passing_atoms" not in gates["verifier_oracle"].metrics
