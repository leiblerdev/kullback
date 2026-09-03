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
from kullback.runner.confinement import confine
from kullback.runner.records import Constraint, Task, UserRules, Verifier
from runner.replay_fixtures import Toolkit, do_replay

# --- reading re-runs from disk (D91) --------------------------------------

def test_load_run_reads_header_plus_event_lines(tmp_path):
    path = Path(write_events_jsonl(reference_run(), tmp_path / "ref.jsonl"))
    run = S.load_run(path)
    assert run.run_id == "ref"
    assert len(run.events) == 8
    assert run.events[0].type == "user_turn"


def test_load_run_reads_a_whole_run_on_one_line(tmp_path):
    path = Path(write_run_json(alt_path_run(), tmp_path / "alt.jsonl"))
    run = S.load_run(path)
    assert run.run_id == "alt"
    assert len(run.events) == 8

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

    assert S.check_run(verifier, with_state("pending")) == (True, None)
    assert S.check_run(verifier, with_state("delivered")) == (False, "hard.k1")


def test_a_hard_rule_that_raises_is_a_failure_and_not_a_silent_pass(tmp_path):
    """A predicate that blows up decided nothing, so the oracle check has to see it (D79 check 2)."""
    rule = Constraint(id="k1", text="never cancel an order that is not pending", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return pre_state['orders']['#W123']['status'] == 'pending'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    assert S.check_run(verifier, reference_run()) == (False, "hard.k1")  # the Run carries no state
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run())}
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
    assert S.hard_holds(atom, reference_run(), WRITE_TOOLS) is False
    assert not marker.exists(), "the escape payload ran outside the sandbox"
    assert S.check_run(verifier, reference_run()) == (False, "hard.k1")


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
    assert S.hard_holds(atom_by_id(verifier, "hard.k1"), reference_run(), WRITE_TOOLS) is False
    assert S.hard_holds(atom_by_id(verifier, "hard.k2"), reference_run(), WRITE_TOOLS) is True


def test_a_hard_rule_sees_the_transcript_before_the_write_and_nothing_after(tmp_path):
    """D43 case 3: a confirmation that arrived after the write did not precede it."""
    never = Constraint(
        id="k1", text="never cancel without a prior user confirmation", compiled=True,
        predicate_src="def check(pre_state, write_call, transcript):\n    return user_confirmed(transcript)\n")
    confirmed = make_run("ref", [
        user("Please cancel my order #W123."),
        assistant("Shall I cancel #W123?"),
        user("yes"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Cancelled."),
    ])
    verifier = V.derive_verifier(TASK, confirmed, [], None, write_tools=WRITE_TOOLS, constraints=[never])
    late = make_run("late", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("I cancelled #W123. Was that what you wanted?"),
        user("yes"),
        assistant("Great."),
    ])
    hard = atom_by_id(verifier, "hard.k1")
    assert S.hard_holds(hard, confirmed, WRITE_TOOLS) is True
    assert S.hard_holds(hard, late, WRITE_TOOLS) is False
    assert S.check_run(Verifier(task_id="t1", atoms=[hard]), late) == (False, "hard.k1")


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
    assert S.check_run(hard_only, reading) == (True, None)
    assert S.check_run(hard_only, reference_run()) == (False, "hard.k1")

# --- the local scorer -----------------------------------------------------

def test_a_different_wording_of_an_allowed_value_passes(tmp_path):
    verifier = derive(tmp_path)
    reworded = make_run("reworded", reference_events(reason="I found it cheaper elsewhere"))
    assert S.check_run(verifier, reworded)[0] is True


def test_a_write_no_successful_run_made_still_fails_on_the_write_cap(tmp_path):
    """The stray write of a failed re-run is not an atom (D43); the cap is what stops a Candidate."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), failed_run()])
    over = make_run("over", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c2"),
        result({"status": "cancelled"}, cid="c2"),
    ])
    passed, failing = S.check_run(verifier, over)
    assert passed is False
    assert failing == "entity_count"


def test_check_run_needs_write_tools_to_catch_an_extra_write_on_an_uncovered_tool(tmp_path):
    """build.py:347 always supplies the mined write_tools in production; no call in this file did
    until this test, so the extra-write safety net (_extra_write) had no coverage of that shape."""
    verifier = derive(tmp_path, reruns=[alt_path_run(), extra_write_run()])
    sneaky = make_run("sneaky", reference_events() + [
        call("refund_order", {"order_id": "#W123", "amount": 25.0}, kind="read", cid="c2"),
        result({"ok": True}, cid="c2"),
    ])
    # With no write_tools, the extra call is invisible: nothing marked it as a write, so it slips
    # through as a full pass.
    assert S.check_run(verifier, sneaky) == (True, None)
    # The same Run, scored the way build.py actually scores it, is caught.
    assert S.check_run(verifier, sneaky, write_tools={"cancel_pending_order", "refund_order"}) == (
        False, "extra_write:refund_order")

# --- D79 validation -------------------------------------------------------

def test_the_suite_runs_the_nine_d79_checks_in_order_and_passes_a_sound_verifier_given_every_run(
        tmp_path, test_model):
    verifier = derive(tmp_path)
    gates = S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()])
    assert [g.stage for g in gates] == [
        "verifier_provenance_spans", "verifier_oracle", "verifier_empty_run",
        "verifier_wrong_run", "verifier_unfinished_run", "verifier_alt_path", "verifier_loophole",
        "verifier_leak", "verifier_mutation",
    ]
    assert all(g.passed for g in gates), [(g.stage, g.failures) for g in gates]


def test_the_unfinished_run_stops_just_before_the_required_write_and_must_fail(tmp_path):
    """D119: the Reference cut one step short (GLM 5.3's unsolved-state check) scores no pass."""
    verifier = derive(tmp_path)
    unfinished = S.unfinished_run(verifier, reference_run())
    assert unfinished.run_id == "ref.unfinished" and unfinished.termination_reason == "max_turns"
    assert [e.type for e in unfinished.events][-2:] == ["tool_call", "tool_result"]  # the read, not the cancel
    assert not any(e.type == "tool_call" and e.payload.get("name") == "cancel_pending_order"
                   for e in unfinished.events)
    assert S.check_run(verifier, unfinished)[0] is False
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run())}
    assert gates["verifier_unfinished_run"].passed is True


def test_a_verifier_that_rewards_an_unfinished_run_is_caught(tmp_path):
    verifier = derive(tmp_path)
    # Only the write cap survives: a Verifier that checks the agent wrote nothing extra and never
    # that the cancellation happened, which a Run that stopped short satisfies.
    hollow = verifier.model_copy(update={"atoms": [a for a in verifier.atoms
                                                    if S.atom_payload(a).get("kind") == "entity_count"]})
    gates = {g.stage: g for g in S.validate_verifier(hollow, reference_run())}
    assert gates["verifier_unfinished_run"].metrics["run_passed"] is True
    assert gates["verifier_unfinished_run"].passed is False


def test_unfinished_run_is_none_for_an_empty_reference_and_cuts_a_writeless_one_before_its_last_turn(tmp_path):
    verifier = derive(tmp_path)
    assert S.unfinished_run(verifier, empty_run()) is None
    talk = make_run("talk", [user("hi"), assistant("hello")])
    assert [e.type for e in S.unfinished_run(verifier, talk).events] == ["user_turn"]


def test_the_suite_reports_every_d79_check_by_the_name_the_verifier_gate_wants(tmp_path, test_model):
    """verifier_gate counts a check it was not told about as a failure, so the names have to line up;
    the mapping is the contract between the two modules and nothing imports it."""
    verifier = derive(tmp_path)
    gates = S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
                                intent_text=TASK.intent, user_rules=UserRules(), model=test_model,
                                run_probe=lambda model, ver: empty_run(),
                                seed_runs=[alt_path_run(), other_reason_run()])
    results = S.d79_results(gates)
    assert sorted(results) == sorted(artifacts.D79_CHECKS)
    assert artifacts.verifier_gate(results).passed is True


def test_a_check_with_no_run_is_reported_as_not_run(tmp_path):
    """A Verifier nobody tried to break is not a validated Verifier; missing input is a failure."""
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run())}
    assert gates["verifier_wrong_run"].passed is False
    assert gates["verifier_alt_path"].passed is False
    assert gates["verifier_loophole"].passed is False
    assert all("not run" in " ".join(gates[stage].failures)
               for stage in ("verifier_wrong_run", "verifier_alt_path", "verifier_loophole"))
    assert artifacts.verifier_gate(S.d79_results(gates.values())).passed is False


def test_a_hollow_verifier_fails_the_suite_with_no_runs_supplied(tmp_path):
    """The empty Run needs no Runner, so check 3 always runs and an atomless Verifier cannot clear it."""
    hollow = Verifier(task_id="t1", atoms=[])
    gates = {g.stage: g for g in S.validate_verifier(hollow, reference_run())}
    assert gates["verifier_empty_run"].passed is False
    assert gates["verifier_empty_run"].metrics["run_passed"] is True
    assert any(not g.passed for g in gates.values())
    supplied = {g.stage: g for g in S.validate_verifier(hollow, reference_run(), empty_run(), wrong_run(),
                                                        alt_path_run(), intent_text="", user_rules=None)}
    assert supplied["verifier_empty_run"].passed is False
    assert supplied["verifier_wrong_run"].passed is False


@pytest.mark.parametrize("break_span", [
    pytest.param(lambda atom: setattr(atom, "spans", []), id="no_span"),
    pytest.param(lambda atom: setattr(atom.spans[0], "msg_index", 3), id="wrong_turn"),
])
def test_validate_flags_a_provenance_atom_whose_span_does_not_hold_the_value(tmp_path, break_span):
    verifier = derive(tmp_path)
    for atom in verifier.atoms:
        if atom.provenance == "user_stated" and atom.spans:
            break_span(atom)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), empty_run(), wrong_run(),
                                                     alt_path_run(), intent_text="", user_rules=None)}
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
    with_seed = {g.stage: g for g in S.validate_verifier(verifier, reference_run(), seed_runs=[extra])}
    assert with_seed["verifier_provenance_spans"].passed is True
    alone = {g.stage: g for g in S.validate_verifier(verifier, reference_run())}
    assert alone["verifier_provenance_spans"].passed is False
    assert "not supplied" in " ".join(alone["verifier_provenance_spans"].failures)


def test_the_mutation_check_flags_an_atom_the_reference_cannot_fail(tmp_path):
    """D79 check 8: an atom whose mutation changes nothing is an atom that is not being checked."""
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    talking = make_run("talking", [user("Hello?"), assistant("Hello, how can I help?")])
    verifier = V.derive_verifier(TASK, talking, [], None, write_tools=WRITE_TOOLS, constraints=[rule])
    gate = [g for g in S.validate_verifier(verifier, talking) if g.stage == "verifier_mutation"][0]
    assert gate.passed is False  # nothing was written, so the rule was never asked anything
    assert "hard.k1" in " ".join(gate.failures)
    lively = [g for g in S.validate_verifier(derive(tmp_path, constraints=[rule]), reference_run())
              if g.stage == "verifier_mutation"][0]
    assert lively.passed is True
    assert lively.metrics["atoms_mutated"] >= 4


def test_leak_check_finds_a_number_the_verifier_read_off_a_tool_result(tmp_path):
    """Check 7 greps for constants only the Verifier should know, whether or not they are strings."""
    def events(amount):
        return [
            user("Please refund my order #W123."),
            call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
            result(ORDER, cid="c0"),
            call("refund_order", {"order_id": "#W123", "amount": amount}, cid="c1"),
            result({"ok": True}, cid="c1"),
            assistant("Refunded."),
        ]
    verifier = V.derive_verifier(TASK, make_run("ref", events(150.0)), [], None,
                                 write_tools={"refund_order"})
    amount = [a for a in verifier.atoms if S.atom_payload(a).get("field") == "amount"][0]
    assert amount.provenance == "system_derived"
    gates = {g.stage: g for g in S.validate_verifier(verifier, make_run("ref", events(150.0)),
                                                     intent_text="refund exactly 150.0 on #W123",
                                                     user_rules=UserRules())}
    assert gates["verifier_leak"].passed is False
    assert "150.0" in " ".join(gates["verifier_leak"].failures)


@pytest.mark.parametrize("intent_text, user_rules, expected_pass, leaked", [
    pytest.param("cancel #W123 and refund exactly 150.0", UserRules(), False, "intent leaks 150.0",
                 id="intent"),
    pytest.param("", UserRules(refusals=["do not accept less than 150.0"]), False,
                 "user_rules leaks 150.0", id="user_rules"),
    pytest.param("cancel order #W123 and record the reason", UserRules(), True, None, id="own_words"),
])
def test_the_leak_check_finds_a_system_derived_constant_in_the_intent_or_the_user_rules_and_never_the_users_own_words(
        tmp_path, intent_text, user_rules, expected_pass, leaked):
    verifier = derive(tmp_path)
    gates = {g.stage: g for g in S.validate_verifier(
        verifier, reference_run(), empty_run(), wrong_run(), alt_path_run(),
        intent_text=intent_text, user_rules=user_rules)}
    assert gates["verifier_leak"].passed is expected_pass
    if leaked is not None:
        assert leaked in " ".join(gates["verifier_leak"].failures)


def test_a_loophole_probe_that_did_not_run_is_not_a_pass(tmp_path):
    """D79 check 6 skipped is 'we do not know', which the suite has to say out loud."""
    verifier = derive(tmp_path)
    gate = S.loophole_probe(verifier, None)
    assert gate.stage == "verifier_loophole"
    assert gate.metrics["skipped"] is True
    assert gate.passed is False
    assert "not run" in " ".join(gate.failures)


def test_loophole_probe_uses_the_run_it_is_given(tmp_path, test_model):
    verifier = derive(tmp_path)
    gate = S.loophole_probe(verifier, test_model, run_probe=lambda model, ver: empty_run())
    assert gate.passed is True
    gate = S.loophole_probe(verifier, test_model, run_probe=lambda model, ver: reference_run())
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
    said = S.communicate_values(run, S.canon_fn(None))
    assert [v["text"] for v in said.values()] == ["#W123"]


# --- the D79 checks over a replayed Reference (wrong Run by code, second path) ---

def test_the_wrong_run_aims_every_required_write_at_another_entity(tmp_path):
    out = do_replay(tmp_path)
    task = Task(id="t1", run_ids=["tr1"])
    verifier = V.derive_verifier(task, out.path, write_tools={"cancel_order"})
    wrong = S.wrong_run(verifier, out.path)
    assert wrong is not None and wrong.run_id.endswith(".wrong")
    call_event = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call_event.payload["args"]["order_id"] != "123"
    passed, failing = S.check_run(verifier, wrong, write_tools={"cancel_order"})
    assert passed is False and failing == "w0"
    # the Reference itself is untouched
    assert S.check_run(verifier, out.path, write_tools={"cancel_order"})[0] is True


def test_the_wrong_run_prefers_an_id_the_reference_showed(tmp_path):
    class TwoOrders(Toolkit):
        def get_order_details(self, order_id):
            return {"id": order_id, "status": "delivered", "total": 25, "other": {"order_id": "456"}}

    out = do_replay(tmp_path, TwoOrders)
    verifier = V.derive_verifier(Task(id="t1", run_ids=["tr1"]), out.path, write_tools={"cancel_order"})
    wrong = S.wrong_run(verifier, out.path)
    call_event = next(e for e in wrong.events if e.type == "tool_call" and e.payload["name"] == "cancel_order")
    assert call_event.payload["args"]["order_id"] == "456"


def test_a_verifier_that_requires_nothing_has_no_wrong_run(tmp_path):
    out = do_replay(tmp_path)
    assert S.wrong_run(Verifier(task_id="t1", atoms=[]), out.path) is None


def test_the_suite_runs_the_wrong_run_and_the_second_path(tmp_path):
    first = do_replay(tmp_path / "a")
    second = do_replay(tmp_path / "b")
    verifier = V.derive_verifier(Task(id="t1", run_ids=["tr1"]), first.path, [second.path],
                                 write_tools={"cancel_order"})
    gates = {g.stage: g for g in S.validate_verifier(
        verifier, first.path, write_tools={"cancel_order"}, wrong_run=S.wrong_run(verifier, first.path),
        alt_path_run=second.path)}
    assert gates["verifier_wrong_run"].passed and gates["verifier_alt_path"].passed
    assert gates["verifier_loophole"].metrics.get("skipped")  # no model, so not known to be tight


def test_the_transcript_helpers_are_one_text_the_derivation_and_the_policy_compiler_both_read(tmp_path):
    """HELPERS_SRC lives in the gates package since phase 5: a Hard rule is scored there, so the helpers
    it may call have to be there too, and the Builder's policy compiler reads the same text rather than
    its own copy. A Hard atom derived for a Verifier runs against exactly these helpers."""
    from kullback.builder import policy

    assert policy.HELPERS_SRC is S.HELPERS_SRC
    assert V._helpers_src() is S.HELPERS_SRC
    namespace: dict = {}
    exec(S.HELPERS_SRC, namespace)
    for helper in ("user_confirmed", "called_before", "said_before"):
        assert callable(namespace[helper]), helper
    transcript = [{"role": "assistant", "content": "Shall I cancel the order?"}, {"role": "user", "content": "yes please"}]
    assert namespace["user_confirmed"](transcript) and namespace["said_before"](transcript, "cancel")
    assert not namespace["called_before"](transcript, "cancel_order")
