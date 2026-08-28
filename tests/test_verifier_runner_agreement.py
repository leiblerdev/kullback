"""The Builder's scorer and the Verdict have to agree, atom for atom, on the same Run.

The D79 suite scores a Run with `check_run` off the atom targets; the Verdict scores it by evaluating
`predicate_src`. Two scorers means a Verifier can clear every gate here and grade something else at
Verdict time, so this file pins them to the same answer on the Runs the gates are made of. It is the
only place the Builder's tests look at the Runner, and it reads it, never imports it into the module
(D89, D91).
"""

from __future__ import annotations

from harness.builder import verifier as V
from harness.runner import verdict as R
from harness.shared.records import Constraint, Task, Verifier

from test_verifier import (
    ORDER, WRITE_TOOLS, alt_path_run, assistant, call, derive, empty_run, make_run, reference_events,
    reference_run, result, user, wrong_run,
)

TASK = Task(id="t1", intent="cancel the pending order and record the reason")


def both(verifier: Verifier, run, write_tools=WRITE_TOOLS):
    """Score one Run the Builder's way and the Runner's way."""
    builder = V.check_run(verifier, run, write_tools=write_tools)
    graded = R.verdict(run, verifier, None, write_tools=write_tools)
    return builder, (graded.passed, graded.failing_atom), graded.notes


def agree(verifier: Verifier, run, expected_pass: bool, write_tools=WRITE_TOOLS):
    builder, runner, notes = both(verifier, run, write_tools)
    assert builder == runner, (builder, runner, notes)
    assert builder[0] is expected_pass, (builder, notes)
    assert not [n for n in notes if n.startswith("atom_error")], notes
    return builder


# --- the Runs the D79 checks are made of ----------------------------------

def test_the_oracle_passes_both_scorers(tmp_path):
    agree(derive(tmp_path), reference_run(), True)


def test_the_second_path_passes_both_scorers(tmp_path):
    agree(derive(tmp_path), alt_path_run(), True)


def test_the_empty_run_fails_both_scorers(tmp_path):
    agree(derive(tmp_path), empty_run(), False)


def test_the_plausible_wrong_run_fails_both_scorers(tmp_path):
    agree(derive(tmp_path), wrong_run(), False)


def test_a_question_the_reference_asked_in_its_own_words(tmp_path):
    """The Reference asked "why", not "reason": an atom keyed on the field name failed its own oracle."""
    verifier = derive(tmp_path)
    question = [a for a in verifier.atoms if a.kind == "question"][0]
    assert V.atom_payload(question)["key"] == "field:reason"
    assert "reason" not in " ".join(
        e.payload["reply"]["content"] for e in reference_run().events if e.type == "model_call")
    agree(verifier, reference_run(), True)


def test_a_wrong_entity_fails_both_scorers_when_the_id_came_from_the_user(tmp_path):
    """The write atom's target is (tool, entity), so its predicate has to name the entity too."""
    def events(written):
        return [
            user("Please cancel one of my orders."),
            assistant("Which order id should I cancel?"),
            user("#W123"),
            call("cancel_pending_order", {"order_id": written, "reason": "customer request"}, cid="c1"),
            result({"order_id": written, "status": "cancelled"}, cid="c1"),
            assistant("Done, it is cancelled."),
        ]
    verifier = V.derive_verifier(TASK, make_run("ref", events("#W123")), [], None, write_tools=WRITE_TOOLS)
    agree(verifier, make_run("ref", events("#W123")), True)
    assert agree(verifier, make_run("wrong", events("#W999")), False)[1] == "w0"


def test_an_elicited_value_is_bound_to_the_users_own_reply(tmp_path):
    """D43: the agent asked, the user answered, and the write has to carry what the user answered."""
    verifier = V.derive_verifier(TASK, make_run("ref", reference_events()), [], None,
                                 write_tools=WRITE_TOOLS)
    invented = make_run("invented", [
        user("Please cancel my order #W123."),
        assistant("Sure. Why do you want to cancel it?"),
        user("changed my mind"),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    assert agree(verifier, invented, False)[1] == "q.field:reason"
    honest = make_run("honest", reference_events(reason="changed my mind"))
    agree(verifier, honest, True)


def test_a_confirmation_in_the_users_own_words_passes_both_scorers(tmp_path):
    """"ok" is a yes to the Builder; a predicate that only knows "yes" failed its own Reference."""
    events = [
        user("Please cancel my order #W123, I no longer need it."),
        assistant("I will cancel #W123 now, is that fine?"),
        user("ok"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Cancelled."),
    ]
    verifier = V.derive_verifier(TASK, make_run("ref", events), [], None, write_tools=WRITE_TOOLS)
    assert any(V.atom_payload(a).get("key") == "confirm:cancel_pending_order" for a in verifier.atoms)
    agree(verifier, make_run("ref", events), True)
    unasked = make_run("unasked", [
        user("Please cancel my order #W123, I no longer need it."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Cancelled."),
    ])
    assert agree(verifier, unasked, False)[1] == "q.confirm:cancel_pending_order"


def test_writing_the_same_entity_twice_fails_both_scorers(tmp_path):
    verifier = derive(tmp_path)
    twice = make_run("twice", reference_events()[:-1] + [
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c2"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c2"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    assert agree(verifier, twice, False)[1] == "entity_count"


def test_a_hard_rule_agrees_on_a_confirmation_that_came_too_late(tmp_path):
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
    hard_only = Verifier(task_id="t1", atoms=[a for a in verifier.atoms if a.kind == "hard"])
    late = make_run("late", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "customer request"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("I cancelled #W123. Was that what you wanted?"),
        user("yes"),
        assistant("Great."),
    ])
    agree(hard_only, confirmed, True)
    assert agree(hard_only, late, False)[1] == "hard.k1"


def test_a_hard_rule_agrees_on_a_tool_the_seed_runs_never_used(tmp_path):
    rule = Constraint(id="k1", text="never call delete_order", compiled=True,
                      predicate_src=("def check(pre_state, write_call, transcript):\n"
                                     "    return write_call['name'] != 'delete_order'\n"))
    verifier = derive(tmp_path, constraints=[rule])
    hard_only = Verifier(task_id="t1", atoms=[a for a in verifier.atoms if a.kind == "hard"])
    deleted = make_run("bad", reference_events()[:-1] + [
        call("delete_order", {"order_id": "#W123"}, cid="c9"),
        result({"deleted": True}, cid="c9"),
        assistant("Your order #W123 is cancelled and 150.0 is refunded."),
    ])
    agree(hard_only, reference_run(), True)
    assert agree(hard_only, deleted, False)[1] == "hard.k1"
