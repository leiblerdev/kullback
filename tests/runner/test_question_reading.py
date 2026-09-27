"""Question atoms read the Run, not one model's habits (G5b).

Invented tools only. A confirmation is a user reply after the agent's question
that holds an affirmation and no negation.
"""

from __future__ import annotations

from kullback.gates import verifier_suite as S
from kullback.runner import target as T
from kullback.runner.canon import CanonRules
from kullback.runner.records import Event, Run, Verifier

WRITE_TOOLS = {"shelve_parcel"}


def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return _ev("user_turn", content=text)


def assistant(text: str) -> dict:
    return _ev("model_call", reply={"content": text})


def call(name: str, args: dict, cid: str = "c1") -> dict:
    return _ev("tool_call", id=cid, name=name, args=args)


def result(data, cid: str = "c1") -> dict:
    return _ev("tool_result", id=cid, result=data)


def make_run(run_id: str, events: list[dict]) -> Run:
    return Run(run_id=run_id, task_id="t9", termination_reason="success",
               events=[Event(idx=i, type=e["type"], payload=e["payload"]) for i, e in enumerate(events)])


def fn():
    return T.canon_fn(CanonRules())


def _confirm_run(reply: str) -> Run:
    return make_run("r1", [
        user("Shelve parcel P-7."),
        assistant("Shall I shelve P-7 now?"),
        user(reply),
        call("shelve_parcel", {"parcel_id": "P-7"}),
        result({"parcel_id": "P-7", "shelved": True}),
    ])


def _asked(run: Run) -> set:
    return set(T.question_keys(run, T.write_effects(run, WRITE_TOOLS, fn()), fn()))


def test_a_yes_buried_mid_reply_counts_as_confirmation():
    """The held-out user said yes, just not as the first word."""
    assert "confirm:shelve_parcel" in _asked(_confirm_run("Um, yes, please go ahead and shelve it."))


def test_a_plain_yes_still_confirms():
    assert "confirm:shelve_parcel" in _asked(_confirm_run("Yes, please."))


def test_a_yes_followed_by_no_is_not_a_confirmation():
    """Risk side: "yes, but wait, no" holds a negation, so it is not consent."""
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("Yes, but wait, no, not yet."))


def test_a_denial_is_not_a_confirmation():
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("No, don't shelve it."))


def test_a_reply_with_no_affirmation_is_not_a_confirmation():
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("Let me check the details first."))


def test_a_not_inside_a_product_option_does_not_veto_the_yes():
    """Bare "not" vetoes only next to an affirmation: "not water resistant"
    describes the option, it does not deny consent."""
    assert "confirm:shelve_parcel" in _asked(_confirm_run(
        "Yes, please proceed with the morning slot, not the evening one."))


def test_a_not_next_to_sure_or_confirm_vetoes():
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("Not sure yet."))
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("I do not confirm."))


def test_know_and_incorrect_read_as_neither_yes_nor_no():
    """Token matching: "know" is not "no", "incorrect" is not "correct"."""
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("I know, let me think."))
    assert "confirm:shelve_parcel" not in _asked(_confirm_run("Incorrect, try again later."))


def _question_verifier() -> Verifier:
    entity = T.text_of(fn()("P-7"))
    return Verifier(task_id="t9", verifier_version="v1", atoms=[
        S.make_atom("w0", "required",
                    {"kind": "write", "tool": "shelve_parcel",
                     "entity": entity, "entity_raw": "P-7", "id_field": "parcel_id"}),
        S.make_atom("q0", "question",
                    {"kind": "question", "key": "field:slot",
                     "tool": "shelve_parcel", "field": "slot"}),
    ])


def test_a_value_the_user_gave_after_being_asked_satisfies_the_question_atom():
    run = make_run("r1", [
        user("Shelve parcel P-7."),
        assistant("Which slot?"),
        user("The morning slot."),
        call("shelve_parcel", {"parcel_id": "P-7", "slot": "morning"}),
        result({"parcel_id": "P-7", "slot": "morning"}),
    ])
    assert T.check_run(_question_verifier(), run, CanonRules(), write_tools=WRITE_TOOLS) == (True, None)


def test_a_guessed_value_fails_the_question_atom():
    """The slot was never stated or read: asking again would have been the point."""
    run = make_run("r1", [
        user("Shelve parcel P-7."),
        call("shelve_parcel", {"parcel_id": "P-7", "slot": "morning"}),
        result({"parcel_id": "P-7", "slot": "morning"}),
    ])
    assert T.check_run(_question_verifier(), run, CanonRules(), write_tools=WRITE_TOOLS) == (False, "q0")


def test_a_guessed_value_still_fails_on_the_write_value_atom():
    """Risk side: G5a loosens nothing about a value the Run invented."""
    key = T._key(fn(), "morning")
    entity = T.text_of(fn()("P-7"))
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[
        S.make_atom("w0", "required",
                    {"kind": "write", "tool": "shelve_parcel",
                     "entity": entity, "entity_raw": "P-7", "id_field": "parcel_id"}),
        S.make_atom("w0.slot", "required",
                    {"kind": "write_value", "tool": "shelve_parcel",
                     "entity": entity, "entity_raw": "P-7", "id_field": "parcel_id",
                     "field": "slot", "value": key, "raw": "morning"}),
    ])
    run = make_run("r1", [
        user("Shelve parcel P-7."),
        call("shelve_parcel", {"parcel_id": "P-7", "slot": "evening"}),
        result({"parcel_id": "P-7", "slot": "evening"}),
    ])
    assert T.check_run(verifier, run, CanonRules(), write_tools=WRITE_TOOLS) == (False, "w0.slot")
