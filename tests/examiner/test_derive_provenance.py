"""D190: what a demand may be written in, and what a Task with no write still claims.

The domain is a staff locker room: a member says which locker they want, the roster says which badge
code that locker is issued against, and the desk assigns it. Nothing here names a customer's tools or
columns; the Runs are hand-built the way tests/gates/verifier_fixtures.py builds its own.
"""

from __future__ import annotations

from typing import Any, Optional

from kullback.examiner import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.atom_context import AtomContext
from kullback.runner.records import Event, Run

ASSIGN = "assign_locker"
ROSTER = "read_roster"
WRITE_TOOLS = {ASSIGN}
# The Starting state a Run carries on its stop event, which a shape atom reads at Verdict time.
WORLD = {"lockers": {"L9": {"locker_id": "L9", "badge_code": "QX41", "floor": 27},
                     "L4": {"locker_id": "L4", "badge_code": "TM08", "floor": 14}},
         "desk": {"free_lockers": 12}}


def _event(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return _event("user_turn", content=text)


def says(text: str) -> dict:
    return _event("model_call", reply={"content": text})


def call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return _event("tool_call", id=cid, name=name, args=args, kind=kind)


def result(data: Any, cid: str = "c1") -> dict:
    return _event("tool_result", id=cid, result=data)


def stop() -> dict:
    return _event("stop", start_state=WORLD)


def run(run_id: str, events: list[dict]) -> Run:
    return Run(run_id=run_id, task_id="t1", termination_reason="success",
               events=[Event(idx=number, type=e["type"], payload=e["payload"])
                       for number, e in enumerate(events + [stop()])])


def assignment_events(badge: str = "QX41", read_badge: Optional[str] = None) -> list[dict]:
    """The user names the locker; the desk reads the roster for the badge code and assigns it.

    `read_badge` is what the roster answered, when that is not what was written: a Run that wrote a
    badge code it never read is the case a shape atom has to keep rejecting.
    """
    return [
        user("Please assign locker L9 to me."),
        call(ROSTER, {"locker_id": "L9"}, kind="read", cid="c0"),
        result({"locker_id": "L9", "badge_code": badge if read_badge is None else read_badge}, cid="c0"),
        call(ASSIGN, {"locker_id": "L9", "badge_code": badge}, cid="c1"),
        result({"locker_id": "L9", "assigned": True}, cid="c1"),
        says("Locker L9 is yours."),
    ]


def derive(reference: Run, reruns: list[Run] = (), **kwargs) -> Any:
    kwargs.setdefault("write_tools", WRITE_TOOLS)
    return V.derive_verifier(kwargs.pop("task", "t1"), reference, list(reruns), None, **kwargs)


def atom(verifier: Any, atom_id: str) -> Any:
    found = [a for a in verifier.atoms if a.id == atom_id]
    assert found, f"no atom {atom_id} in {[a.id for a in verifier.atoms]}"
    return found[0]


# --- rule 1: a required value is one a user said, else a shape ---------------

def test_a_written_value_no_user_said_becomes_a_shape_over_its_own_column():
    verifier = derive(run("ref", assignment_events()))
    shape = atom(verifier, "w0.badge_code")
    assert shape.kind == "hard"
    payload = S.atom_payload(shape)
    assert payload["derived_as"] == V.SHAPE_ATOM
    assert "QX41" not in str(payload) and "QX41" not in (shape.predicate_src or "")
    assert "badge_code" in shape.predicate_src


def test_the_leak_check_passes_over_a_shape_atom_whose_value_the_intent_spells_out():
    """The value the Intent gives away is no longer a constant the Verifier keeps."""
    reference = run("ref", assignment_events())
    verifier = derive(reference)
    gate = S._leak_gate(verifier, reference, "assign the locker whose badge code is QX41", None)
    assert gate.passed, gate.failures


def test_a_shape_atom_still_rejects_a_value_the_run_never_read():
    """Relaxed is not empty: the badge code written has to be one this Run was given."""
    verifier = derive(run("ref", assignment_events()))
    wrong = run("other", assignment_events(badge="TM08", read_badge="QX41"))
    passed, failing = S.check_run(verifier, wrong, write_tools=WRITE_TOOLS)
    assert not passed and failing == "w0.badge_code"


def test_a_written_value_a_user_said_stays_the_literal_it_was():
    verifier = derive(run("ref", assignment_events()))
    literal = atom(verifier, "w0.locker_id")
    assert literal.kind == "required"
    assert S.atom_payload(literal)["raw"] == "L9"
    assert literal.provenance == "user_stated"


def test_what_a_user_said_is_read_the_way_the_leak_check_reads_it():
    """`user_said` and the frozen leak check agree value by value, which is why one may relax the other."""
    reference = run("ref", assignment_events())
    spoken = V.spoken_text([reference])
    for value in ("L9", "QX41", "locker", "true", "9", 12, ["L9", "QX41"], ["L9"]):
        one = S.make_atom("a0", "communicate", {"kind": "communicate", "value": "k", "text": value},
                          provenance="system_derived")
        held = S.Verifier(task_id="t1", atoms=[one])
        gate = S._leak_gate(held, reference, " ".join(S._texts(value)), None)
        assert gate.passed is V.user_said(spoken, value), value


# --- D206: the shape is checked against what the Run read, not against the row it wrote ------

def copy_across_rows_events() -> list[dict]:
    """The member wants locker L4 issued against the badge code the roster holds for L9.

    The value is real and was read off the roster, but off another locker's row than the one the
    desk writes, which is an ordinary copy across rows and not an invention.
    """
    return [
        user("Please assign locker L4 to me, under the badge code locker L9 is issued against."),
        call(ROSTER, {"locker_id": "L9"}, kind="read", cid="c0"),
        result({"locker_id": "L9", "badge_code": "QX41"}, cid="c0"),
        call(ASSIGN, {"locker_id": "L4", "badge_code": "QX41"}, cid="c1"),
        result({"locker_id": "L4", "assigned": True}, cid="c1"),
        says("Locker L4 is yours, under badge QX41."),
    ]


def prose_events() -> list[dict]:
    """The roster answers in prose, the way a reader-built tool does, and the desk writes what it read."""
    return [
        user("Please assign locker L4 to me, under the badge code locker L9 is issued against."),
        call(ROSTER, {"locker_id": "L9"}, kind="read", cid="c0"),
        result("Locker L9 on floor 27 is issued against badge QX41.", cid="c0"),
        call(ASSIGN, {"locker_id": "L4", "badge_code": "QX41"}, cid="c1"),
        result({"locker_id": "L4", "assigned": True}, cid="c1"),
        says("Locker L4 is yours, under badge QX41."),
    ]


def unread_value_events() -> list[dict]:
    """The desk writes a badge code no result of this Run ever held under that column or as a word."""
    return [
        user("Please assign locker L4 to me."),
        call(ROSTER, {"locker_id": "L4"}, kind="read", cid="c0"),
        result({"locker_id": "L4", "spare_badge": "ZZ01"}, cid="c0"),
        call(ASSIGN, {"locker_id": "L4", "badge_code": "ZZ01"}, cid="c1"),
        result({"locker_id": "L4", "assigned": True}, cid="c1"),
        says("Locker L4 is yours."),
    ]


def source_of(verifier: Any, atom_id: str) -> str:
    return S.atom_payload(atom(verifier, atom_id)).get("shape_source")


def test_a_value_read_off_another_rows_result_satisfies_the_shape():
    reference = run("ref", copy_across_rows_events())
    verifier = derive(reference)
    assert source_of(verifier, "w0.badge_code") == "prior_result"
    assert S.check_run(verifier, reference, write_tools=WRITE_TOOLS)[0]


def test_a_value_a_prose_result_states_satisfies_the_shape():
    """A tool that answers in words is still the Run reading the value for itself."""
    reference = run("ref", prose_events())
    verifier = derive(reference)
    assert source_of(verifier, "w0.badge_code") == "result_text"
    assert S.check_run(verifier, reference, write_tools=WRITE_TOOLS)[0]


def test_the_shape_is_satisfied_by_the_write_s_own_row_as_it_always_was():
    verifier = derive(run("ref", assignment_events()))
    assert source_of(verifier, "w0.badge_code") == "own_row"
    assert S.check_run(verifier, run("again", assignment_events()), write_tools=WRITE_TOOLS)[0]


def test_a_value_the_run_never_read_is_rejected_wherever_the_write_lands():
    """The demand stays real: a copy across rows passes, a value out of nowhere does not."""
    verifier = derive(run("ref", copy_across_rows_events()))
    invented = run("other", copy_across_rows_events()[:3] + [
        call(ASSIGN, {"locker_id": "L4", "badge_code": "ZZ99"}, cid="c1"),
        result({"locker_id": "L4", "assigned": True}, cid="c1"),
        says("Locker L4 is yours."),
    ])
    passed, failing = S.check_run(verifier, invented, write_tools=WRITE_TOOLS)
    assert not passed and failing == "w0.badge_code"


def test_a_shape_that_would_reject_its_own_reference_is_counted_and_not_stored():
    """A rule the Reference itself fails is a derivation defect, and a Verifier no Run can pass."""
    reference = run("ref", unread_value_events())
    verifier = derive(reference)
    assert [a.id for a in verifier.atoms if a.id == "w0.badge_code"] == []
    assert V.derivation_counts(verifier)["atoms_dropped_as_shape_self_reject"] == 1
    assert S.atom_payload(atom(verifier, "w0"))["shape_self_reject"] == ["badge_code"]
    assert S.check_run(verifier, reference, write_tools=WRITE_TOOLS)[0]


def test_a_row_whose_column_is_empty_demands_nothing_under_it():
    """A row seeded before the write it is about holds every column empty, and empty is not a value."""
    empty = {"lockers": {"L4": {"locker_id": "L4", "badge_code": None}}}
    seeded = Run(run_id="ref", task_id="t1", termination_reason="success",
                 events=[Event(idx=number, type=e["type"], payload=e["payload"])
                         for number, e in enumerate(unread_value_events()
                                                    + [_event("stop", start_state=empty)])])
    verifier = derive(seeded)
    assert source_of(verifier, "w0.badge_code") == "no_column"
    assert S.check_run(verifier, seeded, write_tools=WRITE_TOOLS)[0]


def test_a_shape_the_row_rule_accepted_keeps_the_row_rule_and_turns_a_swap_away():
    """The widening is spent only where the row rule rejected the Run it was read from.

    Here it did not, so the stored rule is D190's: a Candidate that writes another locker's badge
    code onto this one is turned away even though it read that code off a result of its own, which
    is the swap the wrong Run and the mutation check are made of.
    """
    verifier = derive(run("ref", assignment_events()))
    assert source_of(verifier, "w0.badge_code") == "own_row"
    swapped = run("other", [
        user("Please assign locker L9 to me."),
        call(ROSTER, {"locker_id": "L4"}, kind="read", cid="c0"),
        result({"locker_id": "L4", "badge_code": "TM08"}, cid="c0"),
        call(ASSIGN, {"locker_id": "L9", "badge_code": "TM08"}, cid="c1"),
        result({"locker_id": "L9", "assigned": True}, cid="c1"),
        says("Locker L9 is yours."),
    ])
    passed, failing = S.check_run(verifier, swapped, write_tools=WRITE_TOOLS)
    assert not passed and failing == "w0.badge_code"


def test_the_status_row_counts_which_source_satisfied_each_shape_and_keeps_no_value():
    verifier = derive(run("ref", copy_across_rows_events()))
    counts = V.derivation_counts(verifier)
    assert counts["shape_sources"] == {"prior_result": 1}
    assert "QX41" not in str(counts)


def test_the_two_scorers_hand_a_rule_the_same_transcript():
    """The gates score a Verifier and the Verdict scores it again; a rule reads one Run either way."""
    reference = run("ref", copy_across_rows_events())
    assert S._transcript(reference) == AtomContext(reference).transcript()


# --- rule 2: a Task with no write still carries something to falsify ---------

def desk_events(answer: str) -> list[dict]:
    return [
        user("How many free lockers are left?"),
        call(ROSTER, {"scope": "desk"}, kind="read", cid="c0"),
        result({"free_lockers": 12, "floor": 27}, cid="c0"),
        says(answer),
    ]


def test_a_read_only_task_demands_the_fact_the_request_asked_the_desk_about():
    verifier = derive(run("ref", desk_events("There are 12 free lockers left.")))
    facts = [a for a in verifier.atoms if S.atom_payload(a).get("kind") == "communicate"]
    assert [a.kind for a in facts] == ["communicate"]
    assert S.atom_payload(facts[0])["text"] == "12"


def test_a_fact_the_request_never_asked_about_stays_allowed_and_rejects_no_run():
    """D182 is not weakened: the number the desk volunteered beside the answer is reported."""
    verifier = derive(run("ref", desk_events("There are 12 free lockers left, on floor 27.")))
    reported = [a for a in verifier.atoms
                if S.atom_payload(a).get("kind") == V.REPORTED_COMMUNICATE]
    assert [S.atom_payload(a)["text"] for a in reported] == ["27"]
    assert reported[0].kind == "allowed"
    quiet = run("quiet", desk_events("There are 12 free lockers left."))
    assert S.check_run(verifier, quiet, write_tools=WRITE_TOOLS)[0]


def handover_events() -> list[dict]:
    """Nothing written and no fact told: the desk reads who is on shift and hands the request on."""
    return [
        user("I would rather speak to a person about this."),
        call(ROSTER, {"scope": "shift"}, kind="read", cid="c0"),
        result({"on_shift": "yes"}, cid="c0"),
        says("Of course. I am handing you to a colleague now."),
    ]


def test_a_task_that_wrote_nothing_and_told_nothing_claims_its_end_state_is_its_start_state():
    verifier = derive(run("ref", handover_events()))
    claim = atom(verifier, V.NO_WRITE_ATOM)
    assert claim.kind == "hard"
    assert V.derivation_counts(verifier)["atoms_added_for_falsification"] == 1
    assert S.check_run(verifier, run("ref", handover_events()), write_tools=WRITE_TOOLS)[0]


def test_a_run_that_wrote_fails_the_no_write_claim():
    verifier = derive(run("ref", handover_events()))
    wrote = run("wrote", handover_events()[:-1] + [
        call(ASSIGN, {"locker_id": "L4", "badge_code": "TM08"}, cid="c1"),
        result({"locker_id": "L4", "assigned": True}, cid="c1"),
        says("I have assigned you locker L4 instead."),
    ])
    assert S.hard_holds(atom(verifier, V.NO_WRITE_ATOM), wrote, WRITE_TOOLS) is False
    assert not S.check_run(verifier, wrote, write_tools=WRITE_TOOLS)[0]


def test_the_claim_is_one_the_mutation_check_can_flip_on_a_read_only_reference():
    """The D79 check that a Verifier says something falsifiable about its own Reference (D173)."""
    reference = run("ref", handover_events())
    verifier = derive(reference)
    gates = {g.stage: g for g in S.validate_verifier(verifier, reference, write_tools=WRITE_TOOLS)}
    assert gates["verifier_mutation"].passed, gates["verifier_mutation"].failures
    assert gates["verifier_oracle"].passed


# --- rule 3: the mined Intent is not a user ---------------------------------

def audit_events(question: str) -> list[dict]:
    return [
        user(question),
        call(ROSTER, {"scope": "desk"}, kind="read", cid="c0"),
        result({"free_lockers": 12, "audit_age": 30}, cid="c0"),
        says("There are 12 free lockers left, and the last audit was 30 days ago."),
    ]


INTENT = "tell the member there are 12 free lockers and that the last audit was 30 days ago"


def _kind_of(verifier: Any, text: str) -> list[str]:
    return [a.kind for a in verifier.atoms if S.atom_payload(a).get("text") == text
            and S.atom_payload(a).get("kind") in ("communicate", V.REPORTED_COMMUNICATE)]


def test_a_number_only_the_mined_intent_repeats_does_not_make_a_fact_a_demand():
    """The Intent is mined from the recordings (D83), so a number in it may be one the desk found."""
    verifier = derive(run("ref", audit_events("How many free lockers are left?")), intent=INTENT)
    assert _kind_of(verifier, "30") == ["allowed"]


def test_the_same_number_is_a_demand_when_a_user_of_the_task_said_it():
    verifier = derive(run("ref", audit_events("How many free lockers are left, and was the "
                                              "last audit 30 days ago?")), intent=INTENT)
    assert _kind_of(verifier, "30") == ["communicate"]
