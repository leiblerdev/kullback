"""A Verifier has to demand something of its own Task (D285), and every demanded value is swapped (D286).

Invented world: a desk that looks bookings up and moves seats. Every row shares a status, every date
the same year and the same midnight, so a Verifier that demands only those passes any Run that
writes nothing and mentions a date, whichever booking it answered about.
"""

from __future__ import annotations

import pytest

from gates.verifier_fixtures import assistant, call, make_run, result, user, write_events_jsonl
from kullback.examiner import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules
from kullback.runner.records import Event, Run, Task, Verifier

LOOKUP, MOVE = "lookup_booking", "move_seat"
ROWS = {
    "RX7Q2P": {"booking_id": "RX7Q2P", "ticket": "TK-5501234", "travel_date": "2024-05-15T00:00:00",
               "status": "confirmed", "seat": "14C"},
    "HT3K9L": {"booking_id": "HT3K9L", "ticket": "TK-7702211", "travel_date": "2024-06-02T00:00:00",
               "status": "confirmed", "seat": "3A"},
    "MV8D1Q": {"booking_id": "MV8D1Q", "ticket": "TK-1180044", "travel_date": "2024-07-21T00:00:00",
               "status": "confirmed", "seat": "22F"},
    "ZP4W6N": {"booking_id": "ZP4W6N", "ticket": "TK-9034567", "travel_date": "2024-08-09T00:00:00",
               "status": "confirmed", "seat": "9D"},
}
TASK = Task(id="t1", intent="look up the caller's latest booking")
FN = S.canon_fn(CanonRules())


def _with_world(run: Run) -> Run:
    """The Run with the Starting state its stop event carries: every booking of the world."""
    stop = Event(idx=len(run.events), type="stop", payload={"start_state": {"bookings": ROWS}})
    return run.model_copy(update={"events": [*run.events, stop]})


def lookup_run(run_id: str, code: str) -> Run:
    """A read-only answer about one booking: its code and seat, its date and midnight, its status."""
    row = ROWS[code]
    return _with_world(make_run(run_id, [
        user("Can you look up my latest booking?"),
        call(LOOKUP, {"customer": "me"}, kind="read", cid="c0"),
        result(row, cid="c0"),
        assistant(f"Your booking {code} is {row['status']} for {row['travel_date'][:10]} at 00:00, "
                  f"seat {row['seat']}."),
    ]))


def move_run(run_id: str = "ref") -> Run:
    """A write on one booking's row: the seat of RX7Q2P moves to 15A."""
    return _with_world(make_run(run_id, [
        user("Please move my seat to 15A."),
        call(LOOKUP, {"customer": "me"}, kind="read", cid="c0"),
        result(ROWS["RX7Q2P"], cid="c0"),
        call(MOVE, {"booking_id": "RX7Q2P", "seat": "15A"}, cid="c1"),
        result({"ok": True}, cid="c1"),
        assistant("Done, you are in 15A."),
    ]))


def _gates(verifier: Verifier, reference: Run) -> dict:
    return {g.stage: g for g in S.validate_verifier(verifier, reference, canon=CanonRules(), write_tools={MOVE})}


def _fact(atom_id: str, text: str) -> S.Atom:
    return S.make_atom(atom_id, "communicate", {"kind": "communicate", "value": S._key(FN, text), "text": text},
                       provenance="system_derived")


CAP_ZERO = S.make_atom("entity_count", "required", {"kind": "entity_count", "count": 0})


def test_a_seed_sharing_only_generic_values_leaves_the_references_own_facts_demanded(tmp_path):
    """The seed answered about another booking: the two agree on midnight alone, which is no agreement."""
    seed = write_events_jsonl(lookup_run("seed", "HT3K9L"), tmp_path / "seed.jsonl")
    verifier = V.derive_verifier(TASK, lookup_run("ref", "RX7Q2P"), [seed], CanonRules(), write_tools={MOVE})
    demanded = {S.atom_payload(a)["text"] for a in verifier.atoms if a.kind == "communicate"}
    assert demanded == {"RX7Q2P", "14C"}
    reported = {S.atom_payload(a)["text"]: S.atom_payload(a) for a in verifier.atoms
                if S.atom_payload(a).get("kind") == V.REPORTED_COMMUNICATE}
    assert set(reported) == {"00"} and reported["00"]["generic"] == S.FRAGMENT
    assert all(S.atom_payload(a).get("field") == "booking_id" for a in verifier.atoms
               if S.atom_payload(a).get("text") == "RX7Q2P")


@pytest.mark.parametrize("atoms, generic", [
    pytest.param([_fact("c0", "00"), CAP_ZERO], ["c0", "entity_count"], id="zeros_of_a_time"),
    pytest.param([_fact("c0", "2024"), CAP_ZERO], ["c0", "entity_count"], id="year_of_a_date"),
    pytest.param([_fact("c0", "15")], ["c0"], id="day_of_a_date"),
    pytest.param([_fact("c0", "2024-05-15"), CAP_ZERO], ["c0", "entity_count"], id="date_cut_from_a_timestamp"),
    pytest.param([_fact("c0", "5501234"), CAP_ZERO], ["c0", "entity_count"], id="digits_of_a_prefixed_id"),
    pytest.param([_fact("c0", "confirmed"), CAP_ZERO], ["c0", "entity_count"], id="value_most_rows_hold"),
    pytest.param([CAP_ZERO], ["entity_count"], id="write_count_zero_only"),
    pytest.param([V.no_write_atom({MOVE}), CAP_ZERO], [V.NO_WRITE_ATOM, "entity_count"], id="nothing_written_claim"),
])
def test_a_verifier_demanding_only_generic_values_or_no_writes_is_refused_and_names_each_generic_atom(atoms, generic):
    gate = _gates(Verifier(task_id="t1", atoms=atoms), lookup_run("ref", "RX7Q2P"))["verifier_specific"]
    assert gate.passed is False
    assert sorted(gate.metrics["generic"]) == sorted(generic)
    assert all(f"{atom_id} (" in gate.failures[0] for atom_id in generic)


def test_a_verifier_demanding_the_bookings_own_code_survives_and_fails_the_swap_to_another_booking():
    verifier = Verifier(task_id="t1", atoms=[_fact("c0", "RX7Q2P"), _fact("c1", "00"), CAP_ZERO])
    gates = _gates(verifier, lookup_run("ref", "RX7Q2P"))
    assert gates["verifier_specific"].passed and gates["verifier_specific"].metrics["specific"] == ["c0"]
    wrong = gates["verifier_wrong_run"]
    assert wrong.passed, wrong.failures
    assert wrong.metrics["swaps"] == 1 and wrong.metrics["swaps_passed"] == []


def test_the_swap_probe_refuses_a_verifier_that_accepts_any_value_of_the_field_and_names_the_swap():
    """The write names its tool and no row, so moving another booking's seat passes it."""
    reference = move_run()
    derived = V.derive_verifier(Task(id="t1", intent="move my seat"), reference, [], CanonRules(), write_tools={MOVE})
    assert _gates(derived, reference)["verifier_wrong_run"].passed
    loose = [S.make_atom(a.id, a.kind, dict(S.atom_payload(a), entity="", entity_raw=None, id_field=""))
             if S.atom_payload(a).get("kind") == "write" else a
             for a in derived.atoms if a.kind != "hard" and S.atom_payload(a).get("kind") != "write_value"]
    gate = _gates(Verifier(task_id="t1", atoms=loose), reference)["verifier_wrong_run"]
    assert gate.passed is False
    assert "swap of w0 (booking_id RX7Q2P -> HT3K9L) scored pass" in " ".join(gate.failures)


def test_a_swap_takes_another_value_the_world_holds_under_the_same_field_and_never_an_invented_one():
    reference = move_run()
    derived = V.derive_verifier(Task(id="t1", intent="move my seat"), reference, [], CanonRules(), write_tools={MOVE})
    swaps = S.swapped_runs(derived, reference, FN, S.world_rows([reference]))
    assert {swap for _, swap, _ in swaps} >= {"booking_id RX7Q2P -> HT3K9L"}
    known = {str(value) for row in ROWS.values() for value in row.values()}
    assert all(swap.rsplit(" -> ", 1)[1] in known | {"15A"} for _, swap, _ in swaps)
