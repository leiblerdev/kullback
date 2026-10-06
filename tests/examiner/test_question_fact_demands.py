"""Question demands stay general (G5a).

Invented tools only. A field every seed could read from the world keeps no
question atom.
"""

from __future__ import annotations

from gates.verifier_fixtures import (
    assistant,
    call,
    make_run,
    result,
    user,
    write_events_jsonl,
)
from kullback import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules
from kullback.runner.records import Task

WRITE_TOOLS = {"shelve_parcel"}
PARCEL = {"parcel_id": "P-7", "slot": "morning", "weight": "2kg"}


def _seed(run_id: str, slot_line: str, final: str, stock: dict) -> object:
    return make_run(run_id, [
        user("Shelve my parcel."),
        assistant("Which slot should I shelve it in?"),
        user(slot_line),
        call("fetch_parcel", {"parcel_id": "P-7"}, kind="read", cid="c0"),
        result(stock, cid="c0"),
        call("shelve_parcel", {"parcel_id": "P-7", "slot": "morning"}, cid="c1"),
        result({"parcel_id": "P-7", "slot": "morning"}, cid="c1"),
        assistant(final),
    ], task_id="t2")


def _derive(tmp_path, seeds, intent="shelve the parcel"):
    ref, *reruns = seeds
    paths = [write_events_jsonl(r, tmp_path / f"{r.run_id}.jsonl") for r in reruns]
    return V.derive_verifier(Task(id="t2", intent=intent), ref, paths,
                             CanonRules(), write_tools=WRITE_TOOLS)


def _question_keys(verifier) -> list:
    return sorted(p["key"] for p in map(S.atom_payload, verifier.atoms) if p.get("kind") == "question")


def test_a_field_every_seed_read_from_the_world_keeps_no_question_atom(tmp_path):
    """G5a: both seeds asked, both could have read the slot; reading or asking are both right."""
    seeds = [_seed("ref", "Morning please, parcel P-7.", "Done.", PARCEL),
             _seed("alt", "The morning slot for P-7, thanks.", "Done.", PARCEL)]
    assert _question_keys(_derive(tmp_path, seeds)) == []


def test_a_field_no_seed_could_read_keeps_its_question_atom(tmp_path):
    """Risk side: where the world never held the slot, asking stays demanded."""
    stock = {"parcel_id": "P-7", "weight": "2kg"}
    seeds = [_seed("ref", "Morning please, parcel P-7.", "Done.", stock),
             _seed("alt", "The morning slot for P-7, thanks.", "Done.", stock)]
    assert _question_keys(_derive(tmp_path, seeds)) == ["field:slot"]


def test_a_field_one_seed_could_not_read_keeps_its_question_atom(tmp_path):
    """The demand stands when some seed could not read the slot first."""
    seeds = [_seed("ref", "Morning please, parcel P-7.", "Done.", PARCEL),
             _seed("alt", "The morning slot for P-7, thanks.", "Done.",
                   {"parcel_id": "P-7", "weight": "2kg"})]
    assert _question_keys(_derive(tmp_path, seeds)) == ["field:slot"]

