"""Hand-built Run records: the events a Run holds, the Run around them, and one atom of a Verifier."""

from __future__ import annotations

from kullback.runner.records import Atom, Event, Run, Verifier


def event(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def user(text: str) -> dict:
    return event("user_turn", content=text)


def assistant(text: str) -> dict:
    return event("model_call", reply={"content": text})


def call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return event("tool_call", id=cid, name=name, args=args, kind=kind)


def result(data, cid: str = "c1") -> dict:
    return event("tool_result", id=cid, result=data)


def make_run(run_id: str, events: list[dict], task_id: str = "t1", termination_reason: str = "success") -> Run:
    return Run(
        run_id=run_id,
        task_id=task_id,
        termination_reason=termination_reason,
        events=[Event(idx=i, type=e["type"], payload=e["payload"]) for i, e in enumerate(events)],
    )


def atom_by_id(verifier: Verifier, atom_id: str) -> Atom:
    found = [a for a in verifier.atoms if a.id == atom_id]
    assert found, f"no atom {atom_id} in {[a.id for a in verifier.atoms]}"
    return found[0]
