"""The section builders derive_verifier assembles, pinned through derive_verifier.

The split gives each atom section its own builder, called in a fixed order;
a Task carries no kind field, so there is no kind dispatch to hold. Every
test here goes through derive_verifier and asserts the outcome. The domain is
a reading room: a member reserves a study desk, the chart says which desks
are free, and the desk writes the reservation. Nothing here names a customer
tool or column.
"""

from __future__ import annotations

from kullback.examiner import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.records import Constraint, Event, Run, Task

WRITE_TOOLS = {"reserve_desk", "assign_desk", "sell_ticket"}
TASK = Task(id="t9", intent="reserve a quiet desk for Thursday and say which one")


def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def _run(run_id: str, events: list[dict], task_id: str = "t9") -> Run:
    return Run(run_id=run_id, task_id=task_id, termination_reason="success",
               events=[Event(idx=i, type=e["type"], payload=e["payload"])
                       for i, e in enumerate(events)])


def desk_events(reason: str = "thesis work") -> list[dict]:
    return [
        _ev("user_turn", content="Please reserve desk D7 for Thursday."),
        _ev("model_call", reply={"content": "Sure. What is the visit for?"}),
        _ev("user_turn", content=reason),
        _ev("tool_call", id="c0", name="desk_chart", args={"desk_id": "D7"}, kind="read"),
        _ev("tool_result", id="c0", result={"desk_id": "D7", "quiet": True}),
        _ev("tool_call", id="c1", name="reserve_desk",
            args={"desk_id": "D7", "reason": reason}, kind="write"),
        _ev("tool_result", id="c1", result={"desk_id": "D7", "held": True}),
        _ev("model_call", reply={"content": "Desk D7 is yours for Thursday."}),
    ]


def ticket_events() -> list[dict]:
    """No writes; the answer states three facts the vague request names none of."""
    return [
        _ev("user_turn", content="Can you look into my ticket for me?"),
        _ev("tool_call", id="r0", name="get_ticket_record",
            args={"ticket_id": "T-41"}, kind="read"),
        _ev("tool_result", id="r0",
            result={"ticket_id": "T-41", "hall": "B", "fee": 12.5}),
        _ev("model_call", reply={"content": "Ticket T-41 for hall B carries a fee of 12.5."}),
    ]


def caps(verifier) -> list:
    return [a for a in verifier.atoms if S.atom_payload(a)["kind"] == "entity_count"]


def test_task_id_reads_all_three_caller_shapes():
    assert V.derive_verifier("t9", _run("ref", desk_events()), [], None,
                             write_tools=WRITE_TOOLS).task_id == "t9"
    assert V.derive_verifier(TASK, _run("ref", desk_events()), [], None,
                             write_tools=WRITE_TOOLS).task_id == "t9"
    assert V.derive_verifier(7, _run("ref", desk_events()), [], None,
                             write_tools=WRITE_TOOLS).task_id == "7"


def test_cap_branches():
    """A cap of 1 over the writing run, of 0 over the empty one, and none when
    the Task's own runs contradict a cap nothing else asked for."""
    writing = V.derive_verifier(TASK, _run("ref", desk_events()), [], None,
                                write_tools=WRITE_TOOLS, writes_elsewhere=True)
    assert [S.atom_payload(a)["count"] for a in caps(writing)] == [1]
    empty = V.derive_verifier(TASK, _run("e", []), [], None, write_tools=WRITE_TOOLS)
    assert [S.atom_payload(a)["count"] for a in caps(empty)] == [0]
    vague = Task(id="t9", intent="look into what the caller asked about")
    contradicted = V.derive_verifier(vague, _run("e", [], task_id="t9"), [], None,
                                     write_tools=WRITE_TOOLS, writes_elsewhere=True)
    assert caps(contradicted) == []


def test_constraints_with_code_become_hard_atoms_and_residuals_do_not():
    compiled = Constraint(id="k1", text="never double book a desk",
                          predicate_src="def check(): return True", compiled=True)
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    residual = Constraint(id="k3", text="use good judgement", residual_reason="not checkable")
    verifier = V.derive_verifier(TASK, _run("ref", desk_events()), [], None,
                                 write_tools=WRITE_TOOLS,
                                 constraints=[compiled, judged, residual])
    assert [a.id for a in verifier.atoms if a.kind == "hard"] == ["hard.k1", "hard.k2"]
    plain = V.derive_verifier(TASK, _run("ref", desk_events()), [], None,
                              write_tools=WRITE_TOOLS)
    assert [a for a in plain.atoms if a.kind == "hard"] == []


def test_each_section_owns_one_atom_family_in_order():
    verifier = V.derive_verifier(TASK, _run("ref", desk_events()), [], None,
                                 write_tools=WRITE_TOOLS)
    assert [a.id for a in verifier.atoms] == [
        "w0", "w0.desk_id", "w0.reason", "q.field:reason", "c0", "entity_count"]
    assert verifier.task_id == "t9"
    assert verifier.seed_run_ids == ["ref"]


def test_facts_with_nothing_else_to_fail_are_demanded_again():
    """With no write and no fact named, what the answer stated is the only evidence."""
    vague = Task(id="t7", intent="look into the ticket the caller asked about")
    verifier = V.derive_verifier(vague, _run("kref", ticket_events(), task_id="t7"), [], None,
                                 write_tools=WRITE_TOOLS)
    kinds = {S.atom_payload(a)["kind"] for a in verifier.atoms}
    assert V.REPORTED_COMMUNICATE not in kinds
    demanded = sorted(S.atom_payload(a)["text"] for a in verifier.atoms
                      if S.atom_payload(a)["kind"] == "communicate")
    # Hall B stays out: a single character is never a fact, so nothing states it.
    assert demanded == ["12.5", "T-41"]
