"""The per-kind builders derive_verifier assembles: each one owns one atom family.

The derivation moved to one builder per atom section when derive_verifier
outgrew a readable complexity; these tests hold the split. The domain is a
reading room: a member reserves a study desk, the catalog says which desks
are free, and the desk writes the reservation. Nothing here names a
customer tool or column.
"""

from __future__ import annotations

from kullback.examiner import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.records import Constraint, Event, Run, Task

WRITE_TOOLS = {"reserve_desk"}
TASK = Task(id="t9", intent="reserve a quiet desk for Thursday and say which one")


def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def _run(run_id: str, events: list[dict]) -> Run:
    return Run(run_id=run_id, task_id="t9", termination_reason="success",
               events=[Event(idx=i, type=e["type"], payload=e["payload"])
                       for i, e in enumerate(events)])


def desk_events(reason: str = "thesis work") -> list[dict]:
    return [
        _ev("user_turn", content="Please reserve desk D7 for Thursday."),
        _ev("model_call", reply={"content": "Sure. What is the visit for?"}),
        _ev("user_turn", content=reason),
        _ev("tool_call", id="c0", name="desk_catalog", args={"desk_id": "D7"}, kind="read"),
        _ev("tool_result", id="c0", result={"desk_id": "D7", "quiet": True}),
        _ev("tool_call", id="c1", name="reserve_desk",
            args={"desk_id": "D7", "reason": reason}, kind="write"),
        _ev("tool_result", id="c1", result={"desk_id": "D7", "held": True}),
        _ev("model_call", reply={"content": "Desk D7 is yours for Thursday."}),
    ]


def test_task_id_reads_all_three_caller_shapes():
    assert V._task_id("t9") == "t9"
    assert V._task_id(TASK) == "t9"
    assert V._task_id(7) == "7"


def test_demand_kind_names_both_kinds():
    assert V._demand_kind(True) == "required"
    assert V._demand_kind(False) == "allowed"


def test_cap_atoms_cover_the_early_return_and_both_cap_branches():
    assert V._cap_atoms([], False, False) == []
    capped = V._cap_atoms([{"a": 1}, {"a": 1, "b": 2}], False, False)
    assert [a.id for a in capped] == ["entity_count"]
    assert S.atom_payload(capped[0])["count"] == 2
    assert V._cap_atoms([{}], True, False) == []
    kept = V._cap_atoms([{}], True, True)
    assert [a.id for a in kept] == ["entity_count"]


def test_constraint_atoms_keep_code_and_skip_residuals():
    compiled = Constraint(id="k1", text="never double book a desk",
                          predicate_src="def check(): return True", compiled=True)
    judged = Constraint(id="k2", text="be polite", judge_atom=True)
    residual = Constraint(id="k3", text="use good judgement", residual_reason="not checkable")
    assert V._constraint_atoms(None, set(), []) == []
    atoms = V._constraint_atoms([compiled, judged, residual], set(), [])
    assert [a.id for a in atoms] == ["hard.k1", "hard.k2"]


def test_each_builder_owns_one_atom_family_in_section_order():
    verifier = V.derive_verifier(TASK, _run("ref", desk_events()),
                                 [], None, write_tools=WRITE_TOOLS)
    assert [a.id for a in verifier.atoms] == [
        "w0", "w0.desk_id", "w0.reason", "q.field:reason", "c0", "entity_count"]
    assert verifier.task_id == "t9"
    assert verifier.seed_run_ids == ["ref"]
