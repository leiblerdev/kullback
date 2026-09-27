"""A Spec and a Reference built from records only: one user moves one item to a new slot."""

from __future__ import annotations

from kullback.runner.records import Event, Run
from kullback.spec.schema import Check, FactSource, IntentFact, Spec, SpecIntent, intent_text, tier_of

WRITE_TOOLS = {"update_item"}
FACTS = [
    IntentFact(id="f1", text="Please move item A1 to slot seven.", stance="volunteered",
               source=FactSource(recording="rec1", turn=0), witnesses=["rec2"]),
    IntentFact(id="f2", text="I also want a note saying it was urgent.", stance="accepted",
               source=FactSource(recording="rec1", turn=2)),
]
WRITE = {"demand": "write", "tool": "update_item", "entity": "A1", "id_field": "item_id",
         "values": {"slot": "seven"}}


def fn(value):
    return str(value).strip().lower()


def check(check_id="c1", kind="required", demand=None, because="move item A1 to slot seven", fact_ids=("f1",)):
    demand = dict(WRITE if demand is None else demand)
    return Check(id=check_id, kind=kind, demand=demand, because=because,
                 tier=tier_of(kind, demand), fact_ids=list(fact_ids))


def spec(checks=None, gaps=()):
    intent = SpecIntent(task_id="t1", facts=FACTS, text=intent_text(FACTS), model="m", written_at="2026-01-01")
    return Spec(task_id="t1", intent=intent, checks=[check()] if checks is None else checks, gaps=list(gaps))


def _event(idx, type_, **payload):
    return Event(idx=idx, type=type_, payload=payload)


def reference_run() -> Run:
    events = [
        _event(0, "user_turn", content="Please move item A1 to slot seven."),
        _event(1, "tool_call", id="c0", name="list_items", args={}, kind="read"),
        _event(2, "tool_result", id="c0", result=[{"item_id": "A1", "slot": "two"},
                                                   {"item_id": "B2", "slot": "four"}]),
        _event(3, "tool_call", id="c1", name="update_item", args={"item_id": "A1", "slot": "seven"}, kind="write"),
        _event(4, "tool_result", id="c1", result={"item_id": "A1", "slot": "seven"}),
        _event(5, "model_call", reply={"content": "Item A1 is now in slot seven."}),
    ]
    return Run(run_id="ref", task_id="t1", termination_reason="success", events=events)
