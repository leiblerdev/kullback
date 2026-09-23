"""A listed tool no code can answer ends the Run as Environment cannot answer (G28).

These tests need both frozen patches applied: body-transactions (the transaction
snapshot the Router builds on) and cannot-answer (the new outcome itself). Without
them they skip: the Router knows no cannot-answer route, so every assertion here
would fail for the wrong reason.
"""

from __future__ import annotations

import json
import types
from typing import get_args

import pytest

from kullback.ai.provider import TestModel
from kullback.runner import loop as loop_mod
from kullback.runner import route as route_mod
from kullback.runner.records import ErrorClass, Route, ToolSig, Verifier
from kullback.runner.route import Router
from kullback.runner.state import StateView
from kullback.runner.verdict import verdict

CANNOT_ANSWER_REASON = getattr(route_mod, "CANNOT_ANSWER_REASON", "environment_cannot_answer")

HAS_BODY = hasattr(route_mod, "_snapshot_world")
HAS_CANNOT = (hasattr(route_mod, "refuse_stand_in") and hasattr(route_mod, "CANNOT_ANSWER_REASON")
              and "cannot_answer" in get_args(ErrorClass) and "cannot_answer" in get_args(Route))
pytestmark = pytest.mark.skipif(not (HAS_BODY and HAS_CANNOT),
                                reason="needs the body-transactions and cannot-answer frozen patches applied")


def tools_module() -> types.ModuleType:
    """A compiled toolkit with code for exactly one tool; the ledger tool has none."""
    module = types.ModuleType("ledger_tools")

    def fetch_ledger(state, ledger_id):
        row = state.row("ledgers", ledger_id)
        if row is None:
            raise KeyError(ledger_id)
        return row

    module.fetch_ledger = fetch_ledger
    return module


def shared() -> dict:
    return {"ledgers": {"l1": {"id": "l1", "balance": 40}}}


def sigs() -> list[ToolSig]:
    """The mined surface: the coded tool, a listed tool with no body, and a user-side tool."""
    return [
        ToolSig(name="fetch_ledger"),
        ToolSig(name="quote_ledger"),
        ToolSig(name="tap_counter", callers=["user"]),
    ]


def make_router(**kwargs) -> Router:
    kwargs.setdefault("env_tools_module", tools_module())
    kwargs.setdefault("starting_state", StateView(shared=json.loads(json.dumps(shared()))))
    kwargs.setdefault("tool_sigs", sigs())
    return Router(**kwargs)


def test_a_listed_tool_with_no_code_and_no_recording_ends_the_run(workdir):
    """The Candidate sees no invented answer: the transcript stops at the call."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "quote_ledger", "arguments": {"ledger_id": "l1"}}]},
        {"content": "this reply must never be spoken"},
    ])
    state = loop_mod.new_run_state("r1", workdir=workdir, first_user="quote my ledger")
    loop_mod.run(state, model, tools=[{"name": "fetch_ledger"}, {"name": "quote_ledger"}],
                 router=make_router())
    assert state.run.termination_reason == CANNOT_ANSWER_REASON
    assert state.stopped is True
    roles = [message["role"] for message in state.messages]
    assert roles == ["user", "assistant"]
    assert state.run.route_counts == {"cannot_answer": 1}
    result_event = next(event for event in state.run.events if event.type == "tool_result")
    assert result_event.route == "cannot_answer"
    assert result_event.payload["name"] == "quote_ledger"
    assert result_event.payload["reason"] == CANNOT_ANSWER_REASON
    assert result_event.payload["error"]["class"] == "cannot_answer"


def test_calls_after_a_cannot_answer_in_one_response_are_never_routed(workdir):
    """Greptile P1: the stop takes effect at once, not after the rest of the response."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "quote_ledger", "arguments": {"ledger_id": "l1"}},
                          {"id": "c2", "name": "fetch_ledger", "arguments": {"ledger_id": "l1"}}]},
    ])
    state = loop_mod.new_run_state("r1", workdir=workdir)
    loop_mod.run(state, model, router=make_router())
    assert state.run.termination_reason == CANNOT_ANSWER_REASON
    assert [event.type for event in state.run.events].count("tool_call") == 1
    assert [event.type for event in state.run.events].count("tool_result") == 1
    assert state.run.route_counts == {"cannot_answer": 1}
    assert [message["role"] for message in state.messages] == ["assistant"]


def test_a_cannot_answer_verdict_carries_no_pass_and_no_fail(workdir):
    """The Run leaves the counts: env_error with the environment, not the Candidate, blamed."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "quote_ledger", "arguments": {"ledger_id": "l1"}}]},
    ])
    state = loop_mod.new_run_state("r1", workdir=workdir)
    loop_mod.run(state, model, router=make_router())
    done = verdict(state.run, Verifier(task_id="t", atoms=[]))
    assert done.class_ == "env_error"
    assert done.passed is False
    assert done.cause == "environment"
    assert done.environment_suspected is True
    assert any(note.startswith("environment_cannot_answer:quote_ledger") for note in done.notes)


def test_an_unlisted_or_other_caller_tool_name_is_still_an_ordinary_refusal(workdir):
    """A Candidate mistake stays scoreable: the Run carries on with a refusal it can recover from,
    and a tool listed only to another caller keeps that refusal (D164), not the Environment failing."""
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "conjure_ledger", "arguments": {}}]},
        {"content": "sorry, I cannot do that"},
    ])
    state = loop_mod.new_run_state("r1", workdir=workdir)
    loop_mod.run(state, model, router=make_router())
    assert state.run.termination_reason != CANNOT_ANSWER_REASON
    result_event = next(event for event in state.run.events if event.type == "tool_result")
    assert result_event.payload["error"]["class"] == "tool_not_found"
    assert result_event.route == "code"
    assert state.messages[-1]["role"] == "assistant"
    out = make_router().route("tap_counter", {}, requestor="assistant")
    assert out.route == "code"
    assert out.error is not None and out.error.class_ == "tool_not_found"


def test_a_scoring_caller_refuses_a_router_with_a_stand_in_and_a_listed_tool_never_reaches_it():
    """G28 as a lock, not a convention: the scoring path raises instead of answering by model, and
    even where a stand-in was handed in, a listed tool ends the Run instead of asking the model."""
    router = make_router(stand_in_model=TestModel(['{"quoted": true}']))
    with pytest.raises(ValueError):
        route_mod.refuse_stand_in(router)
    route_mod.refuse_stand_in(make_router())
    model = TestModel(['{"quoted": true}'])
    out = make_router(stand_in_model=model).route("quote_ledger", {"ledger_id": "l1"})
    assert out.route == "cannot_answer"
    assert model.calls == []


def test_a_recorded_answer_still_answers_a_listed_tool_without_code():
    """The recording stays the second route: only no code and no recording ends the Run."""
    router = make_router()
    recordings = [{"tool": "quote_ledger", "args": {"ledger_id": "l1"},
                   "state_hash": router.state_hash(), "result": {"quoted": 40}, "error": None,
                   "writes": {}}]
    router = make_router(recordings=recordings)
    out = router.route("quote_ledger", {"ledger_id": "l1"})
    assert out.route == "recording"
    assert out.result == {"quoted": 40}


def test_a_body_fault_and_a_cannot_answer_read_apart():
    """G27 and G28 are siblings in the record: a fault continues the Run, cannot-answer ends it."""
    module = tools_module()

    def quote_ledger(state, ledger_id):
        raise RuntimeError("boom")

    module.quote_ledger = quote_ledger
    out = make_router(env_tools_module=module).route("quote_ledger", {"ledger_id": "l1"})
    assert out.error is not None and out.error.class_ == "body_fault"
    assert out.route == "code"

    del module.quote_ledger
    out = make_router().route("quote_ledger", {"ledger_id": "l1"})
    assert out.route == "cannot_answer"
    assert out.error is not None and out.error.class_ == "cannot_answer"
