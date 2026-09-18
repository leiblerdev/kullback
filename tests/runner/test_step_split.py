"""The Runner's step is two functions: ask the policy, advance the world (G2).

Needs the step-split patch (docs/frozen-patches/step-split.patch): ask and advance live in
the frozen runner tree, so this module skips unless the patch is applied.
"""

from __future__ import annotations

import copy
import inspect
import types

import pytest

from kullback.ai.provider import TestModel
from kullback.runner import loop
from kullback.runner.loop import finish, new_run_state, run, step
from kullback.runner.records import ToolSig, as_dict
from kullback.runner.route import Router
from kullback.runner.state import StateView

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)


WIDGET_TOOLS = [{"name": "describe_widget"}, {"name": "rename_widget"}]


def tools_module() -> types.ModuleType:
    """Two invented widget tools: one read, one write that refuses an empty label."""
    module = types.ModuleType("step_split_tools")

    def describe_widget(state, widget_id):
        row = state.row("widgets", widget_id)
        if row is None:
            raise KeyError(widget_id)
        return row

    def rename_widget(state, widget_id, label):
        if not label:
            raise ValueError("a widget needs a label")
        row = dict(state.row("widgets", widget_id) or {})
        row["label"] = label
        state.shared.setdefault("widgets", {})[str(widget_id)] = row
        return row

    module.describe_widget = describe_widget
    module.rename_widget = rename_widget
    return module


def make_router(**kwargs) -> Router:
    return Router(
        env_tools_module=tools_module(),
        recordings=None,
        starting_state=StateView(shared={"widgets": {
            "w1": {"id": "w1", "label": "plain", "status": "on shelf"}}}),
        overlay=None,
        tool_sigs=[ToolSig(name="describe_widget"), ToolSig(name="rename_widget")],
        **kwargs,
    )


class Poison:
    """Any touch explodes: proves which half uses the router, the user or the model."""

    def __getattr__(self, name):
        raise AssertionError(f"must not touch {name}")


class ScriptedUser:
    def __init__(self, answers):
        self.answers = list(answers)
        self.done = False

    def reply(self, transcript):
        return self.answers.pop(0) if self.answers else "###STOP###"


def state_shape(state) -> dict:
    return {
        "messages": copy.deepcopy(state.messages),
        "tools": copy.deepcopy(state.tools),
        "turn": state.turn,
        "stopped": state.stopped,
        "termination_reason": state.run.termination_reason,
        "route_counts": dict(sorted(state.run.route_counts.items())),
        "assisted": state.run.assisted,
        "events": [as_dict(e) for e in state.run.events],
    }


def world_shape(state) -> dict:
    """Everything advance owns: the transcript, the world events, the stops and counts.

    The model_call events stay with ask: advance never calls the model, so a hand-written
    message cannot reproduce their priced reply payload. The check mirrors the trainer
    flow instead: the policy has spoken (its message is already in the transcript) and
    advance moves the world from there.
    """
    shape = state_shape(state)
    shape["events"] = [
        {k: v for k, v in e.items() if k != "idx"}  # idx counts ask's calls too
        for e in shape["events"] if e["type"] != "model_call"
    ]
    return shape


def given(message: dict, state, router) -> None:
    """Mirror ask's transcript half by hand, then advance the world."""
    state.messages.append(copy.deepcopy(message))
    state.tools = list(WIDGET_TOOLS)
    state.turn += 1
    loop.advance(state, message, router)


def test_advance_of_a_hand_written_tool_message_matches_step(workdir):
    reply = {"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]}
    whole = new_run_state("r1", workdir=workdir, first_user="what is on shelf w1")
    run(whole, TestModel([reply, {"content": "here it is."}]),
        tools=WIDGET_TOOLS, router=make_router())

    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]}
    state = new_run_state("r2", first_user="what is on shelf w1")
    router = make_router()
    given(message, state, router)
    given({"role": "assistant", "content": "here it is.", "tool_calls": []}, state, router)
    finish(state, router)

    assert world_shape(state) == world_shape(whole)


def test_advance_of_a_hand_written_text_turn_matches_step_with_a_user(workdir):
    whole = new_run_state("r1", workdir=workdir, first_user="what is on shelf w1",
                          user=ScriptedUser(["w1 please."]))
    run(whole, TestModel([{"content": "which one?"}], loop=True),
        tools=WIDGET_TOOLS, router=make_router())

    state = new_run_state("r2", first_user="what is on shelf w1",
                          user=ScriptedUser(["w1 please."]))
    router = make_router()
    text = {"role": "assistant", "content": "which one?", "tool_calls": []}
    given(text, state, router)
    given(text, state, router)  # the looping model asks again; the user ends it
    finish(state, router)

    assert world_shape(state) == world_shape(whole)


def test_advance_of_several_calls_and_a_refusal_matches_step():
    replies = [
        {"content": None, "tool_calls": [
            {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}},
            {"id": "c2", "name": "rename_widget",
             "arguments": {"widget_id": "w1", "label": ""}},
            {"id": "c3", "name": "levitate_widget", "arguments": {}}]},
        {"content": "nothing worked."},
    ]
    whole = new_run_state("r1", first_user="what is on shelf w1")
    run(whole, TestModel(replies), tools=WIDGET_TOOLS, router=make_router())

    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}},
        {"id": "c2", "name": "rename_widget", "arguments": {"widget_id": "w1", "label": ""}},
        {"id": "c3", "name": "levitate_widget", "arguments": {}}]}
    state = new_run_state("r2", first_user="what is on shelf w1")
    router = make_router()
    given(message, state, router)
    given({"role": "assistant", "content": "nothing worked.", "tool_calls": []}, state, router)
    finish(state, router)

    assert world_shape(state) == world_shape(whole)


def test_advance_never_touches_a_model():
    """Advance takes no model: the policy's answer arrives as a message."""
    assert "model" not in inspect.signature(loop.advance).parameters
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]}
    state = new_run_state("r1", first_user="what is on shelf w1")
    state.tools = list(WIDGET_TOOLS)
    state.turn += 1
    model, poison = None, Poison()
    loop.advance(state, message, make_router())
    assert model is None
    with pytest.raises(AssertionError):
        poison.query([])  # the poison explodes when touched; advance never touched it
    assert [e.type for e in state.run.events] == ["tool_call", "tool_result"]
    assert "on shelf" in state.messages[-1]["content"]


def test_ask_never_touches_the_router_or_the_user():
    """Ask takes neither: the world half belongs to advance."""
    assert "router" not in inspect.signature(loop.ask).parameters
    assert "user" not in inspect.signature(loop.ask).parameters
    reply = {"content": None, "tool_calls": [
        {"id": "c1", "name": "rename_widget",
         "arguments": {"widget_id": "w1", "label": "shiny"}}]}
    state = new_run_state("r1", first_user="what is on shelf w1", user=Poison())
    message = loop.ask(state, TestModel([reply]), WIDGET_TOOLS)
    assert message == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "name": "rename_widget",
         "arguments": {"widget_id": "w1", "label": "shiny"}}]}
    assert [e.type for e in state.run.events] == ["model_call"]
    assert state.messages[-1] == message
    assert state.tools == WIDGET_TOOLS
    assert state.turn == 1


def test_ask_then_advance_is_step_event_for_event():
    state = new_run_state("r1", first_user="what is on shelf w1")
    router = make_router()
    model = TestModel([{"content": "w1 is on the shelf."}])
    message = loop.ask(state, model, WIDGET_TOOLS)
    loop.advance(state, message, router)

    whole = new_run_state("r2", first_user="what is on shelf w1")
    step(whole, TestModel([{"content": "w1 is on the shelf."}]),
         WIDGET_TOOLS, make_router())

    assert state_shape(state) == state_shape(whole)
