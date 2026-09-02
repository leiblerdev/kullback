"""The loop, asserted on the event sequence it emits and the transcript it leaves."""

from __future__ import annotations

import asyncio
from collections import deque

from kullback.agent.events import AgentEnd, ToolExecutionEnd, TurnEnd
from kullback.agent.loop import Hooks, LoopState, interrupted_tool_results, run_agent_loop
from kullback.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from kullback.agent.tools import AgentTool, ToolRegistry, ToolResult
from kullback.ai.provider import TestModel
from tests.agent.conftest import AddArgs, AddResult, call, reply, types_of


def run(model, tools=None, hooks=None, prompts=None, state=None):
    """Run the loop over a fresh state with an event collector; returns (events, state)."""
    state = state or LoopState(system="sys")
    events = []
    asyncio.run(
        run_agent_loop(
            state,
            model,
            ToolRegistry(list(tools or [])),
            hooks,
            events.append,
            prompts=prompts if prompts is not None else [UserMessage(content="go")],
        )
    )
    return events, state


def test_a_plain_answer_emits_one_turn_of_message_events_and_leaves_the_answer_in_the_transcript():
    events, state = run(TestModel(["done."]))
    assert types_of(events) == [
        "agent_start",
        "turn_start",
        "message_start",  # the prompt
        "message_end",
        "message_start",  # the assistant, partial
        "message_update",  # start
        "message_update",  # text_start
        "message_update",  # text_delta
        "message_update",  # text_end
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.messages[-1].content == "done."
    assert isinstance(events[-1], AgentEnd) and len(events[-1].messages) == 2
    # message_update carries the stream event and the partial at that point
    updates = [e for e in events if e.type == "message_update"]
    assert [u.stream_event.type for u in updates] == ["start", "text_start", "text_delta", "text_end"]
    assert updates[-1].message.content == "done."


def test_a_tool_call_runs_and_its_result_reaches_the_transcript_and_the_next_model_call(add_tool):
    model = TestModel([reply("adding", call("add", {"a": 2, "b": 3})), reply("five")])
    events, state = run(model, [add_tool])
    kinds = types_of(events)
    assert kinds[: kinds.index("tool_execution_start")].count("turn_start") == 1
    assert [k for k in kinds if k.startswith("tool_") or k in ("turn_start", "turn_end", "agent_end")] == [
        "turn_start",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "turn_end",
        "agent_end",
    ]
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.tool_name == "add" and end.is_error is False and end.result.details == {"total": 5}
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant"]
    tool_message = state.messages[2]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.tool_call_id == state.messages[1].tool_calls[0].id
    assert tool_message.content == '{"total": 5}' and tool_message.is_error is False
    first_turn_end = next(e for e in events if isinstance(e, TurnEnd))
    assert first_turn_end.turn == 1 and first_turn_end.tool_results == [tool_message]
    # the second model call saw the tool result on the wire, without details
    second = model.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["content"] == '{"total": 5}'
    assert model.calls[0]["tools"][0]["name"] == "add"


def test_tool_calls_run_sequentially_in_order(add_tool, echo_tool):
    model = TestModel([reply(None, call("echo", {"text": "b"}), call("add", {"a": 1, "b": 1})), reply("ok")])
    events, state = run(model, [add_tool, echo_tool])
    starts = [e.tool_name for e in events if e.type == "tool_execution_start"]
    ends = [e.tool_name for e in events if e.type == "tool_execution_end"]
    assert starts == ends == ["echo", "add"]
    order = types_of([e for e in events if e.type.startswith("tool_execution")])
    assert order == ["tool_execution_start", "tool_execution_end"] * 2
    assert [m.tool_name for m in state.messages if isinstance(m, ToolResultMessage)] == ["echo", "add"]


def test_validation_error_becomes_an_is_error_result(add_tool):
    model = TestModel([reply(None, call("add", {"a": "two", "b": 3})), reply("sorry")])
    events, state = run(model, [add_tool])
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error is True
    assert "invalid arguments for add" in end.result.content
    tool_message = state.messages[2]
    assert tool_message.is_error is True and tool_message.content == end.result.content
    assert state.messages[-1].content == "sorry"


def test_unknown_tool_is_an_error_result():
    model = TestModel([reply(None, call("nope", {})), reply("ok")])
    events, state = run(model, [])
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error and end.result.content == "no tool named nope"


def test_raising_tool_call_hook_blocks_the_call_and_names_the_hook():
    ran = []

    async def counted(args):
        ran.append(args)
        return {"total": 0}

    counting_tool = AgentTool("add", "Add two integers.", AddArgs, AddResult, counted)

    def no_writes(call):
        raise PermissionError("gates/ is read-only")

    model = TestModel([reply(None, call("add", {"a": 1, "b": 2})), reply("ok")])
    events, state = run(model, [counting_tool], Hooks(tool_call=[no_writes]))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error is True
    assert end.result.content == "add blocked by no_writes: gates/ is read-only"
    assert ran == []
    assert state.messages[2].is_error and "no_writes" in state.messages[2].content


def test_tool_call_hook_can_rewrite_arguments(add_tool):
    def double(call):
        return {"a": call.arguments["a"] * 2, "b": call.arguments["b"]}

    async def noop(call):
        return None

    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("ok")])
    events, _ = run(model, [add_tool], Hooks(tool_call=[double, noop]))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.result.details == {"total": 3}
    start = next(e for e in events if e.type == "tool_execution_start")
    assert start.arguments == {"a": 1, "b": 1}  # what the model asked, before the hook


def test_tool_result_hook_rewrites_the_result(add_tool):
    def append_ruling(call, result):
        return ToolResult(
            content=result.content + "\ngate: accepted",
            details={**(result.details or {}), "ruling": "accepted"},
            is_error=result.is_error,
        )

    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("ok")])
    events, state = run(model, [add_tool], Hooks(tool_result=[append_ruling]))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.result.content == '{"total": 2}\ngate: accepted'
    assert end.result.details == {"total": 2, "ruling": "accepted"}
    assert state.messages[2].content.endswith("gate: accepted")
    assert state.messages[2].details["ruling"] == "accepted"
    # the model read the appended ruling and not the details
    seen = model.calls[1]["messages"][-1]
    assert "gate: accepted" in seen["content"] and "ruling" not in seen["content"]


def test_raising_tool_result_hook_fails_the_result_not_the_run(add_tool):
    def crash(call, result):
        raise ValueError("cannot read ruling")

    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("ok")])
    events, state = run(model, [add_tool], Hooks(tool_result=[crash]))
    end = next(e for e in events if isinstance(e, ToolExecutionEnd))
    assert end.is_error and "rejected by crash" in end.result.content
    assert end.result.details == {"total": 2}
    assert state.messages[-1].content == "ok"


def test_steer_is_delivered_after_the_tool_batch_before_the_next_turn():
    state = LoopState(system="sys")
    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("ok")])

    async def steer_during_tool(args):
        state.steering.append(UserMessage(content="stop and report"))
        return {"total": 2}

    steering_tool = AgentTool("add", "Add two integers.", AddArgs, AddResult, steer_during_tool)
    events, state = run(model, [steering_tool], state=state)
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert state.messages[3].content == "stop and report"
    # the steer's message events come after turn_end of turn 1 and after turn_start of turn 2
    kinds = types_of(events)
    turn2 = [i for i, k in enumerate(kinds) if k == "turn_start"][1]
    first_after = next(i for i in range(turn2, len(events)) if events[i].type == "message_start")
    assert events[first_after].message.content == "stop and report"
    # and the model saw it right after the tool result
    wire = model.calls[1]["messages"]
    assert [m["role"] for m in wire[-2:]] == ["tool", "user"]
    assert wire[-1] == {"role": "user", "content": "stop and report"}


def test_steer_queued_before_the_run_goes_in_with_the_prompt():
    state = LoopState(system="sys", steering=deque([UserMessage(content="also this")]))
    events, state = run(TestModel(["ok"]), state=state)
    assert [m.role for m in state.messages] == ["user", "user", "assistant"]
    assert [m.content for m in state.messages[:2]] == ["go", "also this"]
    assert types_of(events).count("turn_start") == 1


def test_follow_up_is_delivered_when_the_run_would_otherwise_stop():
    state = LoopState(
        system="sys",
        follow_ups=deque([UserMessage(content="second"), UserMessage(content="third")]),
    )
    model = TestModel(["one", "two", "three"])
    events, state = run(model, state=state)
    assert [m.content for m in state.messages] == ["go", "one", "second", "two", "third", "three"]
    assert types_of(events).count("turn_start") == 3
    turn_ends = [e for e in events if isinstance(e, TurnEnd)]
    assert [t.turn for t in turn_ends] == [1, 2, 3]
    assert events[-1].type == "agent_end" and len(events[-1].messages) == 6
    # each follow-up was its own model call on the transcript so far
    assert len(model.calls) == 3
    assert model.calls[1]["messages"][-1] == {"role": "user", "content": "second"}


def test_follow_up_is_not_delivered_while_tools_are_still_running(add_tool):
    state = LoopState(system="sys", follow_ups=deque([UserMessage(content="later")]))
    model = TestModel([reply(None, call("add", {"a": 1, "b": 1})), reply("done"), reply("after")])
    events, state = run(model, [add_tool], state=state)
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert state.messages[4].content == "later"


def test_a_steer_on_the_stopping_turn_goes_before_a_waiting_follow_up():
    state = LoopState(system="sys", follow_ups=deque([UserMessage(content="follow-up")]))
    model = TestModel(["one", "two", "three"])
    steered = []

    def on_event(event):
        if event.type == "turn_end" and event.turn == 1 and not steered:
            steered.append(True)
            state.steering.append(UserMessage(content="steer"))

    events = []

    def emit(event):
        events.append(event)
        on_event(event)

    asyncio.run(run_agent_loop(state, model, ToolRegistry([]), None, emit, prompts=[UserMessage(content="go")]))
    assert [m.content for m in state.messages] == ["go", "one", "steer", "two", "follow-up", "three"]
    assert types_of(events).count("turn_start") == 3


def test_model_error_ends_the_run_and_stays_in_the_transcript():
    class Broken(TestModel):
        def query(self, messages, tools=None, config=None):
            raise RuntimeError("provider down")

    events, state = run(Broken(["unused"]))
    assert types_of(events)[-3:] == ["message_end", "turn_end", "agent_end"]
    last = state.messages[-1]
    assert isinstance(last, AssistantMessage) and last.stop_reason == "error"
    assert "provider down" in (last.error_message or "")


def test_max_turns_stops_with_an_error_message():
    state = LoopState(system="sys", max_turns=1)
    model = TestModel(["a", "b"], loop=True)
    state.follow_ups.append(UserMessage(content="more"))
    events, state = run(model, state=state)
    assert state.messages[-1].stop_reason == "error"
    assert "max_turns=1" in state.messages[-1].error_message
    assert len(model.calls) == 1


def test_cancel_stops_before_the_next_model_call_and_marks_tools_not_run():
    state = LoopState(system="sys")
    model = TestModel([reply(None, call("add", {"a": 1, "b": 1}), call("add", {"a": 2, "b": 2})), reply("never")])

    async def cancel_then_answer(args):
        state.cancel.cancel()
        return {"total": args.a + args.b}

    cancelling_tool = AgentTool("add", "Add two integers.", AddArgs, AddResult, cancel_then_answer)
    events, state = run(model, [cancelling_tool], state=state)
    results = [m for m in state.messages if isinstance(m, ToolResultMessage)]
    assert results[0].is_error is False
    assert results[1].is_error is True and "cancelled" in results[1].content
    assert state.messages[-1].stop_reason == "error" and "cancelled" in state.messages[-1].error_message
    assert len(model.calls) == 1


def test_interrupted_tool_results_answer_dangling_calls():
    messages = [
        UserMessage(content="go"),
        AssistantMessage(tool_calls=[{"id": "c1", "name": "t", "arguments": {}}, {"id": "c2", "name": "t", "arguments": {}}]),
        ToolResultMessage(tool_call_id="c1", tool_name="t", content="ok"),
    ]
    repairs = interrupted_tool_results(messages)
    assert [r.tool_call_id for r in repairs] == ["c2"]
    assert repairs[0].is_error and "interrupted" in repairs[0].content
    assert interrupted_tool_results(messages + repairs) == []


def test_no_prompts_and_empty_queues_still_asks_the_model_once():
    events, state = run(TestModel(["hi"]), prompts=[])
    assert [m.role for m in state.messages] == ["assistant"]
    assert types_of(events).count("turn_start") == 1
