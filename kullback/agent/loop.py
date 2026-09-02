"""The loop: one assistant message, its tool calls in order, the queues, until the model stops.

`run_agent_loop` is a stateless async function over an explicit transcript. It holds nothing
between calls; everything it reads and writes is on the LoopState it is handed (the transcript,
the system prompt, the two queues, the cancel flag), and everything it does is an emitted event.
That is what lets the harness, a session store and a frontend all watch the same run without
sharing an object, and what lets two agents take turns on one stream (D128): a run is a function
call, and the transcript is what it changes.

Order inside a run. A turn is one assistant message followed by its tool calls, executed
sequentially in the order the model asked for them. After the batch, the steering queue is
drained and its messages are appended as user messages before the next assistant turn, so a
steer reaches the model at the earliest point where a message can be inserted without breaking a
call and its result apart. When the model stops without tool calls, the follow-up queue is
drained one message at a time, each starting a new turn; a follow-up is delivered only when the
run would otherwise end, which is what the Examiner's findings and a scheduler's next target want
(D123). A steer interrupts; a follow-up waits.

Hooks. A `tool_call` hook sees the call before the tool runs and may return rewritten arguments
or raise; a raise blocks the call, and the model reads an is_error result naming the hook. That
is fail-safe by construction: a hook that crashes blocks the call rather than letting it through,
which is why the write block on `gates/` and `runner/` can be one raising hook (D122). A
`tool_result` hook sees the call and the result and may return a rewritten one, which is how a
gate's ruling reaches the model appended to the result it is about to read (D123).
"""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from kullback.agent.events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
)
from kullback.agent.messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from kullback.agent.tools import ToolRegistry, ToolResult
from kullback.ai.provider import Model, ModelConfig
from kullback.ai.stream import StreamDone, StreamError, error_message, stream

ToolCallHook = Callable[[ToolCall], Union[Optional[dict], Awaitable[Optional[dict]]]]
ToolResultHook = Callable[[ToolCall, ToolResult], Union[Optional[ToolResult], Awaitable[Optional[ToolResult]]]]
Emit = Callable[[AgentEvent], Union[None, Awaitable[None]]]


class CancelToken:
    """A flag the harness sets and the loop reads between steps."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class Hooks:
    """The hooks a run applies, in registration order."""

    tool_call: list[ToolCallHook] = field(default_factory=list)
    tool_result: list[ToolResultHook] = field(default_factory=list)


@dataclass
class LoopState:
    """Everything one run reads and changes. The transcript is mutated in place."""

    messages: list[Message] = field(default_factory=list)
    system: str = ""
    steering: deque[Message] = field(default_factory=deque)
    follow_ups: deque[Message] = field(default_factory=deque)
    cancel: CancelToken = field(default_factory=CancelToken)
    config: Optional[ModelConfig] = None
    max_turns: Optional[int] = None


async def run_agent_loop(
    state: LoopState,
    model: Model,
    tools: ToolRegistry,
    hooks: Optional[Hooks] = None,
    emit: Optional[Emit] = None,
    prompts: Optional[list[Message]] = None,
) -> list[Message]:
    """Run until the model stops and both queues are empty. Returns the messages this run appended."""
    hooks = hooks or Hooks()
    new: list[Message] = []

    async def send(event: AgentEvent) -> None:
        if emit is None:
            return
        result = emit(event)
        if inspect.isawaitable(result):
            await result

    async def append(message: Message) -> None:
        state.messages.append(message)
        new.append(message)
        await send(MessageStart(message=message))
        await send(MessageEnd(message=message))

    await send(AgentStart())
    turn = 0
    pending: list[Message] = list(prompts or [])
    pending.extend(_drain_all(state.steering))
    while True:
        has_more_tools = True
        while has_more_tools or pending:
            turn += 1
            await send(TurnStart(turn=turn))
            for message in pending:
                await append(message)
            pending = []
            if state.cancel.cancelled:
                assistant = error_message("cancelled before the model was called", getattr(model, "name", None))
                await append(assistant)
                await send(TurnEnd(turn=turn, message=assistant))
                await send(AgentEnd(messages=new))
                return new
            if state.max_turns is not None and turn > state.max_turns:
                assistant = error_message(f"stopped after max_turns={state.max_turns}", getattr(model, "name", None))
                await append(assistant)
                await send(TurnEnd(turn=turn, message=assistant))
                await send(AgentEnd(messages=new))
                return new
            assistant = await _stream_assistant(state, model, tools, send)
            state.messages.append(assistant)
            new.append(assistant)
            await send(MessageEnd(message=assistant))
            if assistant.stop_reason == "error":
                await send(TurnEnd(turn=turn, message=assistant))
                await send(AgentEnd(messages=new))
                return new
            results: list[ToolResultMessage] = []
            for call in assistant.tool_calls:
                result = await _execute(call, state, tools, hooks, send)
                results.append(result)
                await append(result)
            await send(TurnEnd(turn=turn, message=assistant, tool_results=results))
            has_more_tools = bool(assistant.tool_calls)
            pending = _drain_all(state.steering)
        if state.follow_ups:
            pending = [state.follow_ups.popleft()]
            continue
        break
    await send(AgentEnd(messages=new))
    return new


def _drain_all(queue: deque[Message]) -> list[Message]:
    drained = list(queue)
    queue.clear()
    return drained


async def _stream_assistant(state: LoopState, model: Model, tools: ToolRegistry, send) -> AssistantMessage:
    """One assistant message through the stream, as message events. Always returns a message."""
    started = False
    final: Optional[AssistantMessage] = None
    schemas = tools.schemas() or None
    async for event in stream(model, state.messages, tools=schemas, system=state.system, config=state.config):
        if isinstance(event, StreamDone):
            final = event.message
            if not started:
                await send(MessageStart(message=final))
        elif isinstance(event, StreamError):
            final = event.error
            if not started:
                await send(MessageStart(message=final))
        else:
            if not started:
                started = True
                await send(MessageStart(message=event.partial))
            await send(MessageUpdate(message=event.partial, stream_event=event))
    if final is None:  # pragma: no cover - the stream always ends in done or error
        final = error_message("the stream ended without a message", getattr(model, "name", None))
        await send(MessageStart(message=final))
    return final


async def execute_tool_call(call: ToolCall, tools: ToolRegistry, hooks: Hooks, send,
                            *, cancelled: bool = False) -> ToolResult:
    """One tool call through the hooks and the registry, bracketed by its two execution events.

    Hooks before, the tool, hooks after; every failure is a result the caller can hand back rather
    than an exception. A `tool_call` hook that raises blocks the call, a `tool_result` hook that
    raises fails the result and not the run, and either may rewrite what it was given by returning
    a value. `cancelled` refuses the tool without running it, the hooks having already been asked.

    This is the loop's own execution, made public because it is the core's contract for what a tool
    call is: builder/agent.py's driver issues the build call through it with no model turn, so the
    two cannot drift apart in hook order or in the events a call emits.
    """
    await send(ToolExecutionStart(tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments)))
    arguments = dict(call.arguments)
    result: Optional[ToolResult] = None
    for hook in hooks.tool_call:
        try:
            rewritten = await _maybe_await(hook(ToolCall(id=call.id, name=call.name, arguments=arguments)))
        except Exception as exc:  # noqa: BLE001 - a raise is the hook's way of saying no
            result = ToolResult(content=f"{call.name} blocked by {_hook_name(hook)}: {exc}", is_error=True)
            break
        if rewritten is not None:
            arguments = dict(rewritten)
    if result is None:
        if cancelled:
            result = ToolResult(content=f"{call.name} not run: the run was cancelled", is_error=True)
        else:
            tool = tools.get(call.name)
            if tool is None:
                result = ToolResult(content=f"no tool named {call.name}", is_error=True)
            else:
                result = await tool.run(arguments)
    for hook in hooks.tool_result:
        try:
            rewritten_result = await _maybe_await(hook(call, result))
        except Exception as exc:  # noqa: BLE001 - fail-safe: a crashing hook fails the result, not the run
            result = ToolResult(
                content=f"{call.name} result rejected by {_hook_name(hook)}: {exc}",
                details=result.details,
                is_error=True,
            )
            continue
        if rewritten_result is not None:
            result = rewritten_result
    await send(ToolExecutionEnd(tool_call_id=call.id, tool_name=call.name, result=result, is_error=result.is_error))
    return result


async def _execute(call: ToolCall, state: LoopState, tools: ToolRegistry, hooks: Hooks, send) -> ToolResultMessage:
    """The loop's own call: `execute_tool_call` plus the transcript message the next turn reads."""
    result = await execute_tool_call(call, tools, hooks, send, cancelled=state.cancel.cancelled)
    return ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=result.content,
        is_error=result.is_error,
        details=result.details,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _hook_name(hook: Callable) -> str:
    return getattr(hook, "hook_name", None) or getattr(hook, "__name__", None) or type(hook).__name__


def interrupted_tool_results(messages: list[Message]) -> list[ToolResultMessage]:
    """The results a transcript is missing: every tool call no result answers, marked as interrupted.

    A run that was cancelled between a call and its result leaves the transcript ending on an
    assistant turn with an unanswered call, which no provider accepts; the harness appends these
    before the next run so the transcript stays valid without rewriting history.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolResultMessage)}
    repairs: list[ToolResultMessage] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls:
            if call.id in answered:
                continue
            answered.add(call.id)
            repairs.append(
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content="tool call interrupted before it returned",
                    is_error=True,
                )
            )
    return repairs


def user_message(content: str, details: Optional[dict] = None) -> UserMessage:
    return UserMessage(content=content, details=details)
