"""A provider-neutral stream of one assistant message, over today's non-streaming Model.query.

The event shape follows tau's AssistantMessageEvent: start, then text and tool-call blocks each
opened, filled by deltas and closed, then done with the assembled message, or error. Every event
before done carries the message as assembled so far, so a consumer can render the partial without
keeping its own copy. In this phase every provider answers whole (Model.query returns a ModelReply),
so the stream emits the assembled events at once after the call returns: one delta per text block,
one delta per tool call. A true token stream is a later phase and changes only this file; the loop
and every consumer already read the events, not the reply.

The model call runs in a worker thread so an async loop is not held by the network wait. That is
also what keeps MemoModel, RecordedModel and TestModel working unchanged: each is a plain
Model.query, and the thread is where it is called.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, AsyncIterator, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.messages import AssistantMessage, Message, StopReason, ToolCall, to_wire
from kullback.ai.provider import Model, ModelConfig, ModelReply


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StreamStart(_Event):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStart(_Event):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDelta(_Event):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEnd(_Event):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStart(_Event):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDelta(_Event):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEnd(_Event):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class StreamDone(_Event):
    type: Literal["done"] = "done"
    reason: StopReason
    message: AssistantMessage


class StreamError(_Event):
    type: Literal["error"] = "error"
    reason: Literal["error"]
    error: AssistantMessage


StreamEvent = Annotated[
    Union[
        StreamStart,
        TextStart,
        TextDelta,
        TextEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        StreamDone,
        StreamError,
    ],
    Field(discriminator="type"),
]

# What the adapters report, folded onto the four reasons the loop acts on. Anything unknown is
# read from the shape of the reply: tool calls mean tool_use, otherwise stop.
_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_use",
    "tool_calls": "tool_use",
    "length": "length",
    "max_tokens": "length",
}


def normalize_stop_reason(raw: Optional[str], has_tool_calls: bool) -> StopReason:
    """A provider's stop reason as one of ours; the reply's shape decides when the word is unknown."""
    if raw is not None:
        known = _STOP_REASONS.get(str(raw).lower())
        if known is not None:
            return known
    return "tool_use" if has_tool_calls else "stop"


def assemble(reply: ModelReply, call_id_prefix: str = "call") -> AssistantMessage:
    """One ModelReply as an AssistantMessage. A tool call with no id gets one from its position:
    a scripted model rarely names its calls, and a transcript needs every result to answer an id."""
    calls = [
        ToolCall(
            id=call.id or f"{call_id_prefix}_{index}",
            name=call.name,
            arguments=dict(call.arguments or {}),
        )
        for index, call in enumerate(reply.tool_calls)
    ]
    return AssistantMessage(
        content=reply.content,
        tool_calls=calls,
        usage=reply.usage,
        stop_reason=normalize_stop_reason(reply.stop_reason, bool(calls)),
        model=reply.model,
    )


def error_message(text: str, model: Optional[str] = None) -> AssistantMessage:
    """The assistant message that records a failed call, so the transcript says what happened."""
    return AssistantMessage(content=None, stop_reason="error", error_message=text, model=model)


async def stream(
    model: Model,
    messages: Sequence[Message],
    tools: Optional[Sequence[dict]] = None,
    system: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> AsyncIterator[StreamEvent]:
    """Stream one assistant message for the transcript, over Model.query.

    A provider error becomes an error event carrying an assistant message with stop_reason
    "error"; it never escapes as an exception, so the loop can record it in the transcript. A
    cancellation of the awaiting task does escape: that is the caller stopping the run.
    """
    wire = to_wire(messages, system)
    tool_list = list(tools) if tools else None
    prefix = f"call_{len(messages)}"
    try:
        reply = await asyncio.to_thread(model.query, wire, tool_list, config)
    # CancelledError is a BaseException on the Python this package requires, so the clause below
    # lets a cancellation through on its own; the caller stopping the run is not a provider error.
    except Exception as exc:  # noqa: BLE001 - the provider is an isolation boundary
        error = error_message(f"{type(exc).__name__}: {exc}", model=getattr(model, "name", None))
        yield StreamError(reason="error", error=error)
        return
    final = assemble(reply, call_id_prefix=prefix)
    partial = AssistantMessage(model=final.model, stop_reason="stop")
    yield StreamStart(partial=partial.model_copy(deep=True))
    index = 0
    if final.content:
        yield TextStart(content_index=index, partial=partial.model_copy(deep=True))
        partial.content = final.content
        yield TextDelta(content_index=index, delta=final.content, partial=partial.model_copy(deep=True))
        yield TextEnd(content_index=index, content=final.content, partial=partial.model_copy(deep=True))
        index += 1
    for call in final.tool_calls:
        yield ToolCallStart(content_index=index, partial=partial.model_copy(deep=True))
        delta = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        yield ToolCallDelta(content_index=index, delta=delta, partial=partial.model_copy(deep=True))
        partial.tool_calls.append(call)
        yield ToolCallEnd(content_index=index, tool_call=call, partial=partial.model_copy(deep=True))
        index += 1
    yield StreamDone(reason=final.stop_reason, message=final)
