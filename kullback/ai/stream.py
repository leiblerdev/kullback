"""One assistant message as canonical events, from a provider stream or from a whole reply.

Mirrors tau_ai/stream.py. `canonicalize_provider_stream` is the one place a provider's own report
(`_provider_events`) becomes the events the loop reads: it opens a block when a channel first
speaks, grows the partial message on every delta, closes the block when the channel changes or the
answer ends, and finishes with done or error. A consumer sees the same events whether the text
arrived token by token from a live endpoint or whole from a replay.

`stream` below is the older path over `Model.query`: a whole reply, taken in a worker thread so an
async loop is not held by the network wait, then emitted as one delta per block. It is what keeps
the three offline models and every caller that holds a `Model` working while the callers move to
the provider protocol; `replay.ReplayProvider` is the same adaptation behind the protocol.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional, Sequence

from kullback.ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEnd,
    ProviderResponseStart,
    ProviderRetry,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderToolCall,
)
from kullback.ai.events import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from kullback.ai.messages import AssistantMessage, Message, StopReason, ToolCall, to_wire

__all__ = [
    "StreamDone",
    "StreamError",
    "StreamEvent",
    "StreamStart",
    "TextDelta",
    "TextEnd",
    "TextStart",
    "ThinkingDelta",
    "ThinkingEnd",
    "ThinkingStart",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "assemble",
    "canonicalize_provider_stream",
    "error_message",
    "normalize_stop_reason",
    "stream",
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


def assemble(reply: Any, call_id_prefix: str = "call") -> AssistantMessage:
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
        thinking_blocks=getattr(reply, "thinking_blocks", None),
    )


def error_message(text: str, model: Optional[str] = None) -> AssistantMessage:
    """The assistant message that records a failed call, so the transcript says what happened."""
    return AssistantMessage(content=None, stop_reason="error", error_message=text, model=model)


async def canonicalize_provider_stream(
    source: AsyncIterator[ProviderEvent],
    *,
    model: Optional[str] = None,
    independent_channels: bool = False,
) -> AsyncIterator[StreamEvent]:
    """One adapter's report as the canonical events, emitted as the report arrives.

    A block is opened the first time its channel speaks and closed when another channel speaks or
    the answer ends. Chat completions interleave text and reasoning freely, so it passes
    `independent_channels`: a switch back and forth reopens neither block and the two keep their
    own indices. Transports that keep their blocks in order (the Responses API, Anthropic) leave it
    off, and a channel switch closes the block it left.

    A provider error never escapes as an exception: it becomes an error event carrying an assistant
    message with stop_reason "error", so the loop can record what happened in the transcript.
    """
    partial = AssistantMessage(model=model, stop_reason="stop")
    active_index: Optional[int] = None
    active_kind: Optional[str] = None
    channel_indices: dict[str, int] = {}
    next_index = 0
    started = False
    terminal = False

    def snapshot() -> AssistantMessage:
        return partial.model_copy(deep=True)

    def end_block(index: Optional[int]) -> list[StreamEvent]:
        if index is None:
            return []
        kind = next((name for name, value in channel_indices.items() if value == index), None)
        if kind == "text":
            return [TextEnd(content_index=index, content=partial.content or "", partial=snapshot())]
        if kind == "thinking":
            return [ThinkingEnd(content_index=index, content=partial.thinking or "", partial=snapshot())]
        return []

    async for event in source:
        if isinstance(event, ProviderRetry):
            # A retry is the adapter's business; the loop is told what finally happened.
            continue
        if isinstance(event, ProviderResponseStart):
            if event.model:
                partial.model = event.model
            if not started:
                started = True
                yield StreamStart(partial=snapshot())
            continue
        if not started:
            started = True
            yield StreamStart(partial=snapshot())

        if isinstance(event, (ProviderTextDelta, ProviderThinkingDelta)):
            thinking = isinstance(event, ProviderThinkingDelta)
            kind = "thinking" if thinking else "text"
            if independent_channels:
                active_index = channel_indices.get(kind)
                active_kind = kind if active_index is not None else None
            if active_kind != kind:
                if not independent_channels:
                    for ended in end_block(active_index):
                        yield ended
                    channel_indices.pop(active_kind, None)
                active_index = next_index
                next_index += 1
                active_kind = kind
                channel_indices[kind] = active_index
                # A reopened channel keeps what it already said: the message carries one text
                # and one thinking string, so a second block on the same channel appends to it.
                if thinking:
                    partial.thinking = partial.thinking or ""
                    yield ThinkingStart(content_index=active_index, partial=snapshot())
                else:
                    partial.content = partial.content or ""
                    yield TextStart(content_index=active_index, partial=snapshot())
            if thinking:
                partial.thinking = (partial.thinking or "") + event.delta
                yield ThinkingDelta(content_index=active_index, delta=event.delta, partial=snapshot())
            else:
                partial.content = (partial.content or "") + event.delta
                yield TextDelta(content_index=active_index, delta=event.delta, partial=snapshot())
        elif isinstance(event, ProviderToolCall):
            for index in (sorted(channel_indices.values()) if independent_channels else [active_index]):
                for ended in end_block(index):
                    yield ended
            channel_indices.clear()
            active_index = None
            active_kind = None
            index = next_index
            next_index += 1
            yield ToolCallStart(content_index=index, partial=snapshot())
            arguments = json.dumps(event.tool_call.arguments, sort_keys=True, ensure_ascii=False)
            yield ToolCallDelta(content_index=index, delta=arguments, partial=snapshot())
            partial.tool_calls.append(event.tool_call.model_copy(deep=True))
            yield ToolCallEnd(content_index=index, tool_call=event.tool_call, partial=snapshot())
        elif isinstance(event, ProviderResponseEnd):
            for index in (sorted(channel_indices.values()) if independent_channels else [active_index]):
                for ended in end_block(index):
                    yield ended
            channel_indices.clear()
            active_index = None
            active_kind = None
            # What the deltas said is the content; the adapter's final message is authoritative
            # for usage, the model name, the stop reason and the signed thinking blocks (which no
            # delta carries whole), and for nothing else.
            final = snapshot()
            final.usage = event.message.usage
            final.model = event.message.model or partial.model
            final.thinking_blocks = event.message.thinking_blocks
            final.error_message = event.message.error_message
            # An adapter that reported no delta but ended with content keeps it, and an answer of
            # empty text stays empty text rather than becoming no text: a reply that said nothing
            # and a reply that said "" are not the same record.
            if final.content is None and not final.tool_calls and event.message.content is not None:
                final.content = event.message.content
            final.stop_reason = normalize_stop_reason(event.finish_reason, bool(final.tool_calls))
            yield StreamDone(reason=final.stop_reason, message=final)
            terminal = True
        elif isinstance(event, ProviderErrorEvent):
            error = snapshot()
            error.stop_reason = "error"
            error.error_message = event.message
            yield StreamError(reason="error", error=error)
            terminal = True

    if not started:
        yield StreamStart(partial=snapshot())
    if not terminal:
        error = snapshot()
        error.stop_reason = "error"
        error.error_message = "the provider stream ended without a done or an error"
        yield StreamError(reason="error", error=error)


async def reply_events(reply: Any, *, call_id_prefix: str = "call") -> AsyncIterator[ProviderEvent]:
    """One whole reply as the adapter report a live stream would have made of it.

    This is the replay seam: a memoised, recorded or scripted reply is reported as a response
    start, one delta per block and a response end, so the canonicaliser above produces the same
    events it produces for a live endpoint.
    """
    final = assemble(reply, call_id_prefix=call_id_prefix)
    yield ProviderResponseStart(model=final.model)
    if final.thinking:
        yield ProviderThinkingDelta(delta=final.thinking)
    if final.content:
        yield ProviderTextDelta(delta=final.content)
    for call in final.tool_calls:
        yield ProviderToolCall(tool_call=call)
    yield ProviderResponseEnd(message=final, finish_reason=final.stop_reason)


async def stream(
    model: Any,
    messages: Sequence[Message],
    tools: Optional[Sequence[dict]] = None,
    system: Optional[str] = None,
    config: Optional[Any] = None,
) -> AsyncIterator[StreamEvent]:
    """Stream one assistant message for the transcript over the older `Model.query` handle.

    The call runs in a worker thread so an async loop is not held by the network wait. A provider
    error becomes an error event; a cancellation of the awaiting task does escape, because that is
    the caller stopping the run.
    """
    wire = to_wire(messages, system)
    tool_list = list(tools) if tools else None
    prefix = f"call_{len(messages)}"
    try:
        reply = await asyncio.to_thread(model.query, wire, tool_list, config)
    # CancelledError is a BaseException on the Python this package requires, so the clause below
    # lets a cancellation through on its own; the caller stopping the run is not a provider error.
    except Exception as exc:  # noqa: BLE001 - the provider is an isolation boundary
        yield StreamError(
            reason="error",
            error=error_message(f"{type(exc).__name__}: {exc}", model=getattr(model, "name", None)),
        )
        return
    async for event in canonicalize_provider_stream(
        reply_events(reply, call_id_prefix=prefix),
        model=reply.model,
    ):
        yield event
