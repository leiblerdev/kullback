"""The canonical events one assistant message is streamed as, whoever produced it.

Mirrors tau_ai/events.py: start, then text, thinking and tool-call blocks each opened, filled by
deltas and closed, then done with the assembled message, or error. Every event before done carries
the message as assembled so far, so a consumer can render the partial without keeping its own copy.

The names are the ones the loop and the frontends already read (StreamStart, TextDelta, ...); the
thinking events are new, because a real provider stream reports reasoning as its own channel and
the old one-shot stream had none to report.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.messages import AssistantMessage, StopReason, ToolCall


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


class ThinkingStart(_Event):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDelta(_Event):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEnd(_Event):
    type: Literal["thinking_end"] = "thinking_end"
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
        ThinkingStart,
        ThinkingDelta,
        ThinkingEnd,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        StreamDone,
        StreamError,
    ],
    Field(discriminator="type"),
]
