"""What one adapter reports while a provider answers, before anything is assembled.

Mirrors tau_ai/_provider_events.py. An adapter parses its own wire shape into these and nothing
else; `stream.canonicalize_provider_stream` turns them into the canonical events the loop reads.
Keeping the two apart is what lets a new provider be one parser rather than one more event shape.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from kullback.ai.messages import AssistantMessage, ToolCall


class _ProviderEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderResponseStart(_ProviderEvent):
    """The provider has started answering."""

    type: Literal["response_start"] = "response_start"
    model: Optional[str] = None
    request_id: Optional[str] = None


class ProviderRetry(_ProviderEvent):
    """The adapter is trying a transient failure again. Never reaches the loop."""

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: Optional[dict[str, Any]] = None


class ProviderTextDelta(_ProviderEvent):
    """A fragment of the answer, as it arrived."""

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderThinkingDelta(_ProviderEvent):
    """A fragment of the reasoning channel, as it arrived."""

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ProviderToolCall(_ProviderEvent):
    """One whole tool call: the arguments are parsed only once they are complete."""

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ProviderResponseEnd(_ProviderEvent):
    """The provider finished. The message here is authoritative for usage and the stop reason
    only: the content order is what the deltas already said."""

    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: Optional[str] = None


class ProviderErrorEvent(_ProviderEvent):
    """A provider-level failure the loop has to be told about rather than raised at."""

    type: Literal["error"] = "error"
    message: str
    data: Optional[dict[str, Any]] = None


ProviderEvent = Union[
    ProviderResponseStart,
    ProviderRetry,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderToolCall,
    ProviderResponseEnd,
    ProviderErrorEvent,
]
