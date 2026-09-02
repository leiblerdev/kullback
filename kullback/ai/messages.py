"""The typed messages of a transcript: what the model said, what it was told, what a tool returned.

They live in `ai` rather than in `agent` because the stream (`stream.py`) assembles an
AssistantMessage from a ModelReply and takes the transcript it is asked to continue, and `ai` may
not import `agent` (D121). `agent/messages.py` re-exports them, so the agent core reads as the owner
and nothing above `ai` has to spell the path. The alternative, a stream generic over whatever
message type the caller passes, would have put a type parameter on every event for one concrete
type; one module and a re-export is the simpler shape.

`details` on a user or tool message is the part that never enters the model context: the wire shape
in `to_wire` drops it. It is where a structured record (a gate ruling, a Run) rides next to the text
the model reads (D123).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.usage import Usage

StopReason = Literal["stop", "tool_use", "length", "error"]


class ToolCall(BaseModel):
    """One tool call the model asked for, with the arguments already parsed into a dict."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class UserMessage(BaseModel):
    """Text the model is told: a prompt, a steer, a follow-up, or an extension's custom message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str
    details: Optional[dict[str, Any]] = None


class AssistantMessage(BaseModel):
    """What the model answered on one turn: text, tool calls, usage, and why it stopped."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    model: Optional[str] = None
    error_message: Optional[str] = None


class ToolResultMessage(BaseModel):
    """What a tool call returned. `content` is what the model reads; `details` never reaches it."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    details: Optional[dict[str, Any]] = None


Message = Annotated[Union[UserMessage, AssistantMessage, ToolResultMessage], Field(discriminator="role")]


def to_wire(messages: Sequence[Message], system: Optional[str] = None) -> list[dict]:
    """The dict shape Model.query takes, from the typed transcript.

    The system prompt goes first as a system message, the way the adapters already read one.
    An assistant message that ended in error and carries neither text nor tool calls is left out:
    it is kept in the transcript as the record of what happened, but a provider cannot be sent an
    empty assistant turn, and it would poison the next request. `details` is dropped everywhere.
    """
    wire: list[dict] = []
    if system:
        wire.append({"role": "system", "content": system})
    for message in messages:
        if isinstance(message, UserMessage):
            wire.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            if message.stop_reason == "error" and not message.content and not message.tool_calls:
                continue
            wire.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                        for call in message.tool_calls
                    ],
                }
            )
        elif isinstance(message, ToolResultMessage):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.tool_name,
                    "content": message.content,
                }
            )
        else:  # pragma: no cover - the union above is closed
            raise TypeError(f"not a transcript message: {type(message).__name__}")
    return wire
