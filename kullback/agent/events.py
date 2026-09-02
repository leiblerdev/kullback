"""The typed events one run of the loop emits. Every state change is one of these.

The transcript is the source of truth and the events are its changelog: a frontend, a session store
or a test reads the stream and never the loop's internals. The agent, turn, message and tool
execution events are tau's; stage, round, compaction and custom_message are ours, carried now so
later phases (the DAG scheduler in 4, rounds in 5, context tools in 7) add emitters, not types.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.messages import AssistantMessage, Message, ToolResultMessage
from kullback.agent.tools import ToolResult
from kullback.ai.stream import StreamEvent


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentStart(_Event):
    type: Literal["agent_start"] = "agent_start"


class AgentEnd(_Event):
    """The messages this run appended, in order; the transcript holds them too."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = Field(default_factory=list)


class TurnStart(_Event):
    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEnd(_Event):
    type: Literal["turn_end"] = "turn_end"
    turn: int
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStart(_Event):
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdate(_Event):
    """One stream event of the assistant message being assembled, with the partial so far."""

    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: StreamEvent


class MessageEnd(_Event):
    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStart(_Event):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionUpdate(_Event):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    partial: ToolResult


class ToolExecutionEnd(_Event):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool


class StageStart(_Event):
    type: Literal["stage_start"] = "stage_start"
    name: str


class StageEnd(_Event):
    type: Literal["stage_end"] = "stage_end"
    name: str
    counts: dict[str, Any] = Field(default_factory=dict)


class RoundStart(_Event):
    type: Literal["round_start"] = "round_start"
    round: int


class RoundEnd(_Event):
    """The counts are all from gates and none from a model (D126); later phases fill them."""

    type: Literal["round_end"] = "round_end"
    round: int
    counts: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str


class Compaction(_Event):
    """A compaction entry landed on the session: what it replaced and what stood in.

    `by` is `model` for a forget, `code_fallback` for the 40% floor (D124, D131), `code` for any
    other code-driven compaction; `entry_id` names the entry, `note` carries the floor's account
    of why it fired and whether the summary was the model's or mechanical.
    """

    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)
    first_kept_entry_id: Optional[str] = None
    by: Literal["model", "code", "code_fallback"] = "model"
    entry_id: Optional[str] = None
    note: Optional[str] = None


class CustomMessage(_Event):
    """An extension queued a message: `content` enters the context, `details` does not."""

    type: Literal["custom_message"] = "custom_message"
    content: str
    details: Optional[dict[str, Any]] = None
    deliver_as: Literal["steer", "follow_up"] = "steer"


AgentEvent = Annotated[
    Union[
        AgentStart,
        AgentEnd,
        TurnStart,
        TurnEnd,
        MessageStart,
        MessageUpdate,
        MessageEnd,
        ToolExecutionStart,
        ToolExecutionUpdate,
        ToolExecutionEnd,
        StageStart,
        StageEnd,
        RoundStart,
        RoundEnd,
        ErrorEvent,
        Compaction,
        CustomMessage,
    ],
    Field(discriminator="type"),
]
