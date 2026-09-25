"""The typed events one run of the loop emits. Every state change is one of these.

The transcript is the source of truth and the events are its changelog: a frontend, a session store,
the bus or a test reads the stream and never the loop's internals. The agent, turn, message and tool
execution events are tau_agent/events.py's, under tau's names. What tau does not have: the
compaction event, the custom message an extension queues, the error event, the generic `CustomEvent`
an application publishes on the bus, and the stage, round and beat events of the scheduler the
overhaul removes (the stream that removes the round driver removes those four with it).

Two field names differ from tau and are kept because the packages above read them: the tool
execution events carry `arguments` (tau's `args`, also readable under that name) and the message
update carries `stream_event` (tau's `assistant_message_event`, also readable under that name).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from kullback.agent.messages import AssistantMessage, Message, ToolResultMessage
from kullback.agent.tools import ToolResult
from kullback.ai.stream import StreamEvent


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentStartEvent(_Event):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(_Event):
    """The messages this run appended, in order; the transcript holds them too."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = Field(default_factory=list)


class TurnStartEvent(_Event):
    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(_Event):
    type: Literal["turn_end"] = "turn_end"
    turn: int
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(_Event):
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(_Event):
    """One stream event of the assistant message being assembled, with the partial so far."""

    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: StreamEvent

    @property
    def assistant_message_event(self) -> StreamEvent:
        """tau's name for `stream_event`."""
        return self.stream_event


class MessageEndEvent(_Event):
    """A message is complete. An assistant message the model streamed carries `request`: the cache
    fingerprint of what was sent for it (kullback.ai.cache.fingerprint), which budget.subscriber puts
    on the feed. Any other message carries none, and the key is then left out of the stored form."""

    type: Literal["message_end"] = "message_end"
    message: Message
    request: Optional[dict[str, Any]] = None

    @model_serializer(mode="wrap")
    def _omit_absent_request(self, handler):
        data = handler(self)
        if self.request is None:
            data.pop("request", None)
        return data


class ToolExecutionStartEvent(_Event):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def args(self) -> dict[str, Any]:
        """tau's name for `arguments`."""
        return self.arguments


class ToolExecutionUpdateEvent(_Event):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    partial: ToolResult


class ToolExecutionEndEvent(_Event):
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
    """The counts are all from gates and none from a model (D126); `exit` is set on the last round."""

    type: Literal["round_end"] = "round_end"
    round: int
    counts: dict[str, Any] = Field(default_factory=dict)
    # `target_built` is a run asked for a stage earlier than the derivation's inputs: the target was
    # built and there was no Verifier to examine, so the run ends rather than rebuilding it.
    exit: Optional[Literal["done", "stalled", "ceiling", "max_rounds", "target_built", "refused"]] = None


class BeatStart(_Event):
    """One agent takes the stream: the Builder or the Examiner, one at a time (D128)."""

    type: Literal["beat_start"] = "beat_start"
    agent: Literal["builder", "examiner"]
    round: int


class BeatEnd(_Event):
    """The agent hands the stream back; `spend` is what its beat cost, read off budget.json."""

    type: Literal["beat_end"] = "beat_end"
    agent: Literal["builder", "examiner"]
    round: int
    spend: float = 0.0


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str


class Compaction(_Event):
    """A compaction entry landed on the session: what it replaced and what stood in.

    `by` is `model` when the model wrote the summary and `code` when the mechanical summary stood
    in for it; `reason` says what asked for the compaction (the line, or a caller), `entry_id`
    names the entry and `first_kept_entry_id` the oldest entry kept verbatim.
    """

    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)
    first_kept_entry_id: Optional[str] = None
    by: Literal["model", "code"] = "model"
    entry_id: Optional[str] = None
    reason: Optional[str] = None
    note: Optional[str] = None


class CustomMessage(_Event):
    """An extension queued a message: `content` enters the context, `details` does not."""

    type: Literal["custom_message"] = "custom_message"
    content: str
    details: Optional[dict[str, Any]] = None
    deliver_as: Literal["steer", "follow_up"] = "steer"


class CustomEvent(_Event):
    """Anything an application publishes on the bus that the core has no type for."""

    type: Literal["custom"] = "custom"
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SteerRequestEvent(_Event):
    """Someone asks a live run to change course from outside its process (agent/steer.py).

    `nudge` is delivered before the next model turn, `tell` when the run would stop, `stop` cancels
    at the next step. The request sits on the bus beside the events it answers, so every watcher
    sees who asked for what, in order; `sender` names the screen or command that asked.
    """

    type: Literal["steer_request"] = "steer_request"
    id: str
    kind: Literal["nudge", "tell", "stop"]
    text: str = ""
    sender: str = ""


class SteerAckEvent(_Event):
    """The run's answer to one SteerRequestEvent: queued on the harness, cancel asked, or refused
    with the reason."""

    type: Literal["steer_ack"] = "steer_ack"
    id: str
    kind: Literal["nudge", "tell", "stop"]
    outcome: Literal["queued", "cancel_asked", "refused"]
    reason: str = ""


AgentEvent = Annotated[
    Union[
        AgentStartEvent,
        AgentEndEvent,
        TurnStartEvent,
        TurnEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
        StageStart,
        StageEnd,
        RoundStart,
        RoundEnd,
        BeatStart,
        BeatEnd,
        ErrorEvent,
        Compaction,
        CustomMessage,
        CustomEvent,
        SteerRequestEvent,
        SteerAckEvent,
    ],
    Field(discriminator="type"),
]

# The names this package used before it took tau's. The packages above still import them; the
# stream that rewrites each of those packages onto the tau names drops its line here.
AgentStart = AgentStartEvent
AgentEnd = AgentEndEvent
TurnStart = TurnStartEvent
TurnEnd = TurnEndEvent
MessageStart = MessageStartEvent
MessageUpdate = MessageUpdateEvent
MessageEnd = MessageEndEvent
ToolExecutionStart = ToolExecutionStartEvent
ToolExecutionUpdate = ToolExecutionUpdateEvent
ToolExecutionEnd = ToolExecutionEndEvent
Custom = CustomEvent
