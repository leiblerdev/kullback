"""The entry types of the session tree, one pydantic model each, discriminated by `type`."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.messages import Message


def new_entry_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_entry_id)
    parent_id: Optional[str] = None
    timestamp: float = Field(default_factory=now)


class SessionInfoEntry(_Entry):
    """The root: when the session began and what it is for. Never compacted away (D124)."""

    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=now)
    cwd: Optional[str] = None
    title: Optional[str] = None
    agent: Optional[str] = None
    # Which context-management arm produced this session (tools, code_only, files; D124, D131),
    # so a build's artifacts say under which arm they were made.
    arm: Optional[str] = None


class MessageEntry(_Entry):
    type: Literal["message"] = "message"
    message: Message


class ToolSetChangeEntry(_Entry):
    """The active tool set changed: names loaded and unloaded. Tool schemas are context too."""

    type: Literal["tool_set_change"] = "tool_set_change"
    loaded: list[str] = Field(default_factory=list)
    unloaded: list[str] = Field(default_factory=list)


class SkillChangeEntry(_Entry):
    """A skill was loaded, unloaded or edited; the hash says which text was in context (D125)."""

    type: Literal["skill_change"] = "skill_change"
    name: str
    action: Literal["load", "unload", "edit"]
    content_hash: Optional[str] = None


class CompactionEntry(_Entry):
    """A summary that stands in for the entries it replaces on replay.

    `by` says who chose what to drop: `model` is a forget (the model chose the ids and wrote the
    summary, D124), `code_fallback` is the 40% floor (code chose the oldest entries and asked the
    model for the summary, or wrote a mechanical one when it could not; D124, D131), and `code`
    is any other code-driven compaction an application appends.
    """

    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)
    first_kept_entry_id: Optional[str] = None
    by: Literal["model", "code", "code_fallback"] = "model"
    note: Optional[str] = None


class CustomEntry(_Entry):
    """Application-owned data under a namespace; the core carries it and never reads it."""

    type: Literal["custom"] = "custom"
    namespace: str
    data: dict[str, Any] = Field(default_factory=dict)


class LeafEntry(_Entry):
    """The leaf pointer: the active branch ends at `entry_id`. Written by `branch`."""

    type: Literal["leaf"] = "leaf"
    entry_id: str


SessionEntry = Annotated[
    Union[
        SessionInfoEntry,
        MessageEntry,
        ToolSetChangeEntry,
        SkillChangeEntry,
        CompactionEntry,
        CustomEntry,
        LeafEntry,
    ],
    Field(discriminator="type"),
]
