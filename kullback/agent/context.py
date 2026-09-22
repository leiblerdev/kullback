"""Context accounting: what the active context costs, where the line is, and the record it is kept on.

The estimate is the last assistant message's reported usage when the provider gave one (input,
cached input and output, which together are what the model saw and said on that call) plus a
characters-over-four heuristic for whatever was appended after it; when no usage was reported, or
a compaction has moved the context since the last report, the heuristic runs over the rendered wire
messages, the system prompt and the tool schemas. The estimate says which of the two it used. The
window is a parameter with a default (the agent core may not import `runner.budget`, so a caller
passes `window_for(model)` in), and the line is a fraction of it, 40% by default (D124).

`ContextManager` is what the harness owns: it records every message the loop appends into the
session tree, hands out short entry ids, keeps the entries an application protected through
`protect` (D124's guard: an unacted ruling, an open finding, an unfinished repair), carries the
skills, counts what happened, and runs the compactor at the end of a turn when the estimate is over
the line. The compaction policy itself is `compaction.py`; the model has no tools for any of it.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.compaction import Compactor, entry_kind, entry_text, units
from kullback.agent.messages import AssistantMessage, Message, ToolResultMessage, to_wire
from kullback.agent.session import (
    CompactionEntry,
    MessageEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionStore,
    SkillChangeEntry,
)
from kullback.agent.skills import Skills, first_line, prompt_block
from kullback.ai.usage import Usage

if TYPE_CHECKING:  # pragma: no cover - the harness imports this module; the reverse is type-only
    from kullback.agent.harness import AgentHarness

__all__ = [
    "ContextConfig",
    "ContextEstimate",
    "ContextManager",
    "ContextStats",
    "entry_kind",
    "entry_text",
    "entry_tokens",
    "estimate_context",
    "first_line",
    "prompt_block",
    "units",
]

DEFAULT_WINDOW = 200_000
DEFAULT_LINE = 0.40
KEEP_RECENT_BATCHES = 3

_ENTRY_ID = re.compile(r"^e(\d+)$")


# --- the estimate ---


class ContextEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: int = Field(ge=0)
    window: int = Field(gt=0)
    line: float = Field(gt=0, le=1)
    source: str = Field(pattern="^(usage|heuristic)$")

    @property
    def line_tokens(self) -> int:
        return int(self.window * self.line)

    @property
    def fill(self) -> float:
        return self.tokens / self.window

    @property
    def over_line(self) -> bool:
        return self.tokens > self.line_tokens

    def note(self) -> str:
        """The one line a tool result can carry (D124): the estimate, the window, the line."""
        how = "usage" if self.source == "usage" else "characters"
        return f"context {self.fill:.0%} of {self.window}, line at {self.line:.0%}, estimated from {how}"


def text_tokens(text: str) -> int:
    """The heuristic: four characters to a token, rounded up."""
    return (len(text) + 3) // 4


def message_tokens(message: Message) -> int:
    wire = to_wire([message])
    return text_tokens(json.dumps(wire, ensure_ascii=False)) if wire else 0


def usage_reported(usage: Usage) -> bool:
    return (usage.input + usage.cache_read + usage.cache_write + usage.output) > 0


def estimate_context(
    messages: Sequence[Message],
    system: str = "",
    tool_schemas: Optional[Sequence[dict]] = None,
    window: int = DEFAULT_WINDOW,
    line: float = DEFAULT_LINE,
    use_usage: bool = True,
) -> ContextEstimate:
    """Tokens in the active context, from usage when the provider reported it, else by characters.

    `use_usage=False` forces the heuristic; the manager passes it after a compaction, because the
    last reported usage counted a context that is no longer the one in front of the model.
    """
    last: Optional[int] = None
    if use_usage:
        for index in range(len(messages) - 1, -1, -1):
            candidate = messages[index]
            if isinstance(candidate, AssistantMessage) and usage_reported(candidate.usage):
                last = index
                break
    if last is not None:
        usage = messages[last].usage
        tokens = usage.input + usage.cache_read + usage.cache_write + usage.output
        tokens += sum(message_tokens(m) for m in messages[last + 1 :])
        return ContextEstimate(tokens=tokens, window=window, line=line, source="usage")
    tokens = text_tokens(json.dumps(to_wire(messages, system or None), ensure_ascii=False))
    if tool_schemas:
        tokens += text_tokens(json.dumps(list(tool_schemas), ensure_ascii=False))
    return ContextEstimate(tokens=tokens, window=window, line=line, source="heuristic")


def entry_tokens(entry: SessionEntry) -> int:
    """What one entry costs in context, by the heuristic; entries that are not context cost nothing."""
    if isinstance(entry, MessageEntry):
        return message_tokens(entry.message)
    if isinstance(entry, CompactionEntry):
        return text_tokens(entry.summary)
    return 0


# --- configuration and counters ---


class ContextConfig(BaseModel):
    """How a harness manages context. The window and the line are the estimate; `floor` is whether
    a turn that ends over the line compacts, and `keep_recent_batches` is how many recent tool
    batches a compaction keeps verbatim (D124)."""

    model_config = ConfigDict(extra="forbid")

    window: int = Field(default=DEFAULT_WINDOW, gt=0)
    line: float = Field(default=DEFAULT_LINE, gt=0, le=1)
    # Off by default: the note changes every tool result the model reads, and an application that
    # wants its build byte-identical to one without it must opt in.
    note: bool = False
    floor: bool = True
    keep_recent_batches: int = Field(default=KEEP_RECENT_BATCHES, ge=0)


class ContextStats(BaseModel):
    """What the footer reports: how often code had to compact, how much it replaced, and how full
    the context was at the end of each turn as the model left it."""

    model_config = ConfigDict(extra="forbid")

    compactions: int = 0
    mechanical_summaries: int = 0
    entries_replaced: int = 0
    fill_at_turn_end: list[float] = Field(default_factory=list)
    # `fallback_compactions` and `cuts` are what the round driver's footer reads today. The stream
    # that removes the round driver removes these two with it; `cuts` is always 0, because the
    # floor no longer cuts a guarded result in place, it compacts the prefix like tau.
    fallback_compactions: int = 0
    cuts: int = 0


# --- the manager ---


class ContextManager:
    """The harness's view of its context: the record, the ids, the guards, the skills, the line."""

    def __init__(self, harness: "AgentHarness", session: Optional[SessionStore], config: ContextConfig):
        self.harness = harness
        self.session = session
        self.config = config
        self.stats = ContextStats()
        self.protected: dict[str, str] = {}
        self.skills = Skills(harness, record=self.record_skill_change)
        self.compactor = Compactor(self)
        self._seq = 1
        self._entry_ids_by_call: dict[str, str] = {}
        self._turn_boundary = 0
        self._usage_stale = False
        if session is not None:
            # Ids must stay unique across every branch on the file, so the counter reads them all.
            for entry in session.entries:
                match = _ENTRY_ID.match(entry.id)
                if match:
                    self._seq = max(self._seq, int(match.group(1)) + 1)

    # --- the record ---

    def bootstrap(self, messages: list[Message]) -> list[Message]:
        """The transcript the harness starts with. Without a session it is the messages given;
        with one, the given messages are recorded and the active path is the transcript."""
        if self.session is None:
            return messages
        if not self.session.entries:
            self.session.append(SessionInfoEntry(id=self.next_entry_id()))
        for message in messages:
            self.record(message)
        return self.session.active_messages()

    def next_entry_id(self) -> str:
        """Short ids a record can name; never reused within a session."""
        while True:
            candidate = f"e{self._seq}"
            self._seq += 1
            if self.session is None or self.session.get(candidate) is None:
                return candidate

    def entry_id_for(self, tool_call_id: str) -> Optional[str]:
        """The entry id a tool result will have, known from `tool_execution_start` on, so a hook or
        a handler can protect the result before the model reads it. None when this session knows no
        such call: a lookup may not allocate an id no entry will ever carry."""
        if self.session is None:
            return None
        pending = self._entry_ids_by_call.get(tool_call_id)
        if pending is not None:
            return pending
        # Already recorded: the newest result answering this call id is the one just appended.
        for entry in reversed(self.session.entries):
            message = getattr(entry, "message", None)
            if isinstance(message, ToolResultMessage) and message.tool_call_id == tool_call_id:
                return entry.id
        return None

    def _assign(self, tool_call_id: str) -> str:
        # Always a fresh id: a scripted call id can repeat after a compaction shrank the
        # transcript, and the record must never pair a new result with an old entry's id.
        entry_id = self.next_entry_id()
        self._entry_ids_by_call[tool_call_id] = entry_id
        return entry_id

    def record(self, message: Message) -> Optional[MessageEntry]:
        if self.session is None:
            return None
        entry_id = None
        if isinstance(message, ToolResultMessage):
            entry_id = self._entry_ids_by_call.pop(message.tool_call_id, None)
        entry = MessageEntry(id=entry_id or self.next_entry_id(), message=message)
        self.session.append(entry)
        if isinstance(message, AssistantMessage) and usage_reported(message.usage):
            self._usage_stale = False
        return entry

    def observe(self, event: Any) -> None:
        """Called by the harness with every event before subscribers see it."""
        kind = getattr(event, "type", None)
        if kind == "turn_start":
            self._turn_boundary = len(self.session.entries) if self.session is not None else 0
        elif kind == "tool_execution_start" and self.session is not None:
            self._assign(event.tool_call_id)
        elif kind == "message_end":
            self.record(event.message)

    # --- the estimate ---

    def estimate(self, extra: Optional[Message] = None) -> ContextEstimate:
        messages = list(self.harness.messages)
        if extra is not None:
            messages.append(extra)
        return estimate_context(
            messages,
            self.harness.system,
            self.harness.registry.schemas(),
            window=self.config.window,
            line=self.config.line,
            use_usage=not self._usage_stale,
        )

    def note_for(self, tool_call_id: str, result_content: str = "") -> str:
        """The line appended to a tool result: the estimate with this result counted, and its entry id."""
        extra = ToolResultMessage(tool_call_id=tool_call_id, tool_name="", content=result_content)
        text = self.estimate(extra).note()
        entry_id = self.entry_id_for(tool_call_id)
        if entry_id is not None:
            text += f"; this result is entry {entry_id}"
        return text

    def refresh(self) -> None:
        """Put the session's active path in front of the model and re-read the system prompt."""
        self.harness.sync_context()

    def after_compaction(self) -> None:
        """What a compaction changes here: the last usage counted a context that is gone."""
        self._usage_stale = True
        self.refresh()

    def entry_tokens(self, entry: SessionEntry) -> int:
        return entry_tokens(entry)

    # --- the guards ---

    def protect(self, entry_ids: Sequence[str], reason: str) -> None:
        for entry_id in entry_ids:
            self.protected[entry_id] = reason

    def unprotect(self, entry_ids: Sequence[str]) -> None:
        for entry_id in entry_ids:
            self.protected.pop(entry_id, None)

    def current_turn_ids(self) -> set[str]:
        if self.session is None:
            return set()
        return {e.id for e in self.session.entries[self._turn_boundary :]}

    def guards(self, path: Optional[Sequence[SessionEntry]] = None) -> dict[str, tuple[str, str]]:
        """Every guarded id on the active path, with the rule and the reason it is guarded."""
        if self.session is None:
            return {}
        path = self.session.active_path() if path is None else path
        guarded: dict[str, tuple[str, str]] = {}
        for entry in path:
            if isinstance(entry, SessionInfoEntry):
                guarded[entry.id] = ("session_info", f"entry {entry.id} is the session root and is never replaced")
        for entry_id, reason in self.protected.items():
            guarded.setdefault(entry_id, ("protected", f"entry {entry_id} is protected: {reason}"))
        for entry_id in self.current_turn_ids():
            guarded.setdefault(entry_id, ("current_turn", f"entry {entry_id} belongs to the current turn"))
        return guarded

    # --- the skills ---

    @property
    def catalog_skills(self) -> dict[str, str]:
        return self.skills.catalog

    @property
    def loaded_skills(self) -> set[str]:
        return self.skills.loaded

    def catalog_skill(self, name: str, text: str, loaded: bool = False) -> Optional[str]:
        """Make a skill available; with `loaded`, put its text in the prompt now and answer the
        hash of the text that went in, which is what a record of the context names (D125)."""
        return self.skills.catalog_skill(name, text, loaded=loaded)

    def record_skill_change(self, name: str, action: str, digest: str) -> None:
        """A skill entered or left the prompt: the session says which text was in context (D125)."""
        if self.session is None:
            return
        self.session.append(
            SkillChangeEntry(id=self.next_entry_id(), name=name, action=action, content_hash=digest)
        )

    def skills_section(self) -> str:
        """The `<skills>` block an extension places by adding a section named `skills`."""
        return self.skills.section()

    def refresh_skills_section(self) -> None:
        self.skills.refresh_section()

    # --- the line ---

    async def after_turn(self, turn: int) -> Optional[CompactionEntry]:
        """Record the fill as the model left it, then compact if it is over the line (D124)."""
        estimate = self.estimate()
        self.stats.fill_at_turn_end.append(round(estimate.fill, 4))
        if not self.config.floor or self.session is None or not estimate.over_line:
            return None
        return await self.compactor.compact(f"the context was over the line at the end of turn {turn}")

    async def compact(self, reason: str = "the run asked for it") -> Optional[CompactionEntry]:
        """Compact now, whatever the estimate says; what `harness.compact()` calls."""
        return await self.compactor.compact(reason)
