"""Provider-safe repair of a transcript whose tool calls and results do not line up.

tau_agent/tool_history.py. No provider accepts a tool call with no result, a result with no call,
or a result that does not sit next to its call, and a run that was cancelled between a call and its
result leaves exactly that. The repair is deterministic and reads the transcript without changing
it: results are moved beside their calls, a missing result becomes one marked interrupted, and a
result whose call is gone is left out, because a call's arguments cannot be reconstructed from it.
When two results answer one call id, a real result is preferred over the synthetic interruption.

The session file keeps every message as it was recorded; this is what the provider is sent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from kullback.agent.messages import AssistantMessage, Message, ToolCall, ToolResultMessage

__all__ = ["INTERRUPTED_TOOL_RESULT", "ToolHistoryRepair", "interrupted_tool_results", "repair_tool_history"]

INTERRUPTED_TOOL_RESULT = "tool call interrupted before it returned"


@dataclass(frozen=True)
class ToolHistoryRepair:
    """A transcript a provider accepts, and the count of each repair that got it there."""

    messages: tuple[Message, ...]
    changed: bool = False
    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_duplicate_results: int = 0
    reordered_results: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "synthesized_results": self.synthesized_results,
            "dropped_orphan_results": self.dropped_orphan_results,
            "dropped_duplicate_results": self.dropped_duplicate_results,
            "reordered_results": self.reordered_results,
        }


def _interruption(call: ToolCall) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call.id, tool_name=call.name, content=INTERRUPTED_TOOL_RESULT, is_error=True
    )


def _is_interruption(message: ToolResultMessage) -> bool:
    return message.is_error and message.content == INTERRUPTED_TOOL_RESULT


def repair_tool_history(messages: Sequence[Message]) -> ToolHistoryRepair:
    """The transcript with every tool call answered by exactly one result, next to it."""
    messages = tuple(messages)
    calls: list[tuple[tuple[int, int], ToolCall, int]] = []
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for offset, call in enumerate(message.tool_calls, start=1):
                calls.append(((index, offset), call, index + offset))

    results_by_id: dict[str, list[tuple[int, ToolResultMessage]]] = defaultdict(list)
    for index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            results_by_id[message.tool_call_id].append((index, message))

    chosen: dict[tuple[int, int], tuple[Optional[int], ToolResultMessage]] = {}
    used: set[int] = set()
    # Pairs already in place are reserved first, so a call id that repeats keeps the result of its
    # own turn rather than having an earlier occurrence take it.
    for occurrence, call, expected in calls:
        if expected >= len(messages):
            continue
        candidate = messages[expected]
        if isinstance(candidate, ToolResultMessage) and candidate.tool_call_id == call.id and expected not in used:
            chosen[occurrence] = (expected, candidate)
            used.add(expected)

    synthesized = 0
    for occurrence, call, _expected in calls:
        if occurrence in chosen:
            continue
        free = [row for row in results_by_id.get(call.id, []) if row[0] not in used]
        if free:
            after = [row for row in free if row[0] > occurrence[0]]
            pool = after or free
            pick = next((row for row in pool if not _is_interruption(row[1])), pool[0])
            chosen[occurrence] = pick
            used.add(pick[0])
            continue
        chosen[occurrence] = (None, _interruption(call))
        synthesized += 1

    # A real result left over is better than a synthetic interruption already picked for that id.
    for occurrence, call, _expected in calls:
        position, result = chosen[occurrence]
        if position is None or not _is_interruption(result):
            continue
        replacement = next(
            (row for row in results_by_id.get(call.id, []) if row[0] not in used and not _is_interruption(row[1])),
            None,
        )
        if replacement is None:
            continue
        used.discard(position)
        used.add(replacement[0])
        chosen[occurrence] = replacement

    repaired: list[Message] = []
    reordered = 0
    for index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            continue
        repaired.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for offset, _call in enumerate(message.tool_calls, start=1):
            position, result = chosen[(index, offset)]
            repaired.append(result)
            if position is not None and position != index + offset:
                reordered += 1

    call_ids = {call.id for _occurrence, call, _expected in calls}
    left_over = [result for rows in results_by_id.values() for position, result in rows if position not in used]
    repaired_messages = tuple(repaired)
    return ToolHistoryRepair(
        messages=repaired_messages,
        changed=repaired_messages != messages,
        synthesized_results=synthesized,
        dropped_orphan_results=sum(r.tool_call_id not in call_ids for r in left_over),
        dropped_duplicate_results=sum(r.tool_call_id in call_ids for r in left_over),
        reordered_results=reordered,
    )


def interrupted_tool_results(messages: Sequence[Message]) -> list[ToolResultMessage]:
    """The results a transcript is missing, in call order, each marked interrupted.

    The harness appends these before the next run, so the transcript stays valid without any
    history being rewritten: what was recorded stays, and what was never answered is answered now.
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
            repairs.append(_interruption(call))
    return repairs
