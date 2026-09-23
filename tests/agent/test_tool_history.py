"""The repair that makes a transcript one a provider accepts: results beside their calls, every
call answered, a leftover result dropped."""

from __future__ import annotations

from kullback.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from kullback.agent.tool_history import (
    INTERRUPTED_TOOL_RESULT,
    interrupted_tool_results,
    repair_tool_history,
)


def assistant(*call_ids: str) -> AssistantMessage:
    return AssistantMessage(tool_calls=[{"id": i, "name": "t", "arguments": {}} for i in call_ids])


def result(call_id: str, content: str = "ok") -> ToolResultMessage:
    return ToolResultMessage(tool_call_id=call_id, tool_name="t", content=content)


def test_a_transcript_already_in_order_is_returned_unchanged():
    messages = [UserMessage(content="go"), assistant("c1"), result("c1"), AssistantMessage(content="done")]
    repair = repair_tool_history(messages)
    assert repair.changed is False and list(repair.messages) == messages
    assert repair.counts() == {
        "synthesized_results": 0,
        "dropped_orphan_results": 0,
        "dropped_duplicate_results": 0,
        "reordered_results": 0,
    }


def test_an_unanswered_call_is_answered_with_an_interruption_in_place():
    repair = repair_tool_history([assistant("c1", "c2"), result("c2")])
    kinds = [(type(m).__name__, getattr(m, "tool_call_id", None)) for m in repair.messages]
    assert kinds == [("AssistantMessage", None), ("ToolResultMessage", "c1"), ("ToolResultMessage", "c2")]
    assert repair.messages[1].content == INTERRUPTED_TOOL_RESULT and repair.messages[1].is_error
    assert repair.changed is True and repair.counts()["synthesized_results"] == 1
    # the dangling call finder answers each unanswered call once
    messages = [UserMessage(content="go"), assistant("c1", "c2"), result("c1")]
    repairs = interrupted_tool_results(messages)
    assert [r.tool_call_id for r in repairs] == ["c2"]
    assert repairs[0].is_error and "interrupted" in repairs[0].content
    assert interrupted_tool_results(messages + repairs) == []


def test_a_result_that_drifted_from_its_call_is_moved_back_beside_it():
    repair = repair_tool_history([assistant("c1"), UserMessage(content="meanwhile"), result("c1")])
    roles = [m.role for m in repair.messages]
    assert roles == ["assistant", "tool", "user"]
    assert repair.counts()["reordered_results"] == 1


def test_a_result_whose_call_is_gone_is_dropped_because_its_call_cannot_be_rebuilt():
    repair = repair_tool_history([UserMessage(content="go"), result("gone")])
    assert [m.role for m in repair.messages] == ["user"]
    assert repair.counts()["dropped_orphan_results"] == 1


def test_a_real_result_is_preferred_over_a_second_one_for_the_same_call():
    repair = repair_tool_history([assistant("c1"), result("c1", "first"), result("c1", "second")])
    assert [m.role for m in repair.messages] == ["assistant", "tool"]
    assert repair.messages[1].content == "first"
    assert repair.counts()["dropped_duplicate_results"] == 1
