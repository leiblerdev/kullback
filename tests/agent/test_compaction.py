"""Compaction: the older prefix becomes one summary at the line and on request, the recent batches
stand verbatim, the guards hold, and the mechanical summary stands in when the model cannot."""

from __future__ import annotations

import asyncio

from kullback.agent.compaction import SUMMARY_PROMPT, mechanical_summary, units
from kullback.agent.context import ContextConfig
from kullback.agent.extensions import ExtensionAPI
from kullback.agent.harness import AgentHarness
from kullback.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from kullback.agent.session import CompactionEntry, MessageEntry, SessionStore
from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from tests.agent.conftest import call, collect, reply, types_of


def harness_with_session(tmp_path, model, tools=(), **config):
    store = SessionStore(tmp_path / "session.jsonl")
    return AgentHarness(model, system="sys", tools=list(tools), session=store, context=ContextConfig(**config))


def two_turn_harness(tmp_path, replies, **config):
    """A harness over two turns of plain text, with the line low enough that turn 2 crosses it."""
    defaults = dict(window=1000, line=0.4, keep_recent_batches=2)
    defaults.update(config)
    harness = harness_with_session(tmp_path, TestModel(replies), **defaults)
    harness.follow_up("next")
    return harness


# --- the line ---


def test_compaction_at_the_line_replaces_the_older_prefix_with_the_models_summary(tmp_path):
    harness = two_turn_harness(tmp_path, ["one", "two", "the user sent a long x; the answer was one"])
    events = collect(harness.prompt("x" * 3000))
    kinds = types_of(events)
    assert kinds.count("compaction") == 1
    turn_ends = [i for i, k in enumerate(kinds) if k == "turn_end"]
    compaction_at = kinds.index("compaction")
    # turn 1 ends over the line with nothing outside the current turn to replace, so the first
    # compaction is at the end of turn 2
    assert turn_ends[0] < compaction_at and turn_ends[1] < compaction_at < kinds.index("agent_end")
    event = events[compaction_at]
    assert event.by == "model" and event.replaces_entry_ids == ["e2", "e3"]
    assert event.first_kept_entry_id == "e4" and event.entry_id == "e6"
    assert event.summary == "the user sent a long x; the answer was one"
    assert "the context was over the line at the end of turn 2" in event.reason
    entry = harness.session.get("e6")
    assert isinstance(entry, CompactionEntry) and entry.by == "model" and "summary by the model" in entry.note
    # the active path and the transcript now start with the summary; the record keeps the originals
    assert [e.id for e in harness.session.active_path()] == ["e1", "e6", "e4", "e5"]
    assert [m.content for m in harness.messages] == [
        "[summary of earlier context]\nthe user sent a long x; the answer was one",
        "next",
        "two",
    ]
    assert isinstance(harness.session.get("e2"), MessageEntry)
    # one summarization call through the same model, with the replaced entries fenced in the prompt
    summary_call = harness.model.calls[2]["messages"]
    assert len(summary_call) == 1 and summary_call[0]["role"] == "user"
    assert summary_call[0]["content"].startswith(SUMMARY_PROMPT) and "[e2] user:" in summary_call[0]["content"]
    stats = harness.context_stats
    assert stats.compactions == 1 and stats.entries_replaced == 2 and stats.mechanical_summaries == 0
    assert len(stats.fill_at_turn_end) == 2 and all(f > 0.4 for f in stats.fill_at_turn_end)


def test_nothing_is_compacted_below_the_line(tmp_path):
    harness = two_turn_harness(tmp_path, ["one", "two"], window=100_000)
    events = collect(harness.prompt("x" * 3000))
    assert "compaction" not in types_of(events)
    assert harness.context_stats.compactions == 0
    assert len(harness.model.calls) == 2
    assert all(f < 0.4 for f in harness.context_stats.fill_at_turn_end)


def test_the_automatic_compaction_can_be_switched_off(tmp_path):
    harness = two_turn_harness(tmp_path, ["one", "two"], floor=False)
    events = collect(harness.prompt("x" * 3000))
    assert "compaction" not in types_of(events)
    assert harness.context_stats.compactions == 0
    # over the line at both turn ends and still no compaction, and no summarization call went out
    fills = harness.context_stats.fill_at_turn_end
    assert len(fills) == 2 and all(f > 0.4 for f in fills)
    assert len(harness.model.calls) == 2


def test_a_mechanical_summary_stands_in_when_the_model_cannot_write_one(tmp_path):
    harness = two_turn_harness(tmp_path, ["one", "two"])
    events = collect(harness.prompt("x" * 3000))
    event = next(e for e in events if e.type == "compaction")
    assert event.by == "code"
    assert event.summary.startswith("[mechanical summary: ") and "ran out of replies" in event.summary
    assert event.summary.splitlines()[1] == "- e2 user: " + "x" * 77 + "..."
    assert "mechanical summary because" in event.note
    stats = harness.context_stats
    assert stats.compactions == 1 and stats.mechanical_summaries == 1
    assert harness.messages[0].content.startswith("[summary of earlier context]\n[mechanical summary")
    # the summary lists kinds and first lines, one entry per line
    entries = [
        MessageEntry(id="a", message=UserMessage(content="\n\nfirst line\nsecond")),
        MessageEntry(id="b", message=AssistantMessage(content=None, tool_calls=[{"id": "c", "name": "add", "arguments": {"a": 1}}])),
        MessageEntry(id="c", message=ToolResultMessage(tool_call_id="c", tool_name="add", content="3")),
    ]
    text = mechanical_summary(entries, "no model")
    assert text.splitlines() == [
        "[mechanical summary: no model]",
        "- a user: first line",
        '- b assistant: call add({"a": 1})',
        "- c tool result add: 3",
    ]


def test_the_estimate_reads_characters_after_a_compaction_made_the_usage_stale(tmp_path):
    # The first assistant reports a huge usage; after the compaction that number counted a context
    # that is gone, and reading it again would compact on every later turn.
    first = ModelReply(content="one", usage=Usage(input=90_000, output=10))
    harness = two_turn_harness(tmp_path, [first, "two", "summary", "three"], window=100_000, line=0.5)
    harness.follow_up("and then")
    events = collect(harness.prompt("go"))
    assert types_of(events).count("compaction") == 1
    fills = harness.context_stats.fill_at_turn_end
    assert fills[0] > 0.5 and fills[-1] < 0.01


# --- what is kept ---


def test_a_protected_entry_and_the_current_turn_are_kept_where_they_stand(tmp_path):
    harness = two_turn_harness(tmp_path, ["one", "two", "summary"])
    api = ExtensionAPI(harness)
    api.protect(["e2"], "open finding refers to it")
    events = collect(harness.prompt("x" * 3000))
    event = next(e for e in events if e.type == "compaction")
    # e2 is protected and e4, e5 are the current turn; only e3 is old and free
    assert event.replaces_entry_ids == ["e3"]
    # e2 is kept, so what went is not the whole prefix and the entry claims none
    assert event.first_kept_entry_id is None
    assert [e.id for e in harness.session.active_path()] == ["e1", "e2", "e6", "e4", "e5"]


def test_the_recent_tool_batches_are_kept_verbatim_and_an_exchange_goes_whole(tmp_path, echo_tool):
    big = "y" * 1200
    replies = [
        reply(None, call("echo", {"text": big}, "c1")),
        reply(None, call("echo", {"text": big}, "c2")),
        "done",
        "summary of the first echo",
    ]
    harness = harness_with_session(
        tmp_path, TestModel(replies), [echo_tool], window=1_000_000, keep_recent_batches=1
    )
    collect(harness.prompt("go"))
    entry = asyncio.run(harness.compact())
    # the first exchange (assistant e3 with its result e4) is the older prefix and goes as one
    # unit with the prompt; the second exchange is the last batch and stays verbatim
    assert entry.replaces_entry_ids == ["e2", "e3", "e4"]
    assert entry.first_kept_entry_id == "e5"
    assert [e.id for e in harness.session.active_path()] == ["e1", "e8", "e5", "e6", "e7"]
    assert [m.role for m in harness.messages] == ["user", "assistant", "tool", "assistant"]
    assert harness.messages[2].content == big
    assert harness.context_stats.compactions == 1
    # an assistant message and the results answering it are one unit
    messages = [
        MessageEntry(id="a", message=UserMessage(content="go")),
        MessageEntry(id="b", message=AssistantMessage(tool_calls=[{"id": "c1", "name": "t", "arguments": {}}])),
        MessageEntry(id="c", message=ToolResultMessage(tool_call_id="c1", tool_name="t", content="ok")),
        MessageEntry(id="d", message=AssistantMessage(content="done")),
    ]
    assert [[e.id for e in unit] for unit in units(messages)] == [["a"], ["b", "c"], ["d"]]


# --- on request ---


def test_compact_on_request_summarizes_whatever_the_estimate_says(tmp_path):
    harness = harness_with_session(
        tmp_path, TestModel(["one", "two", "the summary"]), window=1_000_000, keep_recent_batches=1
    )
    collect(harness.prompt("go"))
    collect(harness.prompt("again"))
    assert harness.context_stats.compactions == 0
    seen = []
    harness.subscribe(seen.append)
    entry = asyncio.run(harness.compact("the run asked for it"))
    assert isinstance(entry, CompactionEntry) and entry.summary == "the summary"
    # the current turn is guarded, so the user message that opened it stays and the summary
    # claims no prefix
    assert entry.replaces_entry_ids == ["e2", "e3"] and entry.first_kept_entry_id is None
    assert "the run asked for it" in entry.note
    assert [e.type for e in seen] == ["compaction"]
    assert [m.content for m in harness.messages] == [
        "[summary of earlier context]\nthe summary",
        "again",
        "two",
    ]


def test_compact_on_request_does_nothing_without_a_session_or_with_nothing_older_than_the_window(tmp_path):
    harness = AgentHarness(TestModel(["one"]))
    collect(harness.prompt("go"))
    assert asyncio.run(harness.compact()) is None
    harness = harness_with_session(tmp_path, TestModel(["one"]), window=1_000_000, keep_recent_batches=10)
    collect(harness.prompt("go"))
    assert asyncio.run(harness.compact()) is None
    assert harness.context_stats.compactions == 0
    assert [m.content for m in harness.messages] == ["go", "one"]
