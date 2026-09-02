"""Context accounting: the estimate, the session record behind a harness, the note on tool results,
the floor at the line with the model's summary and with the mechanical one, protect, the counters."""

from __future__ import annotations

from kullback.agent.context import (
    SUMMARY_PROMPT,
    ContextConfig,
    ContextEstimate,
    estimate_context,
    mechanical_summary,
    text_tokens,
)
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from kullback.agent.session import CompactionEntry, MessageEntry, SessionInfoEntry, SessionStore
from kullback.agent.tools import ToolResult
from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from tests.agent.conftest import call, collect, reply, types_of


def harness_with_session(tmp_path, model, tools=(), **config):
    store = SessionStore(tmp_path / "session.jsonl")
    return AgentHarness(model, system="sys", tools=list(tools), session=store, context=ContextConfig(**config))


# --- the estimate ---


def test_heuristic_estimate_counts_the_wire_and_says_so():
    messages = [UserMessage(content="a" * 400), AssistantMessage(content="b" * 400)]
    estimate = estimate_context(messages, system="s" * 100, tool_schemas=[{"name": "t"}], window=1000, line=0.4)
    assert estimate.source == "heuristic"
    assert estimate.tokens >= text_tokens("a" * 400 + "b" * 400 + "s" * 100)
    assert estimate.line_tokens == 400 and estimate.window == 1000
    assert estimate.note().startswith(f"context {estimate.fill:.0%} of 1000, line at 40%, estimated from characters")


def test_usage_estimate_uses_the_last_reported_usage_plus_the_tail():
    tail = ToolResultMessage(tool_call_id="c", tool_name="t", content="r" * 80)
    messages = [
        UserMessage(content="go"),
        AssistantMessage(content="x", usage=Usage(input=900, cache_read=50, output=10)),
        tail,
    ]
    estimate = estimate_context(messages, window=2000)
    assert estimate.source == "usage"
    # 900 input + 50 cached + 10 output, plus the tail by characters: the 116-character wire
    # rendering of that tool result over four, rounded up
    assert estimate.tokens == 960 + 37
    assert "estimated from usage" in estimate.note()
    # after a compaction the usage is stale and the caller asks for characters only
    assert estimate_context(messages, use_usage=False).source == "heuristic"


def test_over_line_is_strictly_above_the_line():
    assert ContextEstimate(tokens=400, window=1000, line=0.4, source="heuristic").over_line is False
    assert ContextEstimate(tokens=401, window=1000, line=0.4, source="heuristic").over_line is True


# --- the record ---


def test_a_harness_with_a_session_records_every_message_and_reads_the_active_path(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("three")])
    harness = harness_with_session(tmp_path, model, [add_tool], arm="code_only")
    collect(harness.prompt("go"))
    store = harness.session
    kinds = [(e.id, e.type) for e in store.active_path()]
    assert kinds == [("e1", "session_info"), ("e2", "message"), ("e3", "message"), ("e4", "message"), ("e5", "message")]
    assert isinstance(store.entries[0], SessionInfoEntry) and store.entries[0].arm == "code_only"
    assert [e.message.role for e in store.entries[1:]] == ["user", "assistant", "tool", "assistant"]
    assert [m.role for m in harness.messages] == ["user", "assistant", "tool", "assistant"]
    reloaded = SessionStore.load(store.path)
    assert [m.content for m in reloaded.active_messages()] == [m.content for m in harness.messages]


def test_messages_given_at_construction_are_recorded_first(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    harness = AgentHarness(TestModel(["ok"]), messages=[UserMessage(content="earlier")], session=store)
    assert [e.type for e in store.entries] == ["session_info", "message"]
    assert [m.content for m in harness.messages] == ["earlier"]


def test_entry_ids_never_reuse_numbers_on_a_reloaded_session(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    first = AgentHarness(TestModel(["one"]), session=store)
    collect(first.prompt("go"))
    second = AgentHarness(TestModel(["two"]), session=SessionStore.load(store.path))
    collect(second.prompt("again"))
    ids = [e.id for e in second.session.entries]
    assert len(ids) == len(set(ids))
    assert ids[-1] == f"e{len(ids)}"


# --- the note ---


def test_context_note_is_off_by_default(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool])
    collect(harness.prompt("go"))
    assert harness.messages[2].content == '{"total": 3}'
    plain = AgentHarness(TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")]), tools=[add_tool])
    collect(plain.prompt("go"))
    assert plain.messages[2].content == '{"total": 3}'


def test_context_note_on_appends_the_estimate_and_the_entry_id_and_the_hook_is_gone_between_runs(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool], note=True, window=5000)
    events = collect(harness.prompt("go"))
    content = harness.messages[2].content
    first, note = content.split("\n")
    assert first == '{"total": 3}'
    assert note.startswith("context ") and "of 5000, line at 40%, estimated from characters; this result is entry e4" in note
    assert harness.session.get("e4").message.tool_call_id == "c1"
    end = next(e for e in events if e.type == "tool_execution_end")
    assert end.result.content == content and end.result.details == {"total": 3}
    # the model read the note on the wire
    assert model.calls[1]["messages"][-1]["content"] == content
    # and the hook is gone between runs
    assert harness.hooks.tool_result == []


def test_context_note_without_a_session_carries_no_entry_id(add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = AgentHarness(model, tools=[add_tool], context=ContextConfig(note=True))
    collect(harness.prompt("go"))
    note = harness.messages[2].content.split("\n")[1]
    assert note.startswith("context ") and "entry" not in note


def test_context_note_reads_usage_when_the_provider_reported_it(tmp_path, add_tool):
    first = ModelReply(tool_calls=[call("add", {"a": 1, "b": 2}, "c1")], usage=Usage(input=4000, output=20))
    model = TestModel([first, reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool], note=True, window=10000)
    collect(harness.prompt("go"))
    note = harness.messages[2].content.split("\n")[1]
    assert "estimated from usage" in note and note.startswith("context 40% of 10000")


def test_note_hook_runs_after_the_extension_hooks(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool], note=True)

    def setup(api: ExtensionAPI):
        api.tool_result(lambda c, r: ToolResult(content=r.content + "\ngate: accepted", details=r.details))

    load_extensions(harness, [setup])
    collect(harness.prompt("go"))
    lines = harness.messages[2].content.split("\n")
    assert lines[0] == '{"total": 3}' and lines[1] == "gate: accepted" and lines[2].startswith("context ")


# --- the floor ---


def floor_harness(tmp_path, replies, **config):
    defaults = dict(window=1000, line=0.4, recent_tool_turns=1, arm="code_only")
    defaults.update(config)
    harness = harness_with_session(tmp_path, TestModel(replies), **defaults)
    harness.follow_up("next")
    return harness


def test_floor_compacts_at_the_line_with_the_models_summary(tmp_path):
    harness = floor_harness(tmp_path, ["one", "two", "the user sent a long x; the answer was one"])
    events = collect(harness.prompt("x" * 3000))
    kinds = types_of(events)
    assert kinds.count("compaction") == 1
    turn_ends = [i for i, k in enumerate(kinds) if k == "turn_end"]
    compaction_at = kinds.index("compaction")
    assert turn_ends[0] < compaction_at and turn_ends[1] < compaction_at < kinds.index("agent_end")
    event = events[compaction_at]
    # the oldest entry alone brings the estimate under the line, so the floor stops there
    assert event.by == "code_fallback" and event.replaces_entry_ids == ["e2"] and event.first_kept_entry_id == "e3"
    assert event.summary == "the user sent a long x; the answer was one" and event.entry_id == "e6"
    entry = harness.session.get("e6")
    assert isinstance(entry, CompactionEntry) and entry.by == "code_fallback"
    assert "code_fallback at turn 2" in entry.note and "summary by the model" in entry.note
    # the active path and the transcript now start with the summary; the record keeps the originals
    assert [e.id for e in harness.session.active_path()] == ["e1", "e6", "e3", "e4", "e5"]
    assert [m.content for m in harness.messages] == [
        "[summary of earlier context]\nthe user sent a long x; the answer was one",
        "one",
        "next",
        "two",
    ]
    assert isinstance(harness.session.get("e2"), MessageEntry)
    # one summarization call through the same model, with the dropped entries in the prompt
    summary_call = harness.model.calls[2]["messages"]
    assert len(summary_call) == 1 and summary_call[0]["role"] == "user"
    assert summary_call[0]["content"].startswith(SUMMARY_PROMPT) and "[e2] user:" in summary_call[0]["content"]
    stats = harness.context_stats
    assert stats.fallback_compactions == 1 and stats.mechanical_summaries == 0
    assert len(stats.fill_at_turn_end) == 2 and all(f > 0.4 for f in stats.fill_at_turn_end)


def test_floor_does_not_trigger_below_the_line(tmp_path):
    harness = floor_harness(tmp_path, ["one", "two"], window=100_000)
    events = collect(harness.prompt("x" * 3000))
    assert "compaction" not in types_of(events)
    assert harness.context_stats.fallback_compactions == 0
    assert len(harness.model.calls) == 2
    assert all(f < 0.4 for f in harness.context_stats.fill_at_turn_end)


def test_floor_can_be_switched_off(tmp_path):
    harness = floor_harness(tmp_path, ["one", "two"], floor=False)
    events = collect(harness.prompt("x" * 3000))
    assert "compaction" not in types_of(events)
    assert harness.context_stats.fallback_compactions == 0
    # over the line at both turn ends and still no compaction, and no summarization call went out
    fills = harness.context_stats.fill_at_turn_end
    assert len(fills) == 2 and all(f > 0.4 for f in fills)
    assert len(harness.model.calls) == 2


def test_floor_writes_a_mechanical_summary_when_the_model_cannot(tmp_path):
    harness = floor_harness(tmp_path, ["one", "two"])
    events = collect(harness.prompt("x" * 3000))
    event = next(e for e in events if e.type == "compaction")
    assert event.by == "code_fallback"
    assert event.summary.startswith("[mechanical summary: ") and "ran out of replies" in event.summary
    assert event.summary.splitlines()[1] == "- e2 user: " + "x" * 77 + "..."
    assert "mechanical summary because" in event.note
    stats = harness.context_stats
    assert stats.fallback_compactions == 1 and stats.mechanical_summaries == 1
    assert harness.messages[0].content.startswith("[summary of earlier context]\n[mechanical summary")


def test_mechanical_summary_lists_kinds_and_first_lines():
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


def test_floor_keeps_protected_entries_and_the_current_turn(tmp_path):
    harness = floor_harness(tmp_path, ["one", "two", "summary"])
    api = ExtensionAPI(harness)
    api.protect(["e2"], "open finding refers to it")
    events = collect(harness.prompt("x" * 3000))
    event = next(e for e in events if e.type == "compaction")
    # e2 (protected) and e4, e5 (the current turn) stay; only e3 is old and free
    assert event.replaces_entry_ids == ["e3"]
    # e2 is kept, so the dropped set is not a prefix and the entry claims no prefix
    assert event.first_kept_entry_id is None
    assert [e.id for e in harness.session.active_path()] == ["e1", "e2", "e6", "e4", "e5"]
    assert "still over the line" in event.note


def test_floor_keeps_recent_tool_output_and_whole_exchanges(tmp_path, echo_tool):
    big = "y" * 1200
    replies = [
        reply(None, call("echo", {"text": big}, "c1")),
        reply(None, call("echo", {"text": big}, "c2")),
        "summary of the first echo",
        "done",
        "summary again",
    ]
    harness = harness_with_session(tmp_path, TestModel(replies), [echo_tool], window=1500, line=0.4, recent_tool_turns=1)
    events = collect(harness.prompt("go"))
    compactions = [e for e in events if e.type == "compaction"]
    assert len(compactions) == 2
    # at the end of turn 2 the first exchange (assistant e3 with its result e4) went as one unit
    # with the prompt; the second exchange is the current turn and stayed, still over the line
    assert compactions[0].replaces_entry_ids == ["e2", "e3", "e4"] and "still over the line" in compactions[0].note
    # at the end of turn 3 the second exchange is the last tool output and stays; only the
    # summary itself is old and free, so it is summarized again
    assert compactions[1].replaces_entry_ids == ["e7"] and "still over the line" in compactions[1].note
    assert [e.id for e in harness.session.active_path()] == ["e1", "e9", "e5", "e6", "e8"]
    roles = [m.role for m in harness.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert harness.context_stats.fallback_compactions == 2


def test_floor_uses_characters_after_a_compaction_made_usage_stale(tmp_path):
    # The first assistant reports a huge usage; after the floor compacts, that number is stale
    # and the next estimate must not read it, or the floor would fire on every later turn.
    first = ModelReply(content="one", usage=Usage(input=90_000, output=10))
    harness = floor_harness(tmp_path, [first, "two", "summary", "three"], window=100_000, line=0.5)
    harness.follow_up("and then")
    events = collect(harness.prompt("go"))
    assert types_of(events).count("compaction") == 1
    fills = harness.context_stats.fill_at_turn_end
    assert fills[0] > 0.5 and fills[-1] < 0.01


# --- protect through the api, and the counters ---


def test_protect_and_unprotect_through_the_api_by_tool_call_id(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool], arm="code_only")
    ids = []

    def setup(api: ExtensionAPI):
        def on_end(event):
            entry_id = api.entry_id_for(event.tool_call_id)
            ids.append(entry_id)
            api.protect([entry_id], "unacted ruling on add")

        api.on("tool_execution_end", on_end)

    load_extensions(harness, [setup])
    collect(harness.prompt("go"))
    assert ids == ["e4"] and harness.session.get("e4").message.tool_call_id == "c1"
    assert harness.context.protected == {"e4": "unacted ruling on add"}
    assert harness.context.guards()["e4"][0] == "protected"
    ExtensionAPI(harness).unprotect(["e4"])
    assert "e4" not in harness.context.protected


def test_the_fill_is_counted_at_every_turn_end_without_a_session():
    harness = AgentHarness(TestModel(["one", "two"]))
    collect(harness.prompt("go"))
    collect(harness.prompt("again"))
    fills = harness.context_stats.fill_at_turn_end
    assert len(fills) == 2 and all(0 < f < 1 for f in fills)
