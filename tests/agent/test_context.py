"""Context accounting: the estimate, the session record behind a harness, the note on tool results,
protect, the counters. What happens when the estimate crosses the line is test_compaction.py."""

from __future__ import annotations

from kullback.agent.context import ContextConfig, ContextEstimate, estimate_context, text_tokens
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from kullback.agent.session import SessionInfoEntry, SessionStore
from kullback.agent.tools import ToolResult
from kullback.ai.provider import ModelReply, TestModel
from kullback.ai.usage import Usage
from tests.agent.conftest import call, collect, reply


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


def test_a_session_records_every_message_and_reloads_without_reusing_entry_ids(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("three")])
    harness = harness_with_session(tmp_path, model, [add_tool])
    collect(harness.prompt("go"))
    store = harness.session
    kinds = [(e.id, e.type) for e in store.active_path()]
    assert kinds == [("e1", "session_info"), ("e2", "message"), ("e3", "message"), ("e4", "message"), ("e5", "message")]
    assert isinstance(store.entries[0], SessionInfoEntry)
    assert [e.message.role for e in store.entries[1:]] == ["user", "assistant", "tool", "assistant"]
    assert [m.role for m in harness.messages] == ["user", "assistant", "tool", "assistant"]
    reloaded = SessionStore.load(store.path)
    assert [m.content for m in reloaded.active_messages()] == [m.content for m in harness.messages]
    # a harness on the reloaded session never reuses an entry number
    second = AgentHarness(TestModel(["two"]), session=reloaded)
    collect(second.prompt("again"))
    ids = [e.id for e in second.session.entries]
    assert len(ids) == len(set(ids))
    assert ids[-1] == f"e{len(ids)}"
    # messages given at construction are recorded first
    early = SessionStore(tmp_path / "early.jsonl")
    seeded = AgentHarness(TestModel(["ok"]), messages=[UserMessage(content="earlier")], session=early)
    assert [e.type for e in early.entries] == ["session_info", "message"]
    assert [m.content for m in seeded.messages] == ["earlier"]


# --- the note ---


def test_context_note_is_off_by_default(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool])
    collect(harness.prompt("go"))
    assert harness.messages[2].content == '{"total": 3}'
    plain = AgentHarness(TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")]), tools=[add_tool])
    collect(plain.prompt("go"))
    assert plain.messages[2].content == '{"total": 3}'


def test_context_note_on_appends_the_estimate_and_the_entry_id_after_the_extension_hooks(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool], note=True, window=5000)
    events = collect(harness.prompt("go"))
    content = harness.messages[2].content
    first, note = content.split("\n")
    assert first == '{"total": 3}'
    assert (
        note.startswith("context ")
        and "of 5000, line at 40%, estimated from characters; this result is entry e4" in note
    )
    assert harness.session.get("e4").message.tool_call_id == "c1"
    end = next(e for e in events if e.type == "tool_execution_end")
    assert end.result.content == content and end.result.details == {"total": 3}
    # the model read the note on the wire
    assert model.calls[1]["messages"][-1]["content"] == content
    # and the hook is gone between runs
    assert harness.hooks.tool_result == []

    # without a session the note carries no entry id
    bare = AgentHarness(
        TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")]),
        tools=[add_tool],
        context=ContextConfig(note=True),
    )
    collect(bare.prompt("go"))
    bare_note = bare.messages[2].content.split("\n")[1]
    assert bare_note.startswith("context ") and "entry" not in bare_note

    # the note reads usage when the provider reported it
    first_reply = ModelReply(tool_calls=[call("add", {"a": 1, "b": 2}, "c1")], usage=Usage(input=4000, output=20))
    used = harness_with_session(
        tmp_path / "usage", TestModel([first_reply, reply("ok")]), [add_tool], note=True, window=10000
    )
    collect(used.prompt("go"))
    usage_note = used.messages[2].content.split("\n")[1]
    assert "estimated from usage" in usage_note and usage_note.startswith("context 40% of 10000")

    # the note hook runs after the extension hooks
    gated = harness_with_session(
        tmp_path / "gate",
        TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")]),
        [add_tool],
        note=True,
    )

    def setup(api: ExtensionAPI):
        api.tool_result(lambda c, r: ToolResult(content=r.content + "\ngate: accepted", details=r.details))

    load_extensions(gated, [setup])
    collect(gated.prompt("go"))
    lines = gated.messages[2].content.split("\n")
    assert lines[0] == '{"total": 3}' and lines[1] == "gate: accepted" and lines[2].startswith("context ")


# --- protect through the api, and the counters ---


def test_protect_and_unprotect_through_the_api_by_tool_call_id(tmp_path, add_tool):
    model = TestModel([reply(None, call("add", {"a": 1, "b": 2}, "c1")), reply("ok")])
    harness = harness_with_session(tmp_path, model, [add_tool])
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
