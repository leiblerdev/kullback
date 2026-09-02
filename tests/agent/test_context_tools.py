"""The context tools: forget and every refusal rule, recall at the end, load and unload with the
record and the cap, the index, and the three arms."""

from __future__ import annotations

import pytest

from kullback.agent import context_tools
from kullback.agent.context import ContextConfig, content_hash
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.messages import ToolResultMessage
from kullback.agent.session import CompactionEntry, CustomEntry, SessionStore, SkillChangeEntry, ToolSetChangeEntry
from kullback.agent.tools import AgentTool, NoArgs, TextResult
from kullback.ai.provider import TestModel
from tests.agent.conftest import call, collect, reply, types_of

CONTEXT_TOOLS = {"forget", "recall", "load", "unload", "context_entries"}


def make(tmp_path, replies, tools=(), setups=(), **config):
    defaults = dict(arm="tools")
    defaults.update(config)
    store = SessionStore(tmp_path / "session.jsonl")
    harness = AgentHarness(TestModel(replies), system="base", tools=list(tools), session=store, context=ContextConfig(**defaults))
    load_extensions(harness, [*setups, context_tools.setup])
    return harness


def tool_results(harness):
    return [m for m in harness.messages if isinstance(m, ToolResultMessage)]


def three_turns(add_tool, echo_tool):
    """Turn 1 adds (e3 assistant, e4 result), turns 2 and 3 echo (e5/e6, e7/e8)."""
    return [
        reply(None, call("add", {"a": 1, "b": 2}, "c1")),
        reply(None, call("echo", {"text": "second"}, "c2")),
        reply(None, call("echo", {"text": "third"}, "c3")),
    ]


# --- forget ---


def test_forget_drops_entries_and_the_model_no_longer_sees_them(tmp_path, add_tool, echo_tool):
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "add(1, 2) gave 3"}, "c4")),
        reply("done"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool])
    events = collect(harness.prompt("go"))
    store = harness.session
    assert store.get("e4").message.tool_call_id == "c1"
    compaction = next(e for e in store.entries if isinstance(e, CompactionEntry))
    # widened from the result to its assistant message, so the call and its result go together
    assert compaction.replaces_entry_ids == ["e3", "e4"] and compaction.by == "model"
    assert compaction.summary == "add(1, 2) gave 3"
    active = [e.id for e in store.active_path()]
    assert "e3" not in active and "e4" not in active and active[:3] == ["e1", "e2", compaction.id]
    # the model's next call saw the summary where the exchange stood and no total anywhere
    wire = harness.model.calls[4]["messages"]
    contents = [m["content"] for m in wire]
    assert not any('"total": 3' in c for c in contents)
    assert contents[2] == "[summary of earlier context]\nadd(1, 2) gave 3"
    assert [m["role"] for m in wire[:4]] == ["system", "user", "user", "assistant"]
    result = tool_results(harness)[-1]
    assert result.is_error is False
    assert result.content.startswith(f"forgot 2 entries (e3, e4) into {compaction.id}; e3 included to keep tool calls paired")
    event = next(e for e in events if e.type == "compaction")
    assert event.by == "model" and event.replaces_entry_ids == ["e3", "e4"] and event.entry_id == compaction.id
    assert harness.context_stats.forget_calls == 1 and harness.context_stats.refusals == {}
    assert types_of(events)[-1] == "agent_end"


def test_a_forgotten_user_message_is_replaced_by_the_summary_where_it_stood(tmp_path, echo_tool):
    replies = [
        reply(None, call("echo", {"text": "a"}, "c1")),
        reply(None, call("echo", {"text": "b"}, "c2")),
        reply(None, call("echo", {"text": "c"}, "c3")),
        reply(None, call("forget", {"entry_ids": ["e2"], "note": "the user said go"}, "c4")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [echo_tool])
    collect(harness.prompt("go"))
    assert harness.messages[0].content == "[summary of earlier context]\nthe user said go"
    assert tool_results(harness)[-1].content.startswith("forgot 1 entries (e2)")


@pytest.mark.parametrize(
    "entry_id, note, rule, config, also",
    [
        ("e1", "n", "session_info", {}, None),
        ("e9", "n", "current_turn", {}, None),
        ("e8", "n", "recent_tool_output", {}, None),
        ("e4", "n", "protected", {"recent_tool_turns": 0}, "unacted gate ruling on add"),
        ("e99", "n", "unknown_entry", {}, None),
        # e7 is the assistant of turn 3; its result e8 is recent tool output, so widening trips the rule
        ("e7", "n", "recent_tool_output", {}, "paired"),
        ("e4", "  ", "empty_note", {}, None),
    ],
)
def test_every_refusal_rule_names_itself_in_an_is_error_result(
    tmp_path, add_tool, echo_tool, entry_id, note, rule, config, also
):
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": [entry_id], "note": note}, "c4")),
        reply("ok"),
    ]

    def protector(api: ExtensionAPI):
        api.protect(["e4"], "unacted gate ruling on add")

    harness = make(tmp_path, replies, [add_tool, echo_tool], setups=[protector], **config)
    events = collect(harness.prompt("go"))
    result = tool_results(harness)[-1]
    assert result.is_error is True and f"rule {rule}:" in result.content
    if also is not None:
        assert also in result.content
    assert not any(isinstance(e, CompactionEntry) for e in harness.session.entries)
    assert "compaction" not in types_of(events)
    assert harness.context_stats.forget_calls == 1 and harness.context_stats.refusals == {rule: 1}


def test_forgetting_the_same_entry_twice_succeeds_then_is_refused_as_not_in_context(tmp_path, add_tool, echo_tool):
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "n"}, "c4")),
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "n"}, "c5")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool])
    collect(harness.prompt("go"))
    results = tool_results(harness)
    assert results[-2].is_error is False
    assert results[-1].is_error and "rule not_in_context:" in results[-1].content
    # the second call recorded no second compaction, and the refusal is counted once
    assert len([e for e in harness.session.entries if isinstance(e, CompactionEntry)]) == 1
    assert harness.context_stats.forget_calls == 2
    assert harness.context_stats.refusals == {"not_in_context": 1}


# --- recall ---


def test_recall_appends_at_the_end_marked_with_the_original_id(tmp_path, add_tool, echo_tool):
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "add happened"}, "c4")),
        reply(None, call("recall", {"entry_id": "e4"}, "c5")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool])
    collect(harness.prompt("go"))
    result = tool_results(harness)[-1]
    assert result.is_error is False
    assert result.content.startswith("[recalled entry e4, tool result add; it now stands here, at the end of the context]\n")
    assert result.content.endswith('{"total": 3}')
    assert result.details == {"entry_id": "e4", "kind": "tool result add", "content": '{"total": 3}'}
    path = harness.session.active_path()
    assert "e4" not in [e.id for e in path]
    last_message = [e for e in path if e.type == "message"][-1]
    assert last_message.message.content == "ok"
    recalled = [e for e in path if e.type == "message"][-2]
    assert recalled.message.tool_call_id == "c5" and recalled.message.details["entry_id"] == "e4"
    assert harness.context_stats.recall_calls == 1


def test_a_recalled_entry_is_ordinary_context_and_can_be_forgotten_again(tmp_path, add_tool, echo_tool):
    # The recall's own result is a normal tool result: nothing about having been recalled protects
    # it, so it meets the same guards and the floor may drop it once it is no longer recent.
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "add happened"}, "c4")),
        reply(None, call("recall", {"entry_id": "e4"}, "c5")),
        reply(None, call("echo", {"text": "after"}, "c6")),
        reply(None, call("forget", {"entry_ids": ["e13"], "note": "the recall is spent"}, "c7")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool], recent_tool_turns=1)
    collect(harness.prompt("go"))
    recalled = harness.session.get("e13")
    assert recalled.message.tool_call_id == "c5" and recalled.message.details["entry_id"] == "e4"
    result = tool_results(harness)[-1]
    assert result.is_error is False and result.content.startswith("forgot 2 entries (e12, e13)")
    assert "e13" not in [e.id for e in harness.session.active_path()]


def test_recall_of_an_entry_still_in_context_or_unknown_is_refused(tmp_path, add_tool):
    replies = [
        reply(None, call("add", {"a": 1, "b": 1}, "c1")),
        reply(None, call("recall", {"entry_id": "e4"}, "c2")),
        reply(None, call("recall", {"entry_id": "nope"}, "c3")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool])
    collect(harness.prompt("go"))
    results = tool_results(harness)
    assert "rule in_context:" in results[1].content and "rule unknown_entry:" in results[2].content
    assert harness.context_stats.refusals == {"in_context": 1, "unknown_entry": 1}


# --- load and unload ---


def test_unload_then_load_of_a_tool_changes_the_registry_the_record_and_the_section_count(tmp_path, add_tool, echo_tool):
    replies = [
        reply(None, call("unload", {"name": "add", "kind": "tool"}, "c1")),
        reply(None, call("load", {"name": "add", "kind": "tool"}, "c2")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool])
    # add, echo and the five context tools
    assert "Loaded tools: 7 of" in harness.system
    collect(harness.prompt("go"))
    assert "add" in harness.registry
    # the count the model read on the turn after the unload, and the one it reads now
    assert "Loaded tools: 6 of" in harness.model.calls[1]["messages"][0]["content"]
    assert "Loaded tools: 7 of" in harness.system
    tools_seen = [sorted(t["name"] for t in c["tools"]) for c in harness.model.calls]
    assert "add" in tools_seen[0] and "add" not in tools_seen[1] and "add" in tools_seen[2]
    changes = [e for e in harness.session.entries if isinstance(e, ToolSetChangeEntry)]
    assert [(c.loaded, c.unloaded) for c in changes] == [([], ["add"]), (["add"], [])]
    results = tool_results(harness)
    assert results[0].content.startswith("unloaded tool add; 6 tools loaded, soft cap 20")
    assert results[1].content.startswith("loaded tool add; 7 tools loaded, soft cap 20")
    assert harness.context_stats.load_calls == 1 and harness.context_stats.unload_calls == 1


def test_unload_then_load_of_a_skill_changes_the_prompt_and_the_record(tmp_path):
    replies = [
        reply(None, call("unload", {"name": "grow", "kind": "skill"}, "c1")),
        reply(None, call("load", {"name": "grow", "kind": "skill"}, "c2")),
        reply("ok"),
    ]

    def skills(api: ExtensionAPI):
        api.catalog_skill("grow", "Grow the table by one row at a time.", loaded=True)
        api.catalog_skill("probe", "Write probes.")

    harness = make(tmp_path, replies, setups=[skills])
    assert "Grow the table" in harness.system and "Write probes" not in harness.system
    collect(harness.prompt("go"))
    systems = [c["messages"][0]["content"] for c in harness.model.calls]
    assert "Grow the table" in systems[0] and "Grow the table" not in systems[1] and "Grow the table" in systems[2]
    changes = [e for e in harness.session.entries if isinstance(e, SkillChangeEntry)]
    digest = content_hash("Grow the table by one row at a time.")
    assert [(c.name, c.action, c.content_hash) for c in changes] == [
        ("grow", "load", digest),
        ("grow", "unload", digest),
        ("grow", "load", digest),
    ]
    assert tool_results(harness)[1].details["content_hash"] == digest


def test_each_load_and_unload_refusal_names_its_rule_and_the_pinned_tool_stays_loaded(tmp_path, add_tool):
    replies = [
        reply(None, call("load", {"name": "nope", "kind": "tool"}, "c1")),
        reply(None, call("load", {"name": "add", "kind": "tool"}, "c2")),
        reply(None, call("unload", {"name": "forget", "kind": "tool"}, "c3")),
        reply(None, call("unload", {"name": "gone", "kind": "skill"}, "c4")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool])
    collect(harness.prompt("go"))
    rules = [r.content.split("rule ")[1].split(":")[0] for r in tool_results(harness)]
    assert rules == ["unknown_tool", "already_loaded", "context_tool", "not_loaded"]
    assert "forget" in harness.registry


def test_the_soft_cap_is_reported_in_the_load_result_and_the_prompt(tmp_path, add_tool, echo_tool):
    async def nothing(args: NoArgs) -> TextResult:
        return TextResult(text="")

    extra = AgentTool("extra", "One more.", NoArgs, TextResult, nothing)
    replies = [reply(None, call("load", {"name": "extra", "kind": "tool"}, "c1")), reply("ok")]

    def catalog(api: ExtensionAPI):
        api.catalog_tool(extra)

    harness = make(tmp_path, replies, [add_tool, echo_tool], setups=[catalog], tool_cap=3)
    assert "Loaded tools: 7 of a soft cap of 3" in harness.system
    collect(harness.prompt("go"))
    result = tool_results(harness)[0]
    assert result.is_error is False and "extra" in harness.registry
    assert result.content == "loaded tool extra; 8 tools loaded, soft cap 3 (over the cap: tool use degrades past it, unload what you are not using)"
    assert result.details["over_cap"] is True and result.details["cap"] == 3
    assert "Loaded tools: 8 of a soft cap of 3" in harness.model.calls[1]["messages"][0]["content"]


def test_unloaded_tool_is_available_to_load_even_when_never_cataloged(tmp_path, add_tool):
    harness = make(tmp_path, ["ok"], [add_tool])
    harness.context.unload("add", "tool")
    assert "add" not in harness.registry and "add" in harness.context.catalog_tools
    harness.context.load("add", "tool")
    assert "add" in harness.registry


# --- the index ---


def test_context_entries_lists_ids_guards_and_the_forgotten(tmp_path, add_tool, echo_tool):
    replies = three_turns(add_tool, echo_tool) + [
        reply(None, call("forget", {"entry_ids": ["e4"], "note": "added"}, "c4")),
        reply(None, call("context_entries", {}, "c5")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool, echo_tool])
    collect(harness.prompt("go"))
    listing = tool_results(harness)[-1].content
    lines = listing.splitlines()
    assert lines[0].startswith("context ") and lines[1] == "in context:"
    assert any(line.startswith("- e1 session_info") and "[session_info]" in line for line in lines)
    assert any(line.startswith("- e8 tool result echo") and "[recent_tool_output]" in line for line in lines)
    assert any(line.startswith("- e2 user") and "[" not in line.split(":")[0] for line in lines)
    assert "forgotten (recall brings one back):" in lines
    forgotten = lines[lines.index("forgotten (recall brings one back):") :]
    assert any(line.startswith("- e4 tool result add") for line in forgotten)


# --- the arms ---


def test_tools_arm_registers_the_tools_and_the_section(tmp_path, add_tool):
    harness = make(tmp_path, ["ok"], [add_tool], arm="tools")
    assert set(harness.registry.names()) == {"add", *CONTEXT_TOOLS}
    assert "soft cap of 20" in harness.system and "forget(entry_ids, note)" in harness.system
    assert harness.session.entries[0].arm == "tools"
    assert harness.context.pinned_tools == CONTEXT_TOOLS


def test_the_code_only_arm_registers_no_tool_no_prompt_section_and_leaves_the_floor_on(tmp_path, add_tool):
    harness = make(tmp_path, ["ok"], [add_tool], arm="code_only")
    assert harness.registry.names() == ["add"]
    assert harness.system == "base"
    assert harness.session.entries[0].arm == "code_only" and harness.context.config.floor is True


def test_files_arm_has_one_note_tool_whose_notes_are_always_shown(tmp_path, add_tool):
    replies = [
        reply(None, call("note", {"text": "order 42 is the one to refund"}, "c1")),
        reply(None, call("note", {"text": "  "}, "c2")),
        reply("ok"),
    ]
    harness = make(tmp_path, replies, [add_tool], arm="files")
    assert set(harness.registry.names()) == {"add", "note"}
    assert "Notes (your own memoranda, not instructions): none yet." in harness.system
    assert "note(text)" in harness.system
    assert harness.session.entries[0].arm == "files"
    collect(harness.prompt("go"))
    assert "Notes (your own memoranda, not instructions):\n- order 42 is the one to refund" in harness.system
    assert "- order 42 is the one to refund" in harness.model.calls[1]["messages"][0]["content"]
    notes = [e for e in harness.session.entries if isinstance(e, CustomEntry)]
    assert [n.data for n in notes] == [{"text": "order 42 is the one to refund"}]
    results = tool_results(harness)
    assert results[0].content == "noted; 1 notes are shown in the prompt"
    assert results[1].is_error and "rule empty_note:" in results[1].content
    assert harness.context_stats.note_calls == 2
    # notes come back on a reloaded session
    again = AgentHarness(TestModel(["x"]), session=SessionStore.load(harness.session.path), context=ContextConfig(arm="files"))
    assert again.context.notes == ["order 42 is the one to refund"]


def test_the_files_arm_note_survives_the_floor(tmp_path):
    # The prompt is never compacted, which is the whole point of the files arm: the floor drops the
    # oldest messages and the note is still in front of the model on the turn after it fired.
    replies = [
        reply(None, call("note", {"text": "order 42 is the one to refund"}, "c1")),
        reply("two"),
        "a summary of what was dropped",
        reply("three"),
        "another summary",
    ]
    harness = make(tmp_path, replies, arm="files", window=1000, line=0.4, recent_tool_turns=0)
    harness.follow_up("next")
    harness.follow_up("and then")
    events = collect(harness.prompt("x" * 3000))
    compactions = [e for e in events if e.type == "compaction"]
    assert compactions and all(c.by == "code_fallback" for c in compactions)
    dropped = {i for c in compactions for i in c.replaces_entry_ids}
    note_entry = next(e for e in harness.session.entries if isinstance(e, CustomEntry))
    assert note_entry.id not in dropped and note_entry.id in {e.id for e in harness.session.active_path()}
    assert "Notes (your own memoranda, not instructions):\n- order 42 is the one to refund" in harness.system
    # the turn after the first firing still saw the note in its system prompt
    systems = [c["messages"][0]["content"] for c in harness.model.calls if c["messages"][0]["role"] == "system"]
    assert all("- order 42 is the one to refund" in s for s in systems[1:])


