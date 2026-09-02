"""The session tree: append-only JSONL, branch by leaf pointer, replay with compactions applied."""

from __future__ import annotations

import json

import pytest

from kullback.agent.messages import AssistantMessage, UserMessage
from kullback.agent.session import (
    CompactionEntry,
    CustomEntry,
    LeafEntry,
    MessageEntry,
    SessionInfoEntry,
    SessionStore,
    SessionTreeError,
    SkillChangeEntry,
    ToolSetChangeEntry,
)


def test_append_chains_under_the_leaf_and_writes_one_line_each(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    info = store.append(SessionInfoEntry(title="t"))
    m1 = store.append_message(UserMessage(content="hi"))
    m2 = store.append_message(AssistantMessage(content="hello"))
    assert info.parent_id is None and m1.parent_id == info.id and m2.parent_id == m1.id
    assert store.leaf_id == m2.id
    lines = (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["session_info", "message", "message"]


def test_load_replays_the_file_and_the_path(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    store.append(SessionInfoEntry())
    store.append_message(UserMessage(content="a"))
    store.append(ToolSetChangeEntry(loaded=["build"], unloaded=[]))
    store.append(SkillChangeEntry(name="grow", action="edit", content_hash="abc"))
    store.append(CustomEntry(namespace="builder", data={"round": 1}))
    loaded = SessionStore.load(path)
    assert [e.type for e in loaded.path_to_leaf()] == [
        "session_info",
        "message",
        "tool_set_change",
        "skill_change",
        "custom",
    ]
    assert loaded.leaf_id == store.leaf_id
    assert loaded.get(store.entries[3].id).content_hash == "abc"


def test_branch_moves_the_leaf_and_the_old_branch_stays(tmp_path):
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)
    root = store.append(SessionInfoEntry())
    a = store.append_message(UserMessage(content="a"))
    b = store.append_message(AssistantMessage(content="b"))
    store.branch(a.id)
    c = store.append_message(AssistantMessage(content="c"))
    assert c.parent_id == a.id
    assert [e.id for e in store.active_path()] == [root.id, a.id, c.id]
    assert store.get(b.id) is not None  # nothing was deleted
    reloaded = SessionStore.load(path)
    assert [e.id for e in reloaded.active_path()] == [root.id, a.id, c.id]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["session_info", "message", "message", "leaf", "message"]
    assert lines[3]["entry_id"] == a.id


def test_compaction_replaces_entries_on_replay(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    root = store.append(SessionInfoEntry())
    e1 = store.append_message(UserMessage(content="one"))
    e2 = store.append_message(AssistantMessage(content="two"))
    e3 = store.append_message(UserMessage(content="three"))
    e4 = store.append_message(AssistantMessage(content="four"))
    compaction = store.append(
        CompactionEntry(summary="one and two, in short", replaces_entry_ids=[e1.id, e2.id], first_kept_entry_id=e3.id)
    )
    e5 = store.append_message(UserMessage(content="five"))
    active = store.active_path()
    assert [e.id for e in active] == [root.id, compaction.id, e3.id, e4.id, e5.id]
    assert [e.id for e in store.path_to_leaf()] == [root.id, e1.id, e2.id, e3.id, e4.id, compaction.id, e5.id]
    # replaced, not edited: the record still returns the original entry with its content
    assert isinstance(store.get(e1.id), MessageEntry) and store.get(e1.id).message.content == "one"
    messages = store.active_messages()
    assert messages[0].content.endswith("one and two, in short")
    assert [m.content for m in messages[1:]] == ["three", "four", "five"]
    reloaded = SessionStore.load(store.path)
    assert [e.id for e in reloaded.active_path()] == [e.id for e in active]


def test_compaction_by_first_kept_entry_replaces_everything_before_it(tmp_path):
    # The code fallback compacts tau's way: a first kept entry and no explicit list (D124).
    store = SessionStore(tmp_path / "s.jsonl")
    root = store.append(SessionInfoEntry())
    e1 = store.append_message(UserMessage(content="one"))
    e2 = store.append_message(AssistantMessage(content="two"))
    e3 = store.append_message(UserMessage(content="three"))
    compaction = store.append(CompactionEntry(summary="one and two", first_kept_entry_id=e3.id, by="code"))
    e4 = store.append_message(AssistantMessage(content="four"))
    assert [e.id for e in store.active_path()] == [root.id, compaction.id, e3.id, e4.id]
    assert store.get(e1.id) is not None and store.get(e2.id) is not None
    assert [m.content for m in store.active_messages()][1:] == ["three", "four"]


def test_session_info_is_never_compacted_away(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    root = store.append(SessionInfoEntry())
    e1 = store.append_message(UserMessage(content="one"))
    compaction = store.append(CompactionEntry(summary="all of it", replaces_entry_ids=[root.id, e1.id]))
    assert [e.id for e in store.active_path()] == [root.id, compaction.id]


def test_compaction_that_replaces_nothing_on_the_path_stays_where_it_was(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    store.append(SessionInfoEntry())
    a = store.append_message(UserMessage(content="a"))
    c = store.append(CompactionEntry(summary="of something off-path", replaces_entry_ids=["nope"]))
    assert [e.id for e in store.active_path()][1:] == [a.id, c.id]


def test_branching_to_an_unknown_entry_is_a_tree_error(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    store.append(SessionInfoEntry())
    with pytest.raises(SessionTreeError):
        store.branch("missing")


def test_appending_under_an_unknown_parent_is_a_tree_error(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    store.append(SessionInfoEntry())
    with pytest.raises(SessionTreeError):
        store.append(MessageEntry(message=UserMessage(content="x"), parent_id="missing"))


def test_appending_a_leaf_entry_directly_is_a_tree_error(tmp_path):
    store = SessionStore(tmp_path / "s.jsonl")
    store.append(SessionInfoEntry())
    with pytest.raises(SessionTreeError):
        store.append(LeafEntry(entry_id=store.leaf_id))


def test_loading_a_file_with_a_duplicate_entry_id_is_a_tree_error(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"type": "session_info", "id": "r", "parent_id": None, "timestamp": 0, "created_at": 0}) + "\n"
        + json.dumps({"type": "message", "id": "r", "parent_id": "r", "timestamp": 0, "message": {"role": "user", "content": "x"}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionTreeError):
        SessionStore.load(path)


def test_load_of_a_missing_file_starts_empty(tmp_path):
    store = SessionStore.load(tmp_path / "none.jsonl")
    assert store.entries == [] and store.leaf_id is None and store.active_path() == []
