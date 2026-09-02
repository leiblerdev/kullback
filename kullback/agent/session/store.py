"""The JSONL store: load, append, branch, and the active path with compactions applied."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import TypeAdapter

from kullback.agent.messages import Message, UserMessage
from kullback.agent.session.entries import CompactionEntry, LeafEntry, MessageEntry, SessionEntry, SessionInfoEntry

_ENTRY = TypeAdapter(SessionEntry)


class SessionTreeError(ValueError):
    """The entries do not form a tree the store can walk."""


class SessionStore:
    """One session file. Entries are kept in memory in file order and appended to disk one line each."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[SessionEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self.leaf_id: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path) -> "SessionStore":
        """Read the file, or start empty when it does not exist yet."""
        store = cls(path)
        if not store.path.is_file():
            return store
        for line in store.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            store._take(_ENTRY.validate_json(line))
        return store

    def _take(self, entry: SessionEntry) -> None:
        if isinstance(entry, LeafEntry):
            if entry.entry_id not in self._by_id:
                raise SessionTreeError(f"leaf points at a missing entry: {entry.entry_id}")
            self.leaf_id = entry.entry_id
            return
        if entry.id in self._by_id:
            raise SessionTreeError(f"duplicate entry id: {entry.id}")
        if entry.parent_id is not None and entry.parent_id not in self._by_id:
            raise SessionTreeError(f"entry {entry.id} names a missing parent: {entry.parent_id}")
        self.entries.append(entry)
        self._by_id[entry.id] = entry
        self.leaf_id = entry.id

    def _write(self, entry: SessionEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")

    def append(self, entry: SessionEntry) -> SessionEntry:
        """Append under the current leaf (unless the entry names its parent) and make it the leaf."""
        if isinstance(entry, LeafEntry):
            raise SessionTreeError("use branch() to move the leaf")
        if entry.parent_id is None and self.leaf_id is not None:
            entry.parent_id = self.leaf_id
        self._take(entry)
        self._write(entry)
        return entry

    def append_message(self, message: Message) -> MessageEntry:
        return self.append(MessageEntry(message=message))  # type: ignore[return-value]

    def branch(self, entry_id: str) -> None:
        """Move the leaf to an earlier entry; the next append hangs off it and the old branch stays."""
        if entry_id not in self._by_id:
            raise SessionTreeError(f"cannot branch to a missing entry: {entry_id}")
        leaf = LeafEntry(entry_id=entry_id)
        self._take(leaf)
        self._write(leaf)

    def get(self, entry_id: str) -> Optional[SessionEntry]:
        return self._by_id.get(entry_id)

    def path_to_leaf(self) -> list[SessionEntry]:
        """The root-to-leaf path as recorded, compactions not applied."""
        path: list[SessionEntry] = []
        seen: set[str] = set()
        current = self.leaf_id
        while current is not None:
            if current in seen:
                raise SessionTreeError(f"cycle at entry {current}")
            seen.add(current)
            entry = self._by_id.get(current)
            if entry is None:
                raise SessionTreeError(f"missing entry on the path: {current}")
            path.append(entry)
            current = entry.parent_id
        path.reverse()
        return path

    def active_path(self) -> list[SessionEntry]:
        """The root-to-leaf path with every compaction applied.

        The entries a compaction replaces drop out and the compaction stands where the first of
        them stood, so the summary reads at the point in the history it summarizes. What it
        replaces is `replaces_entry_ids` (the model's forget, D124) plus, when `first_kept_entry_id`
        names an entry on the path, everything before that entry (the code fallback, which compacts
        tau's way by a prefix). The root `session_info` is never replaced, and a compaction that
        replaces nothing on the path stays where it was appended.
        """
        path = self.path_to_leaf()
        for compaction in [e for e in path if isinstance(e, CompactionEntry)]:
            replaced = set(compaction.replaces_entry_ids)
            if compaction.first_kept_entry_id is not None:
                kept_from = next((i for i, e in enumerate(path) if e.id == compaction.first_kept_entry_id), None)
                if kept_from is not None:
                    replaced.update(e.id for e in path[:kept_from])
            replaced.discard(compaction.id)
            replaced.difference_update(e.id for e in path if isinstance(e, SessionInfoEntry))
            if not replaced:
                continue
            first = next((i for i, e in enumerate(path) if e.id in replaced), None)
            if first is None:
                continue
            kept = [e for e in path if e.id not in replaced and e.id != compaction.id]
            # first is an index into the old path; count how many kept entries precede it.
            before = sum(1 for e in path[:first] if e.id not in replaced and e.id != compaction.id)
            kept.insert(before, compaction)
            path = kept
        return path

    def active_messages(self) -> list[Message]:
        """The messages the active path puts in front of the model, a compaction as a marked user message."""
        messages: list[Message] = []
        for entry in self.active_path():
            if isinstance(entry, MessageEntry):
                messages.append(entry.message)
            elif isinstance(entry, CompactionEntry):
                messages.append(UserMessage(content=f"[summary of earlier context]\n{entry.summary}"))
        return messages
