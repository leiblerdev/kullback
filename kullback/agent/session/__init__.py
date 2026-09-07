"""The session: an append-only JSONL tree of everything a harness did, and the active path through it.

Every line is an entry with an id, a parent id and a timestamp. Appending never rewrites a line;
branching moves the leaf pointer to an earlier entry, so the next append hangs off it and the old
branch stays on disk. The active context is the root-to-leaf replay with compaction entries
applied: the entries a compaction replaces drop out and its summary stands where they were, and the
entries it cut (`ContentCut`) keep their place with shorter content. Nothing is deleted, which is
what makes a wrong forget cost one recall (D124); phase 7 builds the context tools on this store.
"""

from __future__ import annotations

from kullback.agent.session.entries import (
    CompactionEntry,
    ContentCut,
    CustomEntry,
    LeafEntry,
    MessageEntry,
    SessionEntry,
    SessionInfoEntry,
    SkillChangeEntry,
    ToolSetChangeEntry,
    new_entry_id,
    now,
)
from kullback.agent.session.store import SessionStore, SessionTreeError, cuts_in

__all__ = [
    "CompactionEntry",
    "ContentCut",
    "CustomEntry",
    "LeafEntry",
    "MessageEntry",
    "SessionEntry",
    "SessionInfoEntry",
    "SessionStore",
    "SessionTreeError",
    "SkillChangeEntry",
    "ToolSetChangeEntry",
    "cuts_in",
    "new_entry_id",
    "now",
]
