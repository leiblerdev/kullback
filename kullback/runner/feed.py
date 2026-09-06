"""The live feed of one build: what it is doing right now, in the order it did it.

A build's records are its state, not its story. budget.json says a build has made 1041 calls and
gates.json says which gates hold, but neither says what happened in the last ten seconds, so a
screen watching a running build from another directory could only show a board that barely moved
while the build worked. This file is the story: one JSON line per event, appended as it happens,
read by anyone who wants to follow along.

Three rules hold it together.

It is append-only and one line per event, so a reader that has read N bytes reads from N and never
re-reads; a half-written last line is a line the reader skips and picks up on its next pass.

It is a courtesy, never a step of the build. Every write is wrapped and every failure is dropped:
a full disk stops no build, and a build that cannot tell anyone what it is doing still does it.

It says only what the build already knows. The stage, the model, the tokens and the cost on a
model_call are the same numbers budget.py writes to the ledger, so the feed and the ledger cannot
disagree about what a build spent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

EVENTS_FILE = "events.jsonl"

# Where the provider layer keeps one file per distinct reply. Not a feed, but the next best
# thing when there is no feed: see derived_since.
CACHE_DIR = "model_cache"

# One line is capped so a runaway tool result cannot turn the feed into the largest file in the
# workdir. The cap is generous: it is here to bound the worst case, not to trim ordinary events.
MAX_LINE = 4000


def path_for(workdir: Any) -> Path:
    return Path(workdir) / EVENTS_FILE


def append(workdir: Any, kind: str, **fields: Any) -> None:
    """Add one event to this build's feed. Never raises: see the third rule above."""
    try:
        row = {"at": time.time(), "kind": kind, **fields}
        line = json.dumps(row, default=str)
        if len(line) > MAX_LINE:
            row = {"at": row["at"], "kind": kind, "note": "event too long to record in full"}
            line = json.dumps(row)
        path = path_for(workdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def read_since(workdir: Any, offset: int = 0) -> tuple[list[dict], int]:
    """Every whole event written since byte `offset`, and where to read from next.

    A trailing partial line is left for the next call: its bytes are not counted as read, so the
    line is delivered once, whole, when the writer has finished it."""
    path = path_for(workdir)
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    # A workdir that was rebuilt from scratch has a shorter feed than the reader's offset; start
    # again rather than seeking past the end and reporting silence forever.
    if size < offset:
        offset = 0
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            body = handle.read()
    except OSError:
        return [], offset
    consumed = offset
    for line in body.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        consumed += len(line.encode("utf-8"))
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, consumed


def describe(row: dict) -> str:
    """One event as one line a person reads, in the words the build itself uses.

    Kept here beside the writer so the feed's shapes and their wording change together."""
    kind = str(row.get("kind") or "?")
    if kind == "model_call":
        usage = f"{_int(row, 'input'):,} in / {_int(row, 'output'):,} out"
        cached = _int(row, "cache_read")
        if cached:
            usage += f" / {cached:,} cached"
        parts = [str(row.get("stage") or "call"), str(row.get("model") or "?"), usage]
        # A cost of zero and an unknown cost are different things, and a derived row knows the
        # tokens but not the dollars, so an absent number is left out rather than printed as 0.
        if row.get("usd") is not None:
            parts.append(f"${float(row['usd']):,.4f}")
        if row.get("wall_ms") is not None:
            parts.append(f"{float(row['wall_ms']) / 1000:,.1f}s")
        return "  ".join(parts)
    if kind in ("round", "beat", "stage"):
        parts = [str(row.get(key)) for key in ("stage", "agent", "round", "state") if row.get(key) is not None]
        return f"{kind} " + " ".join(parts)
    if kind == "message":
        return f"message {row.get('state') or ''} {row.get('role') or ''}".rstrip()
    if kind == "tool":
        return f"tool {row.get('state') or ''} {row.get('name') or ''}".rstrip()
    return kind + (f" {row.get('note')}" if row.get("note") else "")


def _int(row: dict, key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def derived_since(workdir: Any, cursor: float = 0.0,
                  limit: Optional[int] = None) -> tuple[list[dict], float]:
    """A build's story read off its reply cache, for a build that writes no feed of its own.

    Two builds cannot tell you what they are doing through the feed: one started before this file
    existed, and one whose workdir was copied without it. Both still write one JSON file per
    distinct model call into the reply cache, and that file is nearly a feed line already: its
    mtime is when the reply landed and its body holds the model and the tokens. What it does not
    hold is the stage or the cost, so neither is invented.

    A call the cache answers writes nothing, so this shows new work only and never replays old
    work as if it just happened. `cursor` is the mtime read up to, not a byte offset.
    """
    try:
        entries = [(entry.stat().st_mtime, entry)
                   for entry in (Path(workdir) / CACHE_DIR).glob("*.json")]
    except OSError:
        return [], cursor
    fresh = sorted((pair for pair in entries if pair[0] > cursor), key=lambda pair: pair[0])
    # Opening a watch on a build with a thousand cached calls should show the last few, not all
    # of them; every one it skips is older than the ones it keeps, so the cursor still moves past.
    if limit is not None:
        fresh = fresh[-limit:]
    rows = [_cached_call_row(mtime, entry) for mtime, entry in fresh]
    return rows, max([cursor, *(mtime for mtime, _ in fresh)])


def _cached_call_row(mtime: float, entry: Path) -> dict:
    row: dict[str, Any] = {"at": mtime, "kind": "model_call"}
    try:
        body = json.loads(entry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return row
    usage = body.get("usage") if isinstance(body, dict) else None
    row["model"] = body.get("model") if isinstance(body, dict) else None
    for field in ("input", "output", "cache_read", "cache_write"):
        row[field] = (usage or {}).get(field) or 0
    return row


def event_row(event: Any) -> Optional[dict]:
    """A typed event of the agent core as a feed row, or None when it is not for the feed.

    The message and tool events are tau's and only exist in the agent arm, where a model drives
    the session; the round, beat and stage events are ours and are emitted in both arms. Partial
    message updates are deliberately absent: one row per token would be a feed nobody can read
    and a file nobody wants on disk."""
    kind = getattr(event, "type", None)
    if kind in ("stage_start", "stage_end"):
        return {"kind": "stage", "stage": getattr(event, "name", None),
                "state": "start" if kind == "stage_start" else "end"}
    if kind in ("round_start", "round_end"):
        return {"kind": "round", "round": getattr(event, "round", None),
                "state": "start" if kind == "round_start" else "end",
                "exit": getattr(event, "exit", None)}
    if kind in ("beat_start", "beat_end"):
        return {"kind": "beat", "agent": getattr(event, "agent", None),
                "round": getattr(event, "round", None),
                "state": "start" if kind == "beat_start" else "end"}
    if kind in ("message_start", "message_end"):
        message = getattr(event, "message", None)
        return {"kind": "message", "state": "start" if kind == "message_start" else "end",
                "role": getattr(message, "role", None)}
    if kind in ("tool_execution_start", "tool_execution_end"):
        return {"kind": "tool", "state": "start" if kind == "tool_execution_start" else "end",
                "name": getattr(event, "name", None) or getattr(event, "tool", None)}
    if kind == "error":
        return {"kind": "error", "note": str(getattr(event, "message", "") or event)[:400]}
    return None


def from_event(workdir: Any, event: Any) -> None:
    """Append a typed agent-core event, if it is one the feed carries."""
    row = event_row(event)
    if row is not None:
        append(workdir, str(row.pop("kind")), **row)


def pid_of(workdir: Any) -> Optional[int]:
    """The pid recorded on this feed's first line, when it has one."""
    rows, _ = read_since(workdir, 0)
    for row in rows:
        if row.get("pid"):
            try:
                return int(row["pid"])
            except (TypeError, ValueError):
                return None
    return None


def start(workdir: Any, **fields: Any) -> None:
    """Open a build's feed. Truncates, because a feed is one build's story and a workdir that is
    built again is a new one; the ledger keeps the history."""
    try:
        path = path_for(workdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    except OSError:
        return
    append(workdir, "build_start", pid=os.getpid(), **fields)
