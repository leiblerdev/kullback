"""The Claude Code JSONL adapter: one session transcript mapped to one Trace.

One JSON object per line. Lines of type user, assistant and system carrying a message in the
Anthropic Messages shape (role, content as a string or a list of text, tool_use {id, name, input}
and tool_result {tool_use_id, content, is_error} blocks) are turns; every other line (summary,
result and the bookkeeping types a transcript carries) is not a turn and is only counted. One
recording per sessionId; a line without one belongs to the session around it. Turns follow the
parentUuid chain, with file order as the fallback when the chain is broken, which the sidecar says.
A tool_use block is a call whose pointer is the line index and the block; the tool_result with the
same id answers it as a list of content blocks, an is_error result going through the error classes. The transcript carries no
system prompt and no tool list, so both stay None.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Iterator, Optional

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn, as_dict, content_hash

TURN_TYPES = ("user", "assistant", "system")


class ClaudeCodeJsonlDetector:
    """Votes on line-delimited transcript rows and maps one session to one Trace."""

    name = "claude_code_jsonl"
    maps = True
    display = "Claude Code JSONL"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, list):
            if jsonl and any(_looks_claude_code(item) for item in document[:20]
                             if isinstance(item, dict)):
                return (0.8, ["line-delimited rows carry transcript fields"])
            return (0.0, ["no transcript row marker, or the file is one JSON document, not lines"])
        return (0.0, ["not a record list, so not line-delimited rows"])

    def recordings(self, document: Any) -> Iterator[Any]:
        """One recording per sessionId in first-seen order, each line kept with its file index."""
        sessions: dict[str, list] = {}
        current: Optional[str] = None
        orphans: list = []
        for index, line in enumerate(document if isinstance(document, list) else []):
            session = line.get("sessionId") if isinstance(line, dict) else None
            if isinstance(session, str) and session:
                current = session
                sessions.setdefault(session, orphans)
                orphans = []
            if current is None:
                orphans.append((index, line))
            else:
                sessions[current].append((index, line))
        if orphans and not sessions:
            sessions[""] = orphans
        return iter({"id": session or None, "lines": lines} for session, lines in sessions.items())

    def environment(self, document: Any) -> dict:
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        """The non-turn line counts by type and whether the parentUuid chain held."""
        lines = recording.get("lines") or []
        counts = Counter(str(line.get("type")) if isinstance(line, dict) else "not_an_object"
                         for _, line in lines if not _is_turn(line))
        _, broken = _ordered(lines)
        return {"non_turn_lines": dict(sorted(counts.items())),
                "chain": "broken, file order used" if broken else "parentUuid order"}

    def to_trace(self, recording: dict, ctx: Any) -> Trace:
        from kullback.builder.ingest import classify_error, detect_truncation

        trace_id = str(recording.get("id") or f"{ctx.raw_hash[:12]}-{ctx.index}")
        ordered, _ = _ordered(recording.get("lines") or [])
        turns, calls, pending = [], [], {}
        for index, line in ordered:
            message = line["message"]
            role = message.get("role") if message.get("role") in TURN_TYPES else line["type"]
            blocks = message.get("content")
            here = RawPtr(file_hash=ctx.raw_hash, sim_index=index, section="message")
            if not isinstance(blocks, list):
                turns.append(Turn(idx=len(turns), role=role, content=_text(blocks), raw_ptr=here))
                continue
            texts, ids, answered = [], [], False
            for position, block in enumerate(blocks):
                kind = block.get("type")
                at = RawPtr(file_hash=ctx.raw_hash, sim_index=index,
                            section=f"message.content[{position}]")
                if kind == "text":
                    texts.append(block.get("text") or "")
                elif kind == "tool_use":
                    call = ToolCall(id=block.get("id"), name=block.get("name") or "",
                                    args=block.get("input") if isinstance(block.get("input"), dict)
                                    else {}, requestor=role, raw_ptr=at, trace_id=trace_id)
                    calls.append(call)
                    if call.id:
                        pending[call.id] = (call, line.get("timestamp"))
                        ids.append(call.id)
                elif kind == "tool_result":
                    answered = True
                    ids.append(block.get("tool_use_id"))
                    waiting = pending.pop(block.get("tool_use_id"), None)
                    if waiting is not None:
                        _attach_result(waiting[0], block, at, waiting[1], line.get("timestamp"),
                                       classify_error, detect_truncation)
            turns.append(Turn(idx=len(turns), role="tool" if answered and not texts else role,
                              content="\n".join(texts) if texts else None,
                              tool_call_ids=[i for i in ids if i], raw_ptr=here))
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=turns,
            tool_calls=calls,
            tools_declared=None,
            system_prompt=None,
            tools_declared_ptr=None,
            system_prompt_ptr=None,
            info_ptr=None,
            # The ruling finds a Trace's recording by this index, so it is the recording's own.
            raw_ptr=RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index),
        )


def _attach_result(call: ToolCall, block: dict, where: RawPtr, asked: Any, answered: Any,
                   classify_error: Any, detect_truncation: Any) -> None:
    """A tool result is text, never a JSON string: a string content is the Messages API shorthand
    for one text block, so the result is stored as that block list, and truncation reads the joined
    text for cut markers only (text that opens with a bracket is not a cut JSON body)."""
    from kullback.builder.ingest import UNPARSED_JSON_MARKER

    content = block.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    text = "\n".join(item.get("text") or "" for item in blocks or []
                     if isinstance(item, dict) and item.get("type") == "text")
    cut = detect_truncation(text)
    if cut[2] != UNPARSED_JSON_MARKER:
        call.truncated, call.visible_len, call.cut_marker = cut
    if block.get("is_error") is True:
        call.error = classify_error(content, content if isinstance(content, dict) else None,
                                    ptr=where)
    else:
        call.result = blocks
    call.has_result = True
    call.resolved = True
    call.result_ptr = where
    call.latency_ms = _latency_ms(asked, answered)


def _is_turn(line: Any) -> bool:
    return (isinstance(line, dict) and line.get("type") in TURN_TYPES
            and isinstance(line.get("message"), dict))


def _ordered(lines: list) -> tuple[list, bool]:
    """Turn lines in parentUuid order, depth first from each root in file order.

    Non-turn lines sit in the chain too, so the walk crosses them and keeps only turns. The chain
    is broken when a parent is named but absent, or when the walk misses a line (a cycle); then
    file order stands and the second value says so."""
    by_uuid = {line.get("uuid"): index for index, line in lines
               if isinstance(line, dict) and line.get("uuid")}
    children: dict = {}
    roots = []
    broken = False
    for index, line in lines:
        parent = line.get("parentUuid") if isinstance(line, dict) else None
        if parent and parent in by_uuid:
            children.setdefault(parent, []).append((index, line))
        else:
            broken = broken or bool(parent)
            roots.append((index, line))
    walked, stack = [], list(reversed(roots))
    while stack:
        index, line = stack.pop()
        walked.append((index, line))
        uuid = line.get("uuid") if isinstance(line, dict) else None
        stack.extend(reversed(children.get(uuid, []) if uuid else []))
    if broken or len(walked) != len(lines):
        return ([pair for pair in lines if _is_turn(pair[1])], True)
    return ([pair for pair in walked if _is_turn(pair[1])], False)


def _latency_ms(asked: Any, answered: Any) -> Optional[float]:
    try:
        start = datetime.fromisoformat(str(asked).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(answered).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (end - start).total_seconds() * 1000.0


def _text(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _looks_claude_code(item: dict) -> bool:
    if item.get("type") not in ("user", "assistant", "system", "summary", "result"):
        return False
    return any(key in item for key in ("message", "content", "uuid", "sessionId"))


def trace_hash(trace: Trace) -> str:
    """Content hash of a Trace with its own hash field blanked, stable across runs."""
    body = as_dict(trace)
    body["hash"] = ""
    return content_hash(body)
