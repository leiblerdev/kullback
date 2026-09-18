"""Outline-first reading helpers shared by the harness's agents (G21).

A read used to hand over up to 60,000 characters and cut the rest with no way back. The pattern
instead: read the map, then pull one small piece of evidence at a time and know where the next
piece is. `outline` is the map, `page` is the paging, `locate` finds phrases, `around` shows the
neighbours, and `part` pulls one addressed piece.

The Builder, the Examiner and the user agent all share this module, so it imports nothing from any
of them, only the standard library. A record is the JSON-ready dict of a Run or a Trace, the way
`as_dict` renders it. Every locator this module answers (`turn:2`, `call:1`, `event:3`, `header`)
names one part of that record, and the Examiner's `read` takes it back.
"""

from __future__ import annotations

import json
from typing import Any

PAGE_CHARS = 4000  # one page of evidence, far below the old 60,000 character cut
OUTLINE_PARTS = 50  # how many part skeletons one outline lists; the locator scheme covers the rest
OUTLINE_TOOLS = 50  # how many distinct tool names one outline lists; the rest fold into "other"
SNIPPET_CHARS = 120  # how much of a part one locate match shows
LOCATE_MAX = 100  # how many matches locate answers before it says it stopped


def _dump(body: Any) -> str:
    """One record or part as text, the way the Examiner's `read` renders it."""
    return json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def kind_of(record: dict) -> str:
    """Which record this is: a Run holds events, a Trace holds turns and tool calls."""
    if not isinstance(record, dict):
        raise ValueError("a record is a dict, the JSON of a Run or a Trace")
    if isinstance(record.get("events"), list):
        return "run"
    if isinstance(record.get("turns"), list) or isinstance(record.get("tool_calls"), list):
        return "trace"
    raise ValueError("not a Run or a Trace record: neither holds an events, turns or tool_calls list")


def _dicts(rows: Any) -> list[dict]:
    """The rows that are dicts, in order: an outline never counts a ragged row as a part."""
    return [row for row in rows or [] if isinstance(row, dict)]


def _cap(names: dict[str, int]) -> dict[str, int]:
    """Tool counts that fit an outline: the most frequent names, the rest folded into "other"."""
    if len(names) <= OUTLINE_TOOLS:
        return dict(sorted(names.items()))
    ranked = sorted(names.items(), key=lambda pair: (-pair[1], pair[0]))
    kept = dict(sorted(ranked[: OUTLINE_TOOLS - 1]))
    kept["other"] = sum(count for _, count in ranked[OUTLINE_TOOLS - 1 :])
    return kept


def _named(rows: list[dict], field: str) -> dict[str, int]:
    """How many rows carry each name: tool calls by name with counts."""
    counts: dict[str, int] = {}
    for row in rows:
        name = row.get(field)
        if name is not None:
            counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _turn_skeleton(position: int, turn: dict) -> dict:
    """One turn as an outline lists it: where it is, who spoke, how big, never what was said."""
    return {"locator": f"turn:{position}", "role": turn.get("role"), "size": len(_dump(turn))}


def _call_skeleton(position: int, call: dict) -> dict:
    """One recorded tool call as an outline lists it: where, which tool, how big, did it error."""
    return {"locator": f"call:{position}", "name": call.get("name"),
            "size": len(_dump(call)), "error": bool(call.get("error"))}


def _payload(event: dict) -> dict:
    """The dict payload of one Run event, or no rows when the payload is not a dict."""
    return event.get("payload") if isinstance(event.get("payload"), dict) else {}


def _event_errored(event: dict) -> bool:
    """Whether one Run event errored: its own error type, or an error its payload carries."""
    return event.get("type") == "error" or bool(_payload(event).get("error"))


def _event_name(event: dict) -> Any:
    """The tool one Run event belongs to, or None when it belongs to none."""
    name = _payload(event).get("name")
    return name if isinstance(name, str) else None


def _event_skeleton(position: int, event: dict) -> dict:
    """One Run event as an outline lists it: where, which kind, which tool, how big, did it error."""
    row: dict[str, Any] = {"locator": f"event:{position}", "type": event.get("type"),
                           "size": len(_dump(event)), "error": _event_errored(event)}
    if _event_name(event) is not None:
        row["name"] = _event_name(event)
    return row


def _of_type(events: list[dict], kind: str) -> list[dict]:
    """The events of one type, in order."""
    return [event for event in events if event.get("type") == kind]


def _calls_by_tool(events: list[dict]) -> dict[str, int]:
    """Tool calls by name with counts: the tool an event belongs to lives in its payload."""
    counts: dict[str, int] = {}
    for event in _of_type(events, "tool_call"):
        name = _event_name(event)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _trace_outline(record: dict) -> dict:
    """The map of a Trace: counts, calls by name, which calls errored, one skeleton per part."""
    turns = _dicts(record.get("turns"))
    calls = _dicts(record.get("tool_calls"))
    return {
        "kind": "trace",
        "outline": True,
        "trace_id": record.get("trace_id"),
        "turn_count": len(turns),
        "turns": [_turn_skeleton(pos, turn) for pos, turn in enumerate(turns[:OUTLINE_PARTS])],
        "turns_total": len(turns),
        "call_count": len(calls),
        "calls": [_call_skeleton(pos, call) for pos, call in enumerate(calls[:OUTLINE_PARTS])],
        "calls_total": len(calls),
        "by_tool": _cap(_named(calls, "name")),
        "errored": [f"call:{pos}" for pos, call in enumerate(calls) if call.get("error")],
        "end": None,
        "total_chars": len(_dump(record)),
        "note": ("a Trace in outline; read one part with its locator (`turn:<n>` to "
                 f"{len(turns) - 1}, `call:<n>` to {len(calls) - 1}); locators stay stable"),
    }


def _run_outline(record: dict) -> dict:
    """The map of a Run: counts, calls by name, which events errored, one skeleton per event."""
    events = _dicts(record.get("events"))
    header = {key: value for key, value in record.items() if key != "events"}
    shown = [_event_skeleton(pos, event) for pos, event in enumerate(events[:OUTLINE_PARTS])]
    return {
        "kind": "run",
        "outline": True,
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "model": record.get("model"),
        "turn_count": len(_of_type(events, "user_turn")),
        "call_count": len(_of_type(events, "tool_call")),
        "events": shown,
        "events_total": len(events),
        "by_tool": _cap(_calls_by_tool(events)),
        "errored": [row["locator"] for row in shown if row["error"]]
        + [f"event:{pos}" for pos in range(OUTLINE_PARTS, len(events))
           if _event_errored(events[pos])],
        "end": record.get("termination_reason"),
        "total_chars": len(_dump(record)),
        "header": {"locator": "header", "size": len(_dump(header))},
        "note": ("a Run in outline; read one part with its locator (`event:<n>` to "
                 f"{len(events) - 1}, `header` for the Run fields); locators stay stable"),
    }


def outline(record: dict) -> dict:
    """The map of a Run or a Trace, bounded whatever the record's length.

    Turn count, call count, tool calls by name with counts, which calls errored, the end kind,
    the size in characters of the record and of each part, and a stable locator for every part.
    Only the first `OUTLINE_PARTS` skeletons are listed; the totals and the locator scheme
    (`turn:<n>`, `call:<n>`, `event:<n>`, `header`) address the rest, so the outline stays small
    while no part is unreachable.
    """
    if kind_of(record) == "run":
        return _run_outline(record)
    return _trace_outline(record)


def page(text: str, offset: int = 0, limit: int = PAGE_CHARS) -> dict:
    """One page of a text: the slice, the next offset or None, and the total length.

    Offsets walk forward only: an offset past the end answers an empty slice with no next page,
    so paging a record end to end loses no character and repeats none.
    """
    if limit < 1:
        raise ValueError(f"a page holds at least one character, not {limit}")
    at = max(0, offset)
    total = len(text)
    if at >= total:
        return {"text": "", "offset": total, "next": None, "total": total}
    end = at + limit
    return {"text": text[at:end], "offset": at, "next": end if end < total else None, "total": total}


def _snippet(text: str, start: int, end: int) -> str:
    """The match with `SNIPPET_CHARS` characters of its part around it, the match inside."""
    left = max(0, start - (SNIPPET_CHARS - (end - start)) // 2)
    cut = text[max(0, left):left + SNIPPET_CHARS]
    return ("..." if left > 0 else "") + cut + ("..." if left + SNIPPET_CHARS < len(text) else "")


def _located(prefix: str, rows: list[dict]) -> list[tuple[str, dict]]:
    """Every part of one list with its locator, in order."""
    return [(f"{prefix}:{pos}", row) for pos, row in enumerate(rows)]


def _part_bodies(record: dict) -> list[tuple[str, dict]]:
    """Every addressable part of the record in locator order, with its locator."""
    if kind_of(record) == "run":
        return _located("event", _dicts(record.get("events")))
    return _located("turn", _dicts(record.get("turns"))) + _located("call", _dicts(record.get("tool_calls")))


def _occurrences(text: str, needle: str, length: int) -> list[dict]:
    """Every occurrence of one phrase in one part's text, with ranges into that text."""
    lowered = text.casefold()
    out: list[dict] = []
    start = lowered.find(needle)
    while start >= 0:
        out.append({"start": start, "end": start + length,
                    "snippet": _snippet(text, start, start + length)})
        start = lowered.find(needle, start + 1)
    return out


def locate(record: dict, phrase: str) -> dict:
    """Where one phrase appears in a record: locators with character ranges and a short snippet.

    The ranges are offsets into the addressed part's own text, the way `read` renders it. At most
    `LOCATE_MAX` matches are answered; `total` says how many there are and `truncated` whether the
    list stopped early.
    """
    if not phrase.strip():
        raise ValueError("locate needs a phrase to look for")
    needle = phrase.casefold()
    found: list[dict] = []
    for locator, body in _part_bodies(record):
        for hit in _occurrences(_dump(body), needle, len(phrase)):
            if len(found) < LOCATE_MAX:
                found.append({"locator": locator, **hit})
    total = sum(len(_occurrences(_dump(body), needle, len(phrase))) for _, body in _part_bodies(record))
    return {"total": total, "matches": found, "truncated": total > len(found)}


def _rows_for(kind: str, record: dict, name: str) -> Any:
    """The part list one locator names: turns, calls or events, else None."""
    if name == "turn" and kind == "trace":
        return _dicts(record.get("turns"))
    if name == "call" and kind == "trace":
        return _dicts(record.get("tool_calls"))
    if name == "event" and kind == "run":
        return _dicts(record.get("events"))
    return None


def _valid_locators(kind: str) -> str:
    """What a locator may name on this kind of record, for a refusal to quote."""
    return "header, event:<n>" if kind == "run" else "turn:<n>, call:<n>"


def part(record: dict, locator: str) -> dict:
    """The one part a locator names, as JSON-ready data: a turn, a call, an event, or a header."""
    kind = kind_of(record)
    text = (locator or "").strip()
    if kind == "run" and text == "header":
        return {key: value for key, value in record.items() if key != "events"}
    name, sep, index = text.partition(":")
    rows = _rows_for(kind, record, name) if sep else None
    if rows is None or not index.isdigit() or int(index) >= len(rows):
        raise ValueError(f"{locator!r} names no part of this {kind}: read one of {_valid_locators(kind)}")
    return rows[int(index)]


def _skeleton_of(locator: str, body: dict) -> dict:
    """One neighbouring part the way an outline lists it: shape and size, never payloads."""
    name, _, index = locator.partition(":")
    position = int(index) if index.isdigit() else 0
    if name == "turn":
        return _turn_skeleton(position, body)
    if name == "call":
        return _call_skeleton(position, body)
    if name == "event":
        return _event_skeleton(position, body)
    return {"locator": "header", "size": len(_dump(body))}


def _neighbours(bodies: list[tuple[str, dict]], locator: str, n: int) -> list[dict]:
    """The skeletons beside one locator in its own list, without any payload."""
    at = next((pos for pos, (name, _) in enumerate(bodies) if name == locator), len(bodies))
    family = locator.split(":")[0]
    near = [pos for pos, (name, _) in enumerate(bodies)
            if name.split(":")[0] == family and abs(pos - at) <= n]
    return [_skeleton_of(name, body) for pos, (name, body) in enumerate(bodies) if pos in near]


def around(record: dict, locator: str, n: int = 1) -> dict:
    """The neighbouring parts of one locator as skeletons without payloads.

    The neighbours are the parts beside it in its own list (turns beside a turn, calls beside a
    call, events beside an event), so an agent sees what came before and after without reading
    any payload twice.
    """
    if n < 0:
        raise ValueError(f"around takes how many neighbours to each side, not {n}")
    kind = kind_of(record)
    text = (locator or "").strip()
    if kind == "run" and text == "header":
        return {"locator": "header", "parts": [_skeleton_of("header", part(record, "header"))]}
    bodies = _part_bodies(record)
    if all(name != text for name, _ in bodies):
        raise ValueError(f"{locator!r} names no part of this {kind}")
    return {"locator": text, "parts": _neighbours(bodies, text, n)}
