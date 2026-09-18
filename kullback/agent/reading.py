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
from typing import Any, Optional

CONTINUE = ("Keep reading with a range (`turn:0-9`, `call:0-4`, `event:0-29`), the conversation "
            "(`turns`), or everything (`all`); each answers in pages, follow `next_offset` to null.")
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
        "errored": [f"call:{pos}" for pos, call in enumerate(calls) if call.get("error")][:OUTLINE_PARTS],
        "errored_total": sum(1 for call in calls if call.get("error")),
        "end": None,
        "total_chars": len(_dump(record)),
        "note": ("a Trace in outline; read one part with its locator (`turn:<n>` to "
                 f"{len(turns) - 1}, `call:<n>` to {len(calls) - 1}); locators stay stable. "
                 + CONTINUE),
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
        "errored": [f"event:{pos}" for pos, event in enumerate(events)
                    if _event_errored(event)][:OUTLINE_PARTS],
        "errored_total": sum(1 for event in events if _event_errored(event)),
        "end": record.get("termination_reason"),
        "total_chars": len(_dump(record)),
        "header": {"locator": "header", "size": len(_dump(header))},
        "note": ("a Run in outline; read one part with its locator (`event:<n>` to "
                 f"{len(events) - 1}, `header` for the Run fields); locators stay stable. "
                 + CONTINUE),
    }


def outline(record: dict) -> dict:
    """The map of a Run or a Trace, bounded whatever the record's length.

    Turn count, call count, tool calls by name with counts, which calls errored (capped, with
    a total), the end kind, the size in characters of the record and of each part, and a stable
    locator for every part. Only the first `OUTLINE_PARTS` skeletons are listed; the totals and
    the locator scheme (`turn:<n>`, `call:<n>`, `event:<n>`, `header`) address the rest, so the
    outline stays small while no part is unreachable.
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


def _folded(text: str) -> tuple[str, list[int]]:
    """The text case-folded for matching, with each folded character's original offset."""
    folded: list[str] = []
    origins: list[int] = []
    for pos, char in enumerate(text):
        for piece in char.casefold():
            folded.append(piece)
            origins.append(pos)
    return "".join(folded), origins


def _occurrences(text: str, phrase: str) -> list[dict]:
    """Every occurrence of one phrase in one part's text, with ranges into that text.

    Matching folds case, and the fold can change lengths (ß folds to ss), so the ranges are
    mapped back to the original offsets: they always slice the text `read` renders."""
    folded, origins = _folded(text)
    needle = phrase.casefold()
    out: list[dict] = []
    start = folded.find(needle)
    while start >= 0:
        first, last = origins[start], origins[start + len(needle) - 1] + 1
        out.append({"start": first, "end": last, "snippet": _snippet(text, first, last)})
        start = folded.find(needle, start + 1)
    return out


def locate(record: dict, phrase: str) -> dict:
    """Where one phrase appears in a record: locators with character ranges and a short snippet.

    The ranges are offsets into the addressed part's own text, the way `read` renders it. At most
    `LOCATE_MAX` matches are answered; `total` says how many there are and `truncated` whether the
    list stopped early.
    """
    if not phrase.strip():
        raise ValueError("locate needs a phrase to look for")
    found: list[dict] = []
    total = 0
    for locator, body in _part_bodies(record):
        hits = _occurrences(_dump(body), phrase)
        total += len(hits)
        for hit in hits:
            if len(found) < LOCATE_MAX:
                found.append({"locator": locator, **hit})
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


def _single(rows: Any, index: str) -> Any:
    """The indexed row, or None when the index is no row of the list."""
    if rows is None or not index.isdigit():
        return None
    at = int(index)
    return rows[at] if at < len(rows) else None


def part(record: dict, locator: str) -> dict:
    """The one part a locator names, as JSON-ready data: a turn, a call, an event, or a header."""
    kind = kind_of(record)
    text = (locator or "").strip()
    if kind == "run" and text == "header":
        return {key: value for key, value in record.items() if key != "events"}
    name, sep, index = text.partition(":")
    rows = _rows_for(kind, record, name) if sep and "-" not in index else None
    row = _single(rows, index)
    if row is None:
        raise ValueError(f"{locator!r} names no part of this {kind}: read one of {_valid_locators(kind)}")
    return row


def _range(record: dict, name: str, index: str) -> dict:
    """The parts one range names, in order: `turn:3-9`, `call:0-4`, `event:10-30`."""
    kind = kind_of(record)
    rows = _rows_for(kind, record, name)
    bounds = index.split("-")
    if rows is None or len(bounds) != 2 or not all(bit.isdigit() for bit in bounds):
        raise ValueError(f"{(name + ':' + index)!r} is no range of this {kind}: "
                         f"read one of {_valid_locators(kind)}")
    low, high = int(bounds[0]), int(bounds[1])
    span = f"{name}:0-{len(rows) - 1}" if rows else f"no {name}s"
    if low > high:
        raise ValueError(f"{(name + ':' + index)!r} runs backward: this {kind} holds {span}")
    if high >= len(rows):
        raise ValueError(f"{(name + ':' + index)!r} runs past the end: this {kind} holds {span}")
    return {"locator": f"{name}:{low}-{high}", "parts": rows[low:high + 1]}


def _spoken(event: dict) -> Optional[tuple[str, str]]:
    """One Run event as conversation, or None when the event says nothing out loud.

    A user turn speaks its content; a model call speaks its reply's content; tool calls and
    their results are evidence, not conversation, and never enter the `turns` view."""
    payload = _payload(event)
    if event.get("type") == "user_turn" and isinstance(payload.get("content"), str):
        return "user", payload["content"]
    reply = payload.get("reply")
    if event.get("type") == "model_call" and isinstance(reply, dict) \
            and isinstance(reply.get("content"), str):
        return "assistant", reply["content"]
    return None


def _conversation(record: dict) -> dict:
    """Every spoken turn in order, no tool payloads: the `turns` whole view."""
    if kind_of(record) == "run":
        events = _dicts(record.get("events"))
        turns = [{"locator": f"event:{pos}", "speaker": speaker, "text": text}
                 for pos, event in enumerate(events) if _spoken(event) is not None
                 for speaker, text in [_spoken(event)]]
        return {"locator": "turns", "turns": turns}
    turns = [{"locator": f"turn:{pos}", "speaker": turn.get("role"),
              "text": turn.get("content") if isinstance(turn.get("content"), str) else ""}
             for pos, turn in enumerate(_dicts(record.get("turns")))]
    return {"locator": "turns", "turns": turns}


def select(record: dict, locator: str) -> Any:
    """One locator's data, whatever it names: a single part, a range in order, or a whole view.

    `turn:2`, `call:1`, `event:3` and `header` answer one part; `turn:3-9`, `call:0-4` and
    `event:10-30` answer the parts in order under `parts`; `turns` answers every spoken turn
    without tool payloads; `all` answers the whole record."""
    kind = kind_of(record)
    text = (locator or "").strip()
    if text == "turns":
        return _conversation(record)
    if text == "all":
        return record
    name, sep, index = text.partition(":")
    if sep and "-" in index:
        return _range(record, name, index)
    if kind == "run" and text == "header":
        return part(record, locator)
    if _rows_for(kind, record, name) is not None and index.isdigit():
        return part(record, locator)
    raise ValueError(f"{locator!r} names nothing of this {kind}: read one of "
                     f"{_valid_locators(kind)}, a range of one kind, `turns`, or `all`")


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
