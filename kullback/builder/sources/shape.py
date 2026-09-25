"""Structure-only summary of a customer file, for formats no reader maps yet.

A newcomer asks whether a file will ingest and what it holds, without storing
anything and without keeping any customer string. This module decodes the file
as ingest does, then walks it and records per key path only the JSON types
seen, how often the path occurs, length ranges, and structural hints. The one
exception is a role-like field, where a handful of short words is the
structure, so those words are kept.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

# A role-like field keeps its words only under this cap: few enough and short
# enough that the set is structure, not content.
_ROLE_VALUES_MAX = 8
_ROLE_VALUE_LEN = 16


def shape_summary(path: str | Path, max_records: int = 200) -> dict:
    """Walk the file and summarize its shape, keeping no customer string but role words.

    At most max_records top-level items are walked (a JSON object counts as one
    record); nested lists are walked fully, so counts are exact within the
    sampled records. Each key path maps to its JSON types, occurrence count,
    length range for strings and lists, hints, and kept values for role-like
    fields only.
    """
    from kullback.builder.ingest import _decode

    payload = Path(path).read_bytes()
    document, jsonl = _decode(payload)
    if document is None:
        return {"file": str(path), "jsonl": bool(jsonl), "records": 0, "paths": {},
                "error": "the file is neither JSON nor JSONL, so it has no shape to summarize"}
    entries: dict[str, dict] = {}
    if isinstance(document, list):
        records = document[:max_records]
        for item in records:
            _walk(item, "[]", entries)
        examined = len(records)
    elif isinstance(document, dict):
        for key, value in document.items():
            _walk(value, str(key), entries)
        examined = 1
    else:
        _walk(document, "value", entries)
        examined = 1
    paths = {name: _finish(entry) for name, entry in entries.items()}
    return {"file": str(path), "jsonl": jsonl, "records": examined, "paths": paths}


def _new_entry() -> dict:
    """One path's running tally while the walk is still collecting values."""
    return {"types": set(), "count": 0, "min_len": None, "max_len": None,
            "distinct": set(), "overflow": False, "sample": [], "tool_call_shape": False}


def _walk(value: Any, path: str, entries: dict[str, dict]) -> None:
    """Fold one value into its path's tally, recursing into lists and objects."""
    entry = entries.setdefault(path, _new_entry())
    entry["types"].add(_json_type(value))
    entry["count"] += 1
    if isinstance(value, str):
        _note_len(entry, len(value))
        _note_string(entry, value)
    elif isinstance(value, list):
        _note_len(entry, len(value))
        for item in value:
            _walk(item, path + "[]", entries)
    elif isinstance(value, dict):
        if "name" in value and "arguments" in value:
            entry["tool_call_shape"] = True
        for key, item in value.items():
            _walk(item, f"{path}.{key}" if path else str(key), entries)


def _json_type(value: Any) -> str:
    """The JSON type name, with bool before int because bool subclasses int."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _note_len(entry: dict, size: int) -> None:
    """Grow the path's length range to cover one more string or list."""
    entry["min_len"] = size if entry["min_len"] is None else min(entry["min_len"], size)
    entry["max_len"] = size if entry["max_len"] is None else max(entry["max_len"], size)


def _note_string(entry: dict, value: str) -> None:
    """Keep one more string for the role and timestamp probes, sampling past the cap."""
    if value not in entry["distinct"]:
        if len(entry["distinct"]) < _ROLE_VALUES_MAX:
            entry["distinct"].add(value)
        else:
            entry["overflow"] = True
    if len(entry["sample"]) < _ROLE_VALUES_MAX and value not in entry["sample"]:
        entry["sample"].append(value)


def _finish(entry: dict) -> dict:
    """One path's public summary: types, counts, lengths, hints, and role words only."""
    out: dict[str, Any] = {"types": sorted(entry["types"]), "count": entry["count"]}
    if entry["min_len"] is not None:
        out["min_len"] = entry["min_len"]
        out["max_len"] = entry["max_len"]
    hints = []
    if _is_role_like(entry):
        hints.append("looks_like_role")
        out["values"] = sorted(entry["distinct"])
    if _is_timestamp_like(entry):
        hints.append("looks_like_timestamp")
    if entry["tool_call_shape"]:
        hints.append("looks_like_tool_call")
    out["hints"] = sorted(hints)
    return out


def _is_role_like(entry: dict) -> bool:
    """True when every value is one short word from a small set, which is structure."""
    if entry["types"] != {"str"} or entry["overflow"] or not entry["distinct"]:
        return False
    return all(0 < len(word) <= _ROLE_VALUE_LEN and word.split() == [word]
               for word in entry["distinct"])


def _is_timestamp_like(entry: dict) -> bool:
    """True when the sampled strings all parse as timestamps."""
    if entry["types"] != {"str"} or not entry["sample"]:
        return False
    for value in entry["sample"]:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
    return True
