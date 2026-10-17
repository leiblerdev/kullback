"""What each tool always does, learned once per Environment build from the corpus's recordings.

Two readings, both code, both read by the Spec writer's checks (spec/items.py):

1. Effects. Every successful recorded call of a write tool runs again in the build's sandbox
   (`Sandbox.run_diff`, the transition gate's per-call world diff) on its own Starting state after
   its trace's prefix. The columns it moved on rows that were there before, and the tables it made
   rows in, are kept per call; a tool's effects are what every one of its calls moved. A list
   position in a column path is a pattern (`history.N`). The sanity item frees these columns where
   the end state expects the tool to run, so a right Run is not failed for what the tool always does.
   Beside them, what the calls wrote (`writes`): per table and field name, the values any call wrote
   there where they are literal (at most MAX_LITERAL distinct), and the arguments every call's written
   values came from (`from_args`), where the field copies what the call passed. A gated value must be
   one a tool can write (D387, spec/items.py `Reach`), and the writer reads these with the tool list.
2. Signatures. Per tool and argument: the JSON types the recorded calls used (tool_sigs.json where it
   says, else the recordings), for a list or a dict of objects the keys every object carried and the
   keys any carried, and the pairs of text arguments no recorded call ever gave the same value. A called
   item whose argument no recorded call could have made is refused with the reason.

The Builder runs the sandbox and stores both at exam/review/effects.json beside the review's tools.json
(builder/tool_effects.py), once per build; the Spec stage only reads the file (`load`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import read_json

EFFECTS_FILE = Path("exam") / "review" / "effects.json"
# What the stored file holds; a build's digest carries it, so a file learned before `writes` is learned again.
EFFECTS_VERSION = 2
# A field a tool wrote more distinct values to than this is filled from its arguments, not literal (D387).
MAX_LITERAL = 12
# A pair of arguments is called never equal only on this many recorded calls that carry both.
MIN_PAIR_CALLS = 3
JSON_TYPES = {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict", type(None): "NoneType"}


def type_of(value: Any) -> str:
    return JSON_TYPES.get(type(value), type(value).__name__)


def _pattern(part: str) -> str:
    """A list position or a keyed position (a value used as a key, holding a digit) as N."""
    return "N" if part.isdigit() or (re.search(r"\d", part) and re.search(r"[^A-Za-z0-9_]", part)) else part


def _leaf_paths(before: Any, after: Any, path: str = "") -> set[str]:
    """Every path where two values differ, cut after the first pattern step (a list entry is one change)."""
    if path and path.rsplit(".", 1)[-1] == "N":
        return {path} if before != after else set()
    if isinstance(before, dict) and isinstance(after, dict):
        out: set[str] = set()
        for key in set(before) | set(after):
            out |= _leaf_paths(before.get(key), after.get(key), f"{path}.{_pattern(str(key))}" if path
                               else str(key))
        return out
    if isinstance(before, list) and isinstance(after, list):
        out = set()
        for index in range(max(len(before), len(after))):
            one = before[index] if index < len(before) else None
            two = after[index] if index < len(after) else None
            out |= _leaf_paths(one, two, f"{path}.N")
        return out
    return {path} if before != after and path else set()


def call_effects(changed: dict) -> tuple[set[tuple[str, str]], set[str]]:
    """One call's diff as ((table, path pattern) moved on rows it kept, tables it made rows in)."""
    moved: set[tuple[str, str]] = set()
    made: set[str] = set()
    for table, rows in (changed or {}).items():
        for change in (rows or {}).values():
            before, after = change.get("before"), change.get("after")
            if before is None and after is not None:
                made.add(str(table))
            elif before is not None and after is not None:
                moved |= {(str(table), path) for path in _leaf_paths(before, after)}
    return moved, made


def _written(before: Any, after: Any, key: str) -> Iterable[tuple[str, Any]]:
    """(field name, value) for each scalar of `after` that differs from `before` at its place; a list
    entry's scalars are named by their own key, or by the list's where they have none."""
    if isinstance(after, dict):
        for name, value in after.items():
            yield from _written(before.get(name) if isinstance(before, dict) else None, value, str(name))
    elif isinstance(after, list):
        for index, value in enumerate(after):
            yield from _written(before[index] if isinstance(before, list) and index < len(before) else None,
                                value, key)
    elif after is not None and after != before:
        yield key, after


def call_writes(changed: dict) -> set[tuple[str, str, str]]:
    """One call's diff as (table, field name, value as JSON) per scalar it wrote, on rows it kept or made."""
    out: set[tuple[str, str, str]] = set()
    for table, rows in (changed or {}).items():
        for change in (rows or {}).values():
            if change.get("after") is not None:
                out |= {(str(table), name, json.dumps(value, sort_keys=True))
                        for name, value in _written(change.get("before"), change["after"], "")}
    return out


def _arg_leaves(args: Any) -> dict[str, set[str]]:
    """Per argument, its scalars as JSON (a list or an object argument by its members)."""
    def leaves(value: Any) -> Iterable[Any]:
        if isinstance(value, dict):
            for one in value.values():
                yield from leaves(one)
        elif isinstance(value, list):
            for one in value:
                yield from leaves(one)
        elif value is not None:
            yield value
    return {str(name): {json.dumps(one, sort_keys=True) for one in leaves(value)}
            for name, value in (args or {}).items()}


def literal_writes(diffs: Iterable[dict], args: Optional[Iterable[dict]] = None) -> list[dict]:
    """[{table, field, values?, from_args?}] per field the calls wrote, sorted: `values` where they wrote at
    most MAX_LITERAL distinct values there; `from_args`, given each call's arguments beside its diff, the
    arguments that held every value each call wrote there. A field with neither is left out."""
    diffs = list(diffs)
    calls = list(args) if args is not None else [None] * len(diffs)
    seen: dict[tuple[str, str], set[str]] = {}
    fed: dict[tuple[str, str], Optional[set[str]]] = {}
    for changed, call_args in zip(diffs, calls, strict=False):
        written: dict[tuple[str, str], set[str]] = {}
        for table, name, value in call_writes(changed):
            written.setdefault((table, name), set()).add(value)
        held = _arg_leaves(call_args) if call_args is not None else {}
        for key, values in written.items():
            seen.setdefault(key, set()).update(values)
            names = {arg for arg, leaves in held.items() if values <= leaves}
            fed[key] = names if fed.get(key) is None else fed[key] & names
    out = []
    for (table, name), values in sorted(seen.items()):
        row: dict = {"table": table, "field": name}
        if len(values) <= MAX_LITERAL:
            row["values"] = [json.loads(value) for value in sorted(values)]
        if fed.get((table, name)):
            row["from_args"] = sorted(fed[(table, name)])
        if name and len(row) > 2:
            out.append(row)
    return out


def always(diffs: Iterable[dict], args: Optional[Iterable[dict]] = None) -> dict:
    """What every one of a tool's calls moved: {"columns": [{table, column, patterns}], "makes": [...]},
    and what any of them wrote where it is literal or copied from an argument ("writes", `literal_writes`).

    A column every call moved is free even where its path differs call to call (a value used as a key);
    `patterns` are the paths every call shared under it, `<column>.*` where they share none.
    """
    diffs = list(diffs)
    moved_all: Optional[set] = None
    columns_all: Optional[set] = None
    made_all: Optional[set] = None
    for changed in diffs:
        moved, made = call_effects(changed)
        columns = {(table, path.split(".")[0]) for table, path in moved}
        moved_all = moved if moved_all is None else moved_all & moved
        columns_all = columns if columns_all is None else columns_all & columns
        made_all = made if made_all is None else made_all & made
    out = []
    for table, column in sorted(columns_all or ()):
        patterns = sorted(path for t, path in moved_all or () if t == table and path.split(".")[0] == column)
        out.append({"table": table, "column": column, "patterns": patterns or [f"{column}.*"]})
    return {"columns": out, "makes": sorted(made_all or ()), "writes": literal_writes(diffs, args)}


def _object_keys(value: Any) -> Optional[list[set]]:
    """The key sets of the objects a list (or a dict of objects) holds; None when it holds none."""
    items = value if isinstance(value, list) else list(value.values()) if isinstance(value, dict) else []
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    return [set(item) for item in items]


def _declared(sigs: Iterable[Any]) -> dict:
    """Per tool, per argument, the types tool_sigs.json declares."""
    out = {}
    for sig in sigs:
        body = sig.model_dump(mode="json", by_alias=True) if hasattr(sig, "model_dump") else dict(sig)
        properties = (body.get("args_schema") or {}).get("properties") or {}
        out[body.get("name")] = {name: list(spec.get("type") or []) for name, spec in properties.items()
                                 if isinstance(spec, dict)}
    return out


def _tally(tool: dict, pairs: dict, args: dict) -> None:
    """One successful call's arguments into its tool's row: types, object keys, and equal pairs."""
    tool["calls"] += 1
    for name, value in args.items():
        arg = tool["args"].setdefault(name, {"types": set(), "required_keys": None, "keys": set()})
        arg["types"].add(type_of(value))
        for keys in _object_keys(value) or ():
            arg["keys"] |= keys
            arg["required_keys"] = keys if arg["required_keys"] is None else arg["required_keys"] & keys
    names = sorted(args)
    for n, one in enumerate(names):
        for two in names[n + 1:]:
            seen = pairs.setdefault((one, two), [0, 0])
            seen[0] += 1
            seen[1] += _same(args[one], args[two])


def _finish(arg: dict, declared: list, keys_named: set) -> None:
    arg["types"] = sorted(set(declared) | arg["types"])
    arg["required_keys"] = sorted(arg["required_keys"] or ())
    arg["keys"] = sorted(arg["keys"])
    arg["key_fields"] = [key for key in arg["required_keys"]
                         if key in keys_named or key.lower().endswith("id")] or arg["required_keys"]


def _never_equal(args: dict, pairs: dict) -> list[list[str]]:
    """The pairs of text arguments no recorded call (of at least MIN_PAIR_CALLS carrying both) gave one value."""
    return sorted([one, two] for (one, two), (count, equal) in pairs.items()
                  if count >= MIN_PAIR_CALLS and not equal and args[one]["types"] == args[two]["types"] == ["str"])


def signatures(calls: Iterable[Any], sigs: Iterable[Any] = (), key_columns: Iterable[str] = ()) -> dict:
    """Per tool, per argument, the types, object keys and never-equal partners the successful calls show.

    An object list's `key_fields` name its objects: the keys every object carried that are a table's key
    column or end in "id", else every key every object carried.
    """
    declared = _declared(sigs)
    out: dict[str, dict] = {}
    pairs: dict[str, dict] = {}
    for call in calls:
        if call.error is None and isinstance(call.args, dict):
            _tally(out.setdefault(call.name, {"calls": 0, "args": {}}), pairs.setdefault(call.name, {}), call.args)
    for name, tool in out.items():
        for arg_name, arg in tool["args"].items():
            _finish(arg, declared.get(name, {}).get(arg_name) or [], set(key_columns))
        tool["never_equal"] = _never_equal(tool["args"], pairs[name])
    return out


def _same(one: Any, two: Any) -> bool:
    return json.dumps(one, sort_keys=True, default=str) == json.dumps(two, sort_keys=True, default=str)


def load(workdir: Any) -> dict:
    """The build's learned effects and signatures (builder/tool_effects.py writes them); none when absent."""
    stored = read_json(Path(workdir) / EFFECTS_FILE, None)
    if not isinstance(stored, dict):
        return {"effects": {}, "signatures": {}}
    return {"effects": stored.get("effects") or {}, "signatures": stored.get("signatures") or {}}


__all__ = ["EFFECTS_FILE", "EFFECTS_VERSION", "MAX_LITERAL", "always", "call_effects", "call_writes", "literal_writes",
           "load", "signatures", "type_of"]
