"""Inventory of what the recordings witness for the tool context, per workdir.

For every recorded write call in every trace, per tool, this counts how many
results carry an id no earlier call of the same recording showed, how that id
relates to what a body could know (equal to an argument, matching a miner
stated id shape, already held by the seed world, or none of those), and how
many result values read as a time the arguments and the world so far do not
explain. Read only: workdirs are opened read only and never written.

Run from the repo root: uv run python scripts/tool_context_inventory.py <workdir>...
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^\d{2}:\d{2}(:\d{2})?$"
)


def is_id_key(key):
    name = str(key).lower()
    return name == "id" or name.endswith("_id")


def walk_strings(obj, key=""):
    if isinstance(obj, dict):
        for name in sorted(obj, key=str):
            yield from walk_strings(obj[name], str(name))
    elif isinstance(obj, (list, tuple)):
        for inner in obj:
            yield from walk_strings(inner, key)
    elif isinstance(obj, str):
        yield key, obj


def collect_strings(obj):
    return {value for _, value in walk_strings(obj)}


def _record_new_ids(row, fresh, arg_strings, patterns, seed_strings):
    for _, value in fresh:
        row["distinct_new"].add(value)
        if value in seed_strings:
            row["in_seed"] += 1
        if value in arg_strings:
            row["eq_arg"] += 1
        elif any(re.match("^(?:" + pattern + ")$", value) for pattern in patterns):
            row["shape"] += 1
        else:
            row["neither"] += 1


def _later_strings(calls):
    return set().union(*(collect_strings(call.get("args") or {}) | collect_strings(call.get("result"))
                         for call in calls))


def _record_write_inventory(row, result, arg_strings, seen, later_calls, patterns, seed_strings):
    row["writes"] += 1
    if not isinstance(result, dict):
        return
    fresh = [(key, value) for key, value in walk_strings(result)
             if is_id_key(key) and value not in seen]
    if fresh:
        row["trace_new"] += 1
        _record_new_ids(row, fresh, arg_strings, patterns, seed_strings)
        later = _later_strings(later_calls)
        if any(value in later for _, value in fresh):
            row["persist"] += 1
    for key, value in walk_strings(result):
        value = value.strip()
        if (len(value) >= 4 and TIME_RE.match(value) and value not in arg_strings
                and value not in seen and value not in seed_strings):
            row["time_unexpl"] += 1
            row["time_cols"][key] += 1


def inventory(workdir):
    sigs = {sig["name"]: sig for sig in json.load(open(os.path.join(workdir, "tool_sigs.json")))}
    write_tools = {name for name, sig in sigs.items() if sig.get("kind") == "write"}
    schema = json.load(open(os.path.join(workdir, "schema.json")))
    patterns = list((schema.get("id_patterns") or {}).values())
    seed_strings = collect_strings(json.load(open(os.path.join(workdir, "db.json"))))
    per_tool = defaultdict(lambda: {"writes": 0, "trace_new": 0, "persist": 0, "in_seed": 0,
                                    "eq_arg": 0, "shape": 0, "neither": 0,
                                    "distinct_new": set(), "time_unexpl": 0,
                                    "time_cols": defaultdict(int)})
    files = sorted(glob.glob(os.path.join(workdir, "traces", "*.json")))
    for path in files:
        calls = json.load(open(path)).get("tool_calls") or []
        seen = set()
        for index, call in enumerate(calls):
            args = call.get("args") or {}
            result = call.get("result")
            arg_strings = collect_strings(args)
            if call.get("name") in write_tools:
                _record_write_inventory(per_tool[call["name"]], result, arg_strings, seen,
                                        calls[index + 1:], patterns, seed_strings)
            seen |= arg_strings
            if result is not None:
                seen |= collect_strings(result)
            if call.get("error"):
                seen |= collect_strings(call.get("error"))
    return files, write_tools, per_tool


def main(argv):
    if len(argv) < 2:
        print(__doc__.splitlines()[-1])
        return 1
    for workdir in argv[1:]:
        files, write_tools, per_tool = inventory(workdir)
        print(f"===== {workdir} ({len(files)} traces, write tools: {len(write_tools)})")
        for tool in sorted(per_tool):
            row = per_tool[tool]
            print(f"  {tool}: writes={row['writes']} trace_new={row['trace_new']} "
                  f"persist={row['persist']} of_new(in_seed={row['in_seed']} "
                  f"eq_arg={row['eq_arg']} shape={row['shape']} neither={row['neither']}) "
                  f"distinct_new={len(row['distinct_new'])} "
                  f"time_unexpl={row['time_unexpl']} {dict(row['time_cols'])}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
