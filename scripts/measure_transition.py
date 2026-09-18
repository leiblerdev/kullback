"""Measure, per recorded write call, what the built body changed against what the traces witness.

Code only, no model calls. Every workdir argument is opened read only; nothing is written
into it. Run from the worktree root:

    uv run python scripts/measure_transition.py <workdir> [<workdir> ...]

For every recorded write call in the compile stage's own evidence set whose tool has a
built body, the body runs in process on a copy of the world the recording shows before the
call, the state change it made is diffed over the validated world, and that change is
compared with the witnessed change under the workdir's own canon rules. Per tool and per
workdir it counts: agrees, disagrees, unwitnessed, body raised. Disagreeing calls are split
by whether the existing per-call replay row already passed them, and by whether a later
call of the same recording fails in the stored replays.

Stdout may name tools, tables and columns; that output is local rerun material and never
reaches git. Anything copied into a report carries counts only.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transition_common import (
    derive_evidence,
    derive_worlds,
    index_replays,
    load_basics,
    passed_ids,
)

from kullback.builder import compile_env
from kullback.builder import effects as effects_mod
from kullback.builder.compile_env import load_toolkit


def dump_db(toolkit) -> dict:
    db = getattr(toolkit, "db", None)
    if hasattr(db, "model_dump"):
        return db.model_dump(mode="json")
    if isinstance(db, dict):
        return copy.deepcopy(db)
    return {"value": str(db)}


def changed_pairs(before: dict, after: dict) -> dict:
    """Rows that part between two validated dumps, as table to row to either side."""
    out: dict = {}
    for table in sorted(set(before) | set(after)):
        old_rows = before.get(table) if isinstance(before.get(table), dict) else {}
        new_rows = after.get(table) if isinstance(after.get(table), dict) else {}
        for row_id in sorted(set(old_rows) | set(new_rows), key=str):
            old, new = old_rows.get(row_id), new_rows.get(row_id)
            if old != new:
                out.setdefault(str(table), {})[str(row_id)] = {"before": old, "after": new}
    return out


def iter_calls(basics: dict, evidenced_ids: set[str]):
    """Every evidence write call with a built body, with its trace."""
    for trace in basics["traces"]:
        for call in trace.tool_calls:
            if call.name not in basics["writes"] or call.error is not None or not call.id:
                continue
            if call.id not in evidenced_ids:
                continue
            if not (basics["bodies"].get(call.name) or "").strip():
                continue
            yield trace, call


def grade_call(basics: dict, states: dict, evidence: dict, call) -> str:
    """One call's ruling: agrees, disagrees, unwitnessed or raised."""
    rows = evidence.get(call.id, [])
    if not rows:
        return "unwitnessed"
    pre = copy.deepcopy(states.get(call.id, basics["db"]))
    try:
        source = compile_env.module_source(basics["schema"], [basics["by_name"][call.name]],
                                           {call.name: basics["bodies"][call.name]})
        toolkit = load_toolkit(source, pre)
        seen_before = dump_db(toolkit)
        getattr(toolkit, call.name)(**(dict(call.args or {})))
        post = dump_db(toolkit)
    except Exception:
        return "raised"
    made = effects_mod.body_made_change(changed_pairs(seen_before, post), basics["schema"],
                                        basics["rules"])
    if effects_mod.compare_transition(made, rows, basics["rules"])["verdict"] == "agrees":
        return "agrees"
    return "disagrees"


def measure_workdir(workdir: Path) -> dict:
    basics = load_basics(workdir)
    maps = derive_worlds(basics)
    shown = derive_evidence(workdir, basics, maps["worlds"])
    evidence = effects_mod.replay_evidence(
        {tool: rows for tool, rows in shown["observed"].items() if tool in basics["bodies"]})
    passed = passed_ids(workdir)
    run_checks = index_replays(workdir)
    per_tool: dict[str, dict[str, int]] = {}
    downstream = 0
    downstream_of = 0

    def row(tool: str) -> dict[str, int]:
        return per_tool.setdefault(tool, {"calls": 0, "agrees": 0, "disagrees": 0,
                                          "unwitnessed": 0, "raised": 0,
                                          "disagree_passed": 0, "disagree_failed": 0})

    positions = {trace.trace_id: [c.id for c in trace.tool_calls] for trace in basics["traces"]}
    for trace, call in iter_calls(basics, shown["evidenced_ids"]):
        counts = row(call.name)
        counts["calls"] += 1
        ruling = grade_call(basics, maps["states"], evidence, call)
        counts[ruling] += 1
        if ruling != "disagrees":
            continue
        if call.id in passed:
            counts["disagree_passed"] += 1
        else:
            counts["disagree_failed"] += 1
        downstream_of += 1
        at = positions[trace.trace_id].index(call.id)
        later = set(positions[trace.trace_id][at + 1:])
        failing = {c.get("call_id") for c in run_checks.get(trace.trace_id, [])
                   if c.get("verdict") == "differs"}
        if later & failing:
            downstream += 1
    return {"workdir": str(workdir), "per_tool": per_tool,
            "downstream": downstream, "downstream_of": downstream_of}


def print_result(result: dict) -> None:
    print(f"== {result['workdir']}")
    total = {"calls": 0, "agrees": 0, "disagrees": 0, "unwitnessed": 0, "raised": 0,
             "disagree_passed": 0, "disagree_failed": 0}
    for tool in sorted(result["per_tool"]):
        counts = result["per_tool"][tool]
        for key in total:
            total[key] += counts.get(key, 0)
        print(f"  {tool}: calls={counts['calls']} agrees={counts['agrees']} "
              f"disagrees={counts['disagrees']} unwitnessed={counts['unwitnessed']} "
              f"raised={counts['raised']} disagree_passed={counts['disagree_passed']} "
              f"disagree_failed={counts['disagree_failed']}")
    print(f"  TOTAL: {total} downstream_failures={result['downstream']} "
          f"of={result['downstream_of']}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for raw in argv[1:]:
        print_result(measure_workdir(Path(raw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
