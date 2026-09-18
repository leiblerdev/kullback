"""Measure, per recorded write call, what the built body changed against what the traces witness.

Code only, no model calls. Every workdir argument is opened read only; nothing is written
into it. Run from the worktree root:

    uv run python scripts/measure_transition.py <workdir> [<workdir> ...]

For every recorded write call whose tool has a built body, the body runs in process on a
copy of the world the recording shows before the call (the same per-call Starting state
the gates score on), the state change it made is diffed, and that change is compared with
the witnessed change (the same per-call evidence the replay checks, compared under the
workdir's own canon rules the way scoring compares). Per tool and per workdir it counts:
agrees, disagrees, unwitnessed, body raised. Disagreeing calls are split by whether the
existing per-call replay row already passed them, and by whether a later call of the same
recording fails in the stored replays.

Stdout may name tools, tables and columns; that output is local rerun material and never
reaches git. Anything copied into a report carries counts only.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kullback.builder import cluster, compile_env
from kullback.builder import effects as effects_mod
from kullback.builder.build import (
    after_write_calls,
    callers_by_tool,
    evidence_calls,
    load_traces,
    replay_failures,
    trace_worlds,
)
from kullback.builder.compile_env import call_starting_states, load_run_overlays, load_toolkit, overlay_values
from kullback.runner.canon import rules_of
from kullback.runner.records import EntitySchema, Task, TaskOverlay, ToolSig, Trace


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_db(toolkit) -> dict:
    db = getattr(toolkit, "db", None)
    if hasattr(db, "model_dump"):
        return db.model_dump(mode="json")
    if isinstance(db, dict):
        return copy.deepcopy(db)
    return json.loads(json.dumps(db, default=str))


def _changed_pairs(before: dict, after: dict) -> dict:
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


def measure_workdir(workdir: Path) -> dict:
    schema = EntitySchema.model_validate(_load_json(workdir / "schema.json"))
    db = _load_json(workdir / "db.json")
    rules = rules_of({"canon_rules": _load_json(workdir / "canon-rules.json")})
    sigs = [ToolSig.model_validate(row) for row in _load_json(workdir / "tool_sigs.json")]
    by_name = {sig.name: sig for sig in sigs}
    writes = set(cluster.write_tool_names(sigs))
    bodies = _load_json(workdir / "bodies.json") or {}
    tasks_doc = _load_json(workdir / "tasks.json")
    task_rows = tasks_doc.get("tasks") if isinstance(tasks_doc, dict) else tasks_doc
    tasks = [Task.model_validate(row) for row in task_rows]
    traces: list[Trace] = load_traces(workdir)
    overlay_dir = workdir / "overlays"
    overlays = []
    if overlay_dir.is_dir():
        for path in sorted(overlay_dir.glob("*.json")):
            payload = _load_json(path)
            overlay = payload.get("overlay", payload)
            overlays.append(TaskOverlay.model_validate(overlay))
    values = overlay_values(workdir)
    run_overlays = load_run_overlays(workdir)

    task_of = {trace.trace_id: next((t.id for t in tasks if trace.trace_id in t.run_ids), None)
               for trace in traces}
    call_tasks = {call.id: task_of[trace.trace_id] for trace in traces for call in trace.tool_calls
                  if call.id and task_of[trace.trace_id]}
    call_runs = {call.id: trace.trace_id for trace in traces for call in trace.tool_calls if call.id}
    states = call_starting_states(db, overlays, values, call_tasks, run_overlays, call_runs)
    worlds = trace_worlds(db, overlays, values, tasks)

    # The compile stage's own evidence set, mirrored exactly: seed traces alone, after-write
    # calls out (a call that followed a write on the same row saw a world no Starting state
    # can put back), the anchor held out, one caller filter, replay failures readmitted.
    held_out_runs = {run for runs in ((_load_json(workdir / "anchor.json") or {}).get("held_out")
                                      or {}).values() for run in runs}
    seeds = {trace.trace_id for trace in traces} - held_out_runs
    calls_by_tool, _, _ = evidence_calls(traces, seeds, after_write_calls(traces, writes),
                                         callers_by_tool(sigs), replay_failures(workdir))
    evidenced_ids = {call.id for calls in calls_by_tool.values() for call in calls if call.id}

    observed = effects_mod.observe_effects([t for t in traces if t.trace_id in seeds], schema,
                                            writes, db=db, worlds=worlds)
    evidence = effects_mod.replay_evidence(
        {tool: rows for tool, rows in observed.items() if tool in bodies})
    outcomes = _load_json(workdir / "tool_call_outcomes.json") or {}
    passed = {row["call_id"] for rows in outcomes.values() for row in rows if row.get("replayed")}

    replays = _load_json(workdir / "replays.json") or {}
    run_checks: dict[str, list[dict]] = {}
    for task_rows in replays.values():
        if isinstance(task_rows, dict):
            for run_id, entry in task_rows.items():
                checks = entry.get("checks") if isinstance(entry, dict) else None
                if isinstance(checks, list):
                    run_checks[run_id] = checks
    call_positions = {trace.trace_id: [c.id for c in trace.tool_calls] for trace in traces}

    per_tool: dict[str, dict[str, int]] = {}
    downstream = 0
    downstream_denominator = 0

    def row(tool: str) -> dict[str, int]:
        return per_tool.setdefault(tool, {"calls": 0, "agrees": 0, "disagrees": 0,
                                          "unwitnessed": 0, "raised": 0,
                                          "disagree_passed": 0, "disagree_failed": 0})

    for trace in traces:
        positions = call_positions[trace.trace_id]
        checks = run_checks.get(trace.trace_id, [])
        later_failing = {c.get("call_id") for c in checks if c.get("verdict") == "differs"}
        for index, call in enumerate(trace.tool_calls):
            if call.name not in writes or call.error is not None or not call.id:
                continue
            if call.id not in evidenced_ids:
                continue
            if call.name not in bodies or not (bodies[call.name] or "").strip():
                continue
            counts = row(call.name)
            counts["calls"] += 1
            pre = copy.deepcopy(states.get(call.id, db))
            try:
                source = compile_env.module_source(schema, [by_name[call.name]],
                                                   {call.name: bodies[call.name]})
                toolkit = load_toolkit(source, pre)
                # The validated world as dumped, before the body runs: validation narrows
                # the world to what the schema declares, so the diff below compares what
                # the body saw with what it left, exactly as the gate's subprocess does.
                seen_before = _dump_db(toolkit)
                getattr(toolkit, call.name)(**(dict(call.args or {})))
            except Exception:
                counts["raised"] += 1
                continue
            try:
                post = _dump_db(toolkit)
            except Exception:
                counts["raised"] += 1
                continue
            made = effects_mod.body_made_change(_changed_pairs(seen_before, post), schema,
                                                 rules)
            rows = evidence.get(call.id, [])
            seen = {(str(r["table"]), str(r["row"]), str(r["path"])): r["after"]
                    for r in rows}
            if not seen:
                counts["unwitnessed"] += 1
                continue
            # The gate's own rule: a missing or wrong witnessed column fails, and so does a
            # row no sighting associates with the call; an extra column on a witnessed row
            # does not, since sightings are partial.
            if effects_mod.compare_transition(made, rows, rules)["verdict"] == "agrees":
                counts["agrees"] += 1
            else:
                counts["disagrees"] += 1
                if call.id in passed:
                    counts["disagree_passed"] += 1
                else:
                    counts["disagree_failed"] += 1
                rest = set(positions[index + 1:])
                downstream_denominator += 1
                if rest & later_failing:
                    downstream += 1

    return {"workdir": str(workdir), "per_tool": per_tool,
            "downstream": downstream, "downstream_of": downstream_denominator}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for raw in argv[1:]:
        result = measure_workdir(Path(raw))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
