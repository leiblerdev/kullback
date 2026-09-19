"""Shared loading for the transition-gate scripts, read only and code only.

Both scripts open stored workdirs read only and mirror the compile stage's own evidence set:
seed traces alone, after-write calls out, the anchor held out, one caller filter, replay
failures readmitted. Anything copied into a report carries counts only.
"""

from __future__ import annotations

import json
from pathlib import Path

from kullback.builder import cluster
from kullback.builder import effects as effects_mod
from kullback.builder.build import (
    after_write_calls,
    callers_by_tool,
    evidence_calls,
    load_traces,
    replay_failures,
    trace_worlds,
)
from kullback.builder.compile_env import call_starting_states, load_run_overlays, overlay_values
from kullback.runner.canon import rules_of
from kullback.runner.records import EntitySchema, Task, TaskOverlay, ToolSig


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_basics(workdir: Path) -> dict:
    """The workdir's records as models and plain data, nothing derived."""
    tasks_doc = load_json(workdir / "tasks.json")
    task_rows = tasks_doc.get("tasks") if isinstance(tasks_doc, dict) else tasks_doc
    overlays = []
    overlay_dir = workdir / "overlays"
    if overlay_dir.is_dir():
        for path in sorted(overlay_dir.glob("*.json")):
            payload = load_json(path)
            overlays.append(TaskOverlay.model_validate(payload.get("overlay", payload)))
    sigs = [ToolSig.model_validate(row) for row in load_json(workdir / "tool_sigs.json")]
    return {"schema": EntitySchema.model_validate(load_json(workdir / "schema.json")),
            "db": load_json(workdir / "db.json"),
            "rules": rules_of({"canon_rules": load_json(workdir / "canon-rules.json")}),
            "sigs": sigs,
            "by_name": {sig.name: sig for sig in sigs},
            "writes": set(cluster.write_tool_names(sigs)),
            "bodies": load_json(workdir / "bodies.json") or {},
            "tasks": [Task.model_validate(row) for row in task_rows],
            "traces": load_traces(workdir),
            "overlays": overlays,
            "values": overlay_values(workdir),
            "run_overlays": load_run_overlays(workdir)}


def derive_worlds(basics: dict) -> dict:
    """Per-call Starting states and per-trace starting worlds, as the gates score on."""
    task_of = {trace.trace_id: next((t.id for t in basics["tasks"]
                                     if trace.trace_id in t.run_ids), None)
               for trace in basics["traces"]}
    call_tasks = {call.id: task_of[trace.trace_id] for trace in basics["traces"]
                  for call in trace.tool_calls if call.id and task_of[trace.trace_id]}
    call_runs = {call.id: trace.trace_id for trace in basics["traces"]
                 for call in trace.tool_calls if call.id}
    states = call_starting_states(basics["db"], basics["overlays"], basics["values"],
                                  call_tasks, basics["run_overlays"], call_runs)
    return {"task_of": task_of, "call_tasks": call_tasks, "call_runs": call_runs,
            "states": states,
            "worlds": trace_worlds(basics["db"], basics["overlays"], basics["values"],
                                   basics["tasks"])}


def seed_ids(workdir: Path, traces) -> set[str]:
    """Trace ids the Builder may learn from: everything the anchor did not hold out."""
    anchor = load_json(workdir / "anchor.json") or {}
    held_out = {run for runs in (anchor.get("held_out") or {}).values() for run in runs}
    return {trace.trace_id for trace in traces} - held_out


def derive_evidence(workdir: Path, basics: dict, worlds: dict) -> dict:
    """The compile stage's own evidence set over the seed traces."""
    seeds = seed_ids(workdir, basics["traces"])
    calls_by_tool, _, _ = evidence_calls(
        basics["traces"], seeds, after_write_calls(basics["traces"], basics["writes"]),
        callers_by_tool(basics["sigs"]), replay_failures(workdir))
    seed_traces = [t for t in basics["traces"] if t.trace_id in seeds]
    observed = effects_mod.observe_effects(seed_traces, basics["schema"], basics["writes"],
                                           db=basics["db"], worlds=worlds)
    return {"seeds": seeds, "calls_by_tool": calls_by_tool, "observed": observed,
            "evidenced_ids": {call.id for calls in calls_by_tool.values()
                              for call in calls if call.id}}


def index_replays(workdir: Path) -> dict[str, list[dict]]:
    """Run id to the stored replay checks, for the downstream reading."""
    out: dict[str, list[dict]] = {}
    replays = load_json(workdir / "replays.json") or {}
    for task_rows in replays.values():
        if isinstance(task_rows, dict):
            for run_id, entry in task_rows.items():
                checks = entry.get("checks") if isinstance(entry, dict) else None
                if isinstance(checks, list):
                    out[run_id] = checks
    return out


def passed_ids(workdir: Path) -> set[str]:
    """Call ids whose per-call replay row passed, the existing gates' answer per call."""
    outcomes = load_json(workdir / "tool_call_outcomes.json") or {}
    return {row["call_id"] for rows in outcomes.values() for row in rows if row.get("replayed")}
