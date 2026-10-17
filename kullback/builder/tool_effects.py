"""The build's tool effects and call signatures, learned once and stored for the Spec stage (spec/effects.py).

Every successful recorded write call runs again in the sandbox on its own Starting state after its
trace's prefix (`Sandbox.run_diff`, the per-call world diff the transition gate reads), and the
recorded calls' arguments give the signatures. The file is keyed by a digest of the tool module, the
recordings and the sigs, so a build whose bodies or recordings changed learns again and an unchanged
one reads what it stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kullback.builder import cluster
from kullback.builder.compile_env import call_starting_states, load_run_overlays, overlay_values
from kullback.builder.env_files import _read_db, _read_schema, _read_sigs, render_module
from kullback.builder.sandbox import Sandbox, SandboxError, calls_by_trace
from kullback.builder.world_tools import load_traces
from kullback.gates.tool_runs import key_fields
from kullback.runner.records import Task, TaskOverlay, content_hash, read_json
from kullback.runner.world.recorded import recorded_call_contexts
from kullback.spec.effects import EFFECTS_FILE, EFFECTS_VERSION, always, signatures


def digest(workdir: Path) -> str:
    folder = workdir / "traces"
    traces = sorted(path.name for path in folder.glob("*.json")) if folder.is_dir() else []
    return content_hash({"module": render_module(workdir), "traces": traces,
                         "sigs": read_json(workdir / "tool_sigs.json", []), "version": EFFECTS_VERSION})


def _tasks(workdir: Path) -> list[Task]:
    doc = read_json(workdir / "tasks.json", []) or []
    return [Task.model_validate(row) for row in (doc.get("tasks") if isinstance(doc, dict) else doc)]


def _overlays(workdir: Path) -> list[TaskOverlay]:
    folder = workdir / "overlays"
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        payload = read_json(path, {}) or {}
        out.append(TaskOverlay.model_validate(payload.get("overlay", payload)))
    return out


def _call_states(workdir: Path, calls: list, db: dict) -> dict:
    """Call id to the Starting state its own Run ran on, as the compile stage serves it."""
    task_of = {run_id: task.id for task in _tasks(workdir) for run_id in task.run_ids}
    call_tasks = {call.id: task_of[call.trace_id] for call in calls if call.id and call.trace_id in task_of}
    call_runs = {call.id: call.trace_id for call in calls if call.id}
    return call_starting_states(db, _overlays(workdir), overlay_values(workdir), call_tasks,
                                load_run_overlays(workdir), call_runs)


def world_diffs(workdir: Path, scratch: Path, traces: list, write_tools: set[str]) -> dict[str, list[dict]]:
    """Per write tool, the world diff of each successful recorded call, run as the transition gate runs it."""
    return {tool: [changed for changed, _ in rows] for tool, rows in call_diffs(workdir, scratch, traces,
                                                                                 write_tools).items()}


def call_diffs(workdir: Path, scratch: Path, traces: list, write_tools: set[str]) -> dict[str, list[tuple]]:
    """Per write tool, (world diff, arguments) of each successful recorded call (`world_diffs` with the args)."""
    calls = [call.model_copy(update={"trace_id": call.trace_id or trace.trace_id})
             for trace in traces for call in trace.tool_calls]
    writes = [call for call in calls if call.name in write_tools and call.error is None and call.id]
    if not writes:
        return {}
    db = _read_db(workdir, None)
    sandbox = Sandbox(render_module(workdir), db, scratch, call_states=_call_states(workdir, calls, db),
                      call_context=recorded_call_contexts(calls, _read_schema(workdir)),
                      trace_calls=calls_by_trace(calls))
    try:
        entries = sandbox.run_diff(writes)
    except SandboxError:
        return {}
    out: dict[str, list[tuple]] = {}
    for call, entry in zip(writes, entries, strict=False):
        if entry.get("changed") is not None and not entry.get("truncated"):
            out.setdefault(call.name, []).append((entry["changed"], call.args))
    return out


def learn(workdir: Any, scratch: Optional[Any] = None) -> dict:
    """Effects and signatures of one Environment build, from its recordings and its compiled bodies."""
    root = Path(workdir)
    sigs = _read_sigs(root)
    traces = load_traces(root)
    writes = set(cluster.write_tool_names(sigs))
    diffs = call_diffs(root, Path(scratch) if scratch is not None else root / EFFECTS_FILE.parent, traces, writes)
    schema = _read_schema(root)
    keys = {name for table in schema.tables for name in key_fields(schema, table)}
    return {"effects": {tool: dict(always([diff for diff, _ in rows], [args for _, args in rows]), calls=len(rows))
                        for tool, rows in sorted(diffs.items())},
            "signatures": signatures([call for trace in traces for call in trace.tool_calls], sigs, keys)}


def store(workdir: Any) -> Path:
    """Learn and write the build's effects file unless the stored one was learned from the same build."""
    root = Path(workdir)
    path = root / EFFECTS_FILE
    key = digest(root)
    stored = read_json(path, None)
    if isinstance(stored, dict) and stored.get("digest") == key:
        return path
    learned = dict(learn(root), digest=key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(learned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = ["call_diffs", "digest", "learn", "store", "world_diffs"]
