"""Grade every stored write body through run_gates with the transition gate on.

Code only, no model calls. Each workdir argument is opened read only; the sandboxes run
under a scratch directory given by --scratch (never inside the workdir). Prints per tool
the transition metrics and which stage refused the body, if any:

    uv run python scripts/run_transition_gate.py --scratch <dir> <workdir> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kullback.builder import cluster, compile_env
from kullback.builder import effects as effects_mod
from kullback.builder import sandbox as sandbox_mod
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


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def grade_workdir(workdir: Path, scratch: Path) -> dict:
    schema = EntitySchema.model_validate(_load_json(workdir / "schema.json"))
    db = _load_json(workdir / "db.json")
    rules = rules_of({"canon_rules": _load_json(workdir / "canon-rules.json")})
    sigs = [ToolSig.model_validate(row) for row in _load_json(workdir / "tool_sigs.json")]
    by_name = {sig.name: sig for sig in sigs}
    writes = set(cluster.write_tool_names(sigs))
    bodies = _load_json(workdir / "bodies.json") or {}
    tasks_doc = _load_json(workdir / "tasks.json")
    tasks = [Task.model_validate(row)
             for row in (tasks_doc.get("tasks") if isinstance(tasks_doc, dict) else tasks_doc)]
    traces = load_traces(workdir)
    overlays = []
    overlay_dir = workdir / "overlays"
    if overlay_dir.is_dir():
        for path in sorted(overlay_dir.glob("*.json")):
            payload = _load_json(path)
            overlays.append(TaskOverlay.model_validate(payload.get("overlay", payload)))
    values = overlay_values(workdir)
    run_overlays = load_run_overlays(workdir)

    task_of = {trace.trace_id: next((t.id for t in tasks if trace.trace_id in t.run_ids), None)
               for trace in traces}
    call_tasks = {call.id: task_of[trace.trace_id] for trace in traces for call in trace.tool_calls
                  if call.id and task_of[trace.trace_id]}
    call_runs = {call.id: trace.trace_id for trace in traces for call in trace.tool_calls if call.id}
    states = call_starting_states(db, overlays, values, call_tasks, run_overlays, call_runs)
    worlds = trace_worlds(db, overlays, values, tasks)
    held_out_runs = {run for runs in ((_load_json(workdir / "anchor.json") or {}).get("held_out")
                                      or {}).values() for run in runs}
    seeds = {trace.trace_id for trace in traces} - held_out_runs
    observed = effects_mod.observe_effects([t for t in traces if t.trace_id in seeds], schema,
                                            writes, db=db, worlds=worlds)

    calls_by_tool, _, _ = evidence_calls(traces, seeds, after_write_calls(traces, writes),
                                         callers_by_tool(sigs), replay_failures(workdir))

    out: dict[str, dict] = {}
    for tool in sorted(calls_by_tool):
        # Production hands the gate evidence on write tools alone; read tools carry none.
        if tool not in writes:
            continue
        calls = [call for call in calls_by_tool[tool] if bodies.get(tool)]
        if not calls:
            continue
        evidence = effects_mod.replay_evidence({tool: observed.get(tool, [])})
        source = compile_env.module_source(schema, [by_name[tool]], {tool: bodies[tool]})
        box = sandbox_mod.Sandbox(source, db, scratch / workdir.name / tool,
                                  call_states=states, call_tasks=call_tasks, timeout=600.0)
        gates = sandbox_mod.run_gates(source, box, calls, [], schema,
                                      rules=rules, sig=by_name[tool],
                                      transition_evidence=evidence)
        failed = [gate for gate in gates if not gate.passed]
        transition = next((gate for gate in gates if gate.stage == sandbox_mod.TRANSITION_STAGE),
                          None)
        # The gate's own verdict beside the chain's: stored bodies usually fall at an earlier
        # gate, which says nothing about what the transition gate rules on them.
        alone = sandbox_mod.gate_transition(box, calls, schema, evidence, rules=rules)
        out[tool] = {"refused": bool(failed),
                     "refused_by": failed[0].stage if failed else None,
                     "transition": dict(transition.metrics) if transition is not None else {},
                     "alone": {"passed": alone.passed, **dict(alone.metrics)},
                     "first_failure": failed[0].failures[:1] if failed else []}
    return {"workdir": str(workdir), "per_tool": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True)
    parser.add_argument("workdirs", nargs="+")
    args = parser.parse_args(argv)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    refused_total = 0
    for raw in args.workdirs:
        result = grade_workdir(Path(raw), scratch)
        print(f"== {result['workdir']}")
        for tool in sorted(result["per_tool"]):
            row = result["per_tool"][tool]
            print(f"  {tool}: refused={row['refused']} refused_by={row['refused_by']} "
                  f"transition={row['transition']} alone={row['alone']}")
            if not row["alone"]["passed"]:
                refused_total += 1
    print(f"REFUSED_BY_TRANSITION_TOOLS={refused_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
