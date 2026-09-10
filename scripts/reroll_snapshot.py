"""Snapshot and stub timing for the rerolls dispatch (D240).

Measures whether the flat (Task, Run number) pool settles the same Runs as the per Task pool:
it drives the production dispatch of whichever tree it runs in and dumps every Run id, seed,
recording file name, model call count, event shape and end kind, plus each Task's written
rerolls record, as sorted JSON.

Snapshot: uv run python scripts/reroll_snapshot.py --workers N --workdir <dir> --out <file>
(run at workers 1 and 8 on origin/main and on the head; all four dumps are byte identical).
Timing: uv run python scripts/reroll_snapshot.py --workers 8 --wait 0.05 --time --workdir <dir>
(3 Tasks of count 4 with a 50 ms stub wait per model call).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

from kullback.ai.provider import Model, _as_reply
from kullback.builder import build as B
from kullback.builder import compile_env, parallel
from kullback.runner.records import EntitySchema, Task, TaskOverlay, content_hash

TASKS = ("invented-alpha", "invented-beta", "invented-gamma")
COUNT = 4


class StubModel(Model):
    """Stateless so the schedule cannot move it: the reply is a function of the messages so far."""

    def __init__(self, wait: float = 0.0):
        self.name = "stub"
        self.wait = wait

    def query(self, messages, tools=None, config=None):
        if self.wait:
            time.sleep(self.wait)
        prior = sum(1 for m in messages
                    if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant")
        if prior == 0:
            return _as_reply({"content": "",
                              "tool_calls": [{"id": "c1", "name": "invented-tool", "arguments": {}}]})
        return _as_reply({"content": "done"})


def setup(workdir: Path, tasks):
    workdir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        compile_env._write_overlay(workdir, TaskOverlay(task_id=task.id), {}, runs={},
                                   reference_run_id="")
    return EntitySchema()


def dump(rolled, workdir: Path) -> str:
    rows = []
    for task_id, runs in sorted(rolled.items()):
        for order, (run, path) in enumerate(runs):
            rows.append({
                "task": task_id,
                "order": order,
                "run_id": run.run_id,
                "seed": run.seed,
                "recording": Path(path).name,
                "model_calls": len([e for e in run.events if e.type == "model_call"]),
                "events": [e.type for e in run.events],
                "end": run.termination_reason,
            })
    # The record each side wrote beside the Task's Runs, which is what the next stage reads.
    records = {}
    for task_id in sorted(rolled):
        path = workdir / "runs" / task_id / B.REROLL_RECORD
        records[task_id] = json.loads(path.read_text()) if path.is_file() else None
    return json.dumps({"runs": sorted(rows, key=lambda r: (r["task"], r["order"])),
                       "records": records}, sort_keys=True, indent=1)


def run_stage(workdir: Path, workers: int, wait: float) -> dict:
    tasks = [Task(id=name, run_ids=[]) for name in TASKS]
    schema = setup(workdir, tasks)
    source = compile_env.module_source(schema, [], {})
    model = StubModel(wait=wait)
    keys = {task.id: {"format": "review", "task": task.id} for task in tasks}
    jobs = [(task, None, None, keys[task.id], None) for task in tasks]
    common = dict(source=source, schema=schema, sigs=[], db={}, env_id=None, canon_rules={})

    if hasattr(B, "_reroll_run_jobs"):  # the head: one flat pool over (Task, Run number)
        run_jobs = B._reroll_run_jobs(workdir, jobs, traces={}, user_model=None, rerolls=COUNT,
                                      **common)
        got = parallel.each(run_jobs, functools.partial(B._reroll_run_one, workdir, model), workers)
        B._gather_reroll_rows(workdir, jobs, got)  # writes each Task's record
        pairs: dict = {}
        for task_id, number, run, path in got:
            pairs.setdefault(task_id, []).append((number, run, path))
        return {tid: [(run, path) for _n, run, path in sorted(rows, key=lambda r: r[0])]
                for tid, rows in pairs.items()}

    # origin/main: the pool fans out over Tasks and each Task's Runs are a serial inner loop
    def reroll(job):
        task, rules, prompt, key, reference_id = job
        B._discard_runs(workdir / "runs" / task.id, f"reroll-{task.id}-")
        runs = B._candidate_runs(workdir, task, model, count=COUNT, prefix="reroll",
                                 tag=f"{content_hash(key)[:8]}-", rules=rules,
                                 seed=B.REROLL_SEED, max_turns=B.REROLL_TURNS,
                                 system_prompt=prompt, members=B._members_of(task, {}),
                                 reference=None, user_agent_model=None, **common)
        rows = [{"run_id": r.run_id, "path": p, "termination_reason": r.termination_reason,
                 "user_end": B.user_sim.end_of_run(r)} for r, p in runs]
        B._write_json(workdir / "runs" / task.id / B.REROLL_RECORD,
                      {"task_id": task.id, "key": key,
                       "runs": B._reroll_rows(workdir, rows, relative=True)})
        return runs

    return {job[0].id: rows for job, rows in zip(jobs, parallel.each(jobs, reroll, workers), strict=True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--wait", type=float, default=0.0)
    ap.add_argument("--out")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    start = time.perf_counter()
    rolled = run_stage(workdir, args.workers, args.wait)
    elapsed = time.perf_counter() - start

    shape = "flat pool" if hasattr(B, "_reroll_run_jobs") else "per Task pool"
    if args.time:
        print(f"{shape}: workers {args.workers} wait {args.wait} elapsed {elapsed:.3f} s")
    if args.out:
        Path(args.out).write_text(dump(rolled, workdir))
        print(f"{shape}: wrote {args.out} ({len(Path(args.out).read_bytes())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
