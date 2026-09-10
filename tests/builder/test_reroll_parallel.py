"""A Task's re-rolled Runs run side by side in Run number order (D240).

Every Task's Runs go to the one shared pool as (Task, Run number) jobs, gathered and written in
Run number order. These tests run the same helpers the stage runs, over invented Tasks, with a
fresh scripted driver per Run (production drivers are stateless across calls, so identical scripts
mean identical Runs whatever the schedule).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from kullback.ai.provider import TestModel
from kullback.builder import build as build_module
from kullback.builder import compile_env, parallel
from kullback.runner.records import EntitySchema, Task, TaskOverlay

CALL_THEN_STOP = [
    {"content": "", "tool_calls": [{"id": "c1", "name": "invented-tool", "arguments": {}}]},
    {"content": "done"},
]


class Scripted(TestModel):
    """The tests' fake driver with per-call waits and a ceiling on calls in flight."""

    def __init__(self, replies, wait=0.0, tracker=None, fail=False):
        super().__init__(list(replies), loop=True)
        self.wait = wait
        self.tracker = tracker
        self.fail = fail

    def query(self, messages, tools=None, config=None):
        if self.fail:
            raise RuntimeError("the invented driver fell over")
        if self.tracker is not None:
            with self.tracker["lock"]:
                self.tracker["live"] += 1
                self.tracker["peak"] = max(self.tracker["peak"], self.tracker["live"])
        try:
            if self.wait:
                time.sleep(self.wait)
            return super().query(messages, tools=tools, config=config)
        finally:
            if self.tracker is not None:
                with self.tracker["lock"]:
                    self.tracker["live"] -= 1


def _ctx(workdir: Path, task: Task, tag="abcd1234-"):
    compile_env._write_overlay(workdir, TaskOverlay(task_id=task.id), {}, runs={},
                               reference_run_id="")
    schema = EntitySchema()
    return build_module._candidate_task_ctx(
        workdir, task, prefix="reroll", source=compile_env.module_source(schema, [], {}),
        schema=schema, sigs=[], db={}, env_id=None, canon_rules={}, rules=None, seed=0,
        max_turns=6, system_prompt=None, tag=tag, members=[], reference=None,
        user_agent_model=None)


def _snapshot(runs) -> str:
    """Every Run's key, seed, turn count and end kind as sorted JSON."""
    rows = [{"run_id": run.run_id, "seed": run.seed,
             "turns": len([e for e in run.events if e.type == "model_call"]),
             "events": [e.type for e in run.events],
             "termination_reason": run.termination_reason} for run, _path in runs]
    return json.dumps(sorted(rows, key=lambda row: row["run_id"]), sort_keys=True)


def _flat(workdir, tasks, workers, wait=0.0, failing=None):
    """The stage's shape: one shared pool over (Task, Run number) jobs, gathered in order."""
    ctxs = {task.id: _ctx(workdir, task) for task in tasks}
    failing = failing or {}
    model_of = {task.id: [Scripted(CALL_THEN_STOP, wait=wait, fail=(number in failing.get(task.id, ())))
                          for number in range(4)] for task in tasks}
    run_jobs = [(task, ctxs[task.id], number) for task in tasks for number in range(4)]

    def run_one(run_job):
        task, task_ctx, number = run_job
        run, path = build_module._candidate_run_once(
            workdir, task, model_of[task.id][number], ctx=task_ctx, number=number)
        return (task.id, number, run, path)

    got = parallel.each(run_jobs, run_one, workers)
    by_task: dict[str, list] = {}
    for task_id, number, run, path in got:
        by_task.setdefault(task_id, []).append((number, run, path))
    return {task.id: [(run, path) for _, run, path in sorted(by_task.get(task.id, []))]
            for task in tasks}


def test_keys_and_seeds_do_not_depend_on_execution_order(tmp_path):
    """Run ids and their seeds are Task, tag and attempt only (D212): workers 1 and 8 agree."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta", "gamma")]
    one = _flat(tmp_path / "one", tasks, workers=1)
    eight = _flat(tmp_path / "eight", tasks, workers=8)
    assert _snapshot([pair for rows in eight.values() for pair in rows]) == \
        _snapshot([pair for rows in one.values() for pair in rows])
    assert [run.run_id for rows in eight.values() for run, _ in rows] == [
        f"reroll-{task.id}-abcd1234-{number}" for task in tasks for number in range(4)]
    seeds = [run.seed for rows in eight.values() for run, _ in rows]
    assert len(set(seeds)) == len(seeds), "one id draws one seed"


def test_results_land_in_run_number_order(tmp_path):
    """Even with waits that scramble completion order, each Task's rows stay numbered 0 to 3."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta")]
    out = _flat(tmp_path, tasks, workers=8, wait=0.02)
    for task in tasks:
        assert [run.run_id for run, _ in out[task.id]] == [
            f"reroll-{task.id}-abcd1234-{number}" for number in range(4)]


def test_the_pool_bound_holds(tmp_path):
    """No more than workers jobs in flight, counted through the drivers' own tracker."""
    tracker = {"lock": threading.Lock(), "live": 0, "peak": 0}
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta", "gamma")]
    workdir = tmp_path
    ctxs = {task.id: _ctx(workdir, task) for task in tasks}
    run_jobs = [(task, ctxs[task.id], number) for task in tasks for number in range(4)]

    def run_one(run_job):
        task, task_ctx, number = run_job
        run, path = build_module._candidate_run_once(
            workdir, task, Scripted(CALL_THEN_STOP, wait=0.05, tracker=tracker),
            ctx=task_ctx, number=number)
        return (task.id, number, run, path)

    got = parallel.each(run_jobs, run_one, 8)
    assert len(got) == 12
    assert tracker["peak"] <= 8, f"{tracker['peak']} jobs in flight with 8 workers"
    assert tracker["peak"] > 4, f"{tracker['peak']} peak shows nothing ran side by side"


def test_a_failing_run_does_not_lose_its_siblings(tmp_path):
    """A Run whose driver falls over still leaves its row, and its siblings still leave theirs."""
    tasks = [Task(id="invented-alpha", run_ids=[])]
    out = _flat(tmp_path, tasks, workers=8, failing={"invented-alpha": (1,)})
    rows = out["invented-alpha"]
    assert [run.run_id for run, _ in rows] == [
        f"reroll-invented-alpha-abcd1234-{number}" for number in range(4)]
    assert all(Path(path).is_file() for _, path in rows)
    assert rows[1][0].termination_reason is not None


def test_the_serial_wrapper_matches_the_flat_pool(tmp_path):
    """`_candidate_runs` (the Examiner verb's path) settles the same Runs the pool settles."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta")]
    flat = _flat(tmp_path / "flat", tasks, workers=8)
    serial = {}
    for task in tasks:
        workdir = tmp_path / "serial"
        ctx = _ctx(workdir, task)
        models = [Scripted(CALL_THEN_STOP) for _ in range(4)]
        runs = [build_module._candidate_run_once(workdir, task, model, ctx=ctx, number=number)
                for number, model in enumerate(models)]
        serial[task.id] = runs
    assert _snapshot([pair for rows in serial.values() for pair in rows]) == \
        _snapshot([pair for rows in flat.values() for pair in rows])
