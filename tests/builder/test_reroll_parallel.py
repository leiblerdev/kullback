"""A Task's re-rolled Runs run side by side in Run number order (D240).

Every Task's Runs go to the one shared pool as (Task, Run number) jobs, gathered and written in
Run number order. These tests drive the same helpers the stage drives, over invented Tasks, with a
fresh scripted driver per Run. Production drivers are stateless across calls, so one driver serves
every job; the scripted driver here keeps per Run state, so each job looks up its own driver by
Task and Run number before calling the shipped pool unit.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kullback.ai.provider import TestModel
from kullback.builder import build as build_module
from kullback.builder import compile_env, parallel
from kullback.runner.records import EntitySchema, OverlayRow, Task, TaskOverlay, content_hash

CALL_THEN_STOP = [
    {"content": "", "tool_calls": [{"id": "c1", "name": "invented-tool", "arguments": {}}]},
    {"content": "done"},
]

REROLLS = 4


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


def _tag(task_id: str) -> str:
    """The Run id tag the stage derives from the Task's key, for the invented keys used here."""
    return f"{content_hash({'format': 'review', 'task': task_id})[:8]}-"


def _jobs(workdir: Path, tasks: list[Task]):
    """The stage's per Task jobs and the flat (Task, Run number) jobs built from them."""
    for task in tasks:
        compile_env._write_overlay(workdir, TaskOverlay(task_id=task.id), {}, runs={},
                                   reference_run_id="")
    schema = EntitySchema()
    common = dict(source=compile_env.module_source(schema, [], {}), schema=schema, sigs=[],
                  db={}, env_id=None, canon_rules={})
    jobs = [(task, None, None, {"format": "review", "task": task.id}, None) for task in tasks]
    run_jobs = build_module._reroll_run_jobs(workdir, jobs, traces={}, user_model=None,
                                             rerolls=REROLLS, **common)
    return jobs, run_jobs


def _snapshot(runs) -> str:
    """Every Run's key, seed, turn count and end kind as sorted JSON."""
    rows = [{"run_id": run.run_id, "seed": run.seed,
             "turns": len([e for e in run.events if e.type == "model_call"]),
             "events": [e.type for e in run.events],
             "termination_reason": run.termination_reason} for run, _path in runs]
    return json.dumps(sorted(rows, key=lambda row: row["run_id"]), sort_keys=True)


def _flat(workdir, tasks, workers, wait=0.0, failing=None):
    """The stage's shape: the shipped jobs, pool unit and gather, over per Run drivers."""
    jobs, run_jobs = _jobs(workdir, tasks)
    failing = failing or {}
    model_of = {(task.id, number): Scripted(CALL_THEN_STOP, wait=wait,
                                            fail=(number in failing.get(task.id, ())))
                for task in tasks for number in range(REROLLS)}

    def run_one(run_job):
        task, _task_ctx, number = run_job
        return build_module._reroll_run_one(workdir, model_of[(task.id, number)], run_job)

    got = parallel.each(run_jobs, run_one, workers)
    rolled = build_module._gather_reroll_rows(workdir, jobs, got, rerolls=REROLLS)
    by_id = {(task_id, run.run_id): (run, path) for task_id, _number, run, path in got}
    return {task.id: [by_id[(task.id, row["run_id"])] for row in rolled[task.id]]
            for task in tasks}


def test_keys_and_seeds_do_not_depend_on_execution_order(tmp_path):
    """Run ids and their seeds are Task, tag and attempt only (D212): workers 1 and 8 agree."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta", "gamma")]
    one = _flat(tmp_path / "one", tasks, workers=1)
    eight = _flat(tmp_path / "eight", tasks, workers=8)
    assert _snapshot([pair for rows in eight.values() for pair in rows]) == \
        _snapshot([pair for rows in one.values() for pair in rows])
    assert [run.run_id for rows in eight.values() for run, _ in rows] == [
        f"reroll-{task.id}-{_tag(task.id)}{number}" for task in tasks for number in range(4)]
    seeds = [run.seed for rows in eight.values() for run, _ in rows]
    assert len(set(seeds)) == len(seeds), "one id draws one seed"


def test_results_land_in_run_number_order(tmp_path):
    """Even with waits that scramble completion order, each Task's rows stay numbered 0 to 3."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta")]
    out = _flat(tmp_path, tasks, workers=8, wait=0.02)
    for task in tasks:
        assert [run.run_id for run, _ in out[task.id]] == [
            f"reroll-{task.id}-{_tag(task.id)}{number}" for number in range(4)]


def test_the_pool_bound_holds(tmp_path):
    """No more than workers jobs in flight, counted through the drivers' own tracker."""
    tracker = {"lock": threading.Lock(), "live": 0, "peak": 0}
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta", "gamma")]
    _jobs_list, run_jobs = _jobs(tmp_path, tasks)

    def run_one(run_job):
        return build_module._reroll_run_one(
            tmp_path, Scripted(CALL_THEN_STOP, wait=0.05, tracker=tracker), run_job)

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
        f"reroll-invented-alpha-{_tag('invented-alpha')}{number}" for number in range(4)]
    assert all(Path(path).is_file() for _, path in rows)
    assert rows[1][0].termination_reason is not None


def test_the_serial_wrapper_matches_the_flat_pool(tmp_path):
    """`_candidate_runs` (the Examiner verb's path) settles the same Runs the pool settles."""
    tasks = [Task(id=f"invented-{name}", run_ids=[]) for name in ("alpha", "beta")]
    flat = _flat(tmp_path / "flat", tasks, workers=8)
    serial = {}
    for task in tasks:
        workdir = tmp_path / "serial"
        compile_env._write_overlay(workdir, TaskOverlay(task_id=task.id), {}, runs={},
                                   reference_run_id="")
        schema = EntitySchema()
        serial[task.id] = build_module._candidate_runs(
            workdir, task, Scripted(CALL_THEN_STOP), count=4, prefix="reroll",
            tag=_tag(task.id), source=compile_env.module_source(schema, [], {}),
            schema=schema, sigs=[], db={}, env_id=None, canon_rules={}, rules=None, seed=0,
            max_turns=6, system_prompt=None, members=[], reference=None, user_agent_model=None)
    assert _snapshot([pair for rows in serial.values() for pair in rows]) == \
        _snapshot([pair for rows in flat.values() for pair in rows])


def test_runs_of_one_task_do_not_share_a_nested_overlay_container(tmp_path, monkeypatch):
    """Two Runs built from one Task's overlay rows hold no nested container in common (D240).

    The toolkit keeps the caller's row objects by reference, so each Run deep copies the rows at
    its own build. An in place append in one Run's copy is invisible in the other Run's copy and
    in the shared ctx.
    """
    task = Task(id="invented-delta", run_ids=[])
    ctx = _ctx(tmp_path, task)
    ctx["overlay"] = TaskOverlay(task_id=task.id, rows=[
        OverlayRow(table="invented-table", id="invented-row", version_hash="invented-v1")])
    ctx["overlay_rows"] = {"invented-v1": {"items": ["a", "b"]}}
    seen = []
    real_load = compile_env.load_toolkit

    def spy(source, db, **kwargs):
        seen.append(kwargs.get("overlay_values"))
        return real_load(source, db, **kwargs)

    monkeypatch.setattr(compile_env, "load_toolkit", spy)
    build_module._candidate_run_once(tmp_path, task, Scripted(CALL_THEN_STOP), ctx=ctx, number=0)
    build_module._candidate_run_once(tmp_path, task, Scripted(CALL_THEN_STOP), ctx=ctx, number=1)
    assert len(seen) == 2
    assert seen[0] is not ctx["overlay_rows"] and seen[1] is not ctx["overlay_rows"]
    assert seen[0] is not seen[1]
    seen[0]["invented-v1"]["items"].append("written-by-run-zero")
    assert seen[1]["invented-v1"]["items"] == ["a", "b"]
    assert ctx["overlay_rows"]["invented-v1"]["items"] == ["a", "b"]


def test_a_helper_edit_moves_the_rerolls_key(monkeypatch):
    """The rerolls key covers the helpers the stage body calls, so editing one re-rolls once."""
    model = SimpleNamespace(name="invented-driver")
    before = build_module._rerolls_stage(model, REROLLS).code_version

    def _reroll_run_one(*args, **kwargs):
        raise AssertionError("a patched helper never runs here, only its source is hashed")

    monkeypatch.setattr(build_module, "_reroll_run_one", _reroll_run_one)
    after = build_module._rerolls_stage(model, REROLLS).code_version
    assert before != after


def test_a_short_answer_list_raises_instead_of_writing_empty_rows(tmp_path):
    """A pool answer count that does not match jobs times rerolls raises (D240)."""
    tasks = [Task(id="invented-alpha", run_ids=[])]
    jobs, run_jobs = _jobs(tmp_path, tasks)
    got = [build_module._reroll_run_one(tmp_path, Scripted(CALL_THEN_STOP), run_job)
           for run_job in run_jobs[:3]]
    with pytest.raises(build_module.BuildError):
        build_module._gather_reroll_rows(tmp_path, jobs, got, rerolls=REROLLS)
    assert not (tmp_path / "runs" / "invented-alpha" / build_module.REROLL_RECORD).is_file()
