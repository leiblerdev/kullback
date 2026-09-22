"""The worlds the Examiner tests share: a small hand-built world with one Task, one confirmed
replay and one finished re-roll of the same End state, which is the world a Verifier is derived
in, probed and refused."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.examiner.plan import ExaminerPlan
from kullback.runner.records import Run, Task, as_dict

WORLD_TASK = "t1"


@dataclass
class World:
    """A workdir with the derivation's inputs and nothing the Examiner may not see."""
    workdir: Path
    inputs: dict
    paths: dict = field(default_factory=dict)

    def plan(self, **kwargs: Any) -> ExaminerPlan:
        return ExaminerPlan(workdir=self.workdir, inputs=self.inputs, **kwargs)


def _write(run: Run, folder: Path) -> str:
    return VF.write_events_jsonl(run, folder / f"{run.run_id}.jsonl")


def make_world(root: Path, *, rerolls: tuple = ("alt",), confirmed: bool = True,
               terminations: Optional[dict] = None, tasks: int = 1) -> World:
    """One Task whose recording is `ref`, with the named re-roll Runs beside it.

    `alt` is the Reference done another way (D46), `bad` a re-roll that died on max_steps, `rr2` a
    finished re-roll that cancels with another reason (a second End state), `wrong` a finished
    re-roll that cancelled the wrong order, `extra` one that cancelled a second order as well.
    `confirmed` is what the replay row says of `ref`; `terminations` overrides what a re-roll row
    says of how its Run ended. `tasks` repeats that Task under the ids t1, t2, ... over the same
    Runs, which is the world several Tasks are derived in at once and cached one by one (D163).
    """
    workdir = root / "world"
    folder = workdir / "runs" / WORLD_TASK
    folder.mkdir(parents=True)
    runs = {"ref": VF.reference_run(), "alt": VF.alt_path_run(), "bad": VF.failed_run(),
            "rr2": VF.other_reason_run(), "wrong": VF.wrong_run(), "extra": VF.extra_write_run()}
    paths = {run_id: _write(run, folder) for run_id, run in runs.items()}
    task_ids = [f"t{n}" for n in range(1, tasks + 1)]
    replays = {task_id: {"ref": {"trace_id": "ref", "run_id": "ref", "confirmed": confirmed,
                                 "path": paths["ref"], "reasons": [] if confirmed else ["writes differ"]}}
               for task_id in task_ids}
    reroll_rows = [{"run_id": run_id, "path": paths[run_id],
                    "termination_reason": (terminations or {}).get(run_id, runs[run_id].termination_reason)}
                   for run_id in rerolls]
    inputs = {
        "tasks": [Task(id=task_id, intent=VF.TASK.intent, run_ids=["ref"]) for task_id in task_ids],
        "sigs": list(SIGS),
        "constraints": [],
        "canon_rules": {},
        "replays": replays,
        "rerolls": {task_id: [dict(row) for row in reroll_rows] for task_id in task_ids},
        "intents": {},
        "user_rules": {},
        "traces": [],
        "assisted_tools": [],
    }
    return World(workdir=workdir, inputs=inputs, paths=paths)


def events_of(run: Run) -> list[dict]:
    """A Run's events as the probe tool takes them."""
    return [as_dict(event) for event in run.events]


def probe_runner_over(run: Optional[Run] = None):
    """A `run_probe(model, verifier)` that returns one hand-built Run: check 6 without a live model."""
    def run_probe(model: Any, verifier: Any) -> Run:
        return run or VF.wrong_run()
    return run_probe


