"""The worlds the Examiner tests share: the fixture build's store (no Task there has a Reference, so it is the
world for the no-Reference rows, the surface and the session tests) and a small hand-built world with
one Task, one confirmed replay and one finished re-roll of the same End state, which is the world a
Verifier is derived in, probed, repaired and refused."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from builder.test_build import Bodies
from gates import verifier_fixtures as VF
from gates.examiner_fixtures import SIGS
from kullback.agent.harness import AgentHarness
from kullback.agent.session import SessionStore
from kullback.agent.tools import ToolResult
from kullback.builder import build as build_module
from kullback.builder import pipeline
from kullback.builder.build import BuildPlan
from kullback.examiner import agent as examiner_agent
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.stage import DERIVE_INPUTS
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
               terminations: Optional[dict] = None) -> World:
    """One Task whose recording is `ref`, with the named re-roll Runs beside it.

    `alt` is the Reference done another way (D46), `bad` a re-roll that died on max_steps, `rr2` a
    finished re-roll that cancels with another reason (a second End state), `wrong` a finished
    re-roll that cancelled the wrong order, `extra` one that cancelled a second order as well.
    `confirmed` is what the replay row says of `ref`; `terminations` overrides what a re-roll row
    says of how its Run ended.
    """
    workdir = root / "world"
    folder = workdir / "runs" / WORLD_TASK
    folder.mkdir(parents=True)
    runs = {"ref": VF.reference_run(), "alt": VF.alt_path_run(), "bad": VF.failed_run(),
            "rr2": VF.other_reason_run(), "wrong": VF.wrong_run(), "extra": VF.extra_write_run()}
    paths = {run_id: _write(run, folder) for run_id, run in runs.items()}
    replays = {WORLD_TASK: {"ref": {"trace_id": "ref", "run_id": "ref", "confirmed": confirmed,
                                    "path": paths["ref"], "reasons": [] if confirmed else ["writes differ"]}}}
    reroll_rows = [{"run_id": run_id, "path": paths[run_id],
                    "termination_reason": (terminations or {}).get(run_id, runs[run_id].termination_reason)}
                   for run_id in rerolls]
    inputs = {
        "tasks": [Task(id=WORLD_TASK, intent=VF.TASK.intent, run_ids=["ref"])],
        "sigs": list(SIGS),
        "constraints": [],
        "canon_rules": {},
        "replays": replays,
        "rerolls": {WORLD_TASK: reroll_rows},
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


def _fixture_path(request) -> Path:
    return Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"


@dataclass
class FixtureBuild:
    """The Builder's store over the small tau2 file, built once per session, and the Examiner's slice of it."""
    workdir: Path
    plan: BuildPlan

    @property
    def inputs(self) -> dict:
        return {key: self.plan.store[key] for key in DERIVE_INPUTS if key in self.plan.store}

    def copy(self, into: Path) -> Path:
        """A copy of the workdir a test may write into, the Run paths in it pointed at the copy."""
        target = into / "build"
        shutil.copytree(self.workdir, target)
        return target

    def inputs_for(self, workdir: Path) -> dict:
        """The inputs with every Run path under the copy rather than the session's workdir."""
        text = json.dumps(_plain(self.inputs), default=str)
        moved = json.loads(text.replace(str(self.workdir.resolve()), str(workdir.resolve()))
                           .replace(str(self.workdir), str(workdir)))
        out = dict(self.inputs)
        out["replays"], out["rerolls"] = moved["replays"], moved["rerolls"]
        return out


def _plain(inputs: dict) -> dict:
    return {"replays": inputs.get("replays") or {}, "rerolls": inputs.get("rerolls") or {}}


def build_fixture(workdir: Path, request) -> FixtureBuild:
    plan = BuildPlan(workdir=workdir, model=Bodies(), files=[_fixture_path(request)], max_attempts=0)
    build_module.execute(plan)
    return FixtureBuild(workdir=workdir, plan=plan)


def anchor_of(workdir: Path):
    return pipeline.load_anchor(workdir)


def session_harness(plan: ExaminerPlan, path: Path, agent_model=None, subscribers=()) -> AgentHarness:
    """An Examiner harness with a session file, so entry ids exist for the context guards."""
    return examiner_agent.examiner_harness(plan, agent_model, subscribers, session=SessionStore.load(path))


def drive(harness: AgentHarness, name: str, arguments: dict, call_id: str = "call-1") -> ToolResult:
    return examiner_agent.drive_tool(harness, name, arguments, call_id=call_id)
