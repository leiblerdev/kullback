"""The runner as a tool: run, replay and reroll over a built Environment.

Three functions with one shape, usable as AgentTool bodies by any extension and callable
from tests. They wire today's loop, route, records, replay and verdict modules without
rewriting them: the candidate model is driven by the runner loop through the route order
the Router owns (refused, real, code, recording, cannot_answer, stand_in), every Run lands
as JSONL under workdir/runs/, and every Verdict carries the runner version beside it.

The candidate's user arrives as an object, never as an import: the caller that owns a user
extension passes it in, so this module never reaches into the user package. Where the Task
carries rules but no user was passed, the Run goes without a Simulated user and ends where
a package Run ends. The runner loop (runner/loop.py) is kept because tests pin its exact
JSONL; the core harness cutover is left to the agent stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kullback.runner import loop, route
from kullback.runner import replay as replay_mod
from kullback.runner.records import as_dict, plain
from kullback.runner.replay import AGREES
from kullback.runner.verdict import verdict as score_verdict
from kullback.runner.world import loading
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.loading import EnvironmentError

RUNS_DIR = "runs"
MAX_TURNS = 30


@dataclass
class RunReport:
    """One Run played: its Verdict, the route each call took, the world diff and the spend."""

    run_id: str
    task_id: str
    path: str
    verdict: Any = None
    routes: list = field(default_factory=list)
    world_diff: dict = field(default_factory=dict)
    spend: dict = field(default_factory=dict)
    runner_version: str = ""
    termination_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "task_id": self.task_id, "path": self.path,
                "verdict": as_dict(self.verdict) if self.verdict is not None else None,
                "routes": list(self.routes), "world_diff": copy.deepcopy(self.world_diff),
                "spend": dict(self.spend), "runner_version": self.runner_version,
                "termination_reason": self.termination_reason}


@dataclass
class ReplayCall:
    """One recorded call replayed: both answers and the first column that parted them."""

    call_id: Optional[str]
    tool: str
    kind: str
    verdict: str
    first_differing_column: Optional[str]
    recorded: str
    ours: str

    def as_dict(self) -> dict:
        return {"call_id": self.call_id, "tool": self.tool, "kind": self.kind,
                "verdict": self.verdict, "first_differing_column": self.first_differing_column,
                "recorded": self.recorded, "ours": self.ours}


@dataclass
class ReplayReport:
    """One Task replayed: per-call answers column by column, and the fidelity they compose."""

    task_id: str
    trace_id: str
    run_id: str
    path: str
    confirmed: bool
    fidelity: float
    calls: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    runner_version: str = ""

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "trace_id": self.trace_id, "run_id": self.run_id,
                "path": self.path, "confirmed": self.confirmed, "fidelity": self.fidelity,
                "calls": [call.as_dict() if isinstance(call, ReplayCall) else call
                          for call in self.calls],
                "counts": dict(self.counts), "reasons": list(self.reasons),
                "runner_version": self.runner_version}


def version() -> str:
    """sha256 over the bytes of every module under kullback/runner plus the core loop it runs on.

    Stored beside every Verdict (`runner_version`) and in every report, so a cached artifact
    graded under an older runner is told apart from a fresh one.
    """
    package = Path(__file__).resolve().parent.parent
    runner_dir = package / "runner"
    digest = hashlib.sha256()
    for path in sorted(runner_dir.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in ("agent/loop.py", "agent/harness.py"):
        path = package / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _router_for(env: BuiltEnvironment, task_id: str, seed: int) -> Any:
    """The Task's Router: built toolkit, Starting state, overlay, signatures, canon rules."""
    overlay, overlay_rows = env.overlay(task_id)
    source = env.toolkit_source()
    db = copy.deepcopy(env.db)
    toolkit = loading.load_toolkit(source, db, overlay=overlay, overlay_values=overlay_rows)
    context = getattr(toolkit, "ctx", None)
    if context is not None:
        context.reseed(seed)
    return route.Router(
        env_tools_module=toolkit, starting_state=copy.deepcopy(env.db),
        overlay=overlay, overlay_rows=overlay_rows, tool_sigs=env.sigs,
        canon_rules=env.canon_rules,
        synthetic_rows=getattr(env.schema, "synthetic_rows", None) or ())


def _freeze(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, default=str)


def _world_diff(before: Any, after: Any) -> dict:
    """Rows changed per table, before and after: what one Run moved in the world."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    diff: dict = {}
    for table in sorted(set(before) | set(after), key=str):
        old_rows = before.get(table) or {}
        new_rows = after.get(table) or {}
        if not isinstance(old_rows, dict) or not isinstance(new_rows, dict):
            if _freeze(old_rows) != _freeze(new_rows):
                diff[str(table)] = {"before": plain(old_rows), "after": plain(new_rows)}
            continue
        changed: dict = {}
        for row_id in sorted(set(old_rows) | set(new_rows), key=str):
            old = plain(old_rows.get(row_id)) if row_id in old_rows else None
            new = plain(new_rows.get(row_id)) if row_id in new_rows else None
            if _freeze(old) != _freeze(new):
                changed[str(row_id)] = {"before": old, "after": new}
        if changed:
            diff[str(table)] = changed
    return diff


def _model_name(model: Any) -> str:
    return str(getattr(model, "name", None) or type(model).__name__)


def run(environment_dir: Any, task_id: str, model: Any, *, user: Any = None, workdir: Any,
        seed: int = 0) -> RunReport:
    """Play one Run of the candidate model in the Environment and report it.

    The candidate is driven by the runner loop with only the Environment's tools registered,
    through the route; the passed user answers when the Task has one. Writes the Run JSONL
    under workdir/runs/ and returns the Verdict, the route taken per call, the world diff
    (rows changed, per table, before and after), and the spend.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    if env.agent_driven(task_id):
        raise EnvironmentError(f"Task {task_id} is driven by the agent user, which needs a model "
                               "the runner is not given")
    router = _router_for(env, task_id, seed)
    runs_dir = Path(workdir) / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{task_id}-{seed}"
    specs = env.tool_specs()
    state = loop.new_run_state(
        run_id, workdir=runs_dir, env_id=env.env_id, task_id=task_id,
        model=_model_name(model), seed=seed, user=user,
        user_rules=env.rules(task),
        max_turns=MAX_TURNS, system_prompt=env.system_prompt(task),
        first_user=task.intent if user is None else None)
    loop.open_with_user(state)
    loop.run(state, model, tools=specs, router=router)
    loop.finish(state, router)
    report_version = version()
    verifier = env.verifier(task_id)
    scored = score_verdict(state.path, verifier, canon=env.canon_rules, environment=env,
                           runner_version=report_version) if verifier is not None else None
    taken = [{"id": event.payload.get("id"), "name": event.payload.get("name"),
              "route": event.route}
             for event in state.run.events if event.type == "tool_result"]
    return RunReport(run_id=run_id, task_id=task_id, path=str(state.path) if state.path else "",
                     verdict=scored, routes=taken,
                     world_diff=_world_diff(router.start_world, router.world()),
                     spend=as_dict(state.run.cost), runner_version=report_version,
                     termination_reason=state.run.termination_reason)


def _first_differing_column(check: dict) -> Optional[str]:
    """The first column one replayed call parted on: the changed key, else the leaf's head."""
    difference = check.get("difference") or {}
    changed = difference.get("keys_changed") or []
    if changed:
        return str(changed[0])
    leaf = difference.get("leaf")
    if isinstance(leaf, str) and leaf:
        head = leaf.split(":", 1)[0].strip()
        return head or None
    columns = difference.get("columns") or []
    if columns:
        return str(columns[0])
    return None


def replay(environment_dir: Any, task_id: str, *, workdir: Any) -> ReplayReport:
    """Replay the Task's recorded calls through the built tools and report them call by call.

    Returns, per call, the recorded result versus ours with the first differing column named,
    plus the fidelity the replay scoring composes: the share of recorded calls the rebuilt
    tools answer the way the recording did. Dense feedback, never a count without its rows.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    reference = env.reference(task)
    if reference is None:
        raise EnvironmentError(f"{environment_dir} holds no reference Trace for Task {task_id}")
    router = _router_for(env, task_id, seed=0)
    runs_dir = Path(workdir) / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = replay_mod.replay_trace(
        reference, router, workdir=runs_dir, task_id=task_id, env_id=env.env_id,
        write_tools=env.write_tools(), canon_rules=env.canon_rules)
    calls = [ReplayCall(call_id=check.get("call_id"), tool=str(check.get("tool")),
                        kind=str(check.get("kind")), verdict=str(check.get("verdict")),
                        first_differing_column=_first_differing_column(check),
                        recorded=str(check.get("recorded")),
                        ours=str(check.get("ours")))
             for check in out.checks]
    total = len(calls)
    fidelity = sum(1 for call in calls if call.verdict in AGREES) / total if total else 1.0
    return ReplayReport(task_id=task_id, trace_id=out.trace_id, run_id=out.run_id,
                        path=out.path, confirmed=out.confirmed, fidelity=fidelity,
                        calls=calls, counts=dict(out.counts), reasons=list(out.reasons),
                        runner_version=version())


def reroll(environment_dir: Any, task_id: str, model: Any, *, count: int,
           workdir: Any) -> list[RunReport]:
    """Play `count` Runs of the candidate model over the Task, seeds counting up from zero."""
    return [run(environment_dir, task_id, model, workdir=workdir, seed=seed)
            for seed in range(max(int(count), 0))]


__all__ = ["ReplayCall", "ReplayReport", "RunReport", "replay", "reroll", "run", "version"]
