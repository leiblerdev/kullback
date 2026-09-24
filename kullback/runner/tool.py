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

from kullback import sampling
from kullback.runner import loop, route
from kullback.runner import replay as replay_mod
from kullback.runner.real_tools import default_world_factory, real_tools_from
from kullback.runner.records import (
    RawPtr,
    ToolCall,
    Trace,
    Turn,
    as_dict,
    plain,
    read_json,
    stored_run_path,
    write_json,
)
from kullback.runner.replay import AGREES
from kullback.runner.target import load_run
from kullback.runner.verdict import verdict as score_verdict
from kullback.runner.world import clock, loading
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.loading import RUN_SEED_KIND, EnvironmentError

RUNS_DIR = "runs"
PROBES_DIR = "probes"
MAX_TURNS = 30
PROBE_TURNS = 6


@dataclass
class RunReport:
    """One Run played: its Verdict, the route each call took, the world diff and the spend.

    `path` is relative to the workdir (runs/...), resolved with records.run_path by every reader.
    """

    run_id: str
    task_id: str
    path: str
    verdict: Any = None
    routes: list = field(default_factory=list)
    world_diff: dict = field(default_factory=dict)
    spend: dict = field(default_factory=dict)
    runner_version: str = ""
    termination_reason: Optional[str] = None
    user_end: Optional[str] = None

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "task_id": self.task_id, "path": self.path,
                "verdict": as_dict(self.verdict) if self.verdict is not None else None,
                "routes": list(self.routes), "world_diff": copy.deepcopy(self.world_diff),
                "spend": dict(self.spend), "runner_version": self.runner_version,
                "termination_reason": self.termination_reason, "user_end": self.user_end}


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
    """One Task replayed: per-call answers column by column, and the fidelity they compose.

    `path` is relative to the workdir, the way RunReport stores it.
    """

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
    held_out: bool = False
    reference: bool = False

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "trace_id": self.trace_id, "run_id": self.run_id,
                "path": self.path, "confirmed": self.confirmed, "fidelity": self.fidelity,
                "calls": [call.as_dict() if isinstance(call, ReplayCall) else call
                          for call in self.calls],
                "counts": dict(self.counts), "reasons": list(self.reasons),
                "runner_version": self.runner_version, "held_out": self.held_out,
                "reference": self.reference}


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


def _router_for(env: BuiltEnvironment, task_id: str, seed: int,
                trace_id: Optional[str] = None) -> Any:
    """The Task's Router: built toolkit, Starting state, overlay, signatures, canon rules.

    With a Trace named, the overlay is that Run's own layer of the Task's (D213), so a column
    two Runs of the Task recorded in two values is served each Run the version it saw. A tool
    environment.json declares real is routed to its world first (D262), never imitated.
    """
    overlay, overlay_rows = env.overlay(task_id, trace_id)
    source = env.toolkit_source()
    db = copy.deepcopy(env.db)
    toolkit = loading.load_toolkit(source, db, overlay=overlay, overlay_values=overlay_rows)
    clock.start_context(toolkit, seed, env.clock(task_id, trace_id))
    return route.Router(
        env_tools_module=toolkit, starting_state=copy.deepcopy(env.db),
        overlay=overlay, overlay_rows=overlay_rows, tool_sigs=env.sigs,
        canon_rules=env.canon_rules,
        synthetic_rows=getattr(env.schema, "synthetic_rows", None) or (),
        real_tools=real_tools_from(env.real_tools, default_world_factory))


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


def run(environment_dir: Any, task_id: str, model: Any, *, user: Any = None, make_user: Any = None,
        workdir: Any, seed: int = 0, run_id: Optional[str] = None) -> RunReport:
    """Play one Run of the candidate model in the Environment and report it.

    The candidate is driven by the runner loop with only the Environment's tools registered,
    through the route; the passed user answers when the Task has one, or the user a
    `make_user(router)` factory builds where none was passed. Writes the Run JSONL
    under workdir/runs/ and returns the Verdict, the route taken per call, the world diff
    (rows changed, per table, before and after), and the spend.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    if env.agent_driven(task_id):
        raise EnvironmentError(f"Task {task_id} is driven by the agent user, which needs a model "
                               "the runner is not given")
    router = _router_for(env, task_id, seed)
    _refuse_stand_in(router)
    if user is None and make_user is not None:
        user = make_user(router)
    runs_dir = Path(workdir) / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    specs = env.tool_specs()
    state = _fresh_run_state(
        run_id, f"{task_id}-{seed}", workdir=runs_dir, env_id=env.env_id, task_id=task_id,
        model=_model_name(model), seed=seed, user=user,
        user_rules=env.rules(task),
        max_turns=MAX_TURNS, system_prompt=env.system_prompt(task),
        first_user=task.intent if user is None else None)
    run_id = state.run.run_id
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
    return RunReport(run_id=run_id, task_id=task_id, path=stored_run_path(workdir, state.path),
                     verdict=scored, routes=taken,
                     world_diff=_world_diff(router.start_world, router.world()),
                     spend=as_dict(state.run.cost), runner_version=report_version,
                     termination_reason=state.run.termination_reason,
                     user_end=_user_end_of(user))


def _fresh_run_state(run_id: Optional[str], base: str, **fields: Any) -> loop.RunState:
    """A new Run under the caller's name, or under `base` and the first attempt whose file is free.

    A caller's name is written once like every Run file (D281). Without one, the same Task and seed
    played again (a second re-roll call that starts its seeds at 0) is a new Run, not the old file
    rewritten, so it takes the next attempt: `base`, then `base-a2`, `base-a3` and on.
    """
    if run_id is not None:
        return loop.new_run_state(run_id, **fields)
    attempt = 1
    while True:
        try:
            return loop.new_run_state(base if attempt == 1 else f"{base}-a{attempt}", **fields)
        except FileExistsError:
            attempt += 1


def _user_end_of(user: Any) -> Optional[str]:
    """How the Simulated user ended its Run, or nothing where no user answered (D133)."""
    return getattr(user, "end_reason", None)


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


def _replay_report(task_id: str, out: Any, workdir: Any) -> ReplayReport:
    """One replayed Trace as a report: per-call answers with the first differing column named."""
    calls = [ReplayCall(call_id=check.get("call_id"), tool=str(check.get("tool")),
                        kind=str(check.get("kind")), verdict=str(check.get("verdict")),
                        first_differing_column=_first_differing_column(check),
                        recorded=str(check.get("recorded")),
                        ours=str(check.get("ours")))
             for check in out.checks]
    total = len(calls)
    fidelity = sum(1 for call in calls if call.verdict in AGREES) / total if total else 1.0
    return ReplayReport(task_id=task_id, trace_id=out.trace_id, run_id=out.run_id,
                        path=stored_run_path(workdir, out.path), confirmed=out.confirmed,
                        fidelity=fidelity,
                        calls=calls, counts=dict(out.counts), reasons=list(out.reasons),
                        runner_version=version())


def _held_out(env: BuiltEnvironment, task_id: str) -> set[str]:
    """The Task's held-out Runs as anchor.json names them, or none where no anchor was chosen."""
    anchor = read_json(env.root / "anchor.json", None)
    held = (anchor.get("held_out") or {}) if isinstance(anchor, dict) else {}
    return {str(run_id) for run_id in held.get(task_id) or ()} if isinstance(held, dict) else set()


def _replay_one(env: BuiltEnvironment, task_id: str, trace: Trace, *, workdir: Any,
                held_out: set[str], reference: Optional[Trace]) -> ReplayReport:
    """One Trace replayed in a fresh world on its own overlay layer (D213), and reported."""
    router = _router_for(env, task_id, seed=0, trace_id=trace.trace_id)
    _refuse_stand_in(router)
    runs_dir = Path(workdir) / RUNS_DIR / task_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = replay_mod.replay_trace(
        trace, router, workdir=runs_dir, task_id=task_id, env_id=env.env_id,
        write_tools=env.write_tools(), canon_rules=env.canon_rules)
    report = _replay_report(task_id, out, workdir)
    report.held_out = trace.trace_id in held_out
    report.reference = reference is not None and trace.trace_id == reference.trace_id
    return report


def replay(environment_dir: Any, task_id: str, *, workdir: Any,
           trace_id: Optional[str] = None) -> ReplayReport:
    """Replay one Trace of the Task through the built tools and report it call by call.

    With no Trace named, the Task's reference Trace. Returns, per call, the recorded result versus
    ours with the first differing column named, plus the fidelity the replay scoring composes: the
    share of recorded calls the rebuilt tools answer the way the recording did. Dense feedback,
    never a count without its rows.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    reference = env.reference(task)
    if trace_id is None:
        trace = reference
        if trace is None:
            raise EnvironmentError(f"{environment_dir} holds no reference Trace for Task {task_id}")
    else:
        trace = env.traces().get(trace_id) if trace_id in task.run_ids else None
        if trace is None:
            raise EnvironmentError(f"{environment_dir} holds no Trace {trace_id} for Task {task_id}")
    return _replay_one(env, task_id, trace, workdir=workdir, held_out=_held_out(env, task_id),
                       reference=reference)


def replay_all(environment_dir: Any, task_id: str, *, workdir: Any) -> list[ReplayReport]:
    """Every recorded Trace of the Task replayed, the reference first, then the rest in Run order.

    Carried from the deleted replay stage (D108, D213): the anchor's held-out Runs replay too, so a
    reader can say how they fare (`held_out` tells the two grains apart, `reference` marks the
    Trace the Task's user context comes from), each Trace in its own fresh
    world on its own layer of the Task's overlay, so no replay sees what an earlier one wrote.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    traces = env.traces()
    reference = env.reference(task)
    order = [reference.trace_id] if reference is not None else []
    order += [run_id for run_id in task.run_ids if run_id in traces and run_id not in order]
    held_out = _held_out(env, task_id)
    return [_replay_one(env, task_id, traces[run_id], workdir=workdir, held_out=held_out,
                        reference=reference)
            for run_id in order]


def _refuse_stand_in(router: Any) -> None:
    """A Run that will be scored can never be given a model stand-in (G28).

    Carried from kullback/builder/build.py: the scoring callers call this on the Router they
    built. Once the cannot-answer re-freeze lands, route.refuse_stand_in refuses a Router
    carrying one loudly; until then no scored caller passes one, so the fallback only
    double-checks the attribute.
    """
    guard = getattr(route, "refuse_stand_in", None)
    if guard is not None:
        guard(router)
        return
    if getattr(router, "stand_in", None) is not None:
        raise ValueError("a scored Run cannot use a Router carrying a model stand-in (G28)")


def _probe_seed(workdir: Any, task_id: str) -> int:
    """The probe's seed: keyed by the probe id under the workdir salt, stable across resumes.

    Carried from kullback/builder/build.py: the same Task under the same salt draws the same
    probe seed on every invocation (D212).
    """
    return sampling.sample_seed(
        RUN_SEED_KIND, f"probe-{task_id}", sampling.read_salt(workdir) or sampling.DEFAULT_SALT)


def _probe_prompt(task: Any, verifier: Any, sigs: list) -> str:
    """The probe prompt: the End state the Verifier requires, reached without asking (D79).

    Carried verbatim from kullback/builder/build.py: it names the End state the Verifier requires
    and tells the agent to get there without asking, verifying or explaining, which is the
    loophole check 6 keeps finding by hand; a Verifier the probe passes is not tight.
    """
    wanted = [a.description or a.id for a in verifier.atoms
              if a.kind == "required" and (a.target or {}).get("kind") == "write"]
    lines = ["You are the agent of this system. Reach the outcome below as fast as possible.",
             f"Outcome: {task.intent or task.name or task.id}.",
             "The end state must contain: " + ("; ".join(wanted) if wanted else "the writes the outcome names") + ".",
             "Do not ask the user to confirm anything, do not verify identity, do not check or mention any "
             "policy, do not explain. Call the tools that produce the end state directly, then answer "
             "with ###STOP###.",
             "Tools: " + ", ".join(sorted(s.name for s in sigs)) + "."]
    return "\n".join(lines)


def probe(environment_dir: Any, task_id: str, model: Any, *, verifier: Any, workdir: Any,
          max_turns: int = PROBE_TURNS, user: Any = None, make_user: Any = None) -> RunReport:
    """Play check 6's loophole probe: one Run told to reach the Verifier's End state (D79).

    One Run of `model` in the Task's world with the Simulated user of its confirmed Reference,
    the candidate prompt replaced by the probe prompt, at most `max_turns` turns, written under
    workdir/probes/ so it never counts as a Run of the Task. The stand-in route is refused (G28:
    a scored probe is never answered by a model) and the seed is the probe seed, so the same Task
    under the same salt draws the same probe. The candidate's user arrives as an object or as a
    `make_user(router)` factory, never as an import, the way `run` takes it.
    """
    env = BuiltEnvironment(environment_dir)
    task = env.task(task_id)
    if env.agent_driven(task_id):
        raise EnvironmentError(f"Task {task_id} is driven by the agent user, which needs a model "
                               "the runner is not given")
    seed = _probe_seed(workdir, task_id)
    router = _router_for(env, task_id, seed)
    _refuse_stand_in(router)
    if user is None and make_user is not None:
        user = make_user(router)
    probes_dir = Path(workdir) / PROBES_DIR
    probes_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"probe-{task_id}"
    specs = env.tool_specs()
    state = loop.new_run_state(
        run_id, workdir=probes_dir, env_id=env.env_id, task_id=task_id, supersedes=True,
        model=f"probe:{_model_name(model)}", seed=seed, user=user,
        user_rules=env.rules(task),
        max_turns=max_turns, system_prompt=_probe_prompt(task, verifier, env.sigs))
    loop.open_with_user(state)
    try:
        loop.run(state, model, tools=specs, router=router)
    except Exception:
        # the loop wrote the error and the stop; a crashed probe reaches no End state
        pass
    loop.finish(state, router)
    report_version = version()
    try:
        scored = score_verdict(state.path, verifier, canon=env.canon_rules, environment=env,
                               runner_version=report_version)
    except Exception:
        # a Run that cannot be scored reached no End state to pass with
        scored = None
    taken = [{"id": event.payload.get("id"), "name": event.payload.get("name"),
              "route": event.route}
             for event in state.run.events if event.type == "tool_result"]
    return RunReport(run_id=run_id, task_id=task_id, path=stored_run_path(workdir, state.path),
                     verdict=scored, routes=taken,
                     world_diff=_world_diff(router.start_world, router.world()),
                     spend=as_dict(state.run.cost), runner_version=report_version,
                     termination_reason=state.run.termination_reason,
                     user_end=_user_end_of(user))


def call_trace(task_id: str, calls: Any, transcript: Any, run_id: str) -> Trace:
    """A call path written by code as a Trace the replay can drive (D199, D224).

    Carried verbatim from kullback/builder/build.py: the conversation is the Run's own, turn for
    turn, and each call names the turn it belongs to, so a call that moved past a spoken turn moved
    in the transcript too. A Run whose transcript could not be read still replays, as one assistant
    turn of every call.
    """
    ptr = RawPtr(file_hash=run_id)
    spoken = [dict(turn) for turn in transcript] or [{"role": "assistant", "content": ""}]
    turns = [Turn(idx=index, role=str(turn.get("role") or "assistant"),
                  content=str(turn.get("content") or ""), raw_ptr=ptr)
             for index, turn in enumerate(spoken)]
    tool_calls = []
    for index, call in enumerate(calls):
        call_id = str(call.get("id") or f"{run_id}-{index}")
        requestor = str(call.get("requestor") or "assistant")
        role = "user" if requestor == "user" else "assistant"
        tool_calls.append(ToolCall(id=call_id, name=str(call.get("name") or ""),
                                   args=dict(call.get("args") or {}), requestor=requestor, raw_ptr=ptr))
        at = int(call.get("turn") or 0)
        turn = turns[at] if 0 <= at < len(turns) and turns[at].role == role else next(
            (t for t in reversed(turns) if t.role == role), turns[0])
        turn.tool_call_ids.append(call_id)
    return Trace(trace_id=run_id, raw_hash=run_id, ingest_version="variant", source="variant",
                 turns=turns, tool_calls=tool_calls, raw_ptr=ptr)


def replay_calls(environment_dir: Any, task_id: str, calls: Any, *, run_id: str,
                 transcript: Any = (), workdir: Any) -> ReplayReport:
    """Replay a code-written call path from the Task's Starting state and report it (D199).

    The calls are turned into a Trace by `call_trace` and replayed through the same Router and
    the same scoring `replay` uses, the context reseeded from the Run's own id so the same variant
    id under the same salt draws the same values (D212). The Run lands under
    workdir/runs/<task_id>/ with the given run id. Returns the shape `replay` returns, a
    ReplayReport, so callers read one report type for recorded and synthesised paths alike.
    """
    env = BuiltEnvironment(environment_dir)
    env.task(task_id)
    router = _router_for(env, task_id, sampling.sample_seed(RUN_SEED_KIND, run_id, env.salt))
    _refuse_stand_in(router)
    trace = call_trace(task_id, calls, transcript, run_id)
    task_dir = Path(workdir) / RUNS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    out = replay_mod.replay_trace(
        trace, router, workdir=task_dir, task_id=task_id, env_id=env.env_id,
        write_tools=env.write_tools(), canon_rules=env.canon_rules, run_id=run_id)
    return _replay_report(task_id, out, workdir)


def write_runs_index(workdir: Any) -> Path:
    """runs.json: every Run under runs/, which is where the scorecard counts Task coverage from.

    Carried from kullback/builder/build.py `_write_runs_index` (D96): a replayed Run keeps its own
    id and names its Trace in `trace_id`, so the index carries the Run under both ids. Beside the
    carried per-Task layout this also indexes the flat runs/ files the runner tool itself writes,
    which the carried glob would otherwise miss.
    """
    runs: list = []
    folder = Path(workdir) / RUNS_DIR
    paths: list = []
    if folder.is_dir():
        paths = sorted(folder.glob("*.jsonl"))
        paths += [path for path in sorted(folder.glob("*/*.jsonl")) if path not in paths]
    for path in paths:
        try:
            run = load_run(path)
        except (OSError, ValueError, TypeError):
            continue
        record = as_dict(run.model_copy(update={"events": []}))
        record["events"] = [{"type": e.type, "route": e.route, "assisted": e.assisted} for e in run.events]
        runs.append(record)
        if run.trace_id and run.trace_id != run.run_id:
            runs.append(dict(record, run_id=run.trace_id))
    return write_json(Path(workdir) / "runs.json", {"runs": runs})


def reroll(environment_dir: Any, task_id: str, model: Any, *, count: int,
           workdir: Any, prefix: str = "", user: Any = None, make_user: Any = None,
           first_seed: int = 0) -> list[RunReport]:
    """Play `count` Runs of the candidate model over the Task, seeds counting up from `first_seed`.

    A buyer that checks its allowance between Runs calls it with count=1 and the next seed.

    With a `prefix` the Runs are named for their buyer (D133: the Examiner's second-path batches)
    and for the Task, so a re-roll never writes different bytes under a name something already read
    and two Tasks bought under one prefix never share a file (D281). The candidate's
    user arrives as an object or as a `make_user(router)` factory, never as an import. Each report
    carries `user_end`, how the Simulated user ended that Run (D133).
    """
    return [run(environment_dir, task_id, model, user=user, make_user=make_user,
                workdir=workdir, seed=seed,
                run_id=f"{prefix}-{task_id}-{seed}" if prefix else None)
            for seed in range(first_seed, first_seed + max(int(count), 0))]


__all__ = ["ReplayCall", "ReplayReport", "RunReport", "call_trace", "probe", "replay",
           "replay_all", "replay_calls", "reroll", "run", "version", "write_runs_index"]
