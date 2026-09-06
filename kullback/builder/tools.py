"""The Builder's stages and repair verbs as tools of the agent core: pydantic arguments in, a short
result out (D120, D123, D135, D138).

Each acting tool runs one target of the build plan's graph through `build.execute`, so the
scheduler, not the model, decides what runs first: a tool whose inputs are stale gets them resolved
upstream, a tool whose inputs are current is served from the cache. The graph is kept for that and
for nothing else; which target to ask for is the model's call. `build(target)` is the general form,
where the target is `environment` (the whole build), a stage name or an artifact name; the others
are the verbs a Builder session reaches for by name. `recluster()` re-runs the clustering under its
fixed configuration, `grow(table, count)` grows one table of the Starting state with synthetic rows
(D107), `compile_tool(name)` recompiles one tool body, `replay(task)` replays one Task's Traces,
`reroll(task)` re-rolls one Task.

`status()` is where a round starts: it reads the red lights off what code wrote (gates.json, the
fidelity records in replays.json, the assisted tools in tool_builds.json) and never off a model,
and names for each the stage, the tool or Task, the failure in the gate's own words and the repair
verb that owns it. The repair verbs are `repair_recompile(name, hint)` and `repair_grow(table,
count)`, which record the request and then run the stage that repairs the artifact, and
`repair_refuse_task` and `repair_escalate` from `builder/repair.py`, which record a decision and
change no artifact. `repair_rewrite_skill` is deliberately not registered (the GEPA caution in
docs/todo.md).

A result is what the model reads plus what it does not. The rendered text is a few lines: the
status, the stages that ran, and the ruling names with pass or fail. Everything else (the Task
ids, the environment id, each stage's report) is in the result model and reaches the transcript's
`details`, which never enters the context.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, NoArgs
from kullback.builder import build as build_module
from kullback.builder import repair as repair_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.gates import Ruling, ruling_line, ruling_of
from kullback.gates.fidelity import unconfirmed_reason

Sink = Callable[[Any], Awaitable[None]]

# The tools that run the graph, so a driver can tell a build from a reading or a decision.
BUILD_TOOLS: tuple[str, ...] = ("build", "recluster", "grow", "compile_tool", "replay", "reroll",
                                "repair_recompile", "repair_grow")


class StageReport(BaseModel):
    """How one stage ended, from the pipeline's own report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    cached: bool = False
    attempts: int = 0
    rulings: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    produced: list[str] = Field(default_factory=list)


class BuildResult(BaseModel):
    """What one Builder tool produced: a summary, the stage gates' rulings pass or fail, and the payload apart.

    `stage_gates` is the ruling of each stage's own gate, the one that can roll a stage back, and
    `passed` rules over exactly those. Every ruling the run recorded through the ledger, which is
    the longer list, is in `payload["rulings"]`.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str
    target: str
    status: str
    passed: bool
    stage_gates: list[Ruling] = Field(default_factory=list)
    stages: list[StageReport] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    stopped: Optional[dict[str, Any]] = None
    payload: dict[str, Any] = Field(default_factory=dict)


def render(result: BuildResult) -> str:
    """The lines the model reads: status, stages, rulings. The payload stays in details."""
    lines = [result.summary]
    # status is already "cached" for a stage the cache served, so the flag would only say it twice.
    ran = [f"{s.name} ({s.status})" for s in result.stages if s.status != "pending"]
    if ran:
        lines.append("stages: " + ", ".join(ran))
    if result.stage_gates:
        lines.append(ruling_line("rulings", result.stage_gates))
    if result.failed_stage:
        lines.append(f"failed stage: {result.failed_stage}")
    if result.stopped:
        lines.append(f"stopped: {result.stopped.get('reason') or 'spend ceiling reached'}")
    return "\n".join(lines)


class BuildArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(default=TARGET_ALL,
                        description="`environment` for the whole build, or the name of one stage or one artifact.")


class GrowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="The table of the Starting state to grow.")
    count: int = Field(ge=1, description="How many rows the table should hold after growing (D107).")


class CompileToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The mined tool whose body to compile again.")


class ReplayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The Task whose Traces to replay through the built tools.")


class RerollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The Task to re-roll with the frontier model (D112).")


# --- the red lights: what is failing, and which verb owns it ------------------

# Which repair verb answers which ruling. The ruling names are the registry's (kullback/gates),
# so the map is over the harness's own vocabulary and holds for any customer's traces.
REPAIR_VERB_FOR: dict[str, str] = {
    "parses": "repair_recompile",
    "executes_on_s0": "repair_recompile",
    "deterministic": "repair_recompile",
    "non_trivial": "repair_recompile",
    "replay_fidelity": "repair_recompile",
    "refuses_unknown": "repair_recompile",
    "confined": "repair_recompile",
    "compile_tools": "repair_recompile",
    "compile_tools.parses": "repair_recompile",
    "compile_tools.executes": "repair_recompile",
    "compile_tools.deterministic": "repair_recompile",
    "compile_tools.non_trivial": "repair_recompile",
    "compile_tools.replay_fidelity": "repair_recompile",
    # A Task whose Traces do not replay to their End state is a tool that answers differently.
    "replay_reference": "repair_recompile",
    # A row the Traces name that the built world does not hold is a table to grow (D107).
    "build_environment": "repair_grow",
    # A Task the frontier cannot finish, or that has no grounded Intent, has no training signal.
    "rerolls": "repair_refuse_task",
    "intent": "repair_refuse_task",
    "derive_verifier": "repair_refuse_task",
}
DEFAULT_REPAIR_VERB = "repair_escalate"
NO_BODY = " has no body"


def verb_for(stage: str) -> str:
    """The repair verb that owns a ruling; anything the Builder cannot repair goes to a person."""
    return REPAIR_VERB_FOR.get(stage, DEFAULT_REPAIR_VERB)


class RedLight(BaseModel):
    """One failing gate as the model reads it: where, on what, in the gate's own words, and the verb."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    kind: str = "build"
    target: str = ""
    failure: str = ""
    verb: str = DEFAULT_REPAIR_VERB


class StatusResult(BaseModel):
    """Every red light the workdir holds, with the rulings that are passing beside them."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    red_lights: list[RedLight] = Field(default_factory=list)
    passing: list[str] = Field(default_factory=list)
    failing: list[str] = Field(default_factory=list)


def _read_json(path: Path, default: Any) -> Any:
    """One record file, or the default when it is missing or half written."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _named_by(failure: str) -> tuple[str, str]:
    """The tool or Task a failure names, and which of the two, from the gate's own wording.

    The gates write `task <id>: <reason>` for a Task, `<tool>: <reason>` for a tool and
    `<tool> has no body` for a tool with nothing compiled; anything else names neither. A sandbox
    ruling names the call rather than the tool (`get_order({"id": 1}): ...`), so the arguments are
    cut off: the verb takes a tool name, and the model would have to cut them itself otherwise.
    """
    text = failure.strip()
    if text.startswith("task "):
        return text[len("task "):].split(":", 1)[0].strip(), "task"
    if text.endswith(NO_BODY):
        return text[: -len(NO_BODY)].strip(), "tool"
    head, _, rest = text.partition(":")
    head = head.strip()
    if rest and head and " " not in head:
        return head.split("(", 1)[0].strip() or head, "tool"
    return "", "build"


def red_lights(workdir: Any) -> list[RedLight]:
    """Every failing gate this workdir holds, off the records code wrote and never off a model.

    Three files, because one is not enough: `gates.json` is every ruling the stages recorded, but
    the compile_tools stage overwrites it with its own per-tool rulings, so the fidelity records in
    `replays.json` are read for the Tasks whose replay ruling is no longer in the file, and
    `tool_builds.json` for the tools that ended assisted, which is a body the Builder could not
    write (D49) and the one red light no ruling names.
    """
    workdir = Path(workdir)
    out: list[RedLight] = []
    for row in _read_json(workdir / "gates.json", []) or []:
        if not isinstance(row, dict) or row.get("pass"):
            continue
        stage = str(row.get("stage") or "")
        failures = [str(f) for f in (row.get("failures") or [])] or [f"{stage} did not pass"]
        for failure in failures:
            target, kind = _named_by(failure)
            out.append(RedLight(stage=stage, kind=kind, target=target, failure=failure,
                                verb=verb_for(stage)))
    seen = {(light.stage, light.target) for light in out}
    for task_id, per_task in sorted((_read_json(workdir / "replays.json", {}) or {}).items()):
        rows = per_task if isinstance(per_task, dict) else {}
        if any(r.get("confirmed") for r in rows.values()) or ("replay_reference", task_id) in seen:
            continue
        out.append(RedLight(stage="replay_reference", kind="task", target=task_id,
                            failure=f"task {task_id}: {unconfirmed_reason(rows)}",
                            verb=verb_for("replay_reference")))
    for name, row in sorted((_read_json(workdir / "tool_builds.json", {}) or {}).items()):
        if isinstance(row, dict) and row.get("assisted"):
            out.append(RedLight(stage="compile_tools", kind="tool", target=name,
                                failure=f"{name} is assisted: no generated body cleared the gates (D49)",
                                verb="repair_recompile"))
    return out


def status_of(workdir: Any) -> StatusResult:
    """The red lights and the rulings that are passing, as the status tool returns them."""
    lights = red_lights(workdir)
    rows = [r for r in (_read_json(Path(workdir) / "gates.json", []) or []) if isinstance(r, dict)]
    passing = sorted({str(r.get("stage") or "") for r in rows if r.get("pass")})
    failing = list(dict.fromkeys(light.stage for light in lights))
    tools = len({light.target for light in lights if light.kind == "tool" and light.target})
    tasks = len({light.target for light in lights if light.kind == "task" and light.target})
    summary = (f"status: {len(lights)} red light{'' if len(lights) == 1 else 's'} over "
               f"{tools} tool{'' if tools == 1 else 's'} and {tasks} task{'' if tasks == 1 else 's'}; "
               f"{len(passing)} ruling{'' if len(passing) == 1 else 's'} passing")
    return StatusResult(summary=summary, red_lights=lights, passing=passing, failing=failing)


STATUS_LINES = 25


def render_status(result: StatusResult) -> str:
    """The summary, then one line per red light: the stage, what it is about, why, and the verb."""
    lines = [result.summary]
    for light in result.red_lights[:STATUS_LINES]:
        where = f" {light.target}" if light.target else ""
        lines.append(f"- {light.stage}{where}: {light.failure} -> {light.verb}")
    left = len(result.red_lights) - STATUS_LINES
    if left > 0:
        lines.append(f"and {left} more, all in details")
    if not result.red_lights:
        lines.append("nothing is failing; the gates are green")
    return "\n".join(lines)


def result_of(plan: BuildPlan, target: str, result: Any, verb: str) -> BuildResult:
    """A pipeline result as the tool's result model, gate rulings read from the run's own reports."""
    stages = [StageReport(name=r.name, status=r.status, cached=r.cached, attempts=r.attempts,
                          rulings=list(r.rulings), elapsed_ms=r.elapsed_ms, produced=list(r.produced))
              for r in result.reports.values()]
    stage_gates = [ruling_of(g) for g in result.gates]
    ran = [s for s in stages if s.status != "pending"]
    cached = sum(1 for s in ran if s.cached)
    summary = (f"{verb} {target}: {result.status}; {len(ran)} stages, {cached} from cache, "
               f"{len(result.rulings)} rulings recorded")
    environment = result.artifacts.get("environment")
    return BuildResult(
        summary=summary, target=target, status=result.status,
        passed=result.status == "complete" and all(r.passed for r in stage_gates),
        stage_gates=stage_gates, stages=stages,
        produced=[name for s in stages for name in s.produced],
        failed_stage=result.failed_stage, stopped=result.stopped,
        payload={"workdir": str(plan.workdir), "env_id": getattr(environment, "env_id", None),
                 "tasks": [t.id for t in result.artifacts.get("tasks", [])],
                 "rulings": list(result.rulings)})


def _bridged(previous: Optional[Callable[[Any], Any]], sink: Optional[Sink],
             loop: asyncio.AbstractEventLoop) -> Any:
    """The plan's stage events, sent on to the harness's stream from the thread the build runs on.

    The build runs on a worker thread so the event loop stays free; each typed event is handed to
    the sink on the loop and waited for, so the stream carries them in the order the stages emitted
    them rather than in the order the loop got round to them. A subscriber that raises still cannot
    stop a build: the exception travels back to `Pipeline._emit_typed`, which drops it, because a
    screen has no business failing a build that was going fine. A cancelled harness is the one case
    that guard cannot catch, since CancelledError is a BaseException, so the event is dropped here:
    a screen that has gone away is not a reason to fail the build that was writing to it.
    """

    def emit(event: Any) -> None:
        if previous is not None:
            previous(event)
        if sink is not None:
            try:
                asyncio.run_coroutine_threadsafe(sink(event), loop).result()
            except asyncio.CancelledError:
                return

    return emit


def _executor(plan: BuildPlan, sink: Optional[Sink], verb: str, target_of: Callable[[Any], str],
              narrowing_of: Callable[[Any], dict]) -> Callable[[Any], Awaitable[BuildResult]]:
    async def execute(args: Any) -> BuildResult:
        target, narrowing = target_of(args), narrowing_of(args)
        loop = asyncio.get_running_loop()
        previous = plan.emit
        plan.emit = _bridged(previous, sink, loop)
        try:
            result = await asyncio.to_thread(build_module.execute, plan, target, **narrowing)
        finally:
            plan.emit = previous
        return result_of(plan, target, result, verb)

    return execute


def _status_executor(plan: BuildPlan) -> Callable[[Any], Awaitable[StatusResult]]:
    """`status()`: the workdir's records read on a worker thread, so the event loop stays free."""

    async def execute(_args: Any) -> StatusResult:
        return await asyncio.to_thread(status_of, plan.workdir)

    return execute


def _repair_executor(plan: BuildPlan, sink: Optional[Sink], verb: str, target_of: Callable[[Any], str],
                     narrowing_of: Callable[[Any], dict], stage: str,
                     before: Optional[Callable[[Any], dict]] = None) -> Callable[[Any], Awaitable[BuildResult]]:
    """A repair verb that acts: the request is recorded, then the stage that repairs the artifact runs.

    `before` runs first and returns what else the request carries (the hint kept as a lesson), so a
    repair leaves a record of what was asked for as well as the artifact it produced.
    """
    run_stage = _executor(plan, sink, verb, lambda _a: stage, narrowing_of)

    async def execute(args: Any) -> BuildResult:
        extra = before(args) if before is not None else {}
        repair_module.record_request(plan.workdir, verb, target_of(args),
                                     {"arguments": args.model_dump(mode="json"), **extra})
        return await run_stage(args)

    return execute


def _keep_hint(plan: BuildPlan) -> Callable[[Any], dict]:
    """The hint a recompile carries, written to the Builder's memory as a lesson for that tool (D87)."""

    def keep(args: Any) -> dict:
        if not args.hint.strip():
            return {}
        repair_module.record_tool_lesson(plan.workdir, args.name, [args.hint.strip()])
        return {"lesson": args.hint.strip()}

    return keep


def repair_verb_tools(plan: BuildPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The four repair verbs a Builder session may call (D135, D138).

    Two act: they record the request and run the stage that repairs the artifact, so the gates rule
    on what came out in the same tool result. Two decide: `repair_refuse_task` and
    `repair_escalate` come straight from `builder/repair.py`, record their request and change no
    artifact. `repair_rewrite_skill` is not here: a model rewriting its own prompt stays proposed,
    gated and versioned, and no gate accepts an edit yet (the GEPA caution in docs/todo.md).
    """
    recording = {tool.name: tool for tool in repair_module.repair_tools(plan.workdir, sink)}
    return [
        AgentTool("repair_recompile",
                  "Repair one tool body: compile it again from its recorded calls, with a hint saying "
                  "what was wrong with the last one. The hint is kept as a lesson for that tool.",
                  repair_module.RecompileArgs, BuildResult,
                  _repair_executor(plan, sink, "repair_recompile", lambda a: a.name,
                                   lambda a: {"tools": [a.name]}, "compile_tools",
                                   before=_keep_hint(plan)), render=render),
        AgentTool("repair_grow",
                  "Repair the Starting state: grow one table to a row count with synthetic rows (D107).",
                  repair_module.GrowRepairArgs, BuildResult,
                  _repair_executor(plan, sink, "repair_grow", lambda a: a.table,
                                   lambda a: {"grow": {**dict(plan.grow or {}), a.table: a.count}},
                                   "starting_state"), render=render),
        recording["repair_refuse_task"],
        recording["repair_escalate"],
    ]


def builder_tools(plan: BuildPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The stage tools and `status` over one plan; `sink` is where the stage events go (`harness.emit`)."""
    return [
        AgentTool("status", "The red lights: every failing gate with the tool or Task it is about, why it "
                  "failed and the repair verb that answers it. Read off the records, never off a model.",
                  NoArgs, StatusResult, _status_executor(plan), render=render_status),
        AgentTool("build", "Build a target of the Environment: `environment` for everything, or one stage "
                  "or artifact by name; whatever it reads that is stale is rebuilt first.",
                  BuildArgs, BuildResult,
                  _executor(plan, sink, "build", lambda a: a.target, lambda a: {}), render=render),
        AgentTool("recluster", "Cluster the Runs into Tasks again under the fixed configuration.",
                  NoArgs, BuildResult,
                  _executor(plan, sink, "recluster", lambda a: "cluster", lambda a: {}), render=render),
        AgentTool("grow", "Grow one table of the Starting state to a row count with synthetic rows (D107).",
                  GrowArgs, BuildResult,
                  _executor(plan, sink, "grow", lambda a: "starting_state",
                            lambda a: {"grow": {**dict(plan.grow or {}), a.table: a.count}}), render=render),
        AgentTool("compile_tool", "Compile one tool's body again from its recorded calls, through the sandbox gates.",
                  CompileToolArgs, BuildResult,
                  _executor(plan, sink, "compile_tool", lambda a: "compile_tools",
                            lambda a: {"tools": [a.name]}), render=render),
        AgentTool("replay", "Replay one Task's Traces through the built tools (the Reference Runs, D108).",
                  ReplayArgs, BuildResult,
                  _executor(plan, sink, "replay", lambda a: "replay_reference",
                            lambda a: {"replay_tasks": [a.task]}), render=render),
        AgentTool("reroll", "Re-roll one Task with the frontier model inside the built Environment (D112).",
                  RerollArgs, BuildResult,
                  _executor(plan, sink, "reroll", lambda a: "rerolls",
                            lambda a: {"reroll_tasks": [a.task]}), render=render),
    ]
