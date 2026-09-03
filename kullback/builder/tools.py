"""The Builder's stages as tools of the agent core: pydantic arguments in, a short result out (D120, D123).

Each tool runs one target of the build plan's graph through `build.execute`, so the scheduler, not
the model, decides what runs first: a tool whose inputs are stale gets them resolved upstream, a
tool whose inputs are current is served from the cache. `build(target)` is the general form, where
the target is `environment` (the whole build), a stage name or an artifact name; the others are the
verbs a Builder session reaches for by name. `recluster()` re-runs the clustering under its fixed
configuration, `grow(table, count)` grows one table of the Starting state with synthetic rows
(D107), `compile_tool(name)` recompiles one tool body, `replay(task)` replays one Task's Traces,
`reroll(task)` re-rolls one Task. There is no repair verb: the Builder does not edit code (D122).

A result is what the model reads plus what it does not. The rendered text is a few lines: the
status, the stages that ran, and the ruling names with pass or fail. Everything else (the Task
ids, the environment id, each stage's report) is in the result model and reaches the transcript's
`details`, which never enters the context.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, NoArgs
from kullback.builder import build as build_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.gates import Ruling, ruling_line, ruling_of

Sink = Callable[[Any], Awaitable[None]]


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


def builder_tools(plan: BuildPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The six tools over one plan; `sink` is where the stage events go (the harness's `emit`)."""
    return [
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
