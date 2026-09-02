"""`run_builder`: one Builder session over one workdir, driven by code or by a model (D120, D128).

The harness is built with the Builder extension and one plan. Without an agent model the driver
issues the `build(target)` tool call itself, through the harness's registry and hooks and no model
turn, which is what `kullback build` does: the same stages, the same gates, the same bytes as
`build.build()`, plus the stage events on the harness's stream. With an agent model
(`kullback build --agent`) the harness is sent one message asking for `build(target)` and the model
drives; a scripted `TestModel` in a test takes the same path. Either way repair verbs are off:
there is no repair tool and the prompt says so.

Subscribers see every event the harness emits: the stage events the pipeline sends through the
extension's sink, and in the model-driven path the loop's own. `on_event` is the dict stream the
TUI reads, handed to the plan unchanged.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.agent.events import ToolExecutionEnd
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.loop import execute_tool_call
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult
from kullback.ai.provider import Model
from kullback.builder import build as build_module
from kullback.builder.build import DEFAULT_REROLLS, TARGET_ALL, BuildError, BuildPlan
from kullback.builder.extension import builder_extension

DRIVER_CALL_ID = "builder-driver"
MAX_TURNS = 8


class DriverModel(Model):
    """The model of a harness no model drives: any call on it is a bug, not a request."""

    name = "builder-driver"

    def query(self, messages: list[dict], tools: Optional[list[dict]] = None, config: Any = None) -> Any:
        raise RuntimeError("the Builder driver issues tool calls itself; no model turn was asked for")


def build_harness(plan: BuildPlan, agent_model: Optional[Model] = None,
                  subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS) -> AgentHarness:
    """A harness with the Builder extension over this plan; subscribers attached before any event."""
    harness = AgentHarness(model=agent_model or DriverModel(), max_turns=max_turns)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    load_extensions(harness, [builder_extension(plan)])
    return harness


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = DRIVER_CALL_ID) -> ToolResult:
    """One tool call through the harness's hooks and registry, with no model turn.

    `loop.execute_tool_call` is the call: the driver issues the build the same way the loop would,
    so the hook order, what a raising hook does and the two execution events are the core's, stated
    in one place rather than copied here.
    """
    call = ToolCall(id=call_id, name=name, arguments=dict(arguments))
    return asyncio.run(execute_tool_call(call, harness.registry, harness.hooks, harness.emit))


def run_builder(workdir: Any, model: Any = None, target: str = TARGET_ALL, *, agent_model: Optional[Model] = None,
                files: Optional[list] = None, iterate: bool = False, ceiling_usd: Optional[float] = None,
                domain: str = "domain", max_attempts: int = 3, memory_dir: Any = None,
                grow: Optional[dict] = None, grow_seed: int = 0, probe_limit: Optional[int] = None,
                rerolls: int = DEFAULT_REROLLS, search: Any = None, workers: int = 1,
                on_event: Optional[Any] = None, emit: Optional[Any] = None,
                subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS) -> dict:
    """Build `target` in `workdir` through the Builder extension; the dict `build.build()` returns, plus the rulings.

    `model` is the Builder's model for the stages that call one (the plan's), `agent_model` the model
    that drives the session, None for the code driver. A tool result that is an error with nothing
    built is raised as a BuildError, so the CLI fails the way it did. The arguments are BuildPlan's,
    the same list `build.build()` takes, `emit` included: the extension puts its own sink in front of
    whatever a caller passes here, so a plan-level listener still sees every stage event.
    """
    plan = BuildPlan(workdir=Path(workdir), iterate=iterate, model=model, files=list(files or []),
                     ceiling_usd=ceiling_usd, domain=domain, max_attempts=max_attempts, memory_dir=memory_dir,
                     on_event=on_event, grow=grow, grow_seed=grow_seed, probe_limit=probe_limit, rerolls=rerolls,
                     search=search, workers=workers, emit=emit)
    harness = build_harness(plan, agent_model, subscribers, max_turns=max_turns)
    if agent_model is None:
        result = drive_tool(harness, "build", {"target": target})
    else:
        result = _model_driven(harness, target)
    # `plan.last` is set on execute's last line, so a call that raised leaves the previous run's
    # result behind: a model-driven session that built once and then failed would otherwise be
    # reported with the first run's status, env_id and rulings.
    if plan.last is None or (result is not None and result.is_error):
        raise BuildError(result.content if result is not None else f"the model never called build({target!r})")
    out = build_module._result(plan.workdir, plan.last, plan.last.artifacts.get("environment"))
    out["target"] = target
    out["rulings"] = list(plan.last.rulings)
    out["tool_result"] = {"content": result.content, "is_error": result.is_error} if result is not None else None
    return out


def _model_driven(harness: AgentHarness, target: str) -> Optional[ToolResult]:
    """One prompt asking for build(target); the last build tool's result, or None when the model never called it."""
    message = (f"Build the target {target!r}: call the build tool with target={target!r} and then answer "
               "with one line saying whether every ruling passed.")
    last: Optional[ToolResult] = None

    async def go() -> None:
        nonlocal last
        async for event in harness.prompt(message):
            if isinstance(event, ToolExecutionEnd) and event.tool_name == "build":
                last = event.result

    asyncio.run(go())
    return last

