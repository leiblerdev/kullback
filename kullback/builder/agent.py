"""`run_builder`: one Builder session over one workdir, driven by code or by a model (D120, D128).

The harness is built with the Builder extension and one plan. Without an agent model the driver
issues the `build(target)` tool call itself, through the harness's registry and hooks and no model
turn, which is what `kullback build` does: the same stages, the same gates, the same bytes as
`build.build()`, plus the stage events on the harness's stream. With an agent model
(`kullback build --agent`) the harness is sent one message asking for `build(target)` and the model
drives; a scripted `TestModel` in a test takes the same path. Either way repair verbs are off:
there is no repair tool and the prompt says so. The build ends at the re-rolls: the Verifiers and
the probes are the Examiner's (`kullback.examiner.agent.run_examiner`, D123), which reads what this
session leaves. A harness given a `session` records its transcript there, the Builder's own file.

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
from kullback.agent.harness import AgentHarness, DriverModel
from kullback.agent.harness import drive_tool as core_drive_tool
from kullback.agent.session import SessionStore
from kullback.agent.tools import ToolResult
from kullback.ai.provider import Model
from kullback.builder import build as build_module
from kullback.builder.build import DEFAULT_REROLLS, TARGET_ALL, BuildError, BuildPlan
from kullback.builder.extension import builder_extension

DRIVER_CALL_ID = "builder-driver"
MAX_TURNS = 8


def build_harness(plan: BuildPlan, agent_model: Optional[Model] = None,
                  subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS,
                  session: Optional[SessionStore] = None) -> AgentHarness:
    """A harness with the Builder extension over this plan; subscribers attached before any event, the
    session (when given) the Builder's own transcript, never the Examiner's (D128)."""
    harness = AgentHarness(model=agent_model or DriverModel("Builder"), max_turns=max_turns, session=session)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    load_extensions(harness, [builder_extension(plan)])
    return harness


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = DRIVER_CALL_ID) -> ToolResult:
    """The core's `drive_tool` under the Builder driver's call id."""
    return core_drive_tool(harness, name, arguments, call_id)


def run_builder(workdir: Any, model: Any = None, target: str = TARGET_ALL, *, agent_model: Optional[Model] = None,
                files: Optional[list] = None, iterate: bool = False, ceiling_usd: Optional[float] = None,
                domain: str = "domain", max_attempts: int = 3, memory_dir: Any = None,
                grow: Optional[dict] = None, grow_seed: int = 0, probe_limit: Optional[int] = None,
                rerolls: int = DEFAULT_REROLLS, search: Any = None, workers: int = 1,
                on_event: Optional[Any] = None, emit: Optional[Any] = None,
                subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS) -> dict:
    """Build `target` in `workdir` through the Builder extension; the dict `build.build()` returns, plus the rulings.

    The build ends at the re-rolls; no Verifier is derived here (D123). `model` is the Builder's model
    for the stages that call one (the plan's), `agent_model` the model that drives the session, None
    for the code driver. A tool result that is an error with nothing
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
    out = build_module.result_of(plan.workdir, plan.last, plan.last.artifacts.get("environment"))
    out["target"] = target
    out["rulings"] = list(plan.last.rulings)
    out["tool_result"] = {"content": result.content, "is_error": result.is_error} if result is not None else None
    return out


def builder_message(target: str) -> str:
    """The one message a model-driven Builder is sent: here and in the round driver's first beat."""
    return (f"Build the target {target!r}: call the build tool with target={target!r} and then answer "
            "with one line saying whether every ruling passed.")


def _model_driven(harness: AgentHarness, target: str) -> Optional[ToolResult]:
    """One prompt asking for build(target); the last build tool's result, or None when the model never called it."""
    last: Optional[ToolResult] = None

    async def go() -> None:
        nonlocal last
        async for event in harness.prompt(builder_message(target)):
            if isinstance(event, ToolExecutionEnd) and event.tool_name == "build":
                last = event.result

    asyncio.run(go())
    return last

