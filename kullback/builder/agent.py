"""`run_builder`: one Builder session over one workdir, driven by code or by a model (D120, D128, D135).

The harness is built with the Builder extension and one plan. Without an agent model the driver
issues the `build(target)` tool call itself, through the harness's registry and hooks and no model
turn, which is what `kullback build` does without `--agent`: the same stages, the same gates, the
same bytes as `build.build()`, plus the stage events on the harness's stream. That path is
unchanged, and it is what the byte-identical comparison rests on.

With an agent model (`kullback build --agent`) the model drives: it is sent one message asking it
to begin, and it calls tools until it answers with no tool call, which is tau's rule and the only
rule the loop keeps. There is no turn cap. The hard stop is the spend ceiling (D86): a guard on the
stream reads the workdir's spend after every tool call and cancels the run the moment it reaches
the ceiling, so a session cannot outspend the number a person gave it whatever the model does. The
soft stop is stalled (D126), which the round driver applies, not this module.

The build ends at the re-rolls: the Verifiers and the probes are the Examiner's
(`kullback.examiner.agent.run_examiner`, D123), which reads what this session leaves. A harness
given a `session` records its transcript there, the Builder's own file.

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
from kullback.builder.tools import BUILD_TOOLS, repair_verb_tools
from kullback.runner import budget

DRIVER_CALL_ID = "builder-driver"
CEILING_STAGE = "builder_session"


def build_harness(plan: BuildPlan, agent_model: Optional[Model] = None,
                  subscribers: Iterable[Callable[[Any], Any]] = (),
                  session: Optional[SessionStore] = None) -> AgentHarness:
    """A harness with the Builder extension over this plan; subscribers attached before any event, the
    session (when given) the Builder's own transcript, never the Examiner's (D128).

    No turn cap (D135): the run ends when the model answers with no tool call. What bounds it is the
    spend ceiling, which `ceiling_guard` watches, and the round's stalled exit (D126).
    """
    harness = AgentHarness(model=agent_model or DriverModel("Builder"), session=session)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    load_extensions(harness, [builder_extension(plan)])
    return harness


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = DRIVER_CALL_ID) -> ToolResult:
    """The core's `drive_tool` under the Builder driver's call id."""
    return core_drive_tool(harness, name, arguments, call_id)


def spent_in(workdir: Any) -> float:
    """What this workdir has paid for so far, off the one ledger budget.py writes."""
    return float(budget.load_totals(workdir)["total"].get("usd") or 0.0)


def ceiling_guard(harness: AgentHarness, plan: BuildPlan) -> tuple[Callable[[], None], dict]:
    """Cancel the run the moment the workdir's spend reaches the build's ceiling (D86, D135).

    The ceiling is the hard stop of a model-driven session: a stage that crosses it raises inside
    its own tool and the pipeline records the stop, but nothing stopped the model from calling the
    next tool, and with no turn cap that is an unbounded run. The guard reads the ledger after every
    tool call and cancels, so the tool that reached the ceiling is the last one. The dict it returns
    is empty until then and afterwards is the ceiling's own stop report, which is what the caller
    hands to a report (D86: stop where you are and say what it cost).
    """
    stop: dict = {}
    ceiling = plan.ceiling
    if ceiling is None:
        return (lambda: None), stop

    def watch(event: Any) -> None:
        if stop or not isinstance(event, ToolExecutionEnd):
            return
        spent = spent_in(plan.workdir)
        if spent < ceiling.usd:
            return
        ceiling.spent = max(ceiling.spent, spent)
        stop.update(ceiling.report(CEILING_STAGE, event.tool_name))
        harness.cancel()

    return harness.subscribe(watch), stop


def run_builder(workdir: Any, model: Any = None, target: str = TARGET_ALL, *, agent_model: Optional[Model] = None,
                files: Optional[list] = None, iterate: bool = False, ceiling_usd: Optional[float] = None,
                domain: str = "domain", max_attempts: int = 3, memory_dir: Any = None,
                grow: Optional[dict] = None, grow_seed: int = 0, probe_limit: Optional[int] = None,
                rerolls: int = DEFAULT_REROLLS, search: Any = None, workers: int = 1,
                on_event: Optional[Any] = None, emit: Optional[Any] = None,
                subscribers: Iterable[Callable[[Any], Any]] = ()) -> dict:
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
    harness = build_harness(plan, agent_model, subscribers)
    stop: dict = {}
    if agent_model is None:
        result = drive_tool(harness, "build", {"target": target})
    else:
        result, stop = _model_driven(harness, plan, target)
    # `plan.last` is set on execute's last line, so a call that raised leaves the previous run's
    # result behind: a model-driven session that built once and then failed would otherwise be
    # reported with the first run's status, env_id and rulings.
    if plan.last is None or (result is not None and result.is_error):
        raise BuildError(result.content if result is not None else NEVER_BUILT.format(target=target))
    out = build_module.result_of(plan.workdir, plan.last, plan.last.artifacts.get("environment"))
    out["target"] = target
    out["rulings"] = list(plan.last.rulings)
    out["stopped"] = out.get("stopped") or stop or None
    out["tool_result"] = {"content": result.content, "is_error": result.is_error} if result is not None else None
    return out


NEVER_BUILT = "the model never called build({target!r}) or any other tool that builds"


def builder_message(target: str) -> str:
    """The one message a model-driven Builder is sent: here and in the round driver's first beat."""
    return (f"Begin on the target {target!r}. Call status first and read the red lights, then use the "
            "tools to make every gate pass. Answer with no tool call when every gate is green, or "
            "when nothing is changing.")


NOTHING_CHANGED = ("nothing changed since the last round: no gate count moved. Every stage of a build "
                   "over a graph nothing moved is served from the cache, so calling build or replay "
                   "again without a repair returns that same cached result.")


def nothing_changed_message(plan: BuildPlan) -> str:
    """The follow-up a model-driven Builder is sent when its round changed nothing (D126, D135).

    Build 9's round 2 answered each finding with replay and build, was served every stage from the
    cache, and moved no count: the beat had no way of knowing that, since a cached build reads like
    any other. So the message says the plain fact first and then names the verbs of this session that
    can change an artifact, each with what it changes. They come from the session's own registry
    (`tools.repair_verb_tools`), so a verb this worktree does not have is never named at a model.
    Finishing is still the way out when no verb answers what is left: stalled is the soft stop.
    """
    verbs = "; ".join(f"{tool.name} ({tool.description.rstrip('.')})"
                      for tool in repair_verb_tools(plan))
    return (f"{NOTHING_CHANGED} What changes an artifact is a repair verb: {verbs}. Call status to see "
            "which red light is yours and the verb that answers it, and call that verb. If no verb "
            "answers what is left, finish with what you have.")


def _model_driven(harness: AgentHarness, plan: BuildPlan, target: str) -> tuple[Optional[ToolResult], dict]:
    """One opening message and then the model, until it answers with no tool call or the ceiling stops it.

    The result handed back is the last one from a tool that ran the graph, whichever verb it was, so
    a session that repaired without ever calling `build` is still a build.
    """
    last: Optional[ToolResult] = None
    unsubscribe, stop = ceiling_guard(harness, plan)

    async def go() -> None:
        nonlocal last
        async for event in harness.prompt(builder_message(target)):
            if isinstance(event, ToolExecutionEnd) and event.tool_name in BUILD_TOOLS:
                last = event.result

    try:
        asyncio.run(go())
    finally:
        unsubscribe()
    return last, stop

