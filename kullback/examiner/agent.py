"""`run_examiner`: one Examiner session over one workdir, driven by code or by a model (D120, D123, D128).

The harness is built with the Examiner extension, one plan and, when a session path is given, its
own session file: the Examiner's transcript is never the Builder's. Without an agent model the driver
issues `derive(target)` itself through the harness's registry and hooks and no model turn, which is
what the round driver does on a code-only beat: the same derivation, the same gates, the same bytes
the Builder's derive_verifier stage used to write. With an agent model the harness is sent one
message asking for the derivation and whatever the rulings call for, and the model drives; a
scripted `TestModel` in a test takes the same path.
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
from kullback.examiner.extension import examiner_extension
from kullback.examiner.plan import ExaminerPlan
from kullback.gates.trust import trusted_gate

DRIVER_CALL_ID = "examiner-driver"
MAX_TURNS = 8


class ExaminerError(RuntimeError):
    """The session left nothing to report: the driver's derive failed, or the model never derived."""


def examiner_harness(plan: ExaminerPlan, agent_model: Optional[Model] = None,
                     subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS,
                     session: Optional[SessionStore] = None) -> AgentHarness:
    """A harness with the Examiner extension over this plan; subscribers attached before any event."""
    harness = AgentHarness(model=agent_model or DriverModel("Examiner"), max_turns=max_turns, session=session)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    load_extensions(harness, [examiner_extension(plan)])
    return harness


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = DRIVER_CALL_ID) -> ToolResult:
    """The core's `drive_tool` under the Examiner driver's call id."""
    return core_drive_tool(harness, name, arguments, call_id)


def run_examiner(workdir: Any, *, inputs: dict, env_id: Optional[str] = None, agent_model: Optional[Model] = None,
                 probe_model: Any = None, judge_model: Any = None, run_probe: Any = None,
                 run_rerolls: Any = None, probe_limit: Optional[int] = None, anchor: Any = None,
                 subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS,
                 target: str = "all", session_path: Any = None, round: int = 0,
                 allowance_remaining: Optional[float] = None) -> dict:
    """Derive the Verifiers of `target` in `workdir` through the Examiner extension; the status and the rulings.

    `inputs` is the derivation's store (the Builder's, filtered through `inputs_from`; a store that
    names a tool body is refused). `agent_model` is the model that drives the session, None for the
    code driver. `session_path` is the Examiner's own session file. An error result from the driver,
    or a model that never called derive, is an ExaminerError.
    """
    plan = ExaminerPlan(workdir=Path(workdir), inputs=inputs, env_id=env_id, probe_model=probe_model,
                        judge_model=judge_model, run_probe=run_probe, run_rerolls=run_rerolls,
                        probe_limit=probe_limit, anchor=anchor, round=round, allowance_remaining=allowance_remaining)
    session = SessionStore.load(session_path) if session_path is not None else None
    harness = examiner_harness(plan, agent_model, subscribers, max_turns=max_turns, session=session)
    if agent_model is None:
        result = drive_tool(harness, "derive", {"target": target})
    else:
        result = _model_driven(harness, target)
    if result is None:
        raise ExaminerError(f"the model never called derive({target!r})")
    if result.is_error:
        raise ExaminerError(result.content)
    return summary_of(plan, result)


def summary_of(plan: ExaminerPlan, result: Optional[ToolResult]) -> dict:
    """What a session leaves: the status per Task, the trusted and refused Tasks, the rulings recorded."""
    plan.load_state()
    store = plan.store
    trusted = trusted_gate(store["task_status"], store["verifiers"], store["probes"], store["history"],
                           store["refusals"], store["task_runs"], store["replays"], store["rerolls"],
                           store["canon_rules"], store["sigs"])
    return {
        "status": "complete",
        "workdir": str(plan.workdir),
        "tasks": dict(store["task_status"]),
        "trusted": list(trusted.metrics.get("trusted", [])),
        "refused": sorted(store["refusals"]),
        "rulings": [r.stage for r in plan.last_rulings],
        "tool_result": ({"content": result.content, "is_error": result.is_error} if result is not None else None),
    }


def examiner_message(target: str = "all") -> str:
    """The one message a model-driven Examiner is sent: here and in the round driver's first beat."""
    return (f"Derive the Verifiers for target {target!r}: call the derive tool with target={target!r}, read "
            "the rulings, then probe, repair, refuse or send a finding as they tell you; answer with one "
            "line when nothing is left to do.")


def _model_driven(harness: AgentHarness, target: str) -> Optional[ToolResult]:
    """One prompt asking for the derivation; the last derive tool's result, or None when the model never called it."""
    last: Optional[ToolResult] = None

    async def go() -> None:
        nonlocal last
        async for event in harness.prompt(examiner_message(target)):
            if isinstance(event, ToolExecutionEnd) and event.tool_name == "derive":
                last = event.result

    asyncio.run(go())
    return last
