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

from kullback.agent.context import ContextConfig
from kullback.agent.events import ToolExecutionEnd
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness, DriverModel
from kullback.agent.harness import drive_tool as core_drive_tool
from kullback.agent.session import SessionStore
from kullback.agent.tools import ToolResult
from kullback.ai.provider import Model
from kullback.examiner.extension import examiner_extension
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.tools import ATOM_SHAPE
from kullback.gates.trust import trusted_gate
from kullback.runner import budget

DRIVER_CALL_ID = "examiner-driver"
MAX_TURNS = 8

# The sentence both driver messages end on: a finding the Builder can act on names a verb and a hint.
# The system prompt says the same at length (examiner/extension.py, FINDINGS); this is the reminder in
# the message the model is actually answering, since build 11 filed every finding as `replay` and the
# Builder replayed a cached result three times.
FINDING_VERBS = ("A finding is only worth filing if it says what to do: suggest `repair_intent` with a hint "
                 "saying what a Task's Runs evidence when its Intent is ungrounded, `repair_recompile` with a "
                 "hint naming the tool and the differing columns when a body is wrong, and `replay`, `reroll` "
                 "or `compile_tool` only when running the same thing again is what you mean.")


class ExaminerError(RuntimeError):
    """The session left nothing to report: the driver's derive failed, or the model never derived."""


def context_config(model: Optional[Model]) -> ContextConfig:
    """The context settings of a session on this model: its own window, which the caller has to pass
    in because the agent core may not import `runner.budget` (D121, D124).

    Without it every harness ran on the 200,000 default, so the Examiner's fill line said "of 200000"
    on a model with twice that window and the floor fired at half the fill it was written for.
    """
    return ContextConfig(window=budget.window_for(getattr(model, "name", None)))


def examiner_harness(plan: ExaminerPlan, agent_model: Optional[Model] = None,
                     subscribers: Iterable[Callable[[Any], Any]] = (), max_turns: int = MAX_TURNS,
                     session: Optional[SessionStore] = None) -> AgentHarness:
    """A harness with the Examiner extension over this plan; subscribers attached before any event."""
    model = agent_model or DriverModel("Examiner")
    harness = AgentHarness(model=model, max_turns=max_turns, session=session,
                           context=context_config(agent_model))
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    load_extensions(harness, [examiner_extension(plan)])
    return harness


def drive_tool(harness: AgentHarness, name: str, arguments: dict, call_id: str = DRIVER_CALL_ID) -> ToolResult:
    """The core's `drive_tool` under the Examiner driver's call id."""
    return core_drive_tool(harness, name, arguments, call_id)


def run_examiner(workdir: Any, *, inputs: dict, env_id: Optional[str] = None, agent_model: Optional[Model] = None,
                 probe_model: Any = None, judge_model: Any = None, judge_agent: bool = False,
                 run_probe: Any = None,
                 run_rerolls: Any = None, probe_limit: Optional[int] = None, workers: int = 1,
                 anchor: Any = None,
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
                        judge_model=judge_model, judge_agent=judge_agent,
                        run_probe=run_probe, run_rerolls=run_rerolls,
                        probe_limit=probe_limit, workers=workers, anchor=anchor, round=round,
                        allowance_remaining=allowance_remaining)
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
            "line when nothing is left to do. " + FINDING_VERBS)


# The verbs a finding may suggest that only the Examiner can call: `repair` rewrites a Verifier and
# `reroll_then_derive` buys a check the second Run it never had. A finding suggesting one of these
# is addressed to the Examiner itself, and nothing was carrying it back: two live builds' Examiners
# suggested `repair` 33 and 18 times and called it once each. Named here rather than imported from
# the round driver, which imports this module and not the other way round.
OWN_VERBS: tuple[str, ...] = ("repair", "reroll_then_derive")
# How many of them one beat is handed. Each costs the beat a turn out of MAX_TURNS, so the list is
# ranked by the Tasks each finding costs and cut at a handful; what is dropped is always what costs
# the fewest Tasks, and the next beat is handed it once the ones above it are closed.
OWN_SUGGESTION_CAP = 5


def own_suggestions(findings: Iterable[Any], cap: Optional[int] = OWN_SUGGESTION_CAP) -> list[dict]:
    """The open findings that suggest a verb only the Examiner can call, costliest first, capped.

    `findings` are the rows of the store's findings file, dicts as they were written. A row that is
    not open, or suggests a verb the Builder owns, is not this list's business. A cap of 0 or None
    is no cap, which is what a count of them wants.
    """
    rows = [row for row in findings or ()
            if isinstance(row, dict) and row.get("status") == "open" and row.get("suggested") in OWN_VERBS]
    rows.sort(key=lambda row: (-len(row.get("task_ids") or []), str(row.get("finding_id") or "")))
    return rows[:cap] if cap else rows


def suggested_line(findings: Iterable[Any], cap: int = OWN_SUGGESTION_CAP) -> str:
    """The one line a beat is handed about its own open suggestions, or empty when there are none.

    The shape of `pending_line` on the Builder's side: what is owed, named, with the verb each one
    asks for, so the beat does not have to read its own transcript to find out what it wrote and
    never did. The payload shape is said once at the end, since a repair refused for the shape of an
    atom is a repair the round loses.
    """
    rows = own_suggestions(findings, cap)
    if not rows:
        return ""
    named = [f"{row.get('finding_id') or '?'} suggests {row.get('suggested')} on "
             f"{row.get('task_id') or ', '.join(row.get('task_ids') or []) or 'no Task'}"
             + (f" ({row.get('hint')})" if row.get("hint") else "") for row in rows]
    return (f"Still open and suggesting your own verbs: {'; '.join(named)}. Call the verb or say why "
            f"it is not worth calling. A repair takes {ATOM_SHAPE}.")


def examiner_round_message(round: int, target: str = "all", findings: Iterable[Any] = ()) -> str:
    """The steer a model-driven Examiner is sent from round 2 on: the round driver's later beats.

    It asks for what the driver requires. A beat keeps the result of the derive the model called,
    and a beat that never called derive fails the round, so the steer names that call rather than
    leaving the model to infer it: build 9's round 2 was steered to read the rulings and act, the
    model read them, filed a finding and answered in one line, and the round failed for the derive
    it was never asked for. The rest is the round-1 message's, since the work after the derivation
    is the same work every round.

    `findings` are the store's finding rows; the ones still open that suggest one of the Examiner's
    own verbs are named in the steer (`suggested_line`), which is what `pending_line` does for the
    Builder. A suggestion no beat ever acts on is a round's work thrown away.
    """
    owed = suggested_line(findings)
    return (f"round {round}: call the derive tool with target={target!r} again, read the rulings, then "
            "probe, repair, refuse or send a finding as they tell you; answer with one line when "
            "nothing is left to do. " + FINDING_VERBS + (f" {owed}" if owed else ""))


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
