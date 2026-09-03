"""The Examiner as an extension on the agent core (D120, D123, D124, ADR-0007).

`examiner_extension(plan)` is a `setup(api)` the harness loads: it registers the seven tools of
examiner/tools.py, adds three short sections to the system prompt (what the Examiner is, what it may
and may not do, the Tasks of this build), catalogs the probe skill loaded from the start, and installs
the hooks. The `tool_call` hook raises on any call whose arguments name a path the Examiner never
reads (a tool body, the db, the schema, the Environment, the sandbox, the overlays) or a path under
kullback/gates or kullback/runner (D122); the loop turns the raise into an is_error result. The
`tool_result` hook runs every registered gate bound to an artifact a tool produced over the plan's
store and appends the rulings, and it is where the context guards live (D124): a failed ruling
protects the result it came with until the next call on the same Task or target, a repair protects
the read it works from while it runs, and an open finding protects the entry it is about until the
Builder closes it. The guards are code over the session's entry ids; the model is told the reason
when a forget is refused and never asked to keep anything itself.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from kullback.agent.extensions import ExtensionAPI, refuse_paths
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.skills import PROBE_SKILL, PROBE_SKILL_NAME
from kullback.examiner.tools import examiner_tools
from kullback.gates import PROTECTED, PROTECTED_PATH, first_string, ruling_line, rulings_over
from kullback.runner.records import as_dict

# What the Examiner never reads: the Builder's compiled side (D123). A path segment equal to one of
# these, bare or under any directory, is refused whatever tool the call is for.
FORBIDDEN_READS = ("bodies.json", "env", "sandbox", "environment.json", "db.json", "schema.json", "overlays", "tools")
_SEPARATORS = re.compile(r"[/\\]")

WHAT = ("You are the Examiner. From the traces, the Intents and the frontier's re-rolls you derive one "
        "Verifier per Task and try to break it with probes; gates rule on everything you make. You never "
        "edit the Environment: what you find wrong there you report to the Builder as a finding.")
RULES = ("You may derive, probe, repair, refuse, reroll and file findings, and read the rulings that come "
         "back. You never read a tool body, the Starting state, the schema, the Environment or the sandbox: "
         "any call naming one is refused, and so is any path under kullback/gates or kullback/runner. A "
         "probe stays in its pool forever; a repair is accepted only when the D79 suite, the pool and the "
         "loosening gate all pass; a refusal is admitted only when no frontier Run finished. The gates are "
         "the standard, not something to argue with; a failed ruling is reported as it is.")


def task_vocabulary(plan: ExaminerPlan) -> str:
    """The Tasks of this build with what each has: a Reference, a Verifier, probes, a refusal."""
    status = plan.store.get("task_status") or {}
    pools = plan.store.get("probes") or {}
    refusals = plan.store.get("refusals") or {}
    lines = []
    for task in plan.inputs.get("tasks") or []:
        row = status.get(task.id) or {}
        parts = [f"{len(task.run_ids)} Runs"]
        if row:
            parts.append("Reference confirmed" if row.get("reference_confirmed") else "no Reference")
            parts.append("Verifier passed" if row.get("verifier_passed") else "no passing Verifier")
        if task.id in pools:
            parts.append(f"{len(pools[task.id].probes)} probes")
        if task.id in refusals:
            parts.append("refused")
        lines.append(f"{task.id}: " + ", ".join(parts))
    head = f"Tasks ({len(lines)}); `derive` takes `all` or one of these ids:"
    return head + ("\n" + "\n".join(lines) if lines else " none yet.")


def names_forbidden_path(value: Any) -> Optional[str]:
    """The first string in a value that names a path the Examiner never reads: the Builder's compiled side
    (a segment in FORBIDDEN_READS, D123) or the two packages no agent writes (the gates' own rule, D122)."""
    return first_string(value, lambda text: PROTECTED_PATH.search(text) is not None
                        or any(segment in FORBIDDEN_READS for segment in _SEPARATORS.split(text) if segment))


# The tool_call hook (D122, D123): the walk is the gates' own, the refusal is the core's.
examiner_reads_only_its_surface = refuse_paths(
    names_forbidden_path,
    "which the Examiner never reads: a tool body, the Starting state, the schema, the Environment, the sandbox, "
    f"or a path under {' or '.join(PROTECTED)} (D122, D123)",
    "examiner_reads_only_its_surface")


def _key_of(call: ToolCall) -> str:
    arguments = call.arguments or {}
    task_id = arguments.get("task_id")
    if task_id:
        return str(task_id)
    target = arguments.get("target")
    if target and target != "all":
        return str(target)
    return "derive"


def guard_hooks(plan: ExaminerPlan, api: Optional[ExtensionAPI]) -> tuple[Callable, Callable]:
    """The repair guard (a tool_call hook) and the gate rulings hook (a tool_result hook) over one set of guards.

    `guards` maps a Task or `derive` to the entry protected for an unacted ruling, and `('read',
    task)` to the entry of the last read of that Task. With no api, or a harness without a session
    (`entry_id_for` gives None), the hooks compute the rulings and protect nothing.
    """
    guards: dict[Any, str] = {}

    def entry_for(call_id: str) -> Optional[str]:
        return api.entry_id_for(call_id) if api is not None else None

    def repair_guard(call: ToolCall) -> None:
        if call.name != "repair" or api is None:
            return None
        task_id = str((call.arguments or {}).get("task_id") or "")
        entry = guards.get(("read", task_id))
        if entry is not None:
            api.protect([entry], f"repair in progress on task {task_id}")
        return None

    repair_guard.hook_name = "repair_guard"  # type: ignore[attr-defined]

    def gate_rulings(call: ToolCall, result: ToolResult) -> Optional[ToolResult]:
        task_id = str((call.arguments or {}).get("task_id") or "")
        if call.name == "repair" and api is not None:
            entry = guards.get(("read", task_id))
            if entry is not None:
                api.unprotect([entry])
        if result.is_error or not result.details:
            return None
        details = dict(result.details)
        entry = entry_for(call.id)
        if call.name == "finding":
            about = (details.get("finding") or {}).get("about_entry_id")
            if about and api is not None:
                api.protect([about], f"open finding {details.get('finding_id')}")
            return None
        if call.name == "read" and entry is not None:
            guards[("read", str((call.arguments or {}).get("id") or ""))] = entry
        produced = list(details.get("produced") or [])
        rulings = rulings_over(plan.store, produced)
        key = _key_of(call)
        if api is not None and entry is not None:
            earlier = guards.pop(key, None)
            if earlier is not None:
                api.unprotect([earlier])
            failed = [r.stage for r in rulings if not r.passed]
            failed += [r["stage"] for r in details.get("rulings") or [] if not r.get("passed")]
            if failed:
                api.protect([entry], f"unacted ruling: {', '.join(dict.fromkeys(failed))} on {key}")
                guards[key] = entry
        if not rulings:
            return None
        line = ruling_line("gate rulings", rulings)
        details["gate_rulings"] = [as_dict(r) for r in rulings]
        return ToolResult(content=f"{result.content}\n{line}", details=details, is_error=False)

    gate_rulings.hook_name = "gate_rulings"  # type: ignore[attr-defined]
    return repair_guard, gate_rulings


def gate_rulings_hook(plan: ExaminerPlan, api: Optional[ExtensionAPI] = None) -> Callable:
    """The tool_result hook alone, for a caller that wants the rulings without the repair guard."""
    return guard_hooks(plan, api)[1]


def examiner_extension(plan: ExaminerPlan) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: tools, prompt sections, the probe skill, the hooks, the plan's context calls."""

    def setup(api: ExtensionAPI) -> None:
        for tool in examiner_tools(plan, sink=api.harness.emit):
            api.register_tool(tool)
        api.add_prompt_section("examiner", WHAT)
        api.add_prompt_section("examiner_rules", RULES)
        api.add_prompt_section("examiner_tasks", task_vocabulary(plan))
        api.catalog_skill(PROBE_SKILL_NAME, PROBE_SKILL, loaded=True)
        repair_guard, gate_rulings = guard_hooks(plan, api)
        api.tool_call(examiner_reads_only_its_surface)
        api.tool_call(repair_guard)
        api.tool_result(gate_rulings)
        plan.unprotect = api.unprotect
        plan.entry_id_for = api.entry_id_for

    return setup
