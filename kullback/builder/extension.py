"""The Builder as an extension on the agent core (D120, D123, D135, D138, ADR-0007).

`builder_extension(plan)` is a `setup(api)` the harness loads: it registers the stage tools and
`status` from builder/tools.py and the five repair verbs beside them, adds four short sections to
the system prompt (what the job is, what may and may not be repaired, the tools by name, the target
vocabulary of this plan's graph), and installs two hooks. The
`tool_result` hook runs every registered gate bound to an artifact a tool produced (`rulings_over`)
over the plan's store and appends the rulings to the result, in the text the model reads and in
`details`; the stages already recorded the same rulings in gates.json on their way, so the hook
writes nothing. On a harness with a session the same hook protects a result carrying a failed
ruling until the artifact is produced again (D124). Verifiers and probes are the Examiner's: no
tool here writes either (D123). The `tool_call` hook raises on any call whose arguments name a path under
kullback/gates or kullback/runner, which the loop turns into an is_error result: the gates and
the Runner are packages no agent writes (D122), and the refusal is in code, not in the prompt.
"""

from __future__ import annotations

from typing import Callable, Optional

from kullback.agent.extensions import ExtensionAPI, refuse_paths
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult
from kullback.builder import build as build_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.builder.tools import builder_tools, repair_verb_tools
from kullback.gates import PROTECTED, names_protected_path, ruling_line, rulings_over
from kullback.runner.records import as_dict

WHAT = ("You are the Builder. From a customer's recorded traces you build an Environment: the tools "
        "with their bodies, the Starting state, the Tasks with a Reference Run each and the frontier's "
        "re-rolls, the Simulated user's rules, and the policy as Constraints. Your job is to make "
        "every gate pass. Every step is a stage of a fixed graph with a gate over what it made; the "
        "graph is there so that a target rebuilds whatever it reads that has gone stale, and which "
        "target to ask for is yours to choose. The Verifiers and the probes are the Examiner's, "
        "derived from what you leave; what it finds wrong on your side comes back to you as a finding.")
RULES = ("Repair only what a model wrote: the tool bodies, the policy predicates and the Intents. Never a gate, "
         "the Runner, the judge or the Simulated user, and any call naming a path under "
         "kullback/gates or kullback/runner is refused in code. You write no Verifier and no probe: "
         "there is no tool for either. The gates are the standard, not something to argue with; a "
         "failed ruling is reported as it is. Call status first and read the red lights. Stop when "
         "every gate is green, or when a round changes nothing.")
TOOLS = ("Tools: status() for the red lights and the verb that owns each; build(target) for any "
         "target of the graph; recluster(), grow(table, count), compile_tool(name), replay(task) and "
         "reroll(task) for one stage by name; and the repair verbs repair_recompile(name, hint), "
         "repair_grow(table, count), repair_intent(task_id, hint), repair_refuse_task(task_id, reason) "
         "and repair_escalate(task_id, queue). A Task the intent gate names is repaired with "
         "repair_intent, whose hint says what that Task's Runs actually evidence; refusing a Task "
         "records a decision and moves no gate.")


def target_vocabulary(plan: BuildPlan) -> str:
    """The names `build(target)` accepts for this plan: the whole build, each stage, each artifact."""
    stages = build_module.stages(plan)
    names = ", ".join(s.name for s in stages)
    artifacts = ", ".join(a for s in stages for a in s.outputs)
    return (f"Targets: `{TARGET_ALL}` is the whole build. Stages: {names}. Artifacts: {artifacts}. "
            "A target's stale inputs are rebuilt first; current ones come from the cache.")


# The tool_call hook (D122): the rule and the walk are the gates' own, the refusal is the core's.
no_agent_writes_gates_or_runner = refuse_paths(
    names_protected_path, f"a path under {' or '.join(PROTECTED)}, which no agent may write (D122)",
    "no_agent_writes_gates_or_runner")


def gate_rulings_hook(plan: BuildPlan, api: Optional[ExtensionAPI] = None) -> Callable[[ToolCall, ToolResult], Optional[ToolResult]]:
    """The tool_result hook: the registered gates over what the tool produced, and the context guard (D124).

    With an `api` over a harness that has a session, a result carrying a failed ruling is protected
    from every forget under the name of the artifact it produced, until a later call produces the
    same artifact again and releases it: an unacted ruling stays in front of the model. Without an
    api, or without a session (`entry_id_for` gives None), the hook appends the rulings and guards
    nothing, which is the code driver's path and byte-identical to before.
    """
    guards: dict[str, str] = {}

    def gate_rulings(call: ToolCall, result: ToolResult) -> Optional[ToolResult]:
        if result.is_error or not result.details:
            return None
        produced = list(result.details.get("produced") or [])
        rulings = rulings_over(plan.store, produced)
        entry = api.entry_id_for(call.id) if api is not None else None
        if entry is not None:
            for name in produced:
                earlier = guards.pop(name, None)
                if earlier is not None:
                    api.unprotect([earlier])
            failed = [r.stage for r in rulings if not r.passed]
            if failed:
                for name in produced:
                    guards[name] = entry
                api.protect([entry], f"unacted ruling: {', '.join(dict.fromkeys(failed))} on "
                                     f"{', '.join(produced)}")
        if not rulings:
            return None
        line = ruling_line("gate rulings", rulings)
        details = dict(result.details)
        details["gate_rulings"] = [as_dict(r) for r in rulings]
        return ToolResult(content=f"{result.content}\n{line}", details=details, is_error=False)

    gate_rulings.hook_name = "gate_rulings"  # type: ignore[attr-defined]
    return gate_rulings


def builder_extension(plan: BuildPlan) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: tools, prompt sections, the two hooks."""

    def setup(api: ExtensionAPI) -> None:
        for tool in [*builder_tools(plan, sink=api.harness.emit),
                     *repair_verb_tools(plan, sink=api.harness.emit)]:
            api.register_tool(tool)
        api.add_prompt_section("builder", WHAT)
        api.add_prompt_section("builder_rules", RULES)
        api.add_prompt_section("builder_tools", TOOLS)
        api.add_prompt_section("builder_targets", target_vocabulary(plan))
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_result(gate_rulings_hook(plan, api))

    return setup
