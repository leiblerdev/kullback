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
from kullback.agent.harness import prompt_block
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult, counted_ruling_line
from kullback.builder import build as build_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.builder.tools import builder_tools, repair_verb_tools
from kullback.builder.triage import TRIAGE_SKILL, TRIAGE_SKILL_NAME
from kullback.gates import PROTECTED, names_protected_path, rulings_over
from kullback.runner.records import as_dict

WHAT = ("You receive a status report: every gate that failed, the tool or Task it failed on, one line "
        "saying why, and the verb that owns the fix. You produce tool calls that turn red lights green, "
        "and one line with no tool call when you are done. What you build is an Environment from a "
        "customer's recorded traces: the tool bodies, the Starting state, the Tasks with a Reference Run "
        "each and the frontier's re-rolls, the Simulated user's rules, the policy as Constraints, and one "
        "Intent per Task. Every step is a stage of a fixed graph with a gate over what it made; a target "
        "rebuilds whatever it reads that has gone stale, and what is current comes from the cache. The "
        "Verifiers and the probes are the Examiner's; what it finds wrong on your side comes back as a "
        "finding that names the verb to call and the hint that verb needs.")
TOOLS = ("Tools, one example call each.\n"
         "status(): the whole picture, grouped by gate, every tool and every Task id; status(gate=\"intent\") "
         "or status(target=\"lookup_account\") lists every red light there in full.\n"
         "build(target=\"environment\"): any target of the graph. Stale inputs rebuild first; a build after "
         "no repair returns the same rulings from the cache.\n"
         "repair_recompile(name=\"update_booking\", hint=\"the body returned the whole row; the recording "
         "returns only the changed fields\"): compile that tool's body again with the hint in the "
         "compiler's prompt. The result opens with that tool's own ruling, `cleared the gates` or "
         "`still assisted:` with the failure to answer next.\n"
         "repair_intent(task_id=\"task_1a2b\", hint=\"the runs say 'store credit', not 'refund voucher'; "
         "use the user's words\"): write that Task's Intent again with the hint. The result opens "
         "with that Task's own ruling, `grounded:` with the line it settled on or `still refused:` "
         "with the reason.\n"
         "repair_grow(table=\"accounts\", count=200) or grow(table=\"accounts\", count=200): grow one table "
         "of the Starting state with synthetic rows.\n"
         "compile_tool(name=\"update_booking\"), replay(task=\"task_1a2b\"), reroll(task=\"task_1a2b\"), "
         "recluster(): one stage by name, with no hint; a stage whose inputs are current comes from the "
         "cache.\n"
         "repair_refuse_task(task_id=\"task_1a2b\", reason=\"...\") and repair_escalate(task_id=\"task_1a2b\", "
         "queue=\"review\"): record a decision for the round report; they move no gate and change no "
         "artifact.")
EXAMPLES = ("Examples of a red light and the call that answers it.\n"
            "1. `replay_fidelity: update_booking (12): hard columns differ: total: ours 118.0, recorded 120.0` -> "
            "repair_recompile(name=\"update_booking\", hint=\"keep the total the recording shows; do not "
            "recompute it\").\n"
            "2. `executes_on_s0: price_quote({...}) raised NameError: name 'math' is not defined` -> "
            "repair_recompile(name=\"price_quote\", hint=\"no imports are available; call the "
            "evaluate_arithmetic helper the loaders bind, never a parser of your own\").\n"
            "3. `intent: noun phrases with no span: task_9f3e: what fails: refund voucher` -> "
            "status(target=\"task_9f3e\") to read the phrase and the runs, then "
            "repair_intent(task_id=\"task_9f3e\", hint=\"the runs say 'store credit'; use those words and "
            "drop 'refund voucher'\").\n"
            "4. `refuses_unknown: update_booking accepted booking_id='B-404', which the world does not "
            "hold` -> repair_recompile(name=\"update_booking\", hint=\"look the id up first and return the "
            "recording's not-found error when it is absent\").\n"
            "5. `derive_verifier: ... -> the Examiner owns it` -> nothing to call; leave it.\n"
            "6. A result that says `nothing changed: all 13 stages from cache` -> your last call changed "
            "no artifact; change the hint or the target, or stop.\n"
            "7. `repair_intent task_9f3e: still refused: ...` over `rulings: intent fail (12 Tasks, "
            "first task_0a1b: ...)` -> the first line is the Task you repaired and the second is the "
            "gate over every Task; a gate still red says nothing about the one you called.")
RULES = ("Choosing. Read the whole status once. Act first on the red light that blocks the most Tasks: a "
         "tool body many Tasks call before one Task's Intent. Send several repairs in one turn when they "
         "touch different tools or Tasks. A hint says what the evidence shows and what the last body or "
         "Intent got wrong, in one line, and never the same hint twice for the same target. Repair only "
         "what a model wrote: the tool bodies, the policy predicates and the Intents. Never a gate, the "
         "Runner, the judge or the Simulated user, and any call naming a path under kullback/gates or "
         "kullback/runner is refused in code. You write no Verifier and no probe: there is no tool for "
         "either. The gates are the standard, not something to argue with; a failed ruling is reported "
         "as it is.")
STOP = ("Stopping. Answer with no tool call, in one line, when every gate is green, or when two status "
        "reports in a row show the same red lights after your repairs, or when a repair answers "
        "`nothing changed`. Before that line, call build on the target once more, after your last "
        "repair: the Examiner reads what that build leaves, and a repair alone leaves it nothing. "
        "Say which red lights remain and what you tried on each.")


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
        line = counted_ruling_line("gate rulings", rulings)
        details = dict(result.details)
        details["gate_rulings"] = [as_dict(r) for r in rulings]
        return ToolResult(content=f"{result.content}\n{line}", details=details, is_error=False)

    gate_rulings.hook_name = "gate_rulings"  # type: ignore[attr-defined]
    return gate_rulings


def builder_extension(plan: BuildPlan) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: tools, prompt sections, the triage skill, the two hooks."""

    def setup(api: ExtensionAPI) -> None:
        for tool in [*builder_tools(plan, sink=api.harness.emit),
                     *repair_verb_tools(plan, sink=api.harness.emit)]:
            api.register_tool(tool)
        api.add_prompt_section("builder", prompt_block("task", WHAT))
        api.add_prompt_section("builder_tools", prompt_block("tools", TOOLS))
        api.add_prompt_section("skills", api.context.skills_section())
        api.catalog_skill(TRIAGE_SKILL_NAME, TRIAGE_SKILL, loaded=True)
        api.add_prompt_section("builder_examples", prompt_block("examples", EXAMPLES))
        api.add_prompt_section("builder_rules", prompt_block("rules", RULES))
        api.add_prompt_section("builder_targets", prompt_block("targets", target_vocabulary(plan)))
        api.add_prompt_section("builder_stop", prompt_block("stop", STOP))
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_result(gate_rulings_hook(plan, api))

    return setup
