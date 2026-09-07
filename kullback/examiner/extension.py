"""The Examiner as an extension on the agent core (D120, D123, D124, ADR-0007).

`examiner_extension(plan)` is a `setup(api)` the harness loads: it registers the eight tools of
examiner/tools.py, adds four short sections to the system prompt (what the Examiner is, what it may
and may not do, which Builder verb a finding should suggest and with what hint, the Tasks of this
build), catalogs the probe skill loaded from the start, and installs the hooks. The `tool_call`
hook raises on any call whose arguments name a path the Examiner never
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
from kullback.agent.harness import prompt_block
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult, counted_ruling_line
from kullback.examiner.plan import ExaminerPlan
from kullback.examiner.skills import PROBE_SKILL, PROBE_SKILL_NAME
from kullback.examiner.tools import examiner_tools
from kullback.gates import PROTECTED, PROTECTED_PATH, first_string, rulings_over
from kullback.runner.records import as_dict

# What the Examiner never reads: the Builder's compiled side (D123). A path segment equal to one of
# these, bare or under any directory, is refused whatever tool the call is for.
FORBIDDEN_READS = ("bodies.json", "env", "sandbox", "environment.json", "db.json", "schema.json", "overlays", "tools")
_SEPARATORS = re.compile(r"[/\\]")

WHAT = ("You receive the rulings of a derivation: one Verifier per Task from its References through the "
        "D79 suite, with the checks that failed on each. You produce probes that try to break a Verifier, "
        "repairs of a Verifier's atoms, refusals of Tasks no frontier Run finished, re-rolls, and findings "
        "for the Builder that name the verb it should call. Gates rule on everything you make. You never "
        "edit the Environment: what you find wrong there is a finding, not a fix.")
TOOLS = ("Tools, one example call each.\n"
         "derive(target=\"all\"): every Task, or derive(target=\"task_1a2b\") for one; every round starts "
         "with it.\n"
         "read(kind=\"intent\", id=\"task_1a2b\"): one record as JSON; kinds are task, trace, intent, run, "
         "verifier, probes, task_status, gates, rerolls, replays, references. The intent record lists "
         "`ungrounded_phrases` and `run_coverage`.\n"
         "search(text=\"priority handling\", kinds=[\"intent\", \"run\"]): where one phrase appears across "
         "the records, without reading any of them. Kinds are trace, run, intent, task_status, verifier; "
         "it answers a count per kind and one line per match saying the record, the Task and where in it "
         "the phrase is. `regex=true` reads the text as a regular expression, `tool=\"renew_membership\"` "
         "narrows to one tool's calls, `task_id` to one Task. Search before you read: a count over every "
         "Task costs one call, and reading the records that hold it costs one call each.\n"
         "probe(task_id=\"task_1a2b\", bug_class=\"extra-field acceptance\", note=\"writes the change "
         "and also a refund the user never asked for\", events=[...]): a hand-written Run the Verifier "
         "should reject, kept in the Task's pool forever.\n"
         "repair(task_id=\"task_1a2b\", reason=\"the Verifier accepts a Run that never confirms the new "
         "date\", drop=[\"atom-3\"], add=[{...}]): a new Verifier version; the D79 suite, the pool and "
         "the loosening gate decide.\n"
         "refuse(task_id=\"task_1a2b\", reason=\"no frontier Run finishes it\"): admitted only when no "
         "frontier Run finished.\n"
         "reroll(task_id=\"task_1a2b\", count=2): more frontier Runs of a Task.\n"
         "finding(task_id=\"task_1a2b\", kind=\"fidelity\", text=\"the Intent says 'refund voucher'; both "
         "runs say 'store credit'\", suggested=\"repair_intent\", hint=\"the runs say 'store credit'; use "
         "those words\"): what is wrong on the Builder's side, as the call the Builder can make.")
EXAMPLES = ("Examples of a ruling and the call that answers it.\n"
            "1. `derive_verifier: task_1a2b: the D79 suite did not pass: plausible_wrong_fails` -> read the "
            "Verifier, then repair(task_id=\"task_1a2b\", reason=\"the required write atom names no "
            "value, so a write to the wrong record passes\", drop=[\"atom-2\"], add=[{...}]).\n"
            "2. `intent: task_9f3e: noun phrases with no span: refund voucher` -> "
            "read(kind=\"intent\", id=\"task_9f3e\"), then finding(task_id=\"task_9f3e\", kind=\"fidelity\", "
            "text=\"...\", suggested=\"repair_intent\", hint=\"the runs say 'store credit'; use those "
            "words\").\n"
            "3. `replay_reference: task_1a2b: update_booking write: differs` -> "
            "read(kind=\"replays\", id=\"task_1a2b\"), then finding(task_id=\"task_1a2b\", "
            "kind=\"fidelity\", text=\"...\", suggested=\"repair_recompile\", hint=\"update_booking "
            "returns the whole row; the recording returns total and status only\").\n"
            "4. `refuse: task_1a2b is not refused: a frontier Run finished` -> the Task stays; probe or "
            "repair its Verifier instead.\n"
            "5. A Verifier that passed the suite and every probe in its pool -> nothing to call for that "
            "Task.\n"
            "6. Grounding a claim across Tasks before filing it: search(text=\"priority handling\", "
            "kinds=[\"intent\", \"run\"]) answers `4 matches (intent 4, run 0)` and four lines, each an "
            "Intent whose text holds the phrase -> the phrase is in four Intents and in no Run of any "
            "Task, so finding(task_id=\"task_9f3e\", kind=\"fidelity\", text=\"the Intent of 4 Tasks "
            "says 'priority handling'; no Run of any of them says it\", suggested=\"repair_intent\", "
            "hint=\"the Runs say 'same day pickup'; use those words\"). Search first, then read the one "
            "record you are going to quote.")
RULES = ("Choosing. Derive first, every round. Act first on the Tasks with a confirmed Reference whose "
         "Verifier failed the suite, then on the Tasks with no Verdict, and say in each finding which "
         "Builder verb answers it. You never read a tool body, the Starting state, the schema, the "
         "Environment or the sandbox: any call naming one is refused, and so is any path under "
         "kullback/gates or kullback/runner. A probe stays in its pool forever; a repair is accepted only "
         "when the D79 suite, the pool and the loosening gate all pass; a refusal is admitted only when "
         "no frontier Run finished. When one check has rejected one Task's repairs twice in a session, a "
         "third against that check is refused: file a finding, refuse the Task, or reroll_then_derive. "
         "A finding's `kind` is one of assisted_tool, fidelity, "
         "reference_disagreement, suite, false_rejection, environment, other; the name of the ruling you "
         "are answering is taken as the kind it is about, and so is the name of a tool. The gates are "
         "the standard, not something to argue with; a failed ruling is reported as it is.")
FINDINGS = ("A finding names the Builder verb that answers it and the one line that verb needs. When a Task "
            "has no Verdict because its Intent says something no Run says, read the Intent "
            "(`read` with kind `intent`), which lists the phrases the intent gate refused in "
            "`ungrounded_phrases` and what each grounded phrase is evidenced by in `run_coverage`, and "
            "suggest `repair_intent` with a hint saying what the Task's Runs actually evidence. When a tool's "
            "body comes out different on replay, suggest `repair_recompile` with a hint naming the tool and "
            "the columns whose values differ. Suggest `replay`, `reroll` or `compile_tool` only when running "
            "the same thing again with nothing new to say is what you mean; those take no hint. "
            "`derive` has already filed the findings its own records show, ranked by the Tasks they cost: "
            "the assisted tools blocking Tasks with no Reference, the D79 checks that failed, the Verifiers "
            "that reject every held-out Run, the Tasks whose recordings disagree. File what those do not say. "
            "A finding whose kind and subject is already open is refused with the id of the one that holds "
            "it: act on that one instead of filing it again. Two of those verbs are yours and not the "
            "Builder's: `repair` on a Verifier, and `reroll_then_derive` on a check that had no second "
            "finished Run to score, which is your own `reroll` of the Task and then `derive` again.")
STOP = ("Stopping. Answer with one line and no tool call when every Task is trusted or refused, or when "
        "the rulings after your repairs and findings are the ones you already answered. Say which Tasks "
        "remain and what you filed for each.")


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
        line = counted_ruling_line("gate rulings", rulings)
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
        api.add_prompt_section("examiner", prompt_block("task", WHAT))
        api.add_prompt_section("examiner_tools", prompt_block("tools", TOOLS))
        api.add_prompt_section("skills", api.context.skills_section())
        api.catalog_skill(PROBE_SKILL_NAME, PROBE_SKILL, loaded=True)
        api.add_prompt_section("examiner_examples", prompt_block("examples", EXAMPLES))
        api.add_prompt_section("examiner_rules", prompt_block("rules", RULES))
        api.add_prompt_section("examiner_findings", prompt_block("findings", FINDINGS))
        api.add_prompt_section("examiner_tasks", prompt_block("tasks", task_vocabulary(plan)))
        api.add_prompt_section("examiner_stop", prompt_block("stop", STOP))
        repair_guard, gate_rulings = guard_hooks(plan, api)
        api.tool_call(examiner_reads_only_its_surface)
        api.tool_call(repair_guard)
        api.tool_result(gate_rulings)
        plan.unprotect = api.unprotect
        plan.entry_id_for = api.entry_id_for

    return setup
