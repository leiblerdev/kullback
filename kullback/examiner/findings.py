"""The findings the round's records file by themselves, before a model chooses what to say (D170).

Over three model-driven builds the Examiner filed 13 findings and 11 of them said the same thing:
an Intent phrase with no grounded span, `repair_intent`. The Builder followed nearly all of them
and no Task became trusted, because the losses that decide the number were never named: the
assisted seed tool that blocked 52 Tasks with no Reference, the D79 check that failed on most
Tasks with one, the Verifier whose required atoms reject every held-out Run, the Task whose own
recordings do not agree on an End state, the Task no Trace of which replays to its End state. One
build filed nothing at all in three rounds while 80 of 119 Tasks had no Reference, and another
labelled one finding of forty-six fidelity while the replay ruling was what blocked fifty-eight
Tasks. A finding the model chooses to write is a finding the model may not write; these five are
read off task_status.json, gates.json, replays.json and references.json by code, so they exist
whether or not there is a model and whether or not it says anything.

They are filed by the derive stage, which is the first call of every Examiner beat and the only
call of a code-driven one: the list is complete before the model's first choice. They are ranked by
the Tasks they cost, most first, which is the order the round driver hands them to the Builder in,
so the Builder's round opens on the tool that blocks fifty Tasks and not on the fifty-first Intent.
A finding whose key is already open is not filed again; once the Builder has answered it and the
loss is still there, the next round files it as a new finding, which is how a repair that did not
work stays visible.

Nothing here decides anything a gate has not already decided: every row is read back out of the
rulings and the status the derivation wrote (D66, D122).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.examiner.plan import ExaminerPlan
from kullback.gates import verifier_suite
from kullback.gates.fidelity import unconfirmed_reason
from kullback.gates.loosening import discarded_runs, false_rejection, legitimate_runs, over_strict
from kullback.gates.probes import write_tools_of
from kullback.runner.records import Finding, as_dict, read_json

# How much of a gate's own failure line a rule quotes. The line names the leaf and both values
# (D154) and is the whole of what the next body has to answer, but a keys-differ failure over a
# wide table runs to thousands of characters, and a lesson that long is not a lesson. Only the
# rules cut: a hint the model wrote is its own sentence and is kept whole.
HINT_CHARS = 400
# How many rule findings one round files. Each one reaches the Builder as its own follow-up message
# and costs it a turn, so a round that files one per Task spends its whole allowance restating the
# list instead of working it. The list is ranked before it is cut, so what is dropped is always
# what costs the fewest Tasks, and the next round files it once the ones above it are closed.
RULE_FINDING_LIMIT = 10

# Which verb answers which D79 check. Two of the nine have an owner outside the Verifier: the leak
# check fails on a value that reached the Intent, which is a line the Builder's model wrote and
# `repair_intent` rewrites; the rest are the Verifier's own atoms, too weak or too tight, and the
# Examiner's `repair` is the only verb that touches a Verifier (D123). Naming a Builder verb for
# those would send the Builder to rewrite an artifact that is not the one at fault, which is the
# mistake this whole file exists to stop.
SUITE_VERB: dict[str, str] = {
    "provenance_spans": "repair",       # an atom with no span in any Run: drop it
    "oracle_passes": "repair",          # the Reference fails its own Verifier: drop what rejects it
    "empty_fails": "repair",            # a Run that did nothing passes: the Verifier requires nothing
    "plausible_wrong_fails": "repair",  # the wrong entity passes: no atom names the entity
    "unsolved_state_fails": "repair",   # a Run cut short passes: no atom names the last write
    "second_path_passes": "repair",     # the second path is rejected: an atom is over-specific
    "loophole_probe_fails": "repair",   # the probe passes: the atoms leave it open
    "leak_check_clean": "repair_intent",  # the Intent holds a value only the Verifier should know
    "mutation_flips": "repair",         # an atom nothing can fail: it names no value
}
# A check whose gate never ran is not a wrong Verifier, it is a missing input, and the verb that
# answers it buys the input rather than changing the Verifier. `second_path_passes` on a Task with
# one Reference has no second path to score (D173), and what buys one is the Examiner's own `reroll`
# followed by `derive`: the Builder's re-roll would fill the corpus without deriving over it, and
# the Verifier is the Examiner's to rewrite either way (D123). The loophole probe needs probe
# budget, which no verb buys, so it is reported with no verb rather than with a wrong one.
MISSING_INPUT_VERB: dict[str, str] = {"second_path_passes": "reroll_then_derive"}
# The check name each D79 gate stage reports under, inverted from the suite's own map, so a status
# row's `not_run` (which holds stage names) can be read against its `checks` (which holds these).
CHECK_OF_STAGE: dict[str, str] = dict(verifier_suite.D79_STAGES)


def _json(path: Any, default: Any) -> Any:
    """One record file the rules read; a missing or half-written one is the default.

    A build killed mid-write must leave the round with the findings it can read rather than an
    exception out of the derivation that was otherwise finished.
    """
    try:
        return read_json(path, default)
    except (OSError, ValueError):
        return default


# --- what makes two findings the same finding ------------------------------------------

def finding_key(kind: str, subject: str = "", task_id: str = "") -> str:
    """The kind and the thing it is about: a tool, a D79 check, a Task, or nothing.

    Two findings with one key say the same thing, whoever filed them. The key is opaque and is only
    ever compared, so a tool name and a check name share the middle slot without ambiguity: a
    finding of one kind is never about both.
    """
    return f"{kind}:{subject}:{task_id}"


def open_by_key(plan: ExaminerPlan) -> dict[str, str]:
    """The finding id of every open finding, by key; a row written before D170 has no key and is skipped."""
    return {row["key"]: row["finding_id"] for row in plan.store.get("findings", [])
            if isinstance(row, dict) and row.get("status") == "open" and row.get("key")}


def told_task_tools(plan: ExaminerPlan) -> set[tuple[str, str]]:
    """Every Task and tool pair a finding of this build already covers, whatever kind it was filed
    under and whether it is open or closed.

    A fidelity row about a Task and a tool says the same thing as the assisted_tool finding about
    that tool which lists the Task among the ones it costs, and two messages saying one thing cost
    the Builder two turns. The pair is what makes them the same, not the kind, so this is compared
    beside the key and not through it.

    Closed counts as told, not only open. A pair covered by an open finding and skipped for it would
    come back as news the round after, when the Builder has answered that finding and closed it, and
    the round would hand it the same loss under a second name; `cost_by_key` refuses that for a key
    and this refuses it for a pair.
    """
    pairs: set[tuple[str, str]] = set()
    for row in plan.store.get("findings", []):
        if not isinstance(row, dict) or not row.get("tool"):
            continue
        tasks = list(row.get("task_ids") or ([row["task_id"]] if row.get("task_id") else []))
        pairs.update((str(task), str(row["tool"])) for task in tasks)
    return pairs


def covered_pairs(row: dict) -> set[tuple[str, str]]:
    """The Task and tool pairs one row covers; empty for a row that names no tool."""
    return {(str(task), str(row["tool"])) for task in row.get("task_ids") or ()} if row.get("tool") else set()


def cost_by_key(plan: ExaminerPlan) -> dict[str, list[str]]:
    """The Tasks the last finding under each key said it cost, whatever became of that finding.

    This is what tells a loss that is still there from a loss that has changed. A round exits only
    when no finding is pending (D126), so filing the same loss again every round would keep the
    loop running forever over a build that has stopped moving: a code-driven build reached round
    454 on a three Task fixture before this rule existed. The answer is not to file it once and
    forget it, which would hide a repair that made things worse, but to file it again only when the
    Tasks it costs are a different set: fewer Tasks means the repair worked and the rest is new
    news, more means it did not, and the same set means the loop has said all it has to say and the
    stalled exit is the honest end (D170).
    """
    seen: dict[str, list[str]] = {}
    for row in plan.store.get("findings", []):
        if isinstance(row, dict) and row.get("key"):
            seen[row["key"]] = sorted(row.get("task_ids") or ([row["task_id"]] if row.get("task_id") else []))
    return seen


def file_finding(plan: ExaminerPlan, *, kind: str, text: str, key: str, suggested: str = "none",
                 hint: str = "", task_id: Optional[str] = None, task_ids: Iterable[str] = (),
                 tool: Optional[str] = None, run_id: Optional[str] = None,
                 about_entry_id: Optional[str] = None) -> Finding:
    """One finding into the plan's store and onto disk, numbered after the ones already there.

    The caller checks `open_by_key` first: filing is not where a duplicate is refused, because the
    model's tool has to answer a duplicate with the id of the finding that already says it, and the
    rules just skip it.
    """
    rows = plan.store.setdefault("findings", [])
    ids = sorted(task_ids)
    record = Finding(finding_id=f"finding-{len(rows) + 1}", task_id=task_id or (ids[0] if ids else None),
                     kind=kind, text=text, run_id=run_id, tool=tool, suggested=suggested,
                     hint=hint.strip(), about_entry_id=about_entry_id, round=plan.round,
                     status="open", task_ids=ids, key=key)
    rows.append(as_dict(record))
    plan.write_state()
    return record


# --- the five rules ---------------------------------------------------------------------

def assisted_tool_rows(status: dict, fidelity: dict) -> list[dict]:
    """One row per tool whose own differing calls leave Tasks with no Reference, the costliest first.

    D171 decides which Tasks a tool costs. Assisted is a corpus ruling: one recorded call the kept
    body answers differently is enough for it, and reading that as "every Task whose seed Trace
    calls this tool" put one tool's name on 51 Tasks of a live build that make no call it answers
    differently. Only the Tasks the status row lists in `blocking_tools` are counted here, which are
    the Tasks whose own recorded calls part from the body, and they are the Tasks a recompile buys.
    A tool that is assisted and blocks nobody is not a finding: the red light still stands in the
    Builder's status and the ranking would put it last anyway.

    The counts and the first difference come from `tool_fidelity.json` (`tools` is the corpus
    ruling per tool, `tasks` the per Task grain with the corpus gate's own words for each differing
    call), so the hint quotes the leaf and both values or the error class (D154) without anything
    here having to parse a gate's failure line.
    """
    per_tool = (fidelity or {}).get("tools") or {}
    per_task = (fidelity or {}).get("tasks") or {}
    blocked: dict[str, list[str]] = {}
    for task_id, row in sorted((status or {}).items()):
        if not isinstance(row, dict) or row.get("reference_confirmed"):
            continue
        for name in row.get("blocking_tools") or []:
            blocked.setdefault(str(name), []).append(task_id)
    rows = []
    for name, task_ids in blocked.items():
        corpus = per_tool.get(name) or {}
        calls, replayed = int(corpus.get("calls") or 0), int(corpus.get("replayed") or 0)
        detail = first_difference(per_task, name, task_ids)[:HINT_CHARS]
        callers = sum(1 for row in per_task.values() if isinstance(row, dict) and name in row)
        rows.append({
            "kind": "assisted_tool", "tool": name, "task_ids": task_ids, "task_id": None,
            "key": finding_key("assisted_tool", name),
            "suggested": "repair_recompile",
            "hint": detail or "a recorded call of this Task replays differently",
            "text": (f"{name} answers {len(task_ids)} Tasks' own recorded calls differently, so none of them "
                     f"replays to its End state and none has a Reference (D49, D171). It replays "
                     f"{replayed} of {calls} recorded calls of the corpus and {callers} Tasks call it."
                     + (f" First difference: {detail}" if detail else "")),
        })
    return rows


def first_difference(per_task: dict, name: str, task_ids: Iterable[str]) -> str:
    """The corpus gate's own words for the first call of these Tasks that the body answers differently.

    The per Task grain carries the reasons already worded by the gate that ruled on the call, so the
    hint the recompile is given is the same sentence the gate would have written about it.
    """
    for task_id in task_ids:
        reasons = ((per_task.get(task_id) or {}).get(name) or {}).get("reasons") or []
        if reasons:
            return str(reasons[0])
    return ""


def leak_rows(status: dict, task_ids: Iterable[str]) -> list[dict]:
    """One row per Task and column the D196 strip missed, plus one row for the Tasks that named none.

    The leak check is an audit of a strip that has already run (D196), so what it reports is a
    column the strip does not cover yet, and the column is the whole of the news: the Builder's next
    Intent for that Task has to be written without it, and the strip that reads the same column
    class will take it out by itself once it sees it. The key is the Task and the column, so two
    rounds finding the same column on the same Task are one finding (D192) while a second column on
    the same Task is a second one. A status row from before D196 names no column, and those Tasks
    keep the one grouped row the check always filed.
    """
    task_ids = list(task_ids)
    named = {task_id: [str(c) for c in (status.get(task_id) or {}).get("leak_columns") or []]
             for task_id in task_ids}
    rows = [{
        "kind": "intent_leak", "tool": None, "task_ids": [task_id], "task_id": task_id,
        "key": finding_key("intent_leak", column, task_id), "suggested": SUITE_VERB["leak_check_clean"],
        "hint": (f"the Intent of this Task states a value of {column} that no user said; write the "
                 f"line without it"),
        "text": (f"The D79 check leak_check_clean failed on {task_id}: its Intent states a value of "
                 f"the column {column} that no recorded user said, so the Simulated user would hand "
                 f"the Candidate something only the system knew."),
    } for task_id in sorted(named) for column in sorted(set(named[task_id]))]
    unnamed = sorted(task_id for task_id, columns in named.items() if not columns)
    if unnamed:
        rows.append({
            "kind": "suite", "tool": None, "task_ids": unnamed, "task_id": None,
            "key": finding_key("suite", "leak_check_clean"), "suggested": SUITE_VERB["leak_check_clean"],
            "hint": "the D79 check leak_check_clean fails on this Task",
            "text": (f"The D79 check leak_check_clean failed on {len(unnamed)} Tasks that have a "
                     f"Reference, so none of them has a trusted Verifier."),
        })
    return rows


def suite_rows(status: dict) -> list[dict]:
    """One row per D79 check that cost Tasks, the check that cost most first.

    A check is counted only for a Task that has a Reference and no passing Verifier: a Task with no
    Reference never reached the suite, and its loss is the assisted tool's or the corpus's. The
    checks that failed and the checks that never ran are separate rows with separate keys, because
    they have separate answers: a check whose gate never ran is a Task short of an input, and
    telling the Examiner to repair a Verifier over it is telling it to repair the wrong thing.

    The leak check is the one exception, and `leak_rows` files it per Task and column (D196).
    """
    failed: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for task_id, row in sorted((status or {}).items()):
        if not isinstance(row, dict) or not row.get("reference_confirmed") or row.get("verifier_passed"):
            continue
        not_run = {CHECK_OF_STAGE[stage] for stage in row.get("not_run") or [] if stage in CHECK_OF_STAGE}
        for check, ok in sorted((row.get("checks") or {}).items()):
            if ok:
                continue
            (missing if check in not_run else failed).setdefault(check, []).append(task_id)
    rows = leak_rows(status or {}, failed.pop("leak_check_clean", []))
    for check, task_ids in failed.items():
        verb = SUITE_VERB.get(check, "none")
        rows.append({
            "kind": "suite", "tool": None, "task_ids": task_ids, "task_id": None,
            "key": finding_key("suite", check), "suggested": verb,
            "hint": (f"the D79 check {check} fails on this Task's Verifier" if verb == "repair"
                     else f"the D79 check {check} fails on this Task"),
            "text": (f"The D79 check {check} failed on {len(task_ids)} Tasks that have a Reference, so none of "
                     f"them has a trusted Verifier."),
        })
    for check, task_ids in missing.items():
        verb = MISSING_INPUT_VERB.get(check, "none")
        why = verifier_suite.ALT_PATH_NOT_RUN if check == "second_path_passes" else "the input it needs is not there"
        rows.append({
            "kind": "suite", "tool": None, "task_ids": task_ids, "task_id": None,
            "key": finding_key("suite", check, "not_run"), "suggested": verb,
            "hint": "", "text": (f"The D79 check {check} never ran on {len(task_ids)} Tasks ({why}), so the "
                                 f"check counts as not passed and none of them is trusted. This is a missing "
                                 f"Run, not a wrong Verifier (D173)."),
        })
    return rows


def false_rejection_rows(store: dict) -> list[dict]:
    """One row per Task whose required atoms reject every held-out frontier Run (D133).

    The number itself is the loosening gate's (`false_rejection`), read here over the same store and
    the same pool the trusted ruling reads it over, the recordings the round's Reference rule
    discarded taken out (D173): a Verifier that rejects a Run which did not do the job is right, and
    counting that made the number say the opposite of what it means. The atom is what the gate does
    not record, so it is recomputed by scoring the first rejected Run against the atoms that did the
    rejecting. Hard atoms are left out for the same reason the gate leaves them out: a Run that broke
    the policy is rightly rejected.
    """
    write_tools = write_tools_of(store.get("sigs") or [])
    legitimate = legitimate_runs(store.get("replays") or {}, store.get("rerolls") or {},
                                 discarded_runs(store.get("task_status") or {}))
    canon_rules = store.get("canon_rules")
    task_runs = store.get("task_runs") or {}
    rows = []
    for verifier in sorted(store.get("verifiers") or [], key=lambda v: v.task_id):
        runs = task_runs.get(verifier.task_id, [])
        seen = false_rejection(verifier, runs, legitimate.get(verifier.task_id, set()), canon_rules, write_tools)
        if not over_strict(seen):
            continue
        run_id = seen["rejected_ids"][0]
        run = next((r for r in runs if r.run_id == run_id), None)
        required = verifier.model_copy(update={"atoms": [a for a in verifier.atoms if a.kind != "hard"]})
        atom = verifier_suite.check_run(required, run, canon_rules, write_tools=write_tools)[1] if run else None
        names_atom = f" Atom {atom} rejects it." if atom else ""
        rows.append({
            "kind": "false_rejection", "tool": None, "task_ids": [verifier.task_id],
            "task_id": verifier.task_id, "key": finding_key("false_rejection", "", verifier.task_id),
            "suggested": "repair", "run_id": run_id,
            "hint": (f"drop or loosen {atom}, which rejects Run {run_id}" if atom
                     else f"the required atoms reject Run {run_id}"),
            "text": (f"The Verifier's required atoms reject all {seen['held_out']} held-out frontier Runs of this "
                     f"Task, so it recognises no path but its own seeds (D133)." + names_atom),
        })
    return rows


def fidelity_rows(status: dict, fidelity: dict, replays: dict) -> list[dict]:
    """One row per Task no Trace of which replays to its End state, named as the fidelity loss it is.

    The label is the point. `replay_reference` is the ruling that blocks a Task whose recorded calls
    the bodies answer differently, and on one live build it blocked 58 Tasks while 1 of the round's
    46 findings was labelled fidelity: the loss reached the Builder under the name of whatever
    grading layer the Examiner happened to read, so the Builder worked the Verifiers of Tasks that
    had no Reference to derive one from. The rule reads the same records the reference replay gate
    reads (`replays.json`, per Task per Trace, with `confirmed` and the gate's own `reasons`), so
    nothing here decides anything a gate has not decided.

    The tool named is the Task's own blocker where the status row has one (D171's `blocking_tools`,
    which are the tools whose own differing calls cost this Task), otherwise the first tool
    `tool_fidelity.json` records a difference for on this Task. A Task the records name no tool for
    keeps the reason line and suggests nothing: a recompile of no tool is not a verb.
    """
    per_task = (fidelity or {}).get("tasks") or {}
    rows = []
    for task_id, traces in sorted((replays or {}).items()):
        traces = {trace: row for trace, row in (traces or {}).items() if isinstance(row, dict)} \
            if isinstance(traces, dict) else {}
        if not traces or any(row.get("confirmed") for row in traces.values()):
            continue
        row = (status or {}).get(task_id) or {}
        blocking = [str(name) for name in (row.get("blocking_tools") or [])]
        differing = sorted(name for name, seen in (per_task.get(task_id) or {}).items()
                           if isinstance(seen, dict) and seen.get("differing"))
        tool = (blocking or differing or [""])[0]
        reason = unconfirmed_reason(traces)[:HINT_CHARS]
        detail = first_difference(per_task, tool, [task_id])[:HINT_CHARS] if tool else ""
        rows.append({
            "kind": "fidelity", "tool": tool or None, "task_ids": [task_id], "task_id": task_id,
            "key": finding_key("fidelity", tool, task_id),
            "suggested": "repair_recompile" if tool else "none",
            "hint": detail or reason,
            "text": (f"No Trace of this Task replays to its End state, so it has no Reference and no "
                     f"Verifier can be derived from it: {reason}."
                     + (f" The tool blocking it is {tool}." if tool else " No tool is named as the blocker.")),
        })
    return rows


def disagreement_rows(status: dict, references: dict) -> list[dict]:
    """One row per Task whose own recordings do not settle on an End state.

    Two shapes, one loss: the recordings reach two End states and the residue judge could not tell
    them apart, or the judge failed every recording there was. Either way the corpus does not say
    what the Task's answer is, and no Verifier can be derived from it. A Task blocked by an assisted
    tool is left out: its recordings disagree because the tool replays differently, and refusing it
    would be refusing a Task the Builder can still fix. Blocked means blocked by its own differing
    calls (D171): a Task that merely calls an assisted tool the body answers correctly is not
    waiting on a recompile, and on one live build 51 of the 64 Tasks that read as blocked were that
    kind, their real reason sitting later in the same sentence and never reaching a finding.
    """
    rows = []
    for task_id, row in sorted((references or {}).items()):
        if not isinstance(row, dict) or row.get("references"):
            continue
        if ((status or {}).get(task_id) or {}).get("blocking_tools"):
            continue
        groups = [g for g in row.get("groups") or [] if isinstance(g, dict)]
        failed = row.get("failed") or {}
        recordings = row.get("recordings") or []
        judge_failed = bool(failed) and len(failed) >= len(recordings) and bool(recordings)
        if not (groups or judge_failed):
            continue
        # The End states are what the Builder has to look at, but a state over a wide table runs long
        # and the message is a lesson, not a dump; the same cut the gate hints take (D170).
        states = "; ".join(f"{g.get('label')}: {g.get('state')}" for g in groups)[:HINT_CHARS]
        rows.append({
            "kind": "reference_disagreement", "tool": None, "task_ids": [task_id], "task_id": task_id,
            "key": finding_key("reference_disagreement", "", task_id), "suggested": "repair_refuse_task",
            "hint": str(row.get("reason") or "the recordings do not settle on an End state"),
            "text": ("The Task's recordings do not settle on an End state, so it has no Reference: "
                     + (f"{len(groups)} End states, {states}." if groups
                        else f"the judge failed all {len(failed)} recordings.")),
        })
    return rows


# --- the whole pass ---------------------------------------------------------------------

def rule_rows(plan: ExaminerPlan) -> list[dict]:
    """Every loss the round's records show, ranked by the Tasks it costs, most first.

    The tie-break is the key, so the same records give the same order however the dicts were built.
    """
    status = plan.store.get("task_status") or {}
    fidelity = plan.store.get("tool_fidelity") or _json(plan.workdir / "tool_fidelity.json", {}) or {}
    references = _json(plan.workdir / "references.json", {}) or {}
    rows = (assisted_tool_rows(status, fidelity) + suite_rows(status)
            + false_rejection_rows(plan.store) + disagreement_rows(status, references)
            + fidelity_rows(status, fidelity, plan.store.get("replays") or {}))
    return sorted(rows, key=lambda row: (-len(row["task_ids"]), row["key"]))


def file_rule_findings(plan: ExaminerPlan, limit: int = RULE_FINDING_LIMIT) -> list[Finding]:
    """File what the records say, ranked, and only what the Builder has not already been told.

    Two skips. A key that is open is the finding the Builder has not answered yet, and a second copy
    of it would be a second message saying the same thing. A key that was answered and comes back
    costing the very same Tasks is the same news a second time: the repair did not move it, the
    round is stalled and the loop's own exit says so (`cost_by_key`). What does get filed again is
    the same loss over a different set of Tasks, because that is a number that moved.

    The cut is at the bottom of the ranked list, so a round that is over the limit drops the losses
    that cost the fewest Tasks and never the one that costs the most.
    """
    already = open_by_key(plan)
    told = cost_by_key(plan)
    covered = told_task_tools(plan)
    filed: list[Finding] = []
    for row in rule_rows(plan):
        if len(filed) >= limit:
            break
        if row["key"] in already or told.get(row["key"]) == sorted(row["task_ids"]):
            continue
        pairs = covered_pairs(row)
        # The pair skip is the fidelity rule's alone: a Task and tool another finding already names
        # is that finding's news. The other rules are ranked by the Tasks they cost and re-file when
        # that set moves (`cost_by_key`), which a pair cannot see.
        if row["kind"] == "fidelity" and pairs and pairs <= covered:
            continue
        covered |= pairs
        filed.append(file_finding(plan, **row))
    return filed
