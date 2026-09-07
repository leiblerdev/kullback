"""The findings the round's records file by themselves, before a model chooses what to say (D170).

Over three model-driven builds the Examiner filed 13 findings and 11 of them said the same thing:
an Intent phrase with no grounded span, `repair_intent`. The Builder followed nearly all of them
and no Task became trusted, because the losses that decide the number were never named: the
assisted seed tool that blocked 52 Tasks with no Reference, the D79 check that failed on most
Tasks with one, the Verifier whose required atoms reject every held-out Run, the Task whose own
recordings do not agree on an End state. One build filed nothing at all in three rounds while 80
of 119 Tasks had no Reference. A finding the model chooses to write is a finding the model may not
write; these four are read off task_status.json, gates.json and references.json by code, so they
exist whether or not there is a model and whether or not it says anything.

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
from kullback.gates.loosening import false_rejection, legitimate_runs
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
# answers it buys the input rather than changing the Verifier. The second path needs a second
# finished Run of the Task, which is what a re-roll is; the loophole probe needs probe budget,
# which no verb buys, so it is reported with no verb rather than with a wrong one.
MISSING_INPUT_VERB: dict[str, str] = {"second_path_passes": "reroll"}
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


# --- reading a gate's failure line ------------------------------------------------------

def failure_subject(line: str) -> tuple[str, str]:
    """The name a gate failure is about and what the gate says about it.

    The tool gates write `name({arguments}): what differs` and `name: what differs`; anything else
    is a failure about the whole gate and belongs to no one name. Reading the name off the line is
    what lets a per-tool ruling be attributed at all: gates.json holds one row per tool per check
    and the row itself does not carry the tool's name.
    """
    head, separator, rest = line.partition("): ")
    if separator and "(" in head:
        return head.split("(", 1)[0].strip(), rest
    name, separator, rest = line.partition(": ")
    if separator and name and " " not in name:
        return name, rest
    return "", line


def tool_evidence(rulings: list[dict]) -> dict[str, dict]:
    """Per tool named in a failing ruling: its replayed and recorded call counts and its first failure.

    The calls come from the replay_fidelity rulings, whose metrics are that one tool's own split
    (the compile_tools stage rules per tool); a tool that ended assisted on another gate has no
    counts here and is reported without them rather than with someone else's.
    """
    out: dict[str, dict] = {}
    counted: set[tuple[str, str]] = set()  # one tool's split counted once, however many lines name it
    for row in rulings:
        if not isinstance(row, dict) or row.get("pass"):
            continue
        metrics = row.get("metrics") or {}
        stage, split = str(row.get("stage") or ""), str(metrics.get("split") or "")
        for line in row.get("failures") or []:
            name, detail = failure_subject(str(line))
            if not name:
                continue
            slot = out.setdefault(name, {"calls": 0, "replayed": 0, "detail": "", "stage": ""})
            if not slot["detail"]:
                slot["detail"], slot["stage"] = detail, stage
            if stage == "replay_fidelity" and (name, split) not in counted:
                counted.add((name, split))
                slot["calls"] += int(metrics.get("success_calls") or 0) + int(metrics.get("error_calls") or 0)
                slot["replayed"] += int(metrics.get("success_matches") or 0) + int(metrics.get("error_matches") or 0)
    return out


# --- the four rules ---------------------------------------------------------------------

def assisted_tool_rows(status: dict, rulings: list[dict]) -> list[dict]:
    """One row per assisted tool that leaves Tasks with no Reference, the tool it blocks most first.

    A Task whose seed Trace calls an assisted tool cannot be replayed to its End state, so it has no
    Reference and no Verdict (D49); the status row says which tools those are. The hint is the
    gate's own first failure line, which names the column and both values or the error class, since
    a hint written from the grouped line alone tells the compiler nothing it did not know.
    """
    blocked: dict[str, list[str]] = {}
    for task_id, row in sorted((status or {}).items()):
        if not isinstance(row, dict) or row.get("reference_confirmed"):
            continue
        for name in row.get("assisted_tools") or []:
            blocked.setdefault(str(name), []).append(task_id)
    evidence = tool_evidence(rulings)
    rows = []
    for name, task_ids in blocked.items():
        seen = evidence.get(name) or {}
        calls, replayed = seen.get("calls", 0), seen.get("replayed", 0)
        detail = seen.get("detail", "")[:HINT_CHARS]
        fidelity = (f" It replays {replayed} of {calls} recorded calls the way the recording did."
                    if calls else " No ruling names how its recorded calls replay.")
        first = f" First difference: {detail}" if detail else ""
        rows.append({
            "kind": "assisted_tool", "tool": name, "task_ids": task_ids, "task_id": None,
            "key": finding_key("assisted_tool", name),
            "suggested": "repair_recompile",
            "hint": detail or f"the {seen.get('stage') or 'compile_tools'} gate refused every body written for it",
            "text": (f"{name} is assisted, so {len(task_ids)} Tasks have no Reference: a seed Trace of each "
                     f"calls it and the replay cannot reach the End state (D49)." + fidelity + first),
        })
    return rows


def suite_rows(status: dict) -> list[dict]:
    """One row per D79 check that cost Tasks, the check that cost most first.

    A check is counted only for a Task that has a Reference and no passing Verifier: a Task with no
    Reference never reached the suite, and its loss is the assisted tool's or the corpus's. The
    checks that failed and the checks that never ran are separate rows with separate keys, because
    they have separate answers: a check whose gate never ran is a Task short of an input, and
    telling the Examiner to repair a Verifier over it is telling it to repair the wrong thing.
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
    rows = []
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
        rows.append({
            "kind": "suite", "tool": None, "task_ids": task_ids, "task_id": None,
            "key": finding_key("suite", check, "not_run"), "suggested": verb,
            "hint": "", "text": (f"The D79 check {check} never ran on {len(task_ids)} Tasks: the input it needs "
                                 f"is not there, so the check counts as not passed and the Verifier is not "
                                 f"trusted. This is a missing Run, not a wrong Verifier."),
        })
    return rows


def false_rejection_rows(store: dict) -> list[dict]:
    """One row per Task whose required atoms reject every held-out frontier Run (D133).

    The number itself is the loosening gate's (`false_rejection`), read here over the same store the
    trusted ruling reads it over; the atom is what the gate does not record, so it is recomputed by
    scoring the first rejected Run against the atoms that did the rejecting. Hard atoms are left out
    for the same reason the gate leaves them out: a Run that broke the policy is rightly rejected.
    """
    write_tools = write_tools_of(store.get("sigs") or [])
    legitimate = legitimate_runs(store.get("replays") or {}, store.get("rerolls") or {})
    canon_rules = store.get("canon_rules")
    task_runs = store.get("task_runs") or {}
    rows = []
    for verifier in sorted(store.get("verifiers") or [], key=lambda v: v.task_id):
        runs = task_runs.get(verifier.task_id, [])
        seen = false_rejection(verifier, runs, legitimate.get(verifier.task_id, set()), canon_rules, write_tools)
        if not (seen["held_out"] >= 1 and seen["fraction"] == 1.0):
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


def disagreement_rows(status: dict, references: dict) -> list[dict]:
    """One row per Task whose own recordings do not settle on an End state.

    Two shapes, one loss: the recordings reach two End states and the residue judge could not tell
    them apart, or the judge failed every recording there was. Either way the corpus does not say
    what the Task's answer is, and no Verifier can be derived from it. A Task blocked by an assisted
    tool is left out: its recordings disagree because the tool replays differently, and refusing it
    would be refusing a Task the Builder can still fix.
    """
    rows = []
    for task_id, row in sorted((references or {}).items()):
        if not isinstance(row, dict) or row.get("references"):
            continue
        if ((status or {}).get(task_id) or {}).get("assisted_tools"):
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
    rulings = [row for row in (_json(plan.ledger.path, []) or []) if isinstance(row, dict)]
    references = _json(plan.workdir / "references.json", {}) or {}
    rows = (assisted_tool_rows(status, rulings) + suite_rows(status)
            + false_rejection_rows(plan.store) + disagreement_rows(status, references))
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
    filed: list[Finding] = []
    for row in rule_rows(plan):
        if len(filed) >= limit:
            break
        if row["key"] in already or told.get(row["key"]) == sorted(row["task_ids"]):
            continue
        filed.append(file_finding(plan, **row))
    return filed
