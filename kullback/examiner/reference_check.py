"""Is the Task's Reference right? A view the Examiner judges, and the move that acts when it is not.

Every check of the D79 suite compares the Verifier with its own Reference, so a Reference that did
the wrong thing yields a Verifier that faithfully encodes the wrong End state and passes the suite.
Measured on one build, a derived Verifier agreed with the benchmark on 97 percent of recordings when
its Reference was right, and about a third of trusted Tasks rested on a Reference the benchmark
failed; the most common such Reference was a conversation that stopped short and wrote nothing
where the goal needed a write. Nothing in code can know the goal, but the Examiner holds the Intent
and the policy, so `check_reference` hands it the evidence in one view, with no model call: what the
user wanted, what the Reference wrote and what it read first, whether it changed the world at all,
whether the recordings agree, how the re-rolls fared against it, and the Builder's note if any.

`reject_reference` is what a judgement of "wrong" does (D111 continued). The recording is written
to one exclusion file in the exam folder with the reason and the evidence line, and Reference
choice runs again over the references.json row without it: the End state the recording reached is
the answer being rejected, so every Run that reached the same End state goes with it, and the D111
rule re-picks among what is left. A Task left with no End state to stand on waits in the pool: it
stays in the Environment, untrusted for want of a Reference, until a later Run reaches an End
state nobody rejected, which the normal rule picks and which takes the Task out of the pool.

A `builder_right` ruling on a note whose reason says the Reference's outcome is wrong calls the same
move (domain_tools.rule_note), so a note the Examiner agrees with changes the Reference instead of
only writing a ruling file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.examiner import audit as audit_mod
from kullback.examiner import reference as reference_mod
from kullback.examiner.exam_files import ExamRoot, reference_runs
from kullback.examiner.reference import EXCLUSIONS_FILE, excluded_runs, exclusions, exclusions_path
from kullback.gates import verifier_suite
from kullback.gates.probes import write_tools_of
from kullback.runner.canon import load_rules
from kullback.runner.records import EXAM_DIR, read_json, write_json

#: The note reasons that say the Reference's outcome is wrong, so agreeing with them rejects it.
REFERENCE_WRONG_NOTES = ("outcome_not_in_state", "intent_contradicts_reference")
REACHED = "reached"
WROTE_ELSE = "wrote something else"
WROTE_NOTHING = "wrote nothing"
_NOT_CHOSEN = "survivor "  # a failure that only says a survivor was not chosen, which re-picking reopens
ALL_REJECTED = "every End state was rejected or failed"
#: The status field that holds a Task out of the trusted count until it is derived again.
DERIVE_AGAIN = "derive_again"
REPICKED = "reference re-picked, derive again"


class CheckReferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str


class CheckReferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    view: dict = Field(default_factory=dict)


class RejectReferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str = Field(description="The Reference recording whose End state is wrong.")
    why: str = Field(description="One line: what the Intent and the policy needed that this End state lacks.")


class RejectReferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    task_id: str
    excluded: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    pooled: bool = False


# --- re-picking over a references.json row ----------------------------------------

def _eligible(group: dict, failed: dict, excluded: set[str]) -> list[str]:
    """The group's runs still standing: not excluded, and not failed by anything but a survivor choice."""
    return [run_id for run_id in group.get("runs") or ()
            if run_id not in excluded and str(failed.get(run_id, _NOT_CHOSEN)).startswith(_NOT_CHOSEN)]


def repick(row: dict, excluded: dict[str, str]) -> dict:
    """The references.json row with the excluded recordings out and the Reference chosen again (D111).

    The rule is the one `confirm` applies: the one End state left standing is the Reference. A
    survivor that lost only because another survivor was chosen stands again, since the choice it
    lost was made against the End state now rejected. Two or more left standing is a disagreement
    code cannot settle, so no Reference, the same as D111 without a judge.
    """
    out = dict(row)
    failed = dict(out.get("failed") or {})
    for run_id, why in excluded.items():
        failed[run_id] = f"rejected: {why}"
    kinds = {str(r.get("run_id")): r for r in out.get("recordings") or () if isinstance(r, dict)}
    # A row written without its groups still has its References, which share one End state.
    groups = out.get("groups") or [{"runs": [str(r.get("run_id")) for r in out.get("references") or ()]}]
    standing = [runs for runs in (_eligible(g, row.get("failed") or {}, set(excluded)) for g in groups) if runs]
    if len(standing) == 1:
        out["references"] = [dict(kinds.get(run_id) or {"run_id": run_id}) for run_id in standing[0]]
        out["reason"] = None
    else:
        out["references"] = []
        out["reason"] = (ALL_REJECTED if not standing else
                         f"recordings disagree on the End state ({len(standing)} states left after the rejection)")
    out["failed"] = failed
    out["rejected"] = sorted(excluded)
    return out


def _same_state(row: dict, run_id: str) -> list[str]:
    """The run and every Run that reached the same End state, which is the answer being rejected."""
    for group in row.get("groups") or ():
        if run_id in (group.get("runs") or ()):
            return list(group["runs"])
    return [run_id]


def _mark_derive_again(workdir: Path, task_id: str, root: Optional[ExamRoot]) -> None:
    """The Task's status row says its suite no longer passes, until the derivation runs again.

    The Verifier was derived from the Reference just rejected, so it encodes that End state, and a
    suite pass it earned against it is no evidence about the Reference chosen now. The next
    derivation writes the row afresh. With the session's root the existing helper writes both status
    files from what the session holds; a caller with no session merges both files on disk.
    """
    from kullback.examiner.domain_tools import _keep_suite_status  # the one status writer of the session

    row = {"verifier_passed": False, DERIVE_AGAIN: REPICKED}
    if root is not None:
        _keep_suite_status(root, task_id, row)
        return
    for path in (workdir / "task_status.json", workdir / EXAM_DIR / "task_status.json"):
        if not path.is_file():
            continue
        status = read_json(path, {}) or {}
        status[task_id] = {**(status.get(task_id) or {}), **row}
        write_json(path, status)


def reject_reference(workdir: Any, task_id: str, run_id: str, why: str, evidence: str = "",
                     root: Optional[ExamRoot] = None) -> dict:
    """Exclude the recording and every Run of its End state, re-pick, and pool when nothing is left.

    Returns what happened: the run ids excluded, the References now, and whether the Task now waits
    in the pool. The row in the exposed exam copy is written too, so the Examiner reads the Reference
    it now has, and the Task's status row leaves the trusted count until the Task is derived again.
    """
    root_dir = Path(workdir)
    references = read_json(root_dir / "references.json", {}) or {}
    row = references.get(task_id) or {}
    targets = _same_state(row, run_id)
    body = exclusions(root_dir)
    rows = [r for r in body.get(task_id) or [] if r.get("run_id") not in targets]
    for target in targets:
        line = evidence or (f"same End state as {run_id}" if target != run_id else "")
        rows.append({"run_id": target, "why": why, "evidence": line})
    body[task_id] = rows
    write_json(exclusions_path(root_dir), body)
    new_row = repick(row, excluded_runs(root_dir, task_id))
    references[task_id] = new_row
    write_json(root_dir / "references.json", references)
    exposed = root_dir / EXAM_DIR / "references.json"
    if exposed.is_file():
        copy = read_json(exposed, {}) or {}
        copy[task_id] = new_row
        write_json(exposed, copy)
    picked = [str(r.get("run_id")) for r in new_row["references"]]
    pooled = not picked and new_row["reason"] == ALL_REJECTED
    if pooled:
        reference_mod.join_pool(root_dir, task_id, why, targets)
    _mark_derive_again(root_dir, task_id, root)
    return {"excluded": targets, "references": picked, "pooled": pooled}


def reject_on_note(workdir: Any, task_id: str, note: dict, verdict: str, text: str,
                   root: Optional[ExamRoot] = None) -> Optional[dict]:
    """The move a `builder_right` ruling makes on a note saying the outcome is wrong, else None."""
    if verdict != "builder_right" or note.get("reason") not in REFERENCE_WRONG_NOTES:
        return None
    row = (read_json(Path(workdir) / "references.json", {}) or {}).get(task_id) or {}
    current = [str(r.get("run_id")) for r in row.get("references") or () if r.get("run_id")]
    if not current:
        return None
    evidence = f"Builder note {note.get('reason')}: {note.get('sentence')}"
    return reject_reference(workdir, task_id, current[0], text or str(note.get("sentence") or ""), evidence,
                            root=root)


# --- the view --------------------------------------------------------------------

def _fn(root: ExamRoot):
    rules = root.canon_rules if root.canon_rules is not None else load_rules(Path(root.workdir) / "canon-rules.json")
    return verifier_suite.canon_fn(rules)


def _intent(workdir: Path, task_id: str) -> str:
    record = read_json(workdir / "intents" / f"{task_id}.json", {}) or {}
    if record.get("text"):
        return str(record["text"])
    task = read_json(workdir / "tasks" / f"{task_id}.json", {}) or {}
    return str(task.get("intent") or task.get("name") or "")


def _load(root: ExamRoot, task_id: str, run_ids: Iterable[str]) -> list:
    workdir = Path(root.workdir)
    return reference_runs(workdir, {"references": [{"run_id": r} for r in run_ids]},
                          (root.replays or {}).get(task_id), (root.rerolls or {}).get(task_id) or [])


def _writes(run: Any, write_tools: set[str], fn: Any) -> list[dict]:
    """Each write the Run made with the reads that came before it."""
    calls = verifier_suite.run_calls(run)
    out = []
    for effect in verifier_suite.write_effects(run, write_tools, fn).values():
        reads = [c["name"] for c in calls if c["pos"] < effect["pos"] and c["name"] not in write_tools]
        out.append({"tool": effect["tool"], "entity": effect["entity"] or "", "reads_before": reads})
    return out


def _outcome(settled: tuple, target: tuple) -> str:
    if settled == target:
        return REACHED
    return WROTE_NOTHING if not settled else WROTE_ELSE


def reference_view(root: ExamRoot, task_id: str) -> dict:
    """The evidence on one Task's Reference, computed from its files, for the Examiner to judge."""
    from kullback.examiner.domain_tools import read_note, read_ruling  # domain_tools registers this module

    workdir = Path(root.workdir)
    row = (read_json(workdir / "references.json", {}) or {}).get(task_id) or {}
    refs = [str(r.get("run_id")) for r in row.get("references") or () if r.get("run_id")]
    groups = row.get("groups") or []
    view: dict = {"intent": _intent(workdir, task_id), "reference": refs[0] if refs else None,
                  "recordings": len(row.get("recordings") or ()),
                  "recordings_agree": len(groups) == 1, "end_states": len(groups),
                  "excluded": exclusions(workdir).get(task_id) or []}
    note = read_note(workdir, task_id)
    view["note"] = None if note is None else {"reason": note.get("reason"), "sentence": note.get("sentence"),
                                              "ruling": read_ruling(workdir, task_id)}
    loaded = _load(root, task_id, refs[:1])
    if not loaded:
        view["reason"] = row.get("reason") or "no Reference Run on disk"
        return view
    write_tools, fn = set(write_tools_of(root.sigs)), _fn(root)
    reference = loaded[0]
    calls = verifier_suite.run_calls(reference)
    writes = _writes(reference, write_tools, fn)
    view["writes"] = writes
    view["wrote_nothing"] = not any(c["name"] in write_tools for c in calls)
    view["end_equals_start"] = not writes
    target = reference_mod.settled_state(reference, write_tools, fn)
    outcomes: dict[str, str] = {}
    for reroll in (root.rerolls or {}).get(task_id) or []:
        if (reroll.get("termination_reason") or "") not in verifier_suite.SUCCESS_TERMINATIONS:
            continue
        for run in _load(root, task_id, [str(reroll.get("run_id"))]):
            outcomes[str(reroll.get("run_id"))] = _outcome(
                reference_mod.settled_state(run, write_tools, fn), target)
    view["rerolls"] = {kind: sorted(r for r, said in outcomes.items() if said == kind)
                       for kind in (REACHED, WROTE_ELSE, WROTE_NOTHING)}
    return view


def view_line(task_id: str, view: dict) -> str:
    """The view as one line a model reads first; the whole view rides in the result."""
    if view.get("reference") is None:
        return f"task {task_id}: no Reference ({view.get('reason')})"
    writes = view.get("writes") or []
    did = ("wrote nothing" if view.get("wrote_nothing") else
           "left the Starting state unchanged" if view.get("end_equals_start") else
           "wrote " + "; ".join(f"{w['tool']} on {w['entity'] or 'an entity'}" for w in writes))
    rerolls = view.get("rerolls") or {}
    counts = ", ".join(f"{len(rerolls.get(k) or [])} {k}" for k in (REACHED, WROTE_ELSE, WROTE_NOTHING))
    agree = "agree" if view.get("recordings_agree") else f"reach {view.get('end_states')} End states"
    line = (f"task {task_id}: Reference {view['reference']} {did}; {view.get('recordings')} recordings {agree}; "
            f"re-rolls against it: {counts}")
    note = view.get("note")
    if note:
        ruled = note.get("ruling")
        line += f"; note {note.get('reason')} ({'ruled ' + str(ruled.get('verdict')) if ruled else 'open'})"
    return line


# --- the tools ---------------------------------------------------------------------

def _check_reference(root: ExamRoot):
    from kullback.examiner.domain_tools import known_task

    async def check_reference(args: CheckReferenceArgs) -> CheckReferenceResult:
        task_id = known_task(root.workdir, args.task_id)
        view = reference_view(root, task_id)
        audit_mod.record_check(root.workdir, task_id)
        return CheckReferenceResult(summary=view_line(task_id, view), task_id=task_id, view=view)

    return check_reference


__all__ = ["EXCLUSIONS_FILE", "REFERENCE_WRONG_NOTES", "CheckReferenceArgs",
           "CheckReferenceResult", "RejectReferenceArgs", "RejectReferenceResult", "excluded_runs",
           "exclusions", "exclusions_path", "reference_view", "reject_on_note", "reject_reference", "repick",
           "view_line"]
