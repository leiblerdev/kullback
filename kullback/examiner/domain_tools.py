"""The Examiner's domain tools over its root: the ruling tools, finding, no_finding, check_reference.

The Examiner reads and gives feedback; it edits nothing and writes no Verifier (D320, D331). The
ruling tools (rule_tool.py: rule, close, verify, lookup_rows, search_rows) carry its feedback on a
Spec, which the writer applies or rebuts. `finding` files what is wrong with the Environment for the
Builder: the rows and a body edit (D317). `no_finding` files a review that found nothing, with the
one reason the Examiner is satisfied. `check_reference` shows the evidence on a Reference. Every
result carries rows, never a count alone.

The Builder's notes (one per Task under the workdir's notes/, a reason from NOTE_REASONS and one
sentence) are read here too: a finding with `note_ruling` on the Task rules the note, and the ruling
is written next to it for the Builder's next round.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool
from kullback.examiner.exam_files import (
    ExamRoot,
    Finding,
    check_edit,
    finding_key,
    open_by_key,
)
from kullback.runner.records import read_json, write_json

PRODUCED_FINDINGS = ["findings"]
PRODUCED_REVIEWS = ["examiner/reviews.json"]
#: Where the reviews that found nothing are kept, one row per Task, beside findings.json.
REVIEWS_FILE = "examiner/reviews.json"
NO_FINDING = "no finding"
# The shortest reason a no-finding review may give: one clause, not a word.
MIN_REASON_CHARS = 12


def render(result: BaseModel) -> str:
    """The summary line; everything else stays in details for the transcript."""
    return getattr(result, "summary", None) or getattr(result, "text", "") or ""


class FindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = Field(default=None)
    kind: str = Field(description="What the finding is about: assisted_tool, fidelity, "
                                  "reference_disagreement, suite, false_rejection, environment, other.")
    text: str = Field(description="What is wrong, in sentences.")
    rows: list[dict] = Field(default_factory=list, description="The evidence, one record per call or Task.")
    path: str = Field(default="", description="The Environment file the Builder should edit, or empty.")
    change: str = Field(default="", description="One line saying what should differ.")
    edits: list[dict] = Field(default_factory=list, description=(
        "The Environment diff, each with its why: {kind: body, path, call_id, column, recorded, replayed, why}. "
        "What is wrong with a Spec is a ruling, never an edit."))
    note_ruling: Optional[Literal["builder_right", "builder_wrong"]] = Field(
        default=None, description="Set to rule on the Builder's open note on this Task: whether the "
                                  "Builder is right, with the why in text.")


class FindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    finding_id: str
    finding: dict = Field(default_factory=dict)
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_FINDINGS))


FINDING_KINDS = ("assisted_tool", "fidelity", "reference_disagreement", "suite",
                 "false_rejection", "environment", "other")

def _keep_suite_status(root: ExamRoot, task_id: str, row: Optional[dict]) -> None:
    """An accepted proposal's suite and version as the Task's status row, in both status files.

    The exposed exam/task_status.json the session reads, and the workdir's task_status.json the
    Builder's status reads, the rest of each row kept. A ruling with no suite leaves the rows as they were.
    A suite passed now was ruled against the Reference the Task holds now, which is the evidence a
    re-picked Reference's `derive_again` mark waits for (reference_check), so a pass clears the mark.
    """
    if row is None:
        return
    root.task_status[task_id] = _merged_status(root.task_status.get(task_id), row)
    write_json(root.exam_dir / "task_status.json", root.task_status)
    path = Path(root.workdir) / "task_status.json"
    status = read_json(path, {}) if path.is_file() else {}
    status = status if isinstance(status, dict) else {}
    status[task_id] = _merged_status(status.get(task_id), row)
    write_json(path, status)


def _merged_status(old: Optional[dict], row: dict) -> dict:
    """The status row with `row` over it, and without the re-pick mark once the suite passed."""
    merged = {**(old or {}), **row}
    if merged.get("verifier_passed") is True and "derive_again" not in row:
        merged.pop("derive_again", None)
    return merged


# --- the Builder's notes: one per Task, ruled by the Examiner (D280) ---

#: Where the Builder's notes live under the workdir, one file per Task, the ruling next to it.
NOTES_DIR = "notes"
RULING_SUFFIX = ".ruling.json"
#: Why a Task is not verifiable as written: the only reasons a note may give.
NOTE_REASONS = ("outcome_not_in_state", "intent_contradicts_reference", "fact_unavailable_to_user",
                "needs_action_record")
NoteReason = Literal["outcome_not_in_state", "intent_contradicts_reference", "fact_unavailable_to_user",
                     "needs_action_record"]
NOTE_SENTENCE_CHARS = 300
# What a note's sentence may not carry: structure or check source, which is an atom or Verifier text.
_NOTE_FORBIDDEN = ("{", "}", "[", "]", "predicate", "atom", "def ", "lambda", "return ")


def task_ids_of(workdir: Any) -> list[str]:
    """The workdir's Task ids, the stems of tasks/<id>.json: the one set a model-named id is checked against."""
    return [path.stem for path in sorted((Path(workdir) / "tasks").glob("*.json")) if path.name != "tasks.json"]


def known_task(workdir: Any, task_id: Any) -> str:
    """The id when it is one of the workdir's Tasks, else a ValueError naming it unknown.

    Membership, never a path check: an id with a separator, `..` or an absolute path is not a Task.
    """
    if not isinstance(task_id, str) or task_id not in task_ids_of(workdir):
        raise ValueError(f"unknown Task {task_id!r}: it is not one of the Tasks under tasks/")
    return task_id


def task_path(workdir: Any, task_id: Any, folder: Any, suffix: str = ".json") -> Path:
    """Where a Task's file lives under `folder`, resolved only for a known Task id."""
    return Path(folder) / f"{known_task(workdir, task_id)}{suffix}"


def notes_dir(workdir: Any) -> Path:
    return Path(workdir) / NOTES_DIR


def note_path(workdir: Any, task_id: str) -> Path:
    return notes_dir(workdir) / f"{task_id}.json"


def ruling_path(workdir: Any, task_id: str) -> Path:
    return notes_dir(workdir) / f"{task_id}{RULING_SUFFIX}"


def note_refusal(reason: Any, sentence: Any) -> Optional[str]:
    """Why a note is refused, or None: a reason off the fixed list and one plain sentence only."""
    if reason not in NOTE_REASONS:
        return f"the reason {reason!r} is not one of {', '.join(NOTE_REASONS)}"
    text = str(sentence or "").strip()
    if not text:
        return "the note carries no sentence"
    if "\n" in text or len(text) > NOTE_SENTENCE_CHARS:
        return f"the note is one sentence on one line, at most {NOTE_SENTENCE_CHARS} characters"
    lowered = text.lower()
    carried = [mark for mark in _NOTE_FORBIDDEN if mark in lowered]
    if carried:
        return (f"the note carries {', '.join(repr(mark) for mark in carried)}: a note says why the Task "
                "is not verifiable as written, never an atom or Verifier text")
    return None


def note_hash(note: dict) -> str:
    body = json.dumps({key: note.get(key) for key in ("task_id", "reason", "sentence")}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def write_note(workdir: Any, task_id: str, reason: str, sentence: str) -> dict:
    """The Builder's note on one Task, written over any earlier note; refused as a ValueError."""
    why = note_refusal(reason, sentence)
    if why is not None:
        raise ValueError(why)
    note = {"task_id": task_id, "reason": reason, "sentence": str(sentence).strip()}
    note["note_hash"] = note_hash(note)
    write_json(task_path(workdir, task_id, notes_dir(workdir)), note)
    return note


def _read_dict(path: Path) -> Optional[dict]:
    try:
        body = read_json(path, None)
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def read_note(workdir: Any, task_id: str) -> Optional[dict]:
    return _read_dict(note_path(workdir, task_id))


def read_ruling(workdir: Any, task_id: str) -> Optional[dict]:
    """The Examiner's ruling on the Task's note as it stands now; a ruling on an older note is none."""
    note, ruling = read_note(workdir, task_id), _read_dict(ruling_path(workdir, task_id))
    if note is None or ruling is None or ruling.get("note_hash") != note.get("note_hash"):
        return None
    return ruling


def notes_of(workdir: Any) -> dict[str, tuple[dict, Optional[dict]]]:
    """Every note by Task with its ruling, None while the note is open."""
    folder = notes_dir(workdir)
    out: dict[str, tuple[dict, Optional[dict]]] = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name.endswith(RULING_SUFFIX):
            continue
        note = _read_dict(path)
        if note is not None:
            out[path.stem] = (note, read_ruling(workdir, path.stem))
    return out


def open_note(workdir: Any, task_id: str) -> Optional[dict]:
    """The Task's note while no ruling answers it, else None."""
    note = read_note(workdir, task_id)
    return note if note is not None and read_ruling(workdir, task_id) is None else None


def rule_note(workdir: Any, task_id: Optional[str], move: str, verdict: str, text: str,
              root: Optional[ExamRoot] = None) -> Optional[dict]:
    """Rule on the Task's open note with one of the Examiner's moves, written next to the note.

    Returns the ruling, or None when the Task has no open note. Agreeing with a note that says the
    Reference's outcome is wrong also rejects that Reference (reference_check), so the ruling acts.
    The session's `root` carries the status mark into what the session holds, so a later accepted
    write in the same session keeps it.
    """
    from kullback.examiner.reference_check import reject_on_note  # it reads notes through this module

    note = open_note(workdir, task_id) if task_id else None
    if note is None:
        return None
    ruling = {"task_id": task_id, "note_hash": note["note_hash"], "move": move, "verdict": verdict,
              "text": text}
    write_json(task_path(workdir, task_id, notes_dir(workdir), RULING_SUFFIX), ruling)
    rejected = reject_on_note(workdir, task_id, note, verdict, text, root=root)
    if rejected is not None:
        ruling["reference_rejected"] = rejected
    return ruling


def note_line(task_id: str, note: dict, ruling: Optional[dict]) -> str:
    """One note as a line of an opening message: the reason, the sentence, and its ruling or open."""
    head = f"{task_id}: {note.get('reason')}: {note.get('sentence')}"
    if ruling is None:
        return f"{head} (open)"
    return f"{head} (ruled by {ruling.get('move')}, {ruling.get('verdict')}: {ruling.get('text')})"


def _on_task(task_id: Optional[str]) -> str:
    """The ` on task <id>` clause a finding's message carries, empty when it names no Task."""
    return f" on task {task_id}" if task_id else ""


def _check_finding(root: ExamRoot, args: FindingArgs) -> str:
    """The finding's kind once its kind, Task and note ruling hold; raises naming what does not."""
    kind = (args.kind or "").strip()
    if kind not in FINDING_KINDS:
        raise ValueError(f"{args.kind!r} is not a finding kind. The kinds are: "
                         f"{', '.join(FINDING_KINDS)}.")
    if args.task_id is not None:
        known_task(root.workdir, args.task_id)
    if args.note_ruling is not None and open_note(root.workdir, args.task_id or "") is None:
        raise ValueError(f"task {args.task_id} has no open note from the Builder to rule on; "
                         "file the finding without note_ruling")
    return kind


def _refuse_repeat(root: ExamRoot, record: Finding, args: FindingArgs) -> None:
    """Raise when an open finding already says what `record` says."""
    key = finding_key(record.kind, record.path or record.change, args.task_id or "")
    existing = open_by_key(root.findings).get(key)
    if existing is not None:
        raise ValueError(f"{existing} already says this ({record.kind}"
                         + (f" in {args.path}" if args.path else "")
                         + _on_task(args.task_id)
                         + "); it is filed and the Builder has not answered it yet. Act on it, "
                           "or file a finding that says something else.")


def _finding(root: ExamRoot):
    async def finding(args: FindingArgs) -> FindingResult:
        kind = _check_finding(root, args)
        edits = [check_edit(edit, root.exam_dir) for edit in args.edits]
        record = Finding(task_id=args.task_id, kind=kind, text=args.text, rows=list(args.rows),
                         path=args.path, change=args.change, source="model", edits=edits)
        _refuse_repeat(root, record, args)
        root.findings.append(record)
        body = record.as_dict()
        if root.bus is not None:
            root.bus.publish_custom("finding", body)
        summary = (f"finding {body['finding_id']} ({kind}) filed" + _on_task(args.task_id)
                   + (f", naming {args.path}" if args.path else ""))
        if args.note_ruling is not None:
            rule_note(root.workdir, args.task_id, "finding", args.note_ruling, args.text, root=root)
            summary += f"; the Builder's note on task {args.task_id} is ruled: {args.note_ruling}"
        return FindingResult(summary=summary, finding_id=body["finding_id"], finding=body)

    return finding


class NoFindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reason: str = Field(description="The one reason the review found nothing wrong with this Task.")


class NoFindingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    review: dict = Field(default_factory=dict)
    produced: list[str] = Field(default_factory=lambda: list(PRODUCED_REVIEWS))


def reviews_of(workdir: Any) -> dict[str, dict]:
    """The reviews that found nothing, by Task: outcome "no finding" and the reason."""
    body = read_json(Path(workdir) / REVIEWS_FILE, {}) or {}
    return body if isinstance(body, dict) else {}


def _no_finding(root: ExamRoot):
    async def no_finding(args: NoFindingArgs) -> NoFindingResult:
        known_task(root.workdir, args.task_id)
        reason = " ".join(args.reason.split())
        if len(reason) < MIN_REASON_CHARS:
            raise ValueError(f"say the one reason you are satisfied, at least {MIN_REASON_CHARS} characters")
        if any(f.task_id == args.task_id for f in root.findings if f.source == "model"):
            raise ValueError(f"task {args.task_id} already has a finding this session; it is not a review "
                             "that found nothing")
        review = {"task_id": args.task_id, "outcome": NO_FINDING, "reason": reason}
        reviews = reviews_of(root.workdir)
        reviews[args.task_id] = review
        write_json(Path(root.workdir) / REVIEWS_FILE, reviews)
        return NoFindingResult(summary=f"task {args.task_id}: {NO_FINDING}: {reason}", review=review)

    return no_finding


def domain_tools(root: ExamRoot) -> list[AgentTool]:
    """The Examiner's tools over one ExamRoot: it reads and files, nothing here writes a Verifier or runs."""
    from kullback.examiner import reference_check as RC  # it imports this module's note readers
    from kullback.examiner import rule_tool as R

    return [
        *R.rule_tools(root),
        AgentTool("finding", "File what is wrong with the Environment for the Builder: what in `text`, the rows, "
                  "the file in `path`, the one line in `change`, and the body edits, each with its why.",
                  FindingArgs, FindingResult, _finding(root), render=render),
        AgentTool("no_finding", "File a review of a Task that found nothing wrong, with the one reason.",
                  NoFindingArgs, NoFindingResult, _no_finding(root), render=render),
        AgentTool("check_reference", "Show the evidence on a Task's Reference: the Intent, each write with "
                  "the reads before it, whether it wrote nothing, whether the recordings agree, and any "
                  "Builder note. Code only, no model call.",
                  RC.CheckReferenceArgs, RC.CheckReferenceResult, RC._check_reference(root), render=render),
    ]


#: The tools the Examiner holds, so tests pin its surface: none edits a Spec or writes a Verifier.
TOOL_NAMES = ("rule", "close", "verify", "lookup_rows", "search_rows", "finding", "no_finding", "check_reference")


def tool_names() -> tuple[str, ...]:
    """The domain tool names, so tests pin the Builder-called surface."""
    return TOOL_NAMES


__all__ = ["MIN_REASON_CHARS", "NOTES_DIR", "NOTE_REASONS", "NO_FINDING", "REVIEWS_FILE", "TOOL_NAMES",
           "FindingArgs", "FindingResult", "NoFindingArgs", "NoFindingResult", "domain_tools", "known_task",
           "note_line", "note_refusal", "notes_of", "open_note", "read_note", "read_ruling", "reviews_of",
           "rule_note", "ruling_path", "task_ids_of", "task_path", "tool_names", "write_note"]
