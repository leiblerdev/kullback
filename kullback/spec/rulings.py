"""The Examiner's rulings on a Spec: feedback the writer applies or rebuts, never an edit (D331).

A ruling names one item of the Task (a check, an atom, an Intent fact, or the task as a whole), a kind
and a code from RULING_KINDS, the reason in one sentence, the fix the writer should make, and whether it
blocks. The Examiner files rulings in its round; the writer answers each open one in the next round
(applied or rebutted, with why); the Examiner then closes it or keeps it open. ROUNDS_CAP rounds at
most. An open blocking ruling holds the Task untrusted: the Spec's `rulings_open` counts them.

Rulings live one file each under exam/rulings/<task>/<n>.json; this module is their one reader and
writer, so the Spec side never imports the Examiner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import Field

from kullback.runner.records import EXAM_DIR, Record, Verifier, read_json, write_json
from kullback.spec.schema import Spec, load_spec, save_spec

RULINGS_DIR = "rulings"
# The item a ruling about the whole Task names (a scope ruling, or a missing sanity item).
TASK_ITEM = "task"

# What a ruling is about, each with the codes that say how; general, never a domain's words.
RULING_KINDS: dict[str, tuple[str, ...]] = {
    # The item does not follow from the Intent or the policy, or the Intent asks for more than the items check.
    "derivation": ("unasked_item", "uncovered_fact", "fixed_value", "unchecked_prohibition", "unbacked_check"),
    # A judge item (a question a model answers about what the Candidate said) is not fit to ask.
    "judge": ("not_yes_no", "taste", "anchor_wrong", "unanswerable", "leading"),
    # A code check is wrong as code: wrong place read, does not run, or the constructed Runs say so.
    "code": ("wrong_field", "does_not_run", "passes_do_nothing", "fails_reference"),
    # The Task itself: not doable under the policy, too vague, or missing a fact the Task needs.
    "scope": ("not_doable", "vague_intent", "missing_fact"),
}
RulingKind = Literal["derivation", "judge", "code", "scope"]
# The codes that rest on what the Verifier did to a constructed Run or the Reference: the verify tool first.
VERIFY_CODES = ("passes_do_nothing", "fails_reference")
WRONG_SIDES = ("reference", "verifier")
ANSWER_ACTIONS = ("applied", "rebutted")
# One sentence: no line break, between these lengths.
MIN_SENTENCE, MAX_SENTENCE = 12, 400


class Ruling(Record):
    """One ruling on one item of a Task's Spec, with the writer's answer and the Examiner's close."""
    task_id: str
    number: int
    round: int
    item: str
    kind: RulingKind
    code: str
    reason: str
    fix: str
    blocking: bool
    # For fails_reference: which side the Examiner rules wrong, the Reference or the Verifier.
    wrong_side: Optional[Literal["reference", "verifier"]] = None
    # The verify rows the ruling cites (constructed Runs and the Reference through the Verifier).
    verified: list[dict] = Field(default_factory=list)
    status: Literal["open", "closed"] = "open"
    # {action: applied | rebutted, why, round, changed: [check ids], writer: the repair's counts}
    answer: Optional[dict] = None
    # {round, why}
    closed: Optional[dict] = None


def sentence_refusal(text: str, what: str) -> Optional[str]:
    """Why a reason or fix is not one sentence, or None."""
    body = " ".join(str(text or "").split())
    if "\n" in str(text or "").strip() or not MIN_SENTENCE <= len(body) <= MAX_SENTENCE:
        return f"the {what} is one sentence on one line, {MIN_SENTENCE} to {MAX_SENTENCE} characters"
    return None


def code_refusal(kind: str, code: str) -> Optional[str]:
    if kind not in RULING_KINDS:
        return f"kind is one of {', '.join(RULING_KINDS)}"
    if code not in RULING_KINDS[kind]:
        return f"a {kind} ruling's code is one of {', '.join(RULING_KINDS[kind])}"
    return None


def item_ids(spec: Optional[Spec], verifier: Optional[Verifier]) -> list[str]:
    """Every id a ruling may name: the Spec's checks and Intent facts, the Verifier's items, the task."""
    ids: list[str] = []
    if spec is not None:
        ids += [check.id for check in spec.checks] + [fact.id for fact in spec.intent.facts]
    if verifier is not None:
        ids += [atom.id for atom in verifier.atoms]
        for name in ("expected", "forbidden", "conduct"):
            ids += [str(item.id) for item in getattr(verifier, name, None) or [] if getattr(item, "id", None)]
    ids.append(TASK_ITEM)
    return list(dict.fromkeys(ids))


def rulings_dir(workdir: Any, task_id: str) -> Path:
    return Path(workdir) / EXAM_DIR / RULINGS_DIR / task_id


def ruling_path(workdir: Any, task_id: str, number: int) -> Path:
    return rulings_dir(workdir, task_id) / f"{int(number)}.json"


def next_number(workdir: Any, task_id: str) -> int:
    folder = rulings_dir(workdir, task_id)
    numbers = [int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()] if folder.is_dir() else []
    return max(numbers, default=0) + 1


def load_rulings(workdir: Any, task_id: str) -> list[Ruling]:
    """The Task's rulings in number order; a file that does not read as a ruling is skipped."""
    folder = rulings_dir(workdir, task_id)
    paths = sorted((p for p in folder.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)) \
        if folder.is_dir() else []
    out = []
    for path in paths:
        try:
            out.append(Ruling.model_validate(read_json(path, {})))
        except ValueError:
            continue
    return out


def save_ruling(workdir: Any, ruling: Ruling) -> Path:
    path = ruling_path(workdir, ruling.task_id, ruling.number)
    write_json(path, ruling.model_dump(mode="json"))
    return path


def open_blocking(rulings: Iterable[Ruling]) -> list[Ruling]:
    return [r for r in rulings if r.status == "open" and r.blocking]


def unanswered(rulings: Iterable[Ruling]) -> list[Ruling]:
    """The open rulings the writer has not answered yet: its work for the next writer round."""
    return [r for r in rulings if r.status == "open" and r.answer is None]


def answer(workdir: Any, task_id: str, number: int, action: str, why: str, round_number: int,
           changed: Iterable[str] = (), writer: Optional[dict] = None) -> Ruling:
    """The writer's answer on one ruling: applied (with the checks it changed) or rebutted, with why, and
    the writer's counts (what code refused, a parse error) so an empty rebuttal can be told apart."""
    if action not in ANSWER_ACTIONS:
        raise ValueError(f"an answer is one of {', '.join(ANSWER_ACTIONS)}")
    ruling = Ruling.model_validate(read_json(ruling_path(workdir, task_id, number)))
    ruling = ruling.model_copy(update={"answer": {"action": action, "why": " ".join(str(why or "").split()),
                                                  "round": int(round_number), "changed": list(changed),
                                                  "writer": dict(writer or {})}})
    save_ruling(workdir, ruling)
    return ruling


def close(workdir: Any, task_id: str, number: int, keep_open: bool, why: str, round_number: int) -> Ruling:
    """The Examiner's word on an answered ruling: closed, or kept open with why."""
    ruling = Ruling.model_validate(read_json(ruling_path(workdir, task_id, number)))
    if ruling.status != "open":
        raise ValueError(f"ruling {number} on task {task_id} is already closed")
    if ruling.round >= round_number:
        raise ValueError(f"ruling {number} was filed this round; the writer answers it before it is closed")
    update = {"closed": {"round": int(round_number), "why": " ".join(str(why or "").split()),
                         "kept_open": bool(keep_open)}}
    if not keep_open:
        update["status"] = "closed"
    ruling = ruling.model_copy(update=update)
    save_ruling(workdir, ruling)
    return ruling


def sync_spec(workdir: Any, task_id: str, round_number: Optional[int] = None) -> Optional[Spec]:
    """The Spec's rulings_open set to its open blocking rulings, and its round when one is given."""
    spec = load_spec(workdir, task_id)
    if spec is None:
        return None
    update: dict = {"rulings_open": len(open_blocking(load_rulings(workdir, task_id)))}
    if round_number is not None:
        update["round"] = max(spec.round, int(round_number))
    spec = spec.model_copy(update=update)
    save_spec(workdir, spec)
    return spec


def counts(rulings: Iterable[Ruling]) -> dict:
    """Rulings by kind, by kind and code, blocking or note, open or closed, and the writer's answers."""
    out: dict = {"rulings": 0, "by_kind": {}, "by_code": {}, "blocking": 0, "note": 0, "open": 0,
                 "open_blocking": 0, "closed": 0, "applied": 0, "rebutted": 0, "unanswered": 0, "kept_open": 0}
    for r in rulings:
        out["rulings"] += 1
        out["by_kind"][r.kind] = out["by_kind"].get(r.kind, 0) + 1
        key = f"{r.kind}/{r.code}"
        out["by_code"][key] = out["by_code"].get(key, 0) + 1
        out["blocking" if r.blocking else "note"] += 1
        out["open" if r.status == "open" else "closed"] += 1
        out["open_blocking"] += int(r.status == "open" and r.blocking)
        if r.answer:
            out[r.answer.get("action", "applied")] += 1
        elif r.status == "open":
            out["unanswered"] += 1
        out["kept_open"] += int(bool((r.closed or {}).get("kept_open")))
    return out


def rulings_line(workdir: Any, task_ids: Iterable[str]) -> str:
    """The one line the Builder reads about the rulings on these Tasks, empty when there are none."""
    total = counts(r for task_id in task_ids for r in load_rulings(workdir, task_id))
    if not total["rulings"]:
        return ""
    return (f"spec rulings: {total['rulings']} filed, {total['open_blocking']} open and blocking; the Spec writer "
            "answers them in the review loop")


__all__ = ["ANSWER_ACTIONS", "rulings_line", "MAX_SENTENCE", "MIN_SENTENCE", "RULINGS_DIR", "RULING_KINDS", "TASK_ITEM",
           "VERIFY_CODES", "WRONG_SIDES", "Ruling", "answer", "close", "code_refusal", "counts", "item_ids",
           "load_rulings", "next_number", "open_blocking", "ruling_path", "rulings_dir", "save_ruling",
           "sentence_refusal", "sync_spec", "unanswered"]
