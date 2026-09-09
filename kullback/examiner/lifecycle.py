"""When a derived artefact stops being live: its source was withdrawn (D208).

A Verifier is not a free-standing claim. It is derived from a Reference, and it says what the End
state of one Task looks like only for as long as that Reference is the one the Task holds. The
harness already withdraws a Reference in the ordinary course of a round: the residue path settles a
disagreement on a different surviving End state, a frozen-list change moves the Runs a Task is
clustered from, a later round's recordings disagree where an earlier round's agreed. What it did not
do was retire what the withdrawn Reference had produced, so a file written two rounds ago stayed
under verifiers/ and every reader that globs that directory kept scoring it against a pool that had
meanwhile grown with the very Runs the disagreement was about.

The rule here is about lineage and nothing else. An artefact records the source it was derived from
(for a Verifier, `seed_run_ids`, the Run ids of the Reference group it was derived over), the Task's
status row records which source the Task currently holds (`reference_confirmed`, and the Reference's
own Run ids), and an artefact whose source is not the one the row names is retired: kept in version
history under a reason, taken out of the set any gate scores, and never revived by name. Nothing
here reads a column, a tool, a table or a corpus; the two reasons are the two ways a source can stop
being yours, it was withdrawn and nothing replaced it, or it was replaced by another.

The retirement is a step of the derivation, so a round retires and re-derives in one pass, and the
Examiner's plan filters the same way when it reads state back. That second reading is not
belt-and-braces alone: a workdir written by an earlier build still holds the orphaned files, and the
filter is what stops those from being scored before the next derivation has run over them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.gate_support import _get
from kullback.runner.records import Verifier, read_json

# The source was withdrawn and the Task holds none: the artefact claims an End state the Task no
# longer says it has.
REFERENCE_WITHDRAWN = "reference_withdrawn"
# The source was replaced: the Task holds a Reference, and this artefact was derived over a Run that
# is no longer part of it (a survivor choice, a re-clustered Task, a later round's disagreement).
REFERENCE_REDERIVED = "reference_rederived"
# The field the status row carries once a Task's artefact has been retired, so a reader of the row
# sees why the Task has no Verifier rather than reading the absence as an ordinary loss.
RETIRED_FIELD = "verifier_retired"

REASON_TEXT = {
    REFERENCE_WITHDRAWN: "the Reference it was derived from is not confirmed this round",
    REFERENCE_REDERIVED: "the Reference it was derived from was replaced by another",
}


def source_run_ids(artefact: Any) -> list[str]:
    """The Run ids of the source this artefact was derived from, as the artefact itself records them.

    Every artefact this rule covers names its source the same way, which is why one accessor can
    serve all of them: a Verifier carries the Reference group it was derived over, and anything else
    derived from a Reference carries the same field or is not covered here at all.
    """
    return [str(run_id) for run_id in (_get(artefact, "seed_run_ids", None) or ())]


def confirmed_run_ids(row: Any) -> Optional[list[str]]:
    """The Run ids of the Reference the Task holds now, or None when the row does not name them.

    A row written before the field existed says nothing about which Reference is held, so the
    identity half of the rule cannot be applied to it and only the confirmation half is.
    """
    ids = _get(row or {}, "reference_run_ids", None)
    return None if ids is None else [str(run_id) for run_id in ids]


def retirement(artefact: Any, row: Any) -> Optional[dict]:
    """Why this artefact is no longer live, or None while the source it names is the one held.

    Two questions in order. Does the Task hold a confirmed Reference at all; and if it does, was
    this artefact derived over Runs that are still part of it. The second is asked as a subtraction
    rather than an equality: a source Run that has dropped out of the Reference is a source that
    changed, while a Reference that has grown a Run the artefact was not derived over is the same
    source with more evidence behind it, which the next derivation will read anyway.
    """
    sources = source_run_ids(artefact)
    if not _get(row or {}, "reference_confirmed", False):
        return _row(artefact, REFERENCE_WITHDRAWN, sources, [])
    confirmed = confirmed_run_ids(row)
    if confirmed is None:
        return None
    gone = sorted(set(sources) - set(confirmed))
    if sources and gone:
        return _row(artefact, REFERENCE_REDERIVED, sources, gone)
    return None


def _row(artefact: Any, reason: str, sources: list[str], gone: list[str]) -> dict:
    text = REASON_TEXT[reason]
    if gone:
        text = f"{text} ({', '.join(gone)} is no longer part of it)"
    return {"task_id": str(_get(artefact, "task_id", "")), "reason": reason, "text": text,
            "source_run_ids": sources}


def partition(artefacts: Iterable[Any], task_status: Optional[dict]) -> tuple[list, list[dict]]:
    """The artefacts still live and one row per artefact that is not, in Task order.

    A workdir with no status rows at all is a workdir that has not derived yet, and an artefact
    there has no source state to be judged against; everything is live, which is what the derivation
    stage and every test that hands a bare Verifier list expect.
    """
    rows = task_status or {}
    artefacts = list(artefacts or ())
    if not rows:
        return artefacts, []
    live: list[Any] = []
    retired: list[dict] = []
    for artefact in artefacts:
        row = retirement(artefact, rows.get(str(_get(artefact, "task_id", ""))))
        (retired.append(row) if row is not None else live.append(artefact))
    return live, sorted(retired, key=lambda r: r["task_id"])


def live(artefacts: Iterable[Any], task_status: Optional[dict]) -> list:
    """The accessor every consumer reads through: the artefacts whose source the Task still holds."""
    return partition(artefacts, task_status)[0]


def on_disk(workdir: Path) -> list[Verifier]:
    """Every Verifier file the workdir holds, unfiltered; only the retirement step reads this."""
    return [Verifier.model_validate(read_json(path))
            for path in sorted((Path(workdir) / "verifiers").glob("*.json"))]


def retire(workdir: Path, task_status: dict, *, round_number: int = 0) -> list[dict]:
    """Retire, in the derivation's own step, every Verifier on disk whose Reference is not held.

    The file leaves verifiers/ so nothing can pick it up off disk, the Task's status row says it was
    retired and why, and the row returned carries the Verifier itself so the caller can keep it in
    version history. Deriving again is what gives the Task a Verifier back; nothing revives one.
    """
    rows: list[dict] = []
    for verifier in on_disk(workdir):
        row = retirement(verifier, (task_status or {}).get(verifier.task_id))
        if row is None:
            continue
        (Path(workdir) / "verifiers" / f"{verifier.task_id}.json").unlink(missing_ok=True)
        status_row = (task_status or {}).get(verifier.task_id)
        if isinstance(status_row, dict):
            status_row[RETIRED_FIELD] = {"reason": row["reason"], "text": row["text"],
                                         "round": int(round_number),
                                         "source_run_ids": row["source_run_ids"]}
        rows.append({**row, "round": int(round_number), "verifier": verifier})
    return sorted(rows, key=lambda r: r["task_id"])


def carry_forward(workdir: Path, task_status: dict, prior: Optional[dict]) -> list[str]:
    """Keep an earlier round's retirement on a row that has been written again without it.

    A derivation served from its cache writes back the row the entry holds, and that row was written
    before the artefact was retired, so the marker would be dropped every time a Task that still has
    no Verifier is derived again. The marker is the answer to why the Task has none, so it stays
    while that is still the answer: a Task whose file is back on disk has been derived afresh and
    its row rightly carries nothing. Run after the retirement step, whose deletions it reads.
    """
    carried: list[str] = []
    for task_id, row in (task_status or {}).items():
        if not isinstance(row, dict) or retired_row(row) is not None:
            continue
        if (Path(workdir) / "verifiers" / f"{task_id}.json").is_file():
            continue
        earlier = retired_row((prior or {}).get(task_id))
        if earlier is not None:
            row[RETIRED_FIELD] = earlier
            carried.append(str(task_id))
    return sorted(carried)


def retired_row(row: Any) -> Optional[dict]:
    """The retirement a status row carries, or None: how a reader tells a retired Task from a loss."""
    value = _get(row or {}, RETIRED_FIELD, None)
    return value if isinstance(value, dict) else None


def retired_in_round(task_status: Optional[dict], round_number: int) -> list[dict]:
    """The retirements one round made, off the status rows themselves.

    The rows outlive the call that made them, which is what lets a driver read a round's retirements
    after the fact rather than having to be handed them by the stage that ran.
    """
    out = []
    for task_id, row in sorted((task_status or {}).items()):
        gone = retired_row(row)
        if gone is not None and int(gone.get("round") or 0) == int(round_number):
            out.append({"task_id": str(task_id), **gone})
    return out


def counts(rows: Iterable[dict]) -> dict:
    """What a round retired, in the two counts D208 asks to be read: how many, and by which reason."""
    rows = list(rows or ())
    by_reason = {reason: sum(1 for row in rows if row.get("reason") == reason)
                 for reason in (REFERENCE_WITHDRAWN, REFERENCE_REDERIVED)}
    return {"verifiers_retired": len(rows), "verifiers_retired_by_reason": by_reason}
