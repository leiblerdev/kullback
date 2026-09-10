"""One round, one Task table, written once and never rewritten (D218).

A round used to leave its Task level answers spread over three files that move at different times:
gates.json, which the per tool compile step overwrites in the middle of the next round;
gates_by_round.json, frozen at round close; and task_status.json, the live file every later
derivation, loosening and cache recompute writes again. A reader who wanted one row per Task had to
join the three, and the join no longer agreed with itself: rows the frozen ruling calls trusted fail
a check in the live file, rows it calls untrusted pass every check there, and failures the frozen
ruling names cannot be reproduced from what is on disk. Nothing on the record could tell a harness
regression from that drift.

This module is the one pass that answers the question. At round close the driver hands it the state
it already has in memory (the status rows, the replays, the two rulings it just landed, the
difficulty records and the bodies) and gets one row per Task carrying every stage ruling with its
reason: fidelity, reference, the D79 checks in suite order, false rejection with its pool size,
trusted, refused, the difficulty bucket and the kept body hash of every tool the Task calls. The
rows go to rounds/<n>/tasks.json and the round's history row is derived from them rather than
computed a second time, so the two cannot disagree. A snapshot already on disk is left exactly as it
is: a closed round is closed.

`drift` is the other half. The live file goes on moving, which is right, and the snapshot says what
the round ruled, which is also right; the number that matters is how many Tasks the two now disagree
about and at which stage, so a report can say "sixty four rows moved since round three" instead of
letting a reader mistake the movement for a regression.

Nothing here calls a model, reads a body or executes a Run: it is pure over the records it is given.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.gates.fidelity import unconfirmed_reason
from kullback.gates.verifier_suite import D79_STAGES
from kullback.runner.gate_support import _get
from kullback.runner.records import read_json, write_json

# The shape of rounds/<n>/tasks.json. Bumped when a row's fields or their meaning change, so a
# snapshot written under an older meaning reads as an older file rather than as today's answer.
SNAPSHOT_FORMAT = 1
ROUNDS_DIR = "rounds"
TASKS_NAME = "tasks.json"

# The D79 checks in the suite's own order, which is the order a row lists them in. A row that names
# them in some other order would make two snapshots of the same round compare unequal for nothing.
CHECK_ORDER: tuple[str, ...] = tuple(D79_STAGES.values())

# The three words a check's ruling can take. "not_run" is not "failed": a check nobody could run
# says nothing about the Verifier, and a reader who cannot tell them apart cannot group the Tasks
# that stop at each (D173, D198).
PASSED, FAILED, NOT_RUN = "passed", "failed", "not_run"

# The stages of a row the live file can move, in the order a drift is reported under. A Task that
# moved at more than one is reported at the first: it is where the reading of the row changed. The
# trusted ruling is not among them because nothing but these three can move it locally: it is read
# off the same row, so a Task the snapshot trusts whose checks no longer pass drifts at `verifier`.
STAGES: tuple[str, ...] = ("fidelity", "reference", "verifier")

NO_REASON = "no reason was recorded"


def snapshot_path(workdir: Any, round_number: int) -> Path:
    """Where one round's Task table lives."""
    return Path(workdir) / ROUNDS_DIR / str(int(round_number)) / TASKS_NAME


def body_hashes(bodies: Optional[dict], tools: Iterable[str]) -> dict[str, str]:
    """The hash of the kept body of each tool a Task calls, so a row says which bodies it was ruled
    against. A tool with no body on file is named with an empty hash rather than left out: that the
    Task called a tool the build never compiled is part of the reading."""
    kept = bodies or {}
    out: dict[str, str] = {}
    for tool in sorted(set(str(t) for t in tools or ())):
        body = kept.get(tool)
        out[tool] = hashlib.sha256(str(body).encode("utf-8")).hexdigest()[:16] if body is not None else ""
    return out


def tools_of(row: Any) -> list[str]:
    """The tools one Task's Runs called, off the row the derivation wrote."""
    replayed = _get(row, "tool_calls_replayed", None) or {}
    return sorted(str(name) for name in replayed) if isinstance(replayed, dict) else []


def checks_of(row: Any) -> list[dict]:
    """Every D79 check of one Task in suite order, each with its ruling and the reason behind it.

    A check the row does not mention at all is reported as not run with the reason saying so, which
    is what a Task that never reached the suite holds; the alternative, leaving it out, makes a row
    whose length depends on how far the Task got and cannot be tabulated.
    """
    results = _get(row, "checks", None) or {}
    skipped = {D79_STAGES.get(stage, stage) for stage in (_get(row, "not_run", None) or [])}
    reasons = _get(row, "not_run_reasons", None) or {}
    out = []
    for name in CHECK_ORDER:
        if name in skipped:
            ruling, reason = NOT_RUN, str(reasons.get(name) or "") or "the check had no input"
        elif name not in results:
            ruling, reason = NOT_RUN, "the derivation never reached this check"
        elif results.get(name):
            ruling, reason = PASSED, ""
        else:
            ruling, reason = FAILED, f"{name} failed"
        out.append({"check": name, "ruling": ruling, "reason": reason})
    return out


def fidelity_of(task_id: str, replays: Optional[dict]) -> tuple[bool, str]:
    """Whether some Trace of this Task replayed to its End state, and why not when none did."""
    per_task = (replays or {}).get(task_id)
    if not isinstance(per_task, dict) or not per_task:
        return False, "no Trace of the Task was replayed"
    if any(_get(replay, "confirmed", False) for replay in per_task.values()):
        return True, ""
    return False, unconfirmed_reason(per_task)


def task_row(task_id: str, *, round_number: int, row: Any, replays: Optional[dict],
             trusted: Optional[Any], bodies: Optional[dict], buckets: Optional[dict]) -> dict:
    """One Task's row: every stage ruling with its reason, from one reading of one state.

    `trusted` is the trusted ruling the round just landed, which carries the trusted ids, the
    untrusted reasons, the refusals and the false-rejection numbers with their pool sizes. Reading
    them off the ruling rather than computing them again here is the point of the file: the row and
    the round's own counts are then two views of one answer and cannot drift apart.
    """
    metrics = dict(getattr(trusted, "metrics", None) or {})
    untrusted = dict(metrics.get("untrusted") or {})
    refused = dict(metrics.get("refused") or {})
    confirmed, why = fidelity_of(task_id, replays)
    reference = bool(_get(row, "reference_confirmed", False))
    return {
        "task_id": task_id,
        "round": int(round_number),
        "fidelity": confirmed,
        "fidelity_reason": why,
        "reference": reference,
        "reference_reason": "" if reference else str(_get(row, "reason", "") or NO_REASON),
        "verifier_passed": bool(_get(row, "verifier_passed", False)),
        "checks": checks_of(row),
        "false_rejection": (metrics.get("false_rejection") or {}).get(task_id),
        "false_rejection_pool": int((metrics.get("false_rejection_pool") or {}).get(task_id) or 0),
        "false_rejection_ruling": str((metrics.get("false_rejection_ruling") or {}).get(task_id) or ""),
        "trusted": task_id in set(metrics.get("trusted") or ()),
        "trusted_reason": str(untrusted.get(task_id) or ""),
        "refused": task_id in refused,
        "refused_reason": str(refused.get(task_id) or ""),
        "difficulty": str((buckets or {}).get(task_id) or ""),
        "bodies": body_hashes(bodies, tools_of(row)),
    }


def task_rows(round_number: int, *, task_status: Optional[dict], replays: Optional[dict],
              trusted: Optional[Any] = None, bodies: Optional[dict] = None,
              buckets: Optional[dict] = None) -> list[dict]:
    """One row per Task the round knows about, in Task id order.

    A Task with a replay and no status row is still a row: it is a Task the round ruled on at the
    fidelity stage and nowhere else, and a table that dropped it would understate what was measured.
    """
    ids = sorted(set(task_status or {}) | set(replays or {}))
    return [task_row(task_id, round_number=round_number, row=(task_status or {}).get(task_id) or {},
                     replays=replays, trusted=trusted, bodies=bodies, buckets=buckets)
            for task_id in ids]


def counts_of(rows: Iterable[dict]) -> dict:
    """The round's Task level counts, derived from the rows and never computed a second time."""
    rows = list(rows or ())
    return {"tasks": len(rows),
            "fidelity": sum(1 for row in rows if row.get("fidelity")),
            "reference": sum(1 for row in rows if row.get("reference")),
            "verifier_passed": sum(1 for row in rows if row.get("verifier_passed")),
            "trusted": sum(1 for row in rows if row.get("trusted")),
            "refused": sum(1 for row in rows if row.get("refused"))}


def buckets_by_task(body: Optional[dict]) -> dict[str, str]:
    """The difficulty bucket per Task, off the difficulty record the round wrote."""
    return {str(record.get("task_id")): str(record.get("bucket") or "")
            for record in ((body or {}).get("tasks") or []) if isinstance(record, dict)}


def write_snapshot(workdir: Any, round_number: int, rows: Iterable[dict],
                   *, now: Optional[float] = None) -> dict:
    """This round's table, written once. A round already closed keeps the snapshot it has.

    Nothing rewrites a closed round: a later loosening, re-derive or cache recompute moves the live
    file and leaves this one byte for byte as the round left it, which is the only way a reader can
    tell the two apart.

    That holds only while a round number is used once, so the counter and this agree on what a round
    number means: the driver takes the next number past every round the workdir has closed, this
    file's `closed_rounds` among them (`rounds.last_round`, D231). Until it did, a second run over
    the same workdir opened at 1 again and was handed the previous run's table here, so its round
    reported a table written hours before the work it describes.
    """
    path = snapshot_path(workdir, round_number)
    existing = read_json(path, None)
    if isinstance(existing, dict) and existing.get("format") == SNAPSHOT_FORMAT:
        return existing
    rows = [dict(row) for row in rows or ()]
    body = {"format": SNAPSHOT_FORMAT, "round": int(round_number),
            "written_at": float(now if now is not None else time.time()),
            "counts": counts_of(rows), "rows": rows}
    write_json(path, body)
    return body


def read_snapshot(workdir: Any, round_number: Optional[int] = None) -> Optional[dict]:
    """One round's table, or the last closed round's when no round is named."""
    if round_number is not None:
        body = read_json(snapshot_path(workdir, round_number), None)
        return body if isinstance(body, dict) else None
    for number in reversed(closed_rounds(workdir)):
        body = read_json(snapshot_path(workdir, number), None)
        if isinstance(body, dict):
            return body
    return None


def closed_rounds(workdir: Any) -> list[int]:
    """The rounds this workdir holds a snapshot for, lowest first."""
    folder = Path(workdir) / ROUNDS_DIR
    if not folder.is_dir():
        return []
    numbers = []
    for child in folder.iterdir():
        if (child / TASKS_NAME).is_file() and child.name.isdigit():
            numbers.append(int(child.name))
    return sorted(numbers)


def _live_stage(task_id: str, row: dict, *, task_status: Optional[dict],
                replays: Optional[dict]) -> Optional[str]:
    """The first stage of this row the live files now read differently, or None when none does."""
    live = (task_status or {}).get(task_id)
    if live is None:
        return "reference" if row.get("reference") else None
    if fidelity_of(task_id, replays)[0] != bool(row.get("fidelity")):
        return "fidelity"
    if bool(_get(live, "reference_confirmed", False)) != bool(row.get("reference")):
        return "reference"
    if checks_of(live) != row.get("checks") or \
            bool(_get(live, "verifier_passed", False)) != bool(row.get("verifier_passed")):
        return "verifier"
    return None


def drift(snapshot: Optional[dict], *, task_status: Optional[dict], replays: Optional[dict],
          named: int = 3) -> dict:
    """How far the live files have moved from a closed round's table (D218 rule 4).

    A count, the stage each moved Task moved at, and the first few ids so a reader can go and look.
    A workdir with no snapshot has no drift to report and says so with a round of None, which reads
    differently from a drift of zero.
    """
    rows = list((snapshot or {}).get("rows") or ())
    if not isinstance(snapshot, dict) or not rows:
        return {"round": None, "tasks": 0, "status_drift": 0, "by_stage": {}, "first": []}
    moved: list[dict] = []
    by_stage: dict[str, int] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        stage = _live_stage(task_id, row, task_status=task_status, replays=replays)
        if stage is None:
            continue
        by_stage[stage] = by_stage.get(stage, 0) + 1
        moved.append({"task_id": task_id, "stage": stage})
    return {"round": snapshot.get("round"), "tasks": len(rows), "status_drift": len(moved),
            "by_stage": dict(sorted(by_stage.items())), "first": moved[:max(0, int(named))]}


def drift_line(report: dict) -> str:
    """One sentence a report and the status screen both print, so they cannot say different things."""
    if report.get("round") is None:
        return "No round snapshot is on disk, so nothing here is read against a closed round."
    named = ", ".join(f"{row['task_id']} ({row['stage']})" for row in report.get("first") or ())
    by_stage = ", ".join(f"{stage} {count}" for stage, count in (report.get("by_stage") or {}).items())
    return (f"Read from the round {report['round']} snapshot of {report['tasks']} Tasks; "
            f"status_drift {report['status_drift']} Tasks have moved since it closed"
            + (f" ({by_stage})" if by_stage else "")
            + (f", first {named}" if named else "") + ".")


__all__ = ["CHECK_ORDER", "FAILED", "NOT_RUN", "PASSED", "ROUNDS_DIR", "SNAPSHOT_FORMAT", "STAGES",
           "TASKS_NAME", "body_hashes", "buckets_by_task", "checks_of", "closed_rounds", "counts_of",
           "drift", "drift_line", "fidelity_of", "read_snapshot", "snapshot_path", "task_row",
           "task_rows", "tools_of", "write_snapshot"]
