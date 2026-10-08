"""A Reference audit pass over the trusted Tasks the main session never checked.

After the main Examiner session ends, the harness collects every trusted Task
whose Reference no session has checked and runs audit sessions over them, in
chunks. The record of checked References lives in the exam directory, so it
holds across the sessions of one build. Trusted comes from the build's latest
round file, the same source the trust split reads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from kullback.examiner.exam_files import EXAM_DIR
from kullback.runner.records import read_json, write_json

#: At most this many Tasks per audit session, so one session stays answerable.
AUDIT_CHUNK = 30
#: The checked References, task id to when this build checked it.
CHECKS_FILE = "reference_checks.json"
#: One record per audit session: chunk, Tasks, turns, spend.
AUDITS_FILE = "reference_audits.json"
#: The extra opening line of an audit session. Names no corpus, tool or model.
AUDIT_LINE = ("This session audits References: judge each listed Task's Reference "
              "against its Intent and policy.")


def checks_path(workdir: Any) -> Path:
    """Where the checked record lives: the exam directory the session owns."""
    return Path(workdir) / EXAM_DIR / CHECKS_FILE


def record_check(workdir: Any, task_id: str) -> None:
    """Mark one Task's Reference checked by this build, crash-safe per call."""
    path = checks_path(workdir)
    checks = read_json(path, {}) or {}
    if not isinstance(checks, dict):
        checks = {}
    checks[str(task_id)] = {"at": time.time()}
    write_json(path, checks)


def checked_ids(workdir: Any) -> set[str]:
    """The Tasks whose Reference some session already checked."""
    checks = read_json(checks_path(workdir), {}) or {}
    return set(checks) if isinstance(checks, dict) else set()


def trusted_ids(workdir: Any) -> list[str]:
    """The build's latest round file's trusted Tasks, in id order, else none.

    No round file means no build verdict yet, so no audit list rather than a
    guessed one.
    """
    rounds = Path(workdir) / "rounds"
    if not rounds.is_dir():
        return []
    numbered = [p for p in rounds.iterdir() if p.is_dir() and p.name.isdigit()]
    if not numbered:
        return []
    latest = max(numbered, key=lambda p: int(p.name))
    doc = read_json(latest / "tasks.json", {}) or {}
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return sorted(str(r.get("task_id")) for r in rows
                  if isinstance(r, dict) and r.get("trusted") and r.get("task_id"))


def audit_list(workdir: Any) -> list[str]:
    """Every trusted Task whose Reference is still unchecked, in id order."""
    return sorted(set(trusted_ids(workdir)) - checked_ids(workdir))


def chunks(ids: Iterable[str], size: int = AUDIT_CHUNK) -> list[list[str]]:
    """The ids in task order, at most size per chunk."""
    ordered = sorted(set(str(i) for i in ids))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def audits_path(workdir: Any) -> Path:
    """Where the per-audit-session spend lines land."""
    return Path(workdir) / EXAM_DIR / AUDITS_FILE


def record_audit(workdir: Any, record: dict) -> list[dict]:
    """Append one audit session's line (chunk, Tasks, turns, spend) and return all."""
    path = audits_path(workdir)
    records = read_json(path, []) or []
    if not isinstance(records, list):
        records = []
    records.append(record)
    write_json(path, records)
    return records


__all__ = ["AUDIT_CHUNK", "AUDIT_LINE", "AUDITS_FILE", "CHECKS_FILE", "audit_list", "audits_path",
           "checked_ids", "checks_path", "chunks", "record_audit", "record_check", "trusted_ids"]
