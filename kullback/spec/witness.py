"""The Task's Reference as a witness Run, for the reference_passes flag only; the writer never reads it.

The Reference's jobs are world fidelity and one witness: its end state and final answer go through
the Intent-written Verifier, and a fail goes to the Examiner (verifier-redesign-1007, v2). Nothing in
the writer's path imports this module (tests/spec/test_writer_blind.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import Run, load_run_jsonl, read_json, run_path

REFERENCE_KINDS = ("recording", "reroll")


def _rows(index: Any, task_id: str) -> list[dict]:
    rows = (index or {}).get(task_id) if isinstance(index, dict) else None
    rows = rows.values() if isinstance(rows, dict) else rows or []
    return [row for row in rows if isinstance(row, dict)]


def reference_run(workdir: Path, task_id: str) -> Optional[Run]:
    """The Task's first kept Reference as a Run (references.json), a recording before a re-roll; None if none."""
    root = Path(workdir)
    row = (read_json(root / "references.json", {}) or {}).get(task_id) or {}
    refs = [ref for ref in row.get("references") or [] if isinstance(ref, dict) and ref.get("run_id")]
    refs.sort(key=lambda ref: REFERENCE_KINDS.index(ref.get("kind")) if ref.get("kind") in REFERENCE_KINDS else 2)
    paths = {str(r["run_id"]): r["path"] for index in ("replays.json", "rerolls.json", "examiner/rerolls.json")
             for r in _rows(read_json(root / index, {}), task_id) if r.get("run_id") and r.get("path")}
    for ref in refs:
        path = paths.get(str(ref["run_id"]))
        if path and run_path(root, path).is_file():
            return load_run_jsonl(run_path(root, path))
    return None


__all__ = ["REFERENCE_KINDS", "reference_run"]
