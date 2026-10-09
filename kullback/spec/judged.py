"""The judge items' answers that verify and trust read (D346, D347): the Verdict never calls a model.

verify judges the Reference once per (Verifier, Run) and keeps the answers under exam/judged/, which
only the Examiner reads; the score stage keeps each fresh Run's Verdict under verdicts/<task>/. Trust
reads both, so a Run that passes only with its judge items passes in trust too. The judge's reason
goes to the Examiner and nowhere the writer reads (spec/writer.py reads none of these files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.judge.shape import evidence_refs, is_item, resolve
from kullback.runner.records import EXAM_DIR, Run, Verifier, as_dict, content_hash, read_json, write_json

JUDGED_DIR = "judged"
VERDICTS_DIR = "verdicts"
# What verify shows the Examiner of the turns the judge read: the last few, each on one short line.
READ_LINES = 3
LINE_CHARS = 160


def verifier_hash(verifier: Verifier) -> str:
    return content_hash(as_dict(verifier))[:16]


def judged_path(workdir: Any, task_id: str, verifier: Verifier, run_id: str) -> Path:
    return Path(workdir) / EXAM_DIR / JUDGED_DIR / task_id / f"{run_id}.{verifier_hash(verifier)}.json"


def item_ids(verifier: Verifier) -> list[str]:
    """The Verifier's judge items: the atoms only a judge's answer settles."""
    return [atom.id for atom in verifier.atoms if is_item(atom)]


def lines_read(run: Run, target: dict) -> list[str]:
    """The last turns a judge item's evidence names on the Run, each one line, clipped."""
    turns = [text for ref in evidence_refs(target) for _, _, text in resolve(run, ref)]
    return [" ".join(text.split())[:LINE_CHARS] for text in turns[-READ_LINES:]]


def cached_judge(workdir: Any, task_id: str, verifier: Verifier, run_id: str) -> Optional[dict]:
    """atom id -> {score, why, lines} kept by verify for this Verifier and Run, else None."""
    body = read_json(judged_path(workdir, task_id, verifier, run_id), None)
    return body if isinstance(body, dict) else None


def judge_reference(workdir: Any, task_id: str, verifier: Verifier, run: Run,
                    judge: Optional[Callable[[Verifier, Run], dict]]) -> Optional[dict]:
    """The Run's judge items answered once and kept: the cache first, then `judge` (one call per item,
    kullback/judge/items.judge_items), else None. A Verifier with no judge item needs no answer."""
    ids = item_ids(verifier)
    if not ids:
        return {}
    cached = cached_judge(workdir, task_id, verifier, run.run_id)
    if cached is not None or judge is None:
        return cached
    targets = {atom.id: atom.target or {} for atom in verifier.atoms}
    answers = {}
    for atom_id, result in judge(verifier, run).items():
        row = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
        answers[atom_id] = {"score": row.get("score"), "why": str(row.get("why") or row.get("error") or ""),
                            "cost_usd": float(row.get("cost_usd") or 0.0),
                            "lines": lines_read(run, targets.get(atom_id) or {})}
    if all(row["score"] is not None for row in answers.values()):  # a failed call is asked again next time
        write_json(judged_path(workdir, task_id, verifier, run.run_id), answers)
    return answers


def saved_verdict_judges(workdir: Any, task_id: str, run_id: str, verifier: Verifier) -> Optional[dict]:
    """atom id -> {score, why} off the newest Verdict the score stage kept for the Run, judge items of
    this Verifier only; None when no Verdict was kept."""
    folder = Path(workdir) / VERDICTS_DIR / task_id
    paths = sorted(folder.glob(f"{run_id}.*.json"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    wanted = set(item_ids(verifier))
    for path in reversed(paths):
        body = read_json(path, None)
        if not isinstance(body, dict) or body.get("run_id") != run_id:
            continue
        return {row["id"]: {"score": row.get("score"), "why": row.get("why") or ""}
                for row in body.get("items") or []
                if isinstance(row, dict) and row.get("kind") == "judge" and row.get("id") in wanted}
    return None


def judge_results_of(workdir: Any, task_id: str, verifier: Verifier, runs: Iterable[Run]) -> dict[str, dict]:
    """run id -> the judge answers kept for it: verify's for a judged Reference, else the score stage's."""
    out = {}
    for run in runs:
        found = cached_judge(workdir, task_id, verifier, run.run_id)
        if found is None:
            found = saved_verdict_judges(workdir, task_id, run.run_id, verifier)
        if found:
            out[run.run_id] = found
    return out
