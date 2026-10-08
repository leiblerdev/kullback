"""Read-only helper (r5): rebuild per-Run rows from disk for the report.

For each fresh Run file of a sampled Task created after the driver base was captured:
termination reason, last user goal flag, and the stored Verdict's pass/score/judge_use.
Writes nothing; the runs progress file stays as the live process wrote it.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cutoff():
    base = HERE / "out" / "drive_base.json"
    return base.stat().st_mtime if base.exists() else 0.0


def verdict_for(work, task_id, run_id):
    folder = work / "verdicts" / task_id
    if not folder.is_dir():
        return None
    cands = sorted(folder.glob(f"{run_id}.*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    try:
        body = json.loads(cands[-1].read_text())
    except (OSError, ValueError):
        return None
    return {"passed": body.get("passed"), "score": body.get("score"),
            "judge_used": body.get("judge_used")}


def rows(work, task_id):
    out = []
    for path in sorted((work / "runs").glob(f"{task_id}-*.jsonl")):
        try:
            if path.stat().st_mtime < cutoff():
                continue
            events = [json.loads(line) for line in path.open()]
        except (OSError, ValueError):
            continue
        if not events:
            continue
        first = events[0]
        run_id = (first.get("run_id") if isinstance(first, dict) else None) or path.stem
        terms = [e.get("termination_reason") for e in events
                 if isinstance(e, dict) and e.get("termination_reason")]
        goals = [e.get("payload", {}).get("user_goal_met") for e in events
                 if isinstance(e, dict) and isinstance(e.get("payload"), dict)
                 and "user_goal_met" in (e.get("payload") or {})]
        out.append({"run_id": run_id, "file": path.name,
                    "ended": terms[-1] if terms else None,
                    "goal_met_last": goals[-1] if goals else None,
                    "verdict": verdict_for(work, task_id, run_id)})
    return out


def main(corpus):
    work = HERE / corpus
    sample = json.loads((HERE / "sample.json").read_text())[corpus]
    print(json.dumps({t[-4:]: rows(work, t) for t in sample}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
