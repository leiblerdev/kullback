"""Step 1: score every stored Run against its Task Verifier, keep the failed pool.

Reads only. No tool, corpus or model names. All names come from the workdir.
"""
import collections
import json
import sys
from pathlib import Path

from kullback import difficulty
from kullback.runner import canon as canon_mod
from kullback.runner import target as target_mod
from kullback.runner import verdict as verdict_mod
from kullback.runner.records import Verifier, read_json
from kullback.sampling import sample_key


def load_verifier(workdir: Path, task_id: str):
    body = read_json(workdir / "verifiers" / f"{task_id}.json", None)
    if not body:
        return None
    try:
        return Verifier.model_validate(body)
    except Exception:
        return None



def score_all(workdir: Path) -> list[dict]:
    canon_rules = canon_mod.load_rules(workdir / "canon-rules.json")
    status = read_json(workdir / "task_status.json", {}) or {}
    paths = difficulty.run_paths(workdir)
    verifiers = {}
    ref_ids: dict[str, set[str]] = {}
    for task_id, row in status.items():
        verifiers[task_id] = load_verifier(workdir, task_id)
        ids = set()
        for value in (row or {}).get("reference_run_ids") or []:
            ids.add(str(value))
        ref_ids[task_id] = ids
    out = []
    for run_id, rel in paths.items():
        path = workdir / rel if not str(rel).startswith("/") else Path(rel)
        try:
            run = verdict_mod.load_run(path)
        except Exception:
            continue
        task_id = getattr(run, "task_id", None)
        if task_id not in verifiers or verifiers[task_id] is None:
            continue
        if run_id in ref_ids[task_id] or getattr(run, "run_id", "") in ref_ids[task_id]:
            continue
        try:
            passed, reason = target_mod.check_run(verifiers[task_id], run, canon_rules)
        except Exception as error:
            passed, reason = False, f"score_error:{type(error).__name__}"
        out.append({"run_id": getattr(run, "run_id", run_id), "task_id": task_id,
                    "passed": bool(passed), "reason": str(reason or "")})
    return out


def main() -> None:
    workdir = Path(sys.argv[1])
    seed = sys.argv[2] if len(sys.argv) > 2 else "synth0925-d"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else workdir
    scored = score_all(workdir)
    failed = [row for row in scored if not row["passed"]]
    order = sorted(f"{row['task_id']}:{row['run_id']}" for row in failed)
    ranked = sorted(order, key=lambda key: sample_key("arm-d-failed", f"{seed}:{key}", ""))
    sample = ranked[:80]
    by_task = collections.Counter(row["task_id"] for row in failed)
    by_reason = collections.Counter(row["reason"].split(":")[0] for row in failed)
    pool = {"seed": seed, "scored": len(scored),
            "passed": sum(1 for row in scored if row["passed"]),
            "failed": len(failed), "tasks_with_failed": len(by_task),
            "by_reason_head": dict(by_reason.most_common(20)),
            "failed_runs": failed, "sample": sample}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "failed_pool.json").write_text(json.dumps(pool, indent=1), encoding="utf-8")
    print(f"scored {len(scored)} passed {pool['passed']} failed {len(failed)} "
          f"tasks {len(by_task)} sample {len(sample)}")


if __name__ == "__main__":
    main()
