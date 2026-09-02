"""How often the D111 Reference rule agrees with a benchmark's own reward, per build (D111, D112).

Scaffolding, not part of the build: the build never reads a reward, this script reads it afterwards
from the grader sidecars ingest set aside (`grader/<hash>.json`, tau2's `reward`) and prints two
numbers. Kept: of the recordings the rule kept as References, the share the benchmark also passed.
Found: of the recordings the benchmark passed, the share the rule kept. Per compiled rule it also
prints how many recordings the rule failed and how many of those the benchmark passed, which is
what calibrated `reference.MISCOMPILED_SHARE` (D114). It exists to catch a wrong
rule where an answer key happens to exist, before the same rule runs on a customer's traces, where
none does. D112 says to delete it once the rule holds across the domains.

    uv run python scripts/reference_agreement.py .work-retail [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def rewards(workdir: Path) -> dict[str, Optional[float]]:
    """trace_id -> the benchmark's reward for that recording, where the sidecar carries one."""
    by_hash: dict[str, Optional[float]] = {}
    for path in sorted((workdir / "grader").glob("*.json")):
        fields = (_read(path, {}) or {}).get("fields") or {}
        reward = fields.get("reward")
        if reward is None:
            reward = (fields.get("reward_info") or {}).get("reward")
        by_hash[path.stem] = float(reward) if reward is not None else None
    out: dict[str, Optional[float]] = {}
    for path in sorted((workdir / "traces").glob("*.json")):
        trace = _read(path, {}) or {}
        if trace.get("hash") in by_hash:
            out[trace["trace_id"]] = by_hash[trace["hash"]]
    return out


def compare(references: dict, reward_of: dict[str, Optional[float]]) -> dict:
    """The verdict the rule gave each recording against the reward the benchmark gave it."""
    rows = []
    for task_id, row in sorted(references.items()):
        kept = {r["trace_id"] for r in row.get("references", []) if r.get("trace_id")}
        trace_of = {r["run_id"]: r.get("trace_id") for r in row.get("recordings", []) if r.get("trace_id")}
        failed = {trace_of.get(rid, rid): why for rid, why in (row.get("failed") or {}).items()
                  if trace_of.get(rid, rid) in reward_of}
        for trace_id in sorted(kept | set(failed)):
            reward = reward_of.get(trace_id)
            if reward is None:
                continue
            verdict = "reference" if trace_id in kept else "failed"
            rows.append({"task_id": task_id, "trace_id": trace_id, "rule": verdict, "reward": reward,
                         "why": failed.get(trace_id), "unconfirmed_reason": row.get("reason")})
        if not kept and not failed and row.get("reason"):
            rows.append({"task_id": task_id, "trace_id": None, "rule": "unconfirmed", "reward": None,
                         "why": None, "unconfirmed_reason": row.get("reason")})
    kept_rows = [r for r in rows if r["rule"] == "reference"]
    passed_rows = [r for r in rows if r["reward"] == 1.0]
    per_rule: dict[str, dict] = {}
    for r in rows:
        if r["rule"] == "failed" and (r["why"] or "").startswith("violates"):
            for cid in re.findall(r"c_[0-9a-f]+", r["why"]):
                cell = per_rule.setdefault(cid, {"failed": 0, "passed": 0})
                cell["failed"] += 1
                cell["passed"] += int(r["reward"] == 1.0)
    return {
        "per_rule": per_rule,
        "recordings_scored": len([r for r in rows if r["trace_id"]]),
        "kept": len(kept_rows),
        "kept_and_passed": sum(1 for r in kept_rows if r["reward"] == 1.0),
        "passed": len(passed_rows),
        "passed_and_kept": sum(1 for r in passed_rows if r["rule"] == "reference"),
        "failed_by_rule": sum(1 for r in rows if r["rule"] == "failed"),
        "failed_and_benchmark_failed": sum(1 for r in rows if r["rule"] == "failed" and r["reward"] == 0.0),
        "tasks_unconfirmed": sum(1 for r in rows if r["rule"] == "unconfirmed"),
        "rows": rows,
    }


def report(result: dict) -> str:
    def share(num: int, den: int) -> str:
        return f"{100 * num / den:.1f}% ({num}/{den})" if den else "n/a"
    return "\n".join([
        f"recordings with a benchmark reward: {result['recordings_scored']}",
        f"kept as References and the benchmark passed them too: {share(result['kept_and_passed'], result['kept'])}",
        f"benchmark passed and the rule kept: {share(result['passed_and_kept'], result['passed'])}",
        f"failed by the rule and the benchmark failed too: "
        f"{share(result['failed_and_benchmark_failed'], result['failed_by_rule'])}",
        f"Tasks with no Reference: {result['tasks_unconfirmed']}",
    ] + [f"rule {cid}: failed {cell['failed']} recordings, the benchmark passed {cell['passed']} of them"
         for cid, cell in sorted(result["per_rule"].items(), key=lambda kv: -kv[1]["failed"])])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="Also write the rows here (reference_verdicts.json).")
    args = parser.parse_args(argv)
    references = _read(args.workdir / "references.json", None)
    if references is None:
        print(f"no references.json under {args.workdir}; run the build first")
        return 2
    result = compare(references, rewards(args.workdir))
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
