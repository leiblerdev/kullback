"""The kullback.spec writer arm beside intentv r1, on the same sealed sample (sp-writer).

Subcommands:
  write  one writer session per task through kullback.spec.writer; stores the Spec, the counts and
         the compiled Verifier, then scores every recording of the task with the existing Verdict
  table  wrong-failed, right-failed and discrimination with Wilson intervals, and the source of
         every right-reference failure, ids only

The Intent input is the task's spoken text wrapped as one volunteered fact per sentence
(writer.spoken_intent) until a mined Intent exists on this base. Only .work-scratch is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import intentv  # noqa: E402

PRICE_MODEL = intentv.PRICE_MODEL


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _score(clone: Path, task_id: str, verifier) -> list[dict]:
    from kullback.runner import canon
    from kullback.runner.verdict import verdict

    rules = canon.load_rules(clone / "canon-rules.json")
    _, writes, _ = intentv.tool_list(clone)
    rewards = intentv.sidecar_rewards(clone)
    refs = intentv._read_json(clone / "exam" / "references.json").get(task_id, {}).get("references", [])
    id_of = {r["run_id"]: r.get("trace_id") for r in refs}
    out = []
    for run_id, path in sorted(intentv.recording_paths(clone, task_id).items()):
        try:
            result = verdict(str(path), verifier, canon=rules, rules=rules, write_tools=writes)
            scored = {"pass": bool(result.passed), "failing": getattr(result, "failing_atom", None)}
        except Exception as error:
            scored = {"pass": None, "error": type(error).__name__}
        out.append({"run_id": run_id, "is_reference": run_id in id_of,
                    "grade": rewards.get(id_of.get(run_id)), "spec": scored})
    return out


def write(args) -> int:
    intentv._load_main_env()
    from kullback.ai.provider import live_model
    from kullback.spec import writer as W

    if os.environ.get("HARNESS_ALLOW_MODEL_REQUESTS", "") not in ("1", "true", "yes", "on"):
        raise SystemExit("live model requests are off")
    sample = json.loads(args.sample.read_text(encoding="utf-8"))["tasks"]
    sample = [row for row in sample if not args.only or row["task_id"] in args.only]
    model = live_model(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for row in sample:
        if spent >= args.ceiling_usd:
            print(json.dumps({"task": row["task_id"], "status": "skipped_on_ceiling"}), flush=True)
            continue
        clone = intentv.SCRATCH / row["corpus"]
        intent = W.spoken_intent(row["task_id"], intentv.intent_text(clone, row["task_id"]))
        inputs = W.load_inputs(clone, row["task_id"], intent)
        written = W.write_spec(row["task_id"], inputs, model, model_id=PRICE_MODEL)
        spent += written.session["usd"]
        compiled = W.verifier_of(written.spec, inputs)
        out = {**row, "counts": written.counts, "compile_dropped": compiled.dropped,
               "usd": round(written.session["usd"], 4), "tool_rounds": written.session["tool_rounds"],
               "spec": written.spec.model_dump(mode="json"),
               "verifier": compiled.verifier.model_dump(mode="json"),
               "thinking": written.session["thinking"],
               "recordings": _score(clone, row["task_id"], compiled.verifier)}
        (args.out_dir / f"{row['task_id']}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"task": row["task_id"], "kept": written.counts["kept"],
                          "error": written.counts["parse_error"] or None,
                          "usd": out["usd"], "spent": round(spent, 4)}), flush=True)
    print(json.dumps({"spent_usd": round(spent, 4)}))
    return 0


def _source(row: dict, failing: str) -> str:
    """Why a right reference failed: the failing check's kind and what its because rests on."""
    if not failing or failing.startswith("extra_write"):
        return "coverage"
    if failing.startswith("forbidden."):
        return "must_not scope"
    atom_id = failing.split(".")[0]
    number = atom_id[1:] if atom_id.startswith("i") else ""
    checks = row["spec"]["checks"]
    check = checks[int(number)] if number.isdigit() and int(number) < len(checks) else None
    if check is None:
        return f"other:{failing.split(':')[0]}"
    demand = check["demand"].get("demand")
    policy = check["because"].casefold().find("section") >= 0 or not check["fact_ids"]
    if demand == "cap":
        return "policy cap" if policy else "intent cap"
    if demand == "say":
        return "unrepeated user fact"
    if demand in ("write", "shape"):
        return "explicit ask skipped" if not policy else "policy write"
    return f"{demand}:{'policy' if policy else 'intent'}"


def table(args) -> int:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.out_dir.glob("*.json"))]
    cells: dict[str, list[int]] = {}
    sources: dict[str, list[str]] = {}
    for row in rows:
        for rec in row["recordings"]:
            if not rec["is_reference"] or rec["grade"] not in (0, 1):
                continue
            side = "right" if rec["grade"] == 1 else "wrong"
            failed = rec["spec"]["pass"] is False
            for key in (f"total/{side}", f"{row['corpus']}/{side}"):
                cell = cells.setdefault(key, [0, 0])
                cell[0] += failed
                cell[1] += 1
            if side == "right" and failed:
                source = _source(row, rec["spec"].get("failing") or "")
                sources.setdefault(source, []).append(f"{row['task_id']}/{rec['run_id']}")
    out = {}
    for name in ["total"] + sorted({row["corpus"] for row in rows}):
        wrong, right = cells.get(f"{name}/wrong", [0, 0]), cells.get(f"{name}/right", [0, 0])
        out[name] = {"wrong_failed": wrong, "wrong_wilson": wilson(*wrong),
                     "right_failed": right, "right_wilson": wilson(*right),
                     "discrimination": round(wrong[0] / max(wrong[1], 1) - right[0] / max(right[1], 1), 3)}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row["counts"].items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value
        counts["compile_dropped"] = counts.get("compile_dropped", 0) + row["compile_dropped"]
        counts["silent"] = counts.get("silent", 0) + bool(row["counts"]["parse_error"])
    body = {"table": out, "right_failure_sources": sources, "counts": counts,
            "usd": round(sum(row["usd"] for row in rows), 4), "tasks": len(rows)}
    args.out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_write = sub.add_parser("write")
    p_write.add_argument("--sample", type=Path, required=True)
    p_write.add_argument("--model", default=intentv.MODEL_ID)
    p_write.add_argument("--ceiling-usd", type=float, default=3.0)
    p_write.add_argument("--out-dir", type=Path, required=True)
    p_write.add_argument("--only", nargs="*", default=None)
    p_table = sub.add_parser("table")
    p_table.add_argument("--out-dir", type=Path, required=True)
    p_table.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return write(args) if args.cmd == "write" else table(args)


if __name__ == "__main__":
    sys.exit(main())
