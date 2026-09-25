"""Arm D driver: sample failures, judge, cluster, pilot, full run.

Stages persist under R/exp/arm-d/ so a rerun resumes: failed_pool.json,
evidence.json, judge_results.json, clusters.json, tasks.jsonl, summary.json.
Spend ceiling 20 USD: no new model call once reached.
"""
import collections
import json
import sys
import time
from pathlib import Path

from exp.synth0925.arm_d import bind as bind_mod
from exp.synth0925.arm_d import cluster as cluster_mod
from exp.synth0925.arm_d import evidence as evidence_mod
from exp.synth0925.arm_d import judge as judge_mod
from exp.synth0925.arm_d import synthesize as synth_mod
from kullback import synthesise as S

CEILING_USD = 20.0
SEED = "synth0925-d"
LEVELS = ("easy", "medium", "hard")
TARGET = 60


def load(path: Path, default):
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return default


def save(path: Path, body) -> None:
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")


def main() -> None:
    workdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    pilot_only = "--pilot" in sys.argv
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    t0 = time.time()

    pool = load(outdir / "failed_pool.json", None)
    if pool is None:
        print("no failed_pool.json; run score_runs.py first")
        raise SystemExit(2)
    print(f"pool: {pool['failed']} failed of {pool['scored']} scored, "
          f"sample {len(pool['sample'])}", flush=True)

    evidence = load(outdir / "evidence.json", None)
    if evidence is None:
        evidence = evidence_mod.evidence_for(workdir, pool)
        save(outdir / "evidence.json", evidence)
    print(f"evidence: {len(evidence)} bundles", flush=True)

    verdicts = load(outdir / "judge_results.json", None)
    judge_spent = 0.0
    if verdicts is None:
        model = judge_mod.make_model()
        reader = judge_mod.Reader(workdir, evidence)
        verdicts = []
        for pos, item in enumerate(evidence):
            if judge_spent >= CEILING_USD:
                verdicts.append({"run_id": item["run_id"], "task_id": item["task_id"],
                                 "cause": "undetermined", "skill": None,
                                 "confidence": 0.0,
                                 "rationale": "ceiling reached, unjudged",
                                 "rounds": 0, "cost_usd": 0.0})
                continue
            row = judge_mod.judge_one(model, reader, item)
            judge_spent += row.get("cost_usd", 0.0)
            verdicts.append(row)
            if (pos + 1) % 10 == 0:
                save(outdir / "judge_results.json", verdicts)
                print(f"judged {pos + 1}/{len(evidence)} "
                      f"spent {judge_spent:.2f} USD", flush=True)
        save(outdir / "judge_results.json", verdicts)
    else:
        judge_spent = sum(row.get("cost_usd", 0.0) for row in verdicts)
    causes = collections.Counter(row.get("cause") for row in verdicts)
    print(f"causes: {dict(causes)} judge spent {judge_spent:.2f} USD", flush=True)

    clusters = load(outdir / "clusters.json", None)
    cluster_spent = 0.0
    if clusters is None:
        weaknesses = [row for row in verdicts if row.get("cause") == "candidate_weakness"]
        if judge_spent < CEILING_USD:
            model = judge_mod.make_model()
            clusters, cluster_spent = cluster_mod.cluster(weaknesses, model)
        else:
            clusters, cluster_spent = cluster_mod.cluster(weaknesses, None)
        save(outdir / "clusters.json", clusters)
    try:
        filed = float(load(outdir / "cluster_cost.json", {}).get("usd", 0.0))
    except (ValueError, TypeError, AttributeError):
        filed = 0.0
    cluster_spent = max(cluster_spent, filed)
    print(f"clusters: {len(clusters)} cluster spent {cluster_spent:.2f} USD", flush=True)
    for entry in clusters:
        print(f"  {entry['skill']}: {len(entry['member_run_ids'])} runs, "
              f"{len(entry['member_task_ids'])} tasks", flush=True)

    world = S.World(workdir)
    body = S.load_graph(workdir)
    observed = body.get("args") or {}
    from kullback.gates import verifier_suite
    fn = verifier_suite.canon_fn(world.canon_rules)
    synth = synth_mod.make_synth(workdir)
    spent = judge_spent + cluster_spent

    raw = outdir / "tasks.jsonl"
    lines = []
    if raw.is_file():
        for text in raw.read_text(encoding="utf-8").splitlines():
            text = text.strip()
            if text:
                try:
                    lines.append(json.loads(text))
                except ValueError:
                    continue
    counts = collections.Counter(line.get("level") for line in lines
                                 if line.get("status") == "admitted")
    per_level = TARGET // 3
    limit = 10 if pilot_only else TARGET
    scenarios: dict = load(outdir / "scenarios.json", {})
    round_robin = [entry for entry in clusters]
    ci = 0
    made = len(lines)
    spent += sum(line.get("cost_usd", 0.0) for line in lines)
    while len(lines) < limit and spent < CEILING_USD:
        if pilot_only and len(lines) >= 10:
            break
        if not pilot_only and all(counts.get(level, 0) >= per_level for level in LEVELS):
            break
        if not round_robin:
            break
        entry = round_robin[ci % len(round_robin)]
        ci += 1
        asked = collections.Counter(line.get("asked_level", line.get("level"))
                                    for line in lines)
        level = min(LEVELS, key=lambda lv: asked.get(lv, 0))
        if not pilot_only and asked.get(level, 0) >= per_level:
            if all(asked.get(lv, 0) >= per_level for lv in LEVELS):
                break
            continue
        key = f"{entry['skill']}::{level}"
        if key not in scenarios and spent < CEILING_USD:
            write_calls = synth_mod.cluster_writes(entry, evidence)
            scenario, cost = synth_mod.scenario_for(
                synth, entry["skill"], entry.get("evidence", ""), level)
            spent += cost
            scenarios[key] = {"scenario": scenario, "write_calls": write_calls,
                              "cost_usd": cost}
            save(outdir / "scenarios.json", scenarios)
        scenario = scenarios.get(key, {}).get("scenario") or {}
        write_calls = scenarios.get(key, {}).get("write_calls", "")
        if isinstance(write_calls, str):
            write_calls = [write_calls] if write_calls else []
        excluded = synth_mod.member_rows(entry, evidence)
        run_walk = bind_mod.make_run_walk(body, bind_mod.make_bind(),
                                          excluded=excluded)
        ident = f"{SEED}:{entry['skill']}:{level}:{made}"
        line = synth_mod.build_task(world, body, observed, run_walk, entry,
                                    write_calls, level, scenario, synth,
                                    ident, fn, SEED, made)
        spent += line.get("cost_usd", 0.0)
        lines.append(line)
        with open(raw, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        if line.get("status") == "admitted":
            counts[line["level"]] += 1
        made += 1
        print(f"task {made}: {line['level']} {line['status']} "
              f"rev {line['revisions']} {line.get('reasons','')[:80]}", flush=True)

    summary = {
        "seed": SEED, "failed_pool": {key: pool[key] for key in
                                      ("scored", "passed", "failed",
                                       "tasks_with_failed") if key in pool},
        "causes": dict(causes),
        "clusters": [{"skill": entry["skill"],
                      "runs": len(entry["member_run_ids"]),
                      "tasks": len(entry["member_task_ids"])} for entry in clusters],
        "admitted_per_level": {level: counts.get(level, 0) for level in LEVELS},
        "admitted": sum(1 for line in lines if line.get("status") == "admitted"),
        "rejected": sum(1 for line in lines if line.get("status") == "rejected"),
        "tasks_per_cluster": {entry["skill"]: {
            "admitted": sum(1 for line in lines
                            if line.get("cluster") == entry["skill"]
                            and line.get("status") == "admitted"),
            "rejected": sum(1 for line in lines
                            if line.get("cluster") == entry["skill"]
                            and line.get("status") == "rejected")}
            for entry in clusters},
        "revisions": dict(collections.Counter(str(line.get("revisions", 0))
                                              for line in lines)),
        "checks": {key: sum(1 for line in lines
                            if (line.get("checks") or {}).get(key) is True)
                   for key in ("runs", "do_nothing_fails", "prefix_fails",
                               "suite_passed")},
        "judge_usd": round(judge_spent, 4),
        "cluster_usd": round(cluster_spent, 4),
        "synth_usd": round(spent - judge_spent - cluster_spent, 4),
        "total_usd": round(spent, 4),
        "wall_s": round(time.monotonic() - started, 1),
        "finished_at": t0,
    }
    save(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
