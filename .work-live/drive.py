"""lv-run stage driver (r4): fresh blind Specs, two Examiner rounds, two fresh Runs per
Task, judge scoring on every fresh Run, tiers and the Reference witness; one corpus per process.

Adapted from shape-1007-fold/.work-shape/drive.py: same spend arithmetic (stage deltas over a base
captured before the first stage plus witness judge usd; score-stage judges price through a
BudgetedModel on the judge stage, so the ledger holds them), same rounds and witness shape, plus
the runs and score stages the Builder path never reached. No Task outside .work-live/sample.json.
Change no code under kullback/ from here; if a stage breaks the driver stops and the log says so.
"""
import asyncio
import collections
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from kullback.agent.bus import Bus
from kullback.ai.provider import live_model
from kullback.builder import domain_tools as DT
from kullback.examiner.session import review_specs
from kullback.judge.items import judge_items, tally
from kullback.report import load_tool_sigs
from kullback.runner.budget import BudgetedModel, EffortModel
from kullback.runner.canon import load_rules
from kullback.runner.code_hash import CODE_HASH
from kullback.runner.judge import (
    JUDGE_VERSION,
    AgenticJudge,
    judge_atom_results,
    judge_name,
    third_judge,
)
from kullback.runner.records import EntitySchema
from kullback.runner.regrade import regrade_run
from kullback.runner.verdict import load_run, verdict
from kullback.spec.rounds import run_rounds
from kullback.spec.schema import load_spec
from kullback.spec.split import reference_classes
from kullback.spec.stage import write_specs
from kullback.spec.trust import intent_tiers, load_spec_verifier, store_tasks
from kullback.spec.witness import reference_run

OPUS = "bedrock/global.anthropic.claude-opus-5-5"
MUSE = "opencode-go/muse-spark-1.3-contributor"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ATTIC = HERE / "attic" / "pre-r4"
CAP, STOP_AT = 25.0, 22.0
ROUNDS_CEILING = 6.0


def log(corpus, **row):
    line = json.dumps({"corpus": corpus, **row}, default=str)
    with open(HERE / f"drive-{corpus}.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def stage_usd(work):
    doc = json.loads((work / "budget.json").read_text())
    return {k: float(v.get("usd") or 0.0) for k, v in (doc.get("stages") or {}).items()}


def base(work):
    OUT.mkdir(exist_ok=True)
    path = OUT / "drive_base.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    if work.name not in doc:
        doc[work.name] = stage_usd(work)
        path.write_text(json.dumps(doc, indent=1))
    return doc[work.name]


def judge_usd(work):
    total = 0.0
    for name in (f"witness-{work.name}.json", f"score-{work.name}.json"):
        path = OUT / name
        if path.exists():
            total += float(json.loads(path.read_text()).get("judge_usd") or 0.0)
    return total


def spent(work):
    b = base(work)
    return round(sum(v - b.get(k, 0.0) for k, v in stage_usd(work).items()) + judge_usd(work), 4)


def sample(work):
    return json.loads((HERE / "sample.json").read_text())[work.name]


def progress(name, work):
    path = OUT / f"{name}-{work.name}.json"
    return json.loads(path.read_text()) if path.exists() else {"tasks": {}}


def save_progress(name, work, doc):
    (OUT / f"{name}-{work.name}.json").write_text(json.dumps(doc, indent=1, default=str))


def stage_spec(work, model):
    sample_ids = sample(work)
    store_tasks(work, sample_ids)
    log(work.name, stage="manifest", tasks=len(sample_ids))
    dest = ATTIC / work.name
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for task_id in sample_ids:
        for rel in (f"spec/{task_id}.json", f"spec/verifiers/{task_id}.json",
                    f"verifiers/{task_id}.json", f"exam/rulings/{task_id}",
                    f"exam/review/views/{task_id}.json"):
            src = work / rel
            if src.exists() or src.is_symlink():
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(target))
                moved.append(rel)
    log(work.name, stage="attic", moved=len(moved))
    bus = Bus(work / "bus.jsonl", agent="spec")
    done = progress("spec", work)["tasks"]
    for task_id in sample_ids:
        if task_id in done or spent(work) >= STOP_AT:
            continue
        t0 = time.time()
        try:
            counts = write_specs(work, [task_id], model, bus=bus)
            spec = load_spec(work, task_id)
            done[task_id] = {"counts": counts, "seconds": round(time.time() - t0, 1),
                             "set_aside": spec.set_aside if spec else "missing"}
        except Exception as error:
            done[task_id] = {"failed": type(error).__name__, "seconds": round(time.time() - t0, 1)}
        save_progress("spec", work, {"tasks": done})
        row = {"stage": "spec", "task": task_id[-4:], "spent": spent(work), **done[task_id]}
        log(work.name, **{k: v for k, v in row.items() if k != "counts"})
    return done


def guarded(workdir, task_ids, *, model, round_number):
    if spent(Path(workdir)) >= STOP_AT:
        return {"round": round_number, "tasks": [], "capped": False, "counts": {},
                "stopped_at_cap": spent(Path(workdir))}
    t0 = time.time()
    out = review_specs(workdir, task_ids, model=model, round_number=round_number, max_turns=36)
    log(Path(workdir).name, stage="examiner", round=round_number, capped=out.get("capped"),
        seconds=round(time.time() - t0), spent=spent(Path(workdir)), counts=out.get("counts"))
    return out


def stage_rounds(work, model):
    if spent(work) >= STOP_AT:
        return {"stopped_at_cap": spent(work)}
    out = run_rounds(work, sample(work), examine=guarded, model=model, writer_model=model,
                     ceiling_usd=ROUNDS_CEILING)
    (OUT / f"rounds-{work.name}.json").write_text(json.dumps(out, indent=1, default=str))
    log(work.name, stage="rounds", held=[t[-4:] for t in out["held"]], total=out["total"],
        spent=spent(work))
    return out


def run_tools(work, runner, user):
    tools = DT.domain_tools(workdir=work, model=None, run_model=runner, user_model=user)
    return {tool.name: tool for tool in tools}["run"]


def stage_runs(work, runner, user):
    from kullback.runner.world.environment import BuiltEnvironment

    env = BuiltEnvironment(work)
    run_tool = run_tools(work, runner, user)
    done = progress("runs", work)["tasks"]
    for task_id in sample(work):
        if task_id in done or spent(work) >= STOP_AT:
            continue
        try:
            if not (work / "verifiers" / f"{task_id}.json").is_file():
                done[task_id] = {"skipped": "no verifier"}
            elif env.rules(env.task(task_id)) is None:
                done[task_id] = {"skipped": "no user rules"}
            else:
                res = asyncio.run(run_tool.run({"task_ids": [task_id], "count": 2}))
                body = res.details or {}
                done[task_id] = {
                    "runs": [{"run_id": r.get("run_id"), "ended": r.get("termination_reason"),
                              "user_end": r.get("user_end"),
                              "passed": ((r.get("verdict") or {}).get("passed")
                                         if r.get("verdict") else None)}
                             for r in body.get("runs", [])],
                    "not_run": body.get("not_run", []), "reasons": body.get("reasons", []),
                    "error": res.content if res.is_error else None}
        except Exception:
            done[task_id] = {"failed": traceback.format_exc(limit=3).splitlines()[-1]}
        save_progress("runs", work, {"tasks": done})
        log(work.name, stage="runs", task=task_id[-4:], spent=spent(work), **done[task_id])
    return done


def fresh_paths(work, task_id):
    return sorted((work / "runs").glob(f"{task_id}-*.jsonl"))


def stage_score(work, judge_model):
    canon = load_rules(work / "canon-rules.json")
    sigs = load_tool_sigs(work)
    write_tools = {s.name for s in sigs if s.kind == "write"} or None
    flagged = {s.name for s in sigs if s.unclassified}
    schema_path = work / "schema.json"
    schema = EntitySchema.model_validate(json.loads(schema_path.read_text())) if schema_path.is_file() else None
    env_path = work / "environment.json"
    environment = None
    if env_path.is_file():
        from kullback.runner.records import Environment

        environment = Environment.model_validate(json.loads(env_path.read_text()))
    common = dict(environment=environment, runner_version=CODE_HASH, schema=schema,
                  write_tools=write_tools, flagged_tools=flagged, rules=canon)
    first = AgenticJudge(judge_model, name=judge_name(OPUS, "a"))
    judges = (first, third_judge(first, name=judge_name(OPUS, "b")))
    done = progress("score", work)["tasks"]
    for task_id in sample(work):
        if task_id in done or spent(work) >= STOP_AT:
            continue
        try:
            verifier = load_spec_verifier(work, task_id)
            if verifier is None:
                done[task_id] = {"skipped": "no verifier"}
            else:
                scored = []
                for path in fresh_paths(work, task_id):
                    run = load_run(path)
                    results = judge_atom_results(verifier, run, judges[0], judges[1],
                                                 workdir=work, run_id=run.run_id)
                    results.update(judge_items(judge_model, verifier.atoms, run))
                    judged = regrade_run(path, verifier, canon,
                                         out_dir=work / "verdicts" / task_id,
                                         judge_results=results, judge_version=JUDGE_VERSION,
                                         **common)
                    scored.append({"run_id": run.run_id, "passed": judged.passed,
                                   "score": judged.score, "judge_used": judged.judge_used,
                                   "open": [i.id for i in judged.items if i.holds is None]})
                done[task_id] = {"runs": scored} if scored else {"skipped": "no fresh runs"}
        except Exception:
            done[task_id] = {"failed": traceback.format_exc(limit=3).splitlines()[-1]}
        save_progress("score", work, {"tasks": done})
        log(work.name, stage="score", task=task_id[-4:], spent=spent(work), **done[task_id])
    return done


def stage_tiers(work):
    tiers = intent_tiers(work)
    classes = reference_classes(work) or {}
    sample_ids = set(sample(work))
    rows = {t[-4:]: {"tier": v.tier, "reason": v.row.get("reason"), "gates": v.row.get("gates"),
                     "reference_passes": v.row.get("reference_passes"),
                     "solvable": v.row.get("solvable"),
                     "fresh": v.row.get("fresh_runs"), "fresh_passed": v.row.get("fresh_passed"),
                     "reference_class": classes.get(t, "unknown")}
            for t, v in tiers.items() if t in sample_ids}
    out = {"tiers": dict(collections.Counter(r["tier"] for r in rows.values())),
           "reasons": dict(collections.Counter(str(r["reason"]) for r in rows.values())),
           "tasks": rows}
    (OUT / f"tiers-{work.name}.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    log(work.name, stage="tiers", **{k: out[k] for k in ("tiers", "reasons")}, spent=spent(work))
    return out


def stage_witness(work, model):
    path = OUT / f"witness-{work.name}.json"
    done = json.loads(path.read_text()) if path.exists() else {"tasks": {}, "judge_usd": 0.0}
    canon = load_rules(work / "canon-rules.json")
    for task_id in sample(work):
        if task_id[-4:] in done["tasks"] or spent(work) >= STOP_AT:
            continue
        verifier, run = load_spec_verifier(work, task_id), reference_run(work, task_id)
        if verifier is None or run is None:
            done["tasks"][task_id[-4:]] = {"skipped": "no verifier" if verifier is None else "no reference"}
            continue
        results = judge_items(model, verifier.atoms, run, model_id=OPUS)
        judged = verdict(run, verifier, canon, judge_results=results)
        done["judge_usd"] += sum(r.cost_usd for r in results.values())
        done["tasks"][task_id[-4:]] = {
            "passed": judged.passed, "score": judged.score,
            "failing": str(judged.failing_atom or "").split(":")[0] or None,
            "judge": tally(results.values()),
            "answerable": sum(1 for r in results.values() if r.score is not None),
            "items": [{"kind": i.kind, "gate": i.gate, "holds": i.holds} for i in judged.items]}
        path.write_text(json.dumps(done, indent=1, default=str))
        log(work.name, stage="witness", task=task_id[-4:], passed=judged.passed,
            score=judged.score, spent=spent(work))


def main(corpus):
    work = HERE / corpus
    log(corpus, stage="start", sample=len(sample(work)))
    opus = live_model(OPUS)
    muse = EffortModel(live_model(MUSE), "low")
    judge_wrapped = BudgetedModel(live_model(OPUS), stage="judge", workdir=work,
                                  model_id=OPUS, cap_context=False)
    stage_spec(work, opus)
    stage_rounds(work, opus)
    stage_runs(work, muse, muse)
    stage_score(work, judge_wrapped)
    stage_tiers(work)
    stage_witness(work, opus)
    log(corpus, stage="done", spent=spent(work), cap=CAP)


if __name__ == "__main__":
    main(sys.argv[1])
