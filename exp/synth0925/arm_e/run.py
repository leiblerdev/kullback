"""Arm E run: glued validated units against one direct long blueprint.

Paired design: the same 30 goal sets go through both constructions, so pairs
where one passed and the other failed are directly comparable. Keyed seed
synth0925-e for every random draw. Ceiling 10 USD across both constructions.

Usage: PYTHONPATH=$PWD uv run python -m exp.synth0925.arm_e.run [--pilot-only]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from exp.synth0925.arm_e import birth, prompts, units
from kullback import synthesise as S
from kullback.runner import budget as budget_mod
from kullback.sampling import sample_key

RETAIL = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.claude/herdr/research-0925/rs-synth")
OUT = RETAIL / "exp" / "arm-e"
WORKDIR = OUT / "retail"
SEED = "synth0925-e"
CEILING = 10.0
SYNTH_MODEL = "opencode-go/muse-spark-1.3-contributor"
SETS = 30
REVISIONS = 3


def stage_usd(workdir: Any, stage: str) -> float:
    try:
        totals = budget_mod.load_totals(workdir)
    except (OSError, ValueError):
        return 0.0
    return float(((totals.get("stages") or {}).get(stage) or {}).get("usd") or 0.0)


def live_models(workdir: Any):
    from pathlib import Path as _P

    from kullback.ai import provider
    from kullback.runner import budget as _budget

    provider.load_dotenv(_P("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env"))
    provider.enable_live_calls_from_env()
    out = {}
    for stage in ("synth_glued", "synth_direct"):
        inner = provider.live_model(SYNTH_MODEL)
        memo = provider.MemoModel(inner, Path(workdir))
        out[stage] = _budget.BudgetedModel(
            memo,
            stage=stage,
            workdir=Path(workdir),
            model_id=SYNTH_MODEL,
            ceiling=None,
            prompt_cache_key=f"arm-e-{stage}",
        )
    return out


def tool_lines(world: Any) -> list[str]:
    lines = []
    for sig in sorted(world.sigs, key=lambda s: s.name):
        if sig.name not in world.write_tools:
            continue
        _props, req = S.declared(sig)
        lines.append(f"{sig.name} requires {', '.join(req) or 'nothing'}")
    return lines


def to_calls(world: Any, sigs: dict, writes: list[dict], ident: str) -> tuple[list[dict], str]:
    """Model-proposed writes into replayable call rows, declared args only."""
    calls = []
    for index, want in enumerate(writes):
        name = str(want.get("name") or "")
        sig = sigs.get(name)
        if sig is None or name not in world.write_tools:
            return [], f"unknown_step:{name}"
        properties, required = S.declared(sig)
        given = want.get("args") or {}
        args = {}
        for key, value in given.items():
            if key in properties:
                args[key] = S._typed(properties.get(key) or {}, value)
        for key in required:
            if key not in args:
                return [], "unbound"
        calls.append(
            {
                "id": f"{ident}-{index}",
                "name": name,
                "args": args,
                "requestor": "assistant",
                "turn": 1,
                "result": None,
                "error": None,
            }
        )
    return calls, ""


def run_calls(world: Any, calls: list[dict]) -> str:
    """Run the blueprint's writes in one fresh world. Failure class or empty."""
    toolkit = world.toolkit(world.fresh_db())
    for call in calls:
        body = getattr(toolkit, str(call["name"]), None)
        if body is None:
            return "crashed"
        try:
            answer = body(**call["args"])
        except Exception as error:  # noqa: BLE001 - the class is the ruling
            from kullback.gates.tool_runs import CRASH_ERRORS

            return "crashed" if type(error).__name__ in CRASH_ERRORS else "refused"
        call["result"] = answer
    return ""


def draw_sets(valid: list[dict], anchors: list[str], n: int, seed: str) -> list[dict]:
    sets = []
    i = 0
    guard = 0
    while len(sets) < n and guard < n * 60:
        guard += 1
        size = 2 if (len(sets) % 2 == 0) else 3
        a = sample_key("arm-e-pick", f"{seed}:{i}:a", "") % len(valid)
        b = sample_key("arm-e-pick", f"{seed}:{i}:b", "") % len(valid)
        c = sample_key("arm-e-pick", f"{seed}:{i}:c", "") % len(valid)
        i += 1
        idx = [a, b] if size == 2 else [a, b, c]
        if len(set(idx)) != len(idx):
            continue
        got = [valid[k] for k in idx]
        if len({g["tool"] for g in got}) != len(got):
            continue
        anchor = anchors[sample_key("arm-e-anchor", f"{seed}:{len(sets)}", "") % len(anchors)]
        sets.append({"units": got, "anchor": anchor, "size": size})
    return sets


def admit(
    world: Any,
    body: dict,
    sigs: dict,
    askable: set[str],
    calls: list[dict],
    request: str,
    must_tell: list[str],
    task_id: str,
    salt: str,
    set_no: int,
    unit_n: int,
    level: str,
    extra_facts: list[dict],
) -> tuple[dict, dict]:
    said, held = S.facts_of(calls, askable)
    said = said + [f for f in extra_facts if f not in said]
    made = birth.make_task(world, body, calls, request, said, held, f"e{unit_n}", task_id, salt, task_id)
    if made is None:
        return {}, {
            "runs": False,
            "do_nothing_fails": False,
            "prefix_fails": False,
            "suite_passed": False,
            "suite_failures": ["replay_failed"],
        }
    checks = birth.birth_checks(world, made, calls, request)
    checks["suite_passed"] = bool(made["suite_passed"])
    checks["suite_failures"] = list(made["suite_failures"])
    return made, checks


def line_for(
    task_id: str,
    construction: str,
    level: str,
    unit_n: int,
    set_no: int,
    request: str,
    said: list[dict],
    calls: list[dict],
    must_tell: list[str],
    checks: dict,
    revisions: int,
    status: str,
    reasons: list[str],
    cost: float,
) -> dict:
    writes = [{"name": c["name"], "args": c["args"]} for c in calls if c.get("name") is not None]
    return {
        "task_id": task_id,
        "arm": "e",
        "construction": construction,
        "kind": "do",
        "level": level,
        "units": unit_n,
        "scenario_id": None,
        "goal_set": set_no,
        "request": request,
        "user_facts": said,
        "blueprint_writes": writes,
        "must_tell": must_tell,
        "checks": {
            "runs": bool(checks.get("runs")),
            "do_nothing_fails": bool(checks.get("do_nothing_fails")),
            "prefix_fails": bool(checks.get("prefix_fails")),
            "suite_passed": bool(checks.get("suite_passed")),
            "suite_failures": list(checks.get("suite_failures") or []),
        },
        "revisions": revisions,
        "status": status,
        "reasons": reasons,
        "cost_usd": round(cost, 6),
    }


def build_glued(
    world: Any, body: dict, sigs: dict, askable: set[str], model: Any, goal_set: dict, set_no: int, salt: str
) -> tuple[dict, list[str]]:
    """Concatenate validated units, rerun in one world, one request over all goals."""
    reasons: list[str] = []
    lines, _blocks, said_all, held_all = zip(*(prompts.goal_line(u, askable) for u in goal_set["units"]), strict=True)
    facts = [dict(f) for f in said_all[0]]
    seen = {(f["field"], json.dumps(f["value"], default=str)) for f in facts}
    for extra in said_all[1:]:
        for f in extra:
            key = (f["field"], json.dumps(f["value"], default=str))
            if key not in seen:
                seen.add(key)
                facts.append(dict(f))
    held = [dict(f) for f in held_all[0]]
    for extra in held_all[1:]:
        held.extend(dict(f) for f in extra)
    for rev in range(REVISIONS + 1):
        ident = f"e:glued:{set_no}:r{rev}"
        anchor = goal_set["anchor"] if rev == 0 else goal_set["alt"][rev - 1]
        bound = []
        failed = ""
        for ui, unit in enumerate(goal_set["units"]):
            args, why = units.bind_unit(world, sigs, unit, anchor, f"{ident}:u{ui}")
            if args is None:
                failed = f"unit_solo_{why}"
                break
            bound.append(
                {
                    "id": f"{ident}-{len(bound)}",
                    "name": unit["tool"],
                    "args": args,
                    "requestor": "assistant",
                    "turn": 1,
                    "result": None,
                    "error": None,
                }
            )
        if failed:
            reasons.append(failed)
            continue
        why = run_calls(world, bound)
        if why:
            reasons.append(f"glued_{why}")
            continue
        note = f"attempt {rev}: " + reasons[-1] if reasons else ""
        prop, reason = prompts.glued_request(model, list(lines), facts, note)
        if reason:
            reasons.append(reason)
            continue
        leak = prompts.leaked(prop["request"], held)
        if leak:
            reasons.append(f"request_leak:{'+'.join(sorted(set(leak)))}")
            continue
        return {
            "calls": bound,
            "request": prop["request"],
            "must_tell": prop["must_tell"],
            "said": facts,
            "held": held,
            "revisions": rev,
        }, reasons
    return {}, reasons


def build_direct(
    world: Any, sigs: dict, askable: set[str], model: Any, tools: list[str], goal_set: dict, set_no: int, salt: str
) -> tuple[dict, list[str]]:
    """One long blueprint straight from the synthesiser: goals and tools only, no calls."""
    reasons: list[str] = []
    lines, blocks, said_all, held_all = zip(*(prompts.goal_line(u, askable) for u in goal_set["units"]), strict=True)
    facts = [dict(f) for f in said_all[0]]
    seen = {(f["field"], json.dumps(f["value"], default=str)) for f in facts}
    for extra in said_all[1:]:
        for f in extra:
            key = (f["field"], json.dumps(f["value"], default=str))
            if key not in seen:
                seen.add(key)
                facts.append(dict(f))
    held = [dict(f) for f in held_all[0]]
    for extra in held_all[1:]:
        held.extend(dict(f) for f in extra)
    _ = salt
    for rev in range(REVISIONS + 1):
        ident = f"e:direct:{set_no}:r{rev}"
        note = f"attempt {rev}: " + reasons[-1] if reasons else ""
        prop, reason = prompts.direct_blueprint(model, list(lines), list(blocks), tools, facts, note)
        if reason:
            reasons.append(reason if rev else f"direct_{reason}")
            if rev < REVISIONS:
                continue
            return {}, reasons
        calls, why = to_calls(world, sigs, prop["writes"], ident)
        if why:
            reasons.append(f"direct_{why}")
            continue
        if len(calls) < 2:
            reasons.append("direct_shape:fewer_than_two_writes")
            continue
        why = run_calls(world, calls)
        if why:
            reasons.append(f"direct_{why}")
            continue
        leak = prompts.leaked(prop["request"], held)
        if leak:
            reasons.append(f"request_leak:{'+'.join(sorted(set(leak)))}")
            continue
        return {
            "calls": calls,
            "request": prop["request"],
            "must_tell": prop["must_tell"],
            "said": facts,
            "held": held,
            "revisions": rev,
        }, reasons
    return {}, reasons


def main() -> None:
    t0 = time.time()
    pilot_only = "--pilot-only" in sys.argv
    world = S.World(WORKDIR)
    sigs = {s.name: s for s in world.sigs}
    body = S.load_graph(WORKDIR)
    askable = units.askable_fields(WORKDIR)
    models = live_models(WORKDIR)
    tools = tool_lines(world)

    trusted = units.trusted_ids(WORKDIR)
    mined, mstats = units.mine_units(world, WORKDIR, trusted)
    print(f"trusted={len(trusted)} mined_writes={mstats['writes']}", flush=True)

    atable = units.profile_anchor_table(world, WORKDIR, trusted)
    anchors = units.anchor_values(world, atable)
    print(f"anchor_table={atable} rows={len(anchors)}", flush=True)

    valid: list[dict] = []
    unit_reasons: dict[str, int] = {}
    for i, unit in enumerate(mined):
        anchor = anchors[sample_key("arm-e-unit", f"{SEED}:{i}", "") % len(anchors)]
        args, why = units.bind_unit(world, sigs, unit, anchor, f"{SEED}:pre:{i}")
        if args is not None:
            valid.append(unit)
        else:
            unit_reasons[why] = unit_reasons.get(why, 0) + 1
    print(f"units_validated_alone={len(valid)}/{len(mined)} reasons={unit_reasons}", flush=True)
    if len(valid) < 6:
        print("too few validated units, stopping")
        return

    sets = draw_sets(valid, anchors, SETS if not pilot_only else 5, SEED)
    for s in sets:
        alts = []
        for r in range(REVISIONS):
            cand = anchors[sample_key("arm-e-alt", f"{SEED}:{sets.index(s)}:{r}", "") % len(anchors)]
            if cand != s["anchor"]:
                alts.append(cand)
        while len(alts) < REVISIONS:
            alts.append(s["anchor"])
        s["alt"] = alts

    OUT.mkdir(parents=True, exist_ok=True)
    lines_path = OUT / "tasks.jsonl"
    if not pilot_only and lines_path.exists():
        lines_path.unlink()
    results: list[dict] = []
    per: dict[str, dict] = {
        "glued": {"admitted": 0, "rejected": 0, "revs": [], "usd": 0.0},
        "direct": {"admitted": 0, "rejected": 0, "revs": [], "usd": 0.0},
    }
    check_tally: dict[str, dict[str, int]] = {}
    pairs: list[dict] = []

    for set_no, goal_set in enumerate(sets):
        if stage_usd(WORKDIR, "synth_glued") + stage_usd(WORKDIR, "synth_direct") >= CEILING - 0.5:
            print("spend guard reached, stopping")
            break
        unit_n = len(goal_set["units"])
        level = "hard" if unit_n >= 3 else "medium"
        pair_status: dict[str, str] = {}
        for construction, stage, builder in (
            ("glued", "synth_glued", build_glued),
            ("direct", "synth_direct", build_direct),
        ):
            tid = f"synth_e_{construction}_{set_no:02d}"
            before = stage_usd(WORKDIR, stage)
            if construction == "glued":
                prop, reasons = builder(world, body, sigs, askable, models[stage], goal_set, set_no, SEED)
            else:
                prop, reasons = builder(world, sigs, askable, models[stage], tools, goal_set, set_no, SEED)
            status, checks = "rejected", {}
            revisions = prop["revisions"] if prop else REVISIONS
            said, req, must, calls = [], "", [], []
            if prop:
                made, checks = admit(
                    world,
                    body,
                    sigs,
                    askable,
                    prop["calls"],
                    prop["request"],
                    prop["must_tell"],
                    tid,
                    SEED,
                    set_no,
                    unit_n,
                    level,
                    prop["said"],
                )
                said, req, must, calls = prop["said"], prop["request"], prop["must_tell"], prop["calls"]
                if made and all(
                    [
                        checks.get("runs"),
                        checks.get("do_nothing_fails"),
                        checks.get("prefix_fails"),
                        checks.get("suite_passed"),
                    ]
                ):
                    status = "admitted"
                else:
                    if made:
                        failed = [k for k in ("do_nothing_fails", "prefix_fails", "suite_passed") if not checks.get(k)]
                        reasons.append("check:" + "+".join(failed or ["replay"]))
                    else:
                        reasons.append("replay_failed")
                    revisions = min(prop["revisions"] + 1, REVISIONS)
            after_cost = stage_usd(WORKDIR, stage) - before
            line = line_for(
                tid,
                construction,
                level,
                unit_n,
                set_no,
                req,
                said,
                calls,
                must,
                checks,
                revisions,
                status,
                reasons[-4:],
                round(after_cost, 6),
            )
            results.append(line)
            with open(lines_path, "a") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
            per[construction]["admitted" if status == "admitted" else "rejected"] += 1
            per[construction]["revs"].append(revisions)
            per[construction]["usd"] += after_cost
            key = (
                "runs"
                if not checks.get("runs")
                else "do_nothing_fails"
                if not checks.get("do_nothing_fails")
                else "prefix_fails"
                if not checks.get("prefix_fails")
                else "suite_passed"
                if not checks.get("suite_passed")
                else "pass"
            )
            tally = check_tally.setdefault(construction, {})
            tally[key] = tally.get(key, 0) + 1
            pair_status[construction] = status
            print(
                f"set={set_no} {construction} {status} rev={revisions} "
                f"checks={ {k: bool(checks.get(k)) for k in ('runs', 'do_nothing_fails', 'prefix_fails', 'suite_passed')} }",
                flush=True,
            )
        if pair_status.get("glued") != pair_status.get("direct"):
            pairs.append({"goal_set": set_no, "glued": pair_status.get("glued"), "direct": pair_status.get("direct")})
        if pilot_only is False and set_no == 4:
            print(f"PILOT FUNNEL: {json.dumps(per)} tallies={json.dumps(check_tally)} pairs={pairs}", flush=True)

    summary = {
        "seed": SEED,
        "synth_model": SYNTH_MODEL,
        "ceiling_usd": CEILING,
        "wall_s": round(time.time() - t0, 1),
        "mined": mstats,
        "units_validated_alone": len(valid),
        "per_construction": {
            k: {
                "admitted": v["admitted"],
                "rejected": v["rejected"],
                "revisions": v["revs"],
                "usd": round(v["usd"], 4),
                "usd_per_admitted": round(v["usd"] / v["admitted"], 4) if v["admitted"] else None,
            }
            for k, v in per.items()
        },
        "check_tally": check_tally,
        "pairs_split": pairs,
        "total_usd": round(per["glued"]["usd"] + per["direct"]["usd"], 4),
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
