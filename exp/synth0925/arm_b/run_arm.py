"""Arm B driver: compose concrete units into Tasks, birth-check, store.

Pilot 10 first, then the rest to 60 in shares 20/20/20 easy/medium/hard.
Keyed seed synth0925-b for every random draw. Ceiling 5 USD.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import units
from birth import birth_decline, birth_do, store_decline, task_id_of
from requester import Writer, code_fallback, goals_for, write_request

from kullback import synthesise as S
from kullback.gates import verifier_suite
from kullback.sampling import sample_key

HERE = Path(__file__).parent
R = units.R
OUT = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.claude/herdr/research-0925/rs-synth/exp/arm-b")
SALT = "synth0925-b"

def actor_id(u: dict) -> str:
    a = u.get("actor") or {}
    return str(a.get("id") or u.get("user") or "")


def tighten(said: list[dict], held: list[dict], shown: set[str], uni: set[str],
            ) -> tuple[list, list]:
    """Value-level tightening of the field-level said/held split: a row-ref
    value no rebound read reaches (a replacement catalog id, an internal row
    id the user was never shown) is a record fact even when its field is
    askable. Shown ids and free user-supplied values stay said."""
    keep, move = [], []
    for r in said:
        v = r["value"]
        vals = v if isinstance(v, list) else [v]
        is_ref = any(str(x) in uni for x in vals)
        if is_ref and any(str(x) not in shown for x in vals):
            move.append(r)
        else:
            keep.append(r)
    return keep, held + move


def askable_fields(workdir: Path = R) -> set[str]:
    out: set[str] = set()
    d = workdir / "user_rules"
    if not d.exists():
        return set()
    for f in d.iterdir():
        try:
            body = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        facts = body.get("facts") or []
        for x in facts:
            if isinstance(x, dict) and x.get("field"):
                out.add(str(x["field"]))
    return out


def build_universe(world: S.World, workdir: Path) -> tuple[set[str], dict]:
    """Every id-like value the world holds: top-level row keys of every
    schema table, plus the keys of every nested dict-of-dicts map.
    Tables come from schema.json; maps are found by shape."""
    bind = units.Binding(workdir)
    db = world.fresh_db()
    uni: set[str] = set()
    for table in bind.tables:
        uni |= set(map(str, ((db.get(table)) or {}).keys()))
        for body in (db.get(table) or {}).values():
            if not isinstance(body, dict):
                continue
            for _path, names in units.nested_maps(body):
                uni |= set(names)
    return uni, db

def unit_key(u: dict) -> str:
    if u.get("kind") == "do":
        rows = "+".join(sorted(map(str, u.get("rows", []))))
        return f"do:{u['source_task_id']}:{rows}"
    c = u["call"]
    return f"dec:{u['source_task_id']}:{c['name']}:{sorted(map(str, u.get('rows', [])))}:{u.get('error', '')[:40]}"


def plan_counts(mode: str) -> list[tuple[str, str, int, int]]:
    """(kind, level, n_do, n_dec) rows to build."""
    if mode == "pilot":
        return [("do", "easy", 1, 0)] * 4 + [("do", "medium", 2, 0)] * 2 \
            + [("decline", "medium", 0, 1)] + [("mixed", "hard", 1, 1)] * 2 \
            + [("do", "hard", 3, 0)]
    return [("do", "easy", 1, 0)] * 20 + [("do", "medium", 2, 0)] * 10 \
        + [("decline", "medium", 0, 1)] * 10 + [("mixed", "hard", 1, 1)] * 10 \
        + [("mixed", "hard", 2, 1)] * 10


def draw(pool: list, ident: str, skip: set, keyfn) -> dict | None:
    order = sorted(range(len(pool)),
                   key=lambda i: sample_key("arm-b", f"{ident}:{i}", SALT))
    for i in order:
        if keyfn(pool[i]) in skip:
            continue
        return pool[i]
    return None


def build_task(world: S.World, spec: tuple, do_pool: list, dec_pool: list,
               by_actor: dict, askable: set, writer: Writer | None,
               used: set, fn, salt: str, idx: int, uni: set[str],
               revisions: int = 3) -> dict:
    kind, level, n_do, n_dec = spec
    reasons: list[str] = []
    for rev in range(revisions + 1):
        ident = f"{salt}:{kind}:{level}:{idx}:r{rev}"
        dec = None
        if n_dec:
            dec = draw(dec_pool, ident + ":dec", used, unit_key)
            if dec is None:
                reasons.append("no decline unit left")
                continue
        user = actor_id(dec) if dec else None
        dos: list[dict] = []
        for k in range(n_do):
            taken = {r for d in dos for r in d.get("rows", [])}
            if dec:
                taken |= set(dec.get("rows", []))
            if user is not None:
                pool = [x for x in by_actor.get(user, [])
                        if unit_key(x) not in used
                        and taken.isdisjoint(set(x.get("rows", [])))]
                d = draw(pool, ident + f":do{k}", set(), unit_key)
            elif k == 0:
                d = draw(do_pool, ident + f":do{k}", used, unit_key)
            else:
                pool = [x for x in by_actor.get(user, [])
                        if unit_key(x) not in used
                        and taken.isdisjoint(set(x.get("rows", [])))]
                d = draw(pool, ident + f":do{k}", set(), unit_key)
                if d is None:
                    pool2 = [x for x in do_pool
                             if unit_key(x) not in used
                             and taken.isdisjoint(set(x.get("rows", [])))]
                    d = draw(pool2, ident + f":do{k}", set(), unit_key)
            if d is None:
                reasons.append("no do unit left")
                break
            if user is None:
                user = actor_id(d)
            dos.append(d)
        if len(dos) != n_do:
            continue
        # units in one task must touch disjoint rows: a second write to the
        # same row overwrites the first, and the prefix check rightly refuses
        seen: set[str] = set()
        clash = False
        for d in dos:
            for r in d.get("rows", []):
                if r in seen:
                    clash = True
                seen.add(r)
        if clash:
            reasons.append("units share a row")
            for d in dos:
                used.add(unit_key(d))
            continue
        do_rows = {r for d in dos for r in d.get("rows", [])}
        calls = [c for d in dos for c in d["calls"]]
        if dec and not do_rows.isdisjoint(set(dec.get("rows", []))):
            reasons.append("decline rows overlap do rows")
            used.add(unit_key(dec))
            continue
        forbid = [dec["call"]] if dec else []
        said, held = S.facts_of(
            [{"name": c["name"], "args": c["args"]} for c in calls + forbid], askable)
        shown: set[str] = set()
        for d in dos + ([dec] if dec else []):
            shown |= set(d.get("shown", []))
        said, held = tighten(said, held, shown, uni)
        goals = goals_for(len(calls) if kind != "decline" else 0, 1 if dec else 0)
        if kind == "decline":
            goals = goals_for(0, 1)
        fallback = code_fallback(len(calls), 1 if dec else 0)
        request, refusal, spent = write_request(writer, goals, said, held, fallback)
        task_id = task_id_of(ident)
        if kind == "decline":
            made, checks = birth_decline(world, task_id, {**dec, "request": request})
            if made is not None:
                store_decline(world, task_id, request, dec, said, salt)
                used.add(unit_key(dec))
                units_rec = [{"source": dec["source_task_id"], "sort": dec["sort"],
                              "rows": dec["rows"], "error": dec["error"]}]
                return admitted(task_id, kind, level, spec, request, said, [],
                                dec, checks, rev, refusal, spent, units_rec)
            reasons.append(f"decline check failed: {checks}")
            used.add(unit_key(dec))
            continue
        made, checks = birth_do(world, task_id, calls, request, said, salt, fn)
        if made is None:
            reasons.append(f"do check failed: {checks}")
            for d in dos:
                used.add(unit_key(d))
            continue
        if dec:
            dmade, dchecks = birth_decline(world, task_id, {**dec, "request": request})
            checks = {**checks, "decline": dchecks}
            if dmade is None:
                reasons.append(f"mixed decline failed: {dchecks}")
                used.add(unit_key(dec))
                continue
            store_decline(world, task_id + "-dec", request, dec, said, salt)
        for d in dos:
            used.add(unit_key(d))
        if dec:
            used.add(unit_key(dec))
        units_rec = [{"source": d["source_task_id"], "scope": d.get("scope"),
                      "rows": d["rows"]} for d in dos]
        if dec:
            units_rec.append({"source": dec["source_task_id"], "sort": dec["sort"],
                              "rows": dec["rows"], "error": dec["error"]})
        return admitted(task_id, kind, level, spec, request, said, calls,
                        dec, checks, rev, refusal, spent, units_rec)
    return rejected(kind, level, spec, reasons, revisions, 0.0)


def admitted(task_id, kind, level, spec, request, said, calls, dec,
             checks, rev, refusal, spent, units_rec=None):
    return {"task_id": task_id, "arm": "b", "kind": kind, "level": level,
            "units": units_rec or [], "scenario_id": None, "request": request,
            "user_facts": said, "blueprint_writes": calls,
            "must_tell": [],
            "checks": checks, "revisions": rev, "status": "admitted",
            "reasons": [refusal] if refusal else [], "cost_usd": spent}


def rejected(kind, level, spec, reasons, rev, spent):
    return {"arm": "b", "kind": kind, "level": level,
            "units": [], "scenario_id": None, "request": "",
            "user_facts": [], "blueprint_writes": [],
            "must_tell": [],
            "checks": {}, "revisions": rev, "status": "rejected",
            "reasons": reasons[-3:], "cost_usd": spent}


def main(mode: str = "pilot"):
    t0 = time.time()
    world = S.World(R)
    fn = verifier_suite.canon_fn(world.canon_rules)
    askable = askable_fields(R)
    print(f"askable fields: {len(askable)}", flush=True)
    y = json.loads((OUT / "yields.json").read_text())
    do_pool = y["do_units"]
    dec_pool = y["decline_units"]
    print(f"do={len(do_pool)} decline={len(dec_pool)}", flush=True)
    by_actor: dict[str, list] = {}
    for u in do_pool:
        by_actor.setdefault(actor_id(u), []).append(u)
    uni, _db = build_universe(world, R)
    try:
        writer = Writer(OUT)
        print("model writer live", flush=True)
    except Exception as e:  # noqa: BLE001 - code fallback keeps the run going
        print(f"model writer off ({type(e).__name__}): code fallback", flush=True)
        writer = None
    used: set[str] = set()
    if mode == "topup":
        rows = [json.loads(line) for line in (OUT / "tasks.jsonl").read_text().splitlines()]
        for i, r in enumerate(rows):
            if "task_id" not in r:
                r["task_id"] = task_id_of(f"rej:{r.get('kind')}:{r.get('level')}:{i}")
        for r in rows:
            r["used_keys"] = []
        for r in rows:
            if r["status"] != "admitted":
                continue
            for u in r.get("units", []):
                for p in do_pool:
                    if p["source_task_id"] == u.get("source") and set(p.get("rows", [])) == set(u.get("rows", [])):
                        used.add(unit_key(p))
                        r["used_keys"].append(unit_key(p))
                for p in dec_pool:
                    if (p["source_task_id"] == u.get("source")
                            and set(p.get("rows", [])) == set(u.get("rows", []))
                            and p.get("error", "") == u.get("error", p.get("error", ""))):
                        used.add(unit_key(p))
                        r["used_keys"].append(unit_key(p))
        targets = [i for i, r in enumerate(rows) if r["status"] == "rejected"]
        print(f"topup targets: {targets}", flush=True)
        prev_spend = writer.spent_usd() if writer else 0.0
        for j, i in enumerate(targets):
            spec = ("mixed", "hard", 1, 1)
            if writer is not None and writer.spent_usd() >= 5.0:
                rows[i] = rejected(*spec[:2], spec, ["ceiling reached"], 0, 0.0)
                continue
            row = build_task(world, spec, do_pool, dec_pool, by_actor, askable,
                             writer, used, fn, SALT, 100 + j, uni)
            now_spend = writer.spent_usd() if writer else 0.0
            row["cost_usd"] = round(now_spend - prev_spend, 6)
            prev_spend = now_spend
            rows[i] = row
            print(f"topup {i}: {row['status']} rev={row['revisions']} "
                  f"reasons={row['reasons']} spend={row['cost_usd']:.4f}", flush=True)
        for r in rows:
            r.pop("used_keys", None)
            if "task_id" not in r:
                r["task_id"] = task_id_of(f"rej:{r.get('kind')}:{r.get('level')}:{rows.index(r)}")
        specs = [("kept", "", 0, 0)] * len(rows)
    else:
        specs = plan_counts(mode)
        used = set()
        rows = []
        prev_spend = writer.spent_usd() if writer else 0.0
        for i, spec in enumerate(specs):
            if writer is not None and writer.spent_usd() >= 5.0:
                rows.append(rejected(spec[0], spec[1], spec, ["ceiling reached"], 0, 0.0))
                continue
            row = build_task(world, spec, do_pool, dec_pool, by_actor, askable,
                             writer, used, fn, SALT, i, uni)
            now_spend = writer.spent_usd() if writer else 0.0
            row["cost_usd"] = round(now_spend - prev_spend, 6)
            prev_spend = now_spend
            rows.append(row)
            print(f"{i}: {spec[0]}/{spec[1]} {row['status']} rev={row['revisions']} "
                  f"reasons={row['reasons']} spend={row['cost_usd']:.4f}", flush=True)
    dest = OUT / ("tasks-pilot.jsonl" if mode == "pilot" else "tasks.jsonl")
    with open(dest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "mode": mode, "n": len(rows),
        "admitted": sum(1 for r in rows if r["status"] == "admitted"),
        "by_level": {lv: sum(1 for r in rows if r["level"] == lv and r["status"] == "admitted")
                     for lv in ("easy", "medium", "hard")},
        "by_kind": {k: sum(1 for r in rows if r["kind"] == k and r["status"] == "admitted")
                    for k in ("do", "decline", "mixed")},
        "revisions": {str(i): sum(1 for r in rows if r["revisions"] == i) for i in range(5)},
        "spend_usd": writer.spent_usd() if writer else 0.0,
        "wall_s": round(time.time() - t0, 1),
    }
    (OUT / ("summary-pilot.json" if mode == "pilot" else "summary.json")).write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
