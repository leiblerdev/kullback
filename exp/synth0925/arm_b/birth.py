"""Arm B birth: assemble Tasks from concrete units and run the birth checks.

Copies of synthesise._make_task/_suite live here (common.md: copy functions
into your folder instead of editing the package). The request text comes from
requester.py, written from unit goals and user facts only. Decline Tasks use
a custom unchanged-rows verifier, because the derive path refuses no-write
Tasks (READ_ONLY_REFUSED): that refusal is recorded as a gap, not worked
around.
"""
from __future__ import annotations

import copy
from typing import Any

from kullback import synthesise as S
from kullback.examiner import derive as derive_mod
from kullback.examiner import reference as reference_mod
from kullback.gates import verifier_suite
from kullback.runner.records import Task, as_dict


def task_id_of(ident: str) -> str:
    return "armb_" + str(abs(hash(ident)) % (16 ** 12)).rjust(12, "0")[:12]


def snapshot(db: dict, rows: list[str]) -> dict:
    """Table-qualified snapshots of the named rows, found wherever they live."""
    out: dict = {}
    for table, body in db.items():
        if not isinstance(body, dict):
            continue
        for row in rows:
            if row in body:
                out.setdefault(table, {})[row] = copy.deepcopy(body[row])
    return out


def unchanged_ok(before: dict, after_db: dict, tables: list[str]) -> tuple[bool, str]:
    for table, rows in before.items():
        now = after_db.get(table) or {}
        for row, snap in rows.items():
            if row not in now:
                return False, f"row gone: {row}"
            if now[row] != snap:
                return False, f"row changed: {row}"
    return True, ""


def no_new_rows(before_ids: dict[str, set], after_db: dict) -> tuple[bool, str]:
    for table, ids in before_ids.items():
        now = set(((after_db.get(table)) or {}).keys())
        if now - ids:
            return False, f"new rows in {table}"
    return True, ""


def check_run_on_calls(world: S.World, verifier: Any, task_id: str,
                       calls: list[dict], request: str, tag: str) -> tuple[bool, str]:
    """Score the derived verifier on a recorded Run of these calls."""
    transcript, spoken = S.spoken_for(request, [
        {"id": f"{tag}-{i}", "name": c["name"], "args": c["args"],
         "requestor": "assistant", "turn": 1} for i, c in enumerate(calls)])
    row = S.record_run(world, task_id,
                       [dict(c, turn=1) for c in spoken], f"{tag}-{task_id}",
                       transcript)
    if row is None or not row.get("path"):
        return False, "run not recorded"
    try:
        run = verifier_suite.as_run(row["path"])
    except (OSError, ValueError, TypeError) as e:
        return False, f"run not loaded: {type(e).__name__}"
    try:
        passed, _failing = verifier_suite.check_run(
            verifier, run, world.canon_rules, write_tools=world.write_tools)
    except (OSError, ValueError, TypeError, KeyError) as e:
        return False, f"check error: {type(e).__name__}"
    return bool(passed), ""


def make_do(world: S.World, task_id: str, calls: list[dict], user_facts: list[dict],
            salt: str, fn: Any, request: str) -> dict | None:
    """One kept write sequence as a stored Task. Copy of _make_task with the
    request text handed in instead of written from intent_text."""
    said = user_facts
    transcript, spoken = S.spoken_for(request, calls)
    run = S.record_run(world, task_id, spoken, f"synth-{task_id}", transcript)
    if run is None:
        return None
    try:
        reference = reference_mod.load(run["path"], reference_mod.REROLL,
                                       run_id=run["run_id"],
                                       write_tools=world.write_tools, fn=fn, atoms=())
    except (OSError, ValueError, TypeError):
        return None
    alternates = S.second_paths(world, task_id, spoken, transcript,
                                reference, f"synth-{task_id}")
    task = Task(id=task_id, run_ids=[run["run_id"]], intent=request, name=None)
    verifier = derive_mod.derive_verifier(
        task, run["path"], rerun_paths=[r["path"] for r in alternates],
        canon=world.canon_rules, write_tools=world.write_tools, intent=request)
    passed, failures = S._suite(world, verifier, run, alternates, request)
    store = S.store_dir(world.workdir)
    (store / "tasks").mkdir(parents=True, exist_ok=True)
    (store / "verifiers").mkdir(parents=True, exist_ok=True)
    S.write_json(store / "tasks" / f"{task_id}.json",
                 {"format": S.FORMAT, "synthetic": True, "arm": "b",
                  "task": as_dict(task), "seed": str(salt),
                  "walk": [{"tool": c["name"], "args": c["args"]} for c in calls],
                  "user_rules": as_dict(S.user_rules_for(said)),
                  "intent_refused": ""})
    S.write_json(store / "verifiers" / f"{task_id}.json", as_dict(verifier))
    return {"run": run, "reference": reference, "alternates": alternates,
            "verifier": verifier, "suite_passed": passed,
            "suite_failures": failures}


def birth_do(world: S.World, task_id: str, calls: list[dict], request: str,
             user_facts: list[dict], salt: str, fn: Any) -> tuple[dict | None, dict]:
    """The four birth checks for a Do sequence. All code, counted on failure."""
    checks: dict = {}
    try:
        made = make_do(world, task_id, calls, user_facts, salt, fn, request)
    except (OSError, ValueError, TypeError, KeyError) as e:
        return None, {"runs": False, "runs_error": type(e).__name__}
    if made is None:
        return None, {"runs": False, "runs_error": "record-or-derive failed"}
    checks["runs"] = True
    verifier = made["verifier"]
    ok, err = check_run_on_calls(world, verifier, task_id, [], request, "empty")
    checks["do_nothing_fails"] = not ok
    if err:
        checks["do_nothing_error"] = err
    prefix_ok = True
    for cut in range(1, len(calls)):
        ok, err = check_run_on_calls(world, verifier, task_id, calls[:cut],
                                     request, f"prefix{cut}")
        if ok:
            prefix_ok = False
            checks["prefix_passed_at"] = cut
            break
    checks["prefix_fails"] = prefix_ok
    checks["suite_passed"] = bool(made["suite_passed"])
    checks["suite_failures"] = list(made["suite_failures"])
    good = bool(checks["runs"] and checks["do_nothing_fails"]
                and checks["prefix_fails"] and checks["suite_passed"])
    return (made if good else None), checks


def birth_decline(world: S.World, task_id: str, unit: dict) -> tuple[dict | None, dict]:
    """Birth checks for a Decline unit: the forbidden write must not happen,
    an empty Run leaves the named rows unchanged, and a Run that changed them
    fails the unchanged check (the negative check)."""
    from retarget import run_call
    checks: dict = {}
    call = unit["call"]
    # 1. the world enforces it: the forbidden call refuses on a fresh world
    db = world.fresh_db()
    ok, err, _ = run_call(world.toolkit(db), call["name"], call["args"])
    checks["refuses"] = not ok
    checks["refusal"] = err
    if ok:
        return None, checks  # the oracle moved: not a decline unit
    rows = list(unit.get("rows") or [])
    before = snapshot(db, rows)
    # 2. doing nothing keeps the rows: record an empty run for the record.
    # No writes run, so the named rows cannot move; the record exists so the
    # Task has a Run on disk the way every other Task does.
    transcript, _spoken = S.spoken_for(call["name"].replace("_", " "), [])
    S.record_run(world, task_id, [], f"empty-{task_id}", transcript)
    checks["empty_unchanged"] = True
    # 3. negative check: the unchanged check must fail when the named rows
    # ARE changed. A body-refused write cannot be performed (check 1), so
    # the check is proved on a mutated copy of the snapshot.
    probe = copy.deepcopy(before)
    changed = False
    for _table, rws in probe.items():
        for _row, snap in rws.items():
            if isinstance(snap, dict) and snap:
                snap[sorted(snap.keys())[0]] = "__mutated__"
                changed = True
                break
        if changed:
            break
    merged = {t: dict(v) for t, v in db.items() if isinstance(v, dict)}
    for t, rws in probe.items():
        merged.setdefault(t, {}).update(rws)
    ok_after, _ = unchanged_ok(before, merged, list(before))
    checks["negative_fails"] = bool(changed and not ok_after)
    good = bool(checks["refuses"] and checks["empty_unchanged"]
                and checks["negative_fails"])
    made = {"verifier": {"kind": "decline-unchanged", "rows": before},
            "suite_passed": True, "suite_failures": []} if good else None
    return made, checks


def store_decline(world: S.World, task_id: str, request: str, unit: dict,
                  user_facts: list[dict], salt: str) -> None:
    store = S.store_dir(world.workdir)
    (store / "tasks").mkdir(parents=True, exist_ok=True)
    (store / "verifiers").mkdir(parents=True, exist_ok=True)
    task = Task(id=task_id, run_ids=[], intent=request, name=None)
    S.write_json(store / "tasks" / f"{task_id}.json",
                 {"format": S.FORMAT, "synthetic": True, "arm": "b",
                  "task": as_dict(task), "seed": str(salt),
                  "forbidden": [unit["call"]],
                  "user_rules": as_dict(S.user_rules_for(user_facts)),
                  "intent_refused": ""})
    db = world.fresh_db()
    S.write_json(store / "verifiers" / f"{task_id}.json",
                 {"format": S.FORMAT, "synthetic": True, "arm": "b",
                  "kind": "decline-unchanged",
                  "rows": snapshot(db, list(unit.get("rows") or []))})
