"""Arm E birth: store a blueprint like _make_task, then run the four checks.

_make_task is copied from kullback/synthesise.py and given the request text
instead of intent_text (common.md). The checks: runs (the blueprint's calls
run in a fresh world without refusal), do_nothing_fails, prefix_fails, and
suite_passed (the D79 suite). All code, no model.
"""

from __future__ import annotations

from typing import Any, Optional

from kullback import synthesise as S

reference_mod = S.reference_mod
derive_mod = S.derive_mod
verifier_suite = S.verifier_suite
difficulty = S.difficulty


def make_task(
    world: Any,
    body: dict,
    calls: list[dict],
    request: str,
    said: list[dict],
    held: list[dict],
    bucket: str,
    ident: str,
    salt: str,
    task_id: str,
) -> Optional[dict]:
    """One blueprint as a stored Task: the Run, the Verifier, the suite row.

    Copy of synthesise._make_task with the request text passed in. The
    transcript carries the request, so the leak gate audits this text.
    """
    askable = S.askable_fields(world.workdir)
    _ = askable
    fallback = S.written_intent(calls, world.write_tools)
    text = request.strip() or fallback
    transcript, spoken = S.spoken_for(text, calls)
    run = S.record_run(world, task_id, spoken, f"synth-{task_id}", transcript)
    if run is None:
        return None
    fn = verifier_suite.canon_fn(world.canon_rules)
    try:
        reference = reference_mod.load(
            run["path"], reference_mod.REROLL, run_id=run["run_id"], write_tools=world.write_tools, fn=fn, atoms=()
        )
    except (OSError, ValueError, TypeError):
        return None
    alternates = S.second_paths(world, task_id, spoken, transcript, reference, f"synth-{task_id}")
    task = S.Task(id=task_id, run_ids=[run["run_id"]], intent=text, name=None)
    verifier = derive_mod.derive_verifier(
        task,
        run["path"],
        rerun_paths=[row["path"] for row in alternates],
        canon=world.canon_rules,
        write_tools=world.write_tools,
        intent=text,
    )
    passed, failures = S._suite(world, verifier, run, alternates, text)
    from kullback import graph as graph_mod

    writes, tools = graph_mod.reached([{"tool": call["name"]} for call in calls], world.write_tools)
    reached = difficulty.bucket_key(difficulty.write_count(verifier), tools, 1 + len(alternates))
    store = S.store_dir(world.workdir)
    S.write_json(
        store / "tasks" / f"{task_id}.json",
        {
            "format": S.FORMAT,
            "synthetic": True,
            "task": S.as_dict(task),
            "bucket_requested": bucket,
            "bucket": reached,
            "seed": str(salt),
            "walk": [{"tool": c["name"], "args": c["args"]} for c in calls],
            "user_rules": S.as_dict(S.user_rules_for(said)),
            "record_facts": [row["field"] for row in held],
            "arm": "e",
            "construction": ident.split(":")[0],
        },
    )
    S.write_json(store / "verifiers" / f"{task_id}.json", S.as_dict(verifier))
    return {
        "task_id": task_id,
        "bucket": reached,
        "writes": writes,
        "tools": tools,
        "pool": len(alternates),
        "suite_passed": passed,
        "suite_failures": failures,
        "run_id": run["run_id"],
        "run_path": run["path"],
        "alternates": alternates,
        "verifier": verifier,
        "request": text,
    }


def _as_run(path: str) -> Any:
    from kullback.runner import target as target_mod

    return target_mod.as_run(path)


def birth_checks(world: Any, made: dict, calls: list[dict], request: str) -> dict:
    """The four birth checks over one stored blueprint. All code."""
    checks: dict[str, Any] = {
        "runs": True,
        "do_nothing_fails": False,
        "prefix_fails": False,
        "suite_passed": bool(made["suite_passed"]),
        "suite_failures": list(made["suite_failures"]),
    }
    fn = verifier_suite.canon_fn(world.canon_rules)
    verifier = made["verifier"]
    ref = _as_run(made["run_path"])
    n = len([c for c in calls if str(c.get("name")) in set(world.write_tools)])
    if n < 2:
        checks["prefix_fails"] = False
        checks["shape"] = "fewer_than_two_writes"
        return checks
    try:
        transcript, _spoken = S.spoken_for(request, [])
        row = S.record_run(world, made["task_id"], [], f"synth-{made['task_id']}-empty", transcript)
        if row is None:
            return checks
        from kullback.runner import target as target_mod

        passed, _failing = target_mod.check_run(verifier, _as_run(row["path"]), fn, write_tools=world.write_tools)
        checks["do_nothing_fails"] = not passed
    except (OSError, ValueError, TypeError):
        return checks
    prefixes_ok = True
    writes = [c for c in calls if str(c.get("name")) in set(world.write_tools)]
    for cut in range(1, len(writes)):
        try:
            transcript, spoken = S.spoken_for(request, writes[:cut])
            row = S.record_run(world, made["task_id"], spoken, f"synth-{made['task_id']}-pre{cut}", transcript)
            if row is None:
                prefixes_ok = False
                break
            from kullback.runner import target as target_mod

            passed, _failing = target_mod.check_run(verifier, _as_run(row["path"]), fn, write_tools=world.write_tools)
            if passed:
                prefixes_ok = False
                checks["prefix_passed_at"] = cut
                break
        except (OSError, ValueError, TypeError):
            prefixes_ok = False
            break
    checks["prefix_fails"] = prefixes_ok
    _ = ref
    return checks
