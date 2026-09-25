"""Birth path for arm C: run a blueprint, derive, check. All gates are code.

Copies the shape of the harness _make_task (request text supplied by our
synthesiser instead of intent_text) and the coherent rule from the arm A
scratch binder: a row-naming argument of a write must come from what reads
returned, and a refused write is retried on a fresh draw with the reason
kept. No names of any corpus, tool, task or model appear here; every name
is read off the workdir at run time.
"""
from __future__ import annotations

from typing import Any

RETRIES = 8
SEED = "synth0925-c"


def _norm(args: dict, spec_props: dict) -> dict:
    """List-wrap a scalar where the signature declares a list. General fix."""
    out = dict(args)
    for name, spec in (spec_props or {}).items():
        kinds = [str(k) for k in ((spec or {}).get("type") or [])]
        if "list" in kinds and name in out and not isinstance(out[name], list):
            out[name] = [out[name]]
    return out


def _row_args(sig_props: dict) -> list:
    """Required arguments that name rows: the coherent rule applies to these."""
    return [n for n in (sig_props or {}) if n.endswith("_id") or n.endswith("_ids")]


def _leaves(value: Any) -> set:
    """Every scalar string inside an answer, for the read-before-write check."""
    found: set = set()
    if isinstance(value, dict):
        for item in value.values():
            found |= _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= _leaves(item)
    elif isinstance(value, str) and value:
        found.add(value)
    return found


def try_blueprint(world: Any, reads: list, writes: list, ident: str) -> tuple:
    """Run the blueprint reads then writes on a fresh world. Returns (calls, why).

    why is "" on success, else one of read_failed:<Class>, refused:<Class>,
    crashed:<Class>, unknown_tool, or unread_row:<arg>. One attempt: the body
    is the oracle, and a refusal goes back to the synthesiser as a revision
    reason. Row ids a write names must have appeared in an earlier answer of
    this same run (the coherent rule); pure user values are exempt.
    """
    from kullback.gates.tool_runs import CRASH_ERRORS

    sigs = {s.name: s for s in world.sigs}
    db = world.fresh_db()
    toolkit = world.toolkit(db)
    seen: set = set()
    calls: list = []
    seq = [{"kind": "read", **r} for r in (reads or [])]
    seq += [{"kind": "write", **w} for w in (writes or [])]
    for index, item in enumerate(seq):
        name = str(item.get("name") or "")
        args = dict(item.get("args") or {})
        body = getattr(toolkit, name, None)
        if body is None:
            return calls, "unknown_tool"
        sig = sigs.get(name)
        props = {}
        if sig is not None:
            schema = getattr(sig, "args_schema", None) or {}
            props = dict(schema.get("properties") or {})
        args = _norm(args, props)
        if item["kind"] == "write":
            for arg in _row_args(props):
                vals = args.get(arg)
                pool = vals if isinstance(vals, list) else [vals]
                if any(str(v) not in seen for v in pool if v is not None):
                    return calls, f"unread_row:{arg}"
        try:
            answer = body(**args)
        except Exception as error:
            cls = type(error).__name__
            if cls in CRASH_ERRORS:
                return calls, f"crashed:{cls}"
            if item["kind"] == "read":
                return calls, f"read_failed:{cls}"
            return calls, f"refused:{cls}"
        seen |= _leaves(answer)
        calls.append({"id": f"{ident}-{index}", "name": name, "args": args,
                      "requestor": "assistant", "turn": index,
                      "result": answer, "error": None})
    if not [c for c in calls if c["name"] in world.write_tools]:
        return calls, "no_writes"
    return calls, ""


def make_task(world: Any, calls: list, request: str, said: list,
              task_id: str, salt: str) -> tuple:
    """The walk replayed as a Run, the derived Verifier, the suite and checks.

    Returns (row, info) where row is the stored-task record or None and info
    carries the four birth checks. Mirrors the harness _make_task, with the
    caller-supplied request text in place of intent_text.
    """
    from kullback import difficulty
    from kullback import graph as graph_mod
    from kullback import synthesise as S
    from kullback.examiner import derive as derive_mod
    from kullback.examiner import reference as reference_mod
    from kullback.gates import verifier_suite
    from kullback.runner.records import Task, as_dict, write_json

    transcript, spoken = S.spoken_for(request, calls)
    run = S.record_run(world, task_id, spoken, f"synth-{task_id}", transcript)
    checks = {"runs": False, "do_nothing_fails": False,
              "prefix_fails": False, "suite_passed": False, "suite_failures": []}
    if run is None:
        return None, checks
    fn = verifier_suite.canon_fn(world.canon_rules)
    try:
        reference = reference_mod.load(run["path"], reference_mod.REROLL,
                                       run_id=run["run_id"],
                                       write_tools=world.write_tools, fn=fn, atoms=())
    except (OSError, ValueError, TypeError):
        return None, checks
    checks["runs"] = True
    alternates = S.second_paths(world, task_id, spoken, transcript,
                                reference, f"synth-{task_id}")
    task = Task(id=task_id, run_ids=[run["run_id"]], intent=request, name=None)
    verifier = derive_mod.derive_verifier(
        task, run["path"], rerun_paths=[row["path"] for row in alternates],
        canon=world.canon_rules, write_tools=world.write_tools, intent=request)
    passed, failures = S._suite(world, verifier, run, alternates, request)
    checks["suite_failures"] = list(failures)
    checks["suite_passed"] = bool(passed)
    checks["do_nothing_fails"] = "verifier_empty_run" not in set(failures)
    checks["prefix_fails"] = "verifier_unfinished_run" not in set(failures)
    writes, tools = graph_mod.reached([{"tool": c["name"]} for c in calls],
                                      world.write_tools)
    reached = difficulty.bucket_key(difficulty.write_count(verifier), tools,
                                    1 + len(alternates))
    store = S.store_dir(world.workdir)
    (store / "tasks").mkdir(parents=True, exist_ok=True)
    (store / "verifiers").mkdir(parents=True, exist_ok=True)
    write_json(store / "tasks" / f"{task_id}.json",
               {"format": S.FORMAT, "synthetic": True, "task": as_dict(task),
                "bucket": reached, "seed": str(salt),
                "walk": [{"tool": c["name"], "args": c["args"]} for c in calls],
                "user_rules": as_dict(S.user_rules_for(said))})
    write_json(store / "verifiers" / f"{task_id}.json", as_dict(verifier))
    row = {"task_id": task_id, "bucket": reached, "writes": writes,
           "tools": tools, "pool": len(alternates),
           "suite_passed": passed, "suite_failures": failures,
           "run_id": run["run_id"]}
    return row, checks


def task_uid(tag: str) -> str:
    import hashlib
    digest = hashlib.sha256(str(tag).encode()).hexdigest()
    return "synth_" + str(int(digest[:12], 16) % (16 ** 12)).rjust(12, "0")[:12]


def scrub(text: str) -> str:
    """No em or en dashes anywhere: fold them to hyphens."""
    return str(text or "").replace("\u2014", "-").replace("\u2013", "-")
