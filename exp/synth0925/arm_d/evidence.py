"""Evidence bundles for the agentic judge: one compact record per failed Run.

All names (calls, atoms, markers) are read from the workdir, never typed here.
No customer strings leave this module except row ids, which reports may quote.
"""
import json
from pathlib import Path

from kullback.runner import canon as canon_mod
from kullback.runner import target as target_mod
from kullback.runner import verdict as verdict_mod
from kullback.runner.records import Verifier, read_json

USER_MARKERS = ("unavailable", "goal_restated", "scenario_exhausted", "handed_off",
                "fact_unavailable")


def _text(event) -> str:
    payload = event.payload or {}
    return str(payload.get("text") or payload.get("content") or "")


def user_signals(run) -> dict:
    counts = {mark: 0 for mark in USER_MARKERS}
    turns = 0
    for event in run.events or ():
        if event.type != "user_turn":
            continue
        turns += 1
        blob = (_text(event) + " " + json.dumps(event.payload or {},
                                                default=str)).lower()
        for mark in USER_MARKERS:
            if mark in blob:
                counts[mark] += 1
    kinds = {}
    for event in run.events or ():
        kinds[event.type] = kinds.get(event.type, 0) + 1
    return {"user_turns": turns, "markers": counts, "event_kinds": kinds,
            "termination": str(getattr(run, "termination_reason", "") or "")}


def call_shape(run, write_tools: set[str], fn) -> dict:
    calls = target_mod.run_calls(run)
    seq = [{"name": call["name"], "error": bool(call["error"])} for call in calls]
    writes = []
    for key in target_mod.write_effects(run, write_tools, fn):
        call, _, row = key.partition("|")
        writes.append({"call": call, "row": row})
    return {"calls": seq, "writes": writes}
def verifier_shape(verifier: Verifier) -> dict:
    from kullback.runner.target import atom_payload
    atoms = []
    for atom in verifier.atoms or ():
        payload = atom_payload(atom)
        atoms.append({"id": str(atom.id), "kind": str(atom.kind),
                      "checks": str(payload.get("kind") or "")})
    return {"atoms": atoms}


def grader_map(workdir: Path) -> dict:
    """trace id to tau2 reward, from the grader sidecars. Counts only."""
    out = {}
    for path in (workdir / "grader").glob("*.json"):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        info = (body.get("fields") or {}).get("reward_info") or {}
        checks = info.get("action_checks") or []
        reward = info.get("reward")
        if reward is None and checks:
            reward = 1.0 if all(check.get("action_match") for check in checks) else 0.0
        if body.get("trace_id") is not None and reward is not None:
            out[str(body["trace_id"])] = float(reward)
    return out


def load_by_id(workdir: Path, paths: dict, run_id: str):
    rel = paths.get(run_id)
    if not rel:
        return None
    path = workdir / rel if not str(rel).startswith("/") else Path(rel)
    if not path.is_file():
        return None
    try:
        return verdict_mod.load_run(path)
    except Exception:
        return None


def reference_of(paths: dict, workdir: Path, row: dict):
    for run_id in (row or {}).get("reference_run_ids") or []:
        run = load_by_id(workdir, paths, str(run_id))
        if run is not None:
            return run
    return None


def evidence_for(workdir: Path, pool: dict) -> list[dict]:
    from kullback import difficulty
    from kullback.gates import verifier_suite
    paths = difficulty.run_paths(workdir)
    canon_rules = canon_mod.load_rules(workdir / "canon-rules.json")
    try:
        fn = verifier_suite.canon_fn(canon_rules)
    except Exception:
        fn = lambda value: value  # noqa: E731 - scoring fallback only
    sigs = read_json(workdir / "tool_sigs.json", []) or []
    write_tools = {str(row.get("name")) for row in sigs
                   if row.get("kind") == "write"}
    grades = grader_map(workdir)
    status = read_json(workdir / "task_status.json", {}) or {}
    by_id = {f"{row['task_id']}:{row['run_id']}": row for row in pool["failed_runs"]}
    out = []
    for key in pool["sample"]:
        row = by_id.get(key)
        if row is None:
            continue
        task_id, run_id = row["task_id"], row["run_id"]
        verifier = Verifier.model_validate(
            read_json(workdir / "verifiers" / f"{task_id}.json", {}))
        run = load_by_id(workdir, paths, run_id)
        if run is None:
            out.append({"run_id": run_id, "task_id": task_id,
                        "failed_atom": row["reason"], "unreadable": True})
            continue
        ref = reference_of(paths, workdir, status.get(task_id))
        item = {"run_id": run_id, "task_id": task_id,
                "failed_atom": row["reason"],
                "run": call_shape(run, write_tools, fn),
                "user": user_signals(run),
                "verifier": verifier_shape(verifier)}
        if ref is not None:
            item["reference"] = call_shape(ref, write_tools, fn)
            item["reference_run_id"] = getattr(ref, "run_id", "")
        if getattr(run, "trace_id", None) in grades:
            item["run_reward"] = grades[run.trace_id]
        if ref is not None and getattr(ref, "trace_id", None) in grades:
            item["reference_reward"] = grades[ref.trace_id]
        failed = (status.get(task_id) or {}).get("failed_recordings") or {}
        if run_id in failed:
            item["harness_note"] = str(failed[run_id])[:300]
        out.append(item)
    return out
