"""Shared readers for the vf test harness (vf_split.py, vf_measure.py).

Measurement only: nothing here writes to a workdir, and nothing here is read by a
build. The sidecar (the benchmark's own reward per recording) is loaded from a
directory the harness is pointed at, never from a path a model reads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from kullback.runner.canon import CanonRules, canon_value, load_rules
from kullback.runner.records import Run, Verifier

RIGHT = "right"
WRONG = "wrong"
MIXED = "mixed"
UNKNOWN = "unknown"
CLASSES = (RIGHT, WRONG, MIXED, UNKNOWN)


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def latest_round_rows(workdir: Path) -> dict[str, dict]:
    """task_id -> the latest round's row (trusted, checks, false_rejection)."""
    rounds = sorted((workdir / "rounds").glob("*/tasks.json")) if (workdir / "rounds").is_dir() else []
    if not rounds:
        return {}
    body = _read(rounds[-1], {}) or {}
    return {row["task_id"]: row for row in body.get("rows", []) if row.get("task_id")}


def trusted_ids(workdir: Path) -> set[str]:
    """The task ids the latest round calls trusted."""
    return {task_id for task_id, row in latest_round_rows(workdir).items() if row.get("trusted")}


def sidecar_rewards(sidecar: Path) -> dict[str, Optional[float]]:
    """trace_id -> the benchmark's reward, None where the sidecar carries none."""
    out: dict[str, Optional[float]] = {}
    if not sidecar.is_dir():
        return out
    for path in sorted(sidecar.glob("*.json")):
        body = _read(path, {}) or {}
        trace_id = body.get("trace_id")
        if not trace_id:
            continue
        fields = body.get("fields") or {}
        reward = fields.get("reward")
        if reward is None:
            reward = (fields.get("reward_info") or {}).get("reward")
        out[str(trace_id)] = float(reward) if reward is not None else None
    return out


def reference_class(references_row: dict, rewards: dict[str, Optional[float]]) -> str:
    """Right when every scored kept Reference scores 1, wrong when every one scores 0.

    Mixed when the kept References score both ways, unknown when none of them is
    scored. Rerolls carry no trace id and no reward, so only trace-linked kept
    References answer.
    """
    scored = [rewards[trace] for r in (references_row or {}).get("references", [])
              if (trace := (r or {}).get("trace_id")) and rewards.get(trace) is not None]
    if not scored:
        return UNKNOWN
    if all(s == 1.0 for s in scored):
        return RIGHT
    if all(s == 0.0 for s in scored):
        return WRONG
    return MIXED


def classes(workdir: Path, sidecar: Path) -> dict[str, str]:
    """task_id -> reference class for every task with a references row."""
    references = _read(workdir / "references.json", {}) or {}
    rewards = sidecar_rewards(sidecar)
    return {task_id: reference_class(row, rewards) for task_id, row in references.items()}


def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson 95% interval for k of n; None when there is nothing to rate."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> Optional[float]:
    return k / n if n > 0 else None


def task_runs(workdir: Path) -> dict[str, list[Run]]:
    """task_id -> the recorded Runs the Examiner's rulings compare versions over."""
    body = _read(workdir / "exam" / "task_runs.json", {}) or {}
    out: dict[str, list[Run]] = {}
    for task_id, rows in body.items():
        runs = []
        for row in rows or []:
            try:
                runs.append(row if isinstance(row, Run) else Run.model_validate(row))
            except ValueError:
                continue
        out[str(task_id)] = runs
    return out


def run_trace_of(workdir: Path) -> dict[str, dict[str, Optional[str]]]:
    """task_id -> {run_id: trace_id} over every recording the references row names."""
    references = _read(workdir / "references.json", {}) or {}
    out: dict[str, dict[str, Optional[str]]] = {}
    for task_id, row in references.items():
        out[str(task_id)] = {str(r.get("run_id")): r.get("trace_id")
                             for r in (row.get("recordings") or []) if r.get("run_id")}
    return out


def load_verifier(workdir: Path, task_id: str) -> Optional[Verifier]:
    path = workdir / "verifiers" / f"{task_id}.json"
    body = _read(path, None)
    if body is None:
        return None
    try:
        return Verifier.model_validate(body)
    except ValueError:
        return None


def load_canon(workdir: Path) -> CanonRules:
    return load_rules(workdir / "canon-rules.json")


def canon_fn_of(workdir: Path):
    rules = load_canon(workdir)
    return lambda value: canon_value(value, rules=rules)


def write_tools_of(workdir: Path) -> set[str]:
    sigs = _read(workdir / "tool_sigs.json", []) or []
    names = sigs if isinstance(sigs, list) else sigs.get("sigs", [])
    return {str(s.get("name")) for s in names if isinstance(s, dict) and s.get("kind") == "write"}


def intent_text_of(workdir: Path) -> dict[str, Optional[str]]:
    """task_id -> the Intent words, where the workdir carries them."""
    out: dict[str, Optional[str]] = {}
    body = _read(workdir / "tasks.json", {}) or {}
    tasks = body.get("tasks", body) if isinstance(body, dict) else body
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict) or not task.get("id"):
            continue
        intent = task.get("intent")
        text = intent.get("text") if isinstance(intent, dict) else intent
        out[str(task["id"])] = str(text) if text else (str(task["name"]) if task.get("name") else None)
    return out


def policy_rules_of(workdir: Path) -> set[str]:
    """The names of the compiled policy rules, whatever the constraints file holds."""
    body = _read(workdir / "constraints_check.json", {}) or {}
    constraints = body.get("constraints", body) if isinstance(body, dict) else body
    names = set()
    for constraint in constraints if isinstance(constraints, list) else []:
        if isinstance(constraint, dict) and constraint.get("name"):
            names.add(str(constraint["name"]))
    return names


def because_ok(atom: Any, intent_text: Optional[str], policy_rules: set[str]) -> bool:
    """A check's because is valid: its quote sits in the Intent, or its rule names policy."""
    because = (atom.target or {}).get("because") if getattr(atom, "target", None) else None
    if not isinstance(because, dict):
        return False
    quote, rule = because.get("quote"), because.get("rule")
    if quote and intent_text and str(quote) in intent_text:
        return True
    if rule and str(rule) in policy_rules:
        return True
    return False


def build_cost_usd(workdir: Path) -> Optional[float]:
    total = (_read(workdir / "budget.json", {}) or {}).get("total") or {}
    usd = total.get("usd")
    return float(usd) if usd is not None else None


def run_signature(workdir: Path) -> dict[str, list[str]]:
    """task_id -> sorted run ids, the Run sets two workdirs must share to compare."""
    return {task_id: sorted(run.run_id for run in runs) for task_id, runs in task_runs(workdir).items()}
