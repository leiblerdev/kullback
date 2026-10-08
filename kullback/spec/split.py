"""The Reference class of a Task from the benchmark's own rewards: measurement only, never read by trust.

A Task's kept References are right when every scored one has reward 1, wrong when every one has reward 0,
mixed otherwise, and unknown when none is scored. The sidecar carries one reward per recording, found by
trace id. scripts/measure/trust_split.py and kullback/spec/report.py both read the split here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

RIGHT = "right"
WRONG = "wrong"
MIXED = "mixed"
UNKNOWN = "unknown"
CLASSES = (RIGHT, WRONG, MIXED, UNKNOWN)

SIDECAR_DIR = "grader"
TRACES_DIR = "traces"
REWARD_FIELD = "reward_info.reward"


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _dig(doc: Any, dotted: str) -> Any:
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _hash_to_trace(traces_dir: Optional[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    if traces_dir is not None and traces_dir.is_dir():
        for path in sorted(traces_dir.glob("*.json")):
            trace = _read(path, {}) or {}
            if trace.get("hash") and trace.get("trace_id"):
                out[str(trace["hash"])] = str(trace["trace_id"])
    return out


def _reward(doc: dict, reward_field: str) -> Optional[float]:
    fields = doc.get("fields") or {}
    reward = _dig(fields, reward_field)
    if reward is None:
        reward = fields.get("reward", doc.get("reward"))
    try:
        return None if reward is None else float(reward)
    except (TypeError, ValueError):
        return None


def sidecar_rewards(
    sidecar_dir: Path, reward_field: str, traces_dir: Optional[Path]
) -> tuple[dict[str, float], int]:
    """trace id -> the benchmark's own reward, plus how many sidecars were read.

    The trace id is read off the sidecar itself; where the sidecar carries none,
    the file stem (the recording hash) is resolved through the traces. The reward
    is the dotted field inside the sidecar's fields, falling back to a top-level
    reward in either place.
    """
    files = sorted(sidecar_dir.glob("*.json")) if sidecar_dir.is_dir() else []
    hash_to_trace = _hash_to_trace(traces_dir)
    reward_of: dict[str, float] = {}
    for path in files:
        doc = _read(path, {}) or {}
        trace_id = doc.get("trace_id") or hash_to_trace.get(path.stem)
        reward = _reward(doc, reward_field) if trace_id else None
        if reward is not None:
            reward_of[str(trace_id)] = reward
    return reward_of, len(files)


def classify(rewards: list[float]) -> str:
    if not rewards:
        return UNKNOWN
    if all(r == 1.0 for r in rewards):
        return RIGHT
    if all(r == 0.0 for r in rewards):
        return WRONG
    return MIXED


def reference_classes(workdir: Any, sidecar: Optional[Any] = None) -> Optional[dict[str, str]]:
    """task_id -> the class of its kept References' rewards, None without a sidecar folder."""
    root = Path(workdir)
    folder = Path(sidecar) if sidecar is not None else root / SIDECAR_DIR
    if not folder.is_dir():
        return None
    rewards, _files = sidecar_rewards(folder, REWARD_FIELD, root / TRACES_DIR)
    references = _read(root / "references.json", None) or {}
    out = {}
    for task_id, row in references.items() if isinstance(references, dict) else ():
        traces = [(ref or {}).get("trace_id") for ref in (row or {}).get("references") or []]
        out[str(task_id)] = classify([rewards[t] for t in traces if t in rewards])
    return out
