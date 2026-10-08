"""Local backends for observing training runs, read off files on disk.

All layout lives under `<workdir>/training/<run_id>/`. A missing file means
nothing has been written yet, never an error, so every reader here returns an
empty shape instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kullback import curriculum
from kullback.runner.world.environment import BuiltEnvironment

_ERROR_NEEDLES = ("Error", "Traceback", "CUDA out of memory")


def runs(workdir: Any) -> list[str]:
    """Run ids under training/, sorted; empty where nothing has run yet."""
    root = Path(workdir) / "training"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir()
                  and ((p / "state.json").is_file() or (p / "solve_rates.jsonl").is_file()))


def _run_dir(workdir: Any, run_id: str) -> Path:
    return Path(workdir) / "training" / run_id


def training_status(workdir: Any, run_id: str, tail: int = 40) -> dict:
    """State fields, the last log lines, error lines among them, rollouts rate."""
    folder = _run_dir(workdir, run_id)
    state: dict[str, Any] = {"status": None, "step": None, "checkpoint": None,
                             "updated_at": None}
    try:
        body = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = None
    if isinstance(body, dict):
        for key in state:
            state[key] = body.get(key)
    try:
        lines = (folder / "run.log").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    tail_lines = lines[-tail:] if tail > 0 else []
    errors = [line for line in tail_lines
              if any(needle in line for needle in _ERROR_NEEDLES)]
    records = curriculum.read(folder / "solve_rates.jsonl")
    rpm: Optional[float] = None
    if len(records) >= 2:
        span = max(r.ended_at for r in records) - min(r.started_at for r in records)
        if span > 0:
            rpm = sum(r.k for r in records) / (span / 60.0)
    return {**state, "log_tail": tail_lines, "errors": errors,
            "rollouts_per_minute": rpm}


def metrics(workdir: Any, run_id: str, window: int = 3) -> dict:
    """Bands, decision and pass estimates over the window, plus logged trends."""
    folder = _run_dir(workdir, run_id)
    records = curriculum.read(folder / "solve_rates.jsonl")
    views = curriculum.fold(records, window)
    bands = curriculum.window_summary(views)
    kept = curriculum.window_records(records, window)
    if views:
        spread = sum(1 for v in views if v.band != "mixed")
        zero_spread: Optional[float] = spread / len(views)
    else:
        zero_spread = None
    at_vals: list[float] = []
    hat_vals: list[float] = []
    for view in views:
        n = view.passes + view.fails
        if n < 4:
            continue
        at_vals.append(curriculum.pass_at_k(n, view.passes, 4))
        hat_vals.append(curriculum.pass_hat_k(n, view.passes, 4))
    stop: dict[str, int] = {}
    for record in kept:
        for name, count in record.stop_reasons.items():
            stop[name] = stop.get(name, 0) + count
    trend = _trend(folder / "metrics.jsonl")
    return {
        "bands": bands,
        "decision": curriculum.decide(bands),
        "zero_spread_share": zero_spread,
        "pass_at_4": (sum(at_vals) / len(at_vals) if at_vals else None),
        "pass_hat_4": (sum(hat_vals) / len(hat_vals) if hat_vals else None),
        "stop_reasons": stop,
        "trend": trend,
    }


def _metric_rows(path: Path) -> list[dict]:
    """The dict bodies of the metric lines, in file order."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            rows.append(body)
    return rows


def _trend(path: Path) -> dict:
    """First, last and mean per numeric key over the last 20 metric lines."""
    rows = _metric_rows(path)[-20:]
    out: dict[str, dict] = {}
    keys = sorted({k for row in rows for k, v in row.items()
                   if k != "step" and isinstance(v, (int, float))
                   and not isinstance(v, bool)})
    for key in keys:
        vals = [row[key] for row in rows
                if isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)]
        if vals:
            out[key] = {"first": vals[0], "last": vals[-1],
                        "mean": sum(vals) / len(vals)}
    return out


def task_pool(workdir_env: Any, since: Any = None) -> dict:
    """Task totals for a built environment, plus ids unseen in `since`."""
    env = BuiltEnvironment(workdir_env)
    ids = env.task_ids()
    known = set(since) if since is not None else None
    return {
        "total": len(ids),
        "agent_driven": sum(1 for tid in ids if env.agent_driven(tid)),
        "with_verifier": sum(1 for tid in ids if env.verifier(tid) is not None),
        "new": [tid for tid in ids if known is not None and tid not in known]
        if known is not None else [],
    }


def _ask_event(item: Any):
    """The event name and payload of one bus item, or None where it has none."""
    event = item.get("event") if isinstance(item, dict) else getattr(item, "event", None)
    if isinstance(event, dict):
        name = event.get("name")
        payload = event.get("payload") or {}
    else:
        name = getattr(event, "name", None)
        payload = getattr(event, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    return name, payload


def open_asks(events: Any, train_run_id: str) -> tuple[int, int, list[str]]:
    """Asked count, answered count and open ask ids for one run."""
    asked: list[str] = []
    answered: set[str] = set()
    for item in events:
        found = _ask_event(item)
        if found is None:
            continue
        name, payload = found
        if name == "task_ask" and payload.get("train_run_id") == train_run_id:
            ask_id = payload.get("ask_id")
            if ask_id is not None:
                asked.append(ask_id)
        elif name == "task_ask_answered" and payload.get("ask_id") in asked:
            answered.add(payload.get("ask_id"))
    asked_count = len(asked)
    answered_count = sum(1 for a in asked if a in answered)
    open_ids = list(dict.fromkeys(a for a in asked if a not in answered))
    return asked_count, answered_count, open_ids


def synth_status(bus_path: Any, train_run_id: str) -> dict:
    """Ask counts and open ask ids for one run, read off bus.jsonl lines."""
    try:
        text = Path(bus_path).read_text(encoding="utf-8")
    except OSError:
        return {"asked": 0, "answered": 0, "open_ask_ids": []}
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            events.append(body)
    asked, answered, open_ids = open_asks(events, train_run_id)
    return {"asked": asked, "answered": answered, "open_ask_ids": open_ids}
