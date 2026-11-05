"""The rollup of a call log (calllog.py): a calls.jsonl read back and added up per stage, per step and in total.

A step is one label within a stage (the label a caller set with calllog.labelled). Each bucket holds the calls, every token
kind, the cached share of all input (and on the first call of a group, a Run, against the later ones), the
dollars actually paid beside the gateway's estimate (the list price on a subscription route) and the upstream's billed cost, the wall time
p50 and p90, retries, errors, cache misses and the expected-cached counters. A pin row (a model a run pinned) is
no call: it is printed, never counted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

CALLS_FILE = "calls.jsonl"
PIN = "pin"  # the kind of a row that records a pinned model, not a call
TOKENS = ("input", "cache_read", "cache_write", "output", "reasoning")
DOLLARS = ("usd_actual", "usd_gateway", "usd_billed")
TARGET = 0.95  # the founder's target for actual / expected cached
TOTAL = "total"


def read(path: Path | str) -> list[dict]:
    """The rows of a calls.jsonl, or of <path>/calls.jsonl when path is a folder; unreadable lines skipped."""
    path = Path(path)
    path = path / CALLS_FILE if path.is_dir() else path
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def calls_only(rows: Iterable[dict]) -> list[dict]:
    """The rows that are calls (a pin row records a choice, not a call)."""
    return [row for row in rows if row.get("kind") != PIN]


def pins(rows: Iterable[dict]) -> list[dict]:
    return [row for row in rows if row.get("kind") == PIN]


def _share(cached: int, sent: int) -> float:
    return round(cached / sent, 4) if sent else 0.0


def _bucket() -> dict:
    return {"calls": 0, **dict.fromkeys(TOKENS, 0), **dict.fromkeys(DOLLARS, 0.0), "reasoning_unreported": 0,
            "retries": 0, "errors": 0, "failed_statuses": {}, "expected_cached": 0, "actual_on_expected": 0,
            "misses": 0, "uncacheable_shared": 0, "first": [0, 0], "later": [0, 0], "wall": []}


def _add(b: dict, row: dict, first: Optional[bool]) -> None:
    b["calls"] += 1
    for key in TOKENS:
        b[key] += int(row.get(key) or 0)
    b["reasoning_unreported"] += "reasoning" in row and row["reasoning"] is None
    for key in DOLLARS:
        b[key] += float(row.get(key) or 0.0)
    b["retries"] += int(row.get("attempt") or 1) > 1 or int(row.get("http_attempts") or 1) > 1
    b["errors"] += bool(row.get("error"))
    for status in row.get("failed_statuses") or ():
        b["failed_statuses"][str(status)] = b["failed_statuses"].get(str(status), 0) + 1
    b["uncacheable_shared"] += int(row.get("uncacheable_shared") or 0)
    if row.get("expected_cached"):
        b["expected_cached"] += int(row["expected_cached"])
        b["actual_on_expected"] += int(row.get("cache_read") or 0)
    b["misses"] += bool(row.get("miss"))
    b["wall"].append(float(row.get("wall_ms") or 0.0))
    if first is not None:
        when = b["first" if first else "later"]
        when[0] += int(row.get("cache_read") or 0)
        when[1] += int(row.get("input") or 0) + int(row.get("cache_read") or 0)


def percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile (0 for no values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


def _finish(b: dict) -> dict:
    for key in DOLLARS:
        b[key] = round(b[key], 6)
    b["cached_share"] = _share(b["cache_read"], b["input"] + b["cache_read"])
    b["actual_over_expected"] = _share(b["actual_on_expected"], b["expected_cached"])
    b["cached_share_first"] = _share(*b.pop("first"))
    b["cached_share_later"] = _share(*b.pop("later"))
    wall = b.pop("wall")
    b["p50_ms"], b["p90_ms"] = round(percentile(wall, 0.5)), round(percentile(wall, 0.9))
    return b


def rollup(rows: Iterable[dict], by: str = "stage") -> dict[str, dict]:
    """The calls added up per stage (by="stage"), per step, a stage and a label (by="step", keyed "stage/label"),
    or all together (by="total", one key). The first call of a group (a Run) within a stage counts as first."""
    out: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for row in calls_only(rows):
        name, group = str(row.get("stage") or ""), row.get("group")
        first = None if group is None else (name, str(group)) not in seen
        if group is not None:
            seen.add((name, str(group)))
        key = {"stage": name, "step": f"{name}/{row.get('label') or ''}", "total": TOTAL}[by]
        _add(out.setdefault(key, _bucket()), row, first)
    return {key: _finish(b) for key, b in out.items()}


def _line(kind: str, name: str, b: dict) -> str:
    mean = round((b["input"] + b["cache_read"]) / b["calls"]) if b["calls"] else 0
    statuses = ", ".join(f"{k}x{v}" for k, v in sorted(b["failed_statuses"].items())) or "none"
    return (f"{kind} {name}: {b['calls']} calls, input {b['input']} uncached + {b['cache_read']} cached (mean "
            f"{mean}/call, cached share {b['cached_share']:.2%}, first {b['cached_share_first']:.2%}, later "
            f"{b['cached_share_later']:.2%}), cache_write {b['cache_write']}, output {b['output']} (reasoning "
            f"{b['reasoning']}, unreported on {b['reasoning_unreported']}), usd actual {b['usd_actual']:.4f} "
            f"(gateway {b['usd_gateway']:.4f}, billed {b['usd_billed']:.4f}), p50 {b['p50_ms']} ms, p90 {b['p90_ms']} ms, "
            f"retries {b['retries']}, failed statuses {statuses}, errors {b['errors']}, misses {b['misses']}")


def _cache_line(name: str, b: dict) -> str:
    return (f"cache {name}: actual/expected {b['actual_over_expected']:.2%} (target {TARGET:.0%}; "
            f"{b['actual_on_expected']}/{b['expected_cached']}), misses {b['misses']}, uncacheable shared "
            f"{b['uncacheable_shared']}")


def rollup_lines(rows: Iterable[dict]) -> list[str]:
    """The pinned models, then per stage a calls line and a cache line, per step a calls line, and the total."""
    rows = list(rows)
    lines = [f"model {row['role']}: {row['model']} on {', '.join(row.get('routes') or [])} ({row['source']})"
             for row in pins(rows)]
    stages = rollup(rows)
    for name, b in sorted(stages.items()):
        lines += [_line("calls", name, b), _cache_line(name, b)]
    for name, b in sorted(rollup(rows, "step").items()):
        lines.append(_line("step", name, b))
    if stages:
        lines.append(_line("run", TOTAL, rollup(rows, "total")[TOTAL]))
    return lines or ["calls: none"]
