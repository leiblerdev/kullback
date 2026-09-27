"""Shared curriculum records and counting rules for the training agent.

Pure code. No model calls. Records are written as JSON lines; code computes
every number the training agent reads when it decides what to ask for next.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Level = Literal["easy", "medium", "hard"]
LEVEL_ORDER: tuple[str, str, str] = ("easy", "medium", "hard")


class SolveRate(BaseModel):
    """One GRPO group for one task: k rollouts, split by outcome."""

    model_config = ConfigDict(extra="forbid")

    format: int = 1
    train_run_id: str
    task_id: str
    task_origin: Literal["recorded", "synthetic"]
    birth_level: Optional[Level] = None
    model_id: str
    checkpoint: str
    step: Optional[int] = None
    source: Literal["training", "smoke"]
    k: int = Field(ge=1)
    passes: int = Field(ge=0)
    fails: int = Field(ge=0)
    interrupted: int = Field(ge=0)
    no_signal: dict[str, int] = Field(default_factory=dict)
    stop_reasons: dict[str, int] = Field(default_factory=dict)
    user_driver: str
    sampling: dict = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    env_hash: dict[str, Optional[str]] = Field(default_factory=dict)
    run_files: list[str] = Field(default_factory=list)
    started_at: float
    ended_at: float

    @model_validator(mode="after")
    def _counts_match_k(self) -> "SolveRate":
        total = self.passes + self.fails + sum(self.no_signal.values()) + self.interrupted
        if total != self.k:
            raise ValueError(f"passes + fails + no_signal + interrupted ({total}) must equal k ({self.k})")
        return self

    @property
    def scored(self) -> int:
        return self.passes + self.fails

    @property
    def solve_rate(self) -> Optional[float]:
        if self.scored == 0:
            return None
        return self.passes / self.scored

    @property
    def band(self) -> str:
        if self.scored == 0:
            return "no_signal"
        if self.passes == 0:
            return "all_fail"
        if self.fails == 0:
            return "all_pass"
        return "mixed"


def _comb_or_zero(n: int, k: int) -> int:
    if k < 0 or n < k:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    """Unbiased pass@k estimate from n samples with c passes."""
    if k < 1 or n < k:
        return None
    return 1.0 - _comb_or_zero(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> Optional[float]:
    """Unbiased pass^k estimate from n samples with c passes."""
    if k < 1 or n < k:
        return None
    return _comb_or_zero(c, k) / math.comb(n, k)


class TaskAsk(BaseModel):
    """A request for new tasks of one level, published on the bus."""

    model_config = ConfigDict(extra="forbid")

    format: int = 1
    ask_id: str
    train_run_id: str
    model_id: str
    checkpoint: str
    step: int
    level: Level
    kind: Optional[str] = None
    count: int = Field(ge=1, le=60)
    evidence: dict = Field(default_factory=dict)
    exemplars: list[str] = Field(default_factory=list, max_length=5)

    @staticmethod
    def ask_id_for(train_run_id: str, step: int, level: str) -> str:
        raw = "|".join((str(train_run_id), str(step), str(level)))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def append(path: str | Path, record: SolveRate) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def read(path: str | Path) -> list[SolveRate]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[SolveRate] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(SolveRate.model_validate_json(line))
    return out


class TaskView(BaseModel):
    """Pooled record for one task over the folded window."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    birth_level: Optional[Level] = None
    runs: int = Field(ge=0)
    passes: int = Field(ge=0)
    fails: int = Field(ge=0)
    no_signal: int = Field(ge=0)
    solve_rate: Optional[float] = None
    band: Literal["no_signal", "all_fail", "all_pass", "mixed"]


def _pooled_band(passes: int, fails: int) -> str:
    if passes + fails == 0:
        return "no_signal"
    if passes == 0:
        return "all_fail"
    if fails == 0:
        return "all_pass"
    return "mixed"


def window_records(records: list[SolveRate], last_checkpoints: int = 3) -> list[SolveRate]:
    """The records of the last N checkpoints, ordered by max ended_at.

    Step None records each count as their own checkpoint.
    """
    if last_checkpoints < 1 or not records:
        return []
    bucket_end: dict[tuple, float] = {}
    for idx, r in enumerate(records):
        key = ("step", r.step) if r.step is not None else ("none", idx)
        prev = bucket_end.get(key)
        if prev is None or r.ended_at > prev:
            bucket_end[key] = r.ended_at
    ordered = sorted(bucket_end, key=lambda k: (bucket_end[k], str(k)))
    kept_keys = set(ordered[-last_checkpoints:])
    return [r for idx, r in enumerate(records)
            if ((("step", r.step) if r.step is not None else ("none", idx)) in kept_keys)]


def fold(records: list[SolveRate], last_checkpoints: int = 3) -> list[TaskView]:
    """Keep the last N checkpoints, then pool counts per task.

    Step None records each count as their own checkpoint. Checkpoints are
    ordered by max ended_at and the last N are kept. runs is the pooled k.
    """
    kept = window_records(records, last_checkpoints)
    by_task: dict[str, list[SolveRate]] = {}
    for r in kept:
        by_task.setdefault(r.task_id, []).append(r)
    views: list[TaskView] = []
    for task_id in sorted(by_task):
        group = sorted(by_task[task_id], key=lambda r: r.ended_at)
        runs = sum(r.k for r in group)
        passes = sum(r.passes for r in group)
        fails = sum(r.fails for r in group)
        no_signal = sum(sum(r.no_signal.values()) for r in group)
        scored = passes + fails
        views.append(
            TaskView(
                task_id=task_id,
                birth_level=group[-1].birth_level,
                runs=runs,
                passes=passes,
                fails=fails,
                no_signal=no_signal,
                solve_rate=(passes / scored if scored else None),
                band=_pooled_band(passes, fails),
            )
        )
    return views


def _level_counts(views: list[TaskView]) -> dict:
    scored = [v for v in views if v.passes + v.fails > 0]
    mixed = sum(1 for v in views if v.band == "mixed")
    runs = sum(v.runs for v in views)
    nosig_runs = sum(v.no_signal for v in views)
    return {
        "tasks": len(views),
        "mixed": mixed,
        "all_pass": sum(1 for v in views if v.band == "all_pass"),
        "all_fail": sum(1 for v in views if v.band == "all_fail"),
        "no_signal": sum(1 for v in views if v.band == "no_signal"),
        "mixed_share": (mixed / len(scored) if scored else None),
        "no_signal_share": (nosig_runs / runs if runs else None),
    }


def window_summary(views: list[TaskView]) -> dict:
    """Task counts, shares, and the same counts per birth level."""
    summary = _level_counts(list(views))
    by_level: dict[str, dict] = {}
    groups: dict[str, list[TaskView]] = {}
    for v in views:
        key = v.birth_level if v.birth_level is not None else "none"
        groups.setdefault(key, []).append(v)
    for key in sorted(groups):
        by_level[key] = _level_counts(groups[key])
    summary["by_level"] = by_level
    return summary


def _pct(x: float) -> int:
    return int(round(x * 100))


def _no_signal_action(summary: dict, no_signal_bar: float):
    """The report no signal answer, or None where the share is below the bar."""
    nosig = summary.get("no_signal_share")
    if nosig is not None and nosig >= no_signal_bar:
        return {
            "action": "report_no_signal",
            "reason": (
                f"No signal share is {_pct(nosig)} percent across "
                f"{summary.get('tasks', 0)} tasks "
                f"at the {_pct(no_signal_bar)} percent bar so reporting."
            ),
        }
    return None


def _saturated_action(by_level: dict, levels: list, saturated_bar: float):
    """The saturation answer, or None where no level is saturated."""
    saturated: list[str] = []
    for lv in levels:
        entry = by_level[lv]
        scored = entry["tasks"] - entry["no_signal"]
        if scored > 0 and entry["all_pass"] / scored >= saturated_bar:
            saturated.append(lv)
    if not saturated:
        return None
    top = max(saturated, key=lambda lv: LEVEL_ORDER.index(lv))
    nxt = LEVEL_ORDER[LEVEL_ORDER.index(top) + 1] if top != "hard" else None
    entry = by_level[top]
    scored = entry["tasks"] - entry["no_signal"]
    if nxt is None:
        return {
            "action": "none",
            "reason": (
                f"{top} has {entry['all_pass']} of {scored} scored tasks all pass "
                f"at the {_pct(saturated_bar)} percent bar "
                f"with no higher level so no ask."
            ),
        }
    harder_count = max(1, min(60, entry["all_pass"]))
    return {
        "action": "ask_harder",
        "level": nxt,
        "count": harder_count,
        "reason": (
            f"{top} has {entry['all_pass']} of {scored} scored tasks all pass "
            f"at the {_pct(saturated_bar)} percent bar so asking {harder_count} {nxt} tasks."
        ),
    }


def _thin_ask_action(summary: dict, by_level: dict, levels: list, mixed_bar: float):
    """The thin band ask, or None where the band is healthy."""
    mixed_share = summary.get("mixed_share")
    mixed = summary.get("mixed", 0)
    scored_tasks = summary.get("tasks", 0) - summary.get("no_signal", 0)
    needs_ask = mixed_share is None or mixed_share < mixed_bar
    if not needs_ask:
        return None
    candidates = [(lv, by_level[lv]["mixed_share"]) for lv in levels if by_level[lv].get("mixed_share") is not None]
    if candidates:
        best = candidates[0][0]
        best_share = candidates[0][1]
        for lv, share in candidates[1:]:
            if share > best_share:
                best, best_share = lv, share
        level = best
    elif summary.get("all_fail", 0) > summary.get("all_pass", 0):
        level = "easy"
    else:
        level = "medium"
    count = int(math.ceil((mixed_bar * scored_tasks - mixed) / (1 - mixed_bar)))
    count = max(1, min(60, count))
    have = "none" if mixed_share is None else f"{_pct(mixed_share)} percent"
    return {
        "action": "ask",
        "level": level,
        "count": count,
        "reason": (
            f"Mixed share is {have} with {mixed} of {scored_tasks} scored tasks mixed "
            f"below the {_pct(mixed_bar)} percent bar so asking {count} {level} tasks."
        ),
    }


def decide(
    summary: dict,
    *,
    mixed_bar: float = 0.30,
    no_signal_bar: float = 0.05,
    saturated_bar: float = 0.80,
) -> dict:
    """Apply the thin band rules in order: no signal, saturation, thin band."""
    if mixed_bar >= 1:
        raise ValueError("mixed_bar must be below 1")
    by_level: dict = summary.get("by_level", {}) or {}
    levels = [lv for lv in LEVEL_ORDER if lv in by_level]
    hit = _no_signal_action(summary, no_signal_bar)
    if hit is not None:
        return hit
    hit = _saturated_action(by_level, levels, saturated_bar)
    if hit is not None:
        return hit
    hit = _thin_ask_action(summary, by_level, levels, mixed_bar)
    if hit is not None:
        return hit
    mixed_share = summary.get("mixed_share")
    mixed = summary.get("mixed", 0)
    scored_tasks = summary.get("tasks", 0) - summary.get("no_signal", 0)
    have = "none" if mixed_share is None else f"{_pct(mixed_share)} percent"
    return {
        "action": "none",
        "reason": (
            f"Mixed share is {have} with {mixed} of {scored_tasks} scored tasks mixed "
            f"at or above the {_pct(mixed_bar)} percent bar "
            f"with no saturation so no ask."
        ),
    }
