"""The solve-rate stage: a cheap model over the Tasks, through the interface (G1).

One table per run: Task, Runs, passes, no-signal Runs counted apart, solve rate over scoreable
Runs only. A no-signal Run is one whose reward is None: not verdicted (a must-hold judge atom no
code result covers) or an environment failure. Those Runs say nothing about the policy, so they
stand outside the rate and inside the count.

It is a measuring instrument, not a trainer: the policy is handed in, every Run is recorded under
the Episode's own folder, and the spend stops at the ceiling through the budget wrapper. No live
model call happens in this change; the tests drive it with a scripted policy.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Optional

from kullback import sampling
from kullback.episode.environment import BuiltEnvironment, episode_message
from kullback.episode.episode import Episode, EpisodeError
from kullback.runner import budget
from kullback.runner.records import write_json

SOLVE_SEED_KIND = "solve_seed"  # D212 after: one attempt's seed from its Task and number, keyed apart
STAGE = "solve-rate"


def attempt_seed(env: BuiltEnvironment, task_id: str, attempt: int) -> int:
    """The seed of one attempt, drawn from its Task and number under the build salt."""
    return sampling.sample_seed(SOLVE_SEED_KIND, f"{task_id}-{attempt}", env.salt)


class _PolicyAdapter:
    def __init__(self, inner: Any):
        self.inner = inner
        self.name = getattr(inner, "name", None) or "model"

    @property
    def last_hit(self) -> bool:
        return bool(getattr(self.inner, "last_hit", False))

    def query(self, messages, tools=None, config=None):
        return self.inner.query(messages, tools)


def _settings(runs_per_task: int, max_turns: int, ceiling_usd: Optional[float]) -> None:
    if any(type(value) is not int or value < 1 for value in (runs_per_task, max_turns)):
        raise ValueError("runs_per_task and max_turns must be positive integers")
    if ceiling_usd is not None and (isinstance(ceiling_usd, bool) or not math.isfinite(ceiling_usd) or ceiling_usd <= 0):
        raise ValueError("ceiling_usd must be finite and positive")


def _termination(episode: Episode) -> Optional[str]:
    """The Run's stop reason where the record can still be read, else nothing."""
    try:
        return episode.run_record().get("termination_reason")
    except Exception:
        return None


def _attempt_cost(episode: Episode, model: Any, before: dict) -> None:
    after = budget.load_totals(model.workdir)["total"]
    fields = ("usd", "calls", "unpriced_calls", "memo_hits", "input", "output", "cache_read", "cache_write")
    write_json(episode.run_path().with_suffix(".cost.json"), {
        "run_id": episode.run_id, "model": model.model_id,
        "termination_reason": _termination(episode),
        "accounting": "captured_usage_estimate",
        **{name: after[name] - before[name] for name in fields},
    })


def _attempt(env: BuiltEnvironment, task_id: str, attempt: int, model: Any,
             outdir: Any, max_turns: int) -> tuple[Any, bool]:
    """One attempt measured by captured usage; interrupted attempts carry no reward."""
    episode = Episode(env, outdir=outdir, max_turns=max_turns)
    info = episode.reset(task_id, attempt_seed(env, task_id, attempt))
    before = budget.load_totals(model.workdir)["total"]
    try:
        while True:
            reply = model.query(episode.transcript(), info.tools)
            out = episode.step(episode_message(getattr(reply, "content", None),
                                               getattr(reply, "tool_calls", None)))
            if out.done:
                return episode.reward().score, False
    except budget.BudgetExceeded:
        episode.abort("budget_exceeded")
        return None, True
    except EpisodeError:
        if _termination(episode) == "env_error":
            return None, False
        episode.abort("policy_error")
        raise
    except (KeyboardInterrupt, SystemExit):
        episode.abort("cancelled")
        raise
    except Exception:
        episode.abort("policy_error")
        raise
    finally:
        _attempt_cost(episode, model, before)


def _task_row(env, task_id, model, runs_per_task, outdir, max_turns, stopped):
    row = {"task": task_id, "runs": 0, "passes": 0, "no_signal": 0, "interrupted": 0, "solve_rate": None}
    for attempt in range(runs_per_task):
        if stopped:
            break
        score, stopped = _attempt(env, task_id, attempt, model, outdir, max_turns)
        if stopped:
            row["interrupted"] += 1
            break
        row["runs"] += 1
        if score is None:
            row["no_signal"] += 1
        elif score == 1:
            row["passes"] += 1
    scoreable = row["runs"] - row["no_signal"]
    row["solve_rate"] = row["passes"] / scoreable if scoreable else None
    return row, stopped


def solve_rate(env: BuiltEnvironment, task_ids: list[str], model: Any, *, runs_per_task: int = 1,
               ceiling_usd: Optional[float] = None, outdir: Any = None,
               max_turns: int = 30) -> dict:
    """Run the policy over Tasks; spend is the captured usage estimate with D86 post-call stop."""
    _settings(runs_per_task, max_turns, ceiling_usd)
    ceiling = budget.Ceiling(usd=ceiling_usd) if ceiling_usd is not None else None
    adapter = _PolicyAdapter(model)
    if ceiling is not None:
        ceiling.require_priced(adapter.name)
    root = Path(outdir) if outdir is not None else env.root / "episodes"
    root.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="solve-", dir=root))
    policy = budget.BudgetedModel(adapter, STAGE, session, model_id=adapter.name,
                                   ceiling=ceiling, cap_context=False)
    rows = []
    stopped = False
    for task_id in task_ids:
        row, stopped = _task_row(env, task_id, policy, runs_per_task, session, max_turns, stopped)
        rows.append(row)
    totals = budget.load_totals(session)["total"]
    table = {"tasks": rows, "runs_per_task": runs_per_task, "spent_usd": totals["usd"],
             "unpriced_calls": totals["unpriced_calls"], "calls": totals["calls"],
             "memo_hits": totals["memo_hits"], "model": adapter.name, "max_turns": max_turns,
             "ceiling_usd": ceiling_usd, "ceiling_stopped": stopped,
             "ceiling_mode": "D86_post_call", "accounting": "captured_usage_estimate",
             "session_dir": str(session)}
    write_json(session / "solve_rate.json", table)
    write_json(root / "solve_rate.json", table)
    return table


def markdown_table(table: dict) -> list[str]:
    """The table as markdown rows: Task, Runs, passes, no-signal apart, rate over scoreable only."""
    lines = ["| task | runs | passes | no-signal | solve rate |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in table.get("tasks") or []:
        rate = row["solve_rate"]
        lines.append(f"| {row['task']} | {row['runs']} | {row['passes']} | {row['no_signal']} | "
                     + ("-" if rate is None else f"{rate:.3f}"))
    return lines
