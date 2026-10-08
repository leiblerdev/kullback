"""The review loop: Examiner round, writer round, Examiner round, at most ROUNDS_CAP rounds (D331).

The writer's first round is the Spec itself. Each Examiner round files rulings (and from round two
closes or keeps open the ones the writer answered); each writer round answers every open ruling,
applied or rebutted. The loop stops early when no ruling is open. The Spec's `round` and
`rulings_open` carry the state; every round's counts go to the bus (review.round) and to
spec/state/rounds.jsonl, so a report reads what each round did.

The Examiner's round is injected (examiner/session.py review_specs), so the Spec never imports the
Examiner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.review import answer_rulings
from kullback.spec.router import ROUNDS_CAP

LOG_FILE = "spec/state/rounds.jsonl"


def log_path(workdir: Any) -> Path:
    return Path(workdir) / LOG_FILE


def _log(workdir: Any, row: dict) -> None:
    path = log_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def task_rows(workdir: Any, task_ids: Iterable[str]) -> dict[str, dict]:
    """Per Task: its rulings' counts and whether an open blocking ruling holds it untrusted."""
    out = {}
    for task_id in task_ids:
        rulings = R.load_rulings(workdir, task_id)
        out[task_id] = dict(R.counts(rulings), held=bool(R.open_blocking(rulings)))
    return out


def _route_reference_fails(root: Path, task_ids: list[str]) -> dict[str, list[int]]:
    """Trust's kept-Reference failures filed as rulings before the Examiner's first round (D333, fold)."""
    from kullback.spec.trust import intent_tiers, route_reference_fails

    if not (root / "runs").is_dir():
        return {}
    tiers = {task_id: tier for task_id, tier in intent_tiers(root).items() if task_id in task_ids}
    return {task_id: [n] for task_id in tiers for n in route_reference_fails(root, {task_id: tiers[task_id]})}


def _ask_confirm(root: Path, task_id: str, model: Any, ceiling_usd: Optional[float] = None) -> dict:
    from kullback.spec.writer import ask_confirm

    return ask_confirm(root, task_id, model, ceiling_usd)


def run_rounds(workdir: Any, task_ids: Iterable[str], *, examine: Callable[..., dict], model: Any,
               writer_model: Any = None, rounds: int = ROUNDS_CAP, ceiling_usd: Optional[float] = None,
               answer: Callable[..., dict] = answer_rulings, confirm: Optional[Callable[..., dict]] = None) -> dict:
    """Examiner and writer rounds over the Tasks until no ruling is open or `rounds` Examiner rounds ran.

    `examine(workdir, task_ids, model=, round_number=)` is one Examiner round; `answer(workdir, task_id,
    model, round_number, ceiling_usd=)` one writer answer per Task. Before the first round the
    writer is asked the fixed confirm question where a Spec writes with no confirm item (`confirm`,
    spec/writer.ask_confirm). Returns each round's counts, the findings the Examiner rounds filed
    for the Builder, and each Task's row at the end.
    """
    from kullback.agent.bus import Bus

    root = Path(workdir)
    task_ids = sorted(set(task_ids))
    bus = Bus(root / "bus.jsonl", agent="spec")
    rounds = min(int(rounds), ROUNDS_CAP)
    log: list[dict] = []
    found: list[dict] = []
    routed = _route_reference_fails(root, task_ids)
    if routed:
        _log(root, {"side": "trust", "reference_fails_filed": routed})
    confirm = confirm or _ask_confirm
    asked = {task_id: confirm(root, task_id, writer_model or model, ceiling_usd=ceiling_usd) for task_id in task_ids}
    if any(row.get("asked") for row in asked.values()):
        _log(root, {"side": "writer", "confirm_asked": sorted(t for t, row in asked.items() if row.get("asked")),
                    "confirm_added": sum(row.get("added", 0) for row in asked.values())})
    for number in range(1, rounds + 1):
        exam = examine(root, task_ids, model=model, round_number=number)
        for task_id in task_ids:
            R.sync_spec(root, task_id, number)
        row = {"round": number, "side": "examiner", "capped": exam.get("capped"), "counts": exam.get("counts", {})}
        found += exam.get("findings", [])
        events.review_round(bus, number, "examiner", row["counts"])
        _log(root, row)
        log.append(row)
        open_tasks = [t for t in task_ids if R.unanswered(R.load_rulings(root, t))]
        if number == rounds or not open_tasks:
            break
        answers = [answer(root, t, writer_model or model, number + 1, ceiling_usd=ceiling_usd) for t in open_tasks]
        totals = {key: sum(a.get(key, 0) for a in answers) for key in ("answered", "applied", "rebutted", "skipped")}
        row = {"round": number + 1, "side": "writer", "counts": totals, "tasks": answers}
        events.review_round(bus, number + 1, "writer", totals)
        _log(root, row)
        log.append(row)
    for task_id in task_ids:
        R.sync_spec(root, task_id)
    tasks = task_rows(root, task_ids)
    summary = {"rounds": log, "tasks": tasks, "held": sorted(t for t, row in tasks.items() if row["held"]),
               "total": R.counts(r for t in task_ids for r in R.load_rulings(root, t)), "findings": found}
    _log(root, {"side": "summary", "held": summary["held"], "total": summary["total"]})
    return summary


__all__ = ["LOG_FILE", "log_path", "run_rounds", "task_rows"]
