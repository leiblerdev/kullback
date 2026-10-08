"""The Spec stage of a build: each named Task's Intent mined, its Spec and Verifier written, the bus told.

A Task whose Spec and Spec Verifier are both on disk is skipped, so a resumed build writes only
what is missing; a Spec of the older demand grammar (no item) is rewritten as items, Intent and all,
so every Task gets the sanity item (automatic, no flag). An empty Spec is written once more; empty again, the Task is set aside with no
Verifier. A Spec no Run can pass after its one send-back is set aside the same way, as
unsatisfiable_spec. A set-aside Spec counts as done. Every call is priced under the `spec` stage of the workdir ledger (budget.json):
the miner through the harness subscriber, the writer through BudgetedModel, the two seams every
other stage uses. A Task that fails is counted by its error class and the loop goes on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.agent.bus import Bus
from kullback.runner.budget import BudgetedModel
from kullback.spec import events
from kullback.spec.intent import STAGE, mine_intent
from kullback.spec.schema import SpecIntent, is_empty, load_spec, save_spec, spec_path
from kullback.spec.writer import is_item_check, load_inputs, spec_verifier_path, write_spec, write_verifier

# What one Task's mining and writing may spend together when the caller names no ceiling.
SPEC_TASK_CEILING_USD = 1.0


def older_grammar(spec: Any) -> bool:
    """A Spec of the older demand grammar: checks, none of them an item, so its Verifier has no sanity item."""
    return spec is not None and bool(spec.checks) and not any(is_item_check(check) for check in spec.checks)


def written_already(workdir: Path, task_id: str) -> bool:
    """A Task whose Spec and its compiled Verifier are both on disk, or whose Spec is set aside; a Spec of
    the older grammar is not written yet: the stage rewrites it as items."""
    if not spec_path(workdir, task_id).is_file():
        return False
    spec = load_spec(workdir, task_id)
    if spec.set_aside is not None:
        return True
    return spec_verifier_path(workdir, task_id).is_file() and not older_grammar(spec)


def _version(workdir: Path, task_id: str) -> int:
    spec = load_spec(workdir, task_id)
    return spec.version if spec is not None else 0


def write_from_intent(workdir: Path, task_id: str, intent: SpecIntent, model: Any, ceiling_usd: float,
                      bus: Bus) -> float:
    """Write and compile one Task's Spec from a given Intent under `intent`; return the spend.

    An empty first Spec is written once more in a fresh session. Empty again, the Spec is saved set aside
    as empty_spec, task.set_aside is published and no Verifier is written; a Spec no Run can pass is set
    aside so as unsatisfiable_spec; else spec.written is published.
    """
    model_id = getattr(model, "name", None)
    inputs = load_inputs(workdir, task_id, intent)
    priced = BudgetedModel(model, stage=STAGE, workdir=workdir, model_id=model_id)
    tries = [write_spec(task_id, inputs, priced, model_id=model_id, ceiling_usd=ceiling_usd)]
    empty_first = is_empty(tries[0].spec)
    if empty_first:
        left = max(ceiling_usd - float(tries[0].session.get("usd") or 0.0), 0.0)
        tries.append(write_spec(task_id, inputs, priced, model_id=model_id, ceiling_usd=left))
    written, empty = tries[-1], is_empty(tries[-1].spec)
    spent = sum(float(t.session.get("usd") or 0.0) for t in tries)
    rounds = sum(int(t.session.get("tool_rounds") or 0) for t in tries)
    writer = dict(written.counts, usd=spent, read_rounds=rounds, attempts=len(tries), empty_first=empty_first,
                  empty=empty)
    if len(tries) > 1:
        writer["first_attempt"] = dict(tries[0].counts, usd=float(tries[0].session.get("usd") or 0.0))
    spec = written.spec.model_copy(update={"version": _version(workdir, task_id) + 1, "writer": writer})
    aside = events.EMPTY_SPEC if empty else events.UNSATISFIABLE_SPEC if written.counts.get("unsatisfiable") else None
    if aside:
        save_spec(workdir, spec.model_copy(update={"set_aside": aside}))
        events.task_set_aside(bus, task_id, aside, 0)
        return spent
    write_verifier(spec, workdir, inputs)
    events.spec_written(bus, task_id, spec.version)
    return spent


def write_one(workdir: Path, task_id: str, model: Any, ceiling_usd: float, bus: Bus) -> float:
    """Mine one Task's Intent, then write its Spec from it (`write_from_intent`); return the spend."""
    mined = mine_intent(workdir, task_id, model, ceiling_usd=ceiling_usd)
    rest = max(ceiling_usd - mined.spent_usd, 0.0)
    return mined.spent_usd + write_from_intent(workdir, task_id, mined.intent, model, rest, bus)


def write_specs(workdir: Any, task_ids: Iterable[str], model: Any, *,
                ceiling_usd: Optional[float] = None, bus: Optional[Bus] = None) -> dict:
    """Write the Spec of every named Task that has none; counts of written, skipped, failed, empty, and the spend."""
    workdir = Path(workdir)
    bus = bus or Bus(workdir / "bus.jsonl", agent="spec")
    ceiling = SPEC_TASK_CEILING_USD if ceiling_usd is None else ceiling_usd
    counts: dict = {"written": 0, "skipped_existing": 0, "failed": 0, "spent_usd": 0.0, "failures": {},
                    "empty_seen": 0, "retried": 0, "set_aside_empty": 0, "set_aside_unsatisfiable": 0,
                    "rewritten_older": 0}
    for task_id in task_ids:
        if written_already(workdir, task_id):
            counts["skipped_existing"] += 1
            continue
        counts["rewritten_older"] += int(older_grammar(load_spec(workdir, task_id)))
        try:
            counts["spent_usd"] += write_one(workdir, task_id, model, ceiling, bus)
        except Exception as error:  # one Task's failure is recorded; the others are still written
            counts["failed"] += 1
            counts["failures"][task_id] = type(error).__name__
            continue
        spec = load_spec(workdir, task_id)
        counts["empty_seen"] += int(bool(spec.writer.get("empty_first")))
        counts["retried"] += int(spec.writer.get("attempts", 1) > 1)
        aside = {events.EMPTY_SPEC: "set_aside_empty", events.UNSATISFIABLE_SPEC: "set_aside_unsatisfiable"}
        counts[aside.get(spec.set_aside, "written")] += 1
    return counts


__all__ = ["SPEC_TASK_CEILING_USD", "older_grammar", "write_from_intent", "write_one", "write_specs", "written_already"]
