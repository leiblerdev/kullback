"""The Spec stage of a build: each named Task's Intent mined, its Spec and Verifier written, the bus told.

A Task whose Spec and Spec Verifier are both on disk is skipped, so a resumed build writes only
what is missing. Every call is priced under the `spec` stage of the workdir ledger (budget.json):
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
from kullback.spec.schema import load_spec, spec_path
from kullback.spec.writer import load_inputs, spec_verifier_path, write_spec, write_verifier

# What one Task's mining and writing may spend together when the caller names no ceiling.
SPEC_TASK_CEILING_USD = 1.0


def written_already(workdir: Path, task_id: str) -> bool:
    """A Task whose Spec and its compiled Verifier are both on disk; a Spec left half-written is not."""
    return spec_path(workdir, task_id).is_file() and spec_verifier_path(workdir, task_id).is_file()


def _version(workdir: Path, task_id: str) -> int:
    spec = load_spec(workdir, task_id)
    return spec.version if spec is not None else 0


def write_one(workdir: Path, task_id: str, model: Any, ceiling_usd: float, bus: Bus) -> float:
    """Mine, write and compile one Task's Spec under `intent`, publish spec.written, return the spend."""
    model_id = getattr(model, "name", None)
    mined = mine_intent(workdir, task_id, model, ceiling_usd=ceiling_usd)
    inputs = load_inputs(workdir, task_id, mined.intent)
    priced = BudgetedModel(model, stage=STAGE, workdir=workdir, model_id=model_id)
    written = write_spec(task_id, inputs, priced, model_id=model_id,
                         ceiling_usd=max(ceiling_usd - mined.spent_usd, 0.0))
    spec = written.spec.model_copy(update={"version": _version(workdir, task_id) + 1})
    write_verifier(spec, workdir, inputs, verifier_from="intent")
    events.spec_written(bus, task_id, spec.version)
    return mined.spent_usd + float(written.session.get("usd") or 0.0)


def write_specs(workdir: Any, task_ids: Iterable[str], model: Any, *,
                ceiling_usd: Optional[float] = None, bus: Optional[Bus] = None) -> dict:
    """Write the Spec of every named Task that has none; counts of written, skipped and failed, and the spend."""
    workdir = Path(workdir)
    bus = bus or Bus(workdir / "bus.jsonl", agent="spec")
    ceiling = SPEC_TASK_CEILING_USD if ceiling_usd is None else ceiling_usd
    counts: dict = {"written": 0, "skipped_existing": 0, "failed": 0, "spent_usd": 0.0, "failures": {}}
    for task_id in task_ids:
        if written_already(workdir, task_id):
            counts["skipped_existing"] += 1
            continue
        try:
            counts["spent_usd"] += write_one(workdir, task_id, model, ceiling, bus)
            counts["written"] += 1
        except Exception as error:  # one Task's failure is recorded; the others are still written
            counts["failed"] += 1
            counts["failures"][task_id] = type(error).__name__
    return counts


__all__ = ["SPEC_TASK_CEILING_USD", "write_one", "write_specs", "written_already"]
