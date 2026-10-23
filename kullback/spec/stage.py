"""The Spec stage of a build: each named Task's Intent mined, its Spec and Verifier written, the bus told.

A Task whose Spec and Spec Verifier are both on disk is skipped, so a resumed build writes only
what is missing; a Spec of the older demand grammar (no item) is rewritten as items, Intent and all,
so every Task gets the sanity item (automatic, no flag). An empty Spec is written once more; empty again, the Task is set aside with no
Verifier. A Spec no Run can pass after its one send-back is set aside the same way, as
unsatisfiable_spec. A Spec of a born Task leaving an end fact of its scenario unchecked (writer.unchecked_end,
D414) is set aside so as unchecked_end. A born Task's Spec goes back to the writer once where a write its end
names has no gate differing from the start, or a value the end names of the changed row is in no gate
(spec/ungated.py, D419); still found, it is marked (writes_ungated, end_values_ungated) and stands. A Verifier
whose code side holds on the Run that does nothing while its end names a write is marked held_by_idle (the trust
gate refuses it, spec/trust.py). The stage counts each mark. The miner also says whether the recorded user held to their opening asks, checked by
code (spec/coherence.py); a Task whose user strayed is set aside as intent_strayed before the writer, no
Verifier written. A set-aside Spec counts as done. Every call is priced under the `spec` stage of the workdir ledger (budget.json):
the miner through the harness subscriber, the writer through BudgetedModel, the two seams every
other stage uses. A Task that fails is counted with its error class and message, its Spec file put back as it was
before the attempt (so a rerun writes it again), and the loop goes on; the first failure message stands once
in the counts as first_failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.agent.bus import Bus
from kullback.runner.budget import BudgetedModel
from kullback.spec import events
from kullback.spec.events import UNCHECKED_END
from kullback.spec.intent import STAGE, mine_intent
from kullback.spec.schema import SpecIntent, is_empty, load_spec, save_spec, spec_path
from kullback.spec.trust import HELD_BY_IDLE, held_by_idle, idle_run
from kullback.spec.ungated import END_VALUES_UNGATED, WRITES_UNGATED, send_back
from kullback.spec.writer import (
    EFFECTS_COUNT,
    NO_EFFECTS,
    is_item_check,
    load_inputs,
    spec_verifier_path,
    verifier_of,
    write_spec,
    write_verifier,
)

# What one Task's mining and writing may spend together when the caller names no ceiling.
SPEC_TASK_CEILING_USD = 1.0
# How much of a failed Task's error message its failure entry keeps.
FAILURE_CHARS = 200


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
                      bus: Bus, coherence: Optional[dict] = None) -> float:
    """Write and compile one Task's Spec from a given Intent under `intent`; return the spend.

    An empty first Spec is written once more in a fresh session. Empty again, the Spec is saved set aside
    as empty_spec, task.set_aside is published and no Verifier is written; a Spec no Run can pass is set
    aside so as unsatisfiable_spec; else spec.written is published. `coherence` is the miner stage's
    record (spec/coherence.py), kept on the Spec.
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
    if not empty:
        written, again = send_back(written, inputs, priced, model_id=model_id,
                                   ceiling_usd=max(ceiling_usd - spent, 0.0))
        spent += again
    rounds = sum(int(t.session.get("tool_rounds") or 0) for t in tries)
    writer = dict(written.counts, usd=spent, read_rounds=rounds, attempts=len(tries), empty_first=empty_first,
                  empty=empty)
    if len(tries) > 1:
        writer["first_attempt"] = dict(tries[0].counts, usd=float(tries[0].session.get("usd") or 0.0))
    spec = written.spec.model_copy(update={"version": _version(workdir, task_id) + 1, "writer": writer,
                                           "coherence": coherence or {}})
    aside = events.EMPTY_SPEC if empty else events.UNSATISFIABLE_SPEC if written.counts.get("unsatisfiable") else \
        UNCHECKED_END if written.counts.get(UNCHECKED_END) else None
    if aside:
        save_spec(workdir, spec.model_copy(update={"set_aside": aside}))
        events.task_set_aside(bus, task_id, aside, 0)
        return spent
    write_verifier(_idle_marked(spec, inputs), workdir, inputs)
    events.spec_written(bus, task_id, spec.version)
    return spent


def _idle_marked(spec: Any, inputs: Any) -> Any:
    """The Spec marked held_by_idle when its Verifier's code side holds on the Run that does nothing (D419)."""
    verifier = verifier_of(spec, inputs).verifier
    if not held_by_idle(spec, verifier, idle_run(spec.task_id, inputs.state), inputs.canon, inputs.write_tools):
        return spec
    return spec.model_copy(update={"writer": dict(spec.writer, **{HELD_BY_IDLE: True})})


def write_one(workdir: Path, task_id: str, model: Any, ceiling_usd: float, bus: Bus) -> float:
    """Mine one Task's Intent with its coherence record, then write its Spec from it (`write_from_intent`);
    return the spend. A Task whose user strayed from an opening ask (the miner's outcomes, checked by code)
    is saved set aside as intent_strayed with the record, task.set_aside published, no writer."""
    mined = mine_intent(workdir, task_id, model, ceiling_usd=ceiling_usd)
    coherence, spent = mined.coherence, mined.spent_usd
    if coherence.get("strayed"):
        spec = load_spec(workdir, task_id)
        save_spec(workdir, spec.model_copy(update={"coherence": coherence, "set_aside": events.INTENT_STRAYED}))
        events.task_set_aside(bus, task_id, events.INTENT_STRAYED, 0)
        return spent
    rest = max(ceiling_usd - spent, 0.0)
    return spent + write_from_intent(workdir, task_id, mined.intent, model, rest, bus, coherence=coherence)


def failure_line(error: Exception) -> str:
    """A failed Task's entry: the error class and the front of its message."""
    return f"{type(error).__name__}: {error}"[:FAILURE_CHARS]


def _restore(path: Path, before: Optional[bytes]) -> None:
    """Put the Spec file back as it was before a failed attempt, so a rerun writes the Task again (D421)."""
    if before is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(before)


def write_specs(workdir: Any, task_ids: Iterable[str], model: Any, *,
                ceiling_usd: Optional[float] = None, bus: Optional[Bus] = None) -> dict:
    """Write the Spec of every named Task that has none; counts of written, skipped, failed, set aside, and the spend."""
    workdir = Path(workdir)
    bus = bus or Bus(workdir / "bus.jsonl", agent="spec")
    ceiling = SPEC_TASK_CEILING_USD if ceiling_usd is None else ceiling_usd
    counts: dict = {"written": 0, "skipped_existing": 0, "failed": 0, "spent_usd": 0.0, "failures": {},
                    "empty_seen": 0, "retried": 0, "set_aside_empty": 0, "set_aside_unsatisfiable": 0,
                    "set_aside_strayed": 0, "set_aside_unchecked_end": 0, "rewritten_older": 0,
                    "effects_none_learned": 0, WRITES_UNGATED: 0, END_VALUES_UNGATED: 0, HELD_BY_IDLE: 0}
    for task_id in task_ids:
        if written_already(workdir, task_id):
            counts["skipped_existing"] += 1
            continue
        counts["rewritten_older"] += int(older_grammar(load_spec(workdir, task_id)))
        path = spec_path(workdir, task_id)
        before = path.read_bytes() if path.is_file() else None
        try:
            counts["spent_usd"] += write_one(workdir, task_id, model, ceiling, bus)
        except Exception as error:  # one Task's failure is recorded; the others are still written
            _restore(path, before)
            counts["failed"] += 1
            counts["failures"][task_id] = failure_line(error)
            counts.setdefault("first_failure", counts["failures"][task_id])
            continue
        spec = load_spec(workdir, task_id)
        counts["empty_seen"] += int(bool(spec.writer.get("empty_first")))
        counts["retried"] += int(spec.writer.get("attempts", 1) > 1)
        counts["effects_none_learned"] += int(spec.writer.get(EFFECTS_COUNT) == NO_EFFECTS)
        for mark in (WRITES_UNGATED, END_VALUES_UNGATED, HELD_BY_IDLE):
            counts[mark] += int(bool(spec.writer.get(mark)))
        aside = {events.EMPTY_SPEC: "set_aside_empty", events.UNSATISFIABLE_SPEC: "set_aside_unsatisfiable",
                 events.INTENT_STRAYED: "set_aside_strayed", UNCHECKED_END: "set_aside_unchecked_end"}
        counts[aside.get(spec.set_aside, "written")] += 1
    return counts


__all__ = ["FAILURE_CHARS", "SPEC_TASK_CEILING_USD", "failure_line", "older_grammar", "write_from_intent", "write_one",
           "write_specs", "written_already"]
