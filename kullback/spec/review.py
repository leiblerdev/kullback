"""The writer's round of the review loop: each open ruling applied or rebutted (D331).

The Examiner files rulings and edits nothing (examiner/rule_tool.py). Here the Spec writer answers
each ruling it has not answered yet, one repair session per ruling (spec/writer.py repair): it applies
the fix (changes or drops the ruled check, adds a check for an uncovered fact) or rebuts it with why.
An applied answer rewrites the Verifier; the answer is kept on the ruling for the Examiner's next round,
which closes the ruling or keeps it open.

An intent_unfounded ruling the writer applies marks its fact `unfounded` on the Spec: the writer's tool
then refuses any gate resting on it alone (spec/items.py). A not_enough_truth ruling cannot be applied:
whatever the repair changed is not kept, and the answer is the writer's rebuttal, or none.

A fails_reference ruling filed before D367 asks nothing of the writer: the Reference never repairs a Verifier.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from kullback.runner.records import Verifier
from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.items import SANITY_ID
from kullback.spec.router import no_verifier_why
from kullback.spec.schema import Spec, load_spec

REBUT_NONE = "the writer made no grounded change and gave no reason"
# Spec fields the review loop and the Spec stage own, never the writer's repair: a repaired Spec keeps
# them as they stood, so an examined Task stays examined (D343).
REVIEW_FIELDS = ("rulings_open", "truth_open", "round", "examined_rounds", "pending_repair", "set_aside")


def atom_checks(spec: Spec, verifier: Optional[Verifier]) -> dict[str, str]:
    """Each atom id of the Verifier with the check that made it: an item's atom is its check's id, the
    sanity item is "code"; compile names an older check's atoms i<n>, n its index among the older checks;
    "unknown" for an atom no check made."""
    from kullback.spec.writer import is_item_check

    if verifier is None:
        return {}
    items = {check.id for check in spec.checks if is_item_check(check)}
    checks = [check for check in spec.checks if not is_item_check(check)]
    out: dict[str, str] = {}
    for atom in verifier.atoms:
        if atom.id in items or atom.id == SANITY_ID:
            out[atom.id] = atom.id if atom.id in items else "code"
            continue
        number = re.match(r"i(\d+)", atom.id)
        index = int(number.group(1)) if number else -1
        out[atom.id] = checks[index].id if 0 <= index < len(checks) else "unknown"
    return out


def ruled_checks(spec: Spec, verifier: Optional[Verifier], item: str) -> list[str]:
    """The checks a ruling on `item` lets the writer change: the check itself, the check that made the
    atom, or the checks citing the fact; none for the task as a whole, where the writer may only add."""
    ids = {check.id for check in spec.checks}
    if item in ids:
        return [item]
    made = atom_checks(spec, verifier).get(item)
    if made in ids:
        return [made]
    return [check.id for check in spec.checks if item in check.fact_ids]


def _writer_ruling(spec: Spec, verifier: Optional[Verifier], ruling: R.Ruling) -> Any:
    from kullback.spec.writer import Ruling

    return Ruling(check_ids=ruled_checks(spec, verifier, ruling.item), reason=ruling.reason, item=ruling.item,
                  fix=ruling.fix, kind=f"{ruling.kind}/{ruling.code}", excerpt=_kept_open_note(ruling))


def _kept_open_note(ruling: R.Ruling) -> str:
    """For a ruling the Examiner kept open after the writer's answer: that answer and the Examiner's why."""
    if not R.kept_open_after_answer(ruling):
        return ""
    return (f"Your last answer ({ruling.answer.get('action')}): {ruling.answer.get('why') or 'none'}. "
            f"The Examiner kept the ruling open: {ruling.closed.get('why') or 'no reason given'}.")


def _examiners_side(ruling: R.Ruling) -> bool:
    """A fails_reference ruling filed before D367 asks nothing of the writer: no Reference repairs a Verifier."""
    return ruling.code == "fails_reference"


def unfounded_intent(intent: Any, fact_id: str) -> Any:
    """The Intent with one fact marked unfounded: the Examiner ruled the user never asked for it."""
    facts = [f.model_copy(update={"unfounded": True}) if f.id == fact_id else f for f in intent.facts]
    return intent.model_copy(update={"facts": facts})


def _inputs_for(inputs: Any, ruling: R.Ruling) -> Any:
    """The writer's inputs for one ruling: an intent_unfounded ruling's fact marked unfounded."""
    if ruling.code != R.UNFOUNDED_CODE:
        return inputs
    return inputs._replace(intent=unfounded_intent(inputs.intent, ruling.item))


def _settle(root: Path, ruling: R.Ruling, spec: Spec, written: Any, used: Any) -> tuple[str, str, list[str]]:
    """The answer to one ruling from the repair: applied when it changed, dropped or added a check (the
    Verifier rewritten, an unfounded fact kept marked), else rebutted; a not_enough_truth ruling is never
    applied, whatever the repair did."""
    from kullback.spec.writer import write_verifier

    counts = written.counts
    moved = counts.get("changed", 0) + counts.get("dropped", 0) + counts.get("added", 0)
    if ruling.code == R.NO_TRUTH_CODE or not moved:
        return "rebutted", counts.get("rebut") or REBUT_NONE, []
    changed = [c.id for c in written.spec.checks if c not in spec.checks]
    changed += [c.id for c in spec.checks if c.id not in {k.id for k in written.spec.checks}]
    update = {name: getattr(spec, name) for name in REVIEW_FIELDS}
    if ruling.code == R.UNFOUNDED_CODE:
        update["intent"] = unfounded_intent(written.spec.intent, ruling.item)
    write_verifier(written.spec.model_copy(update=update), root, used)
    return "applied", counts.get("rebut") or ruling.fix, changed


def answer_rulings(workdir: Any, task_id: str, model: Any, round_number: int, *,
                   ceiling_usd: Optional[float] = None, repair: Optional[Callable] = None) -> dict:
    """The writer's answer to every open ruling on one Task it has not answered; returns the counts.

    `repair(spec, ruling, inputs, model)` is the writer's repair (spec/writer.py), priced under the Spec's
    stage of the ledger, unless a caller gives another; it returns the repaired Spec and its counts. Applied when it changed, dropped or added a
    check, the Verifier rewritten; rebutted otherwise, with the writer's why.
    """
    from kullback.agent.bus import Bus
    from kullback.runner.budget import BudgetedModel
    from kullback.spec.stage import STAGE
    from kullback.spec.trust import load_spec_verifier
    from kullback.spec.writer import load_inputs
    from kullback.spec.writer import repair as writer_repair

    root = Path(workdir)
    if repair is None:
        model_id = getattr(model, "name", None)
        model = BudgetedModel(model, stage=STAGE, workdir=root, model_id=model_id)

        def repair(spec: Spec, ruled: Any, inputs: Any, m: Any) -> Any:
            return writer_repair(spec, ruled, inputs, m, model_id=model_id, ceiling_usd=ceiling_usd)

    out = {"task_id": task_id, "answered": 0, "applied": 0, "rebutted": 0, "skipped": 0}
    spec = load_spec(root, task_id)
    # No Verifier can be made for the Task: nothing asks the writer to change a check to pass the Reference.
    ended = spec is not None and no_verifier_why(spec) is not None
    todo = [r for r in R.unanswered(R.load_rulings(root, task_id)) if not ended and not _examiners_side(r)]
    out["skipped"] = len(R.unanswered(R.load_rulings(root, task_id))) - len(todo)
    if not todo:
        return out
    bus = Bus(root / "bus.jsonl", agent="spec")
    inputs = None
    for ruling in todo:
        spec = load_spec(root, task_id)
        if spec is None:
            break
        inputs = inputs or load_inputs(root, task_id, spec.intent)
        used = _inputs_for(inputs, ruling)
        written = repair(spec, _writer_ruling(spec, load_spec_verifier(root, task_id), ruling), used, model)
        action, why, changed = _settle(root, ruling, spec, written, used)
        inputs = used if action == "applied" else inputs
        kept = {k: v for k, v in written.counts.items() if k != "rebut" and v}
        R.answer(root, task_id, ruling.number, action, why, round_number, changed, writer=kept)
        events.ruling_answered(bus, task_id, ruling.number, action, changed)
        out["answered"] += 1
        out[action] += 1
    R.sync_spec(root, task_id)
    return out


__all__ = ["REBUT_NONE", "REVIEW_FIELDS", "answer_rulings", "atom_checks", "ruled_checks", "unfounded_intent"]
