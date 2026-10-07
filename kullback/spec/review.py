"""The writer's round of the review loop: each open ruling applied or rebutted (D331).

The Examiner files rulings and edits nothing (examiner/rule_tool.py). Here the Spec writer answers
each ruling it has not answered yet, one repair session per ruling (spec/writer.py repair): it applies
the fix (changes or drops the ruled check, adds a check for an uncovered fact) or rebuts it with why.
An applied answer rewrites the Verifier; the answer is kept on the ruling for the Examiner's next round,
which closes the ruling or keeps it open.

A ruling that says the Reference is wrong (fails_reference, wrong side reference) asks nothing of the
writer: it is a note for the witness flag and is left for the Examiner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from kullback.runner.records import Verifier
from kullback.spec import events
from kullback.spec import rulings as R
from kullback.spec.items import SANITY_ID
from kullback.spec.schema import Spec, load_spec

REBUT_NONE = "the writer made no grounded change and gave no reason"


def atom_checks(spec: Spec, verifier: Optional[Verifier]) -> dict[str, str]:
    """Each atom id of the Verifier with the check that made it: an item's atom is its check's id, the
    sanity item is "code"; compile names an older check's atoms i<n>, n its index among the older checks;
    "unknown" for an atom no check made."""
    from kullback.spec.writer import ATOM_DEMANDS, is_item_check

    if verifier is None:
        return {}
    items = {check.id for check in spec.checks if is_item_check(check)}
    checks = [check for check in spec.checks if not is_item_check(check)]
    if verifier.expected and not items:
        checks = [check for check in checks if check.demand.get("demand") in ATOM_DEMANDS]
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
                  fix=ruling.fix, kind=f"{ruling.kind}/{ruling.code}")


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
    from kullback.spec.writer import load_inputs, write_verifier
    from kullback.spec.writer import repair as writer_repair

    root = Path(workdir)
    if repair is None:
        model_id = getattr(model, "name", None)
        model = BudgetedModel(model, stage=STAGE, workdir=root, model_id=model_id)

        def repair(spec: Spec, ruled: Any, inputs: Any, m: Any) -> Any:
            return writer_repair(spec, ruled, inputs, m, model_id=model_id, ceiling_usd=ceiling_usd)

    out = {"task_id": task_id, "answered": 0, "applied": 0, "rebutted": 0, "skipped": 0}
    todo = [r for r in R.unanswered(R.load_rulings(root, task_id)) if r.wrong_side != "reference"]
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
        written = repair(spec, _writer_ruling(spec, load_spec_verifier(root, task_id), ruling), inputs, model)
        counts = written.counts
        changed = [c.id for c in written.spec.checks if c not in spec.checks]
        changed += [c.id for c in spec.checks if c.id not in {k.id for k in written.spec.checks}]
        if counts.get("changed", 0) + counts.get("dropped", 0) + counts.get("added", 0):
            write_verifier(written.spec.model_copy(update={"rulings_open": spec.rulings_open, "round": spec.round}),
                           root, inputs)
            action, why = "applied", counts.get("rebut") or ruling.fix
        else:
            action, why = "rebutted", counts.get("rebut") or REBUT_NONE
        kept = {k: v for k, v in counts.items() if k != "rebut" and v}
        R.answer(root, task_id, ruling.number, action, why, round_number, changed, writer=kept)
        events.ruling_answered(bus, task_id, ruling.number, action, changed)
        out["answered"] += 1
        out[action] += 1
    R.sync_spec(root, task_id)
    return out


__all__ = ["REBUT_NONE", "answer_rulings", "atom_checks", "ruled_checks"]
