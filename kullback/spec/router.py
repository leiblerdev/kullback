"""The set-aside rule of the review loop (DESIGN.md rule 4, D331).

The bus-driven router that moved each ruling as it was filed (a re-roll, a writer repair, a Builder
note) is retired: rulings are feedback the writer answers in its own round (spec/rounds.py). What
stays is the one source of the round cap and the set-aside rule, which the trust gates read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import read_json
from kullback.spec import events
from kullback.spec.schema import Spec

# Writer rounds per Task: the first write and two answers to rulings (D350).
ROUNDS_CAP = 3
STATE_FILE = "state/router.json"
NO_AGREEMENT = events.NO_AGREEMENT


def state_path(workdir: Any) -> Path:
    return Path(workdir) / "spec" / STATE_FILE


def load_state(workdir: Any) -> dict:
    body = read_json(state_path(workdir), {}) or {}
    return {"last_seq": int(body.get("last_seq") or 0), "rounds": dict(body.get("rounds") or {}),
            "queue": list(body.get("queue") or []), "actions": list(body.get("actions") or [])}


def writer_rounds(rulings: Iterable[Any]) -> int:
    """The writer rounds a Task had: its first write, then each round in which it answered a ruling."""
    answered = {int(r.answer["round"]) for r in rulings if isinstance(getattr(r, "answer", None), dict)
                and r.answer.get("round") is not None}
    return 1 + len(answered)


# Why a Task ends with no Verifier, never a check bent to pass: the Examiner found the Reference wrong,
# or the turns never show what the user wants and no answer settled it. Recorded on the Spec's set_aside.
REFERENCE_WRONG, INTENT_UNCLEAR = "reference_wrong", "intent_unclear"
NO_VERIFIER_WHYS = (REFERENCE_WRONG, INTENT_UNCLEAR)


def no_verifier_why(spec: Spec, rulings: Iterable[Any] = ()) -> Optional[str]:
    """Why no Verifier can be made for the Task, else None: marked so on the Spec, or a not_enough_truth
    ruling still open when the router sets the Task aside at the cap."""
    if spec.set_aside in NO_VERIFIER_WHYS:
        return spec.set_aside
    return INTENT_UNCLEAR if spec.truth_open and is_set_aside(spec, rulings) else None


def is_set_aside(spec: Spec, rulings: Iterable[Any] = ()) -> bool:
    """Is the Task set aside? Marked so, or at ROUNDS_CAP writer rounds with a ruling still open.

    The writer rounds are read off the Task's rulings when given (spec/rulings.py records); else the
    Spec's round stands in, which the review loop keeps equal to them (Examiner round n follows writer
    round n). sp-trust reads it."""
    if spec.set_aside is not None:
        return True
    records = [r for r in rulings if hasattr(r, "answer")]
    done = writer_rounds(records) if records else spec.round
    return done >= ROUNDS_CAP and spec.rulings_open > 0


__all__ = ["INTENT_UNCLEAR", "NO_AGREEMENT", "NO_VERIFIER_WHYS", "REFERENCE_WRONG", "ROUNDS_CAP", "STATE_FILE",
           "is_set_aside", "load_state", "no_verifier_why", "state_path", "writer_rounds"]
