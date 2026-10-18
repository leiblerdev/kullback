"""The set-aside rule of the review loop (DESIGN.md rule 4, D331).

The bus-driven router that moved each ruling as it was filed (a re-roll, a writer repair, a Builder
note) is retired: rulings are feedback the writer answers in its own round (spec/rounds.py). What
stays is the one source of the round cap and the set-aside rule, which the trust gates read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import read_json
from kullback.spec import events
from kullback.spec.rulings import answer_rounds
from kullback.spec.schema import Spec

# Writer rounds per Task: the first write and four answers to rulings (D350, D359), unless the
# environment variable ROUNDS_CAP_ENV sets another cap of at least one (D359).
ROUNDS_CAP = 5
ROUNDS_CAP_ENV = "KULLBACK_ROUNDS_CAP"
STATE_FILE = "state/router.json"
NO_AGREEMENT = events.NO_AGREEMENT


def state_path(workdir: Any) -> Path:
    return Path(workdir) / "spec" / STATE_FILE


def load_state(workdir: Any) -> dict:
    body = read_json(state_path(workdir), {}) or {}
    return {"last_seq": int(body.get("last_seq") or 0), "rounds": dict(body.get("rounds") or {}),
            "queue": list(body.get("queue") or []), "actions": list(body.get("actions") or [])}


def rounds_cap() -> int:
    """The writer-round cap in force: ROUNDS_CAP_ENV when set, else ROUNDS_CAP; a value that is not a
    whole number of at least one is refused, never read as some other cap."""
    raw = os.environ.get(ROUNDS_CAP_ENV, "").strip()
    if not raw:
        return ROUNDS_CAP
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{ROUNDS_CAP_ENV} must be a whole number of at least 1, got {raw!r}")
    return int(raw)


def capped_rounds(rounds: Optional[int] = None) -> int:
    """The Examiner rounds a review loop runs: `rounds` when given, never past the cap in force."""
    cap = rounds_cap()
    return min(int(rounds or cap), cap)


def writer_rounds(rulings: Iterable[Any]) -> int:
    """The writer rounds a Task had: its first write, then each round in which it answered a ruling,
    counting every round a ruling kept open was answered in (`rulings.answer_rounds`)."""
    return 1 + len({n for r in rulings for n in answer_rounds(r)})


# Why a Task ends with no Verifier, never a check bent to pass: the turns never show what the user wants and
# no answer settled it (intent_unclear), the recorded user strayed from an opening ask (intent_strayed,
# spec/coherence.py), or a born Task's Spec left an end fact of its scenario unchecked (unchecked_end, D414).
# Recorded on the Spec's set_aside. A wrong Reference is no reason (D367): the Verifier comes from the ask, the
# policy and the world, and a Reference failing it is a finding.
INTENT_UNCLEAR = "intent_unclear"
INTENT_STRAYED = events.INTENT_STRAYED
UNCHECKED_END = events.UNCHECKED_END
NO_VERIFIER_WHYS = (INTENT_UNCLEAR, INTENT_STRAYED, UNCHECKED_END)


def no_verifier_why(spec: Spec, rulings: Iterable[Any] = ()) -> Optional[str]:
    """Why no Verifier can be made for the Task, else None: marked so on the Spec, or a not_enough_truth
    ruling still open when the router sets the Task aside at the cap."""
    if spec.set_aside in NO_VERIFIER_WHYS:
        return spec.set_aside
    return INTENT_UNCLEAR if spec.truth_open and is_set_aside(spec, rulings) else None


def is_set_aside(spec: Spec, rulings: Iterable[Any] = ()) -> bool:
    """Is the Task set aside? Marked so, or at the round cap (`rounds_cap`) with a ruling still open.

    The writer rounds are read off the Task's rulings when given (spec/rulings.py records); else the
    Spec's round stands in, which the review loop keeps equal to them (Examiner round n follows writer
    round n). sp-trust reads it."""
    if spec.set_aside is not None:
        return True
    records = [r for r in rulings if hasattr(r, "answer")]
    done = writer_rounds(records) if records else spec.round
    return done >= rounds_cap() and spec.rulings_open > 0


__all__ = ["INTENT_STRAYED", "INTENT_UNCLEAR", "NO_AGREEMENT", "NO_VERIFIER_WHYS", "ROUNDS_CAP", "ROUNDS_CAP_ENV",
           "UNCHECKED_END",
           "STATE_FILE", "capped_rounds", "is_set_aside", "load_state", "no_verifier_why", "rounds_cap", "state_path", "writer_rounds"]
