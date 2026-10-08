"""The set-aside rule of the review loop (DESIGN.md rule 4, D331).

The bus-driven router that moved each ruling as it was filed (a re-roll, a writer repair, a Builder
note) is retired: rulings are feedback the writer answers in its own round (spec/rounds.py). What
stays is the one source of the round cap and the set-aside rule, which the trust gates read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kullback.runner.records import read_json
from kullback.spec import events
from kullback.spec.schema import Spec

ROUNDS_CAP = 2
STATE_FILE = "state/router.json"
NO_AGREEMENT = events.NO_AGREEMENT


def state_path(workdir: Any) -> Path:
    return Path(workdir) / "spec" / STATE_FILE


def load_state(workdir: Any) -> dict:
    body = read_json(state_path(workdir), {}) or {}
    return {"last_seq": int(body.get("last_seq") or 0), "rounds": dict(body.get("rounds") or {}),
            "queue": list(body.get("queue") or []), "actions": list(body.get("actions") or [])}


def is_set_aside(spec: Spec) -> bool:
    """Is the Task set aside? Marked so, or at ROUNDS_CAP rounds with a ruling still open. sp-trust reads it."""
    return spec.set_aside is not None or (spec.round >= ROUNDS_CAP and spec.rulings_open > 0)


__all__ = ["NO_AGREEMENT", "ROUNDS_CAP", "STATE_FILE", "is_set_aside", "load_state", "state_path"]
