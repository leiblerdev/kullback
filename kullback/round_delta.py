"""One line saying what the round before moved, read off rounds.json.

Both agents need it and neither package may import the other: the import contract puts the Builder
and the Examiner side by side in one layer, and the round driver above them both. So the reader they
share sits here, under `kullback` and below both, and reads the file rather than the driver.

One live build's Examiner filed fewer findings every round while trusted went 99, 95, 66: each round
opened on the same picture of the artifacts as they stand and nothing anywhere said the last round
had made things worse. This is that sentence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROUNDS_NAME = "rounds.json"
# What a round is judged by, in the words its own counts use: Tasks whose Verdict is trusted, the
# recorded calls that replay, and the Tasks that have a Reference to judge against.
DELTA_FIELDS = (("trusted", "trusted"), ("fidelity", "fidelity"), ("tasks_with_reference", "References"))


def round_records(workdir: Any) -> list[dict]:
    """Every recorded round as it was written, in order; [] when there is no readable file."""
    path = Path(workdir) / ROUNDS_NAME
    if not path.is_file():
        return []
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [row for row in body if isinstance(row, dict)] if isinstance(body, list) else []


def _moved(now: Any, then: Any) -> str:
    """How one number moved, in words; "" when either side is not a number."""
    if isinstance(now, bool) or isinstance(then, bool):
        return ""
    if not isinstance(now, (int, float)) or not isinstance(then, (int, float)):
        return ""
    move = now - then
    if move == 0:
        return f"{now}, unchanged"
    return f"{now}, {'up' if move > 0 else 'down'} {abs(move)}"


def delta_line(workdir: Any) -> str:
    """What the last closed round moved against the one before it; "" until two rounds have closed.

    The artifacts it changed are named beside the numbers, because a round that moved nothing and a
    round that rewrote every body and lost forty Tasks read the same without them.
    """
    rounds = round_records(workdir)
    if len(rounds) < 2:
        return ""
    last, before = rounds[-1], rounds[-2]
    now, then = last.get("counts") or {}, before.get("counts") or {}
    parts = [f"{name} {moved}" for key, name in DELTA_FIELDS
             if (moved := _moved(now.get(key), then.get(key)))]
    if not parts:
        return ""
    changed = [str(name) for name in (now.get("artifacts_changed") or []) if isinstance(name, str)]
    tail = ", ".join(changed) if changed else "nothing"
    return (f"round {last.get('round')} against round {before.get('round')}: "
            + "; ".join(parts) + f"; it changed {tail}")
