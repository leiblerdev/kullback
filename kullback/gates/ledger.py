"""gates.json under one lock: the one ledger both agents record their rulings through (D122, D128).

The class moved here verbatim from `builder/pipeline.py` in phase 5 so the Builder's stages and the
Examiner's tools write the file through one class with one lock and the same replace-and-append
rule; `builder/pipeline.py` re-imports it under the same name. Turn-taking (D128) makes one writer
at a time, and the lock is what keeps a beat's own threads honest.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from kullback.runner.records import GateResult, as_dict

HISTORY_NAME = "gates_by_round.json"


class GateLedger:
    """gates.json under one lock, with every write remembered per stage.

    A stage records a ruling by dropping the rows of the same stage name and appending (report.py
    reads the file), or overwrites the file with a list of its own. Two stages on two threads would
    race for the file, so each write goes through here, and when stages ran side by side the writes
    are replayed in stage order at the end, so the file reads the same as a one-worker build wrote it.

    `snapshot` keeps what gates.json held at the end of one round in `gates_by_round.json`; the
    round driver calls it, and gates.json itself is untouched by it.
    """

    def __init__(self, workdir: Path):
        self.path = Path(workdir) / "gates.json"
        self.history = Path(workdir) / HISTORY_NAME
        self.lock = threading.Lock()
        self.ops: dict[str, list[tuple[str, list]]] = {}
        self.initial: list = []

    def begin(self) -> None:
        self.ops = {}
        self.initial = self._read()

    def _read(self) -> list:
        if not self.path.is_file():
            return []
        return json.loads(self.path.read_text(encoding="utf-8")) or []

    def _write(self, body: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def _apply(body: list, op: str, rows: list) -> list:
        if op == "write":
            return list(rows)
        for row in rows:
            body = [g for g in body if g.get("stage") != row.get("stage")] + [row]
        return body

    def record(self, stage_name: str, result: GateResult) -> GateResult:
        """Append one ruling, replacing any earlier ruling of the same stage name."""
        with self.lock:
            rows = [as_dict(result)]
            self._write(self._apply(self._read(), "record", rows))
            self.ops.setdefault(stage_name, []).append(("record", rows))
        return result

    def write(self, stage_name: str, results: Iterable[GateResult]) -> None:
        """Overwrite the file with these rulings (the compile_tools stage's per-tool sandbox gates)."""
        with self.lock:
            rows = [as_dict(r) for r in results]
            self._write(rows)
            self.ops.setdefault(stage_name, []).append(("write", rows))

    def replay(self, order: Iterable[str]) -> None:
        """Land the writes in stage order, from what the file held when the run began."""
        with self.lock:
            body = list(self.initial)
            for name in order:
                for op, rows in self.ops.get(name, []):
                    body = self._apply(body, op, rows)
            if any(self.ops.values()):
                self._write(body)

    def snapshot(self, round_no: int) -> list:
        """This round's rulings kept in gates_by_round.json, so an earlier round can be read again.

        gates.json holds one ruling per stage, the last one, which is the state the report reads and
        which nothing here changes. A round that repairs an artifact rules again under the same
        stage names, so without this file a ruling that went from red to green leaves no trace of
        ever having been red, and no repair can be said to have moved a gate. One row per round,
        holding gates.json as it stood when the round ended; a round recorded twice replaces its row.
        """
        with self.lock:
            rulings = self._read()
            rows = [row for row in self._read_history() if row.get("round") != round_no]
            rows.append({"round": round_no, "rulings": rulings})
            rows.sort(key=lambda row: int(row.get("round") or 0))
            self.history.parent.mkdir(parents=True, exist_ok=True)
            self.history.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str),
                                    encoding="utf-8")
        return rulings

    def _read_history(self) -> list:
        if not self.history.is_file():
            return []
        try:
            body = json.loads(self.history.read_text(encoding="utf-8"))
        except ValueError:
            return []
        return [row for row in body if isinstance(row, dict)] if isinstance(body, list) else []

    def rulings(self, stage_name: str) -> list[str]:
        """The distinct ruling names this stage recorded, in order."""
        out: list[str] = []
        for _, rows in self.ops.get(stage_name, []):
            out += [row["stage"] for row in rows if row.get("stage") not in out]
        return out
