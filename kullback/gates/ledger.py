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
from typing import Iterable, Optional

from kullback.runner.records import GateResult, as_dict

HISTORY_NAME = "gates_by_round.json"
# Where the per tool compile rulings go (D218 rule 2). They used to overwrite gates.json, which left
# the file holding one stage's per tool rows and none of the round's own rulings the moment a
# recompile ran, so a reader who opened it mid round found neither the last closed round's answer nor
# this round's. They are a snapshot of one stage over one tool set, so they get a file that says so,
# with the round and the tool on every row.
COMPILE_NAME = "compile_snapshot.json"
COMPILE_FORMAT = 1


class GateLedger:
    """gates.json under one lock, with every write remembered per stage.

    A stage records a ruling by dropping the rows of the same stage name and appending (report.py
    reads the file); nothing overwrites it, because a stage that replaced the file with a list of its
    own left the round's other rulings nowhere (D218 rule 2), so a per tool set goes to
    `write_compile_snapshot` and its own file instead. Two stages on two threads would race for the
    file, so each write goes through here, and when stages ran side by side the writes are replayed
    in stage order at the end, so the file reads the same as a one-worker build wrote it.

    `snapshot` keeps what gates.json held at the end of one round in `gates_by_round.json`; the
    round driver calls it, and gates.json itself is untouched by it.
    """

    def __init__(self, workdir: Path):
        self.path = Path(workdir) / "gates.json"
        self.history = Path(workdir) / HISTORY_NAME
        self.compile = Path(workdir) / COMPILE_NAME
        self.lock = threading.Lock()
        self.ops: dict[str, list[list]] = {}
        self.snapshot_names: dict[str, list[str]] = {}
        self.initial: list = []

    def begin(self) -> None:
        self.ops = {}
        self.snapshot_names = {}
        self.initial = self._read()

    def _read(self) -> list:
        if not self.path.is_file():
            return []
        return json.loads(self.path.read_text(encoding="utf-8")) or []

    def _write(self, body: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def _apply(body: list, rows: list) -> list:
        for row in rows:
            body = [g for g in body if g.get("stage") != row.get("stage")] + [row]
        return body

    def record(self, stage_name: str, result: GateResult) -> GateResult:
        """Append one ruling, replacing any earlier ruling of the same stage name."""
        with self.lock:
            rows = [as_dict(result)]
            self._write(self._apply(self._read(), rows))
            self.ops.setdefault(stage_name, []).append(rows)
        return result

    def write_compile_snapshot(self, stage_name: str, rows: Iterable[dict], round_no: int) -> list[dict]:
        """The per tool rulings of one compile, in their own file with the round and tool on each row.

        This is a snapshot of one stage over the tool set it just released, not the state of the
        workdir's gates, and writing it into gates.json cost a reader both: the round's own rulings
        were gone and the rows that replaced them said nothing about which round they came from. A
        row carries the tool it was measured on, so a red light can name the tool without a reader
        having to know the order the stage happened to record them in.
        """
        body = {"format": COMPILE_FORMAT, "round": int(round_no), "stage": str(stage_name),
                "rows": [dict(row) for row in rows]}
        with self.lock:
            self.compile.parent.mkdir(parents=True, exist_ok=True)
            self.compile.write_text(json.dumps(body, indent=2, sort_keys=True, default=str),
                                    encoding="utf-8")
            named = self.snapshot_names.setdefault(stage_name, [])
            named += [str(row.get("stage") or "") for row in body["rows"] if row.get("stage") not in named]
        return body["rows"]

    def replay(self, order: Iterable[str]) -> None:
        """Land the writes in stage order, from what the file held when the run began."""
        with self.lock:
            body = list(self.initial)
            for name in order:
                for rows in self.ops.get(name, []):
                    body = self._apply(body, rows)
            if any(self.ops.values()):
                self._write(body)

    def snapshot(self, round_no: int, *, tasks: Optional[dict] = None) -> list:
        """This round's rulings kept in gates_by_round.json, so an earlier round can be read again.

        gates.json holds one ruling per stage, the last one, which is the state the report reads and
        which nothing here changes. A round that repairs an artifact rules again under the same
        stage names, so without this file a ruling that went from red to green leaves no trace of
        ever having been red, and no repair can be said to have moved a gate. One row per round,
        holding gates.json as it stood when the round ended; a round recorded twice replaces its row.

        `tasks` is the round's Task level counts as its own table derived them (D218 rule 1). They
        ride here rather than being counted again off the rulings, so the history row and the table
        are two readings of one answer.
        """
        with self.lock:
            rulings = self._read()
            rows = [row for row in self._read_history() if row.get("round") != round_no]
            rows.append({"round": round_no, "rulings": rulings, "tasks": dict(tasks or {})})
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
        """The distinct ruling names this stage recorded, in order.

        A per tool snapshot is a ruling the stage made, so its names belong here too even though its
        rows never enter gates.json (D218 rule 2); a stage that stopped naming what it ruled on would
        be the move hiding itself from the event a reader watches.
        """
        out: list[str] = []
        for rows in self.ops.get(stage_name, []):
            out += [row["stage"] for row in rows if row.get("stage") not in out]
        out += [name for name in self.snapshot_names.get(stage_name, []) if name and name not in out]
        return out
