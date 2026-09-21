from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kullback.journal import RoundJournal
from kullback.runner.records import RoundRecord


@dataclass
class CompletedRound:
    run_id: str
    round: int
    open_seq: int
    close_seq: int
    record: RoundRecord

    @property
    def start_seq(self) -> int:
        return self.open_seq

    @property
    def end_seq(self) -> int:
        return self.close_seq


@dataclass
class AbortedAttempt:
    run_id: str
    round: int | None
    open_seq: int | None
    abort_seq: int
    payload: dict = field(default_factory=dict)

    @property
    def start_seq(self) -> int | None:
        return self.open_seq

    @property
    def end_seq(self) -> int:
        return self.abort_seq


@dataclass
class UnfinishedAttempt:
    run_id: str
    round: int
    open_seq: int
    beats: list = field(default_factory=list)

    @property
    def start_seq(self) -> int:
        return self.open_seq


@dataclass
class RoundTable:
    status: Literal["ok", "missing", "torn"]
    reason: str | None = None
    completed: list = field(default_factory=list)
    aborted: list = field(default_factory=list)
    unfinished: list = field(default_factory=list)

    @property
    def completed_rounds(self) -> list:
        return self.completed

    @property
    def aborted_attempts(self) -> list:
        return self.aborted

    @property
    def unfinished_attempts(self) -> list:
        return self.unfinished


class _TornProjection(Exception):
    pass


def _close_record(event: dict) -> RoundRecord:
    raw = event["payload"].get("record")
    if not isinstance(raw, dict):
        raise _TornProjection("close record not a mapping at seq %d" % event["seq"])
    number = raw.get("round")
    if type(number) is not int or number != event["round"]:
        raise _TornProjection("close record round mismatch at seq %d" % event["seq"])
    try:
        record = RoundRecord.model_validate(copy.deepcopy(raw))
    except Exception as exc:
        raise _TornProjection("close record invalid at seq %d: %s" % (event["seq"], type(exc).__name__)) from exc
    if record.round != event["round"]:
        raise _TornProjection("close record round mismatch at seq %d" % event["seq"])
    return record


def _track_open(active: dict, event: dict) -> None:
    active[event["run_id"]] = [event["round"], event["seq"], []]


def _track_beat(active: dict, event: dict) -> None:
    found = active.get(event["run_id"])
    if found is None:
        raise _TornProjection("beat with no active round at seq %d" % event["seq"])
    found[2].append(copy.deepcopy(event["payload"]))


def _track_close(active: dict, completed: list, event: dict) -> None:
    found = active.pop(event["run_id"], None)
    if found is None:
        raise _TornProjection("close with no active round at seq %d" % event["seq"])
    completed.append(CompletedRound(event["run_id"], event["round"], found[1], event["seq"], _close_record(event)))


def _track_abort(active: dict, aborted: list, event: dict) -> None:
    if event["round"] is None:
        aborted.append(AbortedAttempt(event["run_id"], None, None, event["seq"], copy.deepcopy(event["payload"])))
        return
    found = active.pop(event["run_id"], None)
    if found is None:
        raise _TornProjection("abort with no active round at seq %d" % event["seq"])
    aborted.append(
        AbortedAttempt(event["run_id"], event["round"], found[1], event["seq"], copy.deepcopy(event["payload"]))
    )


def _project_events(events: list) -> tuple:
    completed: list = []
    aborted: list = []
    active: dict = {}
    for event in events:
        kind = event["kind"]
        if kind == "round_open":
            _track_open(active, event)
        elif kind == "beat_end":
            _track_beat(active, event)
        elif kind == "round_close":
            _track_close(active, completed, event)
        elif kind == "abort":
            _track_abort(active, aborted, event)
    unfinished = [UnfinishedAttempt(run_id, opened[0], opened[1], opened[2]) for run_id, opened in active.items()]
    completed.sort(key=lambda row: row.close_seq)
    return completed, aborted, unfinished


def read_round_table(workdir: str | Path) -> RoundTable:
    seen = RoundJournal(workdir).read()
    if seen.status == "missing":
        return RoundTable(status="missing", reason=seen.reason)
    if seen.status == "torn":
        return RoundTable(status="torn", reason=seen.reason)
    try:
        completed, aborted, unfinished = _project_events(seen.value)
    except _TornProjection as exc:
        return RoundTable(status="torn", reason=str(exc))
    return RoundTable(status="ok", completed=completed, aborted=aborted, unfinished=unfinished)
