from __future__ import annotations

import copy
import math
import os
import stat
import time
from pathlib import Path
from typing import Any

from kullback.store import ArtifactSpec, ReadResult, StoreError, WorkdirStore


class JournalError(StoreError):
    pass


_JOURNAL_DIR = "round-journal"
_OWNER = "rounds"
_FORMAT = 1
_KINDS = frozenset({"run_open", "round_open", "beat_end", "round_close", "abort"})
_AGENTS = frozenset({"builder", "examiner"})
_EVENT_KEYS = frozenset({"seq", "run_id", "kind", "round", "recorded_at", "payload"})


def _is_pos_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _check_list(items: list) -> None:
    for item in items:
        _check_value(item)


def _check_dict_values(mapping: dict) -> None:
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError("bad payload key")
        _check_value(item)


def _check_value(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("bad payload float")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        _check_list(value)
        return
    if isinstance(value, dict):
        _check_dict_values(value)
        return
    raise ValueError("bad payload value")


def _check_payload_object(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("bad payload")
    _check_dict_values(payload)


def _require_seq(seq: Any, expected: int) -> None:
    if not _is_pos_int(seq):
        raise ValueError("bad seq")
    if seq != expected:
        raise ValueError("bad seq")


def _require_run_id(run_id: Any) -> None:
    if not isinstance(run_id, str) or run_id == "":
        raise ValueError("bad run_id")


def _require_kind(kind: Any) -> None:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("bad kind")


def _require_round_field(round_value: Any) -> None:
    if round_value is None:
        return
    if not _is_pos_int(round_value):
        raise ValueError("bad round")


def _require_recorded_at(stamp: Any) -> None:
    if isinstance(stamp, bool):
        raise ValueError("bad recorded_at")
    if isinstance(stamp, int):
        return
    if isinstance(stamp, float):
        if not math.isfinite(stamp):
            raise ValueError("bad recorded_at")
        return
    raise ValueError("bad recorded_at")


def _require_agent(payload: dict) -> None:
    agent = payload.get("agent")
    if not isinstance(agent, str) or agent not in _AGENTS:
        raise ValueError("bad beat agent")


def _require_kind_shape(event: dict) -> None:
    kind = event["kind"]
    round_value = event["round"]
    payload = event["payload"]
    if kind == "run_open":
        if round_value is not None:
            raise ValueError("bad run_open round")
        if not isinstance(payload.get("config"), dict):
            raise ValueError("bad run_open config")
        return
    if kind in ("round_open", "beat_end", "round_close"):
        if not _is_pos_int(round_value):
            raise ValueError("bad round")
    if kind == "beat_end":
        _require_agent(payload)


def _check_event_shape(value: Any, expected: int) -> Any:
    if not isinstance(value, dict):
        raise ValueError("bad event")
    if set(value.keys()) != set(_EVENT_KEYS):
        raise ValueError("bad event keys")
    _require_seq(value["seq"], expected)
    _require_run_id(value["run_id"])
    _require_kind(value["kind"])
    _require_round_field(value["round"])
    _require_recorded_at(value["recorded_at"])
    _check_payload_object(value["payload"])
    _require_kind_shape(value)
    return value


def _spec_for_seq(seq: int) -> ArtifactSpec:
    if not _is_pos_int(seq):
        raise ValueError("bad seq")
    fname = "%020d.json" % seq
    return ArtifactSpec(
        name="round_journal:%d" % seq,
        path="%s/%s" % (_JOURNAL_DIR, fname),
        format=_FORMAT,
        owner=_OWNER,
        validate=lambda value: _check_event_shape(value, seq),
    )


def _new_run_state() -> dict:
    return {"started": False, "aborted": False, "active": None, "greatest": 0}


def _apply_run_open(state: dict, event: dict) -> str | None:
    if state["started"]:
        return "duplicate run_open for run %s" % event["run_id"]
    if event["round"] is not None:
        return "run_open round must be None for run %s" % event["run_id"]
    if not isinstance(event["payload"].get("config"), dict):
        return "run_open missing config for run %s" % event["run_id"]
    state["started"] = True
    state["aborted"] = False
    state["active"] = None
    return None


def _apply_round_open(state: dict, event: dict) -> str | None:
    round_value = event["round"]
    if state["active"] is not None:
        return "round_open with open round for run %s" % event["run_id"]
    if round_value <= state["greatest"]:
        return "round_open not greater for run %s" % event["run_id"]
    state["active"] = round_value
    state["greatest"] = round_value
    return None


def _apply_beat_end(state: dict, event: dict) -> str | None:
    if state["active"] is None:
        return "beat_end with no active round for run %s" % event["run_id"]
    if event["round"] != state["active"]:
        return "beat_end round mismatch for run %s" % event["run_id"]
    if event["payload"].get("agent") not in _AGENTS:
        return "beat_end bad agent for run %s" % event["run_id"]
    return None


def _apply_round_close(state: dict, event: dict) -> str | None:
    if state["active"] is None:
        return "round_close with no active round for run %s" % event["run_id"]
    if event["round"] != state["active"]:
        return "round_close round mismatch for run %s" % event["run_id"]
    state["active"] = None
    return None


def _apply_abort(state: dict, event: dict) -> str | None:
    round_value = event["round"]
    if round_value is None:
        if state["active"] is not None:
            return "abort None with active round for run %s" % event["run_id"]
        state["aborted"] = True
        return None
    if state["active"] is None:
        return "abort with no active round for run %s" % event["run_id"]
    if round_value != state["active"]:
        return "abort round mismatch for run %s" % event["run_id"]
    state["active"] = None
    state["aborted"] = True
    return None


def _apply_event(states: dict, event: dict) -> str | None:
    run_id = event["run_id"]
    kind = event["kind"]
    state = states.get(run_id)
    if state is None:
        state = _new_run_state()
        states[run_id] = state
    if kind == "run_open":
        return _apply_run_open(state, event)
    if not state["started"]:
        return "event before run_open for run %s" % run_id
    if state["aborted"]:
        return "event after abort for run %s" % run_id
    if kind == "round_open":
        return _apply_round_open(state, event)
    if kind == "beat_end":
        return _apply_beat_end(state, event)
    if kind == "round_close":
        return _apply_round_close(state, event)
    if kind == "abort":
        return _apply_abort(state, event)
    return "unknown kind for run %s" % run_id


def _lifecycle_reason(events: list) -> str | None:
    states: dict = {}
    for event in events:
        reason = _apply_event(states, event)
        if reason is not None:
            return reason
    return None


def _event_filename(seq: int) -> str:
    return "%020d.json" % seq


def _validate_new_event_shape(run_id: str, kind: str, payload: dict, round: int | None) -> None:
    _require_run_id(run_id)
    _require_kind(kind)
    _require_round_field(round)
    _check_payload_object(payload)
    _require_kind_shape({"kind": kind, "round": round, "payload": payload})


def _has_ascii_digits(stem: str) -> bool:
    if len(stem) != 20:
        return False
    for char in stem:
        if char < "0" or char > "9":
            return False
    return True


def _journal_seqs(journal_dir: Path) -> list:
    names = os.listdir(journal_dir)
    seqs: list = []
    for name in names:
        if not name.endswith(".json"):
            continue
        if len(name) != 25 or not _has_ascii_digits(name[:20]):
            raise JournalError("invalid journal filename %s" % name)
        seq = int(name[:20])
        if seq <= 0:
            raise JournalError("invalid journal filename %s" % name)
        seqs.append(seq)
    if not seqs:
        return []
    seqs.sort()
    if seqs[0] != 1:
        raise JournalError("sequence gap before %d" % seqs[0])
    for want, got in enumerate(seqs, start=1):
        if want != got:
            raise JournalError("sequence gap expected %d found %d" % (want, got))
    return seqs


def _check_links(root_str: str, journal_dir: Path, seqs: list) -> None:
    seen: dict = {}
    for seq in seqs:
        target = journal_dir / _event_filename(seq)
        try:
            detail = os.lstat(target)
        except FileNotFoundError as exc:
            raise JournalError("sequence gap at %d" % seq) from exc
        if stat.S_ISLNK(detail.st_mode):
            raise JournalError("event path is a symlink at %d" % seq)
        if not stat.S_ISREG(detail.st_mode):
            raise JournalError("event path is not a file at %d" % seq)
        if detail.st_nlink != 1:
            raise JournalError("event path is hard-linked at %d" % seq)
        key = (detail.st_dev, detail.st_ino)
        if key in seen:
            raise JournalError("event path is hard-linked at %d" % seq)
        seen[key] = seq
    try:
        lock_detail = os.lstat(Path(root_str) / ".store.lock")
    except FileNotFoundError:
        return
    lock_key = (lock_detail.st_dev, lock_detail.st_ino)
    if lock_key in seen:
        raise JournalError("event path aliases the store lock")


def _read_ordered(root_str: str, seqs: list) -> list:
    events: list = []
    for seq in seqs:
        spec = _spec_for_seq(seq)
        reader = WorkdirStore(root_str, [spec])
        try:
            result = reader.read(spec.name)
        except JournalError:
            raise
        except StoreError as exc:
            raise JournalError("event %d unreadable: %s" % (seq, exc)) from exc
        if result.status == "missing":
            raise JournalError("sequence gap at %d" % seq)
        if result.status == "torn":
            raise JournalError("event %d is torn: %s" % (seq, result.reason))
        events.append(result.value)
    return events


def _collect_locked(root_str: str) -> list:
    journal_dir = Path(root_str) / _JOURNAL_DIR
    try:
        info = os.lstat(journal_dir)
    except FileNotFoundError:
        return []
    if stat.S_ISLNK(info.st_mode):
        raise JournalError("journal directory is a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise JournalError("journal directory is not a directory")
    seqs = _journal_seqs(journal_dir)
    if not seqs:
        return []
    _check_links(root_str, journal_dir, seqs)
    events = _read_ordered(root_str, seqs)
    reason = _lifecycle_reason(events)
    if reason is not None:
        raise JournalError(reason)
    return events


class RoundJournal:
    def __init__(self, root: str | Path) -> None:
        self._root = str(Path(root).resolve())

    def read(self) -> ReadResult:
        journal_dir = Path(self._root) / _JOURNAL_DIR
        try:
            os.lstat(journal_dir)
        except FileNotFoundError:
            return ReadResult(status="missing")
        outer = WorkdirStore(self._root, [])
        with outer.transaction():
            try:
                events = _collect_locked(self._root)
            except JournalError as exc:
                return ReadResult(status="torn", reason=str(exc))
            if not events:
                return ReadResult(status="missing")
            return ReadResult(status="ok", value=copy.deepcopy(events))

    def append(self, *, run_id: str, kind: str, payload: dict, round: int | None = None) -> dict:
        _validate_new_event_shape(run_id, kind, payload, round)
        outer = WorkdirStore(self._root, [])
        with outer.transaction():
            history = _collect_locked(self._root)
            if kind != "run_open" and not history:
                raise JournalError("event before run_open for run %s" % run_id)
            coming = len(history) + 1
            target = Path(self._root) / _JOURNAL_DIR / _event_filename(coming)
            try:
                os.lstat(target)
                raise JournalError("event slot %d already present" % coming)
            except FileNotFoundError:
                pass
            event = {
                "seq": coming,
                "run_id": run_id,
                "kind": kind,
                "round": round,
                "recorded_at": time.time(),
                "payload": copy.deepcopy(payload),
            }
            _check_event_shape(event, coming)
            probe: list = list(history) + [event]
            reason = _lifecycle_reason(probe)
            if reason is not None:
                raise JournalError(reason)
            spec = _spec_for_seq(coming)
            writer = WorkdirStore(self._root, [spec])
            writer.write(spec.name, event)
            return copy.deepcopy(event)
