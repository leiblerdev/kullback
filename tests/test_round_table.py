from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.journal import RoundJournal
from kullback.round_table import (
    AbortedAttempt,
    CompletedRound,
    RoundTable,
    UnfinishedAttempt,
    read_round_table,
)
from kullback.runner.records import RoundRecord


def _open(root, run_id="r1"):
    journal = RoundJournal(root)
    journal.append(run_id=run_id, kind="run_open", payload={"config": {}}, round=None)
    return journal


def _close_payload(number, counts=None):
    return {"record": {"round": number, "counts": {"built": number} if counts is None else counts}}


def _write_raw(root, seq, value):
    target = Path(root) / "round-journal" / ("%020d.json" % seq)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "store_format": 1,
        "artifact": "round_journal:%d" % seq,
        "format": 1,
        "owner": "rounds",
        "value": value,
    }
    target.write_text(json.dumps(envelope, sort_keys=True, indent=2), encoding="utf-8")


def _event(seq, run_id, kind, number, payload, recorded_at=1700000000.0):
    return {
        "seq": seq,
        "run_id": run_id,
        "kind": kind,
        "round": number,
        "recorded_at": recorded_at,
        "payload": payload,
    }


def _snapshot(root):
    out = {}
    base = Path(root) / "round-journal"
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if child.is_file() and not child.is_symlink():
            out[child.name] = child.read_bytes()
    return out


def test_missing_on_absent_root(tmp_path):
    root = tmp_path / "absent"
    seen = read_round_table(root)
    assert isinstance(seen, RoundTable)
    assert seen.status == "missing"
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []
    assert not root.exists()


def test_missing_does_not_read_legacy_rounds(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "rounds.json").write_text(json.dumps([{"round": 1}]), encoding="utf-8")
    seen = read_round_table(root)
    assert seen.status == "missing"
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_run_open_only_is_not_a_completed_round(tmp_path):
    root = tmp_path / "w"
    _open(root)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert seen.reason is None
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_normal_round_record(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1, {"built": 2, "failed": 0}), round=1)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert len(seen.completed) == 1
    assert seen.aborted == []
    assert seen.unfinished == []
    row = seen.completed[0]
    assert isinstance(row, CompletedRound)
    assert (row.run_id, row.round) == ("r1", 1)
    assert (row.open_seq, row.close_seq) == (2, 4)
    assert isinstance(row.record, RoundRecord)
    assert row.record.round == 1
    assert row.record.counts == {"built": 2, "failed": 0}


def test_record_with_aliased_nested_fields(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    payload = {
        "record": {
            "round": 2,
            "counts": {"built": 1},
            "exit": "done",
            "pending_findings": [
                {
                    "finding_id": "f1",
                    "task_id": "t1",
                    "kind": "other",
                    "text": "t",
                    "run_id": "r1",
                    "tool": "tool_a",
                    "suggested": "replay",
                    "hint": "h",
                    "about_entry_id": "e1",
                    "round": 2,
                    "status": "open",
                    "task_ids": ["t1", "t2"],
                    "key": "other:tool_a",
                }
            ],
        }
    }
    journal.append(run_id="r1", kind="round_close", payload=payload, round=2)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert len(seen.completed) == 1
    record = seen.completed[0].record
    assert record.exit == "done"
    assert len(record.pending_findings) == 1
    assert record.pending_findings[0].finding_id == "f1"
    assert record.pending_findings[0].suggested == "replay"
    assert record.pending_findings[0].hint == "h"
    assert record.pending_findings[0].key == "other:tool_a"
    assert record.pending_findings[0].task_ids == ["t1", "t2"]
    assert record.pending_findings[0].cost == 2


def test_multi_round_ordering(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(2), round=2)
    journal.append(run_id="r1", kind="round_open", payload={}, round=3)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(3), round=3)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert [(row.run_id, row.round) for row in seen.completed] == [("r1", 1), ("r1", 2), ("r1", 3)]
    assert [(row.open_seq, row.close_seq) for row in seen.completed] == [(2, 3), (4, 5), (6, 7)]
    assert [row.close_seq for row in seen.completed] == sorted(row.close_seq for row in seen.completed)


def test_interleaved_runs_sharing_round_number_stay_distinct(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    journal.append(run_id="a", kind="run_open", payload={"config": {}}, round=None)
    journal.append(run_id="b", kind="run_open", payload={"config": {}}, round=None)
    journal.append(run_id="a", kind="round_open", payload={}, round=1)
    journal.append(run_id="b", kind="round_open", payload={}, round=1)
    journal.append(run_id="a", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="a", kind="round_open", payload={}, round=2)
    journal.append(run_id="b", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="b", kind="round_open", payload={}, round=2)
    journal.append(run_id="a", kind="round_close", payload=_close_payload(2), round=2)
    journal.append(run_id="b", kind="round_close", payload=_close_payload(2), round=2)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert [(row.run_id, row.round) for row in seen.completed] == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]
    assert [row.close_seq for row in seen.completed] == [5, 7, 9, 10]
    assert len(seen.completed) == 4


def test_abort_during_round_excluded_from_completed(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=2)
    journal.append(run_id="r1", kind="abort", payload={"why": "stop"}, round=2)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert [(row.run_id, row.round) for row in seen.completed] == [("r1", 1)]
    assert seen.unfinished == []
    assert len(seen.aborted) == 1
    attempt = seen.aborted[0]
    assert isinstance(attempt, AbortedAttempt)
    assert attempt.run_id == "r1"
    assert attempt.round == 2
    assert attempt.open_seq == 4
    assert attempt.abort_seq == 6
    assert attempt.payload == {"why": "stop"}


def test_abort_between_rounds_keeps_completed_rows(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="r1", kind="abort", payload={"why": "late"}, round=None)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert [(row.run_id, row.round) for row in seen.completed] == [("r1", 1)]
    assert seen.unfinished == []
    assert len(seen.aborted) == 1
    assert seen.aborted[0].run_id == "r1"
    assert seen.aborted[0].round is None
    assert seen.aborted[0].open_seq is None
    assert seen.aborted[0].abort_seq == 4
    assert seen.aborted[0].payload == {"why": "late"}


def test_abort_setup_without_rounds(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="abort", payload={}, round=None)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert seen.completed == []
    assert seen.unfinished == []
    assert len(seen.aborted) == 1
    assert seen.aborted[0].round is None


def test_unfinished_round_carries_beats_in_order(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
    journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder", "note": 1}, round=2)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "examiner", "note": 2}, round=2)
    seen = read_round_table(root)
    assert seen.status == "ok"
    assert [(row.run_id, row.round) for row in seen.completed] == [("r1", 1)]
    assert seen.aborted == []
    assert len(seen.unfinished) == 1
    pending = seen.unfinished[0]
    assert isinstance(pending, UnfinishedAttempt)
    assert pending.run_id == "r1"
    assert pending.round == 2
    assert pending.open_seq == 4
    assert pending.beats == [{"agent": "builder", "note": 1}, {"agent": "examiner", "note": 2}]


@pytest.mark.parametrize(
    ("prefix_good_round", "bad_payload"),
    [
        pytest.param(True, {"record": [1, 2]}, id="record-not-mapping"),
        pytest.param(False, {"exit": "done"}, id="missing-record"),
        pytest.param(True, _close_payload(3), id="round-mismatch"),
        pytest.param(False, {"record": {"round": True, "counts": {}}}, id="bool-round"),
        pytest.param(False, {"record": {"round": 1, "nope": 1}}, id="invalid-record"),
    ],
)
def test_close_bad_payload_is_torn(tmp_path, prefix_good_round, bad_payload):
    root = tmp_path / "w"
    journal = _open(root)
    if prefix_good_round:
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
        journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
        bad_round = 2
    else:
        bad_round = 1
    journal.append(run_id="r1", kind="round_open", payload={}, round=bad_round)
    journal.append(run_id="r1", kind="round_close", payload=bad_payload, round=bad_round)
    seen = read_round_table(root)
    assert seen.status == "torn"
    assert isinstance(seen.reason, str) and len(seen.reason) > 0
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_torn_journal_gap_has_no_rows(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    _write_raw(root, 3, _event(3, "r1", "round_open", 1, {}, 3.0))
    assert journal.read().status == "torn"
    seen = read_round_table(root)
    assert seen.status == "torn"
    assert isinstance(seen.reason, str) and len(seen.reason) > 0
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_torn_journal_malformed_event_has_no_rows(tmp_path):
    root = tmp_path / "w"
    _open(root)
    target = Path(root) / "round-journal" / "00000000000000000001.json"
    target.write_bytes(b'{"store_format": 1,')
    seen = read_round_table(root)
    assert seen.status == "torn"
    assert isinstance(seen.reason, str) and len(seen.reason) > 0
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_duplicate_terminal_preserved_as_torn(tmp_path):
    root = tmp_path / "w"
    _write_raw(root, 1, _event(1, "r1", "run_open", None, {"config": {}}, 1.0))
    _write_raw(root, 2, _event(2, "r1", "round_open", 1, {}, 2.0))
    _write_raw(root, 3, _event(3, "r1", "round_close", 1, {"record": {"round": 1}}, 3.0))
    _write_raw(root, 4, _event(4, "r1", "round_close", 1, {"record": {"round": 1}}, 4.0))
    assert RoundJournal(root).read().status == "torn"
    seen = read_round_table(root)
    assert seen.status == "torn"
    assert isinstance(seen.reason, str) and len(seen.reason) > 0
    assert seen.completed == []
    assert seen.aborted == []
    assert seen.unfinished == []


def test_outputs_detached_from_journal_and_each_other(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    journal.append(run_id="a", kind="run_open", payload={"config": {}}, round=None)
    journal.append(run_id="b", kind="run_open", payload={"config": {}}, round=None)
    journal.append(run_id="a", kind="round_open", payload={}, round=1)
    journal.append(run_id="a", kind="round_close", payload=_close_payload(1, {"built": 5}), round=1)
    journal.append(run_id="b", kind="round_open", payload={}, round=1)
    journal.append(run_id="b", kind="beat_end", payload={"agent": "builder"}, round=1)
    journal.append(run_id="b", kind="abort", payload={"why": "x"}, round=1)
    first = read_round_table(root)
    assert first.status == "ok"
    first.completed[0].record.counts["built"] = 999
    first.completed[0].record.counts["extra"] = 1
    first.aborted[0].payload["why"] = "changed"
    first.aborted[0].payload["extra"] = 1
    first.completed.pop()
    first.aborted.pop()
    second = read_round_table(root)
    assert len(second.completed) == 1
    assert second.completed[0].record.counts == {"built": 5}
    assert len(second.aborted) == 1
    assert second.aborted[0].payload == {"why": "x"}
    assert journal.read().status == "ok"
    third = read_round_table(root)
    assert second == third
    assert second is not third
    assert second.completed[0] is not third.completed[0]


def test_unfinished_beats_detached(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    first = read_round_table(root)
    first.unfinished[0].beats.append({"agent": "examiner"})
    first.unfinished[0].beats[0]["agent"] = "changed"
    second = read_round_table(root)
    assert second.unfinished[0].beats == [{"agent": "builder"}]


def test_projection_leaves_file_bytes_unchanged(tmp_path):
    root = tmp_path / "w"
    journal = _open(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    journal.append(run_id="r1", kind="round_close", payload=_close_payload(1), round=1)
    before = _snapshot(root)
    assert len(before) == 4
    seen = read_round_table(root)
    assert seen.status == "ok"
    seen.completed[0].record.counts["built"] = -1
    again = read_round_table(root)
    assert again.status == "ok"
    assert _snapshot(root) == before
