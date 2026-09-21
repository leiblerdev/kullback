from __future__ import annotations

import json
import multiprocessing
import os
import stat
import time
from pathlib import Path

import pytest

from kullback.journal import JournalError, RoundJournal
from kullback.store import ReadResult, StoreError, WorkdirStore, WriteDurabilityError


def _run_open_payload(config=None):
    if config is None:
        config = {"target": "t"}
    return {"config": config}


def _journal_dir(root):
    return Path(root) / "round-journal"


def _list_events(root):
    return sorted((Path(root) / "round-journal").iterdir())


def _snapshot_bytes(root):
    out = {}
    base = Path(root) / "round-journal"
    if not base.exists() or not base.is_dir():
        return out
    for child in base.iterdir():
        if child.is_symlink() or not child.is_file():
            continue
        out[child.name] = child.read_bytes()
    return out


def _raw_envelope(name, value, owner="rounds", fmt=1, store_format=1):
    return {"store_format": store_format, "artifact": name, "format": fmt, "owner": owner, "value": value}


def _write_raw(root, seq, value, owner="rounds", fmt=1, store_format=1):
    name = "round_journal:%d" % seq
    fname = "%020d.json" % seq
    target = Path(root) / "round-journal" / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_raw_envelope(name, value, owner, fmt, store_format), sort_keys=True, indent=2), encoding="utf-8")
    return target


def _opened_journal(root):
    journal = RoundJournal(root)
    journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    return journal


def _gated_worker(root_str, run_id, count, ready_evt, go_evt):
    from kullback.journal import RoundJournal as _J
    journal = _J(root_str)
    ready_evt.set()
    if not go_evt.wait(timeout=60):
        raise RuntimeError("release timeout")
    journal.append(run_id=run_id, kind="run_open", payload={"config": {"w": run_id}}, round=None)
    journal.append(run_id=run_id, kind="round_open", payload={}, round=1)
    for _ in range(count):
        journal.append(run_id=run_id, kind="beat_end", payload={"agent": "builder"}, round=1)


def test_missing_on_absent_root(tmp_path):
    root = tmp_path / "absent"
    journal = RoundJournal(root)
    seen = journal.read()
    assert seen.status == "missing"
    assert seen.value is None
    assert not root.exists()


def test_missing_on_empty_journal_dir(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    journal = RoundJournal(root)
    seen = journal.read()
    assert seen.status == "missing"
    assert seen.value is None


def test_read_does_not_create_root(tmp_path):
    root = tmp_path / "nope"
    journal = RoundJournal(root)
    journal.read()
    assert not root.exists()


def test_happy_path_all_kinds(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    first = journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    assert first["seq"] == 1
    assert first["run_id"] == "r1"
    assert first["kind"] == "run_open"
    assert first["round"] is None
    assert isinstance(first["recorded_at"], float) or isinstance(first["recorded_at"], int)
    assert first["payload"] == {"config": {"target": "t"}}
    second = journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    assert second["seq"] == 2
    third = journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=2)
    assert third["seq"] == 3
    fourth = journal.append(run_id="r1", kind="beat_end", payload={"agent": "examiner"}, round=2)
    assert fourth["seq"] == 4
    fifth = journal.append(run_id="r1", kind="round_close", payload={"exit": "done"}, round=2)
    assert fifth["seq"] == 5
    sixth = journal.append(run_id="r1", kind="round_open", payload={}, round=3)
    assert sixth["seq"] == 6
    seventh = journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=3)
    assert seventh["seq"] == 7
    eighth = journal.append(run_id="r1", kind="abort", payload={"why": "x"}, round=3)
    assert eighth["seq"] == 8
    seen = journal.read()
    assert seen.status == "ok"
    assert isinstance(seen, ReadResult)
    assert [e["seq"] for e in seen.value] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [e["kind"] for e in seen.value] == ["run_open", "round_open", "beat_end", "beat_end", "round_close", "round_open", "beat_end", "abort"]


def test_interleaved_two_runs(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    journal.append(run_id="a", kind="run_open", payload=_run_open_payload(), round=None)
    journal.append(run_id="b", kind="run_open", payload=_run_open_payload({}), round=None)
    journal.append(run_id="a", kind="round_open", payload={}, round=1)
    journal.append(run_id="b", kind="round_open", payload={}, round=5)
    journal.append(run_id="a", kind="beat_end", payload={"agent": "builder"}, round=1)
    journal.append(run_id="b", kind="beat_end", payload={"agent": "examiner"}, round=5)
    seen = journal.read()
    assert seen.status == "ok"
    assert [e["seq"] for e in seen.value] == [1, 2, 3, 4, 5, 6]
    assert [e["run_id"] for e in seen.value] == ["a", "b", "a", "b", "a", "b"]


def test_second_invocation_preserves_bytes(tmp_path):
    root = tmp_path / "w"
    first = RoundJournal(root)
    first.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    first.append(run_id="r1", kind="round_open", payload={}, round=1)
    before = _snapshot_bytes(root)
    assert len(before) == 2
    second = RoundJournal(root)
    seen = second.read()
    assert seen.status == "ok"
    assert len(seen.value) == 2
    second.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    after = _snapshot_bytes(root)
    assert len(after) == 3
    for name, data in before.items():
        assert after[name] == data


def test_stable_global_sequence(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r2", kind="run_open", payload=_run_open_payload(), round=None)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    seen = journal.read()
    assert [e["seq"] for e in seen.value] == [1, 2, 3]
    fnames = sorted(p.name for p in _journal_dir(root).glob("*.json"))
    assert fnames == ["%020d.json" % i for i in (1, 2, 3)]


def test_nested_mutation_isolation(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    payload = {"config": {"nested": {"items": [1, 2]}}}
    event = journal.append(run_id="r1", kind="run_open", payload=payload, round=None)
    payload["config"]["nested"]["items"].append(99)
    payload["config"]["extra"] = 1
    seen = journal.read()
    assert seen.value[0]["payload"] == {"config": {"nested": {"items": [1, 2]}}}
    event["payload"]["config"]["nested"]["items"].append(100)
    again = journal.read()
    assert again.value[0]["payload"] == {"config": {"nested": {"items": [1, 2]}}}
    seen.value[0]["payload"]["config"]["nested"]["items"].append(101)
    third = journal.read()
    assert third.value[0]["payload"] == {"config": {"nested": {"items": [1, 2]}}}


def test_output_list_detached(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    first = journal.read()
    second = journal.read()
    assert first.value is not second.value
    assert first.value[0] is not second.value[0]
    assert first.value[0]["payload"] is not second.value[0]["payload"]
    first.value.append({"seq": 999})
    assert len(journal.read().value) == 1


def test_reject_duplicate_run_open(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    assert len(journal.read().value) == 1


def test_reject_second_round_open_without_close(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    assert len(journal.read().value) == 2


def test_reject_event_before_run(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(JournalError):
        journal.append(run_id="ghost", kind="round_open", payload={}, round=1)
    with pytest.raises(JournalError):
        journal.append(run_id="ghost", kind="beat_end", payload={"agent": "builder"}, round=1)


def test_reject_terminal_duplicates(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="abort", payload={}, round=1)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="abort", payload={}, round=1)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=2)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)


def test_reject_round_open_not_greater(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=3)
    journal.append(run_id="r1", kind="round_close", payload={}, round=3)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=3)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=2)


def test_reject_unknown_kind(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="nope", payload={}, round=None)


def test_reject_bool_round(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=True)


def test_reject_nonfinite_payload(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload={"config": {"x": float("nan")}}, round=None)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload={"config": {"x": float("inf")}}, round=None)


def test_reject_bad_config(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload={}, round=None)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload={"config": [1]}, round=None)
    journal.append(run_id="r1", kind="run_open", payload={"config": {}}, round=None)
    assert journal.read().status == "ok"


def test_reject_bad_run_id_and_shapes(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="", kind="run_open", payload=_run_open_payload(), round=None)
    with pytest.raises(ValueError):
        journal.append(run_id=123, kind="run_open", payload=_run_open_payload(), round=None)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload=[], round=None)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload={1: "x", "config": {}}, round=None)


def test_reject_beat_end_bad_agent(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="beat_end", payload={}, round=1)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="beat_end", payload={"agent": "other"}, round=1)


def test_reject_run_open_with_round(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=1)


def test_multiple_beats_same_agent_preserved(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    seen = journal.read()
    assert seen.status == "ok"
    assert len(seen.value) == 4


def test_torn_gap(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    _write_raw(root, 3, {"seq": 3, "run_id": "r1", "kind": "round_open", "round": 1, "recorded_at": 1.0, "payload": {}})
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None
    assert isinstance(seen.reason, str) and len(seen.reason) > 0


def test_torn_first_seq_not_one(tmp_path):
    root = tmp_path / "w"
    _write_raw(root, 2, {"seq": 2, "run_id": "r1", "kind": "run_open", "round": None, "recorded_at": 1.0, "payload": {"config": {}}})
    seen = RoundJournal(root).read()
    assert seen.status == "torn"
    assert seen.value is None


def test_torn_malformed_event(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    target = _journal_dir(root) / "00000000000000000001.json"
    target.write_bytes(b'{"store_format": 1,')
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None
    assert isinstance(seen.reason, str) and len(seen.reason) > 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("format", 999), ("owner", "someone"), ("artifact", "other")],
    ids=["format", "owner", "artifact"],
)
def test_torn_wrong_envelope_field(tmp_path, field, bad_value):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    target = _journal_dir(root) / "00000000000000000001.json"
    obj = json.loads(target.read_text(encoding="utf-8"))
    obj[field] = bad_value
    target.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding="utf-8")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


@pytest.mark.parametrize("fname", ["abc.json", "00000000000000000000.json"], ids=["non-numeric", "zero-seq"])
def test_torn_bad_sequence_filename(tmp_path, fname):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    (_journal_dir(root) / fname).write_text("{}", encoding="utf-8")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_torn_lifecycle_conflict_in_stored_history(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    _write_raw(root, 2, {"seq": 2, "run_id": "r1", "kind": "run_open", "round": None, "recorded_at": 2.0, "payload": {"config": {}}})
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_staging_files_ignored(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    staging = _journal_dir(root) / "tmp_staging_123"
    staging.write_text("partial", encoding="utf-8")
    seen = journal.read()
    assert seen.status == "ok"
    assert len(seen.value) == 1
    assert staging.exists()


def test_io_propagation_on_read(tmp_path, monkeypatch):
    root = tmp_path / "w"
    journal = _opened_journal(root)

    def _denied(self):
        raise PermissionError("injected")

    monkeypatch.setattr(Path, "read_bytes", _denied)
    with pytest.raises(PermissionError):
        journal.read()
    monkeypatch.undo()


def test_symlink_journal_dir_refused(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    real = _journal_dir(root)
    backup = tmp_path / "real_backup"
    try:
        real.rename(backup)
        real.symlink_to(backup, target_is_directory=True)
    except OSError:
        pytest.skip("symlink unavailable")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_symlink_event_refused(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    target = _journal_dir(root) / "00000000000000000001.json"
    data = target.read_bytes()
    other = tmp_path / "other.json"
    other.write_bytes(data)
    try:
        target.unlink()
        target.symlink_to(other)
    except OSError:
        pytest.skip("symlink unavailable")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_hardlink_event_refused(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    first = _journal_dir(root) / "00000000000000000001.json"
    second = _journal_dir(root) / "00000000000000000002.json"
    try:
        second.unlink()
        os.link(first, second)
    except OSError:
        pytest.skip("hardlink unavailable")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_append_refusal_on_torn_preserves_bytes(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    target = _journal_dir(root) / "00000000000000000001.json"
    target.write_bytes(b"broken")
    after_corrupt = _snapshot_bytes(root)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert _snapshot_bytes(root) == after_corrupt
    assert journal.read().status == "torn"


def test_append_never_overwrites_existing_slot(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    _write_raw(root, 2, {"seq": 999, "run_id": "r1", "kind": "round_open", "round": 1, "recorded_at": 1.0, "payload": {}})
    before = _snapshot_bytes(root)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert _snapshot_bytes(root) == before


def test_pre_replacement_failure_preserves_bytes(tmp_path, monkeypatch):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    before = _snapshot_bytes(root)
    real_replace = os.replace

    def _boom(src, dst):
        raise OSError("injected")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert _snapshot_bytes(root) == before
    monkeypatch.setattr(os, "replace", real_replace)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert journal.read().status == "ok"


def test_post_replacement_durability_retains_event(tmp_path, monkeypatch):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    real_fsync = os.fsync

    def _fsync_fail(fd):
        try:
            mode = os.fstat(fd).st_mode
        except OSError:
            return real_fsync(fd)
        if stat.S_ISDIR(mode):
            raise OSError("injected dir fsync")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync_fail)
    with pytest.raises(WriteDurabilityError) as excinfo:
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert excinfo.value.replaced is True
    assert excinfo.value.durability == "unknown"
    monkeypatch.undo()
    target = _journal_dir(root) / "00000000000000000002.json"
    assert target.exists()
    seen = journal.read()
    assert seen.status == "ok"
    assert [e["seq"] for e in seen.value] == [1, 2]
    journal.append(run_id="r1", kind="beat_end", payload={"agent": "builder"}, round=1)
    assert [e["seq"] for e in journal.read().value] == [1, 2, 3]


def test_nested_transaction(tmp_path):
    root = tmp_path / "w"
    outer = WorkdirStore(root, [])
    journal = RoundJournal(root)
    with outer.transaction():
        journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
        with outer.transaction():
            journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert len(journal.read().value) == 2


def test_concurrent_processes_separate_runs(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    journal.append(run_id="seed", kind="run_open", payload=_run_open_payload(), round=None)
    ctx = multiprocessing.get_context("spawn")
    ready_first = ctx.Event()
    ready_second = ctx.Event()
    go = ctx.Event()
    first = ctx.Process(target=_gated_worker, args=(str(root), "pa", 5, ready_first, go))
    second = ctx.Process(target=_gated_worker, args=(str(root), "pb", 5, ready_second, go))
    try:
        first.start()
        second.start()
        assert ready_first.wait(timeout=20)
        assert ready_second.wait(timeout=20)
        go.set()
        first.join(timeout=60)
        second.join(timeout=60)
    finally:
        for proc in (first, second):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    assert first.exitcode == 0
    assert second.exitcode == 0
    seen = journal.read()
    assert seen.status == "ok"
    assert len(seen.value) == 15
    seqs = [e["seq"] for e in seen.value]
    assert seqs == list(range(1, 16))
    assert len(set(seqs)) == 15
    by_run = {}
    for event in seen.value:
        by_run.setdefault(event["run_id"], []).append(event["kind"])
    assert sorted(by_run.keys()) == ["pa", "pb", "seed"]
    assert len(by_run["pa"]) == 7
    assert len(by_run["pb"]) == 7
    assert by_run["seed"] == ["run_open"]


def test_journal_error_is_store_error():
    assert issubclass(JournalError, StoreError)


def test_recorded_at_generated(tmp_path, monkeypatch):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    event = journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    assert event["recorded_at"] == 1700000000.0
    seen = journal.read()
    assert seen.value[0]["recorded_at"] == 1700000000.0


def test_envelope_shape_on_disk(tmp_path):
    root = tmp_path / "w"
    _opened_journal(root)
    target = _journal_dir(root) / "00000000000000000001.json"
    obj = json.loads(target.read_text(encoding="utf-8"))
    assert obj["store_format"] == 1
    assert obj["artifact"] == "round_journal:1"
    assert obj["owner"] == "rounds"
    assert obj["format"] == 1
    assert set(obj.keys()) == {"store_format", "artifact", "format", "owner", "value"}
    assert set(obj["value"].keys()) == {"seq", "run_id", "kind", "round", "recorded_at", "payload"}


def test_root_binding_stable_across_chdir(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    journal = RoundJournal("work")
    journal.append(run_id="r1", kind="run_open", payload=_run_open_payload(), round=None)
    monkeypatch.chdir(second)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    seen = journal.read()
    assert seen.status == "ok"
    assert len(seen.value) == 2
    assert (first / "work" / "round-journal").is_dir()
    assert not (second / "work").exists()


@pytest.mark.parametrize("digit", ["\u00b2", "\u0661"], ids=["superscript", "arabic-indic"])
def test_torn_non_ascii_digit_filename(tmp_path, digit):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    bad = digit * 20 + ".json"
    (_journal_dir(root) / bad).write_text("{}", encoding="utf-8")
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None
    assert isinstance(seen.reason, str) and len(seen.reason) > 0


def test_append_refusal_on_superscript_filename_preserves_bytes(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    bad = "\u00b2" * 20 + ".json"
    (_journal_dir(root) / bad).write_text("{}", encoding="utf-8")
    before = _snapshot_bytes(root)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    assert _snapshot_bytes(root) == before
    assert journal.read().status == "torn"


def test_reject_unhashable_kind(tmp_path):
    root = tmp_path / "w"
    journal = RoundJournal(root)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind=[], payload={}, round=None)


def test_reject_unhashable_agent(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    with pytest.raises(ValueError):
        journal.append(run_id="r1", kind="beat_end", payload={"agent": {}}, round=1)


def test_torn_persisted_unhashable_kind(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    _write_raw(root, 2, {"seq": 2, "run_id": "r1", "kind": [], "round": 1, "recorded_at": 2.0, "payload": {}})
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_torn_persisted_unhashable_agent(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    _write_raw(root, 3, {"seq": 3, "run_id": "r1", "kind": "beat_end", "round": 1, "recorded_at": 3.0, "payload": {"agent": {}}})
    seen = journal.read()
    assert seen.status == "torn"
    assert seen.value is None


def test_abort_between_rounds_retains_records(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    journal.append(run_id="r1", kind="round_close", payload={"exit": "done"}, round=1)
    event = journal.append(run_id="r1", kind="abort", payload={}, round=None)
    assert event["seq"] == 4
    seen = journal.read()
    assert seen.status == "ok"
    assert [e["kind"] for e in seen.value] == ["run_open", "round_open", "round_close", "abort"]
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="round_open", payload={}, round=2)


def test_abort_setup_without_rounds(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    event = journal.append(run_id="r1", kind="abort", payload={}, round=None)
    assert event["seq"] == 2
    assert journal.read().status == "ok"


def test_reject_abort_none_with_active_round(tmp_path):
    root = tmp_path / "w"
    journal = _opened_journal(root)
    journal.append(run_id="r1", kind="round_open", payload={}, round=1)
    with pytest.raises(JournalError):
        journal.append(run_id="r1", kind="abort", payload={}, round=None)
