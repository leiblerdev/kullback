from __future__ import annotations

import errno
import json
import multiprocessing
import os
import stat
import tempfile
import threading
from pathlib import Path

import pytest

from kullback.store import ArtifactSpec, ReadResult, StoreError, WorkdirStore


def _check_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("bad int")
    return value


def _check_dict_n(value):
    if not isinstance(value, dict):
        raise ValueError("bad dict")
    current = value.get("n")
    if isinstance(current, bool) or not isinstance(current, int):
        raise ValueError("bad n")
    return current


def _identity(value):
    return value


def _counter_worker(root_str, spec_name, spec_path, spec_format, spec_owner, rounds):
    from kullback.store import ArtifactSpec as _S
    from kullback.store import WorkdirStore as _W

    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bad int")
        return value

    store = _W(root_str, [_S(name=spec_name, path=spec_path, format=spec_format, owner=spec_owner, validate=_v)])
    for _ in range(rounds):
        with store.transaction():
            seen = store.read(spec_name)
            current = seen.value if seen.status == "ok" else 0
            store.write(spec_name, current + 1)


def _holder_worker(root_str, spec_name, spec_path, spec_format, spec_owner, ready_evt, release_evt):
    from kullback.store import ArtifactSpec as _S
    from kullback.store import WorkdirStore as _W

    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bad int")
        return value

    store = _W(root_str, [_S(name=spec_name, path=spec_path, format=spec_format, owner=spec_owner, validate=_v)])
    with store.transaction():
        ready_evt.set()
        release_evt.wait(timeout=60)


def _case_alias_worker(root_a, root_b, spec_name, spec_path, spec_format, spec_owner, done_evt):
    from kullback.store import ArtifactSpec as _S
    from kullback.store import WorkdirStore as _W

    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bad int")
        return value

    first = _W(root_a, [_S(name=spec_name, path=spec_path, format=spec_format, owner=spec_owner, validate=_v)])
    second = _W(root_b, [_S(name=spec_name, path=spec_path, format=spec_format, owner=spec_owner, validate=_v)])
    with first.transaction():
        with second.transaction():
            done_evt.set()


def _fresh_acquire_worker(root_str, spec_name, spec_path, spec_format, spec_owner, done_path, acquired_evt):
    from pathlib import Path as _P

    from kullback.store import ArtifactSpec as _S
    from kullback.store import WorkdirStore as _W

    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bad int")
        return value

    store = _W(root_str, [_S(name=spec_name, path=spec_path, format=spec_format, owner=spec_owner, validate=_v)])
    with store.transaction():
        _P(done_path).write_text("acquired", encoding="utf-8")
        acquired_evt.set()


def test_round_trip_envelope_and_missing(tmp_path):
    root = tmp_path / "w1"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_a", path="tally_a.json", format=2, owner="owner_a", validate=_check_dict_n)])
    seen = store.read("tally_a")
    assert seen.status == "missing"
    assert seen.value is None
    assert isinstance(seen, ReadResult)
    out = store.write("tally_a", {"n": 3})
    assert isinstance(out, Path)
    assert out.is_file()
    raw = out.read_text(encoding="utf-8")
    obj = json.loads(raw)
    assert obj == {"artifact": "tally_a", "format": 2, "owner": "owner_a", "store_format": 1, "value": {"n": 3}}
    assert list(json.loads(raw).keys()) == sorted(json.loads(raw).keys())
    assert raw.startswith("{\n")
    got = store.read("tally_a")
    assert got.status == "ok"
    assert got.value == 3
    assert got.legacy is False
    assert got.reason is None


def test_torn_cases(tmp_path):
    root = tmp_path / "w2"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_b", path="tally_b.json", format=1, owner="owner_b", validate=_check_int)])
    store.write("tally_b", 1)
    target = root / "tally_b.json"

    def _read_status():
        return store.read("tally_b")

    target.write_bytes(b'{"store_format": 1,')
    first = _read_status()
    assert first.status == "torn"
    assert first.value is None
    assert isinstance(first.reason, str) and first.reason.isidentifier()

    target.write_bytes(b"\xff\xfe\x00not text")
    second = _read_status()
    assert second.status == "torn"
    assert second.value is None
    assert isinstance(second.reason, str) and second.reason.isidentifier()

    bad_format = {"store_format": 1, "artifact": "tally_b", "format": 999, "owner": "owner_b", "value": 1}
    target.write_text(json.dumps(bad_format, sort_keys=True, indent=2), encoding="utf-8")
    third = _read_status()
    assert third.status == "torn"
    assert third.value is None

    bad_name = {"store_format": 1, "artifact": "other", "format": 1, "owner": "owner_b", "value": 1}
    target.write_text(json.dumps(bad_name, sort_keys=True, indent=2), encoding="utf-8")
    fourth = _read_status()
    assert fourth.status == "torn"
    assert fourth.value is None

    bad_owner = {"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "someone", "value": 1}
    target.write_text(json.dumps(bad_owner, sort_keys=True, indent=2), encoding="utf-8")
    fifth = _read_status()
    assert fifth.status == "torn"
    assert fifth.value is None

    extra = {"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "owner_b", "value": 1, "extra": 0}
    target.write_text(json.dumps(extra, sort_keys=True, indent=2), encoding="utf-8")
    sixth = _read_status()
    assert sixth.status == "torn"
    assert sixth.value is None

    bad_value = {"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "owner_b", "value": "nope"}
    target.write_text(json.dumps(bad_value, sort_keys=True, indent=2), encoding="utf-8")
    seventh = _read_status()
    assert seventh.status == "torn"
    assert seventh.value is None
    for result in (first, second, third, fourth, fifth, sixth, seventh):
        assert result.reason is not None
        assert result.legacy is False


def test_legacy_refused_then_accepted_without_migration(tmp_path):
    root = tmp_path / "w3"
    raw = {"n": 9}
    strict = WorkdirStore(root, [ArtifactSpec(name="tally_c", path="tally_c.json", format=1, owner="owner_c", validate=_check_dict_n)])
    target = root / "tally_c.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    before = json.dumps(raw, sort_keys=True, indent=2)
    target.write_text(before, encoding="utf-8")
    refused = strict.read("tally_c")
    assert refused.status == "torn"
    assert refused.legacy is False
    loose = WorkdirStore(root, [ArtifactSpec(name="tally_c", path="tally_c.json", format=1, owner="owner_c", validate=_check_dict_n, allow_legacy=True)])
    accepted = loose.read("tally_c")
    assert accepted.status == "ok"
    assert accepted.value == 9
    assert accepted.legacy is True
    after = target.read_text(encoding="utf-8")
    assert after == before


def test_unknown_duplicate_unsafe_and_symlink_escape(tmp_path):
    root = tmp_path / "w4"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_d", path="tally_d.json", format=1, owner="owner_d", validate=_check_int)])
    with pytest.raises(KeyError):
        store.read("absent")
    with pytest.raises(KeyError):
        store.write("absent", 1)
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="dup", path="a.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="dup", path="b.json", format=1, owner="o", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="x1", path="same.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="x2", path="same.json", format=1, owner="o", validate=_check_int)])
    for bad in ["", "/abs.json", "a/../b.json", "./b.json", "a/./b.json", ".", "..", "a/../../b.json"]:
        with pytest.raises(ValueError):
            WorkdirStore(root, [ArtifactSpec(name="bad", path=bad, format=1, owner="o", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="", path="ok.json", format=1, owner="o", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="ok", path="ok.json", format=1, owner="", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="ok", path="ok.json", format=0, owner="o", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="ok", path="ok.json", format=True, owner="o", validate=_check_int)])
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="ok", path="ok.json", format=1, owner="o", validate="notcallable")])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    link = root / "link"
    root.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink unavailable")
    esc = WorkdirStore(root, [ArtifactSpec(name="esc", path="link/evil.json", format=1, owner="o", validate=_check_int)])
    with pytest.raises(StoreError):
        esc.write("esc", 1)
    target = root / "link" / "evil.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"store_format": 1, "artifact": "esc", "format": 1, "owner": "o", "value": 1}), encoding="utf-8")
    with pytest.raises(StoreError):
        esc.read("esc")


def test_reserved_lock_path_rejected(tmp_path):
    root = tmp_path / "w_reserved"
    with pytest.raises(ValueError) as first:
        WorkdirStore(root, [ArtifactSpec(name="r1", path=".store.lock", format=1, owner="o", validate=_check_int)])
    assert str(first.value) == "reserved store path"
    for spelling in ["./.store.lock", "sub/../.store.lock", "a/./../.store.lock"]:
        with pytest.raises(ValueError) as other:
            WorkdirStore(root, [ArtifactSpec(name="r1", path=spelling, format=1, owner="o", validate=_check_int)])
        assert str(other.value) == "reserved store path"


def test_atomic_replace_failure_and_mode_preserved(tmp_path, monkeypatch):
    root = tmp_path / "w5"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_e", path="tally_e.json", format=1, owner="owner_e", validate=_check_int)])
    store.write("tally_e", 4)
    target = root / "tally_e.json"
    before = target.read_bytes()
    real_replace = os.replace

    def _boom(src, dst):
        raise OSError("injected")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        store.write("tally_e", 5)
    assert target.read_bytes() == before
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []
    monkeypatch.setattr(os, "replace", real_replace)
    real_dumps = json.dumps
    monkeypatch.setattr("kullback.store.json.dumps", lambda *a, **k: (_ for _ in ()).throw(ValueError("injected")))
    with pytest.raises(ValueError):
        store.write("tally_e", 6)
    assert target.read_bytes() == before
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []
    monkeypatch.setattr("kullback.store.json.dumps", real_dumps)
    os.chmod(target, 0o640)
    store.write("tally_e", 7)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o640
    assert store.read("tally_e").value == 7


def test_nested_transactions_two_instances(tmp_path):
    root = tmp_path / "w6"
    first = WorkdirStore(root, [ArtifactSpec(name="tally_f", path="tally_f.json", format=1, owner="owner_f", validate=_check_int)])
    second = WorkdirStore(root, [ArtifactSpec(name="tally_f", path="tally_f.json", format=1, owner="owner_f", validate=_check_int)])
    with first.transaction():
        with second.transaction():
            with first.transaction():
                first.write("tally_f", 11)
    assert first.read("tally_f").value == 11
    try:
        with first.transaction():
            with second.transaction():
                raise RuntimeError("injected")
    except RuntimeError:
        pass
    with second.transaction():
        second.write("tally_f", 12)
    assert first.read("tally_f").value == 12


def test_concurrent_increments_two_processes(tmp_path):
    root = tmp_path / "w7"
    name = "tally_g"
    rel = "tally_g.json"
    store = WorkdirStore(root, [ArtifactSpec(name=name, path=rel, format=1, owner="owner_g", validate=_check_int)])
    store.write(name, 0)
    ctx = multiprocessing.get_context("spawn")
    first = ctx.Process(target=_counter_worker, args=(str(root), name, rel, 1, "owner_g", 10))
    second = ctx.Process(target=_counter_worker, args=(str(root), name, rel, 1, "owner_g", 10))
    try:
        first.start()
        second.start()
        first.join(timeout=60)
        second.join(timeout=60)
    finally:
        for proc in (first, second):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    assert first.exitcode == 0
    assert second.exitcode == 0
    final = store.read(name)
    assert final.status == "ok"
    assert final.value == 20


def test_lock_survives_killed_holder(tmp_path):
    root = tmp_path / "w8"
    name = "tally_h"
    rel = "tally_h.json"
    done = tmp_path / "done.signal"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    acquired = ctx.Event()
    holder = ctx.Process(target=_holder_worker, args=(str(root), name, rel, 1, "owner_h", ready, release))
    try:
        holder.start()
        assert ready.wait(timeout=20)
        assert holder.is_alive()
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(timeout=20)
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=10)
    assert not holder.is_alive()
    fresh = ctx.Process(target=_fresh_acquire_worker, args=(str(root), name, rel, 1, "owner_h", str(done), acquired))
    try:
        fresh.start()
        assert acquired.wait(timeout=20)
        fresh.join(timeout=20)
    finally:
        if fresh.is_alive():
            fresh.terminate()
            fresh.join(timeout=10)
    assert fresh.exitcode == 0
    assert done.exists()


def test_symlink_alias_to_root_and_lock_refused(tmp_path):
    root = tmp_path / "w_alias"
    root.mkdir(parents=True, exist_ok=True)
    base = WorkdirStore(root, [ArtifactSpec(name="base_v", path="base_v.json", format=1, owner="owner_v", validate=_check_int)])
    base.write("base_v", 1)
    lock_path = root / ".store.lock"
    assert lock_path.is_file()
    before_inode = os.stat(lock_path).st_ino
    try:
        alias_lock = root / "alias_l.json"
        if alias_lock.is_symlink() or alias_lock.exists():
            alias_lock.unlink()
        alias_lock.symlink_to(lock_path)
        alias_root = root / "alias_r.json"
        if alias_root.is_symlink() or alias_root.exists():
            alias_root.unlink()
        alias_root.symlink_to(root)
    except OSError:
        pytest.skip("symlink unavailable")
    lock_alias = WorkdirStore(root, [ArtifactSpec(name="alias_l", path="alias_l.json", format=1, owner="owner_v", validate=_check_int)])
    with pytest.raises(StoreError):
        lock_alias.write("alias_l", 2)
    assert os.stat(lock_path).st_ino == before_inode
    with pytest.raises(StoreError):
        lock_alias.read("alias_l")
    assert os.stat(lock_path).st_ino == before_inode
    root_alias = WorkdirStore(root, [ArtifactSpec(name="alias_r", path="alias_r.json", format=1, owner="owner_v", validate=_check_int)])
    with pytest.raises(StoreError):
        root_alias.write("alias_r", 2)
    with pytest.raises(StoreError):
        root_alias.read("alias_r")
    assert os.stat(lock_path).st_ino == before_inode


def test_keyboard_interrupt_during_replace_cleans_staging(tmp_path, monkeypatch):
    root = tmp_path / "w_ki"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_ki", path="tally_ki.json", format=1, owner="owner_ki", validate=_check_int)])
    store.write("tally_ki", 4)
    target = root / "tally_ki.json"
    before = target.read_bytes()

    def _kboom(src, dst):
        raise KeyboardInterrupt()

    real_replace = os.replace
    monkeypatch.setattr(os, "replace", _kboom)
    with pytest.raises(KeyboardInterrupt):
        store.write("tally_ki", 5)
    assert target.read_bytes() == before
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []
    monkeypatch.setattr(os, "replace", real_replace)
    store.write("tally_ki", 6)
    assert store.read("tally_ki").value == 6


def test_fchmod_failure_closes_descriptor(tmp_path, monkeypatch):
    root = tmp_path / "w_fchmod"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_fc", path="tally_fc.json", format=1, owner="owner_fc", validate=_check_int)])
    store.write("tally_fc", 1)
    seen = {}

    real_mkstemp = tempfile.mkstemp

    def _capture(*args, **kwargs):
        fd, tmp = real_mkstemp(*args, **kwargs)
        seen["fd"] = fd
        seen["tmp"] = tmp
        return fd, tmp

    def _fboom(fd, mode):
        raise OSError("injected")

    monkeypatch.setattr(tempfile, "mkstemp", _capture)
    monkeypatch.setattr(os, "fchmod", _fboom)
    with pytest.raises(OSError):
        store.write("tally_fc", 2)
    with pytest.raises(OSError) as exc:
        os.fstat(seen["fd"])
    assert exc.value.errno == errno.EBADF


def test_flock_interrupt_then_thread_recovers(tmp_path, monkeypatch):
    import fcntl as _fcntl

    root = tmp_path / "w_flock"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_fk", path="tally_fk.json", format=1, owner="owner_fk", validate=_check_int)])
    real_flock = _fcntl.flock
    once = {"armed": True}

    def _once(fd, op):
        if once["armed"] and op == _fcntl.LOCK_EX:
            once["armed"] = False
            raise KeyboardInterrupt()
        return real_flock(fd, op)

    monkeypatch.setattr(_fcntl, "flock", _once)
    with pytest.raises(KeyboardInterrupt):
        with store.transaction():
            pass
    monkeypatch.setattr(_fcntl, "flock", real_flock)
    done = threading.Event()

    def _work():
        with store.transaction():
            done.set()

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    assert done.wait(timeout=20)
    worker.join(timeout=20)


def test_strict_json_constants_and_duplicate_keys(tmp_path):
    root = tmp_path / "w_strict"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_s", path="tally_s.json", format=1, owner="owner_s", validate=_identity)])
    target = root / "tally_s.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    for token in ["NaN", "Infinity", "-Infinity"]:
        target.write_text("{" + "\"store_format\": 1, \"artifact\": \"tally_s\", \"format\": 1, \"owner\": \"owner_s\", \"value\": " + token + "}", encoding="utf-8")
        got = store.read("tally_s")
        assert got.status == "torn"
        assert got.value is None
    target.write_text("{" + "\"store_format\": 1, \"store_format\": 1, \"artifact\": \"tally_s\", \"format\": 1, \"owner\": \"owner_s\", \"value\": 1}", encoding="utf-8")
    dup_envelope = store.read("tally_s")
    assert dup_envelope.status == "torn"
    target.write_text("{" + "\"store_format\": 1, \"artifact\": \"tally_s\", \"format\": 1, \"owner\": \"owner_s\", \"value\": {\"n\": 1, \"n\": 2}}", encoding="utf-8")
    dup_value = store.read("tally_s")
    assert dup_value.status == "torn"


def test_io_and_root_errors(tmp_path, monkeypatch):
    root = tmp_path / "w_io"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_io", path="tally_io.json", format=1, owner="owner_io", validate=_check_int)])
    store.write("tally_io", 1)

    def _denied(self):
        raise PermissionError("injected")

    monkeypatch.setattr(Path, "read_bytes", _denied)
    with pytest.raises(PermissionError):
        store.read("tally_io")
    monkeypatch.undo()

    def _osfail(value):
        raise OSError("injected")

    target = root / "tally_io.json"
    target.write_text(json.dumps({"store_format": 1, "artifact": "tally_io", "format": 1, "owner": "owner_io", "value": 1}, sort_keys=True, indent=2), encoding="utf-8")
    failing = WorkdirStore(root, [ArtifactSpec(name="tally_io", path="tally_io.json", format=1, owner="owner_io", validate=_osfail)])
    with pytest.raises(OSError):
        failing.read("tally_io")
    reg = tmp_path / "regfile"
    reg.write_text("x", encoding="utf-8")
    bad_root = WorkdirStore(reg, [ArtifactSpec(name="tally_io", path="tally_io.json", format=1, owner="owner_io", validate=_check_int)])
    with pytest.raises(NotADirectoryError):
        bad_root.read("tally_io")
    absent = tmp_path / "does_not_exist"
    missing_store = WorkdirStore(absent, [ArtifactSpec(name="tally_io", path="tally_io.json", format=1, owner="owner_io", validate=_check_int)])
    got = missing_store.read("tally_io")
    assert got.status == "missing"
    assert not absent.exists()


def test_hardlink_alias_to_lock_refused(tmp_path):
    root = tmp_path / "w_hard"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_hd", path="tally_hd.json", format=1, owner="owner_hd", validate=_check_int)])
    store.write("tally_hd", 1)
    lock_path = root / ".store.lock"
    assert lock_path.is_file()
    before = os.stat(lock_path).st_ino
    alias_path = root / "alias_hd.json"
    try:
        if alias_path.is_symlink() or alias_path.exists():
            alias_path.unlink()
        os.link(lock_path, alias_path)
    except OSError:
        pytest.skip("hardlink unavailable")
    alias_store = WorkdirStore(root, [ArtifactSpec(name="alias_hd", path="alias_hd.json", format=1, owner="owner_hd", validate=_check_int)])
    with pytest.raises(StoreError):
        alias_store.write("alias_hd", 2)
    assert os.stat(lock_path).st_ino == before
    with pytest.raises(StoreError):
        alias_store.read("alias_hd")
    assert os.stat(lock_path).st_ino == before


def test_case_insensitive_lock_alias_refused(tmp_path):
    root = tmp_path / "w_cilock"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_ci", path="tally_ci.json", format=1, owner="owner_ci", validate=_check_int)])
    store.write("tally_ci", 1)
    lock_path = root / ".store.lock"
    upper_path = root / ".STORE.LOCK"
    assert lock_path.is_file()
    if not upper_path.exists():
        pytest.skip("case-sensitive filesystem")
    try:
        same = upper_path.samefile(lock_path)
    except OSError:
        pytest.skip("case-sensitive filesystem")
    if not same:
        pytest.skip("case-sensitive filesystem")
    before = os.stat(lock_path).st_ino
    alias_store = WorkdirStore(root, [ArtifactSpec(name="upper_v", path=".STORE.LOCK", format=1, owner="owner_ci", validate=_check_int)])
    with pytest.raises(StoreError):
        alias_store.write("upper_v", 2)
    assert os.stat(lock_path).st_ino == before
    with pytest.raises(StoreError):
        alias_store.read("upper_v")
    assert os.stat(lock_path).st_ino == before


def test_nested_case_alias_transaction(tmp_path):
    base = tmp_path / "w_cinest"
    lower = base / "nestdir"
    lower.mkdir(parents=True, exist_ok=True)
    upper = base / "NESTDIR"
    try:
        same = lower.samefile(upper)
    except OSError:
        pytest.skip("case-sensitive filesystem")
    if not same:
        pytest.skip("case-sensitive filesystem")
    ctx = multiprocessing.get_context("spawn")
    done = ctx.Event()
    child = ctx.Process(target=_case_alias_worker, args=(str(lower), str(upper), "tally_cn", "tally_cn.json", 1, "owner_cn", done))
    try:
        child.start()
        assert done.wait(timeout=20)
        child.join(timeout=20)
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
    assert child.exitcode == 0


def test_numeric_overflow_torn_and_finite_ok(tmp_path):
    root = tmp_path / "w_overflow"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_ov", path="tally_ov.json", format=1, owner="owner_ov", validate=_identity)])
    target = root / "tally_ov.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    for token in ["1e9999", "-1e9999"]:
        target.write_text("{" + "\"store_format\": 1, \"artifact\": \"tally_ov\", \"format\": 1, \"owner\": \"owner_ov\", \"value\": " + token + "}", encoding="utf-8")
        got = store.read("tally_ov")
        assert got.status == "torn"
        assert got.value is None
    target.write_text("{" + "\"store_format\": 1, \"artifact\": \"tally_ov\", \"format\": 1, \"owner\": \"owner_ov\", \"value\": 1e2}", encoding="utf-8")
    ok = store.read("tally_ov")
    assert ok.status == "ok"
    assert ok.value == 100.0


def test_constructor_casefold_duplicate_without_files(tmp_path):
    root = tmp_path / "w_case_decl"
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="a1", path="data.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="a2", path="DATA.json", format=1, owner="o", validate=_check_int)])


def test_constructor_physical_alias_symlink(tmp_path):
    root = tmp_path / "w_phys_link"
    root.mkdir(parents=True, exist_ok=True)
    real = root / "real.json"
    real.write_text("{}", encoding="utf-8")
    alias = root / "alias.json"
    try:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(real)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="pa", path="real.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="pb", path="alias.json", format=1, owner="o", validate=_check_int)])


def test_constructor_physical_alias_hardlink(tmp_path):
    root = tmp_path / "w_phys_hard"
    root.mkdir(parents=True, exist_ok=True)
    real = root / "real2.json"
    real.write_text("{}", encoding="utf-8")
    alias = root / "alias2.json"
    try:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        os.link(real, alias)
    except OSError:
        pytest.skip("hardlink unavailable")
    with pytest.raises(ValueError):
        WorkdirStore(root, [ArtifactSpec(name="ha", path="real2.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="hb", path="alias2.json", format=1, owner="o", validate=_check_int)])


def test_post_construction_symlink_alias_owner_conflict(tmp_path):
    root = tmp_path / "w_late_alias"
    first = WorkdirStore(root, [ArtifactSpec(name="alpha", path="alpha.json", format=1, owner="owner_a", validate=_check_int)])
    second = WorkdirStore(root, [ArtifactSpec(name="beta", path="beta.json", format=1, owner="owner_b", validate=_check_int)])
    first.write("alpha", 1)
    second.write("beta", 2)
    alpha_path = root / "alpha.json"
    beta_path = root / "beta.json"
    before = alpha_path.read_bytes()
    try:
        beta_path.unlink()
        beta_path.symlink_to(alpha_path)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(StoreError):
        second.write("beta", 3)
    assert alpha_path.read_bytes() == before
    first.write("alpha", 4)
    assert first.read("alpha").value == 4


def test_write_durability_directory_open_failure(tmp_path, monkeypatch):
    from kullback.store import WriteDurabilityError
    root = tmp_path / "w_duropen"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_do", path="tally_do.json", format=1, owner="owner_do", validate=_check_int)])
    store.write("tally_do", 1)
    target = root / "tally_do.json"
    parent = target.parent
    real_open = os.open

    def _open_fail(path, flags, *args, **kwargs):
        if isinstance(path, str) and path == str(parent) and (flags & os.O_CREAT) == 0:
            raise OSError("injected dir open")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _open_fail)
    with pytest.raises(WriteDurabilityError) as excinfo:
        store.write("tally_do", 2)
    assert excinfo.value.replaced is True
    assert excinfo.value.durability == "unknown"
    assert Path(excinfo.value.path).name == target.name
    raw = Path(excinfo.value.path).read_bytes()
    assert json.loads(raw)["value"] == 2
    leftovers = [p for p in parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []


def test_write_durability_directory_fsync_failure(tmp_path, monkeypatch):
    from kullback.store import WriteDurabilityError
    root = tmp_path / "w_durfsync"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_df", path="tally_df.json", format=1, owner="owner_df", validate=_check_int)])
    store.write("tally_df", 1)
    target = root / "tally_df.json"
    parent = target.parent
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
        store.write("tally_df", 2)
    assert excinfo.value.replaced is True
    assert excinfo.value.durability == "unknown"
    assert Path(excinfo.value.path).name == target.name
    raw = Path(excinfo.value.path).read_bytes()
    assert json.loads(raw)["value"] == 2
    leftovers = [p for p in parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []


def test_lock_registry_reclaimed(tmp_path):
    import gc

    from kullback import store as _mod
    before = len(_mod._LOCKS)
    for idx in range(50):
        fresh = tmp_path / str("reclaim_" + str(idx))
        inst = WorkdirStore(fresh, [ArtifactSpec(name="tally_lr", path="tally_lr.json", format=1, owner="owner_lr", validate=_check_int)])
        with inst.transaction():
            pass
    gc.collect()
    assert len(_mod._LOCKS) == before


def test_lock_state_shared_with_waiter(tmp_path):
    from kullback import store as _mod
    root = tmp_path / "w_waiter"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_wt", path="tally_wt.json", format=1, owner="owner_wt", validate=_check_int)])
    store.write("tally_wt", 1)
    identity = Path(store._root_str).stat()
    key = (os.getpid(), identity.st_dev, identity.st_ino)
    seen = {}
    ready = threading.Event()
    done = threading.Event()

    def _waiter(held_state):
        current = _mod._state_for(key)
        seen["same"] = current is held_state
        ready.set()
        got = current.rlock.acquire(timeout=20)
        try:
            if got:
                seen["acquired"] = True
        finally:
            if got:
                current.rlock.release()
        done.set()

    with store.transaction():
        held = _mod._state_for(key)
        worker = threading.Thread(target=_waiter, args=(held,), daemon=True)
        worker.start()
        assert ready.wait(timeout=20)
        assert seen["same"] is True
    assert done.wait(timeout=20)
    worker.join(timeout=20)
    assert seen.get("acquired") is True


def _trace_calls(monkeypatch, calls):
    real_open = os.open
    real_close = os.close
    real_fsync = os.fsync
    real_replace = os.replace
    paths = {}

    def _recording_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        try:
            paths[fd] = os.path.realpath(path)
        except (OSError, TypeError, ValueError):
            paths[fd] = "?"
        return fd

    def _recording_close(fd):
        try:
            return real_close(fd)
        finally:
            paths.pop(fd, None)

    def _recording_fsync(fd):
        try:
            mode = os.fstat(fd).st_mode
        except OSError:
            return real_fsync(fd)
        if stat.S_ISDIR(mode):
            ftype = "dir"
        elif stat.S_ISREG(mode):
            ftype = "file"
        else:
            ftype = "other"
        calls.append((ftype, paths.get(fd, "?")))
        return real_fsync(fd)

    def _recording_replace(src, dst):
        calls.append(("replace", str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "open", _recording_open)
    monkeypatch.setattr(os, "close", _recording_close)
    monkeypatch.setattr(os, "fsync", _recording_fsync)
    monkeypatch.setattr(os, "replace", _recording_replace)


def _dir_fsyncs(calls):
    return [path for ftype, path in calls if ftype == "dir"]


def _fail_dir_at(monkeypatch, victim):
    real_open = os.open
    real_close = os.close
    real_fsync = os.fsync
    paths = {}

    def _recording_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        try:
            paths[fd] = os.path.realpath(path)
        except (OSError, TypeError, ValueError):
            paths[fd] = "?"
        return fd

    def _recording_close(fd):
        try:
            return real_close(fd)
        finally:
            paths.pop(fd, None)

    def _failing(fd):
        try:
            mode = os.fstat(fd).st_mode
        except OSError:
            return real_fsync(fd)
        if stat.S_ISDIR(mode) and paths.get(fd, "?") == victim:
            raise PermissionError("injected")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", _recording_open)
    monkeypatch.setattr(os, "close", _recording_close)
    monkeypatch.setattr(os, "fsync", _failing)


def _ancestor_creator_worker(root_str, name, rel, ready_evt, go_evt):
    from kullback.store import ArtifactSpec as _S
    from kullback.store import WorkdirStore as _W

    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bad int")
        return value

    ready_evt.set()
    if not go_evt.wait(timeout=60):
        raise RuntimeError("release timeout")
    store = _W(root_str, [_S(name=name, path=rel, format=1, owner="o", validate=_v)])
    store.write(name, 1)


def test_ancestor_chain_existing_root_new_subdir(tmp_path, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    store = WorkdirStore(root, [ArtifactSpec(name="sub_a", path="sub/f.json", format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("sub_a", 1)
    assert store.read("sub_a").value == 1
    dirs = _dir_fsyncs(calls)
    sub = str((root / "sub").resolve())
    work = str(root.resolve())
    assert sub in dirs
    assert work in dirs
    assert dirs.index(sub) < dirs.index(work)
    first_replace = next(i for i, entry in enumerate(calls) if entry[0] == "replace")
    first_dir = next(i for i, entry in enumerate(calls) if entry[0] == "dir")
    assert first_replace < first_dir


def test_ancestor_chain_absent_workdir(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "work"
    store = WorkdirStore(root, [ArtifactSpec(name="first_b", path="round-journal/00000000000000000001.json", format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("first_b", 1)
    assert store.read("first_b").value == 1
    dirs = _dir_fsyncs(calls)
    assert str(root.resolve()) in dirs
    assert str(parent.resolve()) in dirs


def test_ancestor_chain_two_missing_levels(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    root = base / "mid" / "work"
    store = WorkdirStore(root, [ArtifactSpec(name="two_c", path="f.json", format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("two_c", 1)
    assert store.read("two_c").value == 1
    dirs = _dir_fsyncs(calls)
    assert str((base / "mid").resolve()) in dirs
    assert str(root.resolve()) in dirs
    assert str(base.resolve()) in dirs


def test_ancestor_chain_deep_bottom_up_no_duplicate(tmp_path, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    store = WorkdirStore(root, [ArtifactSpec(name="deep_d", path="a/b/c.json", format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("deep_d", 1)
    assert store.read("deep_d").value == 1
    dirs = _dir_fsyncs(calls)
    inner = str((root / "a" / "b").resolve())
    mid = str((root / "a").resolve())
    work = str(root.resolve())
    assert inner in dirs and mid in dirs and work in dirs
    assert dirs.index(inner) < dirs.index(mid) < dirs.index(work)
    assert dirs.count(inner) == 1
    first_replace = next(i for i, entry in enumerate(calls) if entry[0] == "replace")
    first_dir = next(i for i, entry in enumerate(calls) if entry[0] == "dir")
    assert first_replace < first_dir


def test_ancestor_chain_top_level_syncs_parent(tmp_path, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    store = WorkdirStore(root, [ArtifactSpec(name="top_e", path="top.json", format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("top_e", 1)
    assert store.read("top_e").value == 1
    dirs = _dir_fsyncs(calls)
    assert str(root.resolve()) in dirs
    assert str(tmp_path.resolve()) in dirs


def test_ancestor_boundary_device_change(tmp_path, monkeypatch):
    root = tmp_path / "w"
    store = WorkdirStore(root, [ArtifactSpec(name="bound_f", path="sub/f.json", format=1, owner="o", validate=_check_int)])
    store.write("bound_f", 1)
    base = tmp_path.resolve()
    real_stat = Path.stat

    def _patched(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        each = Path(self)
        if each != base and base not in each.parents:
            vals = list(result)
            vals[2] = result.st_dev + 1
            return os.stat_result(tuple(vals))
        return result

    monkeypatch.setattr(Path, "stat", _patched)
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("bound_f", 2)
    assert store.read("bound_f").value == 2
    dirs = _dir_fsyncs(calls)
    assert str(base) in dirs
    for raw in dirs:
        assert base == Path(raw) or base in Path(raw).parents


def test_ancestor_boundary_filesystem_root(tmp_path, monkeypatch):
    import kullback.store as store_module

    recorded = []
    real_sync = store_module._sync_directory

    def _recording(path):
        recorded.append(str(Path(path)))
        return real_sync(path)

    monkeypatch.setattr(store_module, "_sync_directory", _recording)
    root = tmp_path / "w"
    store = WorkdirStore(root, [ArtifactSpec(name="top_g", path="top.json", format=1, owner="o", validate=_check_int)])
    store.write("top_g", 1)
    assert store.read("top_g").value == 1
    assert recorded[0] == str((root / "top.json").resolve().parent)
    assert len(set(recorded)) == len(recorded)
    for first, second in zip(recorded, recorded[1:], strict=False):
        assert Path(second) == Path(first).parent
    final = Path(recorded[-1])
    if final.parent != final:
        assert final.parent.stat().st_dev != final.stat().st_dev


def test_ancestor_sync_failure_permission(tmp_path, monkeypatch):
    from kullback.store import WriteDurabilityError

    root = tmp_path / "w"
    store = WorkdirStore(root, [ArtifactSpec(name="keep_h", path="keep.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="deep_h", path="a/b/c.json", format=1, owner="o", validate=_check_int)])
    store.write("keep_h", 1)
    store.write("deep_h", 1)
    before = (root / "keep.json").read_bytes()
    _fail_dir_at(monkeypatch, str(root.resolve()))
    with pytest.raises(WriteDurabilityError) as excinfo:
        store.write("deep_h", 2)
    assert excinfo.value.replaced is True
    assert excinfo.value.durability == "unknown"
    assert isinstance(excinfo.value.__cause__, PermissionError)
    raw = json.loads((root / "a" / "b" / "c.json").read_text(encoding="utf-8"))
    assert raw["value"] == 2
    assert (root / "keep.json").read_bytes() == before


def test_ancestor_retry_resyncs_existing_dirs(tmp_path, monkeypatch):
    from kullback.store import WriteDurabilityError

    root = tmp_path / "w"
    store = WorkdirStore(root, [ArtifactSpec(name="deep_i", path="a/b/c.json", format=1, owner="o", validate=_check_int)])
    store.write("deep_i", 1)
    _fail_dir_at(monkeypatch, str(root.resolve()))
    with pytest.raises(WriteDurabilityError):
        store.write("deep_i", 2)
    monkeypatch.undo()
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("deep_i", 3)
    assert store.read("deep_i").value == 3
    dirs = _dir_fsyncs(calls)
    assert str((root / "a" / "b").resolve()) in dirs
    assert str((root / "a").resolve()) in dirs
    assert str(root.resolve()) in dirs


def test_ancestor_cooperative_creators_same_workdir(tmp_path):
    root = tmp_path / "race"
    ctx = multiprocessing.get_context("spawn")
    ready_first = ctx.Event()
    ready_second = ctx.Event()
    go = ctx.Event()
    first = ctx.Process(target=_ancestor_creator_worker, args=(str(root), "r1", "r1.json", ready_first, go))
    second = ctx.Process(target=_ancestor_creator_worker, args=(str(root), "r2", "r2.json", ready_second, go))
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
    check = WorkdirStore(root, [ArtifactSpec(name="r1", path="r1.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="r2", path="r2.json", format=1, owner="o", validate=_check_int)])
    assert check.read("r1").status == "ok"
    assert check.read("r1").value == 1
    assert check.read("r2").status == "ok"
    assert check.read("r2").value == 1


def test_ancestor_root_equality_no_stat_no_sync(monkeypatch):
    import kullback.store as store_module

    def _no_stat(self, *args, **kwargs):
        raise AssertionError("stat must not be called")

    def _no_sync(path):
        raise AssertionError("sync must not be called")

    monkeypatch.setattr(Path, "stat", _no_stat)
    monkeypatch.setattr(store_module, "_sync_directory", _no_sync)
    store_module._sync_ancestors(Path("/probe.json"))


def _session_spec():
    return ArtifactSpec(name="n", path="n.json", format=1, owner="o", validate=_check_int)


def test_session_acquire_use_release(tmp_path):
    from kullback.store import SessionBusy as _Busy
    del _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    with store.exclusive_session():
        with store.transaction():
            store.write("n", 1)
            assert store.read("n").value == 1
        with store.transaction():
            with store.transaction():
                store.write("n", 2)
    assert store.read("n").value == 2
    with store.exclusive_session():
        assert store.read("n").value == 2


def test_session_second_acquire_same_thread_refuses(tmp_path):
    from kullback.store import SessionBusy as _Busy
    store = WorkdirStore(tmp_path / "w", [_session_spec()])
    with store.exclusive_session():
        with pytest.raises(_Busy):
            with store.exclusive_session():
                pass
    with store.exclusive_session():
        pass


def test_session_other_thread_acquire_and_transact(tmp_path):
    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    outcomes = []
    entered = threading.Event()

    def _other():
        other = WorkdirStore(root, [_session_spec()])
        entered.set()
        try:
            with other.exclusive_session():
                outcomes.append("acquired")
        except _Busy:
            outcomes.append("busy")
        with other.transaction():
            other.write("n", 7)
        outcomes.append("wrote")

    with store.exclusive_session():
        worker = threading.Thread(target=_other)
        worker.start()
        assert entered.wait(timeout=60)
        worker.join(timeout=60)
        assert outcomes == ["busy", "wrote"]
        assert store.read("n").value == 7


def test_session_inside_transaction_refuses(tmp_path):
    from kullback.store import SessionBusy as _Busy
    store = WorkdirStore(tmp_path / "w", [_session_spec()])
    with store.transaction():
        with pytest.raises(_Busy):
            with store.exclusive_session():
                pass
    with store.exclusive_session():
        pass


def test_session_while_other_thread_transacts_refuses(tmp_path):
    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    entered = threading.Event()
    release = threading.Event()

    def _holder():
        other = WorkdirStore(root, [_session_spec()])
        with other.transaction():
            entered.set()
            assert release.wait(timeout=60)

    worker = threading.Thread(target=_holder)
    worker.start()
    try:
        assert entered.wait(timeout=60)
        with pytest.raises(_Busy):
            with store.exclusive_session():
                pass
    finally:
        release.set()
        worker.join(timeout=60)
    with store.exclusive_session():
        pass


def _pipe_read_line(fd, timeout=20):
    import select
    out = b""
    while not out.endswith(b"\n"):
        ready, _, _ = select.select([fd], [], [], timeout)
        assert ready, "pipe read timeout"
        chunk = os.read(fd, 1)
        assert chunk, "pipe eof"
        out += chunk
    return out


def _fork_session_child(go_r, go_w, out_r, out_w, root_str, mode):
    pid = os.fork()
    if pid != 0:
        return pid
    try:
        os.close(go_w)
        os.close(out_r)
        from kullback.store import SessionBusy as _Busy
        from kullback.store import WorkdirStore as _W
        assert os.read(go_r, 1) == b"g"
        if mode == "attempt":
            os.close(go_r)
            other = _W(root_str, [_session_spec()])
            try:
                with other.exclusive_session():
                    os.write(out_w, b"acquired\n")
            except _Busy:
                os.write(out_w, b"busy\n")
        elif mode == "hold":
            other = _W(root_str, [_session_spec()])
            with other.exclusive_session():
                os.write(out_w, b"ready\n")
                second = os.read(go_r, 1)
                assert second == b"r"
                os.write(out_w, b"done\n")
            os.close(go_r)
        os.close(out_w)
    except BaseException:
        try:
            os.close(go_r)
        except OSError:
            pass
        try:
            os.close(out_w)
        except OSError:
            pass
        os._exit(1)
    os._exit(0)


def _read_message(fd):
    return _pipe_read_line(fd)


def test_session_second_process_refuses(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    go_r, go_w = os.pipe()
    out_r, out_w = os.pipe()
    pid = _fork_session_child(go_r, go_w, out_r, out_w, str(root), "attempt")
    os.close(go_r)
    os.close(out_w)
    try:
        with store.exclusive_session():
            os.write(go_w, b"g")
            assert _read_message(out_r) == b"busy\n"
    finally:
        os.close(go_w)
        os.close(out_r)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_session_killed_holder_releases(tmp_path):
    import signal
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    go_r, go_w = os.pipe()
    out_r, out_w = os.pipe()
    pid = _fork_session_child(go_r, go_w, out_r, out_w, str(root), "hold")
    os.close(go_r)
    os.close(out_w)
    try:
        os.write(go_w, b"g")
        assert _pipe_read_line(out_r) == b"ready\n"
        os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGKILL
        with store.exclusive_session():
            pass
    finally:
        os.close(go_w)
        os.close(out_r)


def test_session_alias_root_refuses(tmp_path):
    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    alias = WorkdirStore(root / "sub" / "..", [_session_spec()])
    with store.exclusive_session():
        with pytest.raises(_Busy):
            with alias.exclusive_session():
                pass
        with alias.transaction():
            alias.write("n", 3)
    assert store.read("n").value == 3


def test_session_exit_waits_for_open_transaction(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    from kullback.store import SessionBusy as _Busy
    def _once():
        log = []
        entered = threading.Event()
        release = threading.Event()
        in_txn = threading.Event()
        done = threading.Event()
        driver_done = threading.Event()
        worker_closed = threading.Event()
        def _driver():
            with store.exclusive_session():
                entered.set()
                assert release.wait(timeout=60)
            log.append("session-released")
            driver_done.set()
        def _worker():
            other = WorkdirStore(root, [_session_spec()])
            with other.transaction():
                in_txn.set()
                assert done.wait(timeout=60)
            log.append("txn-closed")
            worker_closed.set()
        driver = threading.Thread(target=_driver)
        driver.start()
        assert entered.wait(timeout=60)
        worker = threading.Thread(target=_worker)
        worker.start()
        try:
            assert in_txn.wait(timeout=60)
            release.set()
            with pytest.raises(_Busy):
                with store.exclusive_session():
                    pass
            assert not driver_done.is_set()
        finally:
            done.set()
            worker.join(timeout=60)
            driver.join(timeout=60)
        assert not driver.is_alive()
        assert not worker.is_alive()
        assert worker_closed.is_set()
        assert driver_done.is_set()
        assert sorted(log) == ["session-released", "txn-closed"]
        with store.exclusive_session():
            pass
    for _rep in range(3):
        _once()


def test_session_exception_paths(tmp_path):
    from kullback.store import SessionBusy as _Busy
    store = WorkdirStore(tmp_path / "w", [_session_spec()])
    with pytest.raises(RuntimeError, match="boom"):
        with store.exclusive_session():
            raise RuntimeError("boom")
    with store.exclusive_session():
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.write("n", 1)
                raise RuntimeError("inner")
        with pytest.raises(_Busy):
            with store.exclusive_session():
                pass
    assert store.read("n").value == 1
    with store.exclusive_session():
        pass


def test_session_acquire_interrupt_cleans_fd(tmp_path, monkeypatch):
    import fcntl as _fcntl

    from kullback import store as _mod
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    real_flock = _fcntl.flock
    once = {"armed": True}
    def _once(fd, op):
        if once["armed"] and op == (_fcntl.LOCK_EX | _fcntl.LOCK_NB):
            once["armed"] = False
            raise KeyboardInterrupt()
        return real_flock(fd, op)
    monkeypatch.setattr(_fcntl, "flock", _once)
    with pytest.raises(KeyboardInterrupt):
        with store.exclusive_session():
            pass
    monkeypatch.setattr(_fcntl, "flock", real_flock)
    assert _mod._OPEN_FDS == set()
    with store.exclusive_session():
        with store.transaction():
            store.write("n", 1)
    assert store.read("n").value == 1
    assert _mod._OPEN_FDS == set()


def test_session_survives_store_drop(tmp_path):
    import gc

    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    holder = store.exclusive_session()
    holder.__enter__()
    try:
        del store
        gc.collect()
        probe = WorkdirStore(root, [_session_spec()])
        with pytest.raises(_Busy):
            with probe.exclusive_session():
                pass
        with probe.transaction():
            probe.write("n", 5)
        assert probe.read("n").value == 5
    finally:
        holder.__exit__(None, None, None)
    probe = WorkdirStore(root, [_session_spec()])
    with probe.exclusive_session():
        assert probe.read("n").value == 5


def _read_line(fd):
    return _pipe_read_line(fd)


def test_session_close_entrant_completes(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    entered = threading.Event()
    release = threading.Event()
    in_txn = threading.Event()
    worker_done = threading.Event()
    entrant_done = threading.Event()
    log = []
    driver_closed = threading.Event()
    worker_closed = threading.Event()
    def _driver():
        with store.exclusive_session():
            entered.set()
            assert release.wait(timeout=20)
        log.append("session-released")
        driver_closed.set()
    def _worker():
        other = WorkdirStore(root, [_session_spec()])
        with other.transaction():
            in_txn.set()
            assert worker_done.wait(timeout=20)
        log.append("txn-closed")
        worker_closed.set()
    def _entrant():
        other = WorkdirStore(root, [_session_spec()])
        with other.transaction():
            other.write("n", 11)
        entrant_done.set()
    driver = threading.Thread(target=_driver)
    driver.start()
    assert entered.wait(timeout=20)
    worker = threading.Thread(target=_worker)
    worker.start()
    assert in_txn.wait(timeout=20)
    entrant = threading.Thread(target=_entrant)
    entrant.start()
    release.set()
    go_r, go_w = os.pipe()
    out_r, out_w = os.pipe()
    check = os.fork()
    if check == 0:
        try:
            os.close(go_w)
            os.close(out_r)
            from kullback.store import SessionBusy as _Busy
            from kullback.store import WorkdirStore as _W
            assert os.read(go_r, 1) == b"g"
            os.close(go_r)
            probe = _W(str(root), [_session_spec()])
            try:
                with probe.exclusive_session():
                    os.write(out_w, b"acquired\n")
            except _Busy:
                os.write(out_w, b"busy\n")
            os.close(out_w)
        except BaseException:
            try:
                os.close(go_r)
            except OSError:
                pass
            try:
                os.close(out_w)
            except OSError:
                pass
            os._exit(1)
        os._exit(0)
    os.close(go_r)
    os.close(out_w)
    try:
        os.write(go_w, b"g")
        assert _pipe_read_line(out_r) == b"busy\n"
    except BaseException:
        try:
            import signal as _sig
            os.kill(check, _sig.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(check, 0)
        except ChildProcessError:
            pass
        raise
    finally:
        os.close(go_w)
        os.close(out_r)
    _, check_status = os.waitpid(check, 0)
    assert os.waitstatus_to_exitcode(check_status) == 0
    assert not driver_closed.is_set()
    worker_done.set()
    worker.join(timeout=20)
    entrant.join(timeout=20)
    driver.join(timeout=20)
    assert not worker.is_alive()
    assert not entrant.is_alive()
    assert not driver.is_alive()
    assert entrant_done.is_set()
    assert worker_closed.is_set()
    assert driver_closed.is_set()
    assert sorted(log) == ["session-released", "txn-closed"]
    assert store.read("n").value == 11
    with store.exclusive_session():
        pass


def test_session_fork_while_thread_holds(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    entered = threading.Event()
    release = threading.Event()
    def _holder():
        with store.exclusive_session():
            entered.set()
            assert release.wait(timeout=20)
    holder = threading.Thread(target=_holder)
    holder.start()
    assert entered.wait(timeout=20)
    go_r, go_w = os.pipe()
    out_r, out_w = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(go_w)
            os.close(out_r)
            from kullback.store import SessionBusy as _Busy
            from kullback.store import WorkdirStore as _W
            assert os.read(go_r, 1) == b"g"
            probe = _W(str(root), [_session_spec()])
            try:
                with probe.exclusive_session():
                    os.write(out_w, b"acquired\n")
            except _Busy:
                os.write(out_w, b"busy\n")
            assert os.read(go_r, 1) == b"r"
            os.close(go_r)
            os.close(out_w)
        except BaseException:
            try:
                os.close(go_r)
            except OSError:
                pass
            try:
                os.close(out_w)
            except OSError:
                pass
            os._exit(1)
        os._exit(0)
    os.close(go_r)
    os.close(out_w)
    try:
        os.write(go_w, b"g")
        assert _pipe_read_line(out_r) == b"busy\n"
        from kullback.store import SessionBusy as _Busy
        with store.transaction():
            store.write("n", 21)
        try:
            with store.exclusive_session():
                raise AssertionError("held")
        except _Busy:
            pass
        release.set()
        holder.join(timeout=20)
        assert not holder.is_alive()
        with store.exclusive_session():
            assert store.read("n").value == 21
        os.write(go_w, b"r")
    except BaseException:
        try:
            import signal as _sig
            os.kill(child, _sig.SIGKILL)
        except OSError:
            pass
        try:
            os.write(go_w, b"r")
        except OSError:
            pass
        release.set()
        holder.join(timeout=20)
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass
        raise
    _, child_status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(child_status) == 0
    os.close(go_w)
    os.close(out_r)
    with store.exclusive_session():
        pass


def test_session_fork_child_does_not_pin(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    b_r, b_w = os.pipe()
    c_r, c_w = os.pipe()
    d_r, d_w = os.pipe()
    outer = os.fork()
    if outer == 0:
        try:
            os.close(b_r)
            os.close(c_r)
            os.close(d_w)
            from kullback.store import WorkdirStore as _W
            other = _W(str(root), [_session_spec()])
            with other.exclusive_session():
                os.write(b_w, b"1\n")
                inner = os.fork()
                if inner == 0:
                    try:
                        os.close(b_w)
                        os.write(c_w, b"%d\n" % os.getpid())
                        assert os.read(d_r, 1) == b"r"
                        os.write(c_w, b"bye\n")
                        os.close(d_r)
                        os.close(c_w)
                    except BaseException:
                        try:
                            os.close(d_r)
                        except OSError:
                            pass
                        try:
                            os.close(c_w)
                        except OSError:
                            pass
                        os._exit(1)
                    os._exit(0)
                os.write(b_w, b"%d\n" % inner)
                os.close(b_w)
                os.close(c_w)
                os.close(d_r)
        except BaseException:
            try:
                os.close(b_w)
            except OSError:
                pass
            try:
                os.close(c_w)
            except OSError:
                pass
            try:
                os.close(d_r)
            except OSError:
                pass
            os._exit(1)
        os._exit(0)
    os.close(b_w)
    os.close(c_w)
    os.close(d_r)
    outer_reaped = False
    try:
        assert _pipe_read_line(b_r) == b"1\n"
        inner_pid = int(_pipe_read_line(b_r))
        assert int(_pipe_read_line(c_r)) == inner_pid
        _, status = os.waitpid(outer, 0)
        outer_reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
        with store.exclusive_session():
            pass
        os.write(d_w, b"r")
        assert _pipe_read_line(c_r) == b"bye\n"
        with store.exclusive_session():
            pass
    except BaseException:
        if not outer_reaped:
            try:
                import signal as _sig
                os.kill(outer, _sig.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(outer, 0)
                outer_reaped = True
            except ChildProcessError:
                pass
        try:
            os.write(d_w, b"r")
        except OSError:
            pass
        raise
    finally:
        os.close(b_r)
        os.close(c_r)
        os.close(d_w)


def test_session_exec_child_does_not_pin(tmp_path):
    import sys

    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    with store.exclusive_session():
        pid = os.fork()
        if pid == 0:
            try:
                os.execv(sys.executable, [sys.executable, "-c", "pass"])
            finally:
                os._exit(127)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        probe = WorkdirStore(root, [_session_spec()])
        with pytest.raises(_Busy):
            with probe.exclusive_session():
                pass
    with store.exclusive_session():
        pass
