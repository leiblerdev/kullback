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
