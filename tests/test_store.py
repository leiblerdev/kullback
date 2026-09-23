from __future__ import annotations

import json
import multiprocessing
import os
import stat
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


@pytest.mark.parametrize(
    ("raw", "check_ident"),
    [
        (b'{"store_format": 1,', True),
        (b"\xff\xfe\x00not text", True),
        (json.dumps({"store_format": 1, "artifact": "tally_b", "format": 999, "owner": "owner_b", "value": 1}, sort_keys=True, indent=2).encode("utf-8"), False),
        (json.dumps({"store_format": 1, "artifact": "other", "format": 1, "owner": "owner_b", "value": 1}, sort_keys=True, indent=2).encode("utf-8"), False),
        (json.dumps({"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "someone", "value": 1}, sort_keys=True, indent=2).encode("utf-8"), False),
        (json.dumps({"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "owner_b", "value": 1, "extra": 0}, sort_keys=True, indent=2).encode("utf-8"), False),
        (json.dumps({"store_format": 1, "artifact": "tally_b", "format": 1, "owner": "owner_b", "value": "nope"}, sort_keys=True, indent=2).encode("utf-8"), False),
    ],
    ids=["truncated", "non-text", "bad-format", "bad-name", "bad-owner", "extra-key", "bad-value"],
)
def test_torn_case_reads_torn(tmp_path, raw, check_ident):
    root = tmp_path / "w2"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_b", path="tally_b.json", format=1, owner="owner_b", validate=_check_int)])
    store.write("tally_b", 1)
    target = root / "tally_b.json"
    target.write_bytes(raw)
    got = store.read("tally_b")
    assert got.status == "torn"
    assert got.value is None
    if check_ident:
        assert isinstance(got.reason, str) and got.reason.isidentifier()
    assert got.reason is not None
    assert got.legacy is False


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


def _assert_symlink_aliases_of_root_and_lock_refused(root):
    root.mkdir(parents=True, exist_ok=True)
    base = WorkdirStore(root, [ArtifactSpec(name="base_v", path="base_v.json", format=1, owner="owner_v", validate=_check_int)])
    base.write("base_v", 1)
    lock_path = root / ".store.lock"
    assert lock_path.is_file()
    before_inode = os.stat(lock_path).st_ino
    (root / "alias_l.json").symlink_to(lock_path)
    (root / "alias_r.json").symlink_to(root)
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


def _assert_other_name_for_the_lock_refused(root, alias_rel, make_alias):
    store = WorkdirStore(root, [ArtifactSpec(name="tally_hd", path="tally_hd.json", format=1, owner="owner_hd", validate=_check_int)])
    store.write("tally_hd", 1)
    lock_path = root / ".store.lock"
    assert lock_path.is_file()
    alias_path = root / alias_rel
    make_alias(lock_path, alias_path)
    if not alias_path.exists() or not alias_path.samefile(lock_path):
        return
    before = os.stat(lock_path).st_ino
    alias_store = WorkdirStore(root, [ArtifactSpec(name="alias_hd", path=alias_rel, format=1, owner="owner_hd", validate=_check_int)])
    with pytest.raises(StoreError):
        alias_store.write("alias_hd", 2)
    assert os.stat(lock_path).st_ino == before
    with pytest.raises(StoreError):
        alias_store.read("alias_hd")
    assert os.stat(lock_path).st_ino == before


def _assert_constructor_refuses_physical_aliases(root):
    root.mkdir(parents=True, exist_ok=True)
    real = root / "real.json"
    real.write_text("{}", encoding="utf-8")
    for link_kind, alias in (("symlink", root / "sym.json"), ("hardlink", root / "hard.json")):
        if link_kind == "symlink":
            alias.symlink_to(real)
        else:
            os.link(real, alias)
        with pytest.raises(ValueError):
            WorkdirStore(root, [ArtifactSpec(name="first", path="real.json", format=1, owner="o", validate=_check_int), ArtifactSpec(name="second", path=alias.name, format=1, owner="o", validate=_check_int)])


def test_unknown_duplicate_unsafe_reserved_and_aliased_paths_are_refused(tmp_path):
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
    for spelling in [".store.lock", "./.store.lock", "sub/../.store.lock", "a/./../.store.lock"]:
        with pytest.raises(ValueError) as reserved:
            WorkdirStore(root, [ArtifactSpec(name="r1", path=spelling, format=1, owner="o", validate=_check_int)])
        assert str(reserved.value) == "reserved store path"
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
    _assert_symlink_aliases_of_root_and_lock_refused(tmp_path / "w_alias")
    _assert_other_name_for_the_lock_refused(tmp_path / "w_hard", "alias_hd.json", lambda lock, alias: os.link(lock, alias))
    _assert_other_name_for_the_lock_refused(tmp_path / "w_cilock", ".STORE.LOCK", lambda lock, alias: None)
    _assert_constructor_refuses_physical_aliases(tmp_path / "w_phys_link")


def _assert_failed_write_preserves_bytes(target, before):
    assert target.read_bytes() == before
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name and p.name != ".store.lock"]
    assert leftovers == []


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
    _assert_failed_write_preserves_bytes(target, before)
    monkeypatch.setattr(os, "replace", real_replace)
    real_dumps = json.dumps
    monkeypatch.setattr("kullback.store.json.dumps", lambda *a, **k: (_ for _ in ()).throw(ValueError("injected")))
    with pytest.raises(ValueError):
        store.write("tally_e", 6)
    _assert_failed_write_preserves_bytes(target, before)
    monkeypatch.setattr("kullback.store.json.dumps", real_dumps)
    os.chmod(target, 0o640)
    store.write("tally_e", 7)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o640
    assert store.read("tally_e").value == 7
    before = target.read_bytes()

    def _interrupt(src, dst):
        raise KeyboardInterrupt()

    monkeypatch.setattr(os, "replace", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.write("tally_e", 8)
    _assert_failed_write_preserves_bytes(target, before)
    monkeypatch.setattr(os, "replace", real_replace)
    store.write("tally_e", 9)
    assert store.read("tally_e").value == 9


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


def test_strict_json_constants_overflow_and_duplicate_keys_read_torn(tmp_path):
    root = tmp_path / "w_strict"
    store = WorkdirStore(root, [ArtifactSpec(name="tally_s", path="tally_s.json", format=1, owner="owner_s", validate=_identity)])
    target = root / "tally_s.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    for token in ["NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"]:
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
    target.write_text("{" + "\"store_format\": 1, \"artifact\": \"tally_s\", \"format\": 1, \"owner\": \"owner_s\", \"value\": 1e2}", encoding="utf-8")
    finite = store.read("tally_s")
    assert finite.status == "ok"
    assert finite.value == 100.0


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


@pytest.mark.parametrize(
    ("made", "workdir", "rel", "synced"),
    [
        ("w", "w", "sub/f.json", ["w/sub", "w"]),
        ("parent", "parent/work", "journal/00000000000000000001.json", ["parent/work", "parent"]),
        ("base", "base/mid/work", "f.json", ["base/mid/work", "base/mid", "base"]),
        ("w", "w", "top.json", ["w", "."]),
    ],
    ids=["existing-root-new-subdir", "absent-workdir", "two-missing-levels", "top-level"],
)
def test_a_write_syncs_each_directory_it_touched_up_to_the_first_existing_one(tmp_path, monkeypatch, made, workdir, rel, synced):
    (tmp_path / made).mkdir()
    root = tmp_path / workdir
    store = WorkdirStore(root, [ArtifactSpec(name="chain", path=rel, format=1, owner="o", validate=_check_int)])
    calls = []
    _trace_calls(monkeypatch, calls)
    store.write("chain", 1)
    assert store.read("chain").value == 1
    dirs = _dir_fsyncs(calls)
    expected = [str((tmp_path / each).resolve()) for each in synced]
    for each in expected:
        assert each in dirs
    if workdir == made and "/" in rel:
        assert dirs.index(expected[0]) < dirs.index(expected[1])
    first_replace = next(i for i, entry in enumerate(calls) if entry[0] == "replace")
    first_dir = next(i for i, entry in enumerate(calls) if entry[0] == "dir")
    assert first_replace < first_dir


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


def _session_spec():
    return ArtifactSpec(name="n", path="n.json", format=1, owner="o", validate=_check_int)


def test_session_acquire_use_release_and_release_on_exception(tmp_path):
    from kullback.store import SessionBusy as _Busy
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


@pytest.mark.parametrize(
    ("outer", "spelling"),
    [("session", "same"), ("transaction", "same"), ("session", "alias")],
    ids=["session", "transaction", "session-aliased-root"],
)
def test_session_acquire_while_held_refuses(tmp_path, outer, spelling):
    from kullback.store import SessionBusy as _Busy
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    inner = store if spelling == "same" else WorkdirStore(root / "sub" / "..", [_session_spec()])
    held = store.exclusive_session() if outer == "session" else store.transaction()
    with held:
        with pytest.raises(_Busy):
            with inner.exclusive_session():
                pass
        with inner.transaction():
            inner.write("n", 3)
    assert store.read("n").value == 3
    with store.exclusive_session():
        pass


def test_session_is_exclusive_across_threads_while_their_transactions_run(tmp_path):
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


def _exit_wait_driver(store, entered, release, log, driver_done):
    with store.exclusive_session():
        entered.set()
        assert release.wait(timeout=60)
    log.append("session-released")
    driver_done.set()


def _exit_wait_worker(root, in_txn, done, log, worker_closed):
    other = WorkdirStore(root, [_session_spec()])
    with other.transaction():
        in_txn.set()
        assert done.wait(timeout=60)
    log.append("txn-closed")
    worker_closed.set()


def _check_session_exit_waits_for_open_transaction(root, store):
    from kullback.store import SessionBusy as _Busy
    log = []
    entered = threading.Event()
    release = threading.Event()
    in_txn = threading.Event()
    done = threading.Event()
    driver_done = threading.Event()
    worker_closed = threading.Event()
    driver = threading.Thread(target=_exit_wait_driver, args=(store, entered, release, log, driver_done))
    driver.start()
    assert entered.wait(timeout=60)
    worker = threading.Thread(target=_exit_wait_worker, args=(root, in_txn, done, log, worker_closed))
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


def test_session_exit_waits_for_open_transaction(tmp_path):
    root = tmp_path / "w"
    store = WorkdirStore(root, [_session_spec()])
    for _rep in range(3):
        _check_session_exit_waits_for_open_transaction(root, store)


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


def _pin_inner_child(b_w, c_w, d_w, d_r):
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


def _pin_outer_child(b_w, c_w, d_w, b_r, c_r, d_r, root_str):
    try:
        os.close(b_r)
        os.close(c_r)
        os.close(d_w)
        from kullback.store import WorkdirStore as _W
        other = _W(root_str, [_session_spec()])
        with other.exclusive_session():
            os.write(b_w, b"1\n")
            inner = os.fork()
            if inner == 0:
                _pin_inner_child(b_w, c_w, d_w, d_r)
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


def _abort_pin_probe(outer, outer_reaped, d_w):
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
    return outer_reaped


def _check_forked_child_does_not_pin(root, store):
    b_r, b_w = os.pipe()
    c_r, c_w = os.pipe()
    d_r, d_w = os.pipe()
    outer = os.fork()
    if outer == 0:
        _pin_outer_child(b_w, c_w, d_w, b_r, c_r, d_r, str(root))
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
        outer_reaped = _abort_pin_probe(outer, outer_reaped, d_w)
        raise
    finally:
        os.close(b_r)
        os.close(c_r)
        os.close(d_w)


def test_session_child_process_does_not_pin_the_session(tmp_path):
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
    _check_forked_child_does_not_pin(root, store)
