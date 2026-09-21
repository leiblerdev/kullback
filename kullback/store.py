from __future__ import annotations

import contextlib
import errno
import json
import math
import os
import posixpath
import stat
import tempfile
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from weakref import WeakValueDictionary


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: str
    format: int
    owner: str
    validate: Callable[[Any], Any]
    allow_legacy: bool = False


@dataclass(frozen=True)
class ReadResult:
    status: Literal["ok", "missing", "torn"]
    value: Any = None
    reason: str | None = None
    legacy: bool = False


class StoreError(RuntimeError):
    pass


class WriteDurabilityError(StoreError):
    replaced = True
    durability = "unknown"

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"artifact replaced but directory durability is unknown: {path}")


class SessionBusy(StoreError):
    pass


@dataclass
class _LockState:
    rlock: Any = field(default_factory=threading.RLock)
    depth: int = 0
    fd: int | None = None
    owner_pid: int = 0
    session_active: bool = False
    session_closing: bool = False
    session_fd: int | None = None


_LOCKS: WeakValueDictionary[tuple, _LockState] = WeakValueDictionary()
_GUARD = threading.Lock()
_ENVELOPE_KEYS = frozenset({"store_format", "artifact", "format", "owner", "value"})


def _canonical_root(root: str | Path) -> str:
    return str(Path(root).resolve())


def _path_key(path: Path) -> str:
    return unicodedata.normalize("NFC", str(path.resolve())).casefold()


def _file_key(path: Path) -> tuple | None:
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return info.st_dev, info.st_ino


def _check_path(path: str) -> None:
    if not isinstance(path, str) or path == "":
        raise ValueError("bad path")
    if posixpath.normpath(path) == ".store.lock":
        raise ValueError("reserved store path")
    if os.path.isabs(path):
        raise ValueError("bad path")
    for part in path.split("/"):
        if part in (".", "..", ""):
            raise ValueError("bad path")


def _check_spec(spec: ArtifactSpec) -> None:
    if not isinstance(spec.name, str) or spec.name == "":
        raise ValueError("bad name")
    if not isinstance(spec.owner, str) or spec.owner == "":
        raise ValueError("bad owner")
    if isinstance(spec.format, bool) or not isinstance(spec.format, int) or spec.format <= 0:
        raise ValueError("bad format")
    if not callable(spec.validate):
        raise ValueError("bad validator")
    _check_path(spec.path)


def _state_for(key: tuple) -> _LockState:
    with _GUARD:
        found = _LOCKS.get(key)
        if found is None:
            found = _LockState()
            found.owner_pid = key[0]
            _LOCKS[key] = found
        return found


_OPEN_FDS: set = set()
_FDS_GUARD = threading.Lock()


def _fork_before() -> None:
    _GUARD.acquire()
    _FDS_GUARD.acquire()


def _fork_after_parent() -> None:
    try:
        _FDS_GUARD.release()
    finally:
        _GUARD.release()


def _fork_after_child() -> None:
    fds = list(_OPEN_FDS)
    _OPEN_FDS.clear()
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass
    global _GUARD
    global _FDS_GUARD
    _GUARD = threading.Lock()
    _FDS_GUARD = threading.Lock()
    for st in list(_LOCKS.values()):
        st.rlock = threading.RLock()
        st.depth = 0
        st.fd = None
        st.session_active = False
        st.session_closing = False
        st.session_fd = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_fork_before, after_in_parent=_fork_after_parent, after_in_child=_fork_after_child)


def _open_lock_file(root_str: str) -> int:
    lock_path = os.path.join(root_str, ".store.lock")
    flags = os.O_CREAT | os.O_RDWR
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None:
        flags |= nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    with _FDS_GUARD:
        fd = os.open(lock_path, flags, 0o600)
        _OPEN_FDS.add(fd)
        return fd


def _close_lock_file(fd: int) -> None:
    with _FDS_GUARD:
        try:
            os.close(fd)
        finally:
            _OPEN_FDS.discard(fd)


def _acquire_exclusive(root_str: str, state: _LockState) -> None:
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    Path(root_str).mkdir(parents=True, exist_ok=True)
    fd = _open_lock_file(root_str)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
    except BaseException:
        _close_lock_file(fd)
        raise
    state.fd = fd


def _release_exclusive(state: _LockState) -> None:
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    fd = state.fd
    state.fd = None
    try:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        _close_lock_file(fd)


def _enter_transaction(root_str: str, state: _LockState) -> tuple:
    state.rlock.acquire()
    mine = False
    covered = False
    try:
        if state.depth == 0:
            if state.session_active and not state.session_closing:
                covered = True
            else:
                _acquire_exclusive(root_str, state)
                mine = True
        state.depth += 1
    except BaseException:
        state.rlock.release()
        raise
    return mine, covered


def _exit_transaction(state: _LockState, mine: bool, covered: bool) -> None:
    if state.owner_pid != os.getpid():
        raise StoreError("store state changed process")
    try:
        state.depth -= 1
        if mine:
            _release_exclusive(state)
    finally:
        state.rlock.release()


def _session_acquire(root_str: str, state: _LockState) -> None:
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    if not state.rlock.acquire(blocking=False):
        raise SessionBusy("workdir driver session busy")
    try:
        if state.session_active or state.session_closing:
            raise SessionBusy("workdir driver session busy")
        if state.depth != 0:
            raise SessionBusy("workdir driver session busy")
        fd = _open_lock_file(root_str)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError as exc:
            _close_lock_file(fd)
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                raise SessionBusy("workdir driver session busy") from exc
            raise
        except BaseException:
            _close_lock_file(fd)
            raise
        state.session_active = True
        state.session_closing = False
        state.session_fd = fd
    finally:
        state.rlock.release()


def _session_release(state: _LockState) -> None:
    if state.owner_pid != os.getpid():
        raise StoreError("store state changed process")
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    state.rlock.acquire()
    try:
        if not state.session_active:
            return
        state.session_closing = True
        fd = state.session_fd
        state.session_fd = None
        try:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            finally:
                _close_lock_file(fd)
        finally:
            state.session_active = False
            state.session_closing = False
    finally:
        state.rlock.release()


def _session_close(state: _LockState) -> None:
    _session_release(state)


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON")


def _unique_pairs(pairs: list) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _decode(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"), parse_constant=_reject_constant,
                      object_pairs_hook=_unique_pairs, parse_float=_finite_float)


def _check_owner(target: Path, spec: ArtifactSpec) -> None:
    try:
        body = _decode(target.read_bytes())
    except FileNotFoundError:
        return
    except (UnicodeError, ValueError, RecursionError):
        return
    if isinstance(body, dict) and body.get("store_format") == 1:
        if body.get("artifact") != spec.name or body.get("owner") != spec.owner:
            raise StoreError("artifact declaration conflicts with existing file")


def _check_envelope(obj: Any, spec: ArtifactSpec) -> None:
    if not isinstance(obj, dict):
        raise ValueError("bad envelope")
    if set(obj.keys()) != set(_ENVELOPE_KEYS):
        raise ValueError("bad envelope")
    if type(obj["store_format"]) is not int or obj["store_format"] != 1:
        raise ValueError("bad store format")
    if obj["artifact"] != spec.name:
        raise ValueError("bad artifact")
    if type(obj["format"]) is not int or obj["format"] != spec.format:
        raise ValueError("bad format")
    if obj["owner"] != spec.owner:
        raise ValueError("bad owner")


def _interpret(obj: Any, spec: ArtifactSpec) -> ReadResult:
    if not isinstance(obj, dict) or "store_format" not in obj:
        if not spec.allow_legacy:
            raise ValueError("legacy refused")
        out = spec.validate(obj)
        return ReadResult(status="ok", value=out, legacy=True)
    _check_envelope(obj, spec)
    out = spec.validate(obj["value"])
    return ReadResult(status="ok", value=out)


def _sync_directory(parent: Path) -> None:
    fd = os.open(str(parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_ancestors(target: Path) -> None:
    node = target.parent
    while True:
        parent = node.parent
        if parent == node:
            return
        try:
            same = parent.stat().st_dev == node.stat().st_dev
            if same:
                _sync_directory(parent)
        except OSError as exc:
            raise WriteDurabilityError(target) from exc
        if not same:
            return
        node = parent


def _stage_and_replace(target: Path, text: str) -> None:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        mode = 0o600
    fd, tmp = tempfile.mkstemp(dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            os.fchmod(stream.fileno(), mode)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        try:
            _sync_directory(parent)
        except OSError as exc:
            raise WriteDurabilityError(target) from exc
    finally:
        if fd is not None:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)


class WorkdirStore:
    def __init__(self, root: str | Path, specs: list[ArtifactSpec]):
        canon = _canonical_root(root)
        seen_names: set = set()
        seen_paths: set = set()
        seen_files: set = set()
        mapping: dict = {}
        for spec in specs:
            _check_spec(spec)
            candidate = Path(canon) / spec.path
            norm = _path_key(candidate)
            if spec.name in seen_names:
                raise ValueError("duplicate name")
            if norm in seen_paths:
                raise ValueError("duplicate path")
            physical = _file_key(candidate)
            if physical is not None and physical in seen_files:
                raise ValueError("duplicate physical path")
            if physical is not None:
                seen_files.add(physical)
            seen_names.add(spec.name)
            seen_paths.add(norm)
            mapping[spec.name] = spec
        self._root_str = canon
        self._specs = mapping

    def _resolve(self, spec: ArtifactSpec) -> Path:
        base = Path(self._root_str)
        target = base / spec.path
        resolved = target.resolve()
        if resolved == base or base not in resolved.parents:
            raise StoreError("symlink escape")
        lock = base / ".store.lock"
        if resolved == lock or (resolved.exists() and resolved.samefile(lock)):
            raise StoreError("reserved store path")
        return resolved

    @contextlib.contextmanager
    def transaction(self):
        root = Path(self._root_str)
        root.mkdir(parents=True, exist_ok=True)
        identity = root.stat()
        key = (os.getpid(), identity.st_dev, identity.st_ino)
        state = _state_for(key)
        if state.owner_pid != os.getpid():
            raise StoreError("store state changed process")
        mine, covered = _enter_transaction(self._root_str, state)
        try:
            yield self
        finally:
            _exit_transaction(state, mine, covered)

    @contextlib.contextmanager
    def exclusive_session(self):
        root = Path(self._root_str)
        root.mkdir(parents=True, exist_ok=True)
        identity = root.stat()
        key = (os.getpid(), identity.st_dev, identity.st_ino)
        state = _state_for(key)
        if state.owner_pid != os.getpid():
            raise StoreError("store state changed process")
        _session_acquire(self._root_str, state)
        try:
            yield self
        finally:
            if state.owner_pid == os.getpid():
                _session_close(state)

    def read(self, name: str) -> ReadResult:
        spec = self._specs[name]
        try:
            mode = Path(self._root_str).stat().st_mode
        except FileNotFoundError:
            return ReadResult(status="missing")
        if not stat.S_ISDIR(mode):
            raise NotADirectoryError(self._root_str)
        with self.transaction():
            target = self._resolve(spec)
            try:
                data = target.read_bytes()
            except FileNotFoundError:
                return ReadResult(status="missing")
            try:
                obj = _decode(data)
            except (UnicodeError, ValueError, RecursionError) as exc:
                return ReadResult(status="torn", reason=type(exc).__name__)
            try:
                return _interpret(obj, spec)
            except OSError:
                raise
            except Exception as exc:
                return ReadResult(status="torn", reason=type(exc).__name__)

    def write(self, name: str, value: Any) -> Path:
        spec = self._specs[name]
        spec.validate(value)
        envelope = {
            "store_format": 1,
            "artifact": spec.name,
            "format": spec.format,
            "owner": spec.owner,
            "value": value,
        }
        text = json.dumps(
            envelope,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.transaction():
            target = self._resolve(spec)
            _check_owner(target, spec)
            _stage_and_replace(target, text)
            _sync_ancestors(target)
            return target
