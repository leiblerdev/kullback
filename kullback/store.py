from __future__ import annotations

import contextlib
import json
import math
import os
import posixpath
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


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


_LOCKS: dict = {}
_GUARD = threading.Lock()
_ENVELOPE_KEYS = frozenset({"store_format", "artifact", "format", "owner", "value"})


def _canonical_root(root: str | Path) -> str:
    return str(Path(root).resolve())


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


def _state_for(key: tuple) -> dict:
    with _GUARD:
        found = _LOCKS.get(key)
        if found is None:
            found = {"rlock": threading.RLock(), "depth": 0, "fd": None}
            _LOCKS[key] = found
        return found


def _acquire_exclusive(root_str: str, state: dict) -> None:
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    Path(root_str).mkdir(parents=True, exist_ok=True)
    lock_path = os.path.join(root_str, ".store.lock")
    flags = os.O_CREAT | os.O_RDWR
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None:
        flags |= nofollow
    fd = os.open(lock_path, flags, 0o600)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
    except BaseException:
        os.close(fd)
        raise
    state["fd"] = fd


def _release_exclusive(state: dict) -> None:
    try:
        import fcntl as _fcntl
    except ImportError as exc:
        raise StoreError("fcntl unavailable") from exc
    fd = state["fd"]
    state["fd"] = None
    try:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


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
        dfd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if fd is not None:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)


class WorkdirStore:
    def __init__(self, root: str | Path, specs: list[ArtifactSpec]):
        canon = _canonical_root(root)
        seen_names: set = set()
        seen_paths: set = set()
        mapping: dict = {}
        for spec in specs:
            _check_spec(spec)
            norm = posixpath.normpath(spec.path)
            if spec.name in seen_names:
                raise ValueError("duplicate name")
            if norm in seen_paths:
                raise ValueError("duplicate path")
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
        rlock = state["rlock"]
        rlock.acquire()
        try:
            if state["depth"] == 0:
                _acquire_exclusive(self._root_str, state)
            state["depth"] += 1
        except BaseException:
            rlock.release()
            raise
        try:
            yield self
        finally:
            try:
                state["depth"] -= 1
                if state["depth"] == 0:
                    _release_exclusive(state)
            finally:
                rlock.release()

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
                obj = json.loads(data.decode("utf-8"), parse_constant=_reject_constant,
                                 object_pairs_hook=_unique_pairs, parse_float=_finite_float)
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
            _stage_and_replace(target, text)
            return target
