"""Remembered provider keys: one JSON file under the home directory, mode 0600.

Why this file exists. A key pasted into /login used to last only for the session, so the
next launch asked again and the only lasting path was .env. The founder decided on
2026-09-25 that a key given once is still there on the next launch. It lives in
~/.kullback/auth.json, never in a workdir, a package or a published file, and only names
are ever listed or printed, never values.

The file holds {"keys": {NAME: value}}. Names follow the same shape as the provider key
variables they feed, so a stored name that cannot be an environment variable is refused
rather than written.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Optional

AUTH_DIR_NAME = ".kullback"
AUTH_FILE_NAME = "auth.json"
AUTH_ENV_VAR = "KULLBACK_AUTH_FILE"

_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def check_name(name: str) -> str:
    """A stored name, or the reason it cannot be one. Environment variables are the shape."""
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            f"a remembered key is named like {name!r}; names look like OPENAI_API_KEY"
        )
    return name


def auth_path() -> Path:
    """Where remembered keys live. The env var exists so tests never touch the real home."""
    override = os.environ.get(AUTH_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / AUTH_DIR_NAME / AUTH_FILE_NAME


def _read_store(path: Path) -> dict[str, str]:
    """The stored keys, or nothing where the file is missing or does not parse."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys = body.get("keys") if isinstance(body, dict) else None
    if not isinstance(keys, dict):
        return {}
    return {name: value for name, value in keys.items()
            if isinstance(name, str) and isinstance(value, str)}


def _write_store(path: Path, keys: dict[str, str]) -> Path:
    """Rewrite the whole file atomically, so a crash never leaves half a file.

    The temp file is created at 0600 beside the target and renamed over it, the way
    tau writes its own credentials file. The parent dir is created at 0700.
    """
    parent = path.parent
    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(parent, 0o700)
    content = json.dumps({"keys": keys}, indent=2, sort_keys=True) + "\n"
    tmp = parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return path


def remember(name: str, value: str) -> Path:
    """Store one key under its variable name and return where it went. Values never print."""
    check_name(name)
    path = auth_path()
    keys = _read_store(path)
    keys[name] = value
    return _write_store(path, keys)


def forget(name: str) -> bool:
    """Drop one remembered key. True when one was there, False when nothing changed."""
    check_name(name)
    path = auth_path()
    if not path.is_file():
        return False
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    keys = body.get("keys") if isinstance(body, dict) else None
    if not isinstance(keys, dict) or name not in keys:
        return False
    remaining = {key: value for key, value in keys.items() if key != name}
    _write_store(path, remaining)
    return True


def stored_names() -> list[str]:
    """The remembered key names, sorted. Names only, never values."""
    return sorted(_read_store(auth_path()))


def load_credentials(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Put remembered keys into the environment without overriding what is set.

    The same rule load_dotenv keeps: an exported variable and a .env entry both beat a
    remembered key. Returns the variables that were added. A file that does not parse
    adds nothing and does not raise. A file with wider permissions than 0600 is still
    read, then fixed to 0600.
    """
    values = os.environ if env is None else env
    path = auth_path()
    if path.is_file():
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = 0o600
        if mode != 0o600:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    added: dict[str, str] = {}
    for name, value in _read_store(path).items():
        if not _NAME_PATTERN.match(name):
            continue
        if name not in values:
            values[name] = added[name] = value
    return added
