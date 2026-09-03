"""Which builds are running, for the screen's session list. One small JSON file per build.

The CLI writes a heartbeat when a build starts and rewrites it when the build leaves;
the TUI only reads. A heartbeat whose pid is gone is a build that died without saying so,
which the screen shows as dead rather than dropping, because a vanished build is the thing
the person watching most needs to know about. Nothing here calls a model or reads a key.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def sessions_dir() -> Path:
    """Where heartbeats live: ~/.kullback/sessions, or KULLBACK_SESSIONS_DIR when set (tests)."""
    override = os.environ.get("KULLBACK_SESSIONS_DIR")
    return Path(override).expanduser() if override else Path.home() / ".kullback" / "sessions"


def _slug(workdir: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(workdir))[-60:].strip("-") or "work"


def beat(workdir: Any, model: Optional[str], status: str, **extra: Any) -> Path:
    """Write (or rewrite) this build's heartbeat. Status is running, done or failed."""
    directory = sessions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path = directory / f"{_slug(workdir)}-{os.getpid()}.json"
    record: dict[str, Any] = {"workdir": str(workdir), "model": model, "pid": os.getpid(),
                              "status": status, "updated_at": now}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        record["started_at"] = previous.get("started_at", now)
    except (OSError, ValueError):
        record["started_at"] = now
    record.update(extra)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def read_all() -> list[dict[str, Any]]:
    """Every heartbeat, newest first. Corrupt files are skipped, not fatal: the screen must
    survive a half-written heartbeat from a build that died mid-write."""
    directory = sessions_dir()
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []
    records = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("workdir"):
            record["_path"] = str(path)
            records.append(record)
    records.sort(key=lambda r: float(r.get("updated_at") or 0), reverse=True)
    return records


def alive(pid: Any) -> bool:
    """Whether a pid still runs. A bad pid is dead, not an error."""
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True
