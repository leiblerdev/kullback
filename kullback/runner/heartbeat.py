"""Which builds are running, for the screen's session list. One small JSON file per build.

The CLI beats while a build runs and rewrites the heartbeat when the build leaves; the TUI only
reads. A heartbeat whose pid is gone is a build that died without saying so, which the screen
shows as dead rather than dropping, because a vanished build is the thing the person watching
most needs to know about. Every beat carries the spend off the build's own ledger, so a session
someone is watching from another directory shows what it is costing while it costs it rather than
only once it is over. Nothing here calls a model or reads a key.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


def sessions_dir() -> Path:
    """Where heartbeats live: ~/.kullback/sessions, or KULLBACK_SESSIONS_DIR when set (tests)."""
    override = os.environ.get("KULLBACK_SESSIONS_DIR")
    return Path(override).expanduser() if override else Path.home() / ".kullback" / "sessions"


def _slug(workdir: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(workdir))[-60:].strip("-") or "work"


def _spent(workdir: Path) -> Optional[float]:
    """What this build has paid so far, off the ledger the Runner keeps in its workdir.

    None when there is no ledger yet, which is a build that has not reached a model call rather
    than a build that spent nothing; the screen shows nothing rather than a confident zero."""
    try:
        from kullback.runner import budget

        return float(budget.load_totals(workdir)["total"].get("usd") or 0.0)
    except Exception:
        return None


def beat(workdir: Any, model: Optional[str], status: str, **extra: Any) -> Path:
    """Write (or rewrite) this build's heartbeat. Status is running, done or failed."""
    directory = sessions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    now = time.time()
    # Absolute, off this build's own directory, because the screen that reads the heartbeat is
    # started somewhere else: `--workdir work` names a real directory to the build and nothing at
    # all to a `/watch` two directories away, which then opens on a workdir that does not exist and
    # says 'no build yet' about a build that is running. Symlinks are left alone (absolute, not
    # resolved) so the path still reads the way the person typed it.
    workdir = Path(workdir).expanduser().absolute()
    path = directory / f"{_slug(workdir)}-{os.getpid()}.json"
    record: dict[str, Any] = {"workdir": str(workdir), "model": model, "pid": os.getpid(),
                              "status": status, "updated_at": now}
    spent = _spent(workdir)
    if spent is not None:
        record["spend_usd"] = spent
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        record["started_at"] = previous.get("started_at", now)
    except (OSError, ValueError):
        record["started_at"] = now
    record.update(extra)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


class Pulse:
    """Rewrites one build's heartbeat every few seconds for as long as the build runs.

    Without it a heartbeat is written twice, at the start and at the end, so a build that spends
    an hour in its stages reads `$0.0000` on the screen the whole time and only tells the truth
    once there is nothing left to watch. The thread is a daemon and every beat is wrapped, so a
    full disk or a deleted sessions directory slows nothing down and kills no build: a heartbeat
    is a courtesy to whoever is watching, never a step of the build."""

    def __init__(self, workdir: Any, model: Optional[str], status: str = "running",
                 every_seconds: float = 5.0, **extra: Any) -> None:
        self.workdir, self.model, self.status = workdir, model, status
        self.every_seconds, self.extra = every_seconds, extra
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kullback-heartbeat", daemon=True)

    def start(self) -> Pulse:
        """Beat now, then keep beating. Beating now means the screen lists the build immediately
        rather than a few seconds in."""
        self._beat()
        self._thread.start()
        return self

    def _beat(self) -> None:
        try:
            beat(self.workdir, self.model, self.status, **self.extra)
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.every_seconds):
            self._beat()

    def stop(self) -> None:
        """Stop beating and wait for the thread, so the caller's closing beat is the last word on
        this build rather than racing a pulse that still says `running`."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.every_seconds))

    def __enter__(self) -> Pulse:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


def pulse(workdir: Any, model: Optional[str], status: str = "running",
          every_seconds: float = 5.0, **extra: Any) -> Pulse:
    """A started Pulse: beating already, stopped by `stop()` or by leaving its `with`."""
    return Pulse(workdir, model, status, every_seconds, **extra).start()


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
