"""Steer a live run from any process: requests and acks travel on the workdir's bus.

A screen that started a build steers it through the harness it holds. Any other process (a second
screen, `kullback steer` over SSH, a cron job) holds nothing, so it appends a SteerRequestEvent to
workdir/bus.jsonl instead. Inside the build a SteerBridge follows the bus from the end, hands each
request to the harness's own `steer`, `follow_up` or `cancel` (the same calls the screen makes from
its thread) and appends a SteerAckEvent, so the sender knows it was taken and every watcher sees
both in the one ordered stream. A file rather than a socket: the harness delivers a nudge only
after the current tool batch, so a faster channel would buy nothing, and the file survives a crash
and needs no port. This module imports only the agent core and the standard library (D121/D123).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from kullback.agent.bus import Bus
from kullback.agent.events import SteerAckEvent, SteerRequestEvent

__all__ = ["BUS_FILE", "SteerBridge", "live_records", "request", "target_session", "wait_for_ack"]

BUS_FILE = "bus.jsonl"

# The writer and the seq of each request this process sent, so waiting for its ack reads only
# what was appended after the request instead of the whole log.
_SENT: dict[str, tuple[Bus, int]] = {}

def request(workdir: Union[str, Path], kind: str, text: str = "", sender: str = "",
            session: str = "") -> str:
    """Append one steer request to the workdir's bus and return its id."""
    request_id = uuid.uuid4().hex[:12]
    bus = Bus(Path(workdir) / BUS_FILE, agent=sender or "steer")
    record = bus.append(SteerRequestEvent(id=request_id, kind=kind, text=text, sender=sender,
                                          session=session))
    _SENT[request_id] = (bus, record.seq)
    return request_id


def _session_file(workdir: Union[str, Path], pid: int) -> Path:
    return Path(workdir).expanduser() / f"steer-{pid}.json"


def live_records(workdir: Union[str, Path]) -> list[dict]:
    """The live records bridges wrote beside their buses, running ones only.

    A build started outside the CLI and the screen beats no heartbeat, so its bridge leaves
    this instead: one small file per live build, named for its pid, removed when the bridge
    closes. Corrupt files are skipped, not fatal, and a dead pid is left for the caller, which
    reads aliveness the way it does for heartbeats."""
    try:
        paths = sorted(Path(workdir).expanduser().glob("steer-*.json"))
    except OSError:
        return []
    out = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("status") == "running":
            out.append(record)
    return out


def target_session(live: list[dict], session: str = "") -> str:
    """The session a sender's request names: the named one, or the only live build's pid.

    A build is steered through the bus it shares with any other build on its workdir, so an
    unnamed request with several live builds names no one build; refuse, listing them, rather
    than steering every build. With none live the request goes unnamed, and the wait for its
    ack says no live build answered.
    """
    if session:
        return session
    if len(live) == 1:
        return str(live[0].get("pid"))
    if not live:
        return ""
    listing = ", ".join(f"pid {record.get('pid')} ({record.get('model') or 'no model'})"
                        for record in live)
    raise ValueError(f"more than one live build ({listing}): name the session to steer")


def wait_for_ack(workdir: Union[str, Path], request_id: str, timeout: float,
                 poll_seconds: float = 0.05) -> Optional[SteerAckEvent]:
    """The ack for `request_id`, or None when none lands within `timeout` seconds (no live run)."""
    bus, seq = _SENT.pop(request_id, (None, 0))
    if bus is None:
        bus = Bus(Path(workdir) / BUS_FILE)
    deadline = time.monotonic() + timeout
    for record in bus.tail(seq, poll_seconds=poll_seconds, stop=lambda: time.monotonic() >= deadline):
        event = record.event
        if isinstance(event, SteerAckEvent) and event.id == request_id:
            return event
    return None


class SteerBridge:
    """A daemon thread in the build process that turns bus requests into harness calls.

    It follows the bus from its end when started, so a request left on the log by an earlier build
    is never replayed into this one. An empty nudge or tell is refused rather than queued, because
    an empty user turn would only confuse the model. The bridge acts only on requests naming its
    own session (its pid by default): two builds on one workdir share one bus, and one stop must
    never cancel both. Anything for another session is left for that build's own bridge, silently,
    so the sender waits on the one ack of the build it named.
    """

    def __init__(self, harness: Any, bus_path: Union[str, Path], poll_seconds: float = 0.25,
                 session: Optional[str] = None):
        self.harness = harness
        self.bus_path = Path(bus_path)
        self.poll_seconds = poll_seconds
        self.session = session if session is not None else str(os.getpid())
        self._acks = Bus(self.bus_path, agent="steer")
        self._stop = threading.Event()
        self._stop_asked = threading.Event()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kullback-steer", daemon=True)

    def start(self) -> SteerBridge:
        """Start following; returns once the starting point is fixed, so a request sent after
        this call is always seen. The live record is left only then, so a request sent once the
        record appears is never read past."""
        self._thread.start()
        self._started.wait(5)
        if self._started.is_set():
            self._register()
        return self

    def _register(self) -> None:
        """One live record beside the bus, named for this pid and removed on close."""
        try:
            model = getattr(getattr(self.harness, "model", None), "name", None)
            record = {"workdir": str(Path(self.bus_path.parent).expanduser().absolute()),
                      "pid": os.getpid(), "status": "running", "model": model,
                      "updated_at": time.time()}
            _session_file(self.bus_path.parent, os.getpid()).write_text(
                json.dumps(record), encoding="utf-8")
        except OSError:
            pass

    @property
    def stop_asked(self) -> bool:
        """A stop request came in. The harness only cancels a run that is going, so a caller that
        starts runs one after another reads this before starting the next."""
        return self._stop_asked.is_set()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.poll_seconds * 4))
        try:
            _session_file(self.bus_path.parent, os.getpid()).unlink(missing_ok=True)
        except OSError:
            pass

    def _run(self) -> None:
        records = Bus(self.bus_path).tail(poll_seconds=self.poll_seconds, stop=self._stop.is_set,
                                         from_end=True)
        self._started.set()
        for record in records:
            event = record.event
            if isinstance(event, SteerRequestEvent):
                ack = self._take(event)
                if ack is not None:
                    self._acks.append(ack)

    def _take(self, event: SteerRequestEvent) -> Optional[SteerAckEvent]:
        def ack(outcome: str, reason: str = "") -> SteerAckEvent:
            return SteerAckEvent(id=event.id, kind=event.kind, outcome=outcome, reason=reason,
                                 session=self.session)

        if event.session != self.session:
            return None
        if event.kind == "stop":
            self._stop_asked.set()
            self.harness.cancel()
            return ack("cancel_asked", "the run stops before its next step")
        if not event.text.strip():
            return ack("refused", f"a {event.kind} needs text")
        if event.kind == "nudge":
            self.harness.steer(event.text)
            return ack("queued", "delivered before the next model turn")
        self.harness.follow_up(event.text)
        return ack("queued", "delivered when the run would otherwise stop")
