"""One bus per workdir: the append-only, sequence-numbered log every harness and every tool writes to.

tau's harness has in-process subscribers and nothing else, so an event is gone the moment the
process is. The bus is the durable form of the same thing: one JSON line per event, a sequence
number that orders every writer in the workdir, and subscribers fed through it, so the tui, the
journal, the feed, a report and a finding collector all read one stream instead of each agent's
internals. Transcripts stay per agent; they are context, not events.

Two processes may append at once (a Builder and an Examiner on one workdir), so a write takes an
exclusive lock on a lock file beside the log, reads whatever lines have appeared since this
process last looked, and appends one whole line with O_APPEND. Sequence numbers are therefore
global and gapless, and no reader ever sees half a line.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from kullback.agent.events import AgentEvent, CustomEvent

__all__ = ["Bus", "BusRecord", "Subscriber"]

Subscriber = Callable[[AgentEvent], Union[None, Awaitable[None]]]

_EVENT = TypeAdapter(AgentEvent)


class BusRecord(BaseModel):
    """One line of the bus: where it sits in the order, when it landed, who wrote it, what it says."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    recorded_at: float
    agent: str = ""
    event: AgentEvent


class Bus:
    """The workdir's event log. `path` is the file; `agent` names this writer on every record."""

    def __init__(self, path: Union[str, Path], agent: str = ""):
        self.path = Path(path)
        self.agent = agent
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._subscribers: list[Subscriber] = []
        self._seq = 0
        self._offset = 0
        self._pending: set[asyncio.Task] = set()

    # --- subscribers ---

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Call `fn` with every event published from this process; returns the unsubscribe."""
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(fn)

        return unsubscribe

    # --- publishing ---

    async def publish(self, event: AgentEvent) -> BusRecord:
        """Append the event and await every subscriber, in registration order."""
        record = self.append(event)
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                await result
        return record

    def publish_sync(self, event: AgentEvent) -> BusRecord:
        """Append the event from synchronous code; an async subscriber is scheduled, or run."""
        record = self.append(event)
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                try:
                    task = asyncio.get_running_loop().create_task(_await(result))
                except RuntimeError:
                    asyncio.run(_await(result))
                else:
                    self._pending.add(task)
                    task.add_done_callback(self._pending.discard)
        return record

    def publish_custom(self, name: str, payload: Optional[dict[str, Any]] = None) -> BusRecord:
        """An application's own event, for what the core has no type for."""
        return self.publish_sync(CustomEvent(name=name, payload=dict(payload or {})))

    def append(self, event: AgentEvent) -> BusRecord:
        """Write one record, under the workdir's lock, and return it with its sequence number."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._catch_up()
            record = BusRecord(
                seq=self._seq + 1, recorded_at=time.time(), agent=self.agent, event=event
            )
            line = record.model_dump_json() + "\n"
            data = line.encode("utf-8")
            handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(handle, data)
                os.fsync(handle)
            finally:
                os.close(handle)
            self._seq = record.seq
            self._offset += len(data)
            return record

    # --- reading ---

    def replay(self, since_seq: int = 0) -> list[BusRecord]:
        """Every record after `since_seq`, in order. A line that does not parse is skipped."""
        return [record for record in self._read_all() if record.seq > since_seq]

    def events(self, since_seq: int = 0) -> list[AgentEvent]:
        return [record.event for record in self.replay(since_seq)]

    def tail(
        self,
        since_seq: int = 0,
        poll_seconds: float = 0.25,
        stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[BusRecord]:
        """Follow the log: every record after `since_seq`, then whatever appears, until `stop`.

        Without a `stop` the follower runs until the consumer stops asking for records, which is
        what a `break` out of the loop does; with one it also ends on its own.
        """
        seq = since_seq
        while True:
            found = False
            for record in self.replay(seq):
                seq = record.seq
                found = True
                yield record
            if stop is not None and stop():
                return
            if not found:
                time.sleep(poll_seconds)

    # --- the file ---

    def _read_all(self) -> list[BusRecord]:
        if not self.path.is_file():
            return []
        records: list[BusRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            record = _parse(line)
            if record is not None:
                records.append(record)
        return records

    def _catch_up(self) -> None:
        """Read whatever another writer appended since this process last wrote, for the next seq."""
        if not self.path.is_file():
            self._seq = 0
            self._offset = 0
            return
        size = self.path.stat().st_size
        if size <= self._offset:
            # A log that shrank was replaced under us; count it again rather than trust the cache.
            if size < self._offset:
                self._seq = max((r.seq for r in self._read_all()), default=0)
                self._offset = size
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            tail = handle.read()
        text = tail.decode("utf-8", errors="replace")
        # A trailing partial line is another writer mid-append; leave it for the next catch-up.
        complete, _, remainder = text.rpartition("\n")
        for line in complete.splitlines():
            record = _parse(line)
            if record is not None:
                self._seq = max(self._seq, record.seq)
        self._offset = size - len(remainder.encode("utf-8"))

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            _lock(handle)
            try:
                yield
            finally:
                _unlock(handle)


def _parse(line: str) -> Optional[BusRecord]:
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        return BusRecord(seq=data["seq"], recorded_at=data["recorded_at"], agent=data.get("agent", ""),
                         event=_EVENT.validate_python(data["event"]))
    except Exception:  # noqa: BLE001 - a line the reader cannot read is a line it skips
        return None


async def _await(value: Awaitable[None]) -> None:
    await value


def _lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - the harness runs on posix
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - the harness runs on posix
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
