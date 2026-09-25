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
        # How far replay and tail have read, and the highest seq up to there: (offset, seq).
        self._reader: tuple[int, int] = (0, 0)

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
                _write_whole(handle, data)
            finally:
                os.close(handle)
            self._seq = record.seq
            self._offset += len(data)
            # _catch_up read every line up to the old offset, so this writer's seq is the highest
            # up to the new one: a later replay or tail on this Bus can start from there.
            if self._offset >= self._reader[0]:
                self._reader = (self._offset, self._seq)
            return record

    # --- reading ---

    def replay(self, since_seq: int = 0) -> list[BusRecord]:
        """Every record after `since_seq`, in order. A line that does not parse is skipped.

        A caller that polls with the last seq it saw reads only the bytes appended since: this
        Bus remembers how far it has read and the highest seq up to there, and starts from that
        offset whenever `since_seq` is at or past it. Asking for older records reads from zero.
        """
        offset, seen = self._reader
        start = offset if since_seq >= seen and self._size() >= offset else 0
        records, _ = self._read_from(start, since_seq)
        return records

    def events(self, since_seq: int = 0) -> list[AgentEvent]:
        return [record.event for record in self.replay(since_seq)]

    def last_seq(self) -> int:
        """The seq of the last whole record on the log, or 0; read off the end, not the whole file."""
        size = self._size()
        chunk = 65536
        while size:
            start = max(0, size - chunk)
            with self.path.open("rb") as handle:
                handle.seek(start)
                data = handle.read(size - start)
            lines = data.split(b"\n")
            # Without a newline before the first piece, that piece may be the middle of a line.
            candidates = lines if start == 0 else lines[1:]
            for line in reversed(candidates):
                record = _parse_bytes(line)
                if record is not None:
                    return record.seq
            if start == 0:
                return 0
            chunk *= 4
        return 0

    def tail(
        self,
        since_seq: int = 0,
        poll_seconds: float = 0.25,
        stop: Optional[Callable[[], bool]] = None,
        from_end: bool = False,
    ) -> Iterator[BusRecord]:
        """Follow the log: every record after `since_seq`, then whatever appears, until `stop`.

        Each poll reads only the bytes appended since the last one, and a trailing partial line
        (a writer mid-append) waits for the next poll. `from_end` starts after the last record
        present now, so a follower that only cares about what happens next never parses the past.
        Without a `stop` the follower runs until the consumer stops asking for records, which is
        what a `break` out of the loop does; with one it also ends on its own.
        """
        # The starting point is fixed now, not at the first `next`: a follower from the end must
        # not miss what lands between this call and its first read.
        seq = since_seq
        if from_end:
            offset = self._whole_size()
            seq = max(seq, self.last_seq())
        else:
            cached, seen = self._reader
            offset = cached if since_seq >= seen and self._size() >= cached else 0
        return self._follow(seq, offset, poll_seconds, stop)

    def _follow(self, seq: int, offset: int, poll_seconds: float,
                stop: Optional[Callable[[], bool]]) -> Iterator[BusRecord]:
        while True:
            if self._size() < offset:
                # A log that shrank was replaced under us; read it again from the start.
                offset = 0
            records, offset = self._read_from(offset, seq)
            for record in records:
                seq = record.seq
                yield record
            if stop is not None and stop():
                return
            if not records:
                time.sleep(poll_seconds)

    # --- the file ---

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def _whole_size(self) -> int:
        """The file's length at a record boundary: taken under the writers' lock, so no append is
        half done."""
        if not self.path.is_file():
            return 0
        with self._locked():
            return self._size()

    def _read_from(self, offset: int, since_seq: int) -> tuple[list[BusRecord], int]:
        """The records after `since_seq` in the whole lines from `offset` on, and the offset after
        the last whole line. The reader cursor moves forward with what was read."""
        if not self.path.is_file():
            return [], 0
        with self.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
        complete, newline, _ = data.rpartition(b"\n")
        if not newline:
            return [], offset
        records: list[BusRecord] = []
        top = 0
        for line in complete.split(b"\n"):
            record = _parse_bytes(line)
            if record is None:
                continue
            top = max(top, record.seq)
            if record.seq > since_seq:
                records.append(record)
        end = offset + len(complete) + 1
        cached, seen = self._reader
        if offset == 0:
            self._reader = (end, top)
        elif offset <= cached < end:
            self._reader = (end, max(top, seen))
        return records, end

    def _read_all(self) -> list[BusRecord]:
        records, _ = self._read_from(0, 0)
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


def _parse_bytes(line: bytes) -> Optional[BusRecord]:
    return _parse(line.decode("utf-8", errors="replace"))


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


def _write_whole(handle: int, data: bytes) -> None:
    """Write every byte of one record, or none: a short write is continued, a failure truncated.

    os.write may write fewer bytes than asked; the rest follows until the count is reached. On
    any exception the file goes back to its length before this record, so no partial line stays
    for the next append to join, and the exception is raised again.
    """
    start = os.fstat(handle).st_size
    view = memoryview(data)
    try:
        while view:
            view = view[os.write(handle, view):]
        os.fsync(handle)
    except BaseException:
        os.ftruncate(handle, start)
        raise
