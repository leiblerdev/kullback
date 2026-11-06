"""One call-log row for each streamed call, and the gateway's model check, so a stage cannot forget either.

Every streaming adapter hands its provider events through `logged_stream`, as HttpModel.query does for a blocking
call. The row has the usage of the response end (or what the parser last saw, on a stream that failed), the client's
wall time, the HTTP requests of every post and the error, if any. A stream the consumer closed early is logged too.

The check is the handle's own check_reply, run as soon as the answer names a model: at the headers where the gateway
sends them, else at the first chunk that says, before any of that chunk's output is passed on. A stream that never
names one is not refused. A refusal ends the stream with an error event, the way a stream reports every failure.

The row is built on a worker thread: it can wait on the gateway's price table and writes a file, and the event loop
never waits for either. What the row says of the call (its sink, its context, when it started and ended) is taken
on the loop first, so a cancel that leaves the thread running changes none of it. With logging off there is no row,
so no price is read.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Optional

from kullback.ai import calllog
from kullback.ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEnd,
)
from kullback.ai._sse import StreamFacts
from kullback.ai.http import Posted
from kullback.ai.http_errors import ModelMismatch

CLOSED_EARLY = "the stream closed before the answer ended"
ERROR_CHARS = 200


class _Watch:
    """One streamed call as it goes: what the events said, then its one row."""

    def __init__(self, handle: Any, wire: list, tools: Optional[list], config: Any, facts: StreamFacts):
        self.handle, self.wire, self.tools, self.config, self.facts = handle, wire, tools, config, facts
        self.started, self.sent_at = time.monotonic(), time.time()
        self.usage: Any = None
        self.error: Optional[str] = None
        self.ended, self.checked = False, False

    def posted(self) -> Posted:
        """The stream as the blocking path's Posted, for the handle's check and its reply meta."""
        parser = self.facts.parser
        data = {key: value for key, value in (("usage", getattr(parser, "raw_usage", None)),
                                              ("model", getattr(parser, "body_model", None))) if value is not None}
        return Posted(data, self.facts.status, max(1, self.facts.posts), self.facts.headers)

    def refusal(self) -> Optional[ModelMismatch]:
        """The mismatch, once the answer has named a model; checked once."""
        if self.checked or not self.handle.names_a_model(self.posted()):
            return None
        self.checked = True
        try:
            self.handle.check_reply(self.posted())
        except ModelMismatch as error:
            return error
        return None

    def see(self, event: ProviderEvent) -> None:
        if isinstance(event, ProviderErrorEvent):
            self.error = str(event.message)[:ERROR_CHARS]
        elif isinstance(event, ProviderResponseEnd):
            self.ended, self.usage = True, event.message.usage

    def finish(self, call: tuple, ended: float, finished_at: float) -> None:
        """The row, exactly once, in that call's log and context, of a stream over at `ended` and `finished_at`. The
        log never fails the call it logs."""
        usage = self.usage if self.usage is not None else getattr(self.facts.parser, "_usage", None)
        error = self.error if self.ended or self.error else CLOSED_EARLY
        with calllog.pinned(call):
            try:
                meta = self.handle._meta(self.posted(), self.facts.body, self.config, self.facts.failed)
            except Exception:  # noqa: BLE001
                meta = {}
            meta.pop("warmup_wait_ms", None)  # that thread-local belongs to the last blocking call
            self.handle._log(self.wire, self.tools, _reported(usage), max(1, self.facts.posts),
                             (ended - self.started) * 1000.0, error, meta, (self.sent_at, finished_at))


def _reported(usage: Any) -> Any:
    """The usage, or None when the stream reported none (all zero): unknown, never a free call."""
    if usage is None or not (usage.input or usage.output or usage.cache_read or usage.cache_write):
        return None
    return usage


async def logged_stream(handle: Any, events: AsyncIterator[ProviderEvent], wire: list, tools: Optional[list],
                        config: Any, facts: StreamFacts) -> AsyncIterator[ProviderEvent]:
    """The events as they come, checked against the model asked, and one call-log row when the stream is over."""
    watch = _Watch(handle, wire, tools, config, facts)
    try:
        async for event in events:
            refused = None if isinstance(event, ProviderErrorEvent) else watch.refusal()
            if refused is not None:
                watch.error = f"{type(refused).__name__}: {str(refused)[:ERROR_CHARS]}"
                yield ProviderErrorEvent(message=str(refused), data={"attempts": max(1, facts.posts)})
                return
            watch.see(event)
            yield event
    except Exception as error:
        watch.error = f"{type(error).__name__}: {str(error)[:ERROR_CHARS]}"
        raise
    finally:
        try:
            if calllog.enabled():
                # A cancel here leaves the thread to finish the row, so all it says of the call is taken now.
                await asyncio.to_thread(watch.finish, calllog.this_call(), time.monotonic(), time.time())
        finally:
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()
