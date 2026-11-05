"""Warm before fan-out: of several calls that share a long prompt prefix, one goes first and the rest wait for it.

A provider writes a cache entry only once a call's response has started (Anthropic and Bedrock say so; OpenAI,
Gemini and DeepSeek build theirs within seconds). Calls sent before that each pay to write the same prefix again.
So a call whose prompt shares a long prefix with a call still in flight waits for that call to finish, then goes;
the waiters go together, bounded only by the route's own limiter.

The prefix is read at step boundaries: the tools, then each message in turn, the last message left out (it is the
part that differs). A call registers every boundary prefix of at least MIN_PREFIX_CHARS as warming, and later calls
look for the longest of their own boundary prefixes that a call has warmed or is warming. Warming: wait for it
(at most MAX_WAIT_S). Warmed within TTL_S: go at once. Neither: this call is the first, and goes at once.

No text is kept: a boundary is a hash of the model id and the prefix bytes. With KULLBACK_WARMUP=0 every call goes
at once (for a measured comparison).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

ENV_VAR = "KULLBACK_WARMUP"
MIN_PREFIX_CHARS = 2048  # below this a prefix is under any provider's cacheable minimum (512 tokens and up)
TTL_S = 300.0  # the shortest provider cache lifetime: a prefix warmed longer ago may be cold again
MAX_WAIT_S = 120.0  # a waiter never waits longer than this for the first call
_LOCAL = threading.local()


def waited_ms() -> float:
    """How long this thread's current call was held back for a first call (0 when it was not)."""
    return getattr(_LOCAL, "waited_ms", 0.0)


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "1").strip() not in ("0", "false", "off")


def boundaries(model: str, messages: list[dict], tools: Optional[list[dict]] = None) -> list[str]:
    """The hashes of every step-boundary prefix of at least MIN_PREFIX_CHARS, shortest first; the last message is
    never part of one."""
    digest = hashlib.sha256(model.encode("utf-8"))
    size = 0
    if tools:
        text = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        digest.update(text.encode("utf-8"))
        size += len(text)
    out = []
    for message in messages[:-1]:
        text = json.dumps(message, sort_keys=True, ensure_ascii=False, default=str)
        digest.update(b"\x00" + text.encode("utf-8"))
        size += len(text)
        if size >= MIN_PREFIX_CHARS:
            out.append(digest.copy().hexdigest())
    return out


class _Entry:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.at = 0.0  # when the first call finished


class Gate:
    """The process-wide record of which prefixes are warming or warm."""

    def __init__(self, clock=time.monotonic) -> None:
        self.lock, self.entries, self.clock = threading.Lock(), {}, clock
        self.waited = 0  # calls that waited for a first call (for tests and reports)

    def _claim(self, keys: list[str]) -> tuple[Optional[_Entry], Optional[_Entry]]:
        """(the entry to wait for, or None) and (this call's own entry, when it goes first on some prefix)."""
        now = self.clock()
        with self.lock:
            for key in reversed(keys):  # the longest shared prefix decides
                entry = self.entries.get(key)
                if entry is None:
                    continue
                if not entry.done.is_set():
                    return entry, None
                if now - entry.at <= TTL_S:
                    return None, None
            mine = _Entry()
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or (entry.done.is_set() and now - entry.at > TTL_S):
                    self.entries[key] = mine
            return None, mine

    @contextmanager
    def first(self, keys: list[str]) -> Iterator[None]:
        """Run the block once the longest shared prefix is warm, or at once when this call warms it."""
        _LOCAL.waited_ms = 0.0
        if not keys or not enabled():
            yield
            return
        wait_for, mine = self._claim(keys)
        if wait_for is not None:
            self.waited += 1
            start = time.monotonic()
            wait_for.done.wait(MAX_WAIT_S)
            _LOCAL.waited_ms = round((time.monotonic() - start) * 1000.0)
        try:
            yield
        finally:
            if mine is not None:
                mine.at = self.clock()
                mine.done.set()  # a failed first call releases its waiters too; one of them writes the entry

    def reset(self) -> None:
        with self.lock:
            self.entries.clear()
            self.waited = 0


GATE = Gate()


def gate(model: str, messages: list[dict], tools: Optional[list[dict]] = None) -> Any:
    """The context a live call runs in: held back while a call sharing its longest long prefix warms it."""
    return GATE.first(boundaries(model, messages, tools))
