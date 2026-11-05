"""Warm before fan-out: of several calls that share a long prompt prefix, one goes first and the rest wait for it.

A provider writes a cache entry only once a call's response has started (Anthropic and Bedrock say so; OpenAI,
Gemini and DeepSeek build theirs within seconds). Calls sent before that each pay to write the same prefix again.
So a call whose prompt shares a long prefix with a call still in flight waits for that call to finish, then goes;
the waiters go together, bounded only by the route's own limiter.

The prefix is read at step boundaries: the tools, then each message in turn, the last message left out (it is the
part that differs). A call looks for the longest of its boundary prefixes that a call is warming or has warmed within TTL_S. Warming:
it waits for that call (all waits of one call share one deadline of MAX_WAIT_S; past it the call goes at once and
claims nothing), then looks again. Warm, or none found: every boundary longer than that one is cold, so this call
claims them all as its own entries and goes first; when there are none, it goes at once and claims nothing.

This nests the warm-up. With the stage prefix warm, the first call on a new Task prefix beneath it claims that
prefix and goes at once, and other calls on the same Task prefix wait for it. When several calls wait on the stage
prefix and share a longer Task prefix, one of them goes first on the Task prefix after the stage warms, and the
others wait for it. When a first call raises, its entries are removed before its waiters are released, so one of
them becomes the first on a look again, instead of all of them going cold together.

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
        """(the warming entry to wait for, or None) and (this call's own entry, when it goes first on cold prefixes)."""
        now = self.clock()
        with self.lock:
            found = -1
            for index in range(len(keys) - 1, -1, -1):  # the longest live prefix decides
                entry = self.entries.get(keys[index])
                if entry is None:
                    continue
                if not entry.done.is_set():
                    return entry, None
                if now - entry.at <= TTL_S:
                    found = index
                    break
            cold = keys[found + 1 :]
            if not cold:
                return None, None
            mine = _Entry()
            for key in cold:
                self.entries[key] = mine
            return None, mine

    def _admit(self, keys: list[str]) -> Optional[_Entry]:
        """Wait while the longest live prefix is warming (one deadline for all waits), then claim what is cold."""
        deadline, held = None, 0.0
        while True:
            wait_for, mine = self._claim(keys)
            if wait_for is None:
                break
            start = self.clock()
            if deadline is None:
                deadline = start + MAX_WAIT_S
                self.waited += 1  # a call counts once however many waits it does
            if deadline - start <= 0:
                break
            wait_for.done.wait(deadline - start)
            held += self.clock() - start
        _LOCAL.waited_ms = round(held * 1000.0)
        return mine

    def _release(self, mine: _Entry, ok: bool) -> None:
        with self.lock:
            if ok:
                mine.at = self.clock()
            else:
                for key in [k for k, e in self.entries.items() if e is mine]:
                    del self.entries[key]
            mine.done.set()

    @contextmanager
    def first(self, keys: list[str]) -> Iterator[None]:
        """Run the block once the longest live prefix is warm, claiming the colder prefixes beneath it."""
        _LOCAL.waited_ms = 0.0
        if not keys or not enabled():
            yield
            return
        mine = self._admit(keys)
        try:
            yield
        except BaseException:
            if mine is not None:
                self._release(mine, ok=False)  # waiters look again; one of them becomes the first
            raise
        if mine is not None:
            self._release(mine, ok=True)

    def reset(self) -> None:
        with self.lock:
            self.entries.clear()
            self.waited = 0


GATE = Gate()


def gate(model: str, messages: list[dict], tools: Optional[list[dict]] = None) -> Any:
    """The context a live call runs in: held back while a call sharing its longest long prefix warms it."""
    return GATE.first(boundaries(model, messages, tools))
