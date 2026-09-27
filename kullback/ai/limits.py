"""One rate limiter per provider key, shared by every caller on that key.

A team key allows a fixed number of requests a minute, and parallel Runs that each
retry on their own multiply every 429 instead of sharing the wait. Every model call
takes a slot from the bucket for its provider key first; a 429 parks every waiter on
that bucket for the delay the provider asked for, not only the caller that drew it.
Buckets come from the model catalogue (pricing.rate_from_catalog): no rate row means
no limiter, exactly today's behaviour.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from kullback.ai.retry import RETRY_POLL_SECONDS

# A minute of starts is what a requests_per_minute cap counts.
WINDOW_SECONDS = 60.0


@dataclass(frozen=True)
class RateSpec:
    """What the catalogue names for one provider entry or model row: either cap, or both."""

    requests_per_minute: Optional[int] = None
    in_flight: Optional[int] = None
    # A model row's rate is that model's own bucket; a provider entry's is one bucket
    # shared by all that provider's models without a row rate.
    model_specific: bool = True


class Bucket:
    """The limiter for one provider key: at most in_flight concurrent slots, at most
    requests_per_minute starts in any rolling 60 s, and no starts while a cooldown runs.

    Thread safe: Runs play in threads, so every field below lives behind one condition.
    Async callers never block the loop on it; they poll try_acquire through acquire_async.
    """

    def __init__(
        self,
        name: str,
        *,
        requests_per_minute: Optional[int] = None,
        in_flight: Optional[int] = None,
        clock: Optional[Any] = None,
    ):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.in_flight = in_flight
        self._clock = clock or time.monotonic
        self._cond = threading.Condition()
        self._starts: deque[float] = deque()
        self._flying = 0
        self._cooldown_until: Optional[float] = None
        self.calls = 0
        self.waits = 0
        self.wait_seconds = 0.0
        self.rate_limited = 0

    def now(self) -> float:
        """This bucket's clock, real or the fake one a test passed in."""
        return self._clock()

    def try_acquire(self) -> bool:
        """One slot now, or False when the bucket is full, metered or cooling down."""
        with self._cond:
            now = self._clock()
            self._prune(now)
            if not self._ready(now):
                return False
            self._take(now)
            return True

    def acquire(self) -> None:
        """One slot, waiting until the bucket has one. Threads share the wait."""
        start = self._clock()
        stalled = False
        with self._cond:
            while True:
                now = self._clock()
                self._prune(now)
                if self._ready(now):
                    self._take(now)
                    if stalled:
                        self.waits += 1
                        self.wait_seconds += now - start
                    return
                stalled = True
                self._cond.wait(timeout=self._hint(now))

    @contextmanager
    def slot(self) -> Iterator["Bucket"]:
        """A slot held for one call, released on every exit path."""
        self.acquire()
        try:
            yield self
        finally:
            self.release()

    def release(self) -> None:
        """Hand a held slot back and wake the waiters."""
        with self._cond:
            if self._flying > 0:
                self._flying -= 1
            self._cond.notify_all()

    def update(self, *, requests_per_minute: Optional[int], in_flight: Optional[int]) -> None:
        """Newest catalogue wins: replace the caps, waking waiters a loosened cap frees."""
        with self._cond:
            self.requests_per_minute = requests_per_minute
            self.in_flight = in_flight
            self._cond.notify_all()

    def note_rate_limited(self, delay_seconds: float) -> None:
        """Park every waiter for delay_seconds: a 429 one caller drew, shared by all."""
        with self._cond:
            until = self._clock() + max(0.0, delay_seconds)
            if self._cooldown_until is None or until > self._cooldown_until:
                self._cooldown_until = until
            self.rate_limited += 1
            self._cond.notify_all()

    def note_wait(self, seconds: float) -> None:
        """Count a wait an async poller already sat through."""
        if seconds <= 0:
            return
        with self._cond:
            self.waits += 1
            self.wait_seconds += seconds

    def counts(self) -> dict[str, float]:
        """What happened on this bucket: calls, waits, wait_seconds, 429s."""
        with self._cond:
            return {
                "calls": self.calls,
                "waits": self.waits,
                "wait_seconds": self.wait_seconds,
                "rate_limited": self.rate_limited,
            }

    def _prune(self, now: float) -> None:
        """Forget the starts that left the 60 s window."""
        while self._starts and self._starts[0] <= now - WINDOW_SECONDS:
            self._starts.popleft()

    def _ready(self, now: float) -> bool:
        """Whether a start now fits under every cap. Callers prune first."""
        if self.in_flight is not None and self._flying >= self.in_flight:
            return False
        if self.requests_per_minute is not None and len(self._starts) >= self.requests_per_minute:
            return False
        return self._cooldown_until is None or now >= self._cooldown_until

    def _take(self, now: float) -> None:
        self._starts.append(now)
        self._flying += 1
        self.calls += 1

    def _hint(self, now: float) -> Optional[float]:
        """How long until a cap may lift, or None when only a release lifts one."""
        hints = []
        if self._cooldown_until is not None and now < self._cooldown_until:
            hints.append(self._cooldown_until - now)
        if self.requests_per_minute is not None and len(self._starts) >= self.requests_per_minute:
            hints.append(self._starts[0] + WINDOW_SECONDS - now)
        return min(hints) if hints else None


async def acquire_async(bucket: Bucket, signal: Optional[Any] = None) -> bool:
    """One slot without blocking the event loop: poll, honouring the cancel signal.

    False means the Run was cancelled while waiting; the caller posts nothing.
    """
    start = bucket.now()
    stalled = False
    while not bucket.try_acquire():
        stalled = True
        if signal is not None and signal.is_cancelled():
            return False
        await asyncio.sleep(RETRY_POLL_SECONDS)
    if stalled:
        bucket.note_wait(bucket.now() - start)
    return True


def note_cooldown(bucket: Optional[Bucket], status: int, delay_seconds: float) -> None:
    """Park every waiter for a delay one caller drew. Anything but a 429, or no bucket, is a no-op."""
    if bucket is None or status != 429:
        return
    bucket.note_rate_limited(delay_seconds)


_registry_lock = threading.Lock()
_buckets: dict[tuple, Bucket] = {}


def _unique_name(wanted: str) -> str:
    """A snapshot key no other bucket holds: numbered suffixes, never key material.

    Call with the registry lock held.
    """
    taken = {bucket.name for bucket in _buckets.values()}
    if wanted not in taken:
        return wanted
    number = 2
    while f"{wanted} #{number}" in taken:
        number += 1
    return f"{wanted} #{number}"


def bucket_for(
    *,
    provider: str,
    host: str,
    key: str,
    key_name: str = "",
    model: str,
    spec: RateSpec,
) -> Bucket:
    """The process-wide bucket for one provider key, made once and shared after.

    Scope is the key VALUE's fingerprint, not the variable name: two handles under one
    variable with different values hold independent limits. The readable name keeps the
    variable name and never carries the key or its fingerprint.
    """
    scope = (host, key, model) if spec.model_specific else (host, key)
    name = _bucket_name(provider, host, key_name, model, spec.model_specific)
    with _registry_lock:
        bucket = _buckets.get(scope)
        if bucket is None:
            bucket = Bucket(
                _unique_name(name),
                requests_per_minute=spec.requests_per_minute,
                in_flight=spec.in_flight,
            )
            _buckets[scope] = bucket
        elif (bucket.requests_per_minute, bucket.in_flight) != (spec.requests_per_minute,
                                                                spec.in_flight):
            bucket.update(requests_per_minute=spec.requests_per_minute, in_flight=spec.in_flight)
        return bucket


def _bucket_name(provider: str, host: str, key_name: str, model: str, model_specific: bool) -> str:
    """A readable bucket name: it names the key variable, never the key."""
    what = model if model_specific else "*"
    return f"{provider} {host} {key_name or 'no-key'} {what}"


def snapshot() -> dict[str, dict[str, float]]:
    """Every bucket's counters by name, for experiment scripts and logs."""
    with _registry_lock:
        buckets = list(_buckets.values())
    return {bucket.name: bucket.counts() for bucket in buckets}


def reset() -> None:
    """Drop every bucket. Tests start clean; live scripts start counted."""
    with _registry_lock:
        _buckets.clear()
