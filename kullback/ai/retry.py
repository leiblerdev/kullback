"""When a failed provider call is tried again, and how long the caller waits first.

Mirrors tau_ai/retry.py. The blocking policy (RetryPolicy, backoff_delay, retry_after_seconds,
retryable_status) is provider.py's, moved here verbatim and used by the synchronous handles. The
async helpers below are the same rules for the streaming providers, which cannot block a loop.
"""

from __future__ import annotations

import asyncio
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Optional, Protocol

from pydantic import BaseModel

# How often a waiting retry looks at the cancel signal, and where the async backoff starts.
RETRY_POLL_SECONDS = 0.05
RETRY_BASE_DELAY_SECONDS = 0.25


class CancelSignal(Protocol):
    """Anything that can say a Run was cancelled. The agent core passes its own."""

    def is_cancelled(self) -> bool:
        ...


class RetryPolicy(BaseModel):
    """Five attempts, exponential backoff with jitter, nothing clever."""

    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: float = 0.25
    # A provider may ask for a wait longer than the build is willing to sit still for; past
    # this the call gives up rather than blocking the build for hours.
    max_retry_after_s: float = 120.0


def retry_after_seconds(headers: Any, now: Optional[float] = None) -> Optional[float]:
    """Retry-After as seconds, whether the provider sent a count or an HTTP date."""
    raw = None
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            raw = headers[key]
            break
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    stamp = parsedate_to_datetime(str(raw))
    if stamp is None:
        return None
    reference = time.time() if now is None else now
    return max(0.0, stamp.timestamp() - reference)


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Exponential backoff with jitter, capped."""
    delay = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
    return delay + rng.random() * policy.jitter * delay


def retryable_status(status: int) -> bool:
    """Only rate limits and server faults. A 400 is our bug and retrying it wastes money."""
    return status == 429 or status >= 500


def retry_delay_seconds(attempt: int, *, max_delay_seconds: float) -> float:
    """The async backoff: the same doubling, capped by what the provider config allows."""
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    return float(min(max_delay_seconds, base_delay * (2**attempt)))


async def wait_for_retry(delay_seconds: float, *, signal: Optional[CancelSignal]) -> bool:
    """Sleep before a retry, looking at the cancel signal often enough to stop early.

    Returns whether the caller should still try again: a cancelled Run answers False.
    """
    if delay_seconds <= 0:
        return signal is None or not signal.is_cancelled()
    remaining = delay_seconds
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(RETRY_POLL_SECONDS, remaining)
        await asyncio.sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()
