"""A process pool for independent CPU-bound work, the trust ruling's Tasks and a Harbor export's (D401).

The Runner's `parallel.each` runs model calls on threads, which is right for work that waits on a
provider. Scoring a Task's Runs and replaying an oracle are pure Python that never wait, so threads
share one interpreter lock and gain nothing; a publish spent most of an hour scoring Tasks one at a
time. `each` here runs `fn` in separate processes and hands the results back in the items' order, so
an artifact built from them is byte-identical to the serial loop's. `fn` must be a module-level
function and its items and results picklable; each item is the whole input of one call, so nothing
is shared between processes. One worker (or one item) is a plain loop in this process.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """Every core but two, which keep the laptop and anything else running on it responsive."""
    return max(1, (os.cpu_count() or 1) - 2)


def resolve(workers: Optional[int]) -> int:
    """The worker count asked for, the default where none was; never below one."""
    return default_workers() if workers is None else max(1, int(workers))


def each(items: Iterable[T], fn: Callable[[T], R], workers: Optional[int] = 1) -> list[R]:
    """`fn` over `items`, in order, on at most `workers` processes (None: the default); results in order.

    Processes are spawned, not forked, so a child starts from a clean interpreter whatever threads
    the parent holds. A worker's exception is raised in item order, as the serial loop would raise it.
    """
    batch = list(items)
    count = resolve(workers)
    if count <= 1 or len(batch) <= 1:
        return [fn(item) for item in batch]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(count, len(batch)), mp_context=context) as pool:
        return list(pool.map(fn, batch))
