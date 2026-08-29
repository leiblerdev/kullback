"""A bounded worker pool for the Builder's independent model calls (D118).

Every stage that asks the model once per tool, per Task, per Run or per policy sentence used to
ask in a single line, and a live build's wall clock was the provider's latency summed over some
3,400 calls: two hours for retail, one for airline. The items are independent, so `each` runs
`fn` over them on a few threads and hands the results back in the items' order, which keeps
every artifact byte-identical to the sequential build. What the threads share is protected at
its own seam: the budget ledger takes a lock (budget.py), the memo's per-call hit flag is
thread-local (provider.MemoModel), and an http client is created once. Nothing here retries or
reorders; a worker's exception is the stage's exception, raised in item order, and the items
not yet started are cancelled.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

DEFAULT_WORKERS = 8  # the CLI's default; build() itself defaults to one, so a scripted test model stays in order

T = TypeVar("T")
R = TypeVar("R")


def each(items: Iterable[T], fn: Callable[[T], R], workers: int = 1) -> list[R]:
    """`fn` over `items`, in order, on at most `workers` threads; results in the items' order.

    With one worker (or one item) this is a plain loop, so a caller that never asked for
    concurrency gets none. The first exception, in item order, is raised as soon as its item's
    turn comes: the items already running finish, the ones not yet started are cancelled, which
    is what a spend ceiling reached on one thread wants (budget.BudgetedModel refuses the running
    ones' next call anyway).
    """
    batch = list(items)
    if workers <= 1 or len(batch) <= 1:
        return [fn(item) for item in batch]
    with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
        return list(pool.map(fn, batch))
