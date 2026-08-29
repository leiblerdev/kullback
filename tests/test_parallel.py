"""D118: the Builder's independent model calls run on a few threads, and every artifact stays the same."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from harness.builder import build as build_module
from harness.shared import budget, parallel, provider
from harness.shared.provider import MemoModel, Model, ModelReply
from test_e2e import TOOL_BODIES


def test_each_keeps_the_items_order_whatever_finishes_first():
    def slow_then_fast(n):
        time.sleep(0.02 * (5 - n))
        return n * n

    assert parallel.each(range(5), slow_then_fast, workers=5) == [0, 1, 4, 9, 16]
    assert parallel.each(range(5), slow_then_fast, workers=1) == [0, 1, 4, 9, 16]
    assert parallel.each([], slow_then_fast, workers=4) == []


def test_each_raises_the_first_failure_in_item_order_and_cancels_what_had_not_started():
    seen = []

    def work(n):
        seen.append(n)
        if n in (1, 3):
            raise ValueError(f"item {n}")
        return n

    with pytest.raises(ValueError, match="item 1"):
        parallel.each(range(5), work, workers=2)
    assert {0, 1} <= set(seen) and len(seen) < 5  # the running items finish, the queued ones never start


class Echo(Model):
    """Answers with the last user message, so the reply is a function of the request and never of order."""

    name = "test/echo"

    def query(self, messages, tools=None, config=None):
        time.sleep(0.005)
        return ModelReply(content=str(messages[-1]["content"]), usage=provider.Usage(input=10, output=5))


def test_the_ledger_counts_every_threaded_call_once_and_the_memo_hits_per_thread(tmp_path):
    model = budget.BudgetedModel(MemoModel(Echo(), tmp_path), stage="s", workdir=tmp_path)

    def ask(n):
        return model.query([{"role": "user", "content": f"q{n % 10}"}]).content

    with ThreadPoolExecutor(max_workers=8) as pool:
        first = list(pool.map(ask, range(40)))
    assert first == [f"q{n % 10}" for n in range(40)]
    totals = budget.load_totals(tmp_path)
    assert totals["total"]["calls"] == 40 and model.calls == 40
    assert totals["total"]["memo_hits"] == 30  # ten distinct requests, each asked four times
    assert totals["total"]["input"] == 100  # only the ten misses carried usage
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(ask, range(40)))
    totals = budget.load_totals(tmp_path)
    assert totals["total"]["calls"] == 80 and totals["total"]["memo_hits"] == 70
    assert totals["total"]["input"] == 100


def test_the_ceiling_holds_under_concurrent_spend(tmp_path, monkeypatch):
    monkeypatch.setitem(budget.PRICES, "test/echo", {"input": 1_000_000.0, "output": 0.0,
                                                     "cache_read": 0.0, "cache_write": 0.0})
    ceiling = budget.Ceiling(usd=50.0, workdir=tmp_path)  # ten dollars a call; the sixth is over
    model = budget.BudgetedModel(Echo(), stage="s", workdir=tmp_path, ceiling=ceiling)
    errors, done = [], []

    def ask(n):
        try:
            model.query([{"role": "user", "content": f"q{n}"}])
            done.append(n)
        except budget.BudgetExceeded as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(ask, range(20)))
    totals = budget.load_totals(tmp_path)
    assert totals["total"]["usd"] == pytest.approx(ceiling.spent)
    assert len(done) == 4 and len(errors) == 16  # four calls clear 50 USD; the fifth reaches it and stops
    # Every recorded call is in the ledger exactly once: ten dollars each, whatever the threads did.
    assert totals["total"]["calls"] == pytest.approx(totals["total"]["usd"] / 10.0)
    assert all(e.spent >= 50.0 for e in errors)


class ByName(Model):
    """The fixture's tool bodies, chosen by the tool named in the prompt; no state, so threads may share it."""

    name = "test/bodies"

    def query(self, messages, tools=None, config=None):
        text = " ".join(str(m.get("content") or "") for m in messages)
        body = next((b for name, b in TOOL_BODIES.items() if f"Tool: {name}" in text), "return None")
        return provider._as_reply(body)


def _build(tmp_path_factory, request, workers: int) -> Path:
    workdir = tmp_path_factory.mktemp(f"workers{workers}")
    fixture = Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"
    build_module.build(workdir, model=ByName(), files=[fixture], max_attempts=0, workers=workers)
    return workdir


def test_a_parallel_build_writes_the_same_environment_as_a_sequential_one(tmp_path_factory, request):
    one = _build(tmp_path_factory, request, workers=1)
    four = _build(tmp_path_factory, request, workers=4)
    for name in ("bodies.json", "tool_builds.json", "constraints.json", "tasks.json", "task_status.json",
                 "env/db.json"):
        assert (one / name).read_text() == (four / name).read_text(), name
    seq, par = budget.load_totals(one), budget.load_totals(four)
    assert seq["total"]["calls"] == par["total"]["calls"]
    assert {k: v for k, v in seq["stages"].items()} .keys() == par["stages"].keys()


def test_the_memo_hit_flag_is_per_thread(tmp_path):
    memo = MemoModel(Echo(), tmp_path)
    memo.query([{"role": "user", "content": "warm"}])
    flags = {}

    def read(thread):
        if thread == "hit":
            memo.query([{"role": "user", "content": "warm"}])
        else:
            memo.query([{"role": "user", "content": "cold"}])
        time.sleep(0.02)  # the other thread has answered by now; the flag it set must not be ours
        flags[thread] = memo.last_hit

    threads = [threading.Thread(target=read, args=(name,)) for name in ("hit", "miss")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert flags == {"hit": True, "miss": False}
    assert memo.calls == 3 and memo.hits == 1
