"""Warm before fan-out (kullback/ai/warmup.py): of calls sharing a long prefix, one goes first and the rest wait."""

from __future__ import annotations

import threading
import time

import pytest

from kullback.ai import warmup

SHARED = [{"role": "system", "content": "S" * 3000}, {"role": "user", "content": "a long shared conversation"}]


@pytest.fixture(autouse=True)
def fresh_gate(monkeypatch):
    monkeypatch.delenv(warmup.ENV_VAR, raising=False)
    warmup.GATE.reset()
    yield
    warmup.GATE.reset()


def _fan_out(questions, first_takes=0.05):
    events, lock = [], threading.Lock()

    def call(question):
        messages = SHARED + [{"role": "user", "content": question}]
        with warmup.gate("route/model-a", messages):
            with lock:
                events.append(("start", question, time.monotonic()))
            time.sleep(first_takes)
            with lock:
                events.append(("end", question, time.monotonic()))

    threads = [threading.Thread(target=call, args=(q,)) for q in questions]
    threads[0].start()
    time.sleep(0.01)
    for thread in threads[1:]:
        thread.start()
    for thread in threads:
        thread.join()
    return events


def test_calls_sharing_a_long_prefix_wait_for_the_first_to_finish_then_go_together():
    events = _fan_out(["q1", "q2", "q3", "q4"])
    first_end = next(t for kind, q, t in events if kind == "end" and q == "q1")
    others = [t for kind, q, t in events if kind == "start" and q != "q1"]
    assert all(t >= first_end for t in others)
    assert max(others) - min(others) < 0.04  # released together, not one by one
    assert warmup.GATE.waited == 3


def test_a_prefix_warmed_within_the_cache_lifetime_lets_later_calls_go_at_once():
    _fan_out(["q1"])
    _fan_out(["q2", "q3"])
    assert warmup.GATE.waited == 0


def test_a_short_prefix_or_a_switched_off_gate_never_holds_a_call_back(monkeypatch):
    assert warmup.boundaries("route/model-a", [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]) == []
    monkeypatch.setenv(warmup.ENV_VAR, "0")
    _fan_out(["q1", "q2"])
    assert warmup.GATE.waited == 0


def test_the_longest_shared_prefix_decides_and_the_model_is_part_of_it():
    longer = SHARED + [{"role": "assistant", "content": "Read."}, {"role": "user", "content": "situation"}]
    keys = warmup.boundaries("route/model-a", longer + [{"role": "user", "content": "q"}])
    assert keys[: len(warmup.boundaries("route/model-a", SHARED + [{"role": "user", "content": "q"}]))] == \
        warmup.boundaries("route/model-a", SHARED + [{"role": "user", "content": "q"}])
    assert warmup.boundaries("route/model-b", longer) != warmup.boundaries("route/model-a", longer)
