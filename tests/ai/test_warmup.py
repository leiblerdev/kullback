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


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _until(condition):
    deadline = time.monotonic() + 5.0
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        threading.Event().wait(0.002)


def _start(gate, keys, name, log, hold=None, fail=False):
    def run():
        try:
            with gate.first(keys):
                log.append(("start", name))
                if hold is not None:
                    hold.wait(5.0)
                log.append(("end", name))
                if fail:
                    raise RuntimeError("boom")
        except RuntimeError as exc:
            log.append(("raised", name, str(exc)))

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _join(*threads):
    for thread in threads:
        thread.join(5.0)
        assert not thread.is_alive()


def test_with_the_stage_prefix_warm_the_first_call_on_a_new_task_goes_at_once_and_the_same_task_waits_for_it():
    gate, log, hold = warmup.Gate(), [], threading.Event()
    with gate.first(["stage"]):
        pass
    first = _start(gate, ["stage", "taskA"], "a1", log, hold)
    _until(lambda: ("start", "a1") in log)
    second = _start(gate, ["stage", "taskA"], "a2", log, hold)
    _until(lambda: gate.waited == 1)
    assert ("start", "a2") not in log
    other = _start(gate, ["stage", "taskB"], "b1", log)
    _until(lambda: ("end", "b1") in log)
    assert gate.waited == 1
    hold.set()
    _join(first, second, other)
    assert log.index(("end", "a1")) < log.index(("start", "a2"))


def test_two_calls_waiting_on_the_stage_prefix_share_a_task_prefix_so_one_goes_first_and_the_other_waits():
    gate, log, hold, stage_hold = warmup.Gate(), [], threading.Event(), threading.Event()
    stage = _start(gate, ["stage"], "stage", log, stage_hold)
    _until(lambda: ("start", "stage") in log)
    one = _start(gate, ["stage", "task"], "one", log, hold)
    two = _start(gate, ["stage", "task"], "two", log, hold)
    _until(lambda: gate.waited == 2)
    stage_hold.set()
    _until(lambda: any(e[0] == "start" and e[1] in ("one", "two") for e in log))
    hold.set()
    _join(stage, one, two)
    starts = [e[1] for e in log if e[0] == "start" and e[1] != "stage"]
    ends = [e[1] for e in log if e[0] == "end" and e[1] != "stage"]
    assert log.index(("end", starts[0])) < log.index(("start", starts[1]))
    assert starts == ends
    assert gate.waited == 2


def test_the_wait_deadline_is_shared_across_waits_so_a_late_recheck_goes_at_once_and_claims_nothing():
    clock = _Clock()
    gate, log, hold, stage_hold = warmup.Gate(clock=clock), [], threading.Event(), threading.Event()
    stage = _start(gate, ["stage"], "stage", log, stage_hold)
    _until(lambda: ("start", "stage") in log)
    one = _start(gate, ["stage", "task"], "one", log, hold)
    two = _start(gate, ["stage", "task"], "two", log, hold)
    _until(lambda: gate.waited == 2)
    clock.now = warmup.MAX_WAIT_S + 10.0
    stage_hold.set()
    _until(lambda: ("start", "one") in log and ("start", "two") in log)  # both inside the block together
    assert sum(1 for e in gate.entries.values() if e is gate.entries["task"]) == 1
    hold.set()
    _join(stage, one, two)
    assert gate.waited == 2


def test_waited_ms_is_the_total_held_time_and_a_call_is_counted_once_however_many_waits_it_did():
    clock, seen = _Clock(), []
    gate, log, hold = warmup.Gate(clock=clock), [], threading.Event()
    stage = _start(gate, ["stage"], "stage", log, hold)
    _until(lambda: ("start", "stage") in log)

    def waiter():
        with gate.first(["stage"]):
            seen.append(warmup.waited_ms())

    thread = threading.Thread(target=waiter)
    thread.start()
    _until(lambda: gate.waited == 1)
    clock.now = 2.0
    hold.set()
    _join(stage, thread)
    assert seen == [2000] and gate.waited == 1


def test_a_call_whose_longest_prefix_is_warm_goes_at_once_and_registers_nothing():
    gate = warmup.Gate()
    with gate.first(["stage"]):
        pass
    before = dict(gate.entries)
    with gate.first(["stage"]):
        pass
    assert gate.entries == before and gate.waited == 0


def test_a_failed_first_call_raises_and_one_waiter_becomes_the_first_instead_of_all_going_cold_together():
    gate, log, fail_hold, hold = warmup.Gate(), [], threading.Event(), threading.Event()
    failing = _start(gate, ["k"], "failing", log, fail_hold, fail=True)
    _until(lambda: ("start", "failing") in log)
    one = _start(gate, ["k"], "one", log, hold)
    two = _start(gate, ["k"], "two", log, hold)
    _until(lambda: gate.waited == 2)
    fail_hold.set()
    _until(lambda: any(e[0] == "start" and e[1] in ("one", "two") for e in log))
    hold.set()
    _join(failing, one, two)
    assert ("raised", "failing", "boom") in log
    starts = [e[1] for e in log if e[0] == "start" and e[1] != "failing"]
    assert log.index(("end", starts[0])) < log.index(("start", starts[1]))
