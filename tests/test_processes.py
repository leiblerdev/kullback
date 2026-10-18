"""The process pool (D401): results in the items' order, one worker a plain loop, more spread over processes."""

from __future__ import annotations

import os

from kullback import processes


def _square_and_pid(value: int) -> tuple[int, int]:
    return value * value, os.getpid()


def _fail_on_two(value: int) -> int:
    if value == 2:
        raise ValueError("two")
    return value


def test_one_worker_runs_every_item_in_this_process_in_order():
    out = processes.each([3, 1, 2], _square_and_pid, workers=1)
    assert [value for value, _ in out] == [9, 1, 4]
    assert {pid for _, pid in out} == {os.getpid()}


def test_several_workers_answer_in_the_items_order_from_other_processes():
    out = processes.each(list(range(6)), _square_and_pid, workers=2)
    assert [value for value, _ in out] == [0, 1, 4, 9, 16, 25]
    assert os.getpid() not in {pid for _, pid in out}


def test_a_worker_exception_reaches_the_caller():
    import pytest

    with pytest.raises(ValueError, match="two"):
        processes.each([1, 2, 3], _fail_on_two, workers=2)


def test_no_worker_count_takes_every_core_but_two_capped_at_four_and_never_less_than_one():
    assert processes.resolve(None) == max(1, min((os.cpu_count() or 1) - 2, 4))
    assert processes.resolve(0) == 1 and processes.resolve(3) == 3


def test_the_default_never_exceeds_four_workers_however_many_cores(monkeypatch):
    for cores in (1, 3, 6, 8, 64, None):
        monkeypatch.setattr(os, "cpu_count", lambda cores=cores: cores)
        assert 1 <= processes.default_workers() <= 4
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert processes.default_workers() == 4 and processes.resolve(12) == 12
