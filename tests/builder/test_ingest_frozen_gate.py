"""Tests for the frozen half of admission: the build gate fails only under the floor.

These run only with docs/frozen-patches/intake-seam.patch applied (the founder's re-freeze);
without it the module has no MIN_TASK_ELIGIBLE_SHARE and the whole file skips.
"""

from __future__ import annotations

import pytest

from kullback.gates import artifacts

pytestmark = pytest.mark.skipif(
    not hasattr(artifacts, "MIN_TASK_ELIGIBLE_SHARE"),
    reason="needs the intake-seam frozen patch applied",
)


def trace(trace_id, calls) -> dict:
    return {"trace_id": trace_id, "hash": "h-" + trace_id, "turns": [], "tool_calls": calls}


def good_call(name) -> dict:
    return {"name": name, "args": {}, "result": {"ok": True}}


def bad_call(name) -> dict:
    return {"name": name, "args": {}}


def test_one_bad_recording_among_good_ones_builds():
    traces = [trace("g1", [good_call("a")]), trace("g2", [good_call("b")]),
              trace("g3", [good_call("c")]), trace("b1", [bad_call("d")])]
    out = artifacts.ingest_gate(traces)
    assert out.passed is True
    assert out.metrics["task_eligible"] == 3
    assert out.metrics["eligible_share"] == 0.75
    assert out.failures == []


def test_under_the_floor_the_build_gate_fails_with_the_counts():
    traces = [trace("g1", [good_call("a")]), trace("b1", [bad_call("d")])]
    out = artifacts.ingest_gate(traces)
    assert out.passed is False
    assert any("1 of 2 recordings task-eligible" in line for line in out.failures)
    assert any("d" in line and "neither a result nor an error" in line for line in out.failures)


def test_a_single_bad_recording_still_fails():
    out = artifacts.ingest_gate([trace("b1", [bad_call("d")])])
    assert out.passed is False
    assert any("d" in line for line in out.failures)


def test_a_grader_field_still_fails_above_the_floor():
    leaked = {"name": "a", "args": {"reward_info": 1}, "result": {"ok": True}}
    traces = [trace("g1", [leaked]), trace("g2", [good_call("b")]),
              trace("g3", [good_call("c")]), trace("g4", [good_call("d")])]
    out = artifacts.ingest_gate(traces)
    assert out.passed is False
    assert any("reward_info" in line for line in out.failures)
