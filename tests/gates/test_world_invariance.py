"""World-invariance: a body that answers distinct worlds alike fails when the recordings did not (D247).

The world here is an invented weaving workshop: looms with a pass (warp or weft). Two Tasks saw
the same loom on two worlds. A body that answers both alike never read the world.
"""
from __future__ import annotations

from conftest import PTR
from kullback.runner.records import ToolCall

try:
    from kullback.gates.tool_runs import WORLD_INVARIANCE_STAGE, body_world_invariance_gate
except ImportError:
    from kullback.builder.sandbox import body_world_invariance_gate
    WORLD_INVARIANCE_STAGE = "world_invariance"


def _call(call_id: str, args: dict, result, name: str = "loom_pass") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args, result=result, raw_ptr=PTR)


def _ok(value) -> dict:
    return {"ok": True, "value": value}


def test_two_worlds_recording_varies_body_constant_fails():
    """(a) The body gave one answer across two worlds; the recordings of those calls did not."""
    calls = [
        _call("c1", {"loom_id": "LM-10"}, {"phase": "warp"}),
        _call("c2", {"loom_id": "LM-10"}, {"phase": "weft"}),
    ]
    constant = [_ok({"phase": "warp"})] * 2
    ruling = body_world_invariance_gate(calls, constant, ["dawn", "dusk"])
    assert ruling.passed is False
    assert ruling.stage == WORLD_INVARIANCE_STAGE
    assert ruling.metrics["worlds"] == 2
    assert ruling.metrics["invariant_calls"] == 2
    line = ruling.failures[0]
    assert "loom_pass" in line
    assert "2 worlds" in line
    assert "2 calls" in line
    assert "warp" not in line and "weft" not in line and "LM-10" not in line


def test_recording_also_constant_passes():
    """(b) The recordings already answered every call the same way, so a constant body is faithful."""
    same = {"phase": "warp"}
    calls = [_call("c1", {"loom_id": "LM-10"}, same), _call("c2", {"loom_id": "LM-10"}, same)]
    ruling = body_world_invariance_gate(calls, [_ok(same)] * 2, ["dawn", "dusk"])
    assert ruling.passed is True
    assert ruling.metrics.get("recorded_constant") is True
    assert ruling.failures == []


def test_body_that_varies_with_the_world_passes():
    """(c) Distinct worlds, distinct recordings, and the body followed the recordings."""
    calls = [
        _call("c1", {"loom_id": "LM-10"}, {"phase": "warp"}),
        _call("c2", {"loom_id": "LM-10"}, {"phase": "weft"}),
    ]
    results = [_ok({"phase": "warp"}), _ok({"phase": "weft"})]
    ruling = body_world_invariance_gate(calls, results, ["dawn", "dusk"])
    assert ruling.passed is True, ruling.failures
    assert ruling.metrics["invariant_calls"] == 0
    assert ruling.failures == []
