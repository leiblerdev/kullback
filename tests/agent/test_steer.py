"""Steering a live run through the bus: a request from any process reaches the running harness as a
nudge, a tell or a stop, and the run answers each one with an ack on the same bus."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

import kullback
from kullback.agent import steer
from kullback.agent.bus import Bus
from kullback.agent.events import SteerAckEvent, SteerRequestEvent
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import AgentTool
from kullback.ai.provider import TestModel
from tests.agent.conftest import call, collect, reply

TOOL_SECONDS = 0.2


class WorkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int


class WorkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: int


async def _work(args: WorkArgs) -> WorkResult:
    await asyncio.sleep(TOOL_SECONDS)
    return WorkResult(done=args.step)


def _harness(workdir: Path, steps: int, *tail: str) -> tuple[AgentHarness, TestModel]:
    """A harness whose model asks for `steps` slow tool calls, then says each line of `tail`."""
    model = TestModel([reply(None, call("work", {"step": i}, f"c{i}")) for i in range(steps)]
                      + [reply(line) for line in tail])
    harness = AgentHarness(model, tools=[AgentTool("work", "Do one slow step.", WorkArgs, WorkResult, _work)],
                           bus=Bus(workdir / "bus.jsonl", agent="builder"))
    return harness, model


def _first_call_with(model: TestModel, text: str) -> int:
    for index, sent in enumerate(model.calls):
        if any(text in str(message.get("content")) for message in sent["messages"] if isinstance(message, dict)):
            return index
    return -1


def _acks(workdir: Path) -> list[SteerAckEvent]:
    return [e for e in Bus(workdir / "bus.jsonl").events() if isinstance(e, SteerAckEvent)]


def _on_first_tool(harness: AgentHarness, action) -> None:
    fired: list = []

    def subscriber(event):
        if event.type == "tool_execution_start" and not fired:
            fired.append(True)
            action()

    harness.subscribe(subscriber)


def test_a_nudge_sent_from_another_process_reaches_the_running_harness_before_the_next_model_turn(tmp_path):
    harness, model = _harness(tmp_path, 3, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02).start()
    text = "nudge from another process"
    code = ("import sys; from kullback.agent import steer; "
            f"rid = steer.request({str(tmp_path)!r}, 'nudge', {text!r}, sender='second screen', "
            f"session={str(os.getpid())!r}); "
            "ack = steer.wait_for_ack(" f"{str(tmp_path)!r}" ", rid, 4); "
            "sys.exit(0 if ack is not None and ack.outcome == 'queued' else 1)")
    env = {**os.environ, "PYTHONPATH": str(Path(kullback.__file__).parents[1])}
    # Sent while the first slow tool is about to run, from its own interpreter, and acked there.
    _on_first_tool(harness, lambda: subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                                   timeout=10))
    try:
        collect(harness.prompt("build it"))
    finally:
        bridge.close()
    assert _first_call_with(model, text) == 1
    requests = [e for e in Bus(tmp_path / "bus.jsonl").events() if isinstance(e, SteerRequestEvent)]
    assert [(r.kind, r.sender) for r in requests] == [("nudge", "second screen")]
    assert [(a.kind, a.outcome) for a in _acks(tmp_path)] == [("nudge", "queued")]


def test_a_tell_is_delivered_only_when_the_run_would_stop_and_is_acked(tmp_path):
    harness, model = _harness(tmp_path, 2, "done", "done after the tell")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02).start()
    text = "then write the report"
    _on_first_tool(harness, lambda: steer.wait_for_ack(tmp_path, steer.request(tmp_path, "tell", text,
                                                                               session=str(os.getpid())), 3))
    try:
        collect(harness.prompt("build it"))
    finally:
        bridge.close()
    # calls 0 to 2 are the prompt and the two tool results; the model then says "done" and the
    # tell starts the fourth call.
    assert _first_call_with(model, text) == 3 and len(model.calls) == 4
    assert [(a.kind, a.outcome) for a in _acks(tmp_path)] == [("tell", "queued")]


def test_a_stop_cancels_the_run_at_its_next_step_and_is_acked(tmp_path):
    harness, model = _harness(tmp_path, 6, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02).start()
    _on_first_tool(harness, lambda: steer.wait_for_ack(tmp_path, steer.request(tmp_path, "stop",
                                                                               session=str(os.getpid())), 3))
    try:
        collect(harness.prompt("build it"))
    finally:
        bridge.close()
    assert len(model.calls) == 1
    assert [(a.kind, a.outcome) for a in _acks(tmp_path)] == [("stop", "cancel_asked")]


def test_a_request_already_on_the_bus_when_the_bridge_starts_is_not_replayed(tmp_path):
    old = steer.request(tmp_path, "nudge", "left over from an earlier build", session=str(os.getpid()))
    harness, model = _harness(tmp_path, 0, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02).start()
    try:
        fresh = steer.request(tmp_path, "nudge", "this build's own nudge", session=str(os.getpid()))
        assert steer.wait_for_ack(tmp_path, fresh, 3) is not None
        collect(harness.prompt("build it"))
    finally:
        bridge.close()
    assert [a.id for a in _acks(tmp_path)] == [fresh] and old != fresh
    assert _first_call_with(model, "this build's own nudge") == 0
    assert _first_call_with(model, "left over") == -1


def test_an_empty_nudge_is_refused_with_a_reason_and_nothing_is_queued(tmp_path):
    harness, model = _harness(tmp_path, 0, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02).start()
    try:
        ack = steer.wait_for_ack(tmp_path, steer.request(tmp_path, "nudge", "  ",
                                                         session=str(os.getpid())), 3)
    finally:
        bridge.close()
    assert ack is not None and ack.outcome == "refused" and ack.reason == "a nudge needs text"


def test_waiting_for_an_ack_nobody_writes_gives_none_after_the_timeout(tmp_path):
    request_id = steer.request(tmp_path, "nudge", "anyone there")
    assert steer.wait_for_ack(tmp_path, request_id, 0.1) is None


def test_a_request_for_another_session_is_ignored_by_this_bridge(tmp_path):
    harness, model = _harness(tmp_path, 0, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02,
                               session="s-one").start()
    try:
        other = steer.request(tmp_path, "nudge", "for the other build", session="s-two")
        assert steer.wait_for_ack(tmp_path, other, 0.3) is None
        own = steer.request(tmp_path, "nudge", "for this build", session="s-one")
        assert steer.wait_for_ack(tmp_path, own, 3) is not None
    finally:
        bridge.close()
    collect(harness.prompt("build it"))
    assert _first_call_with(model, "for this build") == 0
    assert _first_call_with(model, "for the other build") == -1
    assert [(a.id, a.outcome) for a in _acks(tmp_path)] == [(own, "queued")]


def test_target_session_names_the_only_live_build_and_refuses_on_several():
    live = [{"pid": 41, "model": "m/one"}]
    assert steer.target_session(live) == "41"
    assert steer.target_session(live, session="7") == "7"
    assert steer.target_session([]) == ""
    with pytest.raises(ValueError, match="pid 41.*pid 42"):
        steer.target_session([{"pid": 41, "model": "m/one"}, {"pid": 42, "model": "m/two"}])


def test_a_request_sent_once_the_live_record_appears_is_seen_by_the_bridge(tmp_path, monkeypatch):
    import time

    times = {}
    real_tail = steer.Bus.tail

    def timed_tail(self, *args, **kwargs):
        if kwargs.get("from_end"):
            times.setdefault("tail_fixed", time.time())
        return real_tail(self, *args, **kwargs)

    monkeypatch.setattr(steer.Bus, "tail", timed_tail)
    harness, model = _harness(tmp_path, 0, "done")
    bridge = steer.SteerBridge(harness, tmp_path / "bus.jsonl", poll_seconds=0.02,
                               session="s-live")
    real_register = bridge._register

    def timed_register():
        times["registered"] = time.time()
        return real_register()

    bridge._register = timed_register
    bridge.start()
    try:
        assert times["tail_fixed"] <= times["registered"], "the live record is readable before the bridge follows from there"
        request_id = steer.request(tmp_path, "nudge", "right after discovery", session="s-live")
        ack = steer.wait_for_ack(tmp_path, request_id, 3)
        assert ack is not None and ack.outcome == "queued"
        collect(harness.prompt("build it"))
    finally:
        bridge.close()
    assert _first_call_with(model, "right after discovery") == 0
