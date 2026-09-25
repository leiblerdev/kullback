"""Watch: the live transcript, the sidebar numbers, nudges, and stopping."""

from __future__ import annotations

import json

import pytest
from textual.widgets import Input

from kullback.agent.bus import Bus
from kullback.agent.events import SteerRequestEvent
from kullback.agent.steer import SteerBridge
from kullback.tui.app import KullbackApp
from kullback.tui.watch import StopModal, WatchView


class HarnessStandIn:
    """A tiny harness with the three calls the steer bridge makes, recording each."""

    def __init__(self) -> None:
        self.steered: list = []
        self.followed: list = []
        self.cancelled = 0

    def steer(self, text: str) -> None:
        self.steered.append(text)

    def follow_up(self, text: str) -> None:
        self.followed.append(text)

    def cancel(self) -> None:
        self.cancelled += 1


def _bus_event(workdir, text="hello nudge"):
    bus = Bus(workdir / "bus.jsonl", agent="test")
    bus.append(SteerRequestEvent(id="req1", kind="nudge", text=text, sender="t"))
    return bus


def _bridge(workdir, harness):
    return SteerBridge(harness, workdir / "bus.jsonl", poll_seconds=0.05).start()


@pytest.mark.anyio
async def test_watch_shows_bus_events_and_the_input_stays_put(tmp_path):
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.pause(0.3)
        assert isinstance(app.current, WatchView)
        _bus_event(tmp_path)
        await pilot.pause(1.5)
        assert "hello nudge" in app.current.log_text()
        steer = app.query_one("#steer", Input)
        assert steer.is_mounted
        assert isinstance(app.current, WatchView)


@pytest.mark.anyio
async def test_watch_sidebar_shows_the_live_counts(tmp_path):
    (tmp_path / "task_status.json").write_text(json.dumps({"task-1": {}}), encoding="utf-8")
    (tmp_path / "rounds.json").write_text(json.dumps([{
        "round": 1, "counts": {"fidelity": 0, "tasks": 1, "trusted": 0, "refused_count": 0,
                              "probes_passing": 0, "spend": {"total": 0.0}}}]), encoding="utf-8")
    (tmp_path / "gates.json").write_text(json.dumps(
        [{"stage": "intent", "pass": False, "failures": ["too vague"]}]), encoding="utf-8")
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.pause(0.5)
        assert isinstance(app.current, WatchView)
        sidebar = app.current.sidebar_text()
        assert "tasks 1" in sidebar
        assert "trusted 0" in sidebar
        assert "open 1" in sidebar
        assert "1 failed" in sidebar


@pytest.mark.anyio
async def test_enter_in_watch_sends_a_nudge_request_on_the_bus(tmp_path):
    harness = HarnessStandIn()
    bridge = _bridge(tmp_path, harness)
    try:
        app = KullbackApp(workdir=tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("4")
            await pilot.pause(0.3)
            assert isinstance(app.current, WatchView)
            steer = app.query_one("#steer", Input)
            steer.value = "push harder"
            steer.focus()
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(2.0)
            assert harness.steered == ["push harder"]
            assert "steer nudge queued" in app.current.log_text()
    finally:
        bridge.close()


@pytest.mark.anyio
async def test_esc_asks_before_stopping_and_y_sends_a_stop(tmp_path):
    harness = HarnessStandIn()
    bridge = _bridge(tmp_path, harness)
    try:
        app = KullbackApp(workdir=tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("4")
            await pilot.pause(0.3)
            assert isinstance(app.current, WatchView)
            await pilot.press("escape")
            await pilot.pause(0.5)
            assert isinstance(app.screen, StopModal)
            await pilot.press("y")
            await pilot.pause(2.0)
            assert harness.cancelled == 1
    finally:
        bridge.close()
