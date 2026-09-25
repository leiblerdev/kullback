"""The app shell, Home, and Build: opening views, refusals, starts, --plain."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kullback.tui.app import KullbackApp
from kullback.tui.build_view import BuildView
from kullback.tui.home import HomeView
from kullback.tui.watch import WatchView


def _live_env(monkeypatch):
    """A workdir where a build may start: live calls on and the model's key set."""
    monkeypatch.setenv("HARNESS_ALLOW_MODEL_REQUESTS", "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")


@pytest.mark.anyio
async def test_the_app_opens_on_home_with_the_checklist_and_next_step(tmp_path):
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test():
        assert isinstance(app.current, HomeView)
        text = app.current.text()
        assert "traces" in text
        assert "next:" in text


@pytest.mark.anyio
async def test_the_app_opens_on_watch_when_a_build_is_live_here(tmp_path):
    from kullback.runner import heartbeat

    heartbeat.beat(tmp_path, "somemodel", "running")
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test():
        assert isinstance(app.current, WatchView)


@pytest.mark.anyio
async def test_build_refuses_an_empty_ceiling_and_starts_nothing(tmp_path, monkeypatch):
    _live_env(monkeypatch)
    launched: list = []
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("3")
        view = app.query_one(BuildView)
        view.launcher = launched.append
        await pilot.click("#start")
        await pilot.pause(0.3)
        assert launched == []
        assert "ceiling" in str(app.query_one("#build-message").content)


@pytest.mark.anyio
async def test_build_starts_a_detached_child_with_the_typed_ceiling(tmp_path, monkeypatch):
    _live_env(monkeypatch)
    launched: list = []
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("3")
        view = app.query_one(BuildView)
        view.launcher = launched.append
        view.ceiling_input.value = "25"
        await pilot.click("#start")
        await pilot.pause(0.3)
        assert len(launched) == 1
        assert "--ceiling-usd" in launched[0]
        assert launched[0][launched[0].index("--ceiling-usd") + 1] == "25"
        assert str(tmp_path) in launched[0]
        assert isinstance(app.current, WatchView)


def test_tui_plain_opens_the_line_screen(tmp_path):
    from kullback import cli

    result = CliRunner().invoke(cli.app, ["tui", "--workdir", str(tmp_path), "--plain"],
                                input="/quit\n")
    assert result.exit_code == 0
    assert "where this workdir stands" in result.output
