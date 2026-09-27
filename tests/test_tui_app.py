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


@pytest.mark.anyio
async def test_build_passes_the_endpoint_to_the_child(tmp_path, monkeypatch):
    _live_env(monkeypatch)
    launched: list = []
    app = KullbackApp(workdir=tmp_path, base_url="http://localhost:9")
    async with app.run_test() as pilot:
        await pilot.press("3")
        view = app.query_one(BuildView)
        view.launcher = launched.append
        view.ceiling_input.value = "25"
        await pilot.click("#start")
        await pilot.pause(0.3)
        assert len(launched) == 1
        assert launched[0][launched[0].index("--base-url") + 1] == "http://localhost:9"


@pytest.mark.anyio
async def test_each_view_key_and_palette_entry_mounts_its_view(tmp_path):
    from kullback.tui.views import PublishView, RunsView, SynthesiseView, TasksView, TracesView

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        for key, cls in (("2", TracesView), ("5", TasksView), ("6", RunsView)):
            await pilot.press(key)
            await pilot.pause(0.8)
            assert isinstance(app.current, cls)
        for name, cls in (("traces", TracesView), ("tasks", TasksView), ("runs", RunsView),
                          ("publish", PublishView), ("synthesise", SynthesiseView)):
            # A view's input eats ctrl+k for editing, so focus leaves it first.
            app.screen.set_focus(None)
            await pilot.pause(0.3)
            await pilot.press("ctrl+k")
            await pilot.pause(0.8)
            await pilot.click(f"#view-{name}")
            await pilot.pause(0.8)
            assert isinstance(app.current, cls)


def test_tui_plain_opens_the_line_screen(tmp_path):
    from kullback import cli

    result = CliRunner().invoke(cli.app, ["tui", "--workdir", str(tmp_path), "--plain"],
                                input="/quit\n")
    assert result.exit_code == 0
    assert "where this workdir stands" in result.output


def test_palette_table_lists_every_listed_command(tmp_path):
    """One table holds every palette command, so the palette cannot drift from it."""
    from kullback.tui.app import PALETTE_COMMANDS

    app = KullbackApp(workdir=tmp_path)
    table = app._command_table()
    for name in PALETTE_COMMANDS:
        assert name in table, name
    assert table["sessions"] == app.action_sessions
    assert table["quit"] == app.action_quit_app


@pytest.mark.anyio
async def test_each_palette_command_opens_its_view_or_modal(tmp_path):
    """Every palette entry acts: views mount, commands show the slash output."""
    import os

    from kullback.ai.provider import DEFAULT_MODEL
    from kullback.tui import HELP, Screen, _keys
    from kullback.tui.app import LOGIN_HINT, PALETTE_COMMANDS, MessageModal, SessionsModal
    from kullback.tui.home import HomeView

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause(0.8)
        assert set(app._command_table()) >= set(PALETTE_COMMANDS)
        for name, cls in (("build", BuildView), ("watch", WatchView), ("home", HomeView)):
            app._palette_chosen(name)
            await pilot.pause(0.5)
            assert isinstance(app.current, cls), name
        app._palette_chosen("sessions")
        await pilot.pause(0.5)
        assert isinstance(app.screen, SessionsModal)
        app.screen.dismiss(None)
        await pilot.pause(0.5)
        screen = Screen(tmp_path, None, "")
        expected = {
            "status": (screen._status_renderable().plain
                       if hasattr(screen._status_renderable(), "plain")
                       else str(screen._status_renderable())),
            "keys": _keys(dict(os.environ), model=DEFAULT_MODEL, host="").plain,
            "login": screen._login_status().plain + "\n" + LOGIN_HINT,
            "help": HELP,
        }
        for name, body in expected.items():
            app._palette_chosen(name)
            await pilot.pause(0.5)
            assert isinstance(app.screen, MessageModal), name
            assert app.screen._body == body, name
            assert app.screen._body, name
            await pilot.press("escape")
            await pilot.pause(0.5)
        app._palette_chosen("logout")
        await pilot.pause(0.5)
        assert isinstance(app.screen, MessageModal)
        assert "cleared" in app.screen._body
        await pilot.press("escape")
        await pilot.pause(0.5)


def test_default_launcher_creates_a_new_workdir_for_the_build_log(tmp_path):
    """The first build names a workdir that is not there yet, so the log needs it made."""
    import sys

    from kullback.tui.build_view import default_launcher

    workdir = tmp_path / "nested" / "work"
    proc = default_launcher([sys.executable, "-c", "pass"], workdir)
    assert proc.wait(timeout=30) == 0
    assert (workdir / "build.log").is_file()


@pytest.mark.anyio
async def test_build_launch_failure_shows_in_the_view_instead_of_raising(tmp_path, monkeypatch):
    """What the launcher cannot open is named in the view, not raised past it."""
    _live_env(monkeypatch)

    def boom(argv):
        raise OSError("disk gone")

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("3")
        view = app.query_one(BuildView)
        view.launcher = boom
        view.ceiling_input.value = "25"
        await pilot.click("#start")
        await pilot.pause(0.3)
        assert "build did not start" in str(app.query_one("#build-message").content)
        assert "disk gone" in str(app.query_one("#build-message").content)
