"""The app shell: bare kullback, the / menu, and slash commands inside the app.

Pilot tests that type the commands: bare `kullback` opens the app, "/" opens
the menu and filters, /ingest ingests a fixture file and turns Home's traces
row done, the /login modal saves a key where `kullback login` writes it,
/build and /watch switch views, and an unknown command says so. No test calls
a model and no test names a provider: the modal's own provider list decides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kullback.ai.provider import DEFAULT_MODEL
from kullback.tui import COMMANDS, Screen
from kullback.tui.app import KullbackApp, OutputModal
from kullback.tui.build_view import BuildView
from kullback.tui.command_line import CommandLine
from kullback.tui.home import HomeView
from kullback.tui.login_modal import LoginModal
from kullback.tui.watch import WatchView

FIXTURE = Path("tests/fixtures/worlds/world_a/traces_small.json").absolute()


def test_bare_kullback_opens_the_app_on_the_current_directory(tmp_path, monkeypatch):
    """Bare `kullback` opens KullbackApp on this directory instead of missing a command."""
    from kullback import cli
    from kullback.tui import app as app_module

    monkeypatch.chdir(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(app_module.KullbackApp, "run",
                        lambda self: seen.setdefault("workdir", self.workdir))
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0
    assert seen.get("workdir") == Path(".")


def test_kullback_help_still_lists_its_commands():
    """The callback leaves --help and every subcommand alone."""
    from kullback import cli

    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "tui" in result.output


@pytest.mark.anyio
async def test_slash_opens_the_menu_and_typing_filters_it_to_ingest(tmp_path):
    """Typing / lists every command and typing more narrows it to /ingest."""
    from textual.widgets import Input

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.click("#cmd-input")
        await pilot.press("slash")
        await pilot.pause(0.5)
        cmd = app.query_one("#cmdline", CommandLine)
        assert cmd.menu.display
        assert cmd.menu_names() == [row[0] for row in COMMANDS]
        await pilot.press("i", "n", "g")
        await pilot.pause(0.5)
        assert cmd.menu_names() == ["ingest"]
        await pilot.press("tab")
        await pilot.pause(0.3)
        assert app.query_one("#cmd-input", Input).value == "/ingest "
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert "ingest needs a file" in app.last_said


@pytest.mark.anyio
async def test_ingest_ingests_the_file_and_home_traces_row_turns_done(tmp_path):
    """/ingest on a fixture file ingests into the workdir and Home shows it done."""
    from textual.widgets import Input

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.click("#cmd-input")
        app.query_one("#cmd-input", Input).value = f"/ingest {FIXTURE}"
        await pilot.pause(0.5)
        await pilot.press("enter")
        await pilot.pause(2.0)
        assert (tmp_path / "ingest_summary.json").is_file()
        assert isinstance(app.current, HomeView)
        assert "[x] traces" in app.current.text()
        assert app.last_said == "ingested 1 file: 3 traces, 9 tool calls, gate passed"
        assert isinstance(app.screen, OutputModal)
        assert "traces_small.json" in app.screen.text()
        assert str(FIXTURE) not in app.screen.text()
        assert "ready" in app.screen.text()
        await pilot.press("escape")


@pytest.mark.anyio
async def test_login_modal_saves_the_key_where_kullback_login_writes_it(tmp_path):
    """Bare /login opens the modal and saving writes auth.json as `kullback login` does."""
    from textual.widgets import Input, Select

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        app.run_command("/login")
        await pilot.pause(0.5)
        modal = app.screen
        assert isinstance(modal, LoginModal)
        assert modal.providers
        provider = next(iter(modal.providers))
        expected_var = Screen._key_var_for(provider, modal.providers[provider])
        modal.query_one("#login-provider", Select).value = provider
        modal.query_one("#login-key", Input).value = "marker-test-key-9"
        await pilot.click("#login-save")
        await pilot.pause(0.5)
        body = json.loads(Path(os.environ["KULLBACK_AUTH_FILE"]).read_text(encoding="utf-8"))
        assert body.get("keys", {}).get(expected_var) == "marker-test-key-9"
        from kullback.ai.credentials import stored_names

        assert expected_var in stored_names()
        assert app.model == modal.providers[provider]
        assert "marker-test-key-9" not in app.last_said


@pytest.mark.anyio
async def test_build_opens_the_build_view_with_prefilled_arguments(tmp_path):
    """/build opens Build and its flags prefill the view's own fields."""
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        app.run_command("/build --ceiling-usd 25")
        await pilot.pause(0.5)
        assert isinstance(app.current, BuildView)
        assert app.current.ceiling_init == "25"
        assert "build opened" in app.last_said


@pytest.mark.anyio
async def test_watch_one_opens_watch_on_that_session(tmp_path):
    """/watch N leaves Home for the Watch view on that session's workdir."""
    from kullback.runner import heartbeat

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        assert isinstance(app.current, HomeView)
        heartbeat.beat(tmp_path, DEFAULT_MODEL, "running")
        app.run_command("/watch 1")
        await pilot.pause(0.8)
        assert isinstance(app.current, WatchView)


@pytest.mark.anyio
async def test_an_unknown_command_says_so_and_names_help(tmp_path):
    """A verb outside the table says so in one line instead of doing nothing."""
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test():
        app.run_command("/zzz")
        assert "no command zzz" in app.last_said
        assert "/help" in app.last_said


@pytest.mark.anyio
async def test_home_rows_carry_their_fix_commands_into_the_command_line(tmp_path):
    """Enter on a checklist row is one prefill away from running its fix."""
    from textual.widgets import Input

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        home = app.current
        assert isinstance(home, HomeView)
        assert home.row_command(0) == "/login"
        assert home.row_command(2) == "/ingest"
        app.prefill_command(home.row_command(2))
        assert app.query_one("#cmd-input", Input).value == "/ingest"


@pytest.mark.anyio
async def test_home_shows_the_gradient_banner_above_the_checklist(tmp_path):
    """Home opens with the gradient name, six selectable rows and the next step."""
    from textual.widgets import Static

    from kullback.tui import banner
    app = KullbackApp(workdir=tmp_path)
    assert app.title == "kullback"
    async with app.run_test():
        assert isinstance(app.current, HomeView)
        assert app.query_one("#home-banner", Static) is not None
        assert len(banner().plain.strip().split("\n")) == 5
        assert len(app.current.rows) == 6
        assert "next:" in app.current.text()


@pytest.mark.anyio
async def test_help_lists_every_command_with_its_meaning(tmp_path):
    """/help lists the whole shared table with one line per command."""
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test() as pilot:
        app.run_command("/help")
        await pilot.pause(0.5)
        assert isinstance(app.screen, OutputModal)
        said = app.screen.text()
        for name, _, blurb in COMMANDS:
            assert f"/{name}" in said and blurb in said
        await pilot.press("escape")


@pytest.mark.anyio
async def test_screenshots_capture_home_and_the_open_menu(tmp_path):
    """Home after an ingest and the open / menu, at 100x30, as two SVG files."""
    from textual.widgets import Input

    app = KullbackApp(workdir=tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.click("#cmd-input")
        app.query_one("#cmd-input", Input).value = f"/ingest {FIXTURE}"
        await pilot.pause(0.5)
        await pilot.press("enter")
        await pilot.pause(2.0)
        await pilot.press("escape")
        await pilot.pause(0.5)
        app.save_screenshot("/tmp/kb-shell-home.svg")
        await pilot.click("#cmd-input")
        await pilot.press("slash")
        await pilot.pause(0.8)
        app.save_screenshot("/tmp/kb-shell-menu.svg")
    assert Path("/tmp/kb-shell-home.svg").is_file()
    assert Path("/tmp/kb-shell-menu.svg").is_file()


@pytest.mark.anyio
async def test_screenshots_capture_the_login_modal(tmp_path):
    """Bare /login opens the modal with providers and a masked key field."""
    app = KullbackApp(workdir=tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        app.run_command("/login")
        await pilot.pause(0.8)
        assert isinstance(app.screen, LoginModal)
        app.save_screenshot("/tmp/kb-shell-login.svg")
        await pilot.press("escape")
    assert Path("/tmp/kb-shell-login.svg").is_file()
