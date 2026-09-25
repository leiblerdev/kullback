"""The Synthesise view refuses without a ceiling and launches nothing."""

from __future__ import annotations

import pytest
from textual.widgets import Input, RichLog, Static

from conftest import ViewApp, wait_until
from kullback.tui.views.synthesise import SynthesiseView

pytestmark = pytest.mark.anyio


async def test_synthesise_refuses_an_empty_ceiling(tmp_path):
    calls: list = []
    view = SynthesiseView(tmp_path, launcher=lambda command, workdir: calls.append(command))
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#per-archetype", Input).value = "2"
        await pilot.click("#start")
        await wait_until(
            pilot, lambda: "ceiling" in str(view.query_one("#status", Static).content).lower())
        assert calls == []


async def test_synthesise_refuses_when_no_mode_is_chosen(tmp_path):
    calls: list = []
    view = SynthesiseView(tmp_path, launcher=lambda command, workdir: calls.append(command))
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#ceiling", Input).value = "5"
        await pilot.click("#start")
        await wait_until(
            pilot, lambda: "buckets" in str(view.query_one("#status", Static).content).lower())
        assert calls == []


async def test_synthesise_passes_buckets_archetypes_and_model_as_given(tmp_path):
    calls: list = []
    log = tmp_path / "out.log"
    log.write_text("")

    def launch(command: list, workdir):
        calls.append(command)
        return log

    view = SynthesiseView(tmp_path, launcher=launch)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#buckets", Input).value = "band_a=2"
        view.query_one("#archetypes", Input).value = "arch-1"
        view.query_one("#per-archetype", Input).value = "3"
        view.query_one("#model", Input).value = "test/model"
        view.query_one("#ceiling", Input).value = "5"
        view.query_one("#seed", Input).value = "s1"
        await pilot.click("#from-domain")
        await pilot.click("#start")
        await wait_until(pilot, lambda: len(calls) == 1)
        command = calls[0]
        assert "--bucket" in command and "band_a=2" in command
        assert "--archetype" in command and "arch-1" in command
        assert "--per-archetype" in command and "3" in command
        assert "--model" in command and "test/model" in command
        assert "--from-domain" in command
        assert "--seed" in command and "s1" in command
        assert "--ceiling-usd" in command


async def test_synthesise_shows_the_command_output_as_it_grows(tmp_path):
    log = tmp_path / "out.log"
    log.write_text("")
    view = SynthesiseView(tmp_path, launcher=lambda command, workdir: log)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#buckets", Input).value = "band_a=2"
        view.query_one("#ceiling", Input).value = "5"
        await pilot.click("#start")

        def logged() -> str:
            return "\n".join(strip.text for strip in view.query_one("#log", RichLog).lines)

        with open(log, "a") as handle:
            handle.write("shaped band_a: 2 walks\n")
        await wait_until(pilot, lambda: "shaped band_a" in logged())
        with open(log, "a") as handle:
            handle.write("stored task-1 under synthetic/\n")
        await wait_until(pilot, lambda: "task-1" in logged())
