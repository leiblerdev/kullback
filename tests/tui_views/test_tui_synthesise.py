"""The Synthesise view refuses without a ceiling and launches nothing."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

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
