"""The Publish view shows the checklist and never uploads."""

from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable, Input, Static

from conftest import ViewApp, table_text, wait_until
from kullback.tui.views import publish as publish_view
from kullback.tui.views.publish import PublishView

pytestmark = pytest.mark.anyio


async def test_publish_shows_the_checklist_and_uploads_nothing(tmp_path):
    before = {path for path in tmp_path.rglob("*")}
    view = PublishView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#repo", Input).value = "acme/support-env"
        await pilot.click("#check")
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 5)
        words = {"ready to release", "preview only", "not ready"}
        await wait_until(
            pilot, lambda: str(view.query_one("#readiness", Static).content) in words)
        text = table_text(table)
        for name in ("fidelity over Tasks", "runner frozen", "leak scan",
                     "HF token", "repo name"):
            assert name in text
        assert view.query_one(Button).disabled is False
    assert {path for path in tmp_path.rglob("*")} == before


async def test_publish_shows_an_error_as_one_sentence_and_the_app_keeps_running(
        tmp_path, monkeypatch):
    """A hub error reads as one line; the next check still runs, so the app survived."""

    def boom(workdir, repo):
        raise RuntimeError("hub down\nsecond line")

    monkeypatch.setattr(publish_view, "publish_checklist", boom)
    view = PublishView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#repo", Input).value = "acme/support-env"
        await pilot.click("#check")
        pane = view.query_one("#readiness", Static)
        await wait_until(pilot, lambda: str(pane.content) == "hub down")
        monkeypatch.undo()
        view.query_one("#check", Button).press()
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 5)
