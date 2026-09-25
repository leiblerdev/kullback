"""The Publish view shows the checklist and never uploads."""

from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable, Input, Static

from conftest import ViewApp, table_text, wait_until
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
