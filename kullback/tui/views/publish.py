"""Check publish readiness: the checklist rows and the word, never an upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import on, work
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Static

from kullback.hub.checklist import READY_RELEASE, publish_checklist, readiness


class PublishView(Widget):
    """Run the publish checklist in a thread and show the exact command when ready."""

    TITLE = "Publish"

    def __init__(self, workdir: Any) -> None:
        """Keep the workdir; checking stages the package in tmp, it never uploads."""
        super().__init__()
        self.workdir = Path(workdir)

    def compose(self):  # type: ignore[override]
        yield Input(placeholder="Repo as organisation/name", id="repo")
        yield Button("Check", id="check")
        yield DataTable(id="rows")
        yield Static("", id="readiness")
        yield Static("", id="command")

    def on_mount(self) -> None:
        table = self.query_one("#rows", DataTable)
        table.add_column("ok", key="ok")
        table.add_column("check", key="check")
        table.add_column("detail", key="detail")

    @on(Button.Pressed, "#check")
    def _checked(self) -> None:
        repo = self.query_one("#repo", Input).value.strip()
        self.query_one("#readiness", Static).update("Checking...")
        self._check(repo)

    @work(thread=True)
    def _check(self, repo: str) -> None:
        """Run the checklist off the UI thread; it only reads and stages in tmp."""
        rows = publish_checklist(self.workdir, repo)
        word = readiness(rows)
        self.app.call_from_thread(self._show_rows, rows, word, repo)

    def _show_rows(self, rows: list, word: str, repo: str) -> None:
        table = self.query_one("#rows", DataTable)
        table.clear()
        for row in rows:
            table.add_row("x" if row.done else " ", row.name, row.detail)
        self.query_one("#readiness", Static).update(word)
        if word == READY_RELEASE:
            self.query_one("#command", Static).update(
                f"Run: kullback publish --workdir {self.workdir} --repo {repo}")
        else:
            self.query_one("#command", Static).update("")
