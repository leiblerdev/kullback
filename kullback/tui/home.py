"""Where the workdir stands: the Home view of the Textual app.

The entry screen of the app, for a newcomer before anything else. The gradient
name on top, then the same six rows and the same next step the line screen's
/doctor prints, then the recent builds of this workdir only, so the machine's
other worktrees stay one key away on ctrl+r. Each checklist row carries its fix
command, and enter on a row puts that command in the app's command line.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from textual import on
from textual.containers import Vertical
from textual.widgets import Label, ListItem, ListView, Static

from kullback.ai.provider import DEFAULT_MODEL
from kullback.tui import banner
from kullback.tui.checklist import next_step, where_it_stands

# How many of this workdir's recent builds Home keeps: the banner and the six
# rows stay visible on an 80x24 terminal with the command line on screen.
MAX_RECENT = 3


def fix_command(next_key: str) -> str:
    """A checklist fix as a command line: slash commands as is, CLI words slashed."""
    if next_key.startswith("/"):
        return next_key
    if next_key.startswith("kullback "):
        return "/" + next_key[len("kullback "):]
    return next_key


class HomeView(Vertical):
    """The banner, the checklist, the next step, and this workdir's recent builds."""

    def __init__(self, workdir: Any, model: Optional[str] = None, base_url: str = "",
                 session: Optional[dict] = None) -> None:
        super().__init__(id="home")
        self.workdir = Path(workdir)
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url
        self.session = session if session is not None else {}
        self.rows: list = []
        self.next_line = ""
        self.recent: list = []
        self._read()
        # One item per row, built once: ListView.clear needs an await before a
        # re-mount, so later reads update these labels in place instead.
        self._items = [self._row_item(index, row) for index, row in enumerate(self.rows)]
        self.checklist = ListView(*self._items, id="home-checklist")
        self.recent_view = Static("", id="home-recent")

    def compose(self):  # type: ignore[override]
        yield Static(banner(), id="home-banner")
        yield self.checklist
        yield self.recent_view

    def on_mount(self) -> None:
        self._render_rows()

    def refresh_view(self) -> None:
        """Read the workdir again; the app calls this every time Home is shown."""
        self._read()
        if self.is_mounted:
            self._render_rows()

    def _read(self) -> None:
        """The rows, the next step and this workdir's recent builds, off disk and env."""
        from kullback.runner import heartbeat

        self.rows = where_it_stands(self.workdir, dict(os.environ), self.model, self.session)
        self.next_line = f"next: {next_step(self.rows)}"
        here = str(Path(self.workdir).expanduser().absolute())
        self.recent = [record for record in heartbeat.read_all()
                       if str(Path(str(record.get("workdir") or "")).expanduser().absolute())
                       == here][:MAX_RECENT]

    @staticmethod
    def _row_line(row: Any) -> str:
        mark = "x" if row.done else " "
        return f"[{mark}] {row.name:<10} {row.detail}"

    def _row_item(self, index: int, row: Any) -> ListItem:
        return ListItem(Label(self._row_line(row)), id=f"home-row-{index}")

    def _render_rows(self) -> None:
        """The rows as selectable lines, the recent builds under them."""
        for item, row in zip(self._items, self.rows, strict=True):
            item.query_one(Label).update(self._row_line(row))
        lines = ["recent builds in this workdir"]
        if not self.recent:
            lines.append("none")
        for record in self.recent:
            from kullback.runner import heartbeat

            alive = heartbeat.alive(record.get("pid"))
            mark = "live" if alive and record.get("status") == "running" else str(
                record.get("status") or "done")
            lines.append(f"{mark} {record.get('model') or 'no model'}")
        self.recent_view.update("\n".join(lines))

    @on(ListView.Selected, "#home-checklist")
    def _row_chosen(self, event: ListView.Selected) -> None:
        """Enter on a row puts its fix command in the command line."""
        try:
            index = list(self.checklist.children).index(event.item)
        except ValueError:
            return
        if 0 <= index < len(self.rows):
            self.app.prefill_command(fix_command(self.rows[index].next_key))

    def row_command(self, index: int) -> str:
        """The command row `index` puts in the command line, for tests to read."""
        return fix_command(self.rows[index].next_key)

    def text(self) -> str:
        """The view as plain text, for tests to read without rendering."""
        lines = ["where this workdir stands"]
        for row in self.rows:
            mark = "x" if row.done else " "
            lines.append(f"[{mark}] {row.name:<10} {row.detail}")
        lines.append("")
        lines.append(self.next_line)
        lines.append("")
        lines.append("recent builds in this workdir")
        if not self.recent:
            lines.append("none")
        for record in self.recent:
            lines.append(str(record.get("status") or "done"))
        return "\n".join(lines)
