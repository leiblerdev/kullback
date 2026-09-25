"""Where the workdir stands: the Home view of the Textual app.

The entry screen of the app, for a newcomer before anything else. It reads the
same six rows and the same next step the line screen's /doctor prints, then the
recent builds of this workdir only, so the machine's other worktrees stay one
key away on ctrl+r.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from textual.containers import Vertical
from textual.widgets import Static

from kullback.ai.provider import DEFAULT_MODEL
from kullback.tui.checklist import next_step, where_it_stands


class HomeView(Vertical):
    """The checklist, the next step, and this workdir's recent builds."""

    def __init__(self, workdir: Any, model: Optional[str] = None, base_url: str = "") -> None:
        super().__init__(id="home")
        self.workdir = Path(workdir)
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url
        self.panel = Static("", id="home-body")

    def compose(self):  # type: ignore[override]
        yield self.panel

    def on_mount(self) -> None:
        self.refresh_view()

    def refresh_view(self) -> None:
        """Read the workdir again; the app calls this every time Home is shown."""
        from kullback.runner import heartbeat

        rows = where_it_stands(self.workdir, dict(os.environ), self.model)
        lines = ["where this workdir stands"]
        for row in rows:
            mark = "x" if row.done else " "
            lines.append(f"[{mark}] {row.name:<10} {row.detail}")
        lines.append("")
        lines.append(f"next: {next_step(rows)}")
        lines.append("")
        lines.append("recent builds in this workdir")
        here = str(Path(self.workdir).expanduser().absolute())
        recent = [record for record in heartbeat.read_all()
                  if str(Path(str(record.get("workdir") or "")).expanduser().absolute()) == here][:5]
        if not recent:
            lines.append("none")
        for record in recent:
            alive = heartbeat.alive(record.get("pid"))
            mark = "live" if alive and record.get("status") == "running" else str(
                record.get("status") or "done")
            lines.append(f"{mark} {record.get('model') or 'no model'}")
        self.panel.update("\n".join(lines))

    def text(self) -> str:
        """The view as plain text, for tests to read without rendering."""
        content = self.panel.content
        return content.plain if hasattr(content, "plain") else str(content)
