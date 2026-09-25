"""Run a Candidate detached and score it: one line per Task plus the pass table."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from textual import on
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Static

Launcher = Callable[[list[str], Path], Path]


class RunsView(Widget):
    """Pick trusted, all or typed ids plus model and ceiling, start detached, show scores."""

    TITLE = "Runs"

    def __init__(self, workdir: Any, launcher: Optional[Launcher] = None) -> None:
        """Keep the workdir and the launcher; the default detaches a child that outlives us."""
        super().__init__()
        self.workdir = Path(workdir)
        self.launcher = launcher or _popen_launcher

    def compose(self):  # type: ignore[override]
        yield Input(value="trusted", id="tasks")
        yield Input(placeholder="Candidate model, as provider/model", id="model")
        yield Input(placeholder="Ceiling in USD, required", id="ceiling")
        yield Button("Start run", id="start")
        yield Static("Only trusted Tasks are graded by a checked Verifier.", id="status")
        yield DataTable(id="result")

    def on_mount(self) -> None:
        table = self.query_one("#result", DataTable)
        table.add_column("task", key="task")
        table.add_column("runs", key="runs")
        table.add_column("passes", key="passes")
        table.add_column("outcomes", key="outcomes")

    @on(Button.Pressed, "#start")
    def _started(self) -> None:
        tasks = self.query_one("#tasks", Input).value.strip() or "trusted"
        model = self.query_one("#model", Input).value.strip()
        ceiling_text = self.query_one("#ceiling", Input).value.strip()
        try:
            ceiling = float(ceiling_text)
        except ValueError:
            ceiling = -1.0
        if ceiling <= 0:
            self.query_one("#status", Static).update(
                "A positive ceiling is required: nothing was launched.")
            return
        if not model:
            self.query_one("#status", Static).update(
                "A model is required: nothing was launched.")
            return
        command = [sys.executable, "-m", "kullback.cli", "run",
                   "--workdir", str(self.workdir), "--tasks", tasks,
                   "--model", model, "--ceiling-usd", str(ceiling), "--json"]
        try:
            out_path = self.launcher(command, self.workdir)
        except Exception as exc:
            self.query_one("#status", Static).update(f"Launch failed: {exc}")
            return
        self.query_one("#status", Static).update(f"Running detached, output to {out_path}.")
        self._show_result(out_path)

    def _show_result(self, out_path: Path) -> None:
        """The JSON the child wrote as a table: one line per Task, then the pass table."""
        try:
            body = json.loads(Path(out_path).read_text())
        except (OSError, ValueError):
            self.query_one("#status", Static).update(
                f"Running detached, output to {out_path}; no result yet.")
            return
        table = self.query_one("#result", DataTable)
        table.clear()
        for row in body.get("rows", ()):
            words = ["pass" if passed else "fail" for passed in row.get("outcomes", ())]
            if row.get("refused"):
                words = ["refused"]
            elif row.get("no_verifier"):
                words = ["no verifier"]
            table.add_row(str(row.get("task")), str(row.get("runs")),
                          str(row.get("passes")), " ".join(words))
        totals = body.get("totals", {})
        failing = (f"{totals.get('failing_most')} ({totals.get('failing_most_fails')})"
                   if totals.get("failing_most") else "none")
        count = totals.get("k", 1)
        head = f"pass@1 {_rate(totals.get('pass_at_1'))}"
        if count and count > 1:
            head += f" pass^{count} {_rate(totals.get('pass_k'))}"
        self.query_one("#status", Static).update(
            f"{head} runs {totals.get('runs_done')}/{totals.get('runs_planned')} "
            f"spend ${float(totals.get('spend_usd') or 0.0):.2f} failing most: {failing}")


def _popen_launcher(command: list[str], workdir: Path) -> Path:
    """Start the CLI detached, writing its JSON to a file in the workdir, and return the path."""
    out_path = workdir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "wb") as handle:
        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=str(workdir))
    return out_path


def _rate(value: Any) -> str:
    return f"{float(value):.2f}" if value is not None else "n/a"
