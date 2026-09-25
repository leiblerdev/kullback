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

Launcher = Callable[[list[str], Path], tuple[Path, Any]]


class RunsView(Widget):
    """Pick trusted, all or typed ids plus model and ceiling, start detached, show scores."""

    TITLE = "Runs"

    def __init__(self, workdir: Any, launcher: Optional[Launcher] = None) -> None:
        """Keep the workdir and the launcher; the default detaches a child that outlives us."""
        super().__init__()
        self.workdir = Path(workdir)
        self.launcher = launcher or _popen_launcher
        self._result_timer = None
        self._poll_start = 0.0
        self._out_path: Optional[Path] = None
        self._proc: Any = None

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
            out_path, proc = self.launcher(command, self.workdir)
        except Exception as exc:
            self.query_one("#status", Static).update(f"Launch failed: {exc}")
            return
        self._stop_poll()
        self._out_path = Path(out_path)
        self._proc = proc
        self._poll_start = time.monotonic()
        if not self._refresh():
            self._result_timer = self.set_interval(2.0, self._poll_result)

    def on_unmount(self) -> None:
        self._stop_poll()

    def _stop_poll(self) -> None:
        """Stop the result timer, so a new start or unmount leaves no timer behind."""
        if self._result_timer is not None:
            self._result_timer.stop()
            self._result_timer = None

    def _poll_result(self) -> None:
        """One timer step: render the result, report a dead child, or keep waiting."""
        if self._out_path is None:
            return
        if self._refresh():
            self._stop_poll()

    def _refresh(self) -> bool:
        """Poll once: True when the status is final and the timer can stop."""
        if self._try_show():
            return True
        if self._run_ended():
            self._show_no_result()
            return True
        self._show_running()
        return False

    def _run_ended(self) -> bool:
        """True when the child is gone: its poll reports an exit, so no JSON will land."""
        poll = getattr(self._proc, "poll", None)
        return poll is not None and poll() is not None

    def _show_no_result(self) -> None:
        """The child ended and its JSON never parsed: say so with the tail of its log."""
        assert self._out_path is not None
        try:
            text = self._out_path.with_suffix(".log").read_text(encoding="utf-8",
                                                                errors="replace")
        except OSError:
            text = ""
        lines = text.splitlines()[-5:]
        message = "the run ended without a result"
        if lines:
            message += ":\n" + "\n".join(lines)
        self.query_one("#status", Static).update(message)

    def _show_running(self) -> None:
        elapsed = int(time.monotonic() - self._poll_start)
        self.query_one("#status", Static).update(
            f"Running {elapsed}s, output to {self._out_path}.")

    def _try_show(self) -> bool:
        """Render the child's JSON as a table; False while it is missing or partial."""
        assert self._out_path is not None
        try:
            body = json.loads(self._out_path.read_text())
        except (OSError, ValueError):
            return False
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
        return True


def _popen_launcher(command: list[str], workdir: Path) -> tuple[Path, Any]:
    """Start the CLI detached: stdout as JSON, stderr beside it, same shell environment."""
    stem = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    out_path = workdir / f"{stem}.json"
    log_path = workdir / f"{stem}.log"
    with open(out_path, "wb") as out_handle, open(log_path, "wb") as log_handle:
        proc = subprocess.Popen(command, stdout=out_handle, stderr=log_handle,
                                start_new_session=True)
    return out_path, proc


def _rate(value: Any) -> str:
    return f"{float(value):.2f}" if value is not None else "n/a"
