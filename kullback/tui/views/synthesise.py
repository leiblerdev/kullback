"""Generate synthetic Tasks through the CLI and watch its bus as it goes."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from textual import on, work
from textual.widget import Widget
from textual.widgets import Button, Input, RichLog, Static, Switch

from kullback.agent.bus import Bus
from kullback.tui import Transcript
from kullback.tui.views.runs import Launcher


class SynthesiseView(Widget):
    """Newcomer options for synthesise, a detached start, and the bus as a transcript."""

    TITLE = "Synthesise"

    def __init__(self, workdir: Any, launcher: Optional[Launcher] = None) -> None:
        """Keep the workdir; the bus follower only reads, the start only launches."""
        super().__init__()
        self.workdir = Path(workdir)
        self.launcher = launcher or _popen_launcher
        self._stop = threading.Event()
        self._transcript = Transcript(on_line=self._on_line)

    def compose(self):  # type: ignore[override]
        yield Input(value="1", id="per-archetype")
        yield Input(placeholder="Ceiling in USD, required", id="ceiling")
        yield Input(placeholder="Seed, optional", id="seed")
        yield Switch(id="from-domain")
        yield Button("Start synthesis", id="start")
        yield Static("Synthetic Tasks land apart and never count toward trusted.", id="status")
        yield RichLog(id="log", highlight=True)

    def on_mount(self) -> None:
        self._follow_bus()

    def on_unmount(self) -> None:
        self._stop.set()

    @on(Button.Pressed, "#start")
    def _started(self) -> None:
        per_text = self.query_one("#per-archetype", Input).value.strip() or "1"
        ceiling_text = self.query_one("#ceiling", Input).value.strip()
        seed = self.query_one("#seed", Input).value.strip()
        from_domain = self.query_one("#from-domain", Switch).value
        try:
            per_archetype = int(per_text)
        except ValueError:
            per_archetype = 0
        try:
            ceiling = float(ceiling_text)
        except ValueError:
            ceiling = -1.0
        if per_archetype < 1:
            self.query_one("#status", Static).update(
                "Per-archetype must be at least 1: nothing was launched.")
            return
        if ceiling <= 0:
            self.query_one("#status", Static).update(
                "A positive ceiling is required: nothing was launched.")
            return
        command = [sys.executable, "-m", "kullback.cli", "synthesise",
                   "--workdir", str(self.workdir),
                   "--per-archetype", str(per_archetype),
                   "--ceiling-usd", str(ceiling)]
        if from_domain:
            command.append("--from-domain")
        if seed:
            command += ["--seed", seed]
        try:
            out_path = self.launcher(command, self.workdir)
        except Exception as exc:
            self.query_one("#status", Static).update(f"Launch failed: {exc}")
            return
        self.query_one("#status", Static).update(f"Synthesis running detached, log at {out_path}.")

    @work(thread=True)
    def _follow_bus(self) -> None:
        """Replay then follow the workdir bus through the line screen's Transcript."""
        bus_path = self.workdir / "bus.jsonl"
        if not bus_path.is_file():
            return
        bus = Bus(bus_path)
        for record in bus.tail(0, poll_seconds=0.25, stop=self._stop.is_set):
            try:
                self._transcript.event(record.event)
            except Exception:
                continue

    def _on_line(self, text: str, style: str) -> None:
        """A Transcript line lands in the RichLog; called from the follower thread."""
        try:
            self.app.call_from_thread(self.query_one("#log", RichLog).write, text)
        except Exception:
            pass


def _popen_launcher(command: list[str], workdir: Path) -> Path:
    """Start the CLI detached, writing its output to a file in the workdir."""
    out_path = workdir / f"synthesise-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(out_path, "wb") as handle:
        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=str(workdir))
    return out_path
