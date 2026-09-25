"""Generate synthetic Tasks through the CLI and follow its output as it grows."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from textual import on
from textual.widget import Widget
from textual.widgets import Button, Input, RichLog, Static, Switch

from kullback.tui.views.runs import Launcher


class SynthesiseView(Widget):
    """Newcomer options for synthesise, a detached start, and the command log."""

    TITLE = "Synthesise"

    def __init__(self, workdir: Any, launcher: Optional[Launcher] = None) -> None:
        """Keep the workdir; the log timer only reads the file the child writes."""
        super().__init__()
        self.workdir = Path(workdir)
        self.launcher = launcher or _popen_launcher
        self._log_timer = None
        self._log_path: Optional[Path] = None
        self._log_offset = 0
        self._log_tail = ""

    def compose(self):  # type: ignore[override]
        yield Input(placeholder="Buckets as NAME=COUNT, comma separated", id="buckets")
        yield Input(placeholder="Archetype ids, comma separated", id="archetypes")
        yield Input(value="1", id="per-archetype")
        yield Input(placeholder="Model for the Intent writer, optional", id="model")
        yield Input(placeholder="Ceiling in USD, required", id="ceiling")
        yield Input(placeholder="Seed, optional", id="seed")
        yield Switch(id="from-domain")
        yield Button("Start synthesis", id="start")
        yield Static("Synthetic Tasks land apart and never count toward trusted.", id="status")
        yield RichLog(id="log", highlight=True)

    def on_unmount(self) -> None:
        self._stop_log()

    @on(Button.Pressed, "#start")
    def _started(self) -> None:
        buckets_text = self.query_one("#buckets", Input).value.strip()
        archetypes_text = self.query_one("#archetypes", Input).value.strip()
        per_text = self.query_one("#per-archetype", Input).value.strip() or "1"
        model = self.query_one("#model", Input).value.strip()
        ceiling_text = self.query_one("#ceiling", Input).value.strip()
        seed = self.query_one("#seed", Input).value.strip()
        from_domain = self.query_one("#from-domain", Switch).value
        buckets, bad_bucket = _buckets(buckets_text)
        if bad_bucket is not None:
            self.query_one("#status", Static).update(
                f"Buckets read as NAME=COUNT, got {bad_bucket!r}: nothing was launched.")
            return
        archetypes = _entries(archetypes_text)
        try:
            ceiling = float(ceiling_text)
        except ValueError:
            ceiling = -1.0
        if ceiling <= 0:
            self.query_one("#status", Static).update(
                "A positive ceiling is required: nothing was launched.")
            return
        if not buckets and not archetypes and not from_domain:
            self.query_one("#status", Static).update(
                "Choose buckets, archetype ids or from-domain: nothing was launched.")
            return
        command = [sys.executable, "-m", "kullback.cli", "synthesise",
                   "--workdir", str(self.workdir)]
        for bucket in buckets:
            command += ["--bucket", bucket]
        for archetype in archetypes:
            command += ["--archetype", archetype]
        if archetypes or from_domain:
            try:
                per = int(per_text)
            except ValueError:
                per = 0
            if per < 1:
                self.query_one("#status", Static).update(
                    "Per-archetype must be at least 1: nothing was launched.")
                return
            command += ["--per-archetype", str(per)]
        if from_domain:
            command.append("--from-domain")
        if model:
            command += ["--model", model]
        if seed:
            command += ["--seed", seed]
        command += ["--ceiling-usd", str(ceiling)]
        try:
            log_path = self.launcher(command, self.workdir)
        except Exception as exc:
            self.query_one("#status", Static).update(f"Launch failed: {exc}")
            return
        self._stop_log()
        self._log_path = Path(log_path)
        self._log_offset = 0
        self._log_tail = ""
        self.query_one("#status", Static).update(
            f"Synthesis running detached, log at {log_path}.")
        self._log_timer = self.set_interval(1.0, self._poll_log)

    def _stop_log(self) -> None:
        """Stop the log timer, so a new start or unmount leaves no timer behind."""
        if self._log_timer is not None:
            self._log_timer.stop()
            self._log_timer = None

    def _poll_log(self) -> None:
        """Append whatever the child wrote since the last poll to the log."""
        if self._log_path is None:
            return
        try:
            chunk = self._log_path.read_bytes()[self._log_offset:]
        except OSError:
            return
        if not chunk:
            return
        self._log_offset += len(chunk)
        text = self._log_tail + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._log_tail = lines.pop()
        log = self.query_one("#log", RichLog)
        for line in lines:
            log.write(line)


def _entries(text: str) -> list[str]:
    """Comma or space separated entries, in the order typed."""
    return [entry for chunk in text.split(",") for entry in chunk.split() if entry]


def _buckets(text: str) -> tuple[list[str], Optional[str]]:
    """The bucket entries plus the first one that does not read as NAME=COUNT."""
    buckets = _entries(text)
    for bucket in buckets:
        if "=" not in bucket:
            return [], bucket
    return buckets, None


def _popen_launcher(command: list[str], workdir: Path) -> Path:
    """Start the CLI detached, writing its output to a file in the workdir."""
    out_path = workdir / f"synthesise-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(out_path, "wb") as handle:
        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return out_path
