"""Follow a build live: the Watch view of the Textual app.

The transcript scrolls on the left while the steer input below never moves, so
output never lands in the middle of typing. The sidebar on the right shows the
numbers the line screen already reads: spend with its burn rate off
Board.money, the live workdir counts, the rounds table, and the failed gates.
Enter nudges, alt+enter tells, esc asks before stopping, all through the bus,
so quitting the screen leaves the build running.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, RichLog, Static

from kullback.agent.bus import Bus
from kullback.tui import STEER_TIMEOUT, Board, Transcript, live_heartbeats

BUS_FILE = "bus.jsonl"


class StopModal(ModalScreen[bool]):
    """Ask before stopping: y stops after the current step, n and esc keep it."""

    BINDINGS = [Binding("y", "choose(True)", "Stop"), Binding("n", "choose(False)", "Keep"),
                Binding("escape", "choose(False)", "Keep")]

    def compose(self):  # type: ignore[override]
        yield Static("stop the build after its current step? y/n", id="stop-ask")

    def action_choose(self, stop: bool) -> None:
        self.dismiss(stop)


class WatchView(Vertical):
    """The live transcript, the sidebar numbers, and the pinned steer input."""

    BINDINGS = [Binding("escape", "confirm_stop", "Stop"),
                Binding("alt+enter", "tell", "Tell")]

    def __init__(self, workdir: Any, ceiling_usd: Optional[float] = None,
                 poll_seconds: float = 0.25) -> None:
        super().__init__(id="watch")
        self.workdir = Path(workdir)
        self.poll_seconds = poll_seconds
        self.bus = Bus(self.workdir / BUS_FILE, agent="tui")
        self.transcript = Transcript(on_line=self._on_bus_line)
        self.board = Board(self.workdir, title="watch", ceiling=ceiling_usd)
        self.note = Static("", id="watch-note")
        self.transcript_log = RichLog(id="transcript", highlight=True, markup=False)
        self.money_panel = Static("", id="watch-money")
        self.counts_panel = Static("", id="watch-counts")
        self.rounds_panel = Static("", id="watch-rounds")
        self.gates_panel = Static("", id="watch-gates")
        self.steer_input = Input(placeholder="nudge the build (alt+enter tells)", id="steer")
        self._stop = threading.Event()

    def compose(self):  # type: ignore[override]
        yield self.note
        with Horizontal(id="watch-main"):
            yield self.transcript_log
            with Vertical(id="sidebar"):
                yield self.money_panel
                yield self.counts_panel
                yield self.rounds_panel
                yield self.gates_panel
        yield self.steer_input

    def on_mount(self) -> None:
        if not live_heartbeats(self.workdir):
            self.note.update(f"nothing is running in {self.workdir}")
        self.refresh_sidebar()
        self.set_interval(5.0, self.refresh_sidebar)
        self.run_worker(self._follow_bus, thread=True)

    def on_unmount(self) -> None:
        self._stop.set()

    def _on_bus_line(self, text: str, style: str) -> None:
        """A Transcript line from the follower thread, appended on the app thread."""
        try:
            self.app.call_from_thread(self._append_line, text, style)
        except Exception:
            self._append_line(text, style)

    def _append_line(self, text: str, style: str) -> None:
        self.transcript_log.write(Text(text, style=style or None))

    def _follow_bus(self) -> None:
        """Replay the bus, then follow it; every event feeds the transcript."""
        try:
            seq = 0
            for record in self.bus.replay():
                seq = record.seq
                self.transcript.event(record.event)
            for record in self.bus.tail(seq, poll_seconds=self.poll_seconds,
                                        stop=self._stop.is_set):
                self.transcript.event(record.event)
        except Exception as exc:
            self.transcript.say(f"watch cannot follow the bus: {exc}", "red")

    def refresh_sidebar(self) -> None:
        """Schedule a sidebar read off the UI thread; reads never overlap."""
        self.run_worker(self._read_sidebar, thread=True, exclusive=True)

    def _read_sidebar(self) -> None:
        """The four panels plus liveness, read off the workdir's files."""
        from kullback.live_counts import workdir_counts

        try:
            money = self.board.money()
            counts = self._counts_line(workdir_counts(self.workdir))
            rounds = self._rounds_table()
            gates = self._failed_gates()
            live = bool(live_heartbeats(self.workdir))
        except Exception as exc:
            self.transcript.say(f"watch cannot read the workdir: {exc}", "red")
            return
        try:
            self.app.call_from_thread(self._update_sidebar, money, counts, rounds, gates, live)
        except Exception:
            pass

    def _update_sidebar(self, money: Any, counts: Text, rounds: Any, gates: Text,
                        live: bool) -> None:
        """The read panels onto the screen, on the app thread."""
        self.money_panel.update(money)
        self.counts_panel.update(counts)
        self.rounds_panel.update(rounds)
        self.gates_panel.update(gates)
        if live:
            self.note.update("")
        else:
            self.note.update(f"nothing is running in {self.workdir}")

    @staticmethod
    def _counts_line(counts: dict) -> Text:
        """The live counts as one line: fidelity, trusted, refused, open, drifted."""
        drifted = counts.get("drifted")
        drifted_n = len(drifted) if isinstance(drifted, list) else int(drifted or 0)
        out = Text(f"tasks {counts.get('tasks', 0)}", style="bold")
        out.append(f"   fidelity {counts.get('fidelity', 0)}/{counts.get('tasks', 0)}", style="white")
        out.append(f"   trusted {counts.get('trusted', 0)}", style="green")
        out.append(f"   refused {counts.get('refused', 0)}", style="red")
        out.append(f"   open {counts.get('open', 0)}", style="white")
        out.append(f"   drifted {drifted_n}", style="yellow" if drifted_n else "dim")
        return out

    def _rounds_table(self) -> Any:
        """The rounds table off rounds.json, through Board so both screens match."""
        import json

        try:
            rows = json.loads((self.workdir / "rounds.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Text("no rounds yet", style="dim")
        if not isinstance(rows, list) or not rows:
            return Text("no rounds yet", style="dim")
        board = Board(self.workdir, title="rounds")
        board.rounds = [{"round": row.get("round"), "counts": row.get("counts") or {},
                         "exit": row.get("exit")} for row in rows if isinstance(row, dict)]
        board.round = int(board.rounds[-1].get("round") or 0)
        return board.rounds_table()

    def _failed_gates(self) -> Text:
        """The gates that hold the build, one line each, cut like the board's."""
        import json

        try:
            gates = json.loads((self.workdir / "gates.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Text("no gates yet", style="dim")
        if not isinstance(gates, list) or not gates:
            return Text("no gates yet", style="dim")
        failed = [gate for gate in gates if isinstance(gate, dict)
                  and not gate.get("pass", gate.get("passed", True))]
        out = Text(f"{len(gates) - len(failed)} gates passed", style="green")
        if failed:
            out.append(f", {len(failed)} failed", style="red")
            latest = failed[-1]
            reason = "; ".join(latest.get("failures") or []) or "no reason given"
            out.append(f"\n{latest.get('stage')}: {reason[:160]}", style="red")
        return out

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self.steer_input:
            text = event.value
            event.input.value = ""
            self.send_steer("nudge", text)

    def action_tell(self) -> None:
        """Alt+enter tells where enter nudges: queued for when the run would stop."""
        text = self.steer_input.value
        self.steer_input.value = ""
        self.send_steer("tell", text)

    def action_confirm_stop(self) -> None:
        """Esc asks before stopping, since a stop cannot be undone."""
        self.app.push_screen(StopModal(), self._stop_chosen)

    def _stop_chosen(self, stop: Optional[bool]) -> None:
        if stop:
            self.send_steer("stop", "")

    def send_steer(self, kind: str, text: str) -> None:
        """A nudge, tell or stop onto the bus; the ack outcome shows as a log line."""
        self.run_worker(lambda: self._steer_roundtrip(kind, text), thread=True)

    def _steer_roundtrip(self, kind: str, text: str) -> None:
        from kullback.agent import steer

        try:
            request_id = steer.request(self.workdir, kind, text, sender="tui")
            ack = steer.wait_for_ack(self.workdir, request_id, STEER_TIMEOUT)
        except Exception as exc:
            self.transcript.say(f"steer {kind} failed: {exc}", "red")
            return
        if ack is None:
            line = f"steer {kind}: no live build answered"
        else:
            line = f"steer {kind} {ack.outcome.replace('_', ' ')}"
            if ack.reason:
                line += f": {ack.reason}"
        try:
            self.app.call_from_thread(self._append_line, line, "cyan")
        except Exception:
            self._append_line(line, "cyan")

    def log_text(self) -> str:
        """The transcript lines so far, for tests to read without rendering."""
        return "\n".join(text for text, _ in self.transcript.lines)

    def sidebar_text(self) -> str:
        """The sidebar panels as plain text, for tests to read without rendering."""
        parts = []
        for panel in (self.money_panel, self.counts_panel, self.rounds_panel, self.gates_panel):
            content = panel.content
            parts.append(content.plain if hasattr(content, "plain") else str(content))
        return "\n".join(parts)
