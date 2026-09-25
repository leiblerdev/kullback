"""Browse Tasks k9s style: filter the rows, open one Task, nudge about it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from textual import on, work
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import DataTable, Input, Static

from kullback.agent import steer
from kullback.live_counts import task_detail, task_rows

_COLUMNS = ("task_id", "status", "replay", "suite", "probes", "drift",
            "last_finding_kind", "last_finding_path")


class TasksView(Widget):
    """List every Task, filter by text or status, open detail, nudge the Builder."""

    TITLE = "Tasks"

    BINDINGS = [
        Binding("v", "toggle_values", "Show/hide recorded values"),
        Binding("n", "nudge", "Nudge the Builder about this Task"),
    ]

    def __init__(self, workdir: Any) -> None:
        """Keep the workdir; every row is read from its files, nothing is kept here."""
        super().__init__()
        self.workdir = Path(workdir)
        self._rows: list[dict] = []
        self._shown: list[dict] = []
        self._selected: Optional[str] = None
        self._show_values = False

    def compose(self):  # type: ignore[override]
        yield Input(placeholder="Filter text or status word", id="filter")
        yield DataTable(id="rows", cursor_type="row")
        yield Static("Select a Task for its detail.", id="detail")
        yield Input(placeholder="Nudge about the selected Task, enter to send",
                    id="nudge")

    def on_mount(self) -> None:
        table = self.query_one("#rows", DataTable)
        for name in ("id", "status", "replay", "suite", "probes", "drift", "finding"):
            table.add_column(name, key=name)
        self.query_one("#nudge", Input).display = False
        self.refresh_rows()
        self._refresh_timer = self.set_interval(5.0, self._refresh_tick)

    def on_unmount(self) -> None:
        self._refresh_timer.stop()

    def _refresh_tick(self) -> None:
        """Re-read every 5 seconds while mounted, keeping selection, cursor and filter."""
        table = self.query_one("#rows", DataTable)
        try:
            saved = table.cursor_coordinate
        except Exception:
            saved = None
        self.refresh_rows()
        if saved is not None and self._shown:
            row = min(saved[0], len(self._shown) - 1)
            with table.prevent(DataTable.RowSelected):
                table.move_cursor(row=row, column=saved[1])
        if self._selected is not None:
            self._render_detail()

    def refresh_rows(self) -> None:
        """Re-read every row and re-apply the filter; missing files read as no Tasks."""
        try:
            self._rows = task_rows(self.workdir)
        except Exception:
            self._rows = []
        self._apply_filter(self.query_one("#filter", Input).value)

    @on(Input.Changed, "#filter")
    def _filtered(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    def _apply_filter(self, text: str) -> None:
        """Keep rows matching the text in any cell; a bare status word matches status only."""
        needle = (text or "").strip().lower()
        if not needle:
            self._shown = list(self._rows)
        else:
            self._shown = [row for row in self._rows if _matches(row, needle)]
        table = self.query_one("#rows", DataTable)
        table.clear()
        for row in self._shown:
            table.add_row(
                str(row.get("task_id")),
                str(row.get("status")),
                str(row.get("replay")),
                str(row.get("suite")),
                str(row.get("probes")),
                str(row.get("drift")),
                f"{row.get('last_finding_kind') or ''} {row.get('last_finding_path') or ''}".strip(),
                key=str(row.get("task_id")),
            )

    @on(DataTable.RowSelected, "#rows")
    def _opened(self, event: DataTable.RowSelected) -> None:
        task_id = str(event.row_key.value)
        self._selected = task_id
        self._render_detail()

    def _render_detail(self) -> None:
        """Detail beside the table: kinds, paths and counts; values only after v."""
        pane = self.query_one("#detail", Static)
        if self._selected is None:
            pane.update("Select a Task for its detail.")
            return
        try:
            detail = task_detail(self.workdir, self._selected)
        except KeyError:
            pane.update(f"{self._selected}: no longer in the workdir.")
            return
        lines = [
            f"detail {detail.get('task_id')} [{detail.get('status')}]",
            f"replay {detail.get('replay')} suite {detail.get('suite')}",
            f"failing gate: {detail.get('failing_gate') or 'none'}",
            f"open findings: {_findings_line(detail.get('open_findings'))}",
            f"atoms: {_atoms_line(detail.get('atom_counts'))}",
            (f"replay calls: {detail.get('replay_matched')}/{detail.get('replay_total')}"
             if detail.get("replay_total") is not None else "replay calls: none recorded"),
        ]
        if self._show_values:
            lines.append("values: " + json.dumps(detail, indent=2, sort_keys=True, default=str))
        else:
            lines.append("press v to show the recorded values this detail holds")
        pane.update("\n".join(lines))

    def action_toggle_values(self) -> None:
        """Show or hide the values the detail already holds, for the one open Task."""
        self._show_values = not self._show_values
        self._render_detail()

    def action_nudge(self) -> None:
        """Open the nudge box naming the selected Task, so the Builder knows which one."""
        if self._selected is None:
            self.query_one("#detail", Static).update("Select a Task first, then press n.")
            return
        box = self.query_one("#nudge", Input)
        box.value = f"look at {self._selected}: "
        box.display = True
        box.focus()

    @on(Input.Submitted, "#nudge")
    def _nudge_sent(self, event: Input.Submitted) -> None:
        text = str(event.value).strip()
        box = self.query_one("#nudge", Input)
        box.display = False
        self._send_nudge(text)

    @work(thread=True)
    def _send_nudge(self, text: str) -> None:
        """Append the nudge to the workdir bus and report the ack, off the UI thread."""
        try:
            request_id = steer.request(self.workdir, "nudge", text, sender="tasks-view")
            ack = steer.wait_for_ack(self.workdir, request_id, timeout=2.0)
        except Exception as exc:
            self.app.call_from_thread(self._nudge_outcome, f"nudge not sent: {exc}")
        else:
            if ack is None:
                self.app.call_from_thread(
                    self._nudge_outcome, "nudge on the bus; no live build acked it")
            else:
                self.app.call_from_thread(
                    self._nudge_outcome, f"nudge {ack.outcome.replace('_', ' ')}")

    def _nudge_outcome(self, line: str) -> None:
        self.query_one("#detail", Static).update(line)


def _matches(row: dict, needle: str) -> bool:
    """A row matches when any cell holds the text, or the status equals the word."""
    if needle in {"trusted", "refused", "open"}:
        return str(row.get("status")) == needle
    return any(needle in str(value).lower() for value in row.values() if value is not None)


def _findings_line(findings: Any) -> str:
    if not findings:
        return "none open"
    return "; ".join(f"{item.get('kind')} {item.get('path') or ''}".strip()
                     for item in findings)


def _atoms_line(counts: Any) -> str:
    if not counts:
        return "no Verifier yet"
    return ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))
