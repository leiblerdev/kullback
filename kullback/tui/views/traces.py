"""Preview a trace file before storing anything: dry run plus shape, never values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import on, work
from textual.widget import Widget
from textual.widgets import DataTable, Input, Static

from kullback.builder import ingest
from kullback.builder.sources import shape


class TracesView(Widget):
    """Dry-run a trace file: format, counts and shape, never file values."""

    TITLE = "Traces"

    def __init__(self, workdir: Any) -> None:
        """Keep the workdir; its readers take part in the dry run exactly as in ingest."""
        super().__init__()
        self.workdir = Path(workdir)

    def compose(self):  # type: ignore[override]
        yield Input(placeholder="Trace file path, enter for a dry run", id="trace-path")
        yield DataTable(id="dry-run")
        yield Static("No file checked yet.", id="shape")

    def on_mount(self) -> None:
        table = self.query_one("#dry-run", DataTable)
        table.add_column("format", key="format")
        table.add_column("confidence", key="confidence")
        table.add_column("reasons", key="reasons")
        table.add_column("recordings", key="recordings")
        table.add_column("traces", key="traces")
        table.add_column("eligible", key="eligible")
        table.add_column("passed", key="passed")
        table.add_column("tools", key="tools")
        table.add_column("rejected", key="rejected")
        table.add_column("message", key="message")

    @on(Input.Submitted, "#trace-path")
    def _submitted(self, event: Input.Submitted) -> None:
        self._dry_run(str(event.value).strip())

    @work(thread=True)
    def _dry_run(self, path: str) -> None:
        """Run the dry run and the shape summary off the UI thread; both read only."""
        try:
            result = ingest.dry_run(path, self.workdir)
            summary = shape.shape_summary(path)
        except Exception as exc:
            self.app.call_from_thread(self._show_error, _one_sentence(exc))
        else:
            self.app.call_from_thread(self._show_result, result, summary)

    def _show_error(self, sentence: str) -> None:
        table = self.query_one("#dry-run", DataTable)
        table.clear()
        self.query_one("#shape", Static).update(sentence or "The file could not be read.")

    def _show_result(self, result: dict, summary: dict) -> None:
        """One table row from the dry run, structure lines from the shape; no file values."""
        table = self.query_one("#dry-run", DataTable)
        table.clear()
        rejected = result.get("rejected") or {}
        table.add_row(
            str(result.get("format")),
            f"{float(result.get('confidence') or 0.0):.2f}",
            "; ".join(str(reason) for reason in result.get("reasons") or ()),
            str(result.get("recordings")),
            str(result.get("traces")),
            f"{float(result.get('eligible_share') or 0.0):.0%}",
            "yes" if result.get("passed") else "no",
            str(result.get("tools")),
            ", ".join(f"{name}={count}" for name, count in sorted(rejected.items())) or "-",
            _one_sentence(result.get("message")),
        )
        self.query_one("#shape", Static).update("\n".join(_shape_lines(summary)))


def _one_sentence(value: Any) -> str:
    """The first line of a message, so a gate error reads as one sentence, never a traceback."""
    text = "" if value is None else str(value).strip()
    return text.splitlines()[0] if text else ""


def _shape_lines(summary: dict) -> list[str]:
    """One line per key path: types, count, lengths and hints; kept values are never shown."""
    head = (f"{summary.get('file')} jsonl={summary.get('jsonl')} "
            f"records={summary.get('records')}")
    if summary.get("error"):
        return [head, str(summary["error"])]
    lines = [head]
    for path, info in sorted((summary.get("paths") or {}).items()):
        line = f"{path}: {','.join(info.get('types', []))} x{info.get('count', 0)}"
        if info.get("min_len") is not None:
            line += f" len {info['min_len']}..{info['max_len']}"
        if info.get("hints"):
            line += f" ({', '.join(info['hints'])})"
        lines.append(line)
    return lines
