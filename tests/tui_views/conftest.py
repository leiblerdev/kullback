"""Shared pilot harness for the views tests: a tiny app mounting one view."""

from __future__ import annotations

import asyncio
import time

from textual.app import App, ComposeResult


class ViewApp(App):
    """A test app holding a single view, so each view mounts on its own."""

    def __init__(self, view) -> None:
        super().__init__()
        self._view = view

    def compose(self) -> ComposeResult:
        yield self._view


async def wait_until(pilot, pred, timeout: float = 30.0) -> None:
    """Wait for pred to hold, pumping the app; worker threads need real time."""
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise TimeoutError("waited condition never held")
        await asyncio.sleep(0.05)
        await pilot.pause()


def table_text(table) -> str:
    """Every cell of a DataTable as text, row by row."""
    return "\n".join(
        " | ".join(str(cell) for cell in table.get_row_at(index))
        for index in range(table.row_count)
    )
