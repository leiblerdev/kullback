"""The Traces view shows the dry run table and never a value from the file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import DataTable, Input, Static

from conftest import ViewApp, table_text, wait_until
from kullback.tui.views.traces import TracesView

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "tau2_retail_small.json"


def _long_strings(value, out: set) -> None:
    """Every string value of at least 24 chars: customer data, found only at runtime."""
    if isinstance(value, str):
        if len(value) >= 24:
            out.add(value)
    elif isinstance(value, list):
        for item in value:
            _long_strings(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _long_strings(item, out)


async def test_traces_shows_the_dry_run_table_and_never_a_value_from_the_file(tmp_path):
    view = TracesView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        box = view.query_one("#trace-path", Input)
        box.value = str(FIXTURE)
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        table = view.query_one("#dry-run", DataTable)
        await wait_until(pilot, lambda: table.row_count == 1)
        cells = table.get_row_at(0)
        assert str(cells[0]).strip()
        float(cells[1])
        shape = str(view.query_one("#shape", Static).content)
        assert "records=" in shape
        shown = table_text(table) + "\n" + shape
        values: set = set()
        _long_strings(json.loads(FIXTURE.read_text()), values)
        assert values
        for value in values:
            assert value not in shown


async def test_traces_shows_a_gate_error_as_one_sentence(tmp_path):
    target = tmp_path / "odd.jsonl"
    target.write_text('{"zzz_nope": 1}\n')
    view = TracesView(tmp_path / "work")
    (tmp_path / "work").mkdir()
    app = ViewApp(view)
    async with app.run_test() as pilot:
        box = view.query_one("#trace-path", Input)
        box.value = str(target)
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        table = view.query_one("#dry-run", DataTable)
        await wait_until(pilot, lambda: table.row_count == 1)
        cells = table.get_row_at(0)
        assert cells[6] == "no"
        message = str(cells[9]).strip()
        assert message and "\n" not in message
