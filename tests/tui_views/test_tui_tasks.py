"""The Tasks view lists rows, filters, hides values until v, and nudges by name."""

from __future__ import annotations

import json

import pytest
from textual.widgets import DataTable, Input, Static

from conftest import ViewApp, table_text, wait_until
from kullback.tui.views.tasks import TasksView

pytestmark = pytest.mark.anyio


def _workdir(root) -> None:
    """Two neutral Tasks: task-1 open with a passing suite, task-2 refused."""
    (root / "tasks").mkdir()
    (root / "tasks" / "task-1.json").write_text("{}")
    (root / "tasks" / "task-2.json").write_text("{}")
    (root / "task_status.json").write_text(
        json.dumps({"task-1": {"verifier_passed": True}, "task-2": {}}))
    (root / "refusals").mkdir()
    (root / "refusals" / "task-2.json").write_text("{}")


async def test_tasks_lists_every_row_and_filters_by_status(tmp_path):
    _workdir(tmp_path)
    view = TasksView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        assert "task-1" in table_text(table) and "task-2" in table_text(table)
        view.query_one("#filter", Input).focus()
        await pilot.pause()
        await pilot.press("r", "e", "f", "u", "s", "e", "d")
        await wait_until(pilot, lambda: table.row_count == 1)
        assert "task-2" in table_text(table)
        for _ in range(7):
            await pilot.press("backspace")
        await wait_until(pilot, lambda: table.row_count == 2)
        await pilot.press("o", "p", "e", "n")
        await wait_until(pilot, lambda: table.row_count == 1)
        assert "task-1" in table_text(table)


async def test_tasks_hides_values_until_v_and_hides_them_again(tmp_path):
    _workdir(tmp_path)
    view = TasksView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        table.focus()
        await pilot.pause()
        await pilot.press("enter")
        pane = view.query_one("#detail", Static)
        await wait_until(pilot, lambda: "detail task-1" in str(pane.content))
        assert '"atom_counts"' not in str(pane.content)
        await pilot.press("v")
        await wait_until(pilot, lambda: '"atom_counts"' in str(pane.content))
        await pilot.press("v")
        await wait_until(pilot, lambda: '"atom_counts"' not in str(pane.content))


async def test_tasks_n_sends_a_nudge_naming_the_task(tmp_path):
    _workdir(tmp_path)
    view = TasksView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        table.focus()
        await pilot.pause()
        await pilot.press("enter")
        await wait_until(
            pilot, lambda: "detail task-1" in str(view.query_one("#detail", Static).content))
        await pilot.press("n")
        box = view.query_one("#nudge", Input)
        await wait_until(pilot, lambda: box.display)
        assert "task-1" in box.value
        await pilot.press("enter")
        await wait_until(
            pilot, lambda: "nudge" in str(view.query_one("#detail", Static).content),
            timeout=15.0)
        assert "task-1" in (tmp_path / "bus.jsonl").read_text()


async def test_tasks_picks_up_a_new_task_while_mounted(tmp_path):
    """A Task stored by a live build appears on the next 5 second refresh."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task-1.json").write_text("{}")
    (tmp_path / "task_status.json").write_text(json.dumps({"task-1": {}}))
    view = TasksView(tmp_path)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        table = view.query_one("#rows", DataTable)
        await wait_until(pilot, lambda: table.row_count == 1)
        (tmp_path / "tasks" / "task-2.json").write_text("{}")
        (tmp_path / "task_status.json").write_text(
            json.dumps({"task-1": {}, "task-2": {}}))
        await wait_until(pilot, lambda: table.row_count == 2, timeout=20.0)
        assert "task-2" in table_text(table)
