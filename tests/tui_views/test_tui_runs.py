"""The Runs view refuses without a ceiling and launches detached with the chosen Tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import DataTable, Input, Static

from conftest import ViewApp, table_text, wait_until
from kullback.tui.views import runs as runs_view
from kullback.tui.views.runs import RunsView

pytestmark = pytest.mark.anyio

RESULT = {
    "rows": [
        {"task": "task-1", "runs": 2, "passes": 2, "outcomes": [True, True],
         "refused": False, "no_verifier": False},
        {"task": "task-2", "runs": 2, "passes": 0, "outcomes": [False, False],
         "refused": False, "no_verifier": False},
    ],
    "totals": {"pass_at_1": 0.5, "pass_k": 0.5, "k": 2, "runs_done": 4,
               "runs_planned": 4, "spend_usd": 6.4, "failing_most": "task-2",
               "failing_most_fails": 2, "ceiling_stopped": False, "ceiling_usd": 5.0},
}


def _fake_launcher(workdir_file: Path, calls: list):
    """Record the command, write the result JSON at once, and return its path."""

    def launch(command: list, workdir) -> Path:
        calls.append(command)
        out = workdir_file
        out.write_text(json.dumps(RESULT))
        return out

    return launch


async def test_runs_refuses_an_empty_ceiling_and_launches_nothing(tmp_path):
    calls: list = []
    view = RunsView(tmp_path, launcher=_fake_launcher(tmp_path / "out.json", calls))
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#model", Input).value = "test/model"
        await pilot.click("#start")
        await wait_until(
            pilot, lambda: "ceiling" in str(view.query_one("#status", Static).content).lower())
        assert calls == []
        assert view.query_one("#result", DataTable).row_count == 0


async def test_runs_launches_a_detached_run_with_the_chosen_tasks(tmp_path, monkeypatch):
    calls: list = []
    view = RunsView(tmp_path, launcher=_fake_launcher(tmp_path / "out.json", calls))
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#tasks", Input).value = "trusted"
        view.query_one("#model", Input).value = "test/model"
        view.query_one("#ceiling", Input).value = "5"
        await pilot.click("#start")
        table = view.query_one("#result", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        command = calls[0]
        assert "--tasks" in command and "trusted" in command
        assert "--model" in command and "--json" in command
        assert "--ceiling-usd" in command and "5.0" in command
        assert "task-1" in table_text(table) and "task-2" in table_text(table)
        assert "pass@1" in str(view.query_one("#status", Static).content)
    seen: dict = {}

    class _Proc:
        pass

    def _popen(*args, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(runs_view.subprocess, "Popen", _popen)
    out = runs_view._popen_launcher(["echo", "hi"], tmp_path)
    assert seen.get("start_new_session") is True
    assert "cwd" not in seen, "the child reads the same .env as this shell"
    assert Path(out).parent == tmp_path and str(out).endswith(".json")
    assert Path(seen["stdout"].name).resolve() == Path(out).resolve()
    assert Path(seen["stderr"].name).resolve() == Path(out).with_suffix(".log").resolve()


async def test_runs_shows_the_result_once_the_child_writes_it(tmp_path):
    """A launcher whose file appears later: running first, the table once it lands."""
    out = tmp_path / "out.json"
    calls: list = []
    view = RunsView(tmp_path, launcher=lambda command, workdir: calls.append(command) or out)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#model", Input).value = "test/model"
        view.query_one("#ceiling", Input).value = "5"
        await pilot.click("#start")
        await wait_until(
            pilot, lambda: "Running" in str(view.query_one("#status", Static).content))
        assert view.query_one("#result", DataTable).row_count == 0
        out.write_text(json.dumps(RESULT))
        table = view.query_one("#result", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        assert "task-1" in table_text(table)


async def test_runs_reads_the_result_despite_warnings_on_stderr(tmp_path):
    """Warnings ride the sibling .log; the .json still parses into the table."""
    out = tmp_path / "out.json"
    out.with_suffix(".log").write_text("unfrozen runner: verdicts carry no version\n")

    def launch(command: list, workdir) -> Path:
        out.write_text(json.dumps(RESULT))
        return out

    view = RunsView(tmp_path, launcher=launch)
    app = ViewApp(view)
    async with app.run_test() as pilot:
        view.query_one("#model", Input).value = "test/model"
        view.query_one("#ceiling", Input).value = "5"
        await pilot.click("#start")
        table = view.query_one("#result", DataTable)
        await wait_until(pilot, lambda: table.row_count == 2)
        assert "task-2" in table_text(table)
