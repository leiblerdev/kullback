"""The solve-rate stage counts no-signal Runs apart from the rate (G1)."""

from __future__ import annotations

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)

from kullback.ai.provider import TestModel  # noqa: E402
from kullback.episode import BuiltEnvironment, markdown_table, solve_rate  # noqa: E402
from tests.episode.invented import write_env  # noqa: E402


def test_solve_rate_counts_passes_and_leaves_no_signal_out():
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root, task_id="scored")
        (root / "tasks" / "quiet.json").write_text(
            (root / "tasks" / "scored.json").read_text().replace("scored", "quiet"),
            encoding="utf-8")
        (root / "overlays" / "quiet.json").write_text(
            (root / "overlays" / "scored.json").read_text().replace("scored", "quiet"),
            encoding="utf-8")
        (root / "verifiers" / "quiet.json").write_text(json.dumps({
            "task_id": "quiet",
            "atoms": [{"id": "j1", "kind": "hard", "judge": True,
                         "predicate_src": "wrote('rename_widget')",
                         "target": {"kind": "write"}}],
            "verifier_version": "1",
        }), encoding="utf-8")
        env = BuiltEnvironment(root)
        rename = {"content": None, "tool_calls": [
            {"id": "c1", "name": "rename_widget",
             "arguments": {"widget_id": "w1", "label": "striped"}}]}
        model = TestModel([rename, "done", rename, "done"], loop=True)
        table = solve_rate(env, ["scored", "quiet"], model, runs_per_task=1,
                           outdir=Path(tmp) / "out")
        by_task = {row["task"]: row for row in table["tasks"]}
        assert by_task["scored"]["passes"] == 1
        assert by_task["scored"]["solve_rate"] == 1.0
        assert by_task["quiet"]["no_signal"] == 1
        assert by_task["quiet"]["solve_rate"] is None
        assert (Path(tmp) / "out" / "solve_rate.json").is_file()
        assert any(line.startswith("| scored ") for line in markdown_table(table))
