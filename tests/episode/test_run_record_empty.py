"""run_record before any event falls back to the in-memory header (G1).

The Episode is built for real and handed a fresh Run state from the loop, with or without a
record file, so no event has been written yet.
"""

import pytest

from kullback.runner import loop
from kullback.runner.world import BuiltEnvironment, Episode
from tests.episode.invented import write_env


@pytest.mark.parametrize("with_path", [True, False])
def test_run_record_without_events_returns_the_header(tmp_path, with_path):
    episode = Episode(BuiltEnvironment(write_env(tmp_path / "root")), outdir=tmp_path / "out")
    episode._state = loop.new_run_state(
        "widget_task-7", path=tmp_path / "widget_task-7.jsonl" if with_path else None,
        task_id="widget_task", model="episode", seed=7)
    record = episode.run_record()
    assert record["events"] == []
    assert record["run_id"] == "widget_task-7"
    assert record["seed"] == 7
