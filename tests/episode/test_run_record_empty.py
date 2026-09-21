"""run_record before any event falls back to the in-memory header (G1).

Needs no step: the Episode is built without its constructor (which requires the step-split
patch) and handed a fresh Run state, so this runs on any runtime.
"""

from types import SimpleNamespace

from kullback.episode.episode import Episode
from kullback.runner.records import Run


def _episode_with_fresh_run(tmp_path, *, path):
    episode = Episode.__new__(Episode)
    episode._state = SimpleNamespace(
        run=Run(run_id="widget_task-7", task_id="widget_task", model="episode", seed=7),
        path=path)
    return episode


def test_run_record_without_events_returns_the_header(tmp_path):
    episode = _episode_with_fresh_run(tmp_path, path=tmp_path / "widget_task-7.jsonl")
    record = episode.run_record()
    assert record["events"] == []
    assert record["run_id"] == "widget_task-7"
    assert record["seed"] == 7


def test_run_record_without_events_or_path_returns_the_header(tmp_path):
    episode = _episode_with_fresh_run(tmp_path, path=None)
    record = episode.run_record()
    assert record["events"] == []
    assert record["run_id"] == "widget_task-7"
