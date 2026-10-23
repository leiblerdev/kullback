"""run_record reads the recorded file, so later writes cannot rewrite earlier results (G1).

The interface steps the world through `advance` (step-split landed); the guard below skips on
a tree without it.
"""

from __future__ import annotations

import json

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)

from kullback.runner.world import BuiltEnvironment, Episode, EpisodeError  # noqa: E402
from tests.episode.invented import DOMAIN_DB, TOOLKIT_HEAD, WIDGET_ROW, write_env  # noqa: E402

DATA_MODEL_UNCOPYABLE = WIDGET_ROW + '''
    def __deepcopy__(self, memo):
        """A live row cannot be copied once a later write has mutated it."""
        if self.label == "striped":
            raise RuntimeError("the live row must stay in the toolkit")
        return Widget(widget_id=self.widget_id, label=self.label)
''' + DOMAIN_DB


def _write_alias_env(root):
    (root / "env" / "tools.py").write_text(ALIAS_TOOLS, encoding="utf-8")
    (root / "env" / "data_model.py").write_text(DATA_MODEL_UNCOPYABLE, encoding="utf-8")

ALIAS_TOOLS = TOOLKIT_HEAD + '''
    @is_tool(ToolType.READ)
    def describe_widget(self, widget_id: str) -> Any:
        row = self.db.widgets.get(widget_id)
        if row is None:
            raise KeyError(widget_id)
        return row

    @is_tool(ToolType.WRITE)
    def rename_widget(self, widget_id: str, label: str) -> Any:
        row = self.db.widgets.get(widget_id)
        if row is None:
            raise KeyError(widget_id)
        row.label = label
        return {"widget_id": widget_id, "label": label}
'''


def test_run_record_matches_the_recorded_file(tmp_path):
    from kullback.runner.records import as_dict
    from kullback.runner.verdict import load_run

    root = write_env(tmp_path / "root")
    _write_alias_env(root)
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    episode.step({"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]})
    episode.step({"content": None, "tool_calls": [
        {"id": "c2", "name": "rename_widget", "arguments": {"widget_id": "w1", "label": "striped"}}]})
    out = episode.step({"content": "done", "tool_calls": []})
    assert out.done
    stored = json.loads(json.dumps(as_dict(load_run(episode.run_file())),
                                   ensure_ascii=False, default=str))
    record = episode.run_record()
    assert record["events"] == stored["events"]
    results = [e["payload"]["result"] for e in record["events"] if e.get("type") == "tool_result"]
    assert [r["label"] for r in results] == ["plain", "striped"]


def _drive_one_step(tmp_path):
    root = write_env(tmp_path / "root")
    _write_alias_env(root)
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    episode.step({"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]})
    return episode


@pytest.mark.parametrize("lose", ["file", "path"])
def test_run_record_missing_file_or_path_with_events_raises(tmp_path, lose):
    episode = _drive_one_step(tmp_path)
    path = episode.run_file()
    if lose == "file":
        path.unlink()
    else:
        # No public way leaves a started Run without a record path; the state is set directly.
        episode._state.path = None
    with pytest.raises(EpisodeError, match="record is missing") as excinfo:
        episode.run_record()
    if lose == "file":
        assert str(path) in str(excinfo.value)
