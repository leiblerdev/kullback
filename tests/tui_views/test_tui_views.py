"""VIEWS lists every view in order with its palette key."""

from __future__ import annotations

import pytest

from kullback.tui.views import VIEWS, PublishView, RunsView, SynthesiseView, TasksView, TracesView

pytestmark = pytest.mark.anyio


async def test_views_lists_every_view_with_its_key(tmp_path):
    assert [(key, title) for key, title, _ in VIEWS] == [
        ("2", "Traces"), ("5", "Tasks"), ("6", "Runs"),
        ("", "Publish"), ("", "Synthesise"),
    ]
    for _, title, cls in VIEWS:
        assert cls.TITLE == title
        assert cls(tmp_path).workdir is not None
    assert [cls for _, _, cls in VIEWS] == [
        TracesView, TasksView, RunsView, PublishView, SynthesiseView]
