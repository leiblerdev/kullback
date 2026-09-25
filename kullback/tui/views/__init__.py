"""The Textual full-screen views: traces, tasks, runs, publish and synthesise.

Each view takes the workdir and works on any Environment inside any Textual App; the app
shell mounts them by key. Nothing here keeps state of its own: every number is read from
the workdir's files when the view asks.
"""

from __future__ import annotations

from kullback.tui.views.publish import PublishView
from kullback.tui.views.runs import RunsView
from kullback.tui.views.synthesise import SynthesiseView
from kullback.tui.views.tasks import TasksView
from kullback.tui.views.traces import TracesView

VIEWS: tuple = (
    ("2", TracesView.TITLE, TracesView),
    ("5", TasksView.TITLE, TasksView),
    ("6", RunsView.TITLE, RunsView),
    ("", PublishView.TITLE, PublishView),
    ("", SynthesiseView.TITLE, SynthesiseView),
)

__all__ = ["VIEWS", "PublishView", "RunsView", "SynthesiseView", "TasksView", "TracesView"]
