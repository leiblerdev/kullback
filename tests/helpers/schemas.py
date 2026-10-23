"""Entity schemas the tests build Environments over."""

from __future__ import annotations

from kullback.runner.records import Column, EntitySchema


def widget_status_schema() -> EntitySchema:
    """One table of widgets keyed by a W- id, with a status column."""
    columns = [Column(table="widgets", name=name, **{"class": "hard"}) for name in ("widget_id", "status")]
    return EntitySchema(tables=["widgets"], columns=columns, id_patterns={"widgets.widget_id": r"^W-\d+$"})
