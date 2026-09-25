"""Tests for builder/sources/shape.py: structure without customer strings."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_the_shape_summary_holds_no_values_only_types_counts_and_lengths(tmp_path):
    """A planted marker string appears nowhere in the summary, only types and sizes."""
    from kullback.builder.sources.shape import shape_summary

    marker = "shape-marker-that-is-longer-than-sixteen-chars"
    items = [{"content": f"{marker}-{idx}-with-padding-to-stay-unique"} for idx in range(3)]
    target = write_json(tmp_path / "notes.json", {"items": items})
    summary = shape_summary(target)
    assert marker not in json.dumps(summary)
    info = summary["paths"]["items[].content"]
    assert info["types"] == ["str"]
    assert info["count"] == 3
    assert "values" not in info
    assert info["min_len"] > 16


def test_a_role_like_field_is_hinted(tmp_path):
    """A small set of short words reads as a role and keeps its words."""
    from kullback.builder.sources.shape import shape_summary

    target = write_json(tmp_path / "rows.json", {"rows": [
        {"role": "assistant"}, {"role": "user"}, {"role": "tool"}, {"role": "assistant"},
    ]})
    summary = shape_summary(target)
    info = summary["paths"]["rows[].role"]
    assert "looks_like_role" in info["hints"]
    assert info["values"] == ["assistant", "tool", "user"]
