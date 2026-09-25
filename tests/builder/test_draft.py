"""Tests for builder/sources/draft.py: model drafted readers with isolated repairs."""

from __future__ import annotations

import json
from pathlib import Path

from kullback.ai.provider import TestModel
from kullback.builder.sources.draft import draft_reader
from kullback.builder.sources.reader_template import render_reader


def write_neutral(path: Path) -> Path:
    """One neutral toy file: two recordings with a role and content each."""
    path.write_text(json.dumps({"entries": [
        {"id": "rec-1", "role": "user", "content": "first neutral note"},
        {"id": "rec-2", "role": "assistant", "content": "second neutral note"},
    ]}), encoding="utf-8")
    return path


def neutral_mapping() -> dict[str, str]:
    return {"recordings": "entries", "role": "entries[].role",
            "content": "entries[].content"}


def fenced(name: str) -> str:
    """A working reader reply: one fenced python block around a rendered reader."""
    return "```python\n" + render_reader(name, "digest", neutral_mapping()) + "\n```"


def test_a_drafted_reader_that_passes_is_saved(tmp_path, workdir):
    """A scripted working reader is written under workdir/sources on the first call."""
    toy = write_neutral(tmp_path / "neutral.json")
    model = TestModel([fenced("drafted")])
    result = draft_reader(toy, workdir, model, "drafted")
    assert result == {"name": "drafted", "path": str(workdir / "sources" / "drafted.py"),
                      "passed": True, "problems": [],
                      "calls": 1, "attempts": [[]]}
    assert (workdir / "sources" / "drafted.py").is_file()


def test_a_drafted_reader_is_repaired_with_the_problems_listed(tmp_path, workdir):
    """A failing first draft is sent back with its problems, and the fix is saved."""
    toy = write_neutral(tmp_path / "neutral.json")
    model = TestModel(["```python\nnot a reader at all\n```", fenced("drafted")])
    result = draft_reader(toy, workdir, model, "drafted", repairs=1)
    assert result["passed"] is True
    assert result["calls"] == 2
    assert (workdir / "sources" / "drafted.py").is_file()
    second_prompt = model.calls[1]["messages"][1]["content"]
    assert "does not import" in second_prompt


def test_the_draft_prompt_holds_no_value_from_the_file_by_default(tmp_path, workdir):
    """A planted marker string reaches neither the shape summary nor the draft prompt."""
    marker = "field-x-marker-value-kept-out-of-the-prompt-0123456789"
    target = tmp_path / "neutral.json"
    target.write_text(json.dumps({"entries": [
        {"id": "rec-1", "role": "user", "content": marker},
        {"id": "rec-2", "role": "assistant", "content": "second neutral note"},
    ]}), encoding="utf-8")
    model = TestModel(["no code here"])
    draft_reader(target, workdir, model, "drafted", repairs=0)
    prompt = model.calls[0]["messages"][1]["content"]
    assert marker not in prompt


def test_draft_gives_up_after_the_repair_budget_with_the_last_problems(tmp_path, workdir):
    """Two replies with no fenced block spend the budget and keep the last problems."""
    toy = write_neutral(tmp_path / "neutral.json")
    model = TestModel(["no code here", "still no code"])
    result = draft_reader(toy, workdir, model, "drafted", repairs=1)
    assert result["name"] == "drafted"
    assert result["path"] is None
    assert result["passed"] is False
    assert result["calls"] == 2
    assert result["problems"] == ["the reply holds no fenced python block, "
                                  "so there is no reader to check"]
    assert result["attempts"] == [result["problems"], result["problems"]]
    assert not (workdir / "sources").exists()
