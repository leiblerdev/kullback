"""Tests for builder/sources/reader_template.py and the isolated reader check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kullback.builder import ingest, sources
from kullback.builder.sources import workdir_readers
from kullback.builder.sources.reader_template import render_reader


@pytest.fixture
def retire():
    """Unregister every neutral adapter a test registered, keeping the seam clean."""
    names: list[str] = []

    def track(name: str) -> None:
        names.append(name)

    yield track
    for name in names:
        sources.unregister(name)


def write_neutral(path: Path) -> Path:
    """One neutral toy file: two recordings with role, content and a tool round trip."""
    path.write_text(json.dumps({"entries": [
        {"id": "rec-1", "role": "user", "content": "first neutral note",
         "calls": [{"call_id": "c1", "tool_name": "tool_a", "args": {"field_x": 1}}],
         "answers": [{"call_id": "c1", "output": {"field_x": 2}}]},
        {"id": "rec-2", "role": "assistant", "content": "second neutral note",
         "calls": [{"call_id": "c2", "tool_name": "tool_a", "args": {"field_x": 3}}],
         "answers": [{"call_id": "c2", "output": {"field_x": 4}}]},
    ]}), encoding="utf-8")
    return path


def neutral_mapping() -> dict[str, str]:
    return {"recordings": "entries", "role": "entries[].role", "content": "entries[].content",
            "tool_calls": "entries[].calls", "tool_name": "entries[].calls[].tool_name",
            "tool_arguments": "entries[].calls[].args",
            "tool_call_id": "entries[].calls[].call_id",
            "tool_results": "entries[].answers",
            "tool_result_id": "entries[].answers[].call_id",
            "tool_result_content": "entries[].answers[].output"}


def test_a_hand_mapped_reader_for_a_toy_format_passes_its_checks_and_ingest_uses_it(
        tmp_path, workdir, retire):
    """A rendered reader maps a neutral toy file to two Traces ingest stores under its name."""
    toy = write_neutral(tmp_path / "neutral.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    source = render_reader("neutral", digest, neutral_mapping())
    reader = workdir / "sources" / "neutral.py"
    reader.parent.mkdir(parents=True, exist_ok=True)
    reader.write_text(source, encoding="utf-8")
    retire("neutral")
    assert workdir_readers.check_reader_isolated(reader, toy) == []
    assert workdir_readers.load_with_reasons(workdir, toy) == (["neutral"], {})
    summary = ingest.ingest_file(toy, workdir)
    assert summary["format"] == "neutral"
    assert summary["runs"] == 2


def test_the_isolated_check_blocks_the_network(tmp_path):
    """A reader importing socket at module level fails the isolated check on the import."""
    toy = write_neutral(tmp_path / "neutral.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    lines = render_reader("neutral", digest, neutral_mapping()).splitlines(keepends=True)
    lines.insert(8, "import socket\n")
    reader = tmp_path / "reader.py"
    reader.write_text("".join(lines), encoding="utf-8")
    problems = workdir_readers.check_reader_isolated(reader, toy)
    assert any("socket" in problem for problem in problems)
