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


def map_first(tmp_path: Path, name: str, payload: dict, mapping: dict):
    from kullback.builder.ingest import INGEST_VERSION, _decode
    from kullback.builder.sources import MapContext
    from kullback.builder.sources.workdir_readers import adapter_from_path

    target = tmp_path / f"{name}.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    reader = tmp_path / f"{name}.py"
    reader.write_text(render_reader(name, digest, mapping), encoding="utf-8")
    adapter = adapter_from_path(reader)
    document, _ = _decode(target.read_bytes())
    recordings = list(adapter.recordings(document))
    ctx = MapContext(raw_hash=digest, index=0, environment={}, ingest_version=INGEST_VERSION)
    return adapter, recordings, ctx


def base_mapping(**extra) -> dict:
    """A recordings, role and content mapping with optional extra entries."""
    return {"recordings": "entries", "role": "entries[].role",
            "content": "entries[].content", **extra}


def test_an_unmapped_role_word_refuses_the_reader(tmp_path):
    """A role word outside the four and outside the map fails the check naming the word."""
    target = tmp_path / "neutral.json"
    target.write_text(json.dumps({"entries": [
        {"id": "rec-1", "role": "bot", "content": "first neutral note"},
    ]}), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    reader = tmp_path / "reader.py"
    reader.write_text(render_reader("neutral", digest, base_mapping()), encoding="utf-8")
    problems = workdir_readers.check_reader_isolated(reader, target)
    assert problems
    assert any("bot" in problem for problem in problems)


def test_a_role_map_maps_each_word_to_its_role(tmp_path):
    """Words in the roles map read as their mapped role, and the four read as themselves."""
    from kullback.builder.sources import MapContext

    adapter, recordings, ctx = map_first(
        tmp_path, "mapped", {"entries": [
            {"id": "rec-1", "role": "agent", "content": "first neutral note"},
            {"id": "rec-2", "role": "client", "content": "second neutral note"},
        ]}, {**base_mapping(), "roles": {"agent": "assistant", "client": "user"}})
    assert [turn.role for turn in adapter.to_trace(recordings[0], ctx).turns] == ["assistant"]
    ctx2 = MapContext(raw_hash=ctx.raw_hash, index=1, environment={},
                      ingest_version=ctx.ingest_version)
    second = adapter.to_trace(recordings[1], ctx2)
    assert [turn.role for turn in second.turns] == ["user"]


def test_a_call_without_a_mapped_result_has_no_result(tmp_path):
    """A mapped call with no mapped result carries nothing the raw file does not hold."""
    adapter, recordings, ctx = map_first(
        tmp_path, "noresult", {"entries": [
            {"id": "rec-1", "role": "user", "content": "first neutral note",
             "calls": [{"call_id": "c1", "tool_name": "tool_a", "args": {"field_x": 1}}]},
        ]}, {**base_mapping(),
              "tool_calls": "entries[].calls", "tool_name": "entries[].calls[].tool_name",
              "tool_arguments": "entries[].calls[].args",
              "tool_call_id": "entries[].calls[].call_id"})
    (call,) = adapter.to_trace(recordings[0], ctx).tool_calls
    assert call.result is None
    assert call.has_result is False
    assert call.result_ptr is None


def test_calls_without_ids_pair_by_position_only_when_counts_match(tmp_path):
    """Id-free calls take the result at their position when counts match, else stay unpaired."""
    adapter, recordings, ctx = map_first(
        tmp_path, "positional", {"entries": [
            {"id": "rec-1", "role": "user", "content": "first neutral note",
             "calls": [{"tool_name": "tool_a"}, {"tool_name": "tool_b"}],
             "answers": [{"output": "first answer"}, {"output": "second answer"}]},
            {"id": "rec-2", "role": "user", "content": "second neutral note",
             "calls": [{"tool_name": "tool_a"}, {"tool_name": "tool_b"}],
             "answers": [{"output": "only answer"}]},
        ]}, {**base_mapping(),
              "tool_calls": "entries[].calls", "tool_name": "entries[].calls[].tool_name",
              "tool_results": "entries[].answers",
              "tool_result_content": "entries[].answers[].output"})
    matched = adapter.to_trace(recordings[0], ctx).tool_calls
    assert [call.result for call in matched] == ["first answer", "second answer"]
    assert all(call.has_result for call in matched)
    from kullback.builder.sources import MapContext

    ctx2 = MapContext(raw_hash=ctx.raw_hash, index=1, environment={},
                      ingest_version=ctx.ingest_version)
    lone = adapter.to_trace(recordings[1], ctx2).tool_calls
    assert [call.result for call in lone] == [None, None]
    assert all(call.has_result is False for call in lone)


def test_a_path_with_a_quote_renders_a_module_that_imports_and_maps(tmp_path):
    """A field name holding a quote renders quoted source that still imports and maps."""
    adapter, recordings, ctx = map_first(
        tmp_path, "quoted", {"entries": [
            {"id": "rec-1", "role": "user", "odd\"key": "quoted neutral"},
        ]}, {"recordings": "entries", "role": "entries[].role",
              "content": "entries[].odd\"key"})
    assert adapter.to_trace(recordings[0], ctx).turns[0].content == "quoted neutral"
