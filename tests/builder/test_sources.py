"""Tests for the intake seam: adapters register with no seam edit, votes decide the format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kullback.builder import ingest, sources
from kullback.runner.records import RawPtr, Trace


class EmberAdapter:
    """One invented dict format: a payload with an embers list."""

    name = "ember"
    maps = True
    display = "Ember"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, dict) and isinstance(document.get("embers"), list):
            return (0.7, ["has an embers list"])
        return (0.0, ["no embers list"])

    def recordings(self, document: Any):
        return iter(document.get("embers", []))

    def environment(self, document: Any) -> dict:
        return {}

    def sidecar(self, recording: Any, document: Any) -> dict:
        return {}

    def to_trace(self, recording: Any, ctx: Any) -> Trace:
        return Trace(
            trace_id=str(recording.get("id", "ember")),
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=[],
            tool_calls=[],
            raw_ptr=RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index),
        )


class HarborAdapter(EmberAdapter):
    """A second invented dict format, shaped the same way on purpose."""

    name = "harbor"
    maps = True
    display = "Harbor"

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        if isinstance(document, dict) and isinstance(document.get("harbors"), list):
            return (0.7, ["has a harbors list"])
        return (0.0, ["no harbors list"])

    def recordings(self, document: Any):
        return iter(document.get("harbors", []))


@pytest.fixture
def toy_formats():
    """Two invented formats behind the seam; retired after each test."""
    sources.register(EmberAdapter())
    sources.register(HarborAdapter())
    try:
        yield
    finally:
        sources.unregister("ember")
        sources.unregister("harbor")


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# --- the seam needs no edit to add a format ---------------------------------


def test_a_registered_adapter_maps_through_derive_with_no_seam_edit(toy_formats, workdir, tmp_path):
    raw = ingest.store_raw(write_json(tmp_path / "ember.json", {"embers": [{"id": "e1"}]}), workdir)
    assert raw.format_detected == "ember"
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert [t.trace_id for t in traces] == ["e1"]
    assert traces[0].source == "ember"
    assert traces[0].raw_ptr.file_hash == raw.raw_hash
    other = ingest.store_raw(write_json(tmp_path / "harbor.json", {"harbors": [{"id": "h1"}]}), workdir)
    assert other.format_detected == "harbor"
    assert [t.trace_id for t in ingest.derive_traces(other.raw_hash, workdir)] == ["h1"]


# --- ambiguity gives unknown -------------------------------------------------


def test_a_payload_two_adapters_claim_is_refused_as_ambiguous(toy_formats, workdir, tmp_path):
    payload = {"embers": [{"id": "e1"}], "harbors": [{"id": "h1"}]}
    assert ingest.format_detect(payload) == "unknown"
    raw = ingest.store_raw(write_json(tmp_path / "tied.json", payload), workdir)
    with pytest.raises(ValueError, match="unknown"):
        ingest.derive_traces(raw.raw_hash, workdir)


def test_a_payload_no_adapter_claims_is_refused_with_the_reasons_in_words(workdir, tmp_path):
    raw = ingest.store_raw(write_json(tmp_path / "odd.json", {"hello": "world"}), workdir)
    assert ingest.format_detect({"hello": "world"}) == "unknown"
    with pytest.raises(ValueError, match="unknown") as caught:
        ingest.derive_traces(raw.raw_hash, workdir)
    assert "no positive evidence" in str(caught.value)


# --- a declared format is trusted before any guess from shape ----------------


def test_a_declared_format_or_span_kind_wins_over_a_shape_guess(toy_formats, workdir, tmp_path):
    payload = {"format": "harbor", "embers": [{"id": "e1"}], "harbors": [{"id": "h1"}]}
    assert ingest.format_detect(payload) == "harbor"
    raw = ingest.store_raw(write_json(tmp_path / "declared.json", payload), workdir)
    assert [t.trace_id for t in ingest.derive_traces(raw.raw_hash, workdir)] == ["h1"]
    assert ingest.format_detect({"span_kind": "ember", "embers": []}) == "ember"


# --- a messages list and an id alone are unknown ------------------------------


def test_a_bare_messages_list_with_an_id_is_unknown(workdir, tmp_path):
    payload = {"id": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert ingest.format_detect(payload) == "unknown"
    assert any("positive evidence" in reason for reason in ingest.detect_reasons(payload))
    raw = ingest.store_raw(write_json(tmp_path / "bare.json", payload), workdir)
    with pytest.raises(ValueError, match="positive evidence"):
        ingest.derive_traces(raw.raw_hash, workdir)


@pytest.mark.parametrize("payload", [
    {"id": "s1", "task_id": "t1", "messages": [{"role": "user", "content": "hi"}]},
    {"id": "s1", "messages": [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "name": "look_up", "arguments": {}, "requestor": "assistant"}]},
    ]},
], ids=["task-marker", "tool-structure"])
def test_one_recording_with_a_task_marker_or_tool_structure_still_detects(payload):
    assert ingest.format_detect(payload) == "tau2_native"
