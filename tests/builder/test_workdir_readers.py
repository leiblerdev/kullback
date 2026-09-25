"""Tests for builder/sources/workdir_readers.py: loading, checking, and skipping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kullback.builder import ingest, sources


@pytest.fixture
def retire():
    """Unregister every toy adapter a test registered, keeping the seam clean."""
    names: list[str] = []
    yield names.append
    for name in names:
        sources.unregister(name)


def write_toy(path: Path) -> Path:
    path.write_text(json.dumps({"toy_format": "toy1", "runs": [{"id": "r1"}]}), encoding="utf-8")
    return path


def reader_source(name: str, for_file: str, mode: str, omit_for_file: bool = False) -> str:
    """One toy reader file: positive detect except in no_vote mode, clean mapping except in flip mode."""
    detect = 'isinstance(document, dict) and document.get("toy_format") == "toy1"'
    if mode == "no_vote":
        detect = "False"
    extra = ""
    if mode == "flip":
        extra = "        _FLIPS += 1\n"
    content = "\"done-again\" if _FLIPS % 2 == 0 and _FLIPS else \"done\"" if mode == "flip" else "\"done\""
    for_file_line = "" if omit_for_file else f'FOR_FILE = "{for_file}"\n'
    return f'''
from kullback.runner.records import RawPtr, ToolCall, Trace, Turn

_FLIPS = 0
{for_file_line}

class ToyAdapter:
    name = "{name}"
    maps = True
    display = "Toy"

    def detect(self, document, jsonl=False):
        if {detect}:
            return (1.0, ["has the toy marker"])
        return (0.0, ["no toy marker"])

    def recordings(self, document):
        return iter(document["runs"])

    def environment(self, document):
        return {{}}

    def sidecar(self, recording, document):
        return {{}}

    def to_trace(self, recording, ctx):
        global _FLIPS
{extra}        here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=0)
        call = ToolCall(id="c1", name="lookup", args={{}}, result={{"x": 1}},
                        requestor="assistant", has_result=True, resolved=True,
                        raw_ptr=here, result_ptr=here, trace_id=str(recording.get("id")))
        turn = Turn(idx=0, role="assistant", content={content},
                    tool_call_ids=["c1"], raw_ptr=here)
        return Trace(trace_id=str(recording.get("id")), raw_hash=ctx.raw_hash,
                     ingest_version=ctx.ingest_version, source=self.name,
                     turns=[turn], tool_calls=[call], raw_ptr=here)


ADAPTER = ToyAdapter()
'''


def write_reader(workdir: Path, filename: str, name: str, for_file: str, mode: str) -> Path:
    folder = workdir / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    target.write_text(reader_source(name, for_file, mode), encoding="utf-8")
    return target


def test_a_workdir_reader_that_passes_its_checks_is_used_by_ingest(tmp_path, workdir, retire):
    """A toy format with a toy reader ingests for real once the reader is in workdir/sources."""
    from kullback.builder.sources import workdir_readers

    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    write_reader(workdir, "toy_pass.py", "toy_pass", digest, "pass")
    retire("toy_pass")
    assert workdir_readers.load_with_reasons(workdir, toy) == (["toy_pass"], {})
    summary = ingest.ingest_file(toy, workdir)
    assert summary["format"] == "toy_pass"
    assert summary["runs"] == 1
    assert workdir_readers.load_workdir_readers(workdir) == ["toy_pass"]


def test_a_workdir_reader_that_fails_its_checks_is_skipped_with_a_reason(tmp_path, workdir, retire):
    """A reader whose detect votes nothing is left out, with why, and ingest still refuses unknown."""
    from kullback.builder.sources import workdir_readers

    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    write_reader(workdir, "toy_novote.py", "toy_novote", digest, "no_vote")
    retire("toy_novote")
    ingest.store_raw(toy, workdir)
    names, skipped = workdir_readers.load_with_reasons(workdir)
    assert names == []
    assert "toy_novote" in skipped and "no positive vote" in skipped["toy_novote"]
    assert sources.by_name("toy_novote") is None


def test_check_reader_catches_a_reader_that_is_not_deterministic(tmp_path, retire):
    """A reader that maps the same recording two ways fails the check on determinism."""
    from kullback.builder.sources import workdir_readers

    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    reader = tmp_path / "flip_reader.py"
    reader.write_text(reader_source("toy_flip", digest, "flip"), encoding="utf-8")
    retire("toy_flip")
    adapter = workdir_readers.adapter_from_path(reader)
    problems = workdir_readers.check_reader(adapter, toy, fixtures=[toy])
    assert any("not deterministic" in problem for problem in problems)


def test_a_workdir_reader_is_checked_against_the_file_on_its_first_ingest(tmp_path, workdir, retire):
    """A reader that fails its checks on that file is skipped on the first ingest, not run unchecked."""
    from kullback.builder.sources import workdir_readers

    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    write_reader(workdir, "toy_bad.py", "toy_bad", digest, "no_vote")
    retire("toy_bad")
    with pytest.raises(ValueError, match="unknown"):
        ingest.ingest_file(toy, workdir)
    assert sources.by_name("toy_bad") is None
    names, skipped = workdir_readers.load_with_reasons(workdir, toy)
    assert names == []
    assert "no positive vote" in skipped["toy_bad"]


def test_a_workdir_reader_without_for_file_is_skipped_with_a_reason(tmp_path, workdir, retire):
    """A reader that names no FOR_FILE has nothing to check against, so it stays out with why."""
    from kullback.builder.sources import workdir_readers

    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    folder = workdir / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "toy_nofor.py").write_text(
        reader_source("toy_nofor", digest, "pass", omit_for_file=True), encoding="utf-8")
    retire("toy_nofor")
    names, skipped = workdir_readers.load_with_reasons(workdir, toy)
    assert names == []
    assert "FOR_FILE" in skipped["toy_nofor"]
    assert sources.by_name("toy_nofor") is None


def test_workdir_readers_are_gone_from_the_registry_after_the_ingest_ends(tmp_path, workdir, retire):
    """An ingest leaves the registry as it found it, so one workdir never votes on the next file."""

    before = [adapter.name for adapter in sources.registered()]
    toy = write_toy(tmp_path / "toy.json")
    digest = hashlib.sha256(toy.read_bytes()).hexdigest()
    write_reader(workdir, "toy_gone.py", "toy_gone", digest, "pass")
    retire("toy_gone")
    summary = ingest.ingest_file(toy, workdir)
    assert summary["format"] == "toy_gone"
    assert sources.by_name("toy_gone") is None
    assert [adapter.name for adapter in sources.registered()] == before
