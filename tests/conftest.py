"""Fixtures every module's tests share: the small tau2 file, a tmp workdir, and the offline models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.shared import provider as provider_module
from harness.shared.provider import RecordedModel, TestModel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    """No test calls a real model (build brief rule 2)."""
    monkeypatch.setattr(provider_module, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def tau2_small_path() -> Path:
    """The 3-simulation tau2 retail file, grader fields still in it."""
    return FIXTURES / "tau2_retail_small.json"


@pytest.fixture(scope="session")
def tau2_small(tau2_small_path: Path) -> dict:
    return json.loads(tau2_small_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def tau2_retail_dir() -> Path:
    """tau2-bench's own retail domain files: db.json, policy.md, tasks.json."""
    return FIXTURES / "tau2_retail"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    """The full raw traces, never committed; tests that need them skip when they are absent."""
    path = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not path.is_dir():
        pytest.skip("raw traces not present")
    return path


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A fresh working directory; all module state lives under one of these, never in globals."""
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def make_test_model():
    """Factory: make_test_model(["hi", {"tool_calls": [...]}]) -> TestModel."""

    def _make(replies=None, **kwargs) -> TestModel:
        return TestModel(replies or [], **kwargs)

    return _make


@pytest.fixture
def test_model(make_test_model) -> TestModel:
    """A TestModel with one plain reply, for code that just needs a model to exist."""
    return make_test_model(["ok"])


@pytest.fixture
def write_run_jsonl(tmp_path: Path):
    """Factory: write lines (dicts) as a Run JSONL and return its path."""

    def _write(lines, name: str = "run.jsonl") -> Path:
        path = tmp_path / name
        with path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        return path

    return _write


@pytest.fixture
def make_recorded_model(write_run_jsonl):
    """Factory: make_recorded_model(lines) -> RecordedModel over a JSONL written from those lines."""

    def _make(lines, name: str = "run.jsonl") -> RecordedModel:
        return RecordedModel(write_run_jsonl(lines, name))

    return _make
