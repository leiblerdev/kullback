"""Fixtures every module's tests share: the small tau2 file, a tmp workdir, and the offline models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.ai import provider as provider_module
from kullback.ai.provider import RecordedModel, TestModel
from kullback.runner.records import RawPtr

FIXTURES = Path(__file__).parent / "fixtures"

# Every Turn, ToolCall and Trace carries a raw_ptr (D66), so a test that does not care where its
# record came from still has to name a raw location. This is that location.
PTR = RawPtr(file_hash="testfile", sim_index=0)


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    """No test calls a real model (build brief rule 2)."""
    monkeypatch.setattr(provider_module, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture(autouse=True)
def isolated_price_catalog(tmp_path, monkeypatch):
    """No test reads or writes the real ~/.cache/harness/models.dev.json snapshot, for prices or
    for the provider registry, and every test starts with budget's price-catalog cache unloaded
    rather than carrying over the previous test's snapshot."""
    from kullback.ai import provider as provider_module
    from kullback.runner import budget as budget_module

    monkeypatch.setattr(budget_module, "_SNAPSHOT_PATH", tmp_path / "models.dev.json")
    monkeypatch.setattr(provider_module, "REGISTRY_SNAPSHOT_PATH", str(tmp_path / "models.dev.json"))
    monkeypatch.setattr(budget_module, "_CATALOG_LOADED", False)
    monkeypatch.setattr(budget_module, "_CATALOG", None)


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
    """The full raw traces, never committed; tests that need them skip when they are absent.

    Kullback keeps them at data/raw under the package root; the brain keeps them one level
    up, at monitoring-tool/data/raw (../data/raw from here). Try the kullback layout first.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root / "data" / "raw", repo_root.parent / "data" / "raw"):
        if candidate.is_dir():
            return candidate
    pytest.skip("raw traces not present")


# The full raw_dir corpus also holds airline and telecom traces (fetched for a separate,
# cross-domain check); tools mined from the whole folder would be gated against retail-only
# thresholds, so tests scoped to retail read just these two files by name.
RETAIL_RAW_FILES = (
    "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
    "claude-sonnet-4-5_enabled_retail_gpt-5.2_4trials.json",
)


@pytest.fixture
def retail_raw_files(raw_dir: Path) -> list[Path]:
    """The two retail trace files in raw_dir; airline and telecom traces are checked elsewhere."""
    return [raw_dir / name for name in RETAIL_RAW_FILES]


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
