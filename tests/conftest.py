"""Fixtures every module's tests share: the invented worlds, a tmp workdir, and the offline models."""

from __future__ import annotations

import json
import os
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
def isolated_sessions_dir(tmp_path, monkeypatch):
    """No test reads or writes the real ~/.kullback/sessions. Every build a test runs writes a
    heartbeat, so without this a test run leaves a heartbeat per build behind in the person's own
    session list, and a screen test reads whatever builds they happen to be running right now."""
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))


@pytest.fixture(autouse=True)
def isolated_auth_file(tmp_path, monkeypatch):
    """No test reads or writes the real ~/.kullback/auth.json. Remembered keys are loaded by
    live_model and by both entries that open the screen, so without this a developer's own
    remembered keys leak into any test that reaches them."""
    monkeypatch.setenv("KULLBACK_AUTH_FILE", str(tmp_path / "auth.json"))


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


@pytest.fixture(scope="session", autouse=True)
def plain_terminal():
    """The suite never depends on the caller's terminal, so help and error text render plain."""
    saved = {key: os.environ.get(key) for key in ("NO_COLOR", "TERM", "COLUMNS", "FORCE_COLOR")}
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"
    os.environ["COLUMNS"] = "120"
    os.environ.pop("FORCE_COLOR", None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


WORLD_TRACE_FILES = {"world_a": "traces_small.json", "world_b": "traces_small_tau2.json"}


@pytest.fixture(scope="session")
def worlds_dir() -> Path:
    """The invented worlds every module's tests share, beside the retail fixtures."""
    return FIXTURES / "worlds"


@pytest.fixture(scope="session")
def world_dir(worlds_dir):
    """Factory: world_dir("world_a") is the library folder, world_dir("world_b") the nursery."""
    def _get(name: str) -> Path:
        if name not in WORLD_TRACE_FILES:
            raise ValueError(f"unknown world: {name}")
        return worlds_dir / name
    return _get


@pytest.fixture(scope="session")
def world_traces_path(world_dir):
    """Factory: the ingestible traces file of a world; world_b's is its tau2 twin."""
    def _get(name: str) -> Path:
        return world_dir(name) / WORLD_TRACE_FILES[name]
    return _get


@pytest.fixture(scope="session")
def world_traces(world_traces_path):
    """Factory: the parsed ingestible traces of a world, grader fields still in."""
    def _get(name: str) -> dict:
        return json.loads(world_traces_path(name).read_text(encoding="utf-8"))
    return _get


@pytest.fixture(scope="session", params=["world_a", "world_b"])
def world(request, world_dir, world_traces_path, world_traces) -> dict:
    """Each invented world in turn, so a test written once runs on both."""
    name = request.param
    return {
        "name": name,
        "dir": world_dir(name),
        "traces_path": world_traces_path(name),
        "traces": world_traces(name),
    }


@pytest.fixture(scope="session")
def world_a_small_path(world_traces_path) -> Path:
    """The 3-simulation library file, the drop-in replacement for tau2_small_path."""
    return world_traces_path("world_a")


@pytest.fixture(scope="session")
def world_a_small(world_traces) -> dict:
    """The parsed library file, the drop-in replacement for tau2_small."""
    return world_traces("world_a")


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
