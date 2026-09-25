"""The real-export smoke tier's file fixtures: the small recorded file and its helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.runner.records import RawPtr

# Every Turn, ToolCall and Trace carries a raw_ptr (D66), so a test that does not care where its
# record came from still has to name a raw location. This is that location.
PTR = RawPtr(file_hash="testfile", sim_index=0)

_SMOKE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

@pytest.fixture(scope="session")
def tau2_small_path() -> Path:
    """The small recorded file, grader fields still in it."""
    return _SMOKE_FIXTURES / "tau2_retail_small.json"


@pytest.fixture(scope="session")
def tau2_small(tau2_small_path: Path) -> dict:
    return json.loads(tau2_small_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def tau2_retail_dir() -> Path:
    """The recorded domain files: db.json, policy.md, tasks.json."""
    return _SMOKE_FIXTURES / "tau2_retail"


# The full raw_dir also holds other exports' traces (fetched for a separate,
# cross-domain check); tools mined from the whole folder would be gated against export-only
# thresholds, so tests scoped to one export read just these two files by name.
RETAIL_RAW_FILES = (
    "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
    "claude-sonnet-4-5_enabled_retail_gpt-5.2_4trials.json",
)


@pytest.fixture
def retail_raw_files(raw_dir: Path) -> list[Path]:
    """The two trace files named above, in raw_dir; other exports' traces are checked elsewhere."""
    return [raw_dir / name for name in RETAIL_RAW_FILES]
