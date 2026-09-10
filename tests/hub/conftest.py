"""Fixtures for the publishing tests: the invented domain's workdir, its toolkit, and a host in memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nursery_domain import FakeHub, build_workdir

from kullback.runner.records import EntitySchema, RawPtr, ToolSig

# Two test folders both name a module `conftest`, and which one answers a bare `from conftest
# import ...` depends on the order pytest put them on the path. The shared value is repeated here
# so a test folder collected after this one still resolves it either way.
PTR = RawPtr(file_hash="testfile", sim_index=0)


@pytest.fixture
def nursery(tmp_path: Path) -> Path:
    return build_workdir(tmp_path / "work")


@pytest.fixture
def toolkit_source():
    """The module the Runner compiles for a workdir, so a test can call a tool the way a Run does."""
    from kullback.builder import compile_env

    def build(directory: Path):
        directory = Path(directory)
        schema = EntitySchema.model_validate(json.loads((directory / "schema.json").read_text()))
        sigs = [ToolSig.model_validate(row) for row in json.loads((directory / "tool_sigs.json").read_text())]
        bodies = json.loads((directory / "bodies.json").read_text())
        db = json.loads((directory / "env" / "db.json").read_text())
        return compile_env.load_toolkit(compile_env.module_source(schema, sigs, bodies), db)

    return build


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def hub_with_organisation_card() -> FakeHub:
    return FakeHub(organisation_card_supported=True)
