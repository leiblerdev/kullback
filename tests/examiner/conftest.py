"""The fixtures over the hand-built world of worlds.py: the world, and the same world derived."""

from __future__ import annotations

import pytest

from examiner.worlds import World, make_world, probe_runner_over
from kullback.examiner import stage
from kullback.gates.ledger import GateLedger


@pytest.fixture
def world(tmp_path) -> World:
    return make_world(tmp_path)


@pytest.fixture
def derived(world) -> World:
    """The world after the derivation ran: a Verifier for the Task, its status row and its history."""
    ctx = stage.ExamContext(world.workdir, GateLedger(world.workdir))
    stage.derive_all(ctx, world.inputs, probe_model=object(), run_probe=probe_runner_over())
    return world
