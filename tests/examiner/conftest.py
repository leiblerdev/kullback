"""The fixtures over the two worlds of worlds.py: the hand-built world, the same world derived, and the
fixture build made once per session."""

from __future__ import annotations

import pytest

from examiner.worlds import FixtureBuild, World, build_fixture, make_world, probe_runner_over
from kullback.examiner import agent as examiner_agent


@pytest.fixture
def world(tmp_path) -> World:
    return make_world(tmp_path)


@pytest.fixture
def derived(world) -> World:
    """The world after the code driver derived it: a Verifier for the Task, its status row and its history."""
    examiner_agent.run_examiner(world.workdir, inputs=world.inputs, probe_model=object(),
                                run_probe=probe_runner_over())
    return world


@pytest.fixture(scope="session")
def fixture_build(tmp_path_factory, request) -> FixtureBuild:
    return build_fixture(tmp_path_factory.mktemp("fixture-build"), request)
