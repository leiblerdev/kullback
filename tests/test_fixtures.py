"""The checked-in fixtures keep the shape the module tests are written against."""

from __future__ import annotations

GRADER_FIELDS = {"reward_info", "trial"}


def test_grader_fields_are_still_present_so_ingest_can_be_tested_stripping_them(world: dict):
    for sim in world["traces"]["simulations"]:
        assert GRADER_FIELDS.issubset(sim)
        assert "action_checks" in sim["reward_info"]
    for task in world["traces"]["tasks"]:
        assert "evaluation_criteria" in task
