import pytest

from kullback.runner.records import Verdict
from scripts.episode_acceptance import compare_world, stored_seed, verdict_outcome


@pytest.mark.parametrize("event_type,key,value", [
    *[(event_type, "payload", {"value": 2})
      for event_type in ["tool_call", "tool_result", "user_turn", "stop", "error"]],
    ("tool_result", "route", "recording"),
    ("tool_result", "assisted", True),
])
def test_world_comparison_detects_payload_routing_and_assistance_changes_for_every_world_event(
        event_type, key, value):
    event = {"type": event_type, "payload": {"value": 1}}
    assert compare_world([event], [{**event, key: value}])[0] is False


def test_world_comparison_ignores_only_outer_event_metadata():
    left = [{"idx": 1, "ts": "old", "type": "tool_call", "payload": {"name": "get_widget"}}]
    right = [{"idx": 9, "ts": "new", "type": "tool_call", "payload": {"name": "get_widget"}}]
    assert compare_world(left, right) == (True, "")


@pytest.mark.parametrize("footer", [{}, {"seed": None}, {"seed": True}, {"seed": "7"}])
def test_acceptance_uses_the_stored_seed_and_never_invents_a_missing_or_malformed_one(footer):
    assert stored_seed({"seed": 123, "run_id": "task-7"}) == 123
    with pytest.raises(ValueError, match="integer seed"):
        stored_seed(footer)


def test_verdict_comparison_keeps_every_field_except_run_identity():
    verdict = Verdict.model_validate({"run_id": "task-7", "pass": False, "class": "fail",
                                      "failing_atom": "a1", "notes": ["reason"]})
    outcome = verdict_outcome(verdict)
    assert outcome["failing_atom"] == "a1" and outcome["notes"] == ["reason"]
    full = verdict.model_dump(mode="json")
    full.pop("run_id")
    assert "run_id" not in outcome and outcome == full
