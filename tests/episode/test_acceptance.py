import sys
from types import SimpleNamespace

import pytest

from scripts.episode_acceptance import compare_world, main, stored_seed, verdict_outcome


def test_call_without_workdir_refuses_with_exit_two_and_replays_nothing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["episode_acceptance.py"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "--workdir" in capsys.readouterr().err


@pytest.mark.parametrize("event_type", ["tool_call", "tool_result", "user_turn", "stop", "error"])
def test_world_comparison_detects_payload_changes_for_every_world_event(event_type):
    left = [{"type": event_type, "payload": {"value": 1}}]
    right = [{"type": event_type, "payload": {"value": 2}}]
    assert compare_world(left, right)[0] is False


def test_world_comparison_ignores_only_outer_event_metadata():
    left = [{"idx": 1, "ts": "old", "type": "tool_call", "payload": {"name": "get_widget"}}]
    right = [{"idx": 9, "ts": "new", "type": "tool_call", "payload": {"name": "get_widget"}}]
    assert compare_world(left, right) == (True, "")


@pytest.mark.parametrize("key,value", [("route", "recording"), ("assisted", True)])
def test_world_comparison_detects_routing_and_assistance_changes(key, value):
    event = {"type": "tool_result", "payload": {"value": 1}}
    assert compare_world([event], [{**event, key: value}])[0] is False


@pytest.mark.parametrize("footer", [{}, {"seed": None}, {"seed": True}, {"seed": "7"}])
def test_acceptance_never_invents_a_missing_or_malformed_seed(footer):
    with pytest.raises(ValueError, match="integer seed"):
        stored_seed(footer)


def test_acceptance_uses_the_stored_seed_not_the_attempt_suffix():
    assert stored_seed({"seed": 123, "run_id": "task-7"}) == 123


def test_verdict_comparison_keeps_every_field_except_run_identity():
    calls = []
    def dump(**kwargs):
        calls.append(kwargs)
        return {"class_": "fail", "failing_atom": "a1", "notes": ["reason"]}
    assert verdict_outcome(SimpleNamespace(model_dump=dump))["failing_atom"] == "a1"
    assert calls == [{"mode": "json", "exclude": {"run_id"}}]
