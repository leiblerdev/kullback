"""A fresh Run is played only against a Simulated user: a Task with no rules is named, never played (F34)."""

from __future__ import annotations

from kullback.ai.provider import TestModel
from kullback.builder import domain_tools as domain_tools_mod
from kullback.runner import budget
from tests.builder.test_domain_tools import _run, _workdir
from tests.helpers.scripted import reply


def _candidate() -> TestModel:
    return TestModel([reply(None, ("rename_widget", {"widget_id": "w1", "label": "striped"})),
                      reply("Done, widget w1 is labelled striped.")], loop=True)


def _run_tool(root, model):
    (run,) = [tool for tool in domain_tools_mod.domain_tools(workdir=root, model=model) if tool.name == "run"]
    return run


def test_a_task_with_an_unconfirmed_reference_is_named_not_run_and_the_model_is_never_called(tmp_path):
    root = _workdir(tmp_path, with_rules=False)
    candidate = _candidate()
    result = _run(_run_tool(root, candidate), {"task_ids": ["widget_task"], "count": 2})
    assert not result.is_error, result.content
    assert candidate.calls == []
    assert result.details["runs"] == []
    assert result.details["not_run"] == ["widget_task"]
    assert result.details["reasons"] == [domain_tools_mod.NO_USER_REASON]
    assert "not run, no Simulated user: the Reference is not confirmed, replay first: widget_task" in result.content
    assert "runner" not in budget.load_totals(root)["stages"]


def test_a_task_with_a_confirmed_reference_plays_against_its_simulated_user(tmp_path):
    root = _workdir(tmp_path)
    candidate = _candidate()
    result = _run(_run_tool(root, candidate), {"task_id": "widget_task", "count": 1})
    assert not result.is_error, result.content
    (row,) = result.details["runs"]
    assert row["user_end"] == "goal_satisfied"
    assert result.details["not_run"] == [] and result.details["reasons"] == []
    assert result.content.startswith("reroll of task widget_task: 1 Runs, 1 finished")


def test_the_cap_keeps_its_own_reason_beside_the_missing_user():
    text = domain_tools_mod._render_run(domain_tools_mod.RunResult(
        summary="reroll", not_run=["a_task", "b_task"],
        reasons=[domain_tools_mod.NO_USER_REASON, domain_tools_mod.CAP_REASON]))
    assert "not run, no Simulated user: the Reference is not confirmed, replay first: a_task" in text
    assert text.endswith(f"not run, past the cap of {domain_tools_mod.RUNS_PER_CALL} Runs per call: b_task")


class _Router:
    """What the runner hands `make_user`: the Task's Starting state behind `.state`."""

    def __init__(self, root):
        from kullback.runner.state import StateView
        from kullback.runner.world.environment import BuiltEnvironment

        self.state = StateView(BuiltEnvironment(root).db)


def _user_model() -> TestModel:
    return TestModel(["Please give widget w1 the label striped."], name="user-test", loop=True)


def test_a_task_without_replay_rows_gets_the_agent_user_built_from_its_recordings_rules_on_disk(tmp_path):
    from kullback.builder import run_user as R
    from kullback.user.agent import AgentUser

    root = _workdir(tmp_path)
    assert not (root / "replays.json").exists()
    user = R.run_user(root, "widget_task", _Router(root), _user_model())
    assert isinstance(user, AgentUser)
    assert [(fact.field, fact.value) for fact in user.fallback.rules.facts] == [("widget_id", "w1"), ("label", "striped")]


def test_a_task_without_replay_rows_or_user_rules_gets_no_agent_user(tmp_path):
    from kullback.builder import run_user as R

    root = _workdir(tmp_path, with_rules=False)
    assert R.run_user(root, "widget_task", _Router(root), _user_model()) is None


def test_a_task_whose_replay_rows_are_none_confirmed_still_gets_no_agent_user(tmp_path):
    import json

    from kullback.builder import run_user as R

    root = _workdir(tmp_path)
    (root / "replays.json").write_text(json.dumps({"widget_task": {"rec1": {"trace_id": "rec1", "confirmed": False}}}))
    assert R.run_user(root, "widget_task", _Router(root), _user_model()) is None
