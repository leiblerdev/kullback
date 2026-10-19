"""The agent user switch in the Examiner's re-rolls: off is the rule-driven user, on is the agent user
over that floor, priced under its own stage (D214)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.builder import run_user as R
from kullback.runner.records import load_run_jsonl
from kullback.runner.state import StateView
from kullback.runner.world.environment import BuiltEnvironment
from kullback.user.agent import AgentUser
from kullback.user.simulated import SimulatedUser
from tests.examiner.test_runners import _exam_env as _bare_env


def _exam_env(path: Path) -> Path:
    """The examiner env with a Spec that names its one write, which the user's goal reads."""
    from kullback.spec.schema import Check, Spec, SpecIntent, save_spec
    root = _bare_env(path)
    called = {"id": "c1", "kind": "called", "tool": "rename_widget"}
    check = Check(id="c1", kind="allowed", demand=called, because="asked", tier="critical")
    save_spec(root, Spec(task_id="widget_task", intent=SpecIntent(task_id="widget_task"), checks=[check]))
    return root

# A question no cue of the rule-driven user reads as one: it asks the user to pick, in words the
# cues were never written for.
UNCUED_QUESTION = "There are two of those on offer here. Which of them suits you best?"
ANSWER = "The second one suits me best."


def _asking_candidate() -> TestModel:
    """A candidate that asks the uncued question first, then does the rename and stops."""
    return TestModel([
        ModelReply(content=UNCUED_QUESTION, model="test"),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m1", name="describe_widget", arguments={"widget_id": "w1"})]),
        ModelReply(content=None, model="test", tool_calls=[
            ToolCallRequest(id="m2", name="rename_widget",
                            arguments={"widget_id": "w1", "label": "striped"})]),
        ModelReply(content="Done.", model="test"),
    ], loop=True)


def _user_model() -> TestModel:
    return TestModel(["Please give widget w1 the label striped.", ANSWER], name="user-test", loop=True)


class _Router:
    """What the runner hands `make_user`: the Task's Starting state behind `.state`."""

    def __init__(self, root: Path):
        self.state = StateView(BuiltEnvironment(root).db)


def _user_turns(path: Any) -> list[str]:
    run = load_run_jsonl(path)
    return [str(event.payload.get("text")) for event in run.events if event.type == "user_turn"]


def test_with_the_switch_off_the_reroll_user_is_the_rule_driven_user_with_the_builds_vocabulary_and_counts(tmp_path):
    root = _exam_env(tmp_path / "env")
    user = R._make_user(root, "widget_task", None, _Router(root))
    assert type(user) is SimulatedUser
    assert user.goal_counts == {"rename_widget": 1}
    assert "widget_id" in [field.field for field in user.vocab.fields]


def test_with_the_switch_on_the_reroll_user_is_the_agent_user_over_the_rule_floor(tmp_path):
    root = _exam_env(tmp_path / "env")
    user = R._make_user(root, "widget_task", None, _Router(root), user_model=_user_model())
    assert isinstance(user, AgentUser)
    assert type(user.fallback) is SimulatedUser


# --- the Examiner's own reroll tool ----------------------------------------------------------------


def test_the_builders_run_user_is_the_rule_user_alone_without_a_model(tmp_path):
    from kullback.builder import domain_tools as builder_tools

    root = _exam_env(tmp_path / "env")
    user = builder_tools._run_user(root, "widget_task", _Router(root))
    assert type(user) is SimulatedUser and user.goal_counts == {"rename_widget": 1}


def test_the_builders_run_user_is_the_agent_user_over_the_floor_with_a_model(tmp_path):
    from kullback.builder import domain_tools as builder_tools

    root = _exam_env(tmp_path / "env")
    user = builder_tools._run_user(root, "widget_task", _Router(root), _user_model())
    assert isinstance(user, AgentUser) and type(user.fallback) is SimulatedUser
