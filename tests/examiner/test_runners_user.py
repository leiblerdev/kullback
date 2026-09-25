"""The agent user switch in the Examiner's re-rolls: off is the rule-driven user, on is the agent user
over that floor, priced under its own stage (D214)."""

from __future__ import annotations

from pathlib import Path

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import runners as R
from kullback.runner import budget
from kullback.runner.records import load_run_jsonl
from kullback.runner.state import StateView
from kullback.runner.world.environment import BuiltEnvironment
from kullback.user.agent import AgentUser
from kullback.user.simulated import SimulatedUser
from tests.examiner.test_runners import _exam_env

# A question no cue of the rule-driven user reads as one: it asks the user to pick, in words the
# cues were never written for.
UNCUED_QUESTION = "There are two finishes on offer here. Which of those suits you best?"
ANSWER = "The striped one suits me best."


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


def _user_turns(path: str) -> list[str]:
    run = load_run_jsonl(path)
    return [str(event.payload.get("text")) for event in run.events if event.type == "user_turn"]


def test_with_the_switch_off_the_reroll_user_is_the_rule_driven_user(tmp_path):
    root = _exam_env(tmp_path / "env")
    user = R._make_user(root, "widget_task", None, _Router(root))
    assert type(user) is SimulatedUser


def test_with_the_switch_off_a_reroll_run_is_the_same_as_before_the_switch(tmp_path):
    before = _exam_env(tmp_path / "before")
    after = _exam_env(tmp_path / "after")
    old = R.runners_for(before, reroll_model=_asking_candidate())["run_rerolls"]("widget_task", 1, "b")
    new = R.runners_for(after, reroll_model=_asking_candidate(),
                        user_model=None)["run_rerolls"]("widget_task", 1, "b")
    assert _user_turns(old[0]["path"]) == _user_turns(new[0]["path"])
    assert old[0]["user_end"] == new[0]["user_end"]


def test_with_the_switch_on_the_reroll_user_is_the_agent_user_over_the_rule_floor(tmp_path):
    root = _exam_env(tmp_path / "env")
    user = R._make_user(root, "widget_task", None, _Router(root), user_model=_user_model())
    assert isinstance(user, AgentUser)
    assert type(user.fallback) is SimulatedUser


def test_with_the_switch_on_a_question_the_cues_miss_gets_an_answer_and_not_the_goal_again(tmp_path):
    root = _exam_env(tmp_path / "env")
    floor = R._make_user(root, "widget_task", None, _Router(root))
    transcript = [{"role": "user", "content": "Please give widget w1 the label striped."},
                  {"role": "assistant", "content": UNCUED_QUESTION}]
    floor.reply(transcript[:0])
    restated = floor.reply(transcript)
    assert ANSWER not in restated

    rows = R.runners_for(root, reroll_model=_asking_candidate(),
                         user_model=_user_model())["run_rerolls"]("widget_task", 1, "on")
    turns = _user_turns(rows[0]["path"])
    assert turns[1] == ANSWER


def test_the_agent_users_calls_are_priced_under_the_user_stage_and_never_the_runner(tmp_path):
    root = _exam_env(tmp_path / "env")
    seeded = budget.load_totals(root)["stages"].get(R.RUNNER_STAGE, {}).get("calls", 0)
    candidate, user = _asking_candidate(), _user_model()
    R.runners_for(root, reroll_model=candidate, user_model=user)["run_rerolls"]("widget_task", 1, "on")
    stages = budget.load_totals(root)["stages"]
    assert stages[R.USER_STAGE]["calls"] == len(user.calls)
    assert stages[R.RUNNER_STAGE]["calls"] - seeded == len(candidate.calls)
