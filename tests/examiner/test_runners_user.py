"""The agent user switch in the Examiner's re-rolls: off is the rule-driven user, on is the agent user
over that floor, priced under its own stage (D214)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


# --- the Examiner's own reroll tool ----------------------------------------------------------------

def _examiner_reroll(root: Path, user_model: Any) -> Any:
    import asyncio

    from kullback.examiner import domain_tools as D
    from kullback.examiner.exam_files import ExamRoot
    from kullback.runner.canon import CanonRules

    runners = R.runners_for(root, reroll_model=_asking_candidate(), user_model=user_model)
    exam = ExamRoot(workdir=root, reroll_model=_asking_candidate(), canon_rules=CanonRules(),
                    reroll_user=runners["reroll_user"])
    [tool] = [t for t in D.domain_tools(exam) if t.name == "reroll"]
    return asyncio.run(tool.execute(D.RerollArgs(task_id="widget_task", count=1)))


def _run_file(root: Path, run_id: str) -> Path:
    return root / "runs" / f"{run_id}.jsonl"


def test_with_the_switch_on_the_examiners_reroll_tool_meets_the_agent_user(tmp_path):
    root = _exam_env(tmp_path / "env")
    user = _user_model()
    result = _examiner_reroll(root, user)
    assert _user_turns(_run_file(root, result.runs[0]["run_id"]))[1] == ANSWER
    assert budget.load_totals(root)["stages"][R.USER_STAGE]["calls"] == len(user.calls) > 0


def test_with_the_switch_off_the_examiners_reroll_tool_meets_the_rule_driven_user(tmp_path):
    root = _exam_env(tmp_path / "env")
    factory = R.runners_for(root, reroll_model=_asking_candidate())["reroll_user"]
    assert type(factory("widget_task")(_Router(root))) is SimulatedUser
    result = _examiner_reroll(root, None)
    turns = _user_turns(_run_file(root, result.runs[0]["run_id"]))
    floor = R._make_user(root, "widget_task", None, _Router(root))
    floor.reply([])
    assert len(turns) > 1 and turns[1] == floor.reply(
        [{"role": "user", "content": turns[0]}, {"role": "assistant", "content": UNCUED_QUESTION}])
    assert R.USER_STAGE not in budget.load_totals(root)["stages"]


def test_with_the_switch_on_the_examiners_reroll_factory_builds_the_agent_user(tmp_path):
    root = _exam_env(tmp_path / "env")
    factory = R.runners_for(root, reroll_model=_asking_candidate(), user_model=_user_model())["reroll_user"]
    assert isinstance(factory("widget_task")(_Router(root)), AgentUser)


def test_an_effort_wrapped_user_model_carries_the_effort_on_every_user_turn(tmp_path):
    """The agent user's effort follows the run effort: each turn it takes carries it."""
    root = _exam_env(tmp_path / "env")
    user = _user_model()
    R.runners_for(root, reroll_model=_asking_candidate(),
                  user_model=budget.EffortModel(user, "low"))["run_rerolls"]("widget_task", 1, "on")
    assert len(user.calls) > 0
    assert all(call["config"].effort == "low" and call["config"].reasoning_effort == "low"
               for call in user.calls)


def test_the_builders_run_user_is_the_rule_user_alone_without_a_model(tmp_path):
    from kullback.builder import domain_tools as builder_tools

    root = _exam_env(tmp_path / "env")
    assert type(builder_tools._run_user(root, "widget_task", _Router(root))) is SimulatedUser


def test_the_builders_run_user_is_the_agent_user_over_the_floor_with_a_model(tmp_path):
    from kullback.builder import domain_tools as builder_tools

    root = _exam_env(tmp_path / "env")
    user = builder_tools._run_user(root, "widget_task", _Router(root), _user_model())
    assert isinstance(user, AgentUser) and type(user.fallback) is SimulatedUser
