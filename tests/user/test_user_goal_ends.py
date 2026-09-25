"""When a Simulated user ends on its goal: a goal naming no writes is done by a Candidate that used
a tool and then closed, never by the empty write set alone. Both drivers read the one predicate.

Every name is abstract, so the tests prove the rule and not any world.
"""

from __future__ import annotations

from kullback.ai.provider import TestModel
from kullback.runner.records import UserFact, UserRules
from kullback.user import context as context_mod
from kullback.user.agent import AgentUser
from kullback.user.rules import GOAL_SATISFIED, HANDED_OFF
from kullback.user.simulated import SimulatedUser

WRITES = {"write_a"}
QUESTION = "Before I look, could you tell me a bit more about what you need?"
CLOSE = "Is there anything else I can help you with?"


def _rules() -> UserRules:
    return UserRules(facts=[UserFact(field="goal", value="I want to know the state of my thing.")])


def _user(goal_writes) -> SimulatedUser:
    return SimulatedUser(_rules(), write_tools=WRITES, goal_writes=goal_writes)


def _call(name: str, result: str = '{"ok": true}') -> list:
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": name, "arguments": "{}"}]},
            {"role": "tool", "name": name, "tool_call_id": "c1", "content": result}]


def _said(content: str) -> dict:
    return {"role": "assistant", "content": content}


def test_a_goal_with_no_writes_is_not_ended_by_a_candidate_that_asks_a_question_first():
    user = _user(set())
    said = user.reply([_said(QUESTION)])
    assert said.strip()
    assert not user.done and user.end_reason is None


def test_a_goal_with_no_writes_ends_satisfied_when_the_candidate_closes_after_a_tool_call():
    user = _user(set())
    user.reply([_said(QUESTION)])
    user.reply([_said(QUESTION), *_call("read_a"), _said(CLOSE)])
    assert user.done and user.end_reason == GOAL_SATISFIED


def test_a_goal_with_no_writes_is_handed_off_when_the_candidate_closes_without_a_tool_call():
    user = _user(set())
    user.reply([_said(QUESTION)])
    user.reply([_said(QUESTION), _said(CLOSE)])
    assert user.done and user.end_reason == HANDED_OFF


def test_a_goal_with_writes_still_ends_satisfied_once_they_are_made():
    user = _user(WRITES)
    user.reply([_said(QUESTION)])
    assert not user.done
    user.reply([_said(QUESTION), *_call("write_a", '{"before": 1, "after": 2}'), _said("Done. Anything else?")])
    assert user.done and user.end_reason == GOAL_SATISFIED


def _agent(goal_writes) -> AgentUser:
    return AgentUser(context_mod.curate("t", _rules()), _user(goal_writes),
                     TestModel(["I just want to know how it stands."], loop=True),
                     write_tools=WRITES, goal_writes=goal_writes)


def test_the_agent_user_does_not_end_a_goal_with_no_writes_on_a_first_question():
    user = _agent(set())
    user.reply([_said(QUESTION)])
    assert user.protocol.kind(QUESTION, said_anything=True, had_nothing=False) is None


def test_the_agent_user_ends_a_goal_with_no_writes_by_the_same_rule_as_the_rule_user():
    user = _agent(set())
    assert user.protocol.kind(CLOSE, said_anything=True, had_nothing=False, acted=True) == GOAL_SATISFIED
    assert _agent(set()).protocol.kind(CLOSE, said_anything=True, had_nothing=False) == HANDED_OFF
