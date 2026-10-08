"""The rule user hears id asks, confirms only what is proposed, and ends only when the job is done (D326).

Every name here is invented: a ferry desk whose tools take a `passenger_id` and a `voyage_id`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from kullback.runner.records import DisclosureRule, UserFact, UserRules
from kullback.runner.tool import STOP_SHORT_LINE, runner_prompt
from kullback.user.fidelity import vocabulary_of
from kullback.user.rules import (
    CLOSING,
    CONFIRMATION,
    GOAL,
    GOAL_SATISFIED,
    SCENARIO_EXHAUSTED,
    TRANSFER_MARKER,
    asked_fields,
    goal_write_counts,
)
from kullback.user.simulated import REPEAT_ASK, TRANSFER_ACCEPT, USER_REPEAT, SimulatedUser
from kullback.user.vocabulary import GENERIC, _id_arg

SIGS = [
    {"name": "find_passenger", "kind": "read", "args_fields": [{"name": "passenger_id"}]},
    {"name": "rebook_voyage", "kind": "write",
     "args_schema": {"properties": {"voyage_id": {}, "deck": {}, "paid": {}, "valid": {}, "cabinId": {}}}},
]
GOAL_LINE = "I want to move my two crossings to a later day."


def rules(*facts: UserFact, confirmations: int = 0) -> UserRules:
    extra = [UserFact(field=CONFIRMATION, value="Yes, go ahead.")] * confirmations
    return UserRules(facts=[UserFact(field=GOAL, value=GOAL_LINE), *facts, *extra],
                     disclosure=[DisclosureRule(field=f.field, on_request=True) for f in facts])


def ask(text: str, *before: dict) -> list[dict]:
    return [{"role": "user", "content": "Hi."}, *before, {"role": "assistant", "content": text}]


def wrote(n: int) -> list[dict]:
    """n rebook calls that took effect, as tool result messages."""
    return [{"role": "tool", "tool_call_id": f"c{i}", "name": "rebook_voyage",
             "content": json.dumps({"status": "rebooked"})} for i in range(n)]


def test_a_workdir_without_a_vocabulary_hears_an_id_ask_through_its_tool_signatures(tmp_path):
    (tmp_path / "tool_sigs.json").write_text(json.dumps(SIGS))
    vocab = vocabulary_of(tmp_path)
    question = "Could you please provide your passenger ID and the voyage number?"
    assert asked_fields(question, vocab=GENERIC) == []
    assert asked_fields(question, vocab=vocab) == ["passenger_id", "voyage_id"]
    assert vocab.get("deck") is None and vocab.get("email") is not None
    assert vocab.get("paid") is None and vocab.get("valid") is None, "an id argument is id, *_id or *Id only"
    assert vocab.get("cabinId") is not None and all(map(_id_arg, ["id", "order_id", "orderId"]))
    assert vocab.get("voyage_id").sources == ["signature:rebook_voyage"]


def test_a_held_id_is_answered_even_when_the_ask_says_confirm():
    user = SimulatedUser(rules(UserFact(field="passenger_id", value="P-4471")))
    user.reply(ask("Hello, how can I help?"))
    text = user.reply(ask("Sure. Could you confirm your passenger id for me?"))
    assert "P-4471" in text and user.events[-1].payload["sources"] == {"passenger_id": "rules"}


def test_a_confirm_question_that_states_the_held_value_is_a_confirmation_not_an_ask():
    user = SimulatedUser(rules(UserFact(field="passenger_id", value="P-4471"), confirmations=1))
    user.reply(ask("Hello, how can I help?"))
    text = user.reply(ask("Shall I rebook both crossings for passenger P-4471?"))
    assert text == "Yes, go ahead."


def test_once_the_goal_writes_are_made_a_would_you_like_question_does_not_reuse_the_yes():
    user = SimulatedUser(rules(confirmations=1), write_tools={"rebook_voyage"},
                         goal_writes={"rebook_voyage"})
    user.reply(ask("Hello, how can I help?"))
    assert user.reply(ask("Shall I go ahead and rebook?")) == "Yes, go ahead."
    text = user.reply(ask("Done. Would you like me to add a cabin as well?", *wrote(1)))
    assert text != "Yes, go ahead." and not user.done


def test_the_goal_never_ends_a_run_and_is_reported_when_the_candidate_closes():
    user = SimulatedUser(rules(confirmations=1), write_tools={"rebook_voyage"},
                         goal_writes={"rebook_voyage"})
    user.reply(ask("Hello, how can I help?"))
    user.reply(ask("Shall I go ahead and rebook?"))
    user.reply(ask("Done, you are rebooked.", *wrote(1)))
    assert not user.done
    user.reply(ask("Is there anything else I can help with?", *wrote(1)))
    assert user.done and user.end_reason == GOAL_SATISFIED


def test_a_line_said_three_times_without_a_write_asks_what_is_missing_then_ends_the_run():
    user = SimulatedUser(rules(confirmations=1), write_tools={"rebook_voyage"},
                         goal_writes={"rebook_voyage"})
    user.reply(ask("Hello, how can I help?"))
    assert user.reply(ask("Do you want me to look up the timetable?")) == "Yes, go ahead."
    assert user.reply(ask("Do you want me to look up the timetable?")) == REPEAT_ASK
    assert user.done is False
    user.reply(ask("Do you want me to look up the timetable?"))
    assert user.done is True and user.end_reason == SCENARIO_EXHAUSTED
    assert USER_REPEAT in user.events[-1].payload["tags"]


def test_the_transfer_token_waits_one_agent_turn_before_it_is_sent():
    closing = UserFact(field=CLOSING, value=TRANSFER_MARKER)
    user = SimulatedUser(rules(closing))
    user.reply(ask("Hello, how can I help?"))
    user.reply(ask("Okay."))  # the one restatement of the goal
    first = user.reply(ask("I am transferring you to a human agent now."))
    assert first == TRANSFER_ACCEPT and TRANSFER_MARKER not in first and user.done is False
    second = user.reply(ask("Transfer requested.", {"role": "tool", "name": "hand_off", "content": "ok"}))
    assert second == TRANSFER_MARKER and user.done is True


def test_the_goal_is_not_done_with_one_of_two_requested_writes_made():
    reference = SimpleNamespace(tool_calls=[
        SimpleNamespace(name="rebook_voyage", error=None), SimpleNamespace(name="rebook_voyage", error=None),
        SimpleNamespace(name="rebook_voyage", error={"class": "refused"}),
        SimpleNamespace(name="find_passenger", error=None)])
    counts = goal_write_counts(reference, {"rebook_voyage"})
    assert counts == {"rebook_voyage": 2}
    user = SimulatedUser(rules(), write_tools={"rebook_voyage"}, goal_writes={"rebook_voyage"},
                         goal_counts=counts)
    user.reply(ask("Hello, how can I help?"))
    user.reply(ask("I have moved one crossing.", *wrote(1)))
    assert user.end_reason != GOAL_SATISFIED
    user.reply(ask("And now the second one too.", *wrote(2)))
    assert user.end_reason == GOAL_SATISFIED


def test_the_runner_prompt_ends_with_the_stop_short_line():
    prompt = runner_prompt("Be kind to passengers.")
    assert prompt.startswith("Be kind to passengers.") and prompt.endswith(STOP_SHORT_LINE)
    assert "list every item the user asked for" in prompt
    assert runner_prompt(None) is None
