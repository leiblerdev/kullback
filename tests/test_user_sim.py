"""Tests for builder/user_sim.py: rules derived from a trace, and a Simulated user that never invents."""

from __future__ import annotations

import json

import pytest

from harness.builder.user_sim import (
    CHOICE,
    CLOSING,
    CONFIRMATION,
    GOAL,
    FactLookup,
    SimulatedUser,
    asked_fields,
    derive_user_rules,
    dict_reader,
    extracted_values,
)
from harness.shared.records import (
    DisclosureRule,
    RawPtr,
    Trace,
    Turn,
    UserFact,
    UserRules,
    as_dict,
)


def make_trace(pairs, trace_id: str = "t1") -> Trace:
    """pairs is [(role, content), ...] in transcript order."""
    turns = [
        Turn(
            idx=i,
            role=role,
            content=content,
            raw_ptr=RawPtr(file_hash="rawhash", sim_index=0, msg_index=i),
        )
        for i, (role, content) in enumerate(pairs)
    ]
    return Trace(
        trace_id=trace_id,
        raw_hash="rawhash",
        ingest_version="0",
        source="tau2",
        turns=turns,
    )


def facts_by_field(rules: UserRules) -> dict:
    """Field values, the first one recorded per field; free-text fields are asserted by name."""
    values: dict = {}
    for fact in rules.facts:
        values.setdefault(fact.field, fact.value)
    return values


def field_values(rules: UserRules, field: str) -> list:
    return [fact.value for fact in rules.facts if fact.field == field]


def disclosure_by_field(rules: UserRules) -> dict:
    return {rule.field: rule for rule in rules.disclosure}


# --- derive_user_rules ---


def test_facts_come_from_user_turns_with_a_span():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Hi! What is your email address?"),
                ("user", "It is yusuf.rossi@example.com, thanks."),
                ("assistant", "Which order id is this about? my email is agent@shop.com"),
                ("user", "Order #W2378156 please."),
            ]
        )
    )
    assert facts_by_field(rules) == {
        "email": "yusuf.rossi@example.com",
        "order_id": "#W2378156",
    }
    email = rules.facts[0]
    assert isinstance(email, UserFact)
    assert email.span == RawPtr(file_hash="rawhash", sim_index=0, msg_index=1)


def test_assistant_turns_are_not_a_source_of_facts():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "I have you as mei@example.com, order #W1234567."),
                ("user", "Sounds right."),
            ]
        )
    )
    assert [fact.field for fact in rules.facts] == [GOAL]
    assert facts_by_field(rules)[GOAL] == "Sounds right."


def test_disclosure_marks_asked_facts_on_request_and_volunteered_facts_not():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Hi! How can I help?"),
                ("user", "I want to cancel order #W2378156."),
                ("assistant", "Sure. Could you give me your zip code?"),
                ("user", "It is 19122."),
            ]
        )
    )
    rules_by_field = disclosure_by_field(rules)
    assert isinstance(rules_by_field["zip"], DisclosureRule)
    assert rules_by_field["zip"].on_request is True
    assert rules_by_field["order_id"].on_request is False
    assert rules_by_field["order_id"].condition == "volunteered"


def test_refusal_is_recorded_against_the_field_the_agent_asked_for():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Could you please provide your email address?"),
                ("user", "I am sorry, but I do not remember the email address I used."),
                ("assistant", "No problem, what is your name and zip code?"),
                ("user", "Mei Kovacs, 28236."),
            ]
        )
    )
    assert rules.refusals == ["email"]
    # The bare answer to a name and zip ask holds both, and neither is a refusal or a goal.
    assert facts_by_field(rules)["zip"] == "28236"
    assert facts_by_field(rules)["name"] == "Mei Kovacs"
    assert rules.incomplete_reasons == []


def test_walk_away_lines_are_kept():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "I cannot do that."),
                ("user", "Never mind, I will do it myself. ###STOP###"),
            ]
        )
    )
    assert rules.walk_away == ["Never mind, I will do it myself. ###STOP###"]


def test_the_rules_name_the_one_trace_they_came_from():
    """Style is sampled from this trace and every fact points back into it (D44, D66)."""
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Hi! How can I help?"),
                ("user", "My name is Mei Kovacs and I want to return a lamp."),
                ("assistant", "What is your zip code?"),
                ("user", "28236."),
            ],
            trace_id="sim-7",
        )
    )
    assert rules.style_sample == ["sim-7"]
    assert [fact.span.msg_index for fact in rules.facts if fact.field == "zip"] == [3]
    assert {fact.span.file_hash for fact in rules.facts} == {"rawhash"}
    assert all(fact.span is not None for fact in rules.facts)


def test_asked_but_never_answered_field_makes_the_rules_incomplete():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "What is your zip code?"),
                ("user", "Let me think about it."),
            ]
        )
    )
    assert any("zip" in reason for reason in rules.incomplete_reasons)


def test_answered_and_refused_fields_do_not_make_the_rules_incomplete():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "What is your zip code?"),
                ("user", "19122."),
                ("assistant", "And your email address?"),
                ("user", "I would rather not share that."),
            ]
        )
    )
    assert rules.incomplete_reasons == []


def test_a_trace_with_no_user_turns_is_incomplete():
    rules = derive_user_rules(make_trace([("assistant", "Hello?")]))
    assert rules.incomplete_reasons == ["no user turns in the trace"]


def test_rules_round_trip_through_json():
    """Rules with every branch in them survive a round trip, values and order included."""
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Hi! How can I help?"),
                ("user", "I want to exchange a desk lamp."),
                ("assistant", "Your email please?"),
                ("user", "I would rather not share that."),
                ("assistant", "Your first name, last name and zip code then?"),
                ("user", "My name is Mei Kovacs and my zip code is 28236."),
                ("assistant", "Shall I proceed with the exchange?"),
                ("user", "Yes, please go ahead."),
            ]
        )
    )
    again = UserRules.model_validate(as_dict(rules))
    assert again == rules
    assert facts_by_field(again) == facts_by_field(rules)
    assert facts_by_field(again)[CONFIRMATION] == "Yes, please go ahead."
    assert again.refusals == ["email"]
    assert again.facts[0].span == rules.facts[0].span


def rules_from_fixture(tau2_small, index: int) -> UserRules:
    sim = tau2_small["simulations"][index]
    pairs = [
        (message["role"], message.get("content"))
        for message in sim["messages"]
        if message["role"] in ("user", "assistant")
    ]
    return derive_user_rules(make_trace(pairs, trace_id=sim["id"]))


def test_derive_over_the_tau2_fixture_finds_the_recorded_zip_and_name(tau2_small):
    sim = tau2_small["simulations"][0]
    rules = rules_from_fixture(tau2_small, 0)
    assert facts_by_field(rules)["zip"] == "28236"
    # The same user turn states the name, which the rules have to hold as well (D44 facts exact).
    assert facts_by_field(rules)["name"] == "Mei Kovacs"
    assert "email" in rules.refusals
    assert rules.style_sample == [sim["id"]]
    assert facts_by_field(rules)[GOAL].startswith("Hi!")
    assert "water bottle" in facts_by_field(rules)[GOAL]
    assert "I confirm" in " ".join(field_values(rules, CONFIRMATION))
    assert facts_by_field(rules)[CLOSING].startswith("No, that")


def test_the_fixture_traces_are_complete(tau2_small):
    """Every field the agent really asked for is answered or refused in all three traces."""
    for index in range(3):
        rules = rules_from_fixture(tau2_small, index)
        assert rules.incomplete_reasons == [], index
        assert rules.walk_away == [], index


def test_an_order_id_stated_without_a_hash_is_recorded(tau2_small):
    rules = rules_from_fixture(tau2_small, 2)
    assert facts_by_field(rules)["order_id"] == "#W2378156"


def test_a_farewell_is_not_a_refusal_and_a_stop_marker_is_not_a_walk_away():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "Done. Anything else today?"),
                ("user", "No, I do not have any other questions. Thanks, goodbye! ###STOP###"),
            ]
        )
    )
    assert rules.refusals == []
    assert rules.walk_away == []
    assert facts_by_field(rules)[CLOSING].startswith("No, I do not have")


def test_a_fact_given_after_a_refusal_is_no_longer_refused():
    rules = derive_user_rules(
        make_trace(
            [
                ("assistant", "What is your email address?"),
                ("user", "I do not remember the email address I used."),
                ("assistant", "Found you. Is your email mei.kovacs8232@example.com?"),
                ("user", "Oh yes, that is right, mei.kovacs8232@example.com."),
            ]
        )
    )
    assert facts_by_field(rules)["email"] == "mei.kovacs8232@example.com"
    assert rules.refusals == []
    user = SimulatedUser(rules)
    assert "mei.kovacs8232@example.com" in user.reply(ask("What is your email address?"))


def test_a_statement_about_the_order_is_not_a_question():
    """Mentions are not asks: no field is asked for by any of these agent turns (D46 coverage)."""
    for said in (
        "I see the order with the headphones. Shall I start the return?",
        "I found your order. How would you like to proceed?",
        "Thank you. Now let me address your questions one by one.",
        "You will receive an email with the return instructions shortly.",
        "I have located your account using your name and zip code.",
    ):
        assert asked_fields(said) == [], said


def test_a_mention_does_not_make_the_simulated_user_say_a_fact_is_unavailable():
    user = SimulatedUser(UserRules(), starting_state_reader=dict_reader({}))
    text = user.reply(ask("I see the order with the headphones. Shall I start the return?"))
    assert "do not have my phone" not in text
    assert user.events[-1].payload["unavailable_fields"] == [CONFIRMATION]


def test_a_five_digit_number_is_only_a_zip_when_the_turn_says_so():
    assert extracted_values("I want to return item 12345 please.") == []
    assert extracted_values("My zip code is 19122.") == [("zip", "19122")]
    assert extracted_values("19122.", asked=["zip"]) == [("zip", "19122")]


def test_the_name_cue_is_read_whatever_its_case():
    assert ("name", "Mei Kovacs") in extracted_values("my name is Mei Kovacs and my zip is 28236.")
    assert ("name", "Mei Kovacs") in extracted_values("My name is Mei Kovacs and my zip is 28236.")


def test_a_product_name_question_is_not_an_ask_for_the_user_name():
    assert asked_fields("What is the name of the product you want to return?") == []
    user = SimulatedUser(UserRules(facts=[UserFact(field="name", value="Mei Kovacs")]))
    assert "Mei Kovacs" not in user.reply(ask("What is the name of the product you want to return?"))


# --- SimulatedUser ---


@pytest.fixture
def rules_with_zip() -> UserRules:
    return UserRules(
        facts=[UserFact(field="zip", value="19122")],
        disclosure=[DisclosureRule(field="zip", on_request=True)],
        style_sample=["t1"],
    )


def ask(text: str) -> list[dict]:
    return [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": text}]


def test_the_model_only_fills_wording_around_the_recorded_fact(rules_with_zip, make_test_model):
    model = make_test_model(["Sure thing, my zip code is 19122!"])
    user = SimulatedUser(rules_with_zip, model=model)
    assert user.reply(ask("What is your zip code?")) == "Sure thing, my zip code is 19122!"
    assert len(model.calls) == 1


def test_wording_that_drops_the_fact_is_replaced_by_the_plain_sentence(rules_with_zip, make_test_model):
    model = make_test_model(["My zip code is 90210."])
    user = SimulatedUser(rules_with_zip, model=model)
    text = user.reply(ask("What is your zip code?"))
    assert "19122" in text
    assert "90210" not in text


def test_a_simulated_user_works_without_a_model(rules_with_zip):
    user = SimulatedUser(rules_with_zip)
    assert "19122" in user.reply(ask("Could you confirm your zip code?"))


def test_an_on_request_fact_is_not_volunteered(rules_with_zip):
    user = SimulatedUser(rules_with_zip)
    assert "19122" not in user.reply(ask("Thanks, one moment while I look that up."))


def test_a_fact_the_rules_lack_is_answered_from_the_starting_state():
    rules = UserRules(facts=[UserFact(field="zip", value="19122")])
    user = SimulatedUser(rules, starting_state_reader=dict_reader({"email": "mei@example.com"}))
    text = user.reply(ask("What email address is on the account?"))
    assert "mei@example.com" in text
    event = user.events[-1]
    assert event.payload["sources"] == {"email": "world"}
    assert event.assisted is False


def test_a_world_answer_from_a_synthetic_row_marks_the_turn_assisted():
    user = SimulatedUser(
        UserRules(),
        starting_state_reader=dict_reader({"email": FactLookup("mei@example.com", synthetic=True)}),
    )
    user.reply(ask("What is your email address?"))
    assert user.events[-1].assisted is True


def test_a_fact_neither_the_rules_nor_the_world_hold_is_unavailable():
    user = SimulatedUser(UserRules(), starting_state_reader=dict_reader({}))
    text = user.reply(ask("Could you give me the last four digits of your card?"))
    assert "do not have" in text
    event = user.events[-1]
    assert event.type == "user_turn"
    assert "fact_unavailable" in event.payload["tags"]
    assert event.payload["unavailable_fields"] == ["card_last4"]


def test_the_model_is_not_asked_to_word_an_unavailable_fact(make_test_model):
    model = make_test_model(["It is 4321."])
    user = SimulatedUser(UserRules(), model=model, starting_state_reader=dict_reader({}))
    text = user.reply(ask("What are the last four digits of your card?"))
    assert "4321" not in text
    assert model.calls == []


def test_a_refused_fact_stays_refused_even_when_the_world_has_it():
    rules = UserRules(refusals=["email"])
    user = SimulatedUser(rules, starting_state_reader=dict_reader({"email": "mei@example.com"}))
    text = user.reply(ask("What is your email address?"))
    assert "mei@example.com" not in text
    assert user.events[-1].payload["sources"] == {"email": "refused"}


def test_events_are_one_user_turn_per_reply_with_rising_indexes(rules_with_zip):
    user = SimulatedUser(rules_with_zip)
    user.reply(ask("What is your zip code?"))
    user.reply(ask("Thanks, anything else today?"))
    assert [event.idx for event in user.events] == [0, 1]
    assert {event.type for event in user.events} == {"user_turn"}
    assert "19122" in user.events[0].payload["text"]


def test_a_volunteered_fact_is_offered_once_without_being_asked():
    rules = UserRules(
        facts=[UserFact(field="order_id", value="#W2378156")],
        disclosure=[DisclosureRule(field="order_id", on_request=False, condition="volunteered")],
    )
    user = SimulatedUser(rules)
    first = user.reply(ask("Hi! How can I help you today?"))
    second = user.reply(ask("Thanks, one moment."))
    assert "#W2378156" in first
    assert "#W2378156" not in second
    assert user.events[0].payload["sources"] == {"order_id": "volunteered"}


def test_the_world_is_read_as_this_user_and_not_as_the_first_row(tau2_retail_dir):
    """D77: the Starting state is read as the user the rules describe, never as any other row."""
    from harness.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    rules = UserRules(facts=[
        UserFact(field="name", value="Mei Kovacs"),
        UserFact(field="zip", value="28236"),
    ])
    user = SimulatedUser(rules, starting_state_reader=StateView(shared=db))
    text = user.reply(ask("Could you confirm the email address on the account?"))
    assert "mei.kovacs8232@example.com" in text
    assert db["users"]["noah_brown_6181"]["email"] not in text
    assert user.events[-1].payload["sources"] == {"email": "world"}


def test_the_caller_can_name_the_user_whose_row_is_read(tau2_retail_dir):
    from harness.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    user = SimulatedUser(UserRules(), starting_state_reader=StateView(shared=db),
                         identity={"user_id": "mei_kovacs_8020"})
    assert "mei.kovacs8232@example.com" in user.reply(ask("What is your email address?"))


def test_a_user_the_world_does_not_hold_gets_no_row_at_all(tau2_retail_dir):
    from harness.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    rules = UserRules(facts=[
        UserFact(field="name", value="Nobody Here"),
        UserFact(field="zip", value="00000"),
    ])
    user = SimulatedUser(rules, starting_state_reader=StateView(shared=db))
    text = user.reply(ask("What is your email address?"))
    assert "@" not in text
    assert user.events[-1].payload["unavailable_fields"] == ["email"]
    assert "fact_unavailable" in user.events[-1].payload["tags"]


def test_the_overlay_row_wins_over_the_shared_world(tau2_retail_dir):
    """D74: the Task's own rows are read before the shared world."""
    from harness.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    pinned = dict(db["users"]["mei_kovacs_8020"], email="mei.pinned@example.com")
    view = StateView(shared=db, overlay={"users": {"mei_kovacs_8020": pinned}})
    rules = UserRules(facts=[
        UserFact(field="name", value="Mei Kovacs"),
        UserFact(field="zip", value="28236"),
    ])
    user = SimulatedUser(rules, starting_state_reader=view)
    assert "mei.pinned@example.com" in user.reply(ask("What is your email address?"))


def test_wording_that_adds_a_fact_the_rules_never_gave_is_thrown_away(rules_with_zip, make_test_model):
    model = make_test_model(["Sure, my zip is 19122 and my email is invented@example.com."])
    user = SimulatedUser(rules_with_zip, model=model)
    text = user.reply(ask("What is your zip code?"))
    assert "invented@example.com" not in text
    assert text == "My zip is 19122."


def test_the_model_is_not_asked_to_word_a_reply_with_nothing_in_it(make_test_model):
    model = make_test_model(["My order is #W9999999 and my card ends in 1234."])
    user = SimulatedUser(UserRules(), model=model)
    text = user.reply(ask("Thanks, one moment."))
    assert text == "Okay, thank you."
    assert model.calls == []


def test_the_model_cannot_word_its_way_past_a_refusal(make_test_model):
    rules = UserRules(
        facts=[UserFact(field="zip", value="19122")],
        refusals=["email"],
    )
    model = make_test_model(["My zip is 19122 and my email is mei.leaked@example.com."])
    user = SimulatedUser(rules, model=model)
    text = user.reply(ask("What is your zip code and your email address?"))
    assert "mei.leaked@example.com" not in text
    assert "rather not share my email" in text


def test_the_goal_the_recorded_user_opened_with_is_spoken_first(tau2_small):
    rules = rules_from_fixture(tau2_small, 0)
    user = SimulatedUser(rules)
    first = user.reply([{"role": "assistant", "content": "Hi! How can I help you today?"}])
    assert "water bottle" in first and "desk lamp" in first
    assert user.events[0].payload["sources"][GOAL] == "rules"
    second = user.reply(ask("One moment please."))
    assert "water bottle" not in second


def test_a_confirmation_question_is_answered_with_the_recorded_confirmation(tau2_small):
    rules = rules_from_fixture(tau2_small, 0)
    user = SimulatedUser(rules)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Do you confirm the exchange of the desk lamp? Please reply yes to proceed."))
    assert text in field_values(rules, CONFIRMATION)
    assert "proceed" in text.lower()
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "rules"}


def test_a_confirmation_the_trace_never_gave_is_unavailable_not_invented():
    user = SimulatedUser(UserRules(facts=[UserFact(field="zip", value="19122")]))
    text = user.reply(ask("Do you confirm the exchange? Please reply yes to proceed."))
    assert "yes" not in text.lower()
    assert user.events[-1].payload["unavailable_fields"] == [CONFIRMATION]


def test_an_open_question_is_answered_with_the_choice_the_recorded_user_stated(tau2_small):
    rules = rules_from_fixture(tau2_small, 2)
    user = SimulatedUser(rules)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Which items would you like to exchange, and for which options?"))
    assert text == facts_by_field(rules)[CHOICE]
    assert "thermostat" in text


def test_the_simulated_user_closes_the_run_with_the_recorded_closing_line(tau2_small):
    rules = rules_from_fixture(tau2_small, 1)
    user = SimulatedUser(rules)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Your return is processed. Is there anything else I can help you with?"))
    assert text == facts_by_field(rules)[CLOSING]
    assert user.done is True


def test_a_run_against_the_simulated_user_stops_instead_of_burning_its_turns(tau2_small, make_test_model):
    from harness.runner import loop

    user = SimulatedUser(rules_from_fixture(tau2_small, 1))
    model = make_test_model([{"content": "Is there anything else I can help you with today?"}], loop=True)
    state = loop.new_run_state("ends", user=user, max_turns=8)
    loop.run(state, model)
    assert state.run.termination_reason == "user_stop"
    assert state.turn < 8


def test_the_reader_is_only_asked_for_fields_the_rules_lack(rules_with_zip):
    asked = []

    class Reader:
        def get(self, field):
            asked.append(field)
            return None

    user = SimulatedUser(rules_with_zip, starting_state_reader=Reader())
    user.reply(ask("What is your zip code?"))
    assert asked == []
    user.reply(ask("And your email address?"))
    assert asked == ["email"]
