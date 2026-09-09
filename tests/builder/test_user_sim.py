"""Tests for builder/user_sim.py: rules derived from a trace, and a Simulated user that never invents."""

from __future__ import annotations

import json

import pytest

from conftest import PTR
from kullback.builder import intent as intent_mod
from kullback.builder.user_sim import (
    ASKABLE,
    CHOICE,
    CLOSING,
    CONFIRMATION,
    GAVE_UP,
    GENERIC_CONFIRM,
    GOAL,
    GOAL_SATISFIED,
    HANDED_OFF,
    RECORD,
    SCENARIO_EXHAUSTED,
    STRIPPED_TAG,
    FactLookup,
    SimulatedUser,
    asked_fields,
    derive_user_rules,
    dict_reader,
    end_of_run,
    ends_by_kind,
    extracted_values,
    fact_class,
    goal_write_set,
)
from kullback.builder.vocabulary import GENERIC_FIELDS, FieldSpec, Vocabulary
from kullback.runner.records import (
    DisclosureRule,
    RawPtr,
    ToolCall,
    ToolCallError,
    Trace,
    Turn,
    UserFact,
    UserRules,
    as_dict,
)

# What the retail build derives (D115): the generic core plus the fields retail's users state.
RETAIL = Vocabulary(domain="retail", fields=[f.model_copy(deep=True) for f in GENERIC_FIELDS] + [
    FieldSpec(field="order_id", kind="reference", pattern=r"(?:\#)?\bW\d{7}\b", prefix="#",
              cues=[r"\border (?:id|number|no\.?|#|reference|code)\b", r"\bwhich order\b"], aliases=["order id"]),
    FieldSpec(field="card_last4", pattern=r"(?i:(?:last four|last 4|ending in)\D{0,12})(\d{4})\b",
              cues=[r"\blast four\b", r"\blast 4\b", r"\bcard ending\b"], aliases=["last four"]),
    FieldSpec(field="payment_method", pattern=r"\b(?i:gift card|credit card|debit card|paypal|mastercard|visa|amex|american express)\b",
              cues=[r"\bpayment method\b", r"\brefund method\b"], aliases=["payment method"]),
])


def make_trace(pairs, trace_id: str = "t1", calls=()) -> Trace:
    """pairs is [(role, content), ...] in transcript order; calls is [(turn index, tool name), ...]."""
    turns = [
        Turn(
            idx=i,
            role=role,
            content=content,
            raw_ptr=RawPtr(file_hash="rawhash", sim_index=0, msg_index=i),
        )
        for i, (role, content) in enumerate(pairs)
    ]
    tool_calls = []
    for i, (turn_index, name) in enumerate(calls):
        call_id = f"c{i}"
        turns[turn_index].tool_call_ids.append(call_id)
        tool_calls.append(ToolCall(
            id=call_id, name=name,
            raw_ptr=RawPtr(file_hash="rawhash", sim_index=0, msg_index=turn_index)))
    return Trace(
        trace_id=trace_id,
        raw_hash="rawhash",
        ingest_version="0",
        source="tau2",
        turns=turns,
        tool_calls=tool_calls,
        raw_ptr=RawPtr(file_hash="rawhash", sim_index=0),
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
        ),
        vocab=RETAIL,
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
        ),
        vocab=RETAIL,
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
    return derive_user_rules(make_trace(pairs, trace_id=sim["id"]), vocab=RETAIL)


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


def test_a_request_that_ends_in_a_colon_asks_for_the_fields_it_lists():
    """Build 8: 'please provide me with: - your first name - your last name - your zip code' read as no ask
    at all, and the Simulated user answered 'I do not have my name' on 63 re-rolls."""
    listed = "Could you please provide me with:\n- Your first name\n- Your last name\n- Your ZIP code"
    assert asked_fields(listed) == ["name", "zip"]
    rules = derive_user_rules(make_trace([("assistant", listed), ("user", "Olivia Lopez, 76171.")]))
    assert facts_by_field(rules)["name"] == "Olivia Lopez"
    assert facts_by_field(rules)["zip"] == "76171"
    assert field_values(rules, CHOICE) == []


def test_a_list_after_a_statement_that_asks_nothing_is_not_an_ask():
    assert asked_fields("Your order holds:\n- a lamp\n- a name tag with your name on it") == []


def test_a_yes_that_also_names_the_order_is_still_the_recorded_confirmation():
    """Build 8: the recorded yes carried the order id, so it was filed as the id alone and the re-rolls
    answered 'I do not have an answer for that' when asked to confirm."""
    yes = "Yes, please cancel order #W5995614 and refund it to the same card."
    rules = derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I want to cancel an order."),
        ("assistant", "I see two pending orders. Shall I cancel #W5995614? Please confirm."),
        ("user", yes),
    ]), RETAIL)
    assert field_values(rules, CONFIRMATION) == [yes]
    assert facts_by_field(rules)["order_id"] == "#W5995614"


def test_a_yes_that_answers_a_field_ask_is_the_field_and_not_a_confirmation():
    rules = derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I need help with an order."),
        ("assistant", "Could you give me your zip code?"),
        ("user", "Yes, my zip is 19122."),
    ]))
    assert facts_by_field(rules)["zip"] == "19122"
    assert field_values(rules, CONFIRMATION) == []


def test_an_asked_value_is_not_read_where_another_field_already_read_it():
    """Build 8: the Simulated user answered an address question with the order id the same turn named."""
    vocab = Vocabulary(domain="retail", fields=[f.model_copy(deep=True) for f in RETAIL.fields] + [
        FieldSpec(field="address1", kind="value", asked_only=r"\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9 ]+\b",
                  cues=[r"\baddress\b"]),
    ])
    values = extracted_values("For order #W5056519, please update it to 380 Maple Drive.",
                              asked=["address1"], vocab=vocab)
    assert ("order_id", "#W5056519") in values
    assert ("address1", "380 Maple Drive") in values


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
    assert user.events[-1].payload["unavailable_fields"] == []
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "generic_confirm"}


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


def test_a_fact_only_the_world_holds_is_pointed_at_and_never_spoken():
    """D210: the Starting state is still read, and what it holds is a record fact, so the user says
    the Candidate can look it up rather than handing over a value no user of the Task ever said."""
    rules = UserRules(facts=[UserFact(field="zip", value="19122")])
    user = SimulatedUser(rules, starting_state_reader=dict_reader({"email": "mei@example.com"}))
    text = user.reply(ask("What email address is on the account?"))
    assert "mei@example.com" not in text
    assert "look it up" in text
    event = user.events[-1]
    assert event.payload["sources"] == {"email": "record"}
    assert event.payload["record_fields"] == ["email"]
    assert "fact_in_record" in event.payload["tags"]
    assert event.assisted is False


def test_a_world_answer_from_a_synthetic_row_marks_the_turn_assisted():
    user = SimulatedUser(
        UserRules(),
        starting_state_reader=dict_reader({"email": FactLookup("mei@example.com", synthetic=True)}),
    )
    user.reply(ask("What is your email address?"))
    assert user.events[-1].assisted is True


def test_a_fact_neither_the_rules_nor_the_world_hold_is_unavailable():
    user = SimulatedUser(UserRules(), starting_state_reader=dict_reader({}), vocab=RETAIL)
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
    """D77: the Starting state is read as the user the rules describe, never as any other row.

    D210: the row it finds is a record fact and no row's value is spoken, so what the lookup decides
    is whether the user can point the Candidate at it at all.
    """
    from kullback.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    rules = UserRules(facts=[
        UserFact(field="name", value="Mei Kovacs"),
        UserFact(field="zip", value="28236"),
    ])
    user = SimulatedUser(rules, starting_state_reader=StateView(shared=db))
    text = user.reply(ask("Could you confirm the email address on the account?"))
    assert "mei.kovacs8232@example.com" not in text
    assert db["users"]["noah_brown_6181"]["email"] not in text
    assert user.events[-1].payload["sources"] == {"email": "record"}


def test_the_caller_can_name_the_user_whose_row_is_read(tau2_retail_dir):
    from kullback.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    user = SimulatedUser(UserRules(), starting_state_reader=StateView(shared=db),
                         identity={"user_id": "mei_kovacs_8020"})
    user.reply(ask("What is your email address?"))
    assert user.events[-1].payload["record_fields"] == ["email"]
    stranger = SimulatedUser(UserRules(), starting_state_reader=StateView(shared=db),
                             identity={"user_id": "nobody_at_all_0000"})
    stranger.reply(ask("What is your email address?"))
    assert stranger.events[-1].payload["unavailable_fields"] == ["email"]


def test_a_user_the_world_does_not_hold_gets_no_row_at_all(tau2_retail_dir):
    from kullback.runner.route import StateView

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
    """D74: the Task's own rows are read before the shared world.

    The overlay row carries a field the shared row does not, so which of the two the user read is
    what decides whether the field is a record fact it can point at or one nobody holds (D210).
    """
    from kullback.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    shared = json.loads(json.dumps(db))
    shared["users"]["mei_kovacs_8020"].pop("email", None)
    pinned = dict(db["users"]["mei_kovacs_8020"], email="mei.pinned@example.com")
    rules = UserRules(facts=[
        UserFact(field="name", value="Mei Kovacs"),
        UserFact(field="zip", value="28236"),
    ])
    user = SimulatedUser(rules, starting_state_reader=StateView(
        shared=shared, overlay={"users": {"mei_kovacs_8020": pinned}}))
    user.reply(ask("What is your email address?"))
    assert user.events[-1].payload["record_fields"] == ["email"]
    without = SimulatedUser(rules, starting_state_reader=StateView(shared=shared))
    without.reply(ask("What is your email address?"))
    assert without.events[-1].payload["unavailable_fields"] == ["email"]


def test_a_row_with_two_payment_methods_reports_the_field_unavailable_not_a_pick(tau2_retail_dir):
    """Row 11: noah_brown_6181 carries a paypal and a credit card. Answering from whichever one a
    dict happened to iterate to first means the answer depends on key order; the field must be
    reported unavailable instead, in either order."""
    from kullback.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    assert len(db["users"]["noah_brown_6181"]["payment_methods"]) == 2

    def reply_for(methods: dict) -> str:
        row = dict(db["users"]["noah_brown_6181"], payment_methods=methods)
        view = StateView(shared=db, overlay={"users": {"noah_brown_6181": row}})
        user = SimulatedUser(UserRules(), starting_state_reader=view, identity={"user_id": "noah_brown_6181"},
                             vocab=RETAIL)
        return user.reply(ask("What is your payment method?")), user

    forward, user_forward = reply_for(dict(db["users"]["noah_brown_6181"]["payment_methods"]))
    backward, user_backward = reply_for(
        dict(reversed(list(db["users"]["noah_brown_6181"]["payment_methods"].items())))
    )
    for text, user in ((forward, user_forward), (backward, user_backward)):
        assert "paypal" not in text.lower() and "credit_card" not in text.lower()
        assert "fact_unavailable" in user.events[-1].payload["tags"]
        assert user.events[-1].payload["unavailable_fields"] == ["payment_method"]


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
    # A turn that asks for nothing gets the goal a second time, said as a restatement and marked as
    # one, and never a third: the turn after that closes the Run.
    second = user.reply(ask("One moment please."))
    assert user.events[-1].payload["sources"] == {GOAL: "goal_restated"}
    third = user.reply(ask("Still one moment please."))
    assert "water bottle" in second and "desk lamp" in second
    assert "water bottle" not in third


def test_a_confirmation_question_is_answered_with_the_recorded_confirmation(tau2_small):
    rules = rules_from_fixture(tau2_small, 0)
    user = SimulatedUser(rules)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Do you confirm the exchange of the desk lamp? Please reply yes to proceed."))
    assert text in field_values(rules, CONFIRMATION)
    assert "proceed" in text.lower()
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "rules"}


def test_a_confirmation_the_trace_never_gave_gets_the_representative_yes():
    """D44: the recorded user was never asked; a user who asked for the action says yes to doing it, and
    the source names the yes as generic so a report can tell it from a recorded one."""
    user = SimulatedUser(UserRules(facts=[UserFact(field="zip", value="19122")]))
    text = user.reply(ask("Do you confirm the exchange? Please reply yes to proceed."))
    assert text == "Yes, please go ahead."
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "generic_confirm"}
    assert user.events[-1].payload["unavailable_fields"] == []


def test_a_recorded_no_keeps_the_simulated_user_from_saying_yes():
    user = SimulatedUser(UserRules(facts=[UserFact(field=CHOICE, value="No, do not cancel it yet.")]))
    text = user.reply(ask("Shall I cancel the order? Please confirm."))
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
    # Nothing in this transcript wrote, so the first "anything else" gets the goal again; the
    # second one is the close.
    user.reply(ask("Your return is processed. Is there anything else I can help you with?"))
    text = user.reply(ask("Is there anything else I can help you with?"))
    assert text == facts_by_field(rules)[CLOSING]
    assert user.done is True


def test_a_run_against_the_simulated_user_stops_instead_of_burning_its_turns(tau2_small, make_test_model):
    from kullback.runner import loop

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


# --- D136: the Simulated user confirms, and gives the fact the question names ---


def test_a_confirmation_question_gets_the_sentence_the_recorded_user_said_word_for_word(tau2_small):
    """D44: a confirmation is a whole sentence of the recording, repeated exactly as it was given."""
    rules = rules_from_fixture(tau2_small, 2)
    said = [message.get("content") for message in tau2_small["simulations"][2]["messages"]
            if message["role"] == "user"]
    user = SimulatedUser(rules, vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Is this correct? Please confirm and I will process the exchange."))
    assert text in said
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "rules"}


def test_an_answer_to_a_confirmation_question_is_the_confirmation_even_without_the_word_yes():
    rules = derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I need to cancel my order."),
        ("assistant", "I can cancel it for you. Shall I proceed?"),
        ("user", "Please do that now, the sooner the better."),
    ]), RETAIL)
    assert field_values(rules, CONFIRMATION) == ["Please do that now, the sooner the better."]
    user = SimulatedUser(rules, vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    assert user.reply(ask("Do you confirm?")) == "Please do that now, the sooner the better."


def test_a_closing_line_that_opens_with_no_does_not_refuse_a_later_confirmation():
    """Build 8: 'No, that is it for today' was filed as a stated choice, and every later
    'do you confirm' was answered 'I do not have an answer for that' (58 re-rolls)."""
    rules = derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I want to cancel my order."),
        ("assistant", "Anything else today?"),
        ("user", "No, that is it for today."),
    ]), RETAIL)
    assert field_values(rules, CHOICE) == []
    user = SimulatedUser(rules, vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Shall I cancel order #W1234567? Please confirm."))
    assert text == GENERIC_CONFIRM
    assert user.events[-1].payload["unavailable_fields"] == []


def test_the_write_the_recording_made_after_the_confirmation_question_is_the_users_agreement(
        tau2_small_path, workdir):
    """The recorded Run went on to the write, so the recorded user agreed; a read is not agreement."""
    from kullback.builder import ingest

    raw = ingest.store_raw(tau2_small_path, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    agreed = derive_user_rules(trace, RETAIL, writes={"exchange_delivered_order_items"})
    assert agreed.confirmed_by_write is True
    read_only = derive_user_rules(trace, RETAIL, writes={"get_order_details"})
    assert read_only.confirmed_by_write is False


def test_a_no_the_user_said_about_something_else_does_not_outweigh_the_recorded_write():
    trace = make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I want to cancel my order."),
        ("assistant", "Which order id should I cancel? Please confirm the order id."),
        ("user", "Order #W1234567."),
        ("assistant", None),
        ("assistant", "That is cancelled. Would you like a replacement ordered as well?"),
        ("user", "No, do not order a replacement."),
    ], calls=[(4, "cancel_order")])
    rules = derive_user_rules(trace, RETAIL, writes={"cancel_order"})
    assert rules.confirmed_by_write is True
    user = SimulatedUser(rules, vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    assert user.reply(ask("Shall I cancel it? Please confirm.")) == GENERIC_CONFIRM


@pytest.mark.parametrize("question, expected", [
    ("What is the name on the account?", "Yusuf Rossi"),
    ("Just so I have it right, what name is on file?", "Yusuf Rossi"),
    ("May I take a name and a zip code, please?", "Yusuf Rossi"),
    ("May I take a name and a zip code, please?", "19122"),
    ("Could you give me the order id this is about?", "#W2378156"),
])
def test_a_fact_the_recording_holds_is_given_however_the_agent_worded_the_question(
        tau2_small, question, expected):
    """Build 8: the agent asked for the name in wording the cues did not carry and the Simulated
    user answered 'I do not have my name', although the recording gave it (63 re-rolls)."""
    user = SimulatedUser(rules_from_fixture(tau2_small, 2), vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask(question))
    assert expected in text
    assert user.events[-1].payload["unavailable_fields"] == []


def test_a_question_about_another_things_name_is_still_not_a_question_for_the_users_name(tau2_small):
    user = SimulatedUser(rules_from_fixture(tau2_small, 2), vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    assert "Yusuf Rossi" not in user.reply(ask("What is the name of the product you want to exchange?"))


def two_zip_rules() -> UserRules:
    """A recorded user that states the zip on the account and then the zip it is moving to."""
    return derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I want to change the shipping address on my order."),
        ("assistant", "Sure. What is the zip code on the account?"),
        ("user", "My zip code is 19122."),
        ("assistant", "And where should it go instead?"),
        ("user", "Please send it to my new address, zip code 78701."),
    ]), RETAIL)


@pytest.mark.parametrize("question, expected", [
    ("Could you confirm the zip code on file for the account?", "19122"),
    ("What is the zip code of the new address?", "78701"),
])
def test_the_value_the_question_names_is_the_one_given_when_the_recording_holds_two(question, expected):
    """Build 8: the Simulated user gave the zip of the address it was moving to when the agent
    asked for the one on file (27 re-rolls)."""
    rules = two_zip_rules()
    assert [fact.value for fact in rules.facts if fact.field == "zip"] == ["19122", "78701"]
    user = SimulatedUser(rules, vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    assert expected in user.reply(ask(question))


def test_the_recorded_turn_order_breaks_a_tie_between_two_values_of_one_field():
    user = SimulatedUser(two_zip_rules(), vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    assert "19122" in user.reply(ask("What zip code should I use?"))


def test_the_value_on_the_account_is_a_record_fact_when_the_recording_holds_only_the_new_one(
        tau2_retail_dir):
    """D210: a user that stated only the value it is moving to never said the one on file, so the
    stored value is the world's and the user points at it rather than reading it out; and the value
    it did state is not offered in the stored one's place either."""
    from kullback.runner.route import StateView

    db = json.loads((tau2_retail_dir / "db.json").read_text(encoding="utf-8"))
    rules = derive_user_rules(make_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I am moving. Please change my zip code to 78701."),
        ("assistant", "Of course. What is your name?"),
        ("user", "My name is Yusuf Rossi."),
    ]), RETAIL)
    user = SimulatedUser(rules, starting_state_reader=StateView(shared=db), vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("What is the zip code currently on file for the account?"))
    assert "19122" not in text and "78701" not in text
    assert user.events[-1].payload["sources"] == {"zip": "record"}


def test_each_user_turn_counts_the_asks_it_refused_and_the_running_total_for_the_run():
    """The number the report reads: how many asks the Simulated user did not answer, per Run."""
    rules = UserRules(facts=[UserFact(field="zip", value="19122")], refusals=["email"])
    user = SimulatedUser(rules, starting_state_reader=dict_reader({}), vocab=RETAIL)
    user.reply(ask("What is your email address and your zip code?"))
    assert (user.events[-1].payload["refused"], user.events[-1].payload["refused_so_far"]) == (1, 1)
    user.reply(ask("What are the last four digits of your card?"))
    assert (user.events[-1].payload["refused"], user.events[-1].payload["refused_so_far"]) == (1, 2)
    user.reply(ask("And your zip code again, please?"))
    assert (user.events[-1].payload["refused"], user.events[-1].payload["refused_so_far"]) == (0, 2)


def test_a_summary_that_names_the_order_and_asks_to_confirm_is_confirmed_and_not_read_back(tau2_small):
    """Build 8: the agent listed the action, named the order in it and asked 'do you confirm'; the
    Simulated user read the id out of the question and confirmed nothing (58 re-rolls)."""
    user = SimulatedUser(rules_from_fixture(tau2_small, 2), vocab=RETAIL)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("I will exchange the thermostat on order #W2378156. Do you confirm?"))
    assert "#W2378156" not in text
    assert user.events[-1].payload["sources"] == {CONFIRMATION: "rules"}


# --- facts mined from the arguments of the recorded calls ---

# A library help desk, so none of this leans on the shape of any one recorded domain.
LIBRARY = Vocabulary(domain="library", fields=[f.model_copy(deep=True) for f in GENERIC_FIELDS] + [
    FieldSpec(field="member_id", kind="identity", pattern=r"(?:\#)?\bM\d{6}\b", prefix="#",
              cues=[r"\bmember (?:id|number|card)\b"], aliases=["member id"]),
])


def made(turn: int, name: str, args: dict, result=None, error=None) -> dict:
    """One recorded call, tied to the turn that made it."""
    return {"turn": turn, "name": name, "args": args, "result": result, "error": error}


def library_trace(pairs, calls=()) -> Trace:
    """pairs is [(role, content), ...] in transcript order; calls are rows from `made`."""
    turns = [
        Turn(idx=i, role=role, content=content,
             raw_ptr=RawPtr(file_hash=PTR.file_hash, sim_index=0, msg_index=i))
        for i, (role, content) in enumerate(pairs)
    ]
    tool_calls = []
    for i, row in enumerate(calls):
        call_id = f"lc{i}"
        turns[row["turn"]].tool_call_ids.append(call_id)
        tool_calls.append(ToolCall(
            id=call_id, name=row["name"], args=row["args"], result=row["result"], error=row["error"],
            raw_ptr=RawPtr(file_hash=PTR.file_hash, sim_index=0, msg_index=row["turn"])))
    return Trace(trace_id="lib-1", raw_hash=PTR.file_hash, ingest_version="0", source="tau2",
                 turns=turns, tool_calls=tool_calls, raw_ptr=PTR)


BOLD_NAME = ("Sure, let's do that. I do not remember the email on the account, but my name is "
             "**Ada Whitfield** and my zip code is 30318 (Atlanta, GA).")


def bold_name_trace() -> Trace:
    """The recorded user gave a name the cues cannot read, and the agent looked it up with it."""
    return library_trace(
        [
            ("assistant", "Hi! How can I help you today?"),
            ("user", "I would like to renew a book."),
            ("assistant", "Of course. What is the email address on your library card?"),
            ("user", BOLD_NAME),
            ("assistant", None),
        ],
        calls=[made(4, "find_member_by_name_zip",
                    {"first_name": "Ada", "last_name": "Whitfield", "zip": "30318"},
                    result={"member_id": "M400318"})],
    )


def test_a_name_written_in_bold_is_mined_from_the_call_that_looked_the_user_up():
    """Build 12: 108 of 456 traces gave a name the cues missed, and 149 of 600 re-rolls never
    authenticated because the rules held no name fact."""
    trace = bold_name_trace()
    rules = derive_user_rules(trace, LIBRARY)
    assert facts_by_field(rules)["name"] == "Ada Whitfield"
    name = next(fact for fact in rules.facts if fact.field == "name")
    assert name.span == trace.turns[3].raw_ptr
    assert "Ada Whitfield" in name.context
    assert rules.incomplete_reasons == []


def test_the_first_and_last_name_of_one_call_are_one_name_fact():
    rules = derive_user_rules(bold_name_trace(), LIBRARY)
    assert field_values(rules, "name") == ["Ada Whitfield"]
    assert "first_name" not in facts_by_field(rules)
    assert "last_name" not in facts_by_field(rules)


def test_a_value_only_the_tool_returned_is_not_a_fact_the_user_gave():
    """D77: the member id came back from the lookup, so the recorded user never had it to give."""
    rules = derive_user_rules(
        library_trace(
            [
                ("assistant", "Hi! How can I help you today?"),
                ("user", "My name is **Ada Whitfield**, zip 30318."),
                ("assistant", None),
                ("assistant", None),
            ],
            calls=[
                made(2, "find_member_by_name_zip",
                     {"first_name": "Ada", "last_name": "Whitfield", "zip": "30318"},
                     result={"member_id": "M400318"}),
                made(3, "get_member", {"member_id": "M400318"}, result={"loans": []}),
            ],
        ),
        LIBRARY,
    )
    assert "member_id" not in facts_by_field(rules)
    assert "M400318" not in [str(fact.value) for fact in rules.facts]


def test_a_value_the_user_said_only_after_the_call_is_not_a_fact_of_that_call():
    """The user may be repeating what the agent just read out, so later turns do not count."""
    rules = derive_user_rules(
        library_trace(
            [
                ("assistant", "Hi! How can I help you today?"),
                ("user", "I would like to put a book on hold."),
                ("assistant", None),
                ("user", "Yes, Tidewater is the one I want."),
            ],
            calls=[made(2, "search_catalog", {"title": "Tidewater"}, result={"items": ["B7781"]})],
        ),
        LIBRARY,
    )
    assert "title" not in facts_by_field(rules)
    assert "Tidewater" not in [str(fact.value) for fact in rules.facts]


def test_an_argument_of_a_call_that_failed_is_not_a_fact():
    """The world refused the call, so its arguments are what the agent tried, not what it knew."""
    rules = derive_user_rules(
        library_trace(
            [
                ("assistant", "Hi! How can I help you today?"),
                ("user", "I am **Ada Whitfield** and I need to renew a book."),
                ("assistant", None),
            ],
            calls=[made(2, "find_member_by_name_zip",
                        {"first_name": "Ada", "last_name": "Whitfield"},
                        error=ToolCallError(**{"class": "not_found_entity"}, payload="No such member"))],
        ),
        LIBRARY,
    )
    assert "name" not in facts_by_field(rules)


def test_the_members_of_a_list_argument_the_user_named_are_mined_one_by_one():
    rules = derive_user_rules(
        library_trace(
            [
                ("assistant", "Hi! How can I help you today?"),
                ("user", "Please put B7781 and B4402 on hold for me."),
                ("assistant", None),
            ],
            calls=[made(2, "place_holds", {"item_ids": ["B7781", "B4402"]}, result={"held": 2})],
        ),
        LIBRARY,
    )
    assert field_values(rules, "item_ids") == ["B7781", "B4402"]


def test_a_mined_id_keeps_the_mark_the_argument_carries():
    """The user says the bare id and the tools store it with a '#'; the fact is the stored form."""
    rules = derive_user_rules(
        library_trace(
            [
                ("assistant", "Hi! How can I help you today?"),
                ("user", "Could you renew loan L2201 for me?"),
                ("assistant", None),
            ],
            calls=[made(2, "renew_loan", {"loan_id": "#L2201"}, result={"due": "2026-10-01"})],
        ),
        LIBRARY,
    )
    assert facts_by_field(rules)["loan_id"] == "#L2201"


def test_a_simulated_user_built_from_the_mined_rules_answers_the_name_and_zip_ask():
    """End to end: the ask 149 re-rolls of build 12 could not answer is answered."""
    user = SimulatedUser(derive_user_rules(bold_name_trace(), LIBRARY), vocab=LIBRARY)
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Could you give me your name and zip code so I can find the account?"))
    assert "Ada Whitfield" in text
    assert "30318" in text
    assert user.events[-1].payload["unavailable_fields"] == []


# --- the goal restated once before the Run is left ---


def renewal_rules() -> UserRules:
    """A recorded help desk call whose user opened with a request and closed with a line of its own."""
    return derive_user_rules(
        library_trace([
            ("assistant", "Hi! How can I help you today?"),
            ("user", "I would like to renew loan L2201 for another three weeks."),
            ("assistant", "That is renewed. Is there anything else I can help you with?"),
            ("user", "No, that is all for today. Thanks for your help."),
        ]),
        LIBRARY,
    )


def said(user: SimulatedUser) -> dict:
    return user.events[-1].payload["sources"]


def test_the_user_restates_its_goal_once_before_closing_on_a_silent_turn():
    """Build 12: of 381 re-rolls that made no write, 248 closed after one turn where the agent asked
    for nothing. A user whose request has not been acted on says it again before it leaves."""
    user = SimulatedUser(renewal_rules(), vocab=LIBRARY)
    opening = user.reply(ask("Hi! How can I help you today?"))
    assert "L2201" in opening and said(user)[GOAL] == "rules"
    again = user.reply(ask("One moment while I look into that."))
    assert again == opening
    assert said(user) == {GOAL: "goal_restated"}
    assert user.done is False


def test_the_user_restates_its_goal_when_asked_if_that_is_all_before_any_write():
    user = SimulatedUser(renewal_rules(), vocab=LIBRARY, write_tools={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Is there anything else I can help you with?"))
    assert "L2201" in text
    assert said(user) == {GOAL: "goal_restated"}
    assert user.done is False


def test_the_user_closes_at_once_when_asked_if_that_is_all_after_the_write():
    """The Run did what the user came for, so the same question is the end of the call, not a prod."""
    user = SimulatedUser(renewal_rules(), vocab=LIBRARY, write_tools={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply([
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "name": "renew_loan", "arguments": {"loan_id": "#L2201"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "renew_loan", "content": "{}"},
        {"role": "assistant", "content": "That is renewed. Is there anything else I can help you with?"},
    ])
    assert text == facts_by_field(renewal_rules())[CLOSING]
    assert said(user) == {CLOSING: "rules"}
    assert user.done is True


def test_the_second_silent_turn_closes():
    user = SimulatedUser(renewal_rules(), vocab=LIBRARY)
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("One moment while I look into that."))
    text = user.reply(ask("Still checking, sorry for the wait."))
    assert text == facts_by_field(renewal_rules())[CLOSING]
    assert said(user) == {CLOSING: "rules"}
    assert user.done is True


# --- D210: an end carries its reason, a fact carries its class, an answer goes through the strip ---


RENEWED = [
    {"role": "user", "content": "Hi."},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c1", "name": "renew_loan", "arguments": {"loan_id": "#L2201"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "renew_loan", "content": "{}"},
    {"role": "assistant", "content": "That is renewed. Is there anything else I can help you with?"},
]


def renewal_user(**kwargs) -> SimulatedUser:
    return SimulatedUser(renewal_rules(), vocab=LIBRARY, **kwargs)


def test_a_run_whose_goal_writes_are_all_made_ends_goal_satisfied():
    user = renewal_user(write_tools={"renew_loan", "pay_fine"}, goal_writes={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(RENEWED)
    assert user.done is True
    assert user.end_reason == GOAL_SATISFIED
    assert GOAL_SATISFIED in user.events[-1].payload["tags"]


def test_a_goal_that_implies_two_writes_is_not_satisfied_by_one_of_them():
    user = renewal_user(write_tools={"renew_loan", "pay_fine"}, goal_writes={"renew_loan", "pay_fine"})
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(RENEWED)
    assert user.done is False
    assert "L2201" in text and said(user) == {GOAL: "goal_restated"}


def test_the_closing_cue_alone_does_not_end_a_run_whose_goal_writes_are_not_done():
    """The cue was the whole end before D210, so a Run that had done half the work read as finished."""
    user = renewal_user(write_tools={"renew_loan", "pay_fine"}, goal_writes={"renew_loan", "pay_fine"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("All done. Have a great day!"))
    assert user.done is False
    assert user.end_reason is None


def test_a_candidate_that_closes_over_an_unfinished_goal_ends_handed_off():
    user = renewal_user(write_tools={"renew_loan"}, goal_writes={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("Is there anything else I can help you with?"))
    user.reply(ask("Thanks for contacting us, have a great day."))
    assert user.done is True
    assert user.end_reason == HANDED_OFF


def test_a_candidate_that_passes_the_conversation_on_ends_handed_off():
    user = renewal_user(write_tools={"renew_loan"}, goal_writes={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("One moment while I look into that."))
    user.reply(ask("I am transferring you to a colleague who can do that."))
    assert user.end_reason == HANDED_OFF


def test_a_user_with_nothing_left_to_say_ends_scenario_exhausted():
    user = renewal_user(write_tools={"renew_loan"}, goal_writes={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("One moment while I look into that."))
    user.reply(ask("Still checking, sorry for the wait."))
    assert user.done is True
    assert user.end_reason == SCENARIO_EXHAUSTED


def test_two_asks_the_user_has_no_record_of_end_the_run_scenario_exhausted():
    user = renewal_user(write_tools={"renew_loan"}, goal_writes={"renew_loan"})
    user.reply(ask("Hi! How can I help you today?"))
    user.reply(ask("What is your member id?"))
    user.reply(ask("Could you give me your email address?"))
    user.reply(ask("Thanks for contacting us, have a great day."))  # the goal is said once more
    user.reply(ask("Thanks for contacting us, have a great day."))
    assert user.end_reason == SCENARIO_EXHAUSTED


def test_a_run_that_spends_its_turns_without_ending_gave_up(make_test_model):
    from kullback.runner import loop

    user = renewal_user(write_tools={"renew_loan"}, goal_writes={"renew_loan"})
    model = make_test_model([{"content": "What is your member id?"}], loop=True)
    state = loop.new_run_state("spent", user=user, max_turns=4)
    loop.run(state, model)
    assert state.run.termination_reason == "max_turns"
    assert end_of_run(state.run) == GAVE_UP


def test_the_end_kind_of_a_finished_run_is_read_off_the_run_the_loop_wrote(make_test_model):
    from kullback.runner import loop

    user = renewal_user(write_tools={"renew_loan"}, goal_writes=set())
    model = make_test_model([{"content": "Is there anything else I can help you with?"}], loop=True)
    state = loop.new_run_state("finished", user=user, max_turns=8)
    loop.open_with_user(state)
    loop.run(state, model)
    assert end_of_run(state.run) == GOAL_SATISFIED


def test_the_ends_of_a_round_are_counted_in_every_kind():
    rows = [{"user_end": GOAL_SATISFIED}, {"user_end": GOAL_SATISFIED}, {"user_end": GAVE_UP},
            {"user_end": None}]
    assert ends_by_kind(rows) == {GOAL_SATISFIED: 2, SCENARIO_EXHAUSTED: 0, HANDED_OFF: 0, GAVE_UP: 1}


def test_a_fact_the_recorded_user_gave_is_askable_and_one_only_the_world_holds_is_a_record_fact():
    rules = derive_user_rules(bold_name_trace(), LIBRARY)
    assert fact_class(rules, "name") == ASKABLE
    assert fact_class(rules, "member_id") == RECORD


def asked_name_trace() -> Trace:
    """A recording whose user gave its name only because the help desk asked for it."""
    return library_trace([
        ("assistant", "Hi! How can I help you today?"),
        ("user", "I would like to renew a book."),
        ("assistant", "Of course. What is your full name?"),
        ("user", "My name is Ada Whitfield."),
    ])


def test_an_askable_fact_is_given_once_asked_and_not_before():
    rules = derive_user_rules(asked_name_trace(), LIBRARY)
    assert fact_class(rules, "name") == ASKABLE
    user = SimulatedUser(rules, vocab=LIBRARY)
    opening = user.reply(ask("Hi! How can I help you today?"))
    assert "Ada Whitfield" not in opening
    answer = user.reply(ask("What is your full name?"))
    assert "Ada Whitfield" in answer
    assert said(user)["name"] == "rules"


def test_a_question_about_a_record_fact_is_answered_without_the_value():
    """The user knows what it told the help desk and nothing else; the rest is the world's, and the
    Candidate has the tools that read it (D210)."""
    user = SimulatedUser(derive_user_rules(bold_name_trace(), LIBRARY), vocab=LIBRARY,
                         starting_state_reader=dict_reader({"member_id": "M400318"}))
    user.reply(ask("Hi! How can I help you today?"))
    text = user.reply(ask("Could you give me your member id?"))
    assert "M400318" not in text
    assert "look it up" in text
    assert said(user) == {"member_id": "record"}
    assert user.events[-1].payload["record_fields"] == ["member_id"]


def test_a_value_the_strip_removes_is_never_spoken_and_is_counted():
    """A value the recording's tools held and no user of the Task said is not this user's to give,
    however it reached the rules; the fact is answered as absent instead (D196, D210)."""
    trace = library_trace(
        [
            ("assistant", "Hi! How can I help you today?"),
            ("user", "I would like to renew a book."),
            ("assistant", None),
        ],
        calls=[made(2, "get_member", {"member_id": "M400318"}, result={"loans": ["L2201"]})],
    )
    leaky = UserRules(facts=[UserFact(field="member_id", value="M400318", span=PTR)])
    user = SimulatedUser(leaky, vocab=LIBRARY, answer_strip=intent_mod.value_strip([trace]))
    text = user.reply(ask("Could you give me your member id?"))
    assert "M400318" not in text
    assert user.stripped == 1
    assert said(user) == {"member_id": "stripped"}
    assert STRIPPED_TAG in user.events[-1].payload["tags"]


def test_a_fact_a_user_of_the_task_did_say_is_left_alone_by_the_strip():
    trace = library_trace(
        [
            ("assistant", "Hi! How can I help you today?"),
            ("user", "My member id is M400318 and I would like to renew a book."),
            ("assistant", None),
        ],
        calls=[made(2, "get_member", {"member_id": "M400318"}, result={"loans": ["L2201"]})],
    )
    user = SimulatedUser(derive_user_rules(trace, LIBRARY), vocab=LIBRARY,
                         answer_strip=intent_mod.value_strip([trace]))
    text = user.reply(ask("Could you give me your member id?"))
    assert "M400318" in text
    assert user.stripped == 0


def test_the_goal_write_set_is_the_writes_the_recording_made_and_the_world_took():
    trace = library_trace(
        [
            ("assistant", "Hi! How can I help you today?"),
            ("user", "Please renew loan L2201 and pay the fine on it."),
            ("assistant", None),
            ("assistant", None),
            ("assistant", None),
        ],
        calls=[
            made(2, "get_member", {"member_id": "M400318"}, result={"loans": ["L2201"]}),
            made(3, "renew_loan", {"loan_id": "L2201"}, result={"due": "2026-10-01"}),
            made(4, "pay_fine", {"loan_id": "L2201"},
                 error=ToolCallError(**{"class": "not_found_entity"}, payload="no fine")),
        ],
    )
    assert goal_write_set(trace, {"renew_loan", "pay_fine"}) == {"renew_loan"}
    assert goal_write_set(None, {"renew_loan"}) == set()
