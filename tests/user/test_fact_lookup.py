"""The fact store and the one lookup both Simulated users answer from (D332)."""

from __future__ import annotations

import re
from pathlib import Path

from kullback.runner.records import UserFact, UserRules
from kullback.user.lookup import ALIAS, FIELD, OVERLAP, FactStore, asked_in
from kullback.user.simulated import SimulatedUser
from kullback.user.vocabulary import from_signatures

SIGS = [{"name": "fetch_widget", "args_fields": [{"name": "widget_id"}, {"name": "slot_count"}]}]
VOCAB = from_signatures(SIGS)


def store(*facts: tuple[str, str], names=()) -> FactStore:
    return FactStore([UserFact(field=f, value=v) for f, v in facts], VOCAB, names)


def test_a_question_naming_the_field_gets_its_fact_by_field_name():
    found = store(("widget_id", "WG-17")).lookup("What is the widget id?")
    assert [(hit.field, hit.value, hit.how) for hit in found.hits] == [("widget_id", "WG-17", FIELD)]


def test_an_alias_from_the_signatures_finds_the_fact():
    found = store(("widget_id", "WG-17")).lookup("Could you give me the widget number?")
    assert [(hit.field, hit.how) for hit in found.hits] == [("widget_id", ALIAS)]


def test_plain_word_overlap_finds_a_fact_no_name_or_alias_carries():
    found = store(("widget_id", "WG-17"), ("slot_count", "3")).lookup("Which widget is it about?")
    assert [(hit.field, hit.how) for hit in found.hits] == [("widget_id", OVERLAP)]
    plural = store(("slot_count", "3")).lookup("Which slots do you need?")
    assert [hit.field for hit in plural.hits] == ["slot_count"]


def test_a_name_heard_with_no_value_reaches_the_held_facts_sharing_its_words():
    found = store(("address1", "1 Elm Row"), ("address2", "Flat 2")).lookup("What is your address?")
    assert {hit.field for hit in found.hits} == {"address1", "address2"} and found.missing == []


def test_a_field_named_but_held_nowhere_comes_back_missing():
    found = store(("slot_count", "3")).lookup("What is your email?")
    assert found.hits == [] and found.missing == ["email"]


def test_a_value_the_question_states_is_not_asked_for():
    found = store(("widget_id", "WG-17")).lookup("Shall I cancel widget WG-17 for you?")
    assert found.hits == []


def test_a_word_re_pointed_by_of_or_a_yes_ask_names_no_field():
    held = store(("name", "Ann Example"), ("widget_id", "WG-17"))
    assert held.lookup("What is the name of the product you want?").hits == []
    assert held.lookup("Do you confirm the widget change?").hits == []


def test_two_values_of_one_field_come_back_with_the_one_the_question_names_first():
    facts = [UserFact(field="address1", value="1 Elm Row", context="my address is 1 Elm Row"),
             UserFact(field="address1", value="9 Oak Lane", context="please change it to 9 Oak Lane")]
    found = FactStore(facts, VOCAB).lookup("What is the new address you want?")
    assert [hit.value for hit in found.hits] == ["9 Oak Lane", "1 Elm Row"]


def test_the_spoken_lines_are_never_facts_to_look_up():
    found = store(("goal", "I want my widget fixed"), ("confirmation", "yes")).lookup("What is the goal?")
    assert found.hits == [] and store(("goal", "x")).held() == []


def test_only_the_request_sentences_of_a_turn_are_looked_up():
    assert asked_in("Your widget is updated. Is there anything else?") == "is there anything else?"
    assert asked_in("Your widget is updated.") == ""


def test_the_rule_user_answers_through_the_lookup_and_keeps_every_lookup():
    rules = UserRules(facts=[UserFact(field="goal", value="Please fix my widget."),
                             UserFact(field="widget_id", value="WG-17")])
    user = SimulatedUser(rules, vocab=VOCAB)
    user.reply([{"role": "assistant", "content": "Hello, how can I help?"}])
    text = user.reply([{"role": "assistant", "content": "Which widget is it?"}])
    assert "WG-17" in text
    assert [hit.field for hit in user.lookups[-1].hits] == ["widget_id"]


def test_no_trust_or_verifier_module_reads_the_users_goal():
    """The user's goal is reported on the Run; spec, gates and the Verdict never read it (D332)."""
    root = Path(__file__).resolve().parents[2] / "kullback"
    paths = [*sorted((root / "spec").rglob("*.py")), *sorted((root / "gates").rglob("*.py")),
             root / "runner" / "verdict.py", root / "runner" / "records.py"]
    reads = re.compile(r"GOAL_SATISFIED|goal_satisfied|user_goal_met|user_end\b|goal_done|end_reason")
    offenders = [str(path.relative_to(root)) for path in paths if reads.search(path.read_text())]
    assert offenders == []


def test_a_closing_question_or_an_offer_on_your_thing_asks_for_no_fact():
    held = store(("widget_id", "WG-17"))
    assert held.lookup("Anything else, like your other widgets?").hits == []
    assert held.lookup("Would you like me to add a cover to your widget?").hits == []
    assert [hit.field for hit in held.lookup("Which widget should I add the cover to? Reply yes to confirm.").hits] \
        == ["widget_id"]


def test_a_missing_name_reaches_only_held_facts_sharing_its_words():
    found = store(("widget_id", "WG-17")).lookup("Before I change your widget I need your email.")
    assert found.hits == [] and found.missing == ["email"]


def test_a_name_made_only_of_kind_words_is_never_heard():
    vocab = from_signatures([{"name": "fetch", "args_fields": [{"name": "id"}, {"name": "widget_id"}]}])
    found = FactStore([UserFact(field="widget_id", value="WG-17")], vocab).lookup("What is the widget id?")
    assert found.fields() == ["widget_id"]
