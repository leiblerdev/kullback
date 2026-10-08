from kullback.spec.ground import coverage, ground_spec, valid_because, valid_write_because
from tests.spec.fixtures import check, spec

INTENT = "Please move item A1 to slot seven.\nI also want a note saying it was urgent."


def test_a_because_that_quotes_the_intent_is_valid():
    assert valid_because("\"MOVE item a1   to slot seven.\"", INTENT)


def test_a_because_quoting_twelve_characters_not_in_the_intent_is_refused():
    assert not valid_because("move item Z9 to the roof", INTENT)


def test_a_named_policy_section_is_valid_and_an_unknown_one_is_refused():
    sections = ["refunds-2", "section:identity check"]
    assert valid_because("policy refunds-2 requires it", INTENT, sections)
    assert valid_because("per the identity check rule", INTENT, sections)
    assert not valid_because("policy shipping-9 requires it", INTENT, sections)


def test_coverage_lists_the_uncovered_fact_unless_a_check_names_it():
    assert coverage(spec()) == ["f2"]
    assert coverage(spec(gaps=["f2"])) == []
    named = check("c2", "communicate", {"demand": "say", "text": "urgent"},
                  because="a note saying it was urgent", fact_ids=["f2"])
    assert coverage(spec([check(), named])) == []


def test_ground_spec_refuses_an_ungrounded_check_and_an_uncovered_fact():
    refusals = ground_spec(spec([check(because="nothing the user said here")]))
    assert any("check c1" in r for r in refusals)
    assert any("fact f2" in r for r in refusals)
    assert ground_spec(spec(gaps=["f2"])) == []


def test_ground_spec_never_raises_on_a_malformed_spec():
    assert ground_spec(object()) and ground_spec(spec(), policy_sections=None) == ground_spec(spec())


def test_a_section_matches_as_whole_words_only():
    assert not valid_because("a1 is the item", INTENT, ["a", "1"])
    assert valid_because("per the Identity  Check rule", INTENT, ["identity check"])


def test_a_write_because_must_quote_the_intent_or_the_policy_text_and_a_section_name_is_not_enough():
    policy = "## Moving items\nMove an item only to a free slot."
    assert valid_write_because("move item A1 to slot seven", INTENT, policy)
    assert valid_write_because('per moving items: "Move an item only to a free slot"', INTENT, policy)
    assert not valid_write_because("per the moving items rules", INTENT, policy)
    assert not valid_write_because("", INTENT, policy)
