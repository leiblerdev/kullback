"""Tests for the agentic judge: a tool check before every verdict, two judges, the disagreement queue."""

from __future__ import annotations

import json

import pytest

from kullback.ai.provider import ProviderError
from kullback.runner.judge import (
    JUDGE_VERSION,
    AgenticJudge,
    JudgeResult,
    abstain_verdict,
    confirm_reference,
    disagreement_rate,
    judge_name,
    read_disagreement_queue,
    set_task_aside,
    smoke,
    smoke_lines,
    tasks_set_aside,
    third_judge,
    two_judges,
)

# --- read-only tools the judge is given (design 13b: Starting state and End state) ---


def read_order(order_id: str = "o1") -> dict:
    """Read one order out of the End state."""
    return {"order_id": order_id, "status": "cancelled", "reason": "no longer needed"}


def list_writes() -> list:
    """Every write the Run made."""
    return [{"table": "orders", "id": "o1", "status": "cancelled"}]


def read_row(table: str) -> dict:
    """Read one row of a table the caller names, which only the model can choose."""
    return {"table": table, "rows": 1}


TOOLS = {"read_order": read_order, "list_writes": list_writes}


class RefusesToolChoice:
    """A provider that does not carry tool_choice and refuses the whole request when it is sent."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "model")
        self.refusals = 0

    def query(self, messages, tools=None, config=None):
        if config is not None and getattr(config, "tool_choice", None):
            self.refusals += 1
            raise ProviderError("unknown parameter: tool_choice", status=400)
        return self.inner.query(messages, tools=tools, config=config)


def call(name: str = "read_order", **args) -> dict:
    """A scripted model reply that calls one tool."""
    return {"tool_calls": [{"id": "c1", "name": name, "arguments": args}]}


def answer(**payload) -> dict:
    """A scripted model reply that returns the judge's JSON verdict."""
    return {"content": json.dumps(payload)}


def yes(question: str = "did the agent state the action first") -> dict:
    return {"question": question, "answer": "yes", "cited_span": "I will cancel order o1"}


TRANSCRIPT = "agent: I will cancel order o1\nuser: yes please\nagent: cancelled it"


# --- policy atoms (D76, R27 section 8b: atomic yes or no sub-questions) ---


@pytest.mark.parametrize(
    "sub_answers, expected",
    [
        ([yes(), yes("did the user answer yes before the write")], "pass"),
        ([yes(), {"question": "q", "answer": "no"}], "fail"),
        ([yes(), {"question": "q", "answer": "No!"}], "fail"),
        ([{"question": "q", "answer": "Yes."},
          {"question": "q2", "answer": "yes, the agent asked first"}], "pass"),
        ([{"question": "q", "answer": "abstain"}], "abstain"),
        ([{"question": "q", "answer": "maybe"}], "abstain"),
    ],
    ids=["every_sub_answer_yes", "a_plain_no", "a_no_with_punctuation", "a_yes_with_punctuation_or_a_tail",
         "an_unanswered_sub_question", "a_word_that_is_neither"],
)
def test_the_policy_atom_passes_only_when_every_sub_answer_reads_as_yes(make_test_model, sub_answers, expected):
    """Every row scripts the model's own verdict word as "pass", so the verdict can only come from
    the sub-answers, however they are spelled (D76, R27 section 8b)."""
    model = make_test_model(
        [call(order_id="o1"),
         answer(verdict="pass", cited_spans=["I will cancel order o1"], sub_answers=sub_answers)]
    )
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == expected
    assert result.use == "policy_atom"
    assert result.tools_run == ["read_order"]
    assert result.cited_spans == ["I will cancel order o1"]
    assert result.judge_version == JUDGE_VERSION
    assert result.refused is False
    assert result.abstained is (expected == "abstain")


def test_policy_atom_abstains_when_the_model_gives_no_sub_answers(make_test_model):
    model = make_test_model([call(), answer(verdict="pass")])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "abstain"
    assert "sub-answer" in (result.reason or "")


# --- the check rule (D222: the harness runs the check, then the model is asked) ---


def test_a_verdict_with_no_tool_call_stands_when_the_harness_prefilled_the_checks(make_test_model):
    """The check the question needs was already run, so answering straight away is a whole answer."""
    model = make_test_model([answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "pass"
    assert result.refused is False
    assert result.tools_run == []
    assert result.extra_tool_calls == 0
    assert [check["tool"] for check in result.checks] == ["policy_rule", "transcript",
                                                          "read_order", "list_writes"]


def test_a_tool_whose_argument_only_the_model_can_choose_is_left_out_of_the_prefilled_checks(
    make_test_model,
):
    """A view that needs something chosen is the model's to open; rule 2 forces that first turn."""
    model = make_test_model([answer(verdict="acceptable", cited_spans=["x"])])
    judge = judge_of(model, tools={"read_row": read_row, "list_writes": list_writes})
    result = judge.judge_dispute({"orders": {}}, [], [])
    assert "read_row" not in [check["tool"] for check in result.checks]
    assert "list_writes" in [check["tool"] for check in result.checks]
    assert model.calls[0]["config"].tool_choice == "required"
    assert result.tool_choice_forced is True
    assert result.tool_choice_rejected is False


def test_no_tool_choice_is_asked_for_when_every_check_was_already_run(make_test_model):
    model = make_test_model([answer(verdict="acceptable", cited_spans=["x"])])
    result = judge_of(model, tools={"list_writes": list_writes}).judge_dispute({}, [], [])
    assert model.calls[0]["config"].tool_choice is None
    assert result.tool_choice_forced is False


def test_a_provider_that_refuses_tool_choice_is_asked_again_without_it_and_the_refusal_is_recorded(
    make_test_model,
):
    """The checks are in the prompt either way, so rule 1 alone still answers the question."""
    inner = make_test_model([answer(verdict="acceptable", cited_spans=["x"])])
    model = RefusesToolChoice(inner)
    judge = AgenticJudge(model, {"read_row": read_row, "list_writes": list_writes}, name="a")
    result = judge.judge_dispute({"orders": {}}, [], [])
    assert result.verdict == "acceptable"
    assert result.tool_choice_rejected is True
    assert result.tool_choice_forced is False


def test_a_question_with_no_check_to_prefill_is_refused_as_a_harness_bug(make_test_model):
    model = make_test_model([answer(verdict="acceptable")])
    result = judge_of(model, tools={}).ask("dispute", "decide this", {})
    assert result.verdict == "abstain"
    assert result.refused is True
    assert result.checks == []
    assert "D222" in (result.reason or "")


def test_a_call_to_an_unknown_or_failing_tool_is_not_counted_and_its_error_goes_back_to_the_judge(
    make_test_model,
):
    model = make_test_model([call(name="drop_table"), answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "pass"
    assert result.tools_run == []
    tool_reply = model.calls[1]["messages"][-1]
    assert tool_reply["role"] == "tool"
    assert "tool_not_found" in tool_reply["content"]

    def boom() -> dict:
        """Always fails."""
        raise ValueError("no such table")

    model = make_test_model([call(name="boom"), call(order_id="o1"), answer(verdict="pass", sub_answers=[yes()])])
    result = AgenticJudge(model, {"boom": boom, "read_order": read_order}).judge_policy_atom("rule", TRANSCRIPT)
    assert result.tools_run == ["read_order"]
    assert result.verdict == "pass"
    # TestModel keeps the live messages list, so read the final state of the conversation.
    assert any("no such table" in str(m.get("content")) for m in model.calls[-1]["messages"])


def test_the_judge_gives_up_after_max_steps_of_tool_calls(make_test_model):
    model = make_test_model([call()], loop=True)
    judge = AgenticJudge(model, TOOLS, max_steps=2)
    result = judge.judge_dispute({"orders": {"o1": "cancelled"}}, ["cancel o1"], [])
    assert result.verdict == "abstain"
    assert result.refused is True
    assert result.tools_run == ["read_order", "read_order"]


def test_the_model_is_given_the_read_only_tools(make_test_model):
    model = make_test_model([call(), answer(verdict="pass", sub_answers=[yes()])])
    judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert {spec["name"] for spec in model.calls[0]["tools"]} == {"read_order", "list_writes"}


# --- semantic equivalence (D84, R27 section 8a) ---


def test_equivalence_abstains_when_the_judge_flags_a_number_or_unit_mismatch(make_test_model):
    model = make_test_model(
        [call(), answer(verdict="equivalent", flags=["number mismatch"], cited_spans=["25", "25.5"])]
    )
    result = judge_of(model).judge_equivalence("orders.total", "25", "25.5", field_type="currency amount")
    assert result.verdict == "abstain"
    assert "mismatch" in (result.reason or "")


@pytest.mark.parametrize("reply, reason_part", [
    ({"content": "I think the two values mean the same thing."}, None),
    (answer(verdict="probably fine"), "probably fine"),
], ids=["not_json", "an_unknown_verdict_word"])
def test_a_judge_reply_that_cannot_be_read_abstains(make_test_model, reply, reason_part):
    model = make_test_model([call(), reply])
    result = judge_of(model).judge_equivalence("orders.reason", "a", "b")
    assert result.verdict == "abstain"
    assert result.refused is False
    if reason_part is not None:
        assert reason_part in (result.reason or "")


def test_json_wrapped_in_prose_is_still_read(make_test_model):
    reply = ('Here is my answer:\n{"verdict": "equivalent", "evidence": ["value_a", "value_b"]}\n'
             "That is all.")
    model = make_test_model([call(), {"content": reply}])
    assert judge_of(model).judge_equivalence("orders.reason", "a", "b").verdict == "equivalent"


@pytest.mark.parametrize("reply, value_b, expected, uncited", [
    (answer(verdict="equivalent", cited_spans=["they read alike"]), "b", "abstain", True),
    (answer(verdict="equivalent", evidence=["value_a"]), "a ", "equivalent", False),
    # only agreement carries the rule, so a different answer needs no citation
    (answer(verdict="not_equivalent"), "b", "not_equivalent", False),
], ids=["equal_citing_nothing", "equal_naming_a_prefilled_check", "different_citing_nothing"])
def test_an_equal_answer_that_names_no_check_is_unresolved_rather_than_agreement(
    make_test_model, reply, value_b, expected, uncited
):
    """D222 rule 3: an agreement rests on a named check, or nobody settled the pair (D219)."""
    model = make_test_model([reply])
    result = judge_of(model).judge_equivalence("orders.reason", "a", value_b)
    assert result.verdict == expected
    assert (result.reason or "").startswith("uncited") is uncited


# --- a ruling may rest only on what the judge was handed (D93, build 8's hand fix 3) ---


_NO_AUTH = "no evidence of the required authentication and explicit confirmation"


@pytest.mark.parametrize("use, reply, expected, evidence, reason_has, reason_lacks", [
    ("reference", dict(verdict="bad_reference", evidence=["transcript"], reason=_NO_AUTH),
     "abstain", None, ["transcript", "was not given"], []),
    ("reference", dict(verdict="bad_reference", evidence="transcript", reason=_NO_AUTH),
     "abstain", None, ["transcript", "was not given"], []),
    ("reference", dict(verdict="bad_reference", evidence=["end_state", "the transcript"], reason=_NO_AUTH),
     "abstain", None, ["transcript", "was not given"], []),
    # the rule never turns into a pass either: D110 holds in both directions
    ("policy", dict(verdict="pass", evidence=["crm_export"], sub_answers=[yes(), yes()]),
     "abstain", None, ["crm_export"], []),
    ("reference", dict(verdict="bad_reference", evidence=["Intent", "End states"],
                       reason="an order the Intent never names was exchanged"),
     "bad_reference", ["Intent", "End states"], [], []),
    ("policy", dict(verdict="pass", evidence=["transcript", "state_tools"], sub_answers=[yes()]),
     "pass", None, [], ["transcript"]),
    ("reference", dict(verdict="bad_reference", reason="nothing was written"),
     "bad_reference", [], [], []),
], ids=["reference_on_the_transcript_as_a_list", "reference_on_the_transcript_as_one_string",
        "reference_on_the_transcript_beside_a_source_it_does_have", "a_pass_on_a_source_it_was_not_given",
        "reference_on_what_it_was_handed_still_fails", "a_judge_handed_the_transcript_may_rest_on_it",
        "a_reply_with_no_evidence_field_is_read_as_before"])
def test_a_ruling_resting_on_a_source_the_judge_was_not_given_abstains(
    make_test_model, use, reply, expected, evidence, reason_has, reason_lacks
):
    """Build 8: eleven Tasks were failed "without evidence of the required authentication and explicit
    confirmation" by a judge that was handed no transcript, where D93 says a person decides."""
    model = make_test_model([call(), answer(**reply)])
    if use == "reference":
        result = AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "cancel order o1")
    else:
        result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == expected
    if expected == "abstain":
        assert result.abstained is True
        assert result.refused is False
        if use == "reference":
            assert result.evidence[-1].endswith("transcript")
    if evidence is not None:
        assert result.evidence == evidence
    for part in reason_has:
        assert part in (result.reason or "")
    for part in reason_lacks:
        assert part not in (result.reason or "")


def test_a_run_that_matches_a_revised_intent_is_not_failed_for_the_opening_request(make_test_model):
    """Build 8: the Intent said "exchange the desk lamp only" because the user changed their mind, the
    frontier did exactly that, and the judge failed it on what the user had asked for first."""
    model = make_test_model([call(), answer(
        verdict="bad_reference", evidence=["opening_request"],
        reason="the user requested exchanges for both items")])
    result = AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "exchange the desk lamp only")
    assert result.verdict == "abstain"
    assert "opening_request" in (result.reason or "")
    prompt = model.calls[0]["messages"][1]["content"]
    assert "ground truth of what the user wanted by the end of the Run" in prompt


def test_a_verifier_output_passed_to_the_call_wins_over_the_one_on_the_judge(make_test_model):
    model = make_test_model([call(), answer(verdict="bad_reference")])
    judge = AgenticJudge(model, TOOLS, verifier_output={"pass": True})
    judge.judge_reference({"run_id": "r1"}, "intent", verifier_output={"pass": False, "failing_atom": "a2"})
    prompt = model.calls[0]["messages"][1]["content"]
    assert "a2" in prompt


# --- failure cause (D88) ---


def test_cause_uses_the_cause_vocabulary_and_undetermined_is_its_abstain(make_test_model):
    model = make_test_model([call(), answer(verdict="environment", cited_spans=["tool error"])])
    result = judge_of(model).judge_cause({"run_id": "r2"}, {"run_id": "r1"})
    assert result.verdict == "environment"
    assert result.use == "cause"
    assert abstain_verdict("cause") == "undetermined"
    assert result.abstained is False

    model = make_test_model([call(), answer(verdict="undetermined")])
    result = judge_of(model).judge_cause({"run_id": "r2"}, {"run_id": "r1"})
    assert result.verdict == "undetermined"
    assert result.abstained is True


# --- two judges and the disagreement queue (D92) ---


def test_two_judges_that_disagree_queue_both_verdicts_and_abstain_and_two_that_agree_queue_nothing(
    make_test_model, workdir
):
    agreed = workdir / "agreed"
    agreed.mkdir()
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "acceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=agreed, item_id="run1")
    assert disagreement is False
    assert result.verdict == "acceptable"
    assert result.judge == "a"
    assert [entry["verdict"] for entry in result.pair] == ["acceptable", "acceptable"]
    assert read_disagreement_queue(agreed) == []
    assert disagreement_rate(agreed) == {"pairs": 1, "disagreements": 0, "rate": 0.0,
                                         "abstains": 0, "abstain_rate": 0.0,
                                         "by_pair": {"a vs b": {"pairs": 1, "disagreements": 0,
                                                                "rate": 0.0}}}

    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, item_id="run1",
                                      third_sample=False)
    assert disagreement is True
    assert result.verdict == "abstain"
    assert "acceptable" in (result.reason or "") and "unacceptable" in (result.reason or "")
    queue = read_disagreement_queue(workdir)
    assert len(queue) == 1
    assert queue[0]["use"] == "dispute"
    assert queue[0]["item_id"] == "run1"
    assert queue[0]["verdict_a"] == "acceptable"
    assert queue[0]["verdict_b"] == "unacceptable"
    assert queue[0]["judge_a"]["cited_spans"] == ["by a"]
    assert queue[0]["judge_b"]["cited_spans"] == ["by b"]
    # D160: the row says which two judges these verdicts came from, and names the pair the
    # by-pair rate counts them under.
    assert queue[0]["judge_a_name"] == "a" and queue[0]["judge_b_name"] == "b"
    assert queue[0]["judges"] == "a vs b"
    assert queue[0]["reason"] == "split"
    assert queue[0]["disagreement"] is True
    assert disagreement_rate(workdir)["rate"] == 1.0

    # Nowhere to queue the split is not a reason to pick a side: the caller still gets the abstain.
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], third_sample=False)
    assert disagreement is True
    assert result.verdict == "abstain"
    assert [entry["verdict"] for entry in result.pair] == ["acceptable", "unacceptable"]
    assert "acceptable" in (result.reason or "") and "unacceptable" in (result.reason or "")


def test_the_disagreement_rate_can_be_read_for_one_use(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, third_sample=False)
    c = named_judge(make_test_model, "c", "environment")
    d = named_judge(make_test_model, "d", "environment")
    two_judges(c, d, "judge_cause", {}, {}, workdir=workdir)
    assert disagreement_rate(workdir) == {"pairs": 2, "disagreements": 1, "rate": 0.5,
                                          "abstains": 0, "abstain_rate": 0.0,
                                          "by_pair": {"a vs b": {"pairs": 1, "disagreements": 1,
                                                                 "rate": 1.0},
                                                      "c vs d": {"pairs": 1, "disagreements": 0,
                                                                 "rate": 0.0}}}
    assert disagreement_rate(workdir, use="cause") == {"pairs": 1, "disagreements": 0,
                                                       "rate": 0.0, "abstains": 0,
                                                       "abstain_rate": 0.0,
                                                       "by_pair": {"c vs d": {"pairs": 1,
                                                                              "disagreements": 0,
                                                                              "rate": 0.0}}}


@pytest.mark.parametrize("verdict_b, verdict_c, expected, disagreement_expected, queued", [
    ("unacceptable", "unacceptable", "unacceptable", False, []),
    ("unacceptable", "abstain", "abstain", True, ["split"]),
    ("abstain", "abstain", "abstain", None, ["abstain_majority"]),
    # the default third sample reuses judge A's model, which may have nothing left to say
    ("unacceptable", None, "abstain", True, ["split"]),
], ids=["a_majority_settles_it", "a_three_way_split", "an_abstain_majority", "a_third_sample_that_cannot_answer"])
def test_a_third_sample_settles_a_split_by_majority_or_leaves_it_in_the_queue(
    make_test_model, workdir, verdict_b, verdict_c, expected, disagreement_expected, queued
):
    """D97: the two judges split, a third sample from judge A's model decides by majority."""
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", verdict_b)
    third = {} if verdict_c is None else {"judge_c": named_judge(make_test_model, "c", verdict_c)}
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="run1", **third)
    if disagreement_expected is not None:
        assert disagreement is disagreement_expected
    assert result.verdict == expected
    assert [row["reason"] for row in read_disagreement_queue(workdir)] == queued
    if verdict_c is None:
        assert result.pair[2]["refused"] is True
        assert "did not answer" in result.pair[2]["reason"]
    else:
        assert [entry["verdict"] for entry in result.pair] == ["acceptable", verdict_b, verdict_c]


def test_two_judge_models_name_themselves_and_the_pair_they_disagreed_as(make_test_model, workdir):
    """D160: a ruling names the model it ran on, so a split is between two named models, not two personas."""
    a = named_judge(make_test_model, judge_name("vendor/large", "a"), "acceptable")
    b = named_judge(make_test_model, judge_name("other/small", "b"), "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="run1", third_sample=False)
    assert disagreement is True
    assert [entry["judge"] for entry in result.pair] == ["vendor/large:a", "other/small:b"]
    assert result.judge == "vendor/large:a+other/small:b"
    assert disagreement_rate(workdir)["by_pair"] == {
        "vendor/large:a vs other/small:b": {"pairs": 1, "disagreements": 1, "rate": 1.0}}

    # An older build's rows say nothing about which two models ran, so they count in no pair.
    older = workdir / "older"
    older.mkdir()
    (older / "judge_pairs.jsonl").write_text(
        json.dumps({"use": "dispute", "disagreement": True}) + "\n", encoding="utf-8")
    stats = disagreement_rate(older)
    assert stats["pairs"] == 1 and stats["rate"] == 1.0
    assert stats["by_pair"] == {}


def test_a_reference_split_never_takes_a_third_sample(make_test_model, workdir):
    """D93: a Reference sets the bar for its Task, so a split there goes to a person."""
    a = named_judge(make_test_model, "a", "good_reference")
    b = named_judge(make_test_model, "b", "bad_reference")
    result, disagreement = two_judges(a, b, "judge_reference", {}, "intent", workdir=workdir)
    assert disagreement is True
    assert result.verdict == "abstain"
    assert len(result.pair) == 2


def test_the_default_third_sample_reuses_judge_a_under_another_persona(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    third = third_judge(a)
    assert third.model is a.model
    assert third.tools == a.tools
    assert third.name == "a#3"
    # cli.py builds its default second judge this way, and names it, so the tie-breaker that
    # reuses judge A's model is not the same name as judge B (D160).
    assert third_judge(a, name=judge_name("a", "b")).name == "a:b"
    assert third.persona and third.persona != a.persona

    # and it runs: judge A's model is scripted for two turns of its own plus two for the third sample
    model_a = make_test_model([call(), answer(verdict="acceptable", cited_spans=["by a"]),
                               call(), answer(verdict="unacceptable", cited_spans=["by a#3"])])
    left = AgenticJudge(model_a, TOOLS, name="a")
    right = named_judge(make_test_model, "b", "unacceptable")
    result, disagreement = two_judges(left, right, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="run1")
    assert disagreement is False
    assert result.verdict == "unacceptable"
    assert [entry["judge"] for entry in result.pair] == ["a", "b", "a#3"]


# --- a disputed Reference sets the Task aside (D93) ---


def test_a_confirmed_reference_leaves_the_task_in_the_build(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "good_reference")
    b = named_judge(make_test_model, "b", "good_reference")
    confirmed, result = confirm_reference(a, b, {"run_id": "r1"}, "intent", workdir=workdir, task_id="t1")
    assert confirmed is True
    assert result.verdict == "good_reference"
    assert tasks_set_aside(workdir) == []


def test_a_disputed_or_bad_reference_sets_the_task_aside_and_no_third_judge_runs(make_test_model, workdir):
    model_a = make_test_model([call(), answer(verdict="good_reference", cited_spans=["by a"])])
    model_b = make_test_model([call(), answer(verdict="bad_reference", cited_spans=["by b"])])
    a = AgenticJudge(model_a, TOOLS, name="a")
    b = AgenticJudge(model_b, TOOLS, name="b")
    confirmed, result = confirm_reference(a, b, {"run_id": "r1"}, "intent", workdir=workdir, task_id="t1")
    assert confirmed is False
    assert result.verdict == "abstain"
    aside = tasks_set_aside(workdir)
    assert len(aside) == 1
    assert aside[0]["task_id"] == "t1"
    assert aside[0]["reason"] == "reference_disputed"
    assert aside[0]["judge_a"]["verdict"] == "good_reference"
    assert aside[0]["judge_b"]["verdict"] == "bad_reference"
    assert aside[0]["judge_a"]["cited_spans"] == ["by a"]
    assert len(read_disagreement_queue(workdir)) == 1
    assert len(model_a.calls) == 2 and len(model_b.calls) == 2

    both_bad = workdir / "both_bad"
    both_bad.mkdir()
    a = named_judge(make_test_model, "a", "bad_reference")
    b = named_judge(make_test_model, "b", "bad_reference")
    confirmed, _ = confirm_reference(a, b, {"run_id": "r1"}, "intent", workdir=both_bad, task_id="t1")
    assert confirmed is False
    assert tasks_set_aside(both_bad)[0]["reason"] == "reference_unconfirmed"

    # the list of Tasks set aside only grows, in the order they were set aside
    stored = workdir / "stored"
    stored.mkdir()
    first = JudgeResult(use="reference", verdict="good_reference", judge="a")
    second = JudgeResult(use="reference", verdict="bad_reference", judge="b")
    set_task_aside(stored, "t1", "reference_disputed", first, second)
    set_task_aside(stored, "t2", "reference_unconfirmed", first, second)
    assert [row["task_id"] for row in tasks_set_aside(stored)] == ["t1", "t2"]


def test_two_abstaining_judges_leave_the_reference_unconfirmed(make_test_model, workdir):
    """D92: an item neither judge decided is exactly what a person has to see, so it is queued too."""
    a = named_judge(make_test_model, "a", "abstain")
    b = named_judge(make_test_model, "b", "abstain")
    confirmed, result = confirm_reference(a, b, {"run_id": "r1"}, "intent", workdir=workdir, task_id="t1")
    assert confirmed is False
    assert tasks_set_aside(workdir)[0]["reason"] == "reference_unconfirmed"
    queue = read_disagreement_queue(workdir)
    assert len(queue) == 1
    assert queue[0]["reason"] == "agreed_abstain"
    assert queue[0]["disagreement"] is False


# --- helpers ---


def judge_of(model, tools=None) -> AgenticJudge:
    return AgenticJudge(model, TOOLS if tools is None else tools, name="a")


def named_judge(make_test_model, name: str, verdict: str) -> AgenticJudge:
    model = make_test_model([call(), answer(verdict=verdict, cited_spans=[f"by {name}"], sub_answers=[yes()])])
    return AgenticJudge(model, TOOLS, name=name)


# --- D92: an item neither judge decided goes to the queue too, under its own label ---


def test_two_judges_that_agree_on_abstain_or_both_refuse_are_queued_as_such(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "undetermined")
    b = named_judge(make_test_model, "b", "undetermined")
    result, disagreement = two_judges(a, b, "judge_cause", {"run_id": "r2"}, {"run_id": "r1"},
                                      workdir=workdir, item_id="r2")
    assert result.verdict == "undetermined"
    assert disagreement is False
    queue = read_disagreement_queue(workdir)
    assert len(queue) == 1
    assert queue[0]["reason"] == "agreed_abstain"
    assert queue[0]["item_id"] == "r2"
    assert disagreement_rate(workdir) == {"pairs": 1, "disagreements": 0, "rate": 0.0,
                                          "abstains": 1, "abstain_rate": 1.0,
                                          "by_pair": {"a vs b": {"pairs": 1, "disagreements": 0,
                                                                 "rate": 0.0}}}

    # The harness could prefill no check for either judge, so neither verdict counts (D222).
    refused = workdir / "refused"
    refused.mkdir()
    a = AgenticJudge(make_test_model([answer(verdict="acceptable")]), {}, name="a")
    b = AgenticJudge(make_test_model([answer(verdict="unacceptable")]), {}, name="b")
    result, disagreement = two_judges(a, b, lambda judge: judge.ask("dispute", "decide this", {}),
                                      workdir=refused, item_id="x")
    assert result.verdict == "abstain"
    assert disagreement is False
    assert [row["reason"] for row in read_disagreement_queue(refused)] == ["refused"]


# --- the seam verdict.py needs: judge atoms and the failure cause, per Run (D76, D88) ---


def test_judge_atom_results_answers_every_judge_atom_of_a_verifier(make_test_model, workdir):
    from kullback.runner.judge import judge_atom_results
    from kullback.runner.records import Atom, Verifier

    verifier = Verifier(task_id="t1", atoms=[
        Atom(id="a_write", kind="required", predicate_src='wrote("cancel_pending_order")'),
        Atom(id="a_polite", kind="required", judge=True, description="the tone matched the policy"),
    ])
    a = named_judge(make_test_model, "a", "pass")
    b = named_judge(make_test_model, "b", "pass")
    results = judge_atom_results(verifier, TRANSCRIPT, a, b, workdir=workdir, run_id="r1")
    assert list(results) == ["a_polite"]
    assert results["a_polite"].verdict == "pass"
    assert results["a_polite"].use == "policy_atom"
    prompt = a.model.calls[0]["messages"][1]["content"]
    assert "the tone matched the policy" in prompt


def test_judge_cause_result_names_the_cause_for_one_failed_run(make_test_model, workdir):
    from kullback.runner.judge import judge_cause_result

    a = named_judge(make_test_model, "a", "environment")
    b = named_judge(make_test_model, "b", "environment")
    result = judge_cause_result({"run_id": "r2"}, {"run_id": "r1"}, a, b, workdir=workdir, run_id="r2")
    assert result.verdict == "environment"
    assert result.use == "cause"


# --- judge-smoke: can this model be a judge at all (D222 rule 4) ---


@pytest.mark.parametrize("replies, outcomes, routes, agrees, summary", [
    ([answer(verdict="equivalent", evidence=["value_a", "value_b"]),
      answer(verdict="not_equivalent", cited_spans=["gloss white"])],
     ["resolved", "resolved"], ["judge", "judge"], [True, True], "resolved 2 of 2, right on 2 of 2"),
    ([answer(verdict="equivalent"), answer(verdict="not_equivalent")],
     ["refused", "resolved"], ["uncited", "judge"], None, "resolved 1 of 2, right on 1 of 2"),
    ([answer(verdict="equivalent", evidence=["value_a"]), answer(verdict="not_equivalent")],
     ["resolved", "resolved"], ["judge", "judge"], None, None),
], ids=["cites_the_check_it_rests_on", "equal_answer_cites_nothing", "cites_one_value"])
def test_judge_smoke_asks_one_equal_and_one_different_pair_and_resolves_only_cited_agreement(
    make_test_model, replies, outcomes, routes, agrees, summary
):
    rows = smoke(AgenticJudge(make_test_model(replies), {}, name="a"))
    assert [row["outcome"] for row in rows] == outcomes
    assert [row["route"] for row in rows] == routes
    if agrees is not None:
        assert [row["agrees"] for row in rows] == agrees
    if summary is not None:
        assert smoke_lines(rows)[-1] == summary
    # two invented pairs, one the same and one not
    assert [row["expected"] for row in rows] == ["equal", "different"]
    assert rows[0]["a"] != rows[0]["b"] and rows[1]["a"] != rows[1]["b"]
    assert all(row["checks"] == 2 for row in rows)  # value_a and value_b, prefilled either way


