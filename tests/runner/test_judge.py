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
    count_judgement,
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


def test_policy_atom_passes_when_every_sub_answer_is_yes(make_test_model):
    model = make_test_model(
        [
            call(order_id="o1"),
            answer(
                verdict="pass",
                cited_spans=["I will cancel order o1"],
                sub_answers=[yes(), yes("did the user answer yes before the write")],
            ),
        ]
    )
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == "pass"
    assert result.use == "policy_atom"
    assert result.tools_run == ["read_order"]
    assert result.cited_spans == ["I will cancel order o1"]
    assert result.judge_version == JUDGE_VERSION
    assert result.refused is False
    assert result.abstained is False


@pytest.mark.parametrize(
    "sub_answers, expected",
    [
        ([yes(), {"question": "q", "answer": "no"}], "fail"),
        ([yes(), {"question": "q", "answer": "No!"}], "fail"),
        ([{"question": "q", "answer": "Yes."},
          {"question": "q2", "answer": "yes, the agent asked first"}], "pass"),
        ([{"question": "q", "answer": "abstain"}], "abstain"),
        ([{"question": "q", "answer": "maybe"}], "abstain"),
    ],
    ids=["a_plain_no", "a_no_with_punctuation", "a_yes_with_punctuation_or_a_tail",
         "an_unanswered_sub_question", "a_word_that_is_neither"],
)
def test_the_policy_atom_verdict_is_read_off_the_sub_answers_however_they_are_spelled(
    make_test_model, sub_answers, expected
):
    """Every row scripts the model's own verdict word as "pass", so the verdict can only come from
    the sub-answers (D76, R27 section 8b)."""
    model = make_test_model([call(), answer(verdict="pass", sub_answers=sub_answers)])
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == expected


def test_policy_atom_abstains_when_the_model_gives_no_sub_answers(make_test_model):
    model = make_test_model([call(), answer(verdict="pass")])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "abstain"
    assert "sub-answer" in (result.reason or "")


def test_policy_atom_prompt_carries_the_one_rule_and_the_transcript(make_test_model):
    model = make_test_model([call(), answer(verdict="pass", sub_answers=[yes()])])
    judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    prompt = model.calls[0]["messages"][1]["content"]
    assert "confirm before cancelling" in prompt
    assert TRANSCRIPT in prompt
    assert "sub_answers" in prompt


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


def test_the_prompt_carries_every_prefilled_check_with_its_tool_name_and_its_result(make_test_model):
    model = make_test_model([answer(verdict="pass", sub_answers=[yes()])])
    judge_of(model).judge_policy_atom("cancel only what the user named", TRANSCRIPT)
    prompt = model.calls[0]["messages"][1]["content"]
    assert "Checks already run for you" in prompt
    assert "- policy_rule {} ->" in prompt
    assert "- transcript {} ->" in prompt
    # Both tools take nothing the model has to choose, so the harness opened both views itself.
    assert '- read_order {} -> {"order_id":"o1"' in prompt
    assert "- list_writes {} ->" in prompt


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


def test_the_round_counts_what_was_prefilled_forced_and_refused(make_test_model):
    inner = make_test_model([answer(verdict="acceptable", cited_spans=["x"])])
    judge = AgenticJudge(RefusesToolChoice(inner), {"read_row": read_row, "list_writes": list_writes},
                         name="a")
    counts = count_judgement(judge.judge_dispute({"orders": {}}, [], []))
    assert counts["judge_checks_prefilled"] == 4  # end_state, required_set, allowed_set, list_writes
    assert counts["judge_tool_choice_rejected"] == 1
    assert counts["judge_tool_choice_forced"] == 0
    assert counts["judge_refused_no_check"] == 0
    assert counts["judge_extra_tool_calls"] == 0


def test_a_question_with_no_check_to_prefill_is_refused_as_a_harness_bug(make_test_model):
    model = make_test_model([answer(verdict="acceptable")])
    result = judge_of(model, tools={}).ask("dispute", "decide this", {})
    assert result.verdict == "abstain"
    assert result.refused is True
    assert result.checks == []
    assert "D222" in (result.reason or "")


def test_a_call_to_an_unknown_tool_is_not_counted_and_its_error_goes_back_to_the_judge(make_test_model):
    model = make_test_model([call(name="drop_table"), answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "pass"
    assert result.tools_run == []
    tool_reply = model.calls[1]["messages"][-1]
    assert tool_reply["role"] == "tool"
    assert "tool_not_found" in tool_reply["content"]


def test_a_tool_that_raises_does_not_count_and_its_error_goes_back_to_the_judge(make_test_model):
    def boom() -> dict:
        """Always fails."""
        raise ValueError("no such table")

    model = make_test_model([call(name="boom"), call(order_id="o1"), answer(verdict="pass", sub_answers=[yes()])])
    judge = AgenticJudge(model, {"boom": boom, "read_order": read_order})
    result = judge.judge_policy_atom("rule", TRANSCRIPT)
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


def test_tool_results_are_kept_for_the_report(make_test_model):
    model = make_test_model([call(order_id="o1"), answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.tool_results == [
        {"name": "read_order", "args": {"order_id": "o1"}, "result": read_order("o1")}
    ]


# --- semantic equivalence (D84, R27 section 8a) ---


def test_equivalence_shows_the_column_the_field_type_and_both_values(make_test_model):
    model = make_test_model([call(), answer(verdict="not_equivalent", cited_spans=["wrong size"])])
    result = judge_of(model).judge_equivalence(
        "orders.reason", "no longer needed", "wrong size", field_type="free-text category"
    )
    assert result.verdict == "not_equivalent"
    assert result.use == "equivalence"
    prompt = model.calls[0]["messages"][1]["content"]
    assert "orders.reason" in prompt
    assert "free-text category" in prompt
    assert "no longer needed" in prompt
    assert "wrong size" in prompt


def test_equivalence_abstains_when_the_judge_flags_a_number_or_unit_mismatch(make_test_model):
    model = make_test_model(
        [call(), answer(verdict="equivalent", flags=["number mismatch"], cited_spans=["25", "25.5"])]
    )
    result = judge_of(model).judge_equivalence("orders.total", "25", "25.5", field_type="currency amount")
    assert result.verdict == "abstain"
    assert "mismatch" in (result.reason or "")


def test_an_unknown_verdict_word_abstains(make_test_model):
    model = make_test_model([call(), answer(verdict="probably fine")])
    result = judge_of(model).judge_equivalence("orders.reason", "a", "b")
    assert result.verdict == "abstain"
    assert "probably fine" in (result.reason or "")


def test_a_reply_that_is_not_json_abstains(make_test_model):
    model = make_test_model([call(), {"content": "I think the two values mean the same thing."}])
    result = judge_of(model).judge_equivalence("orders.reason", "a", "b")
    assert result.verdict == "abstain"
    assert result.refused is False


def test_json_wrapped_in_prose_is_still_read(make_test_model):
    reply = ('Here is my answer:\n{"verdict": "equivalent", "evidence": ["value_a", "value_b"]}\n'
             "That is all.")
    model = make_test_model([call(), {"content": reply}])
    assert judge_of(model).judge_equivalence("orders.reason", "a", "b").verdict == "equivalent"


def test_an_equal_answer_that_names_no_check_is_unresolved_rather_than_agreement(make_test_model):
    """D222 rule 3: an agreement rests on a named check, or nobody settled the pair (D219)."""
    model = make_test_model([answer(verdict="equivalent", cited_spans=["they read alike"])])
    result = judge_of(model).judge_equivalence("orders.reason", "a", "b")
    assert result.verdict == "abstain"
    assert (result.reason or "").startswith("uncited")


def test_an_equal_answer_naming_a_prefilled_check_stands(make_test_model):
    model = make_test_model([answer(verdict="equivalent", evidence=["value_a"])])
    result = judge_of(model).judge_equivalence("orders.reason", "a", "a ")
    assert result.verdict == "equivalent"
    assert result.reason != "uncited: the agreeing verdict named no check it rests on"


def test_a_different_answer_needs_no_citation_because_only_agreement_carries_the_rule(make_test_model):
    model = make_test_model([answer(verdict="not_equivalent")])
    assert judge_of(model).judge_equivalence("orders.reason", "a", "b").verdict == "not_equivalent"


# --- reference confirmation (D57, Trust or Escalate) ---


def test_reference_judge_sees_the_intent_and_the_verifier_output(make_test_model):
    model = make_test_model([call(), answer(verdict="good_reference", cited_spans=["cancelled it"])])
    judge = AgenticJudge(model, TOOLS, verifier_output={"pass": True, "atoms": 3})
    result = judge.judge_reference({"run_id": "r1"}, "cancel order o1 and say why")
    assert result.verdict == "good_reference"
    assert result.use == "reference"
    system = model.calls[0]["messages"][0]["content"]
    assert "verifier" in system.lower()
    assert '"atoms":3' in system
    prompt = model.calls[0]["messages"][1]["content"]
    assert "cancel order o1 and say why" in prompt
    assert "escalate" in prompt.lower()


def test_the_reference_judge_is_told_the_transcript_is_not_in_evidence_and_to_grade_the_intent(make_test_model):
    """Build 8: eleven Tasks failed 'without evidence of authentication and confirmation' the judge could not see,
    and three were graded on the opening request after the user had changed their mind."""
    model = make_test_model([call(), answer(verdict="abstain")])
    judge = AgenticJudge(model, TOOLS)
    judge.judge_reference({"run_id": "r1"}, "exchange the desk lamp only")
    prompt = model.calls[0]["messages"][1]["content"].lower()
    assert "transcript is not in evidence" in prompt
    assert "authenticated" in prompt and "confirmation" in prompt
    assert "not the opening request" in prompt
    assert "end state" in prompt


# --- a ruling may rest only on what the judge was handed (D93, build 8's hand fix 3) ---


def test_the_system_prompt_names_the_sources_this_use_has_and_says_the_transcript_is_not_one(make_test_model):
    model = make_test_model([call(), answer(verdict="good_reference")])
    AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "cancel order o1")
    system = model.calls[0]["messages"][0]["content"]
    assert "intent, verifier_output, end_state, state_tools" in system
    assert "You do not have the transcript." in system
    assert "authenticated the user" in system and "confirmation" in system
    assert '"evidence"' in system


@pytest.mark.parametrize("evidence", [["transcript"], "transcript", ["end_state", "the transcript"]],
                         ids=["a_list", "one_string", "beside_a_source_it_does_have"])
def test_a_reference_judge_that_rests_on_the_transcript_abstains_rather_than_failing_the_run(
    make_test_model, evidence
):
    """Build 8: eleven Tasks were failed "without evidence of the required authentication and explicit
    confirmation" by a judge that was handed no transcript, where D93 says a person decides."""
    model = make_test_model([call(), answer(
        verdict="bad_reference", evidence=evidence,
        reason="no evidence of the required authentication and explicit confirmation")])
    result = AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "cancel order o1")
    assert result.verdict == "abstain"
    assert result.abstained is True
    assert result.refused is False
    assert result.evidence[-1].endswith("transcript")
    assert "transcript" in (result.reason or "") and "was not given" in (result.reason or "")


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


def test_a_reference_judge_that_rests_on_what_it_was_handed_still_fails_the_run(make_test_model):
    model = make_test_model([call(), answer(
        verdict="bad_reference", evidence=["Intent", "End states"],
        reason="an order the Intent never names was exchanged")])
    result = AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "exchange the desk lamp only")
    assert result.verdict == "bad_reference"
    assert result.evidence == ["Intent", "End states"]


def test_a_judge_handed_the_transcript_may_rest_its_verdict_on_it(make_test_model):
    model = make_test_model([call(), answer(
        verdict="pass", evidence=["transcript", "state_tools"], sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == "pass"
    assert "transcript" not in (result.reason or "")


def test_a_pass_shaped_ruling_that_rests_on_a_source_it_was_not_given_abstains_too(make_test_model):
    """The rule never turns into a pass either: D110 holds in both directions."""
    model = make_test_model([call(), answer(
        verdict="pass", evidence=["crm_export"], sub_answers=[yes(), yes()])])
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == "abstain"
    assert "crm_export" in (result.reason or "")


def test_a_reply_with_no_evidence_field_is_read_as_it_was_before(make_test_model):
    model = make_test_model([call(), answer(verdict="bad_reference", reason="nothing was written")])
    result = AgenticJudge(model, TOOLS).judge_reference({"run_id": "r1"}, "cancel order o1")
    assert result.verdict == "bad_reference" and result.evidence == []


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


def test_cause_abstains_as_undetermined(make_test_model):
    model = make_test_model([call(), answer(verdict="undetermined")])
    result = judge_of(model).judge_cause({"run_id": "r2"}, {"run_id": "r1"})
    assert result.verdict == "undetermined"
    assert result.abstained is True


# --- dispute path (R27 section 8d) ---


def test_dispute_shows_the_end_state_and_the_required_and_allowed_sets(make_test_model):
    model = make_test_model([call(), answer(verdict="acceptable", cited_spans=["status cancelled"])])
    result = judge_of(model).judge_dispute(
        {"orders": {"o1": "cancelled"}}, ["orders.o1.status == cancelled"], ["orders.o1.reason"]
    )
    assert result.verdict == "acceptable"
    assert result.use == "dispute"
    prompt = model.calls[0]["messages"][1]["content"]
    assert "orders.o1.status == cancelled" in prompt
    assert "orders.o1.reason" in prompt
    assert "change your mind" in prompt


def test_a_persona_reaches_the_system_prompt(make_test_model):
    model = make_test_model([call(), answer(verdict="acceptable")])
    judge = AgenticJudge(model, TOOLS, persona="a strict auditor", name="b")
    judge.judge_dispute({}, [], [])
    assert "a strict auditor" in model.calls[0]["messages"][0]["content"]


# --- two judges and the disagreement queue (D92) ---


def test_two_judges_that_agree_report_no_disagreement(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "acceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, item_id="run1")
    assert disagreement is False
    assert result.verdict == "acceptable"
    assert result.judge == "a"
    assert [entry["verdict"] for entry in result.pair] == ["acceptable", "acceptable"]
    assert read_disagreement_queue(workdir) == []
    assert disagreement_rate(workdir) == {"pairs": 1, "disagreements": 0, "rate": 0.0,
                                          "abstains": 0, "abstain_rate": 0.0,
                                          "by_pair": {"a vs b": {"pairs": 1, "disagreements": 0,
                                                                 "rate": 0.0}}}


def test_two_judges_that_disagree_queue_both_verdicts_and_abstain(make_test_model, workdir):
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


def test_two_judges_accepts_an_unbound_method(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "environment")
    b = named_judge(make_test_model, "b", "environment")
    result, disagreement = two_judges(
        a, b, AgenticJudge.judge_cause, {"run_id": "r2"}, {"run_id": "r1"}, workdir=workdir
    )
    assert disagreement is False
    assert result.verdict == "environment"


def test_a_split_without_a_workdir_is_still_returned_as_an_abstain_with_both_verdicts(make_test_model):
    """Nowhere to queue the split is not a reason to pick a side: the caller still gets the abstain."""
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


def test_a_third_sample_settles_a_split_on_a_non_reference_atom(make_test_model, workdir):
    """D97: the two judges split, a third sample from judge A's model decides by majority."""
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    c = named_judge(make_test_model, "c", "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="run1", judge_c=c)
    assert disagreement is False
    assert result.verdict == "unacceptable"
    assert [entry["verdict"] for entry in result.pair] == ["acceptable", "unacceptable", "unacceptable"]
    assert read_disagreement_queue(workdir) == []


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


def test_a_row_written_before_the_judges_were_named_joins_no_pair(workdir):
    """An older build's rows say nothing about which two models ran, so they count in no pair."""
    (workdir / "judge_pairs.jsonl").write_text(
        json.dumps({"use": "dispute", "disagreement": True}) + "\n", encoding="utf-8")
    stats = disagreement_rate(workdir)
    assert stats["pairs"] == 1 and stats["rate"] == 1.0
    assert stats["by_pair"] == {}


def test_a_three_way_split_still_abstains_to_the_queue(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    c = named_judge(make_test_model, "c", "abstain")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="run1", judge_c=c)
    assert disagreement is True
    assert result.verdict == "abstain"
    assert len(read_disagreement_queue(workdir)) == 1


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


def test_a_disputed_reference_sets_the_task_aside_and_no_third_judge_runs(make_test_model, workdir):
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


def test_a_bad_reference_both_judges_agree_on_sets_the_task_aside(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "bad_reference")
    b = named_judge(make_test_model, "b", "bad_reference")
    confirmed, _ = confirm_reference(a, b, {"run_id": "r1"}, "intent", workdir=workdir, task_id="t1")
    assert confirmed is False
    assert tasks_set_aside(workdir)[0]["reason"] == "reference_unconfirmed"


def test_set_task_aside_is_appendable_and_readable(workdir):
    first = JudgeResult(use="reference", verdict="good_reference", judge="a")
    second = JudgeResult(use="reference", verdict="bad_reference", judge="b")
    set_task_aside(workdir, "t1", "reference_disputed", first, second)
    set_task_aside(workdir, "t2", "reference_unconfirmed", first, second)
    assert [row["task_id"] for row in tasks_set_aside(workdir)] == ["t1", "t2"]


def test_queues_are_empty_before_anything_is_written(workdir):
    assert read_disagreement_queue(workdir) == []
    assert tasks_set_aside(workdir) == []
    assert disagreement_rate(workdir) == {"pairs": 0, "disagreements": 0, "rate": 0.0,
                                          "abstains": 0, "abstain_rate": 0.0, "by_pair": {}}


# --- helpers ---


def judge_of(model, tools=None) -> AgenticJudge:
    return AgenticJudge(model, TOOLS if tools is None else tools, name="a")


def named_judge(make_test_model, name: str, verdict: str) -> AgenticJudge:
    model = make_test_model([call(), answer(verdict=verdict, cited_spans=[f"by {name}"], sub_answers=[yes()])])
    return AgenticJudge(model, TOOLS, name=name)


# --- D92: an item neither judge decided goes to the queue too, under its own label ---


def test_two_judges_that_agree_on_abstain_are_queued_as_such(make_test_model, workdir):
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


def test_two_refused_judges_are_queued_as_refused(make_test_model, workdir):
    """The harness could prefill no check for either judge, so neither verdict counts (D222)."""
    a = AgenticJudge(make_test_model([answer(verdict="acceptable")]), {}, name="a")
    b = AgenticJudge(make_test_model([answer(verdict="unacceptable")]), {}, name="b")
    result, disagreement = two_judges(a, b, lambda judge: judge.ask("dispute", "decide this", {}),
                                      workdir=workdir, item_id="x")
    assert result.verdict == "abstain"
    assert disagreement is False
    assert [row["reason"] for row in read_disagreement_queue(workdir)] == ["refused"]


def test_an_abstain_majority_from_the_third_sample_is_queued(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "abstain")
    c = named_judge(make_test_model, "c", "abstain")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir,
                                      item_id="x", judge_c=c)
    assert result.verdict == "abstain"
    assert [row["reason"] for row in read_disagreement_queue(workdir)] == ["abstain_majority"]


def test_a_third_sample_that_cannot_answer_leaves_the_split_in_the_queue(make_test_model, workdir):
    """The default third sample reuses judge A's model, which may have nothing left to say."""
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, item_id="x")
    assert disagreement is True
    assert result.verdict == "abstain"
    assert len(read_disagreement_queue(workdir)) == 1
    assert [row["reason"] for row in read_disagreement_queue(workdir)] == ["split"]
    third = result.pair[2]
    assert third["refused"] is True
    assert "did not answer" in third["reason"]


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


def test_judge_smoke_reports_resolved_for_a_model_that_cites_the_check_it_rests_on(make_test_model):
    model = make_test_model([answer(verdict="equivalent", evidence=["value_a", "value_b"]),
                             answer(verdict="not_equivalent", cited_spans=["gloss white"])])
    rows = smoke(AgenticJudge(model, {}, name="a"))
    assert [row["outcome"] for row in rows] == ["resolved", "resolved"]
    assert [row["route"] for row in rows] == ["judge", "judge"]
    assert [row["agrees"] for row in rows] == [True, True]
    assert smoke_lines(rows)[-1] == "resolved 2 of 2, right on 2 of 2"


def test_judge_smoke_reports_refused_for_a_model_whose_equal_answer_cites_nothing(make_test_model):
    model = make_test_model([answer(verdict="equivalent"),
                             answer(verdict="not_equivalent")])
    rows = smoke(AgenticJudge(model, {}, name="a"))
    assert [row["outcome"] for row in rows] == ["refused", "resolved"]
    assert [row["route"] for row in rows] == ["uncited", "judge"]
    assert smoke_lines(rows)[-1] == "resolved 1 of 2, right on 1 of 2"


def test_judge_smoke_asks_two_invented_pairs_one_the_same_and_one_not(make_test_model):
    model = make_test_model([answer(verdict="equivalent", evidence=["value_a"]),
                             answer(verdict="not_equivalent")])
    rows = smoke(AgenticJudge(model, {}, name="a"))
    assert [row["expected"] for row in rows] == ["equal", "different"]
    assert rows[0]["a"] != rows[0]["b"] and rows[1]["a"] != rows[1]["b"]
    assert all(row["checks"] == 2 for row in rows)  # value_a and value_b, prefilled either way
