"""Tests for the agentic judge: a tool check before every verdict, two judges, the disagreement queue."""

from __future__ import annotations

import json

from harness.runner.judge import (
    JUDGE_VERSION,
    AgenticJudge,
    JudgeResult,
    abstain_verdict,
    confirm_reference,
    disagreement_rate,
    read_disagreement_queue,
    set_task_aside,
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


TOOLS = {"read_order": read_order, "list_writes": list_writes}


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


def test_policy_atom_verdict_is_computed_from_the_sub_answers_not_taken_from_the_model(make_test_model):
    model = make_test_model(
        [call(), answer(verdict="pass", sub_answers=[yes(), {"question": "q", "answer": "no"}])]
    )
    result = judge_of(model).judge_policy_atom("confirm before cancelling", TRANSCRIPT)
    assert result.verdict == "fail"


def test_policy_atom_abstains_when_a_sub_question_is_unanswered(make_test_model):
    model = make_test_model([call(), answer(verdict="pass", sub_answers=[{"question": "q", "answer": "abstain"}])])
    assert judge_of(model).judge_policy_atom("rule", TRANSCRIPT).verdict == "abstain"


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


# --- the tool rule (D92: at least one tool check before a verdict) ---


def test_a_verdict_with_no_tool_call_is_refused(make_test_model):
    model = make_test_model([answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "abstain"
    assert result.refused is True
    assert result.tools_run == []


def test_a_call_to_an_unknown_tool_does_not_count_as_a_check(make_test_model):
    model = make_test_model([call(name="drop_table"), answer(verdict="pass", sub_answers=[yes()])])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.refused is True
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
    model = make_test_model([call(), answer(verdict="equivalent", flags=["number mismatch"], cited_spans=["25", "25.5"])])
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
    model = make_test_model([call(), {"content": 'Here is my answer:\n{"verdict": "equivalent"}\nThat is all.'}])
    assert judge_of(model).judge_equivalence("orders.reason", "a", "b").verdict == "equivalent"


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
                                          "abstains": 0, "abstain_rate": 0.0}


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
    assert disagreement_rate(workdir)["rate"] == 1.0


def test_two_judges_accepts_an_unbound_method(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "environment")
    b = named_judge(make_test_model, "b", "environment")
    result, disagreement = two_judges(
        a, b, AgenticJudge.judge_cause, {"run_id": "r2"}, {"run_id": "r1"}, workdir=workdir
    )
    assert disagreement is False
    assert result.verdict == "environment"


def test_two_judges_without_a_workdir_writes_nothing(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], third_sample=False)
    assert disagreement is True
    assert list(workdir.iterdir()) == []


def test_the_disagreement_rate_can_be_read_for_one_use(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, third_sample=False)
    c = named_judge(make_test_model, "c", "environment")
    d = named_judge(make_test_model, "d", "environment")
    two_judges(c, d, "judge_cause", {}, {}, workdir=workdir)
    assert disagreement_rate(workdir) == {"pairs": 2, "disagreements": 1, "rate": 0.5,
                                          "abstains": 0, "abstain_rate": 0.0}
    assert disagreement_rate(workdir, use="cause") == {"pairs": 1, "disagreements": 0,
                                                       "rate": 0.0, "abstains": 0,
                                                       "abstain_rate": 0.0}


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
                                          "abstains": 0, "abstain_rate": 0.0}


# --- helpers ---


def judge_of(model) -> AgenticJudge:
    return AgenticJudge(model, TOOLS, name="a")


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
                                          "abstains": 1, "abstain_rate": 1.0}


def test_two_refused_judges_are_queued_as_refused(make_test_model, workdir):
    """Neither judge ran a tool, so neither verdict counts and nobody has decided the item."""
    a = AgenticJudge(make_test_model([answer(verdict="pass")]), TOOLS, name="a")
    b = AgenticJudge(make_test_model([answer(verdict="fail")]), TOOLS, name="b")
    result, disagreement = two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, item_id="x")
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


def test_a_split_is_queued_under_its_own_reason(make_test_model, workdir):
    a = named_judge(make_test_model, "a", "acceptable")
    b = named_judge(make_test_model, "b", "unacceptable")
    two_judges(a, b, "judge_dispute", {}, [], [], workdir=workdir, item_id="x", third_sample=False)
    queue = read_disagreement_queue(workdir)
    assert [row["reason"] for row in queue] == ["split"]
    assert queue[0]["disagreement"] is True


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


# --- sub-answers as people and models write them (R27 8b) ---


def test_a_sub_answer_with_punctuation_or_a_tail_still_counts(make_test_model):
    model = make_test_model([
        call(),
        answer(verdict="pass", sub_answers=[{"question": "q", "answer": "Yes."},
                                            {"question": "q2", "answer": "yes, the agent asked first"}]),
    ])
    result = judge_of(model).judge_policy_atom("rule", TRANSCRIPT)
    assert result.verdict == "pass"


def test_a_no_with_punctuation_still_fails_the_atom(make_test_model):
    model = make_test_model([call(), answer(verdict="pass", sub_answers=[yes(), {"question": "q", "answer": "No!"}])])
    assert judge_of(model).judge_policy_atom("rule", TRANSCRIPT).verdict == "fail"


def test_a_sub_answer_that_is_neither_still_abstains(make_test_model):
    model = make_test_model([call(), answer(verdict="pass", sub_answers=[{"question": "q", "answer": "maybe"}])])
    assert judge_of(model).judge_policy_atom("rule", TRANSCRIPT).verdict == "abstain"


# --- the seam verdict.py needs: judge atoms and the failure cause, per Run (D76, D88) ---


def test_judge_atom_results_answers_every_judge_atom_of_a_verifier(make_test_model, workdir):
    from harness.runner.judge import judge_atom_results
    from harness.shared.records import Atom, Verifier

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
    from harness.runner.judge import judge_cause_result

    a = named_judge(make_test_model, "a", "environment")
    b = named_judge(make_test_model, "b", "environment")
    result = judge_cause_result({"run_id": "r2"}, {"run_id": "r1"}, a, b, workdir=workdir, run_id="r2")
    assert result.verdict == "environment"
    assert result.use == "cause"
