"""reference.py: which Runs a Verifier may be derived from (D111), re-rolls (D112) and demoted constraints (D76)."""

from __future__ import annotations

from gates.verifier_fixtures import assistant, call, make_run, result, user
from kullback.ai.provider import TestModel
from kullback.examiner import reference as ref
from kullback.runner.records import Constraint

WRITES = {"cancel_pending_order", "exchange_delivered_order_items"}


def _canon(value):
    return value


def cancel_run(run_id: str, order: str = "#W123", kind: str = ref.RECORDING) -> ref.Recording:
    run = make_run(run_id, [
        user(f"Please cancel my order {order}."),
        call("cancel_pending_order", {"order_id": order, "reason": "no longer needed"}),
        result({"order_id": order, "status": "cancelled"}),
        assistant("Done."),
    ])
    return ref.Recording(run_id=run_id, path=run_id, kind=kind, trace_id=run_id if kind == ref.RECORDING else None,
                         end_state=ref.end_state(run, WRITES, _canon))


def empty_run(run_id: str, kind: str = ref.RECORDING) -> ref.Recording:
    run = make_run(run_id, [user("Please cancel my order #W123."), assistant("I cannot do that.")])
    return ref.Recording(run_id=run_id, path=run_id, kind=kind, end_state=ref.end_state(run, WRITES, _canon))


# --- End states -------------------------------------------------------------

def test_two_runs_that_wrote_the_same_thing_share_an_end_state():
    assert cancel_run("a").end_state == cancel_run("b").end_state
    assert cancel_run("a").end_state != cancel_run("c", order="#W999").end_state
    assert empty_run("d").end_state == ()


def test_describe_names_the_tool_the_entity_and_the_values():
    text = ref.describe(cancel_run("a").end_state)
    assert text.startswith("cancel_pending_order on #W123") and "reason=no longer needed" in text
    assert ref.describe(()) == "no writes; the answer states nothing read from the world"
    assert ref.describe(ref.ANSWERED) == "no writes; the answer states facts read from the world"


# --- End states of a Run that wrote nothing ---------------------------------
# A public library: the reader asks about a loan, the agent looks it up and either states what it
# read or refuses. Neither Run writes, and before this the two were one group.

LIBRARY_WRITES = {"renew_loan"}
LOAN = {"loan_id": "LB-4412", "title": "The Long Ships", "due": "2026-09-20", "fine": 0}


def _library_recording(run_id: str, answer: str, *, renewed: bool = False,
                       kind: str = ref.RECORDING) -> ref.Recording:
    events = [user("When is my copy of The Long Ships due back?"),
              call("get_loan", {"loan_id": "LB-4412"}, kind="read", cid="c0"),
              result(LOAN, cid="c0")]
    if renewed:
        events += [call("renew_loan", {"loan_id": "LB-4412", "weeks": 2}, cid="c1"),
                   result({"loan_id": "LB-4412", "due": "2026-10-04"}, cid="c1")]
    events.append(assistant(answer))
    run = make_run(run_id, events)
    return ref.Recording(run_id=run_id, path=run_id, kind=kind,
                         trace_id=run_id if kind == ref.RECORDING else None,
                         end_state=ref.end_state(run, LIBRARY_WRITES, _canon))


def answered(run_id: str, answer: str = "Loan LB-4412 is due on 2026-09-20.", **kw) -> ref.Recording:
    return _library_recording(run_id, answer, **kw)


def refused(run_id: str, **kw) -> ref.Recording:
    return _library_recording(run_id, "I am unable to verify your library card, so I cannot look that up.", **kw)


def test_a_no_write_recording_that_stated_facts_and_one_that_stated_nothing_are_two_end_states():
    assert answered("a").end_state == ref.ANSWERED
    assert refused("b").end_state == ()
    assert len(ref.group([answered("a"), refused("b")])) == 2


def test_two_answered_no_write_recordings_share_one_end_state():
    """The state says facts were stated, never which ones: which facts are common is derive's question."""
    other = answered("b", answer="It is due back on 2026-09-20; nothing is owed on LB-4412.")
    assert answered("a").end_state == other.end_state == ref.ANSWERED
    assert len(ref.group([answered("a"), other])) == 1


def test_a_recording_that_read_nothing_and_a_refusal_after_a_read_share_the_stated_nothing_state():
    read_nothing = ref.Recording(run_id="c", path="c", end_state=ref.end_state(
        make_run("c", [user("Is the library open on Sunday?"), assistant("I cannot check that.")]),
        LIBRARY_WRITES, _canon))
    assert read_nothing.end_state == refused("b").end_state == ()


def test_a_recording_with_writes_keeps_its_write_state_whatever_it_said():
    stated = answered("a", renewed=True)
    silent = refused("b", renewed=True)
    assert stated.end_state == silent.end_state != ref.ANSWERED
    assert ref.describe(stated.end_state).startswith("renew_loan on LB-4412")


def test_the_judge_fails_the_refusal_and_the_answered_recordings_are_the_references():
    """26 Tasks on the last build had no-write recordings in one group, so the judge was never called
    and refusals stayed References; the two states now reach it."""
    judge = TestModel(['{"failed": ["B"], "evidence": ["intent", "end_states"], '
                       '"reason": "the reader was never told when the book is due"}'])
    out = ref.confirm([answered("a"), answered("b"), refused("c")],
                      intent="tell the reader when loan LB-4412 is due",
                      policy_lines=["a reader may be told the due date of their own loan"], judge=judge)
    assert [r.run_id for r in out.references] == ["a", "b"]
    assert out.failed == {"c": "judge: the reader was never told when the book is due"}
    assert out.judged and not out.judge_abstained
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "A (2 runs): no writes; the answer states facts read from the world" in prompt
    assert "B (1 run): no writes; the answer states nothing read from the world" in prompt


# --- the rule ---------------------------------------------------------------

def test_recordings_that_agree_are_the_references():
    out = ref.confirm([cancel_run("a"), cancel_run("b")])
    assert [r.run_id for r in out.references] == ["a", "b"] and out.failed == {} and out.reason is None


def test_a_recording_that_broke_a_constraint_is_a_failed_recording():
    broken = cancel_run("b")
    broken.violated = ["c_pending"]
    out = ref.confirm([cancel_run("a"), broken])
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "violates c_pending"}


def test_two_end_states_and_no_judge_is_no_reference():
    out = ref.confirm([cancel_run("a"), empty_run("b")])
    assert out.references == [] and out.reason.startswith("recordings disagree on the End state (2 states")
    assert [g["label"] for g in out.groups] == ["A", "B"]


def test_the_judge_can_fail_a_state_and_the_other_one_becomes_the_reference():
    judge = TestModel(['{"failed": ["B"], "reason": "the cancellation the user asked for never happened"}'])
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel order #W123",
                      policy_lines=["pending orders may be cancelled"], judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: the cancellation the user asked for never happened"}
    assert out.judged and not out.judge_abstained
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "cancel order #W123" in prompt and "A (1 run): cancel_pending_order" in prompt


def test_the_judge_prompt_gives_the_intent_as_the_ground_truth_and_says_it_has_no_transcript():
    prompt = ref.judge_prompt("exchange the desk lamp only", ["exchanges need a confirmation"],
                              ref.group([cancel_run("a"), empty_run("b")]))
    assert "intent, policy, end_states" in prompt
    assert "You do not have the transcript." in prompt
    assert "authenticated the user" in prompt and "confirmation" in prompt
    assert "ground truth of what the user wanted by the end of the Run: exchange the desk lamp only" in prompt
    assert "never the opening request" in prompt


def test_a_judgement_that_rests_on_the_transcript_fails_no_state_and_is_recorded_as_an_abstention():
    """Build 8 failed eleven Tasks "without evidence of the required authentication and explicit
    confirmation", which no transcript was ever handed to the judge to show."""
    judge = TestModel(['{"failed": ["A", "B"], "evidence": ["transcript"], "reason": '
                       '"without evidence of the required authentication and explicit confirmation"}'])
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel order #W123", judge=judge)
    assert out.failed == {} and out.references == []
    assert out.judged and out.judge_abstained
    assert "transcript" in out.judge_reason and "was not given" in out.judge_reason
    assert out.reason.startswith("recordings disagree on the End state")
    assert "the judge abstained" in out.reason


def test_a_run_that_matches_a_revised_intent_is_not_failed_for_the_opening_request():
    """The Intent is what the user wanted by the end of the Run; the opening request was not handed
    to this judge and cannot fail the state that matches the Intent."""
    judge = TestModel(['{"failed": ["A"], "evidence": ["opening_request"], "reason": '
                       '"the user requested cancellations for both orders"}'])
    out = ref.confirm([cancel_run("a"), cancel_run("b", order="#W999")],
                      intent="cancel order #W123 only, leave #W999 alone", judge=judge)
    assert out.failed == {} and out.judge_abstained
    assert "opening_request" in out.judge_reason


def test_a_judgement_that_rests_on_what_the_judge_was_given_still_fails_a_state():
    judge = TestModel(['{"failed": ["B"], "evidence": ["Intent", "End states"], "reason": "nothing was written"}'])
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel order #W123", judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: nothing was written"} and not out.judge_abstained


def test_the_judge_never_awards_a_pass():
    """Failing nothing leaves the disagreement; failing everything leaves no Reference."""
    nothing = ref.confirm([cancel_run("a"), empty_run("b")], judge=TestModel(['{"failed": [], "reason": "cannot tell"}']))
    assert nothing.references == [] and nothing.reason.startswith("recordings disagree")
    everything = ref.confirm([cancel_run("a"), empty_run("b")], judge=TestModel(['{"failed": ["A", "B"]}']))
    assert everything.references == [] and everything.reason == "the judge failed every End state"


def test_an_unreadable_judge_reply_fails_nothing():
    for text in ("I think B is wrong", '{"failed": "B"}'):
        unreadable = ref.parse_judgement(text, {"A", "B"})
        assert unreadable.failed == set() and unreadable.reason == "unreadable reply"
        assert not unreadable.abstained
    read = ref.parse_judgement('sure: {"failed": ["b", "Z"], "reason": "x"}', {"A", "B"})
    assert read.failed == {"B"} and read.reason == "x"


def test_the_judge_is_not_called_when_the_recordings_agree():
    judge = TestModel([])  # raises if queried
    out = ref.confirm([cancel_run("a"), cancel_run("b")], judge=judge)
    assert len(out.references) == 2 and not out.judged and judge.calls == []


def test_a_recording_outranks_a_reroll_as_the_reference():
    out = ref.confirm([cancel_run("r1", kind=ref.REROLL), cancel_run("a"), cancel_run("r0", kind=ref.REROLL)])
    assert [r.run_id for r in out.references] == ["a", "r0", "r1"]
    assert out.references[0].kind == ref.RECORDING


def test_rerolls_alone_can_corroborate_or_contradict_a_single_recording():
    agree = ref.confirm([cancel_run("a"), cancel_run("r0", kind=ref.REROLL), cancel_run("r1", kind=ref.REROLL)])
    assert len(agree.references) == 3 and len(agree.groups) == 1
    contradict = ref.confirm([cancel_run("a"), empty_run("r0", kind=ref.REROLL)])
    assert contradict.references == [] and len(contradict.groups) == 2


def test_no_run_and_every_run_broken_have_their_own_reasons():
    assert ref.confirm([]).reason == "no Run to confirm"
    broken = cancel_run("a")
    broken.violated = ["c1"]
    assert ref.confirm([broken]).reason == "every recording broke a Hard constraint"


def test_every_group_gets_its_own_label_past_the_end_of_the_alphabet():
    """confirm() fails and keeps groups by label, so two groups under one label are one group."""
    runs = [cancel_run(f"r{i}", order=f"#W{i}") for i in range(30)]
    labels = [g["label"] for g in ref.group(runs)]
    assert len(set(labels)) == len(labels)
    assert labels[:2] == ["A", "B"] and labels[26] == "A1"


def test_the_judge_prompt_is_bounded():
    groups = ref.group([cancel_run("a"), empty_run("b")])
    prompt = ref.judge_prompt("x" * 5000, ["line %d" % i for i in range(200)], groups)
    assert len(prompt) < 12000 and "line 39" in prompt and "line 40" not in prompt


# --- constraints against the corpus -----------------------------------------

def _rule(rule_id: str, src: str) -> Constraint:
    return Constraint(id=rule_id, text=f"rule {rule_id}", compiled=True, predicate_src=src)


ALWAYS = "def check(pre_state, write_call, transcript):\n    return True\n"
NEVER = "def check(pre_state, write_call, transcript):\n    return False\n"


def _runs(n: int):
    return [make_run(f"run{i}", [call("cancel_pending_order", {"order_id": "#W1"}), result({"status": "cancelled"})])
            for i in range(n)]


def test_a_rule_the_confirmed_recordings_mostly_break_is_demoted_and_the_rest_kept():
    rules = [_rule("ok", ALWAYS), _rule("bad", NEVER)]
    rates = ref.constraint_rates(rules, _runs(4), WRITES, _canon)
    assert rates == {"ok": {"failed": 0, "runs": 4}, "bad": {"failed": 4, "runs": 4}}
    kept, demoted = ref.demote(rules, rates)
    assert [c.id for c in kept] == ["ok"]
    assert demoted[0]["id"] == "bad" and "4 of 4" in demoted[0]["reason"]


def test_too_few_recordings_demote_nothing():
    rules = [_rule("bad", NEVER)]
    rates = ref.constraint_rates(rules, _runs(2), WRITES, _canon)
    kept, demoted = ref.demote(rules, rates)
    assert [c.id for c in kept] == ["bad"] and demoted == []


def test_violations_name_the_constraints_a_run_breaks():
    atoms = ref.hard_atoms([_rule("ok", ALWAYS), _rule("bad", NEVER), Constraint(id="res", text="residual")], WRITES)
    assert [a.id for a in atoms] == ["hard.ok", "hard.bad"]
    assert ref.violations(_runs(1)[0], atoms, WRITES, _canon) == ["bad"]


def test_a_rule_broken_by_a_few_percent_of_the_recordings_is_demoted():
    """Calibrated on the second retail build: rules firing at 2.7% of confirmed recordings were miscompiled (D114)."""
    rules = [_rule("rare", "def check(pre_state, write_call, transcript):\n    return write_call['arguments']['order_id'] != '#W1'\n")]
    runs = _runs(39) + [make_run("odd", [call("cancel_pending_order", {"order_id": "#W2"}), result({"status": "cancelled"})])]
    rates = ref.constraint_rates(rules, runs, WRITES, _canon)
    assert rates["rare"] == {"failed": 39, "runs": 40}
    one_in_forty = {"rare": {"failed": 1, "runs": 40}}
    assert ref.demote(rules, one_in_forty)[1][0]["id"] == "rare"
    assert ref.demote(rules, {"rare": {"failed": 1, "runs": 80}})[1] == []


def test_the_confirmation_lists_every_recording_it_saw():
    broken = cancel_run("b")
    broken.violated = ["c1"]
    out = ref.confirm([cancel_run("a"), broken]).as_dict()
    assert [r["run_id"] for r in out["recordings"]] == ["a", "b"] and out["failed"] == {"b": "violates c1"}
