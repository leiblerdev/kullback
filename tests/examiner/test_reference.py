"""reference.py: which Runs a Verifier may be derived from (D111), re-rolls (D112) and demoted constraints (D76)."""

from __future__ import annotations

import json

from gates.verifier_fixtures import assistant, call, make_run, result, user
from kullback.ai.provider import TestModel
from kullback.examiner import reference as ref
from kullback.runner.records import Constraint

# A parcel courier desk, invented for these tests: a pending delivery is cancelled with a reason.
WRITES = {"cancel_pending_delivery", "redirect_delivered_parcel"}
# One key `cancel_run`'s state holds and `empty_run`'s does not, which is what the two disagree on.
DELIVERY_KEY = "cancel_pending_delivery.#D123.delivery_id"


def ruling(*failures: dict, evidence=("intent", "end_states"), reason: str = "") -> str:
    """One judge reply in D186's shape: every failure cites the state, the key, the value and what it
    should have been. A failure is written `{"group": .., "key": .., "value": .., "expected": ..}`."""
    return json.dumps({"failed": list(failures), "evidence": list(evidence), "reason": reason})


def fails(group: str, key: str, value: str, expected: str, reason: str = "") -> dict:
    return {"group": group, "key": key, "value": value, "expected": expected, "reason": reason}


def _canon(value):
    return value


def cancel_run(run_id: str, delivery: str = "#D123", kind: str = ref.RECORDING) -> ref.Recording:
    run = make_run(run_id, [
        user(f"Please cancel my delivery {delivery}."),
        call("cancel_pending_delivery", {"delivery_id": delivery, "reason": "no longer needed"}),
        result({"delivery_id": delivery, "status": "cancelled"}),
        assistant("Done."),
    ])
    return ref.Recording(run_id=run_id, path=run_id, kind=kind, trace_id=run_id if kind == ref.RECORDING else None,
                         end_state=ref.end_state(run, WRITES, _canon),
                         stated=ref.stated_facts(run, _canon), transferred=ref.transferred(run),
                         settled=ref.settled_state(run, WRITES, _canon))


def empty_run(run_id: str, kind: str = ref.RECORDING) -> ref.Recording:
    run = make_run(run_id, [user("Please cancel my delivery #D123."), assistant("I cannot do that.")])
    return ref.Recording(run_id=run_id, path=run_id, kind=kind, end_state=ref.end_state(run, WRITES, _canon),
                         stated=ref.stated_facts(run, _canon), transferred=ref.transferred(run),
                         settled=ref.settled_state(run, WRITES, _canon))


# --- End states -------------------------------------------------------------

def test_two_runs_that_wrote_the_same_thing_share_an_end_state():
    assert cancel_run("a").end_state == cancel_run("b").end_state
    assert cancel_run("a").end_state != cancel_run("c", delivery="#D999").end_state
    assert empty_run("d").end_state == ()


def test_describe_names_the_tool_the_entity_and_the_values():
    text = ref.describe(cancel_run("a").end_state)
    assert text.startswith("cancel_pending_delivery on #D123") and "reason=no longer needed" in text
    assert ref.describe(()) == "no writes; the answer states nothing read from the world"
    assert ref.describe(ref.ANSWERED) == "no writes; the answer states facts read from the world"


# --- End states of a Run that wrote nothing ---------------------------------
# A public library: the reader asks about a loan, the agent looks it up and either states what it
# read or refuses. Neither Run writes, and before this the two were one group.

LIBRARY_WRITES = {"renew_loan"}
LOAN = {"loan_id": "LB-4412", "title": "The Long Ships", "due": "2026-09-20", "fine": 0}


def _library_recording(run_id: str, answer: str, *, renewed: bool = False, handed_on: bool = False,
                       kind: str = ref.RECORDING) -> ref.Recording:
    events = [user("When is my copy of The Long Ships due back?"),
              call("get_loan", {"loan_id": "LB-4412"}, kind="read", cid="c0"),
              result(LOAN, cid="c0")]
    if renewed:
        events += [call("renew_loan", {"loan_id": "LB-4412", "weeks": 2}, cid="c1"),
                   result({"loan_id": "LB-4412", "due": "2026-10-04"}, cid="c1")]
    if handed_on:
        events += [call("transfer_to_librarian", {"loan_id": "LB-4412"}, kind="read", cid="c2"),
                   result({"queued": True}, cid="c2")]
    events.append(assistant(answer))
    run = make_run(run_id, events)
    return ref.Recording(run_id=run_id, path=run_id, kind=kind,
                         trace_id=run_id if kind == ref.RECORDING else None,
                         end_state=ref.end_state(run, LIBRARY_WRITES, _canon),
                         stated=ref.stated_facts(run, _canon), transferred=ref.transferred(run),
                         settled=ref.settled_state(run, LIBRARY_WRITES, _canon))


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
    due = "the reader was never told when the book is due"
    judge = TestModel([ruling(fails("B", "stated:2026-09-20", ref.NO_VALUE, "A", due), reason=due)])
    out = ref.confirm([answered("a"), answered("b"), refused("c")],
                      intent="tell the reader when loan LB-4412 is due",
                      policy_lines=["a reader may be told the due date of their own loan"], judge=judge)
    assert [r.run_id for r in out.references] == ["a", "b"]
    assert out.failed == {"c": "judge: the reader was never told when the book is due"}
    assert out.judged and not out.judge_abstained
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "A (2 runs): no writes; the answer states facts read from the world" in prompt
    assert "B (1 run): no writes; the answer states nothing read from the world" in prompt


def test_the_judge_is_told_which_facts_each_end_state_stated_back():
    """A Task the agent resolved by answering reads as an abandoned one when the judge sees only "no
    writes"; the values the answers stated are the other half of the End state (D43)."""
    judge = TestModel([ruling(fails("B", "stated:2026-09-20", ref.NO_VALUE, "A"), reason="no due date")])
    ref.confirm([answered("a"), refused("b")], intent="tell the reader when loan LB-4412 is due",
                policy_lines=["a reader may be told the due date of their own loan"], judge=judge)
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "told the user: 2026-09-20, LB-4412; none handed the conversation on" in prompt
    assert "told the user no fact read from the world" in prompt
    assert "you still do not have the transcript" in prompt


def test_a_state_whose_runs_handed_the_conversation_on_says_so():
    judge = TestModel(['{"failed": [], "evidence": ["end_states"], "reason": "cannot tell"}'])
    ref.confirm([answered("a"), refused("b", handed_on=True)],
                intent="tell the reader when loan LB-4412 is due", judge=judge)
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "1 of 1 handed the conversation on" in prompt


def test_the_facts_of_a_state_are_the_ones_all_its_runs_stated_between_them():
    """One state, two Runs, and the judge is told everything that state's Runs told the user."""
    both = ref.group([answered("a"), answered("b", answer="Nothing is owed on LB-4412.")])
    assert both[0]["told"] == ["2026-09-20", "LB-4412"]
    assert ref.told_line(both[0]).startswith("told the user: 2026-09-20, LB-4412")


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
    never = "the cancellation the user asked for never happened"
    judge = TestModel([ruling(fails("B", DELIVERY_KEY, ref.NO_VALUE, "A", never), reason=never)])
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel delivery #D123",
                      policy_lines=["pending deliveries may be cancelled"], judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: the cancellation the user asked for never happened"}
    assert out.judged and not out.judge_abstained
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "cancel delivery #D123" in prompt and "A (1 run): cancel_pending_delivery" in prompt


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
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel delivery #D123", judge=judge)
    assert out.failed == {} and out.references == []
    assert out.judged and out.judge_abstained
    assert "transcript" in out.judge_reason and "was not given" in out.judge_reason
    assert out.reason.startswith("recordings disagree on the End state")
    assert "the judge abstained" in out.reason


def test_a_run_that_matches_a_revised_intent_is_not_failed_for_the_opening_request():
    """The Intent is what the user wanted by the end of the Run; the opening request was not handed
    to this judge and cannot fail the state that matches the Intent."""
    judge = TestModel(['{"failed": ["A"], "evidence": ["opening_request"], "reason": '
                       '"the user requested cancellations for both deliveries"}'])
    out = ref.confirm([cancel_run("a"), cancel_run("b", delivery="#D999")],
                      intent="cancel delivery #D123 only, leave #D999 alone", judge=judge)
    assert out.failed == {} and out.judge_abstained
    assert "opening_request" in out.judge_reason


def test_a_judgement_that_rests_on_what_the_judge_was_given_still_fails_a_state():
    judge = TestModel([ruling(fails("B", DELIVERY_KEY, ref.NO_VALUE, "A"),
                              evidence=("Intent", "End states"), reason="nothing was written")])
    out = ref.confirm([cancel_run("a"), empty_run("b")], intent="cancel delivery #D123", judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: nothing was written"} and not out.judge_abstained


def test_the_judge_never_awards_a_pass():
    """Failing nothing leaves the disagreement; failing everything leaves no Reference."""
    nothing = ref.confirm([cancel_run("a"), empty_run("b")],
                          judge=TestModel(['{"failed": [], "reason": "cannot tell"}'], loop=True))
    assert nothing.references == [] and nothing.reason.startswith("recordings disagree")
    both = ruling(fails("A", DELIVERY_KEY, "#D123", "B"), fails("B", DELIVERY_KEY, ref.NO_VALUE, "A"))
    everything = ref.confirm([cancel_run("a"), empty_run("b")], judge=TestModel([both]))
    assert everything.references == [] and everything.reason.startswith("the judge failed every End state")


def test_an_unreadable_judge_reply_fails_nothing():
    groups = ref.group([cancel_run("a"), empty_run("b")])
    for text in ("I think B is wrong", '{"failed": "B"}'):
        unreadable = ref.parse_judgement(text, groups)
        assert unreadable.failed == set() and unreadable.reason == "unreadable reply"
        assert not unreadable.abstained
    read = ref.parse_judgement(
        "sure: " + ruling(fails("b", DELIVERY_KEY, ref.NO_VALUE, "A"), fails("Z", "anything", "x", "A"),
                          reason="x"), groups)
    assert read.failed == {"B"} and read.reason == "x", "a failure naming no state of this Task fails nothing"


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
    runs = [cancel_run(f"r{i}", delivery=f"#D{i}") for i in range(30)]
    labels = [g["label"] for g in ref.group(runs)]
    assert len(set(labels)) == len(labels)
    assert labels[:2] == ["A", "B"] and labels[26] == "A1"


def test_the_judge_prompt_is_bounded():
    groups = ref.group([cancel_run("a"), empty_run("b")])
    prompt = ref.judge_prompt("x" * 5000, ["line %d" % i for i in range(200)], groups)
    assert len(prompt) < 12000 and "line 39" in prompt and "line 40" not in prompt


# --- what a failing judgement must cite (D186) -------------------------------
# A bike-hire desk, invented for these tests: a hire is read with `find_hire` and its end date moved
# with `extend_hire`. Two Runs extend the same hire to different dates, so the states agree on the
# hire they touched and on the id they told the rider back, and disagree only on the date. That is
# the shape a judge must cite in: the disagreement is one value, and the reasons the judge reached
# for instead on two corpora, a confirmation it was given or not given, are nowhere in front of it.

HIRE_WRITES = {"extend_hire"}
HIRE = {"hire_id": "BH-31", "bike": "city 7", "due": "2026-09-18"}
UNTIL = "extend_hire.BH-31.until"
SHARED = "extend_hire.BH-31.hire_id"


def hire_run(run_id: str, until: str) -> ref.Recording:
    run = make_run(run_id, [
        user("Can I keep the bike for another week?"),
        call("find_hire", {"hire_id": "BH-31"}, kind="read", cid="c0"),
        result(HIRE, cid="c0"),
        call("extend_hire", {"hire_id": "BH-31", "until": until}, cid="c1"),
        result({"hire_id": "BH-31", "due": until}, cid="c1"),
        assistant(f"Hire BH-31 now runs to {until}."),
    ])
    return ref.Recording(run_id=run_id, path=run_id, trace_id=run_id,
                         end_state=ref.end_state(run, HIRE_WRITES, _canon),
                         stated=ref.stated_facts(run, _canon), transferred=ref.transferred(run),
                         settled=ref.settled_state(run, HIRE_WRITES, _canon))


def hire_groups() -> list[dict]:
    return ref.group([hire_run("a", "2026-10-02"), hire_run("b", "2026-09-25")])


def test_differing_keys_names_the_column_and_the_stated_value_the_states_disagree_on_and_nothing_else():
    keys = ref.differing_keys(hire_groups())
    assert set(keys) == {"A", "B"} and keys["A"] == keys["B"]
    assert set(keys["A"]) == {UNTIL, "stated:2026-10-02", "stated:2026-09-25"}
    assert SHARED not in keys["A"] and "stated:BH-31" not in keys["A"]


def test_a_failure_citing_a_differing_key_with_its_own_value_and_another_states_value_is_accepted():
    judge = TestModel([ruling(fails("B", UNTIL, "2026-09-25", "A", "the rider asked for another week"))])
    out = ref.confirm([hire_run("a", "2026-10-02"), hire_run("b", "2026-09-25")],
                      intent="extend hire BH-31 by a week", judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: the rider asked for another week"}
    assert not out.judge_abstained and not out.judge_uncited and out.abstain_reason is None


def test_a_failure_on_a_key_the_states_share_abstains_and_the_task_keeps_no_reference():
    """The confirmation case in code. The judge's reason was that the rider never confirmed, and it
    spelled that as a key both states hold; no value in front of it separates them, so it settles
    nothing and the whole judgement is dropped."""
    judge = TestModel([ruling(fails("B", SHARED, "BH-31", "A", "no explicit confirmation was given"))])
    out = ref.confirm([hire_run("a", "2026-10-02"), hire_run("b", "2026-09-25")],
                      intent="extend hire BH-31 by a week", judge=judge)
    assert out.references == [] and out.failed == {}
    assert out.judged and out.judge_abstained and out.judge_uncited
    assert out.abstain_reason == f"a failure rested on nothing the states differ on: {SHARED}"
    assert out.reason.startswith("recordings disagree on the End state")
    assert "the judge abstained" in out.reason


def test_a_failure_citing_a_value_the_state_does_not_hold_is_uncited():
    groups = hire_groups()
    wrong = ref.parse_judgement(ruling(fails("B", UNTIL, "2026-10-02", "A")), groups)
    assert wrong.failed == set() and wrong.abstained
    assert wrong.uncited == f"a failure rested on nothing the states differ on: {UNTIL}"
    right = ref.parse_judgement(ruling(fails("B", UNTIL, "2026-09-25", "A")), groups)
    assert right.failed == {"B"} and not right.abstained


def test_an_expected_naming_a_policy_line_that_was_shown_is_accepted_and_one_that_was_not_is_uncited():
    policy = [f"section {n}: hires may be extended" for n in range(45)]
    shown = ref.shown_policy(policy)
    assert shown == list(range(1, ref.MAX_POLICY_LINES + 1)), "the prompt shows the first forty lines"
    groups = hire_groups()
    seen = ref.parse_judgement(ruling(fails("B", UNTIL, "2026-09-25", "policy:1")), groups,
                               policy_shown=shown)
    assert seen.failed == {"B"} and not seen.abstained
    unseen = ref.parse_judgement(ruling(fails("B", UNTIL, "2026-09-25", "policy:41")), groups,
                                 policy_shown=shown)
    assert unseen.failed == set() and unseen.abstained
    assert "policy:41" not in ref.judge_prompt("extend the hire", policy, groups)


def test_each_states_block_in_the_prompt_opens_with_the_keys_and_values_it_differs_on():
    groups = hire_groups()
    prompt = ref.judge_prompt("extend hire BH-31 by a week", ["hires may be extended once"], groups)
    lines = prompt.splitlines()
    for row in groups:
        at = next(n for n, text in enumerate(lines) if text.startswith(f"{row['label']} (1 run):"))
        assert lines[at + 1].strip().startswith("differs from the other states on: ")
        assert SHARED not in lines[at + 1]
        # The value is on the line because a failure quotes it back and is checked against the
        # string the prompt used; a key without its value could only be answered by guessing.
        assert f"{UNTIL} = {ref.state_values(row)[UNTIL]}" in lines[at + 1]
        assert lines[at + 2].strip().startswith("told the user: ")


def test_a_failure_citing_the_no_value_a_state_holds_where_the_other_wrote_is_accepted():
    groups = ref.group([cancel_run("a"), empty_run("b")])
    prompt = ref.judge_prompt("cancel the delivery", [], groups)
    assert f"{DELIVERY_KEY} = {ref.NO_VALUE}" in prompt
    seen = ref.parse_judgement(ruling(fails("B", DELIVERY_KEY, ref.NO_VALUE, "A", "nothing was cancelled")),
                               groups)
    assert seen.failed == {"B"} and not seen.abstained
    guessed = ref.parse_judgement(ruling(fails("B", DELIVERY_KEY, "no writes", "A")), groups)
    assert guessed.failed == set() and guessed.abstained


# --- the residue of a fail-only judgement (D193) -----------------------------
# The same bike-hire desk with three Runs, which is the shape the residue needs: a fail-only judge
# may drop one state with confidence and still leave two, and one pass can never settle that.

LATE = "2026-10-09"
ASKED = "2026-10-02"
EARLY = "2026-09-25"
HIRE_INTENT = "extend hire BH-31 by a week"


def three_hires() -> list[ref.Recording]:
    return [hire_run("a", ASKED), hire_run("b", EARLY), hire_run("c", LATE)]


def drops_c(reason: str = "three weeks is more than was asked for") -> str:
    """The first pass: one state dropped with a cited failure, two left in."""
    return ruling(fails("C", UNTIL, LATE, "A", reason))


def _confirm_three(*replies: str) -> ref.Confirmation:
    return ref.confirm(three_hires(), intent=HIRE_INTENT, judge=TestModel(list(replies)))


def test_a_residue_of_two_is_resolved_by_a_cited_second_pass():
    judge = TestModel([drops_c(), ruling(fails("B", UNTIL, EARLY, "A", "the rider asked for a week"))])
    out = ref.confirm(three_hires(), intent=HIRE_INTENT, judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.judge_passes == 2 and out.judge_residue_resolved and not out.judge_residue_abstained
    assert out.failed == {"c": "judge: three weeks is more than was asked for",
                          "b": "judge: the rider asked for a week"}
    second = judge.calls[1]["messages"][0]["content"]
    assert "\nC (1 run)" not in second, "the second pass is asked about the survivors only"
    assert ref.SURVIVORS_NOT_TOLD_APART in second and "exactly one of them may remain" in second


def test_an_uncited_second_pass_abstains_and_names_the_keys_the_survivors_were_not_told_apart_on():
    """The second pass rests on a key both survivors hold, so it settles nothing, as the first would."""
    out = _confirm_three(drops_c(), ruling(fails("B", SHARED, "BH-31", "A", "no confirmation was given")))
    assert out.references == [] and out.judge_passes == 2
    assert out.judge_residue_abstained and out.judge_abstained and out.judge_uncited
    assert out.abstain_reason == f"{ref.SURVIVORS_NOT_TOLD_APART}: {UNTIL}, stated:{EARLY}, stated:{ASKED}"
    assert out.reason.startswith("recordings disagree on the End state (2 states")
    assert ref.SURVIVORS_NOT_TOLD_APART in out.reason


def test_a_second_pass_that_fails_nothing_abstains_and_the_first_passs_failure_stands():
    out = _confirm_three(drops_c(), '{"failed": [], "reason": "the survivors cannot be told apart"}')
    assert out.references == [] and out.judge_passes == 2
    assert out.judge_residue_abstained and not out.judge_uncited
    assert out.abstain_reason.startswith(f"{ref.SURVIVORS_NOT_TOLD_APART}: {UNTIL}")
    assert out.failed == {"c": "judge: three weeks is more than was asked for"}
    assert out.judge_second["reason"] == "the survivors cannot be told apart"


def test_failed_all_on_one_key_the_intent_required_is_recorded_as_no_correct_recording():
    """Every state failed on the same key against the same requirement is one claim about the world,
    not a disagreement between the recordings, so it is a reason of its own for a later beat."""
    out = _confirm_three(ruling(fails("A", UNTIL, ASKED, "intent", "the rider asked for two weeks"),
                                fails("B", UNTIL, EARLY, "intent"),
                                fails("C", UNTIL, LATE, "intent")))
    assert out.references == [] and out.no_correct_recording and out.no_correct_key == UNTIL
    assert out.reason.startswith(f"{ref.NO_CORRECT_RECORDING}: every End state was failed on {UNTIL}")
    assert not out.judge_abstained and out.judge_passes == 1


def test_failed_all_on_keys_that_disagree_abstains_instead():
    out = _confirm_three(ruling(fails("A", UNTIL, ASKED, "intent"),
                                fails("B", f"stated:{EARLY}", "told to the user", "intent"),
                                fails("C", UNTIL, LATE, "intent")))
    assert out.references == [] and not out.no_correct_recording
    assert out.judge_abstained and out.abstain_reason == ref.DIFFERING_GROUNDS
    assert out.reason.startswith("the judge failed every End state")


def test_the_reference_row_carries_how_many_times_the_judge_was_asked():
    once = ref.confirm(three_hires(), intent=HIRE_INTENT,
                       judge=TestModel([ruling(fails("B", UNTIL, EARLY, "A"), fails("C", UNTIL, LATE, "A"))]))
    assert once.as_dict()["judge_passes"] == 1 and once.as_dict()["judge_second"] is None
    twice = _confirm_three(drops_c(), ruling(fails("B", UNTIL, EARLY, "A", "a week was asked for")))
    row = twice.as_dict()
    assert row["judge_passes"] == 2 and row["judge_residue_resolved"] is True
    assert row["judge_second"]["failed"] == ["B"] and row["judge_second"]["reason"] == ""
    assert row["judge_citations"]["B"] == {"key": UNTIL, "value": EARLY, "expected": "A"}
    assert ref.confirm(three_hires()).as_dict()["judge_passes"] == 0, "no judge, no pass"


# --- constraints against the corpus -----------------------------------------

def _rule(rule_id: str, src: str) -> Constraint:
    return Constraint(id=rule_id, text=f"rule {rule_id}", compiled=True, predicate_src=src)


ALWAYS = "def check(pre_state, write_call, transcript):\n    return True\n"
NEVER = "def check(pre_state, write_call, transcript):\n    return False\n"


def _runs(n: int):
    return [make_run(f"run{i}", [call("cancel_pending_delivery", {"delivery_id": "#D1"}), result({"status": "cancelled"})])
            for i in range(n)]


def test_a_rule_the_confirmed_recordings_mostly_break_is_demoted_and_the_rest_kept():
    rules = [_rule("ok", ALWAYS), _rule("bad", NEVER)]
    rates = ref.constraint_rates(rules, _runs(4), WRITES, _canon)
    assert rates == {"ok": {"failed": 0, "runs": 4, "judged": 4},
                     "bad": {"failed": 4, "runs": 4, "judged": 4}}
    kept, demoted = ref.demote(rules, rates)
    assert [c.id for c in kept] == ["ok"]
    assert demoted[0]["id"] == "bad" and "4 of 4" in demoted[0]["reason"]


def test_too_few_recordings_demote_nothing():
    rules = [_rule("bad", NEVER)]
    rates = ref.constraint_rates(rules, _runs(2), WRITES, _canon)
    kept, demoted = ref.demote(rules, rates)
    assert [c.id for c in kept] == ["bad"] and demoted == []


# A rule with code that was never asked about anything read as a rule the recordings had upheld.


def _read_only_runs(n: int):
    return [make_run(f"read{i}", [call("look_up_ticket", {"ticket_id": "T-1"}, kind="read"),
                                  result({"status": "open"})]) for i in range(n)]


def test_a_rule_that_judged_no_call_is_not_counted_as_one_the_recordings_upheld():
    rates = ref.constraint_rates([_rule("quiet", NEVER)], _read_only_runs(4), WRITES, _canon,
                                 read_tools={"look_up_ticket"})
    assert rates["quiet"]["failed"] == 0 and rates["quiet"]["runs"] == 4
    assert rates["quiet"]["judged"] == 0
    assert rates["quiet"]["skipped"] == "no recording made a call this rule judges"


def test_a_rule_whose_own_tests_failed_is_still_run_against_the_recordings():
    rules = [_rule("ok", ALWAYS), Constraint(id="uncompiled", text="rule uncompiled", compiled=False,
                                             predicate_src=NEVER)]
    rates = ref.constraint_rates(rules, _runs(4), WRITES, _canon)
    assert rates["uncompiled"]["failed"] == 4 and rates["uncompiled"]["judged"] == 4
    assert rates["uncompiled"]["gates"] is False and "gates" not in rates["ok"]


def test_a_rule_that_compiled_to_no_code_says_so_rather_than_reading_as_checked():
    rates = ref.constraint_rates([Constraint(id="prose", text="rule prose")], _runs(4), WRITES, _canon)
    assert rates["prose"] == {"failed": 0, "runs": 0, "judged": 0, "skipped": "the rule compiled to no code",
                              "gates": False}


def test_a_judge_atom_is_not_run_against_the_recordings():
    rule = Constraint(id="asked", text="rule asked", compiled=True, judge_atom=True, predicate_src=ALWAYS)
    rates = ref.constraint_rates([rule], _runs(4), WRITES, _canon)
    assert rates["asked"]["skipped"] == "the rule is a judge atom"


def test_violations_name_the_constraints_a_run_breaks():
    atoms = ref.hard_atoms([_rule("ok", ALWAYS), _rule("bad", NEVER), Constraint(id="res", text="residual")], WRITES)
    assert [a.id for a in atoms] == ["hard.ok", "hard.bad"]
    assert ref.violations(_runs(1)[0], atoms, WRITES, _canon) == ["bad"]


def test_a_rule_broken_by_a_few_percent_of_the_recordings_is_demoted():
    """Calibrated on the second retail build: rules firing at 2.7% of confirmed recordings were miscompiled (D114)."""
    rules = [_rule("rare", "def check(pre_state, write_call, transcript):\n    return write_call['arguments']['delivery_id'] != '#D1'\n")]
    runs = _runs(39) + [make_run("odd", [call("cancel_pending_delivery", {"delivery_id": "#D2"}), result({"status": "cancelled"})])]
    rates = ref.constraint_rates(rules, runs, WRITES, _canon)
    assert rates["rare"] == {"failed": 39, "runs": 40, "judged": 40}
    one_in_forty = {"rare": {"failed": 1, "runs": 40}}
    assert ref.demote(rules, one_in_forty)[1][0]["id"] == "rare"
    assert ref.demote(rules, {"rare": {"failed": 1, "runs": 80}})[1] == []


def test_the_confirmation_lists_every_recording_it_saw():
    broken = cancel_run("b")
    broken.violated = ["c1"]
    out = ref.confirm([cancel_run("a"), broken]).as_dict()
    assert [r["run_id"] for r in out["recordings"]] == ["a", "b"] and out["failed"] == {"b": "violates c1"}


# --- the order of a list argument (fix C) -----------------------------------
# A caterer: the guest names three dishes and the agent books them in one call. Two recordings list
# the same dishes in different orders; the kitchen answers both with the same booking.

CATERING_WRITES = {"book_dishes"}


def _catering(run_id: str, dishes: list[str], answered: dict, kind: str = ref.RECORDING) -> ref.Recording:
    run = make_run(run_id, [
        user("Please book the soup, the pie and the tart for table 9."),
        call("book_dishes", {"table_id": "T-9", "dishes": dishes}),
        result(answered),
        assistant("Booked."),
    ])
    return ref.Recording(run_id=run_id, path=run_id, kind=kind,
                         trace_id=run_id if kind == ref.RECORDING else None,
                         end_state=ref.end_state(run, CATERING_WRITES, _canon),
                         stated=ref.stated_facts(run, _canon), transferred=ref.transferred(run),
                         settled=ref.settled_state(run, CATERING_WRITES, _canon))


BOOKED = {"table_id": "T-9", "covers": 3, "status": "booked"}


def test_the_same_dishes_booked_in_another_order_are_one_end_state():
    first = _catering("a", ["soup", "pie", "tart"], BOOKED)
    second = _catering("b", ["tart", "soup", "pie"], BOOKED)
    assert first.end_state != second.end_state
    assert first.settled == second.settled
    groups = ref.group([first, second])
    assert len(groups) == 1 and groups[0]["runs"] == ["a", "b"]


def test_the_same_dishes_the_kitchen_answered_differently_stay_two_end_states():
    first = _catering("a", ["soup", "pie", "tart"], BOOKED)
    second = _catering("b", ["tart", "soup", "pie"], dict(BOOKED, status="waitlisted"))
    assert first.settled != second.settled
    assert len(ref.group([first, second])) == 2


def test_two_runs_that_booked_different_dishes_stay_two_end_states():
    first = _catering("a", ["soup", "pie", "tart"], BOOKED)
    second = _catering("b", ["soup", "pie", "cake"], BOOKED)
    assert len(ref.group([first, second])) == 2


def test_a_task_whose_recordings_differ_only_in_that_order_confirms_a_reference():
    out = ref.confirm([_catering("a", ["soup", "pie", "tart"], BOOKED),
                       _catering("b", ["tart", "soup", "pie"], BOOKED)], judge=None)
    assert [r.run_id for r in out.references] == ["a", "b"] and out.reason is None
