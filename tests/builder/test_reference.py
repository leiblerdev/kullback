"""reference.py: which Runs a Verifier may be derived from (D111), re-rolls (D112) and demoted constraints (D76)."""

from __future__ import annotations

from gates.verifier_fixtures import assistant, call, make_run, result, user
from kullback.ai.provider import TestModel
from kullback.builder import reference as ref
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
    assert ref.describe(()) == "no writes"


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
    out = ref.confirm([cancel_run("a"), empty_run("b")], request="cancel order #W123",
                      policy_lines=["pending orders may be cancelled"], judge=judge)
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed == {"b": "judge: the cancellation the user asked for never happened"}
    assert out.judged
    prompt = judge.calls[0]["messages"][0]["content"]
    assert "The user asked: cancel order #W123" in prompt and "A (1 run): cancel_pending_order" in prompt


def test_the_judge_never_awards_a_pass():
    """Failing nothing leaves the disagreement; failing everything leaves no Reference."""
    nothing = ref.confirm([cancel_run("a"), empty_run("b")], judge=TestModel(['{"failed": [], "reason": "cannot tell"}']))
    assert nothing.references == [] and nothing.reason.startswith("recordings disagree")
    everything = ref.confirm([cancel_run("a"), empty_run("b")], judge=TestModel(['{"failed": ["A", "B"]}']))
    assert everything.references == [] and everything.reason == "the judge failed every End state"


def test_an_unreadable_judge_reply_fails_nothing():
    assert ref.parse_judgement("I think B is wrong", {"A", "B"}) == (set(), "unreadable reply")
    assert ref.parse_judgement('{"failed": "B"}', {"A", "B"}) == (set(), "unreadable reply")
    assert ref.parse_judgement('sure: {"failed": ["b", "Z"], "reason": "x"}', {"A", "B"}) == ({"B"}, "x")


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
