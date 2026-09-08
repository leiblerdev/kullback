"""judge.py: the reference judge as an agent with a bounded look, and the one-shot judge as its fallback.

The domain is a parking-permit office, invented for these tests: a permit is read with `find_permit`
and cancelled with `cancel_permit`, and the policy refuses a cancellation in the permit's final month.
That is the shape the one-shot judge got wrong on two corpora: a Run that wrote nothing because the
policy forbade the write read to it as a Run that abandoned the Task.
"""

from __future__ import annotations

import json

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.examiner import judge as J
from kullback.examiner import reference as ref
from kullback.runner.records import Constraint

WRITE_TOOLS = {"cancel_permit"}
READ_TOOLS = {"find_permit"}
PERMIT = {"permit_id": "PK-77", "zone": "north", "expires": "2026-09-30", "final_month": True}
WORLD = {"permits": {"PK-77": PERMIT}}
INTENT = "cancel permit PK-77"
POLICY = ["A permit in its final month may not be cancelled; the holder is told the expiry date instead.",
          "A permit outside its final month is cancelled at the holder's request."]


def _canon(value):
    return value


def _event(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def _write_run(tmp_path, run_id: str, events: list[dict]) -> str:
    """One Run on disk, the way loop.py writes one: event lines and a footer carrying the world."""
    path = tmp_path / f"{run_id}.jsonl"
    lines = [*events, {"run_id": run_id, "task_id": "permit", "termination_reason": "success",
                       "start_state": WORLD}]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return str(path)


def refusing_run(tmp_path, run_id: str = "a") -> ref.Recording:
    """The holder is told the expiry date and nothing is written, which is what the policy asks for."""
    path = _write_run(tmp_path, run_id, [
        _event("user_turn", content="Please cancel my parking permit PK-77."),
        _event("tool_call", id="c0", name="find_permit", args={"permit_id": "PK-77"}, kind="read"),
        _event("tool_result", id="c0", result=PERMIT),
        _event("model_call", reply={"content": "Permit PK-77 runs to 2026-09-30 and is in its final "
                                               "month, so it stays open until then."}),
    ])
    return ref.load(path, ref.RECORDING, run_id=run_id, trace_id=run_id, write_tools=WRITE_TOOLS, fn=_canon)


def cancelling_run(tmp_path, run_id: str = "b") -> ref.Recording:
    path = _write_run(tmp_path, run_id, [
        _event("user_turn", content="Please cancel my parking permit PK-77."),
        _event("tool_call", id="c1", name="cancel_permit", args={"permit_id": "PK-77"}),
        _event("tool_result", id="c1", result={"permit_id": "PK-77", "status": "cancelled"}),
        _event("model_call", reply={"content": "Cancelled."}),
    ])
    return ref.load(path, ref.RECORDING, run_id=run_id, trace_id=run_id, write_tools=WRITE_TOOLS, fn=_canon)


def noting_run(tmp_path, run_id: str = "c") -> ref.Recording:
    """A third End state: the permit is cancelled and a note is written with it."""
    path = _write_run(tmp_path, run_id, [
        _event("user_turn", content="Please cancel my parking permit PK-77."),
        _event("tool_call", id="c2", name="cancel_permit", args={"permit_id": "PK-77", "note": "holder moved"}),
        _event("tool_result", id="c2", result={"permit_id": "PK-77", "status": "cancelled"}),
        _event("model_call", reply={"content": "Cancelled and noted."}),
    ])
    return ref.load(path, ref.RECORDING, run_id=run_id, trace_id=run_id, write_tools=WRITE_TOOLS, fn=_canon)


def _call(call_id: str, name: str, arguments: dict) -> ModelReply:
    return ModelReply(content=None, tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])


def _judge(model, **kw) -> J.AgentJudge:
    return J.AgentJudge(model, constraints=kw.pop("constraints", ()), write_tools=WRITE_TOOLS,
                        read_tools=READ_TOOLS, fn=_canon, **kw)


FINAL_MONTH = "the permit is in its final month, so cancelling it is not what the policy allows"
# The key the two states differ on and the value the cancelling state holds at it: a ruling that fails
# that state cites them, or the judgement is dropped whole and nothing is failed (D186).
CANCELLED_KEY = "cancel_permit.PK-77.permit_id"


def ruling(*failures: dict, evidence=("intent", "policy", "end_states"), reason: str = FINAL_MONTH) -> str:
    return json.dumps({"failed": list(failures), "evidence": list(evidence), "reason": reason})


def fails(group: str, key: str, value: str, expected: str, reason: str = FINAL_MONTH) -> dict:
    return {"group": group, "key": key, "value": value, "expected": expected, "reason": reason}


RULING = ruling(fails("B", CANCELLED_KEY, "PK-77", "A"))


# --- the look ---------------------------------------------------------------

def test_the_judge_asks_for_the_rows_and_the_rules_on_a_refusal_the_policy_allows(tmp_path):
    """The state that wrote nothing was right, and the judge can only tell by reading the rule and
    the row. The one-shot judge sees neither and fails it for making no writes."""
    model = TestModel([_call("t1", "policy", {"query": "final month"}),
                       _call("t2", "rows", {"group": "A"}),
                       ModelReply(content=RULING)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model), phrases=["permit PK-77"])
    assert [r.run_id for r in out.references] == ["a"]
    assert out.failed["b"].startswith("judge: the permit is in its final month")
    assert [row["tool"] for row in out.judge_calls] == ["policy", "rows"]
    assert "A permit in its final month" in out.judge_calls[0]["summary"]
    assert "2026-09-30" in out.judge_calls[1]["summary"]
    assert out.judge_fallback is None


def test_the_row_tool_answers_with_the_world_as_it_stood_and_names_a_value_it_does_not_hold(tmp_path):
    case = _judge(TestModel([])).case(INTENT, POLICY, ref.group([refusing_run(tmp_path)]))
    assert "permits PK-77" in J._rows_text(case, "A") and "north" in J._rows_text(case, "A")
    assert "no state is labelled 'Z'" in J._rows_text(case, "Z")


FINAL_MONTH_RULE = Constraint(
    id="c_1", text="A permit in its final month is not cancelled.", compiled=True,
    predicate_src=("def check(state, call, turns):\n"
                   "    row = (state.get('permits') or {}).get(call['arguments'].get('permit_id')) or {}\n"
                   "    return not row.get('final_month')\n"))


def test_the_constraints_tool_says_which_rules_judged_the_runs_and_which_said_nothing(tmp_path):
    """A rule that judged no call of a state says nothing either way, which is the state the judge
    must not read as the policy having been kept (D173)."""
    groups = ref.group([refusing_run(tmp_path), cancelling_run(tmp_path)])
    case = _judge(TestModel([]), constraints=[FINAL_MONTH_RULE]).case(INTENT, POLICY, groups)
    wrote = J._constraints_text(case, "B")
    assert "c_1: A permit in its final month is not cancelled. | judged 1 of 1 run(s), failed 1" in wrote
    assert "0 judged a call of these Runs and 1 judged none" in J._constraints_text(case, "A")


def test_the_answer_tool_gives_the_last_thing_one_run_said_and_no_more_of_the_conversation(tmp_path):
    case = _judge(TestModel([])).case(INTENT, POLICY, ref.group([refusing_run(tmp_path)]))
    text = J._answer_text(case, "A")
    assert text.endswith("it stays open until then.")
    assert "Please cancel my parking permit" not in text


# --- the cap and the fallback ------------------------------------------------

def test_the_judge_over_the_cap_falls_back_to_the_one_shot_judge_and_the_record_says_so(tmp_path):
    """Seven calls is one past the cap; the session ends and the judge that reads one prompt rules."""
    model = TestModel([_call(f"t{i}", "intent", {}) for i in range(J.MAX_CALLS + 1)]
                      + [ModelReply(content=RULING)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model))
    assert out.judge_fallback == J.OVER_THE_CAP
    assert len(out.judge_calls) == J.MAX_CALLS + 1
    assert [r.run_id for r in out.references] == ["a"]


def test_a_turn_that_asks_for_many_tools_at_once_still_looks_only_six_times(tmp_path):
    """The cap is a hook asked before each tool, not the cancel flag alone: the loop runs a turn's
    whole batch before it reads that flag, and a judge that batched its last turn looked nine times
    under a cap of six and the record said nine."""
    batch = ModelReply(content=None, tool_calls=[ToolCallRequest(id=f"t{i}", name="intent", arguments={})
                                                 for i in range(J.MAX_CALLS + 3)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT, policy_lines=POLICY,
                      judge=_judge(TestModel([batch, ModelReply(content=RULING)])))
    looked = [row for row in out.judge_calls if "Intent" in row["summary"]]
    assert len(looked) == J.MAX_CALLS, "the seventh call is blocked before it reads anything"
    assert len(out.judge_calls) == J.MAX_CALLS + 1 and out.judge_fallback == J.OVER_THE_CAP


def test_the_judge_is_told_to_answer_once_it_has_spent_its_calls(tmp_path):
    model = TestModel([_call(f"t{i}", "intent", {}) for i in range(J.MAX_CALLS)] + [ModelReply(content=RULING)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model))
    said = [message["content"] for call in model.calls for message in call["messages"]
            if message.get("role") == "user"]
    assert any(J.ANSWER_NOW == text for text in said)
    assert out.judge_fallback is None and len(out.judge_calls) == J.MAX_CALLS


def test_a_judgement_that_will_not_parse_falls_back_and_the_fallback_rules(tmp_path):
    model = TestModel([ModelReply(content="B is the one I would fail."), ModelReply(content=RULING)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model))
    assert out.judge_fallback == J.UNREADABLE
    assert [r.run_id for r in out.references] == ["a"]


# --- what it may not rest on -------------------------------------------------

def test_a_ruling_that_rests_on_a_row_it_read_is_not_an_abstention(tmp_path):
    """The one-shot judge had three sources and abstained on any other name; this judge has its six
    tools too, so naming the rows it read is naming something it was given (D93)."""
    model = TestModel([_call("t1", "rows", {"group": "B"}),
                       ModelReply(content=ruling(fails("B", CANCELLED_KEY, "PK-77", "A"),
                                                evidence=("rows", "policy"),
                                                reason="the permit row is in its final month"))])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model))
    assert not out.judge_abstained and [r.run_id for r in out.references] == ["a"]
    assert "rows" in J.JUDGE_SOURCES and "transcript" not in J.JUDGE_SOURCES


def test_a_judgement_that_needs_the_conversation_abstains_and_fails_nothing(tmp_path):
    model = TestModel([_call("t1", "answer", {"group": "B"}),
                       ModelReply(content=ruling(fails("A", CANCELLED_KEY, ref.NO_VALUE, "B"),
                                                evidence=("transcript",),
                                                reason="the holder agreed to the cancellation when asked"))])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                      policy_lines=POLICY, judge=_judge(model))
    assert out.judge_abstained and not out.references and not out.failed
    assert "transcript" in (out.judge_reason or "") and "the judge abstained" in (out.reason or "")
    assert [row["tool"] for row in out.judge_calls] == ["answer"]


def test_a_failure_of_the_agent_that_cites_nothing_the_states_differ_on_abstains(tmp_path):
    """D186 holds for the judge that looks too: it read the row and named the permit id, which both
    states hold, so its failure settles none of the disagreement and the judgement is dropped whole.
    A failure on the key they do differ on, with the value that state holds, still fails it."""
    def rule(reply):
        model = TestModel([_call("t1", "rows", {"group": "B"}), ModelReply(content=reply)])
        return ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                           policy_lines=POLICY, judge=_judge(model))

    shared = rule(ruling(fails("B", "find_permit.PK-77.permit_id", "PK-77", "A"), evidence=("rows",)))
    assert shared.judge_abstained and shared.judge_uncited
    assert shared.references == [] and shared.failed == {}
    assert shared.abstain_reason == ("a failure rested on nothing the states differ on: "
                                     "find_permit.PK-77.permit_id")
    assert [row["tool"] for row in shared.judge_calls] == ["rows"], "the look is still on the record"

    cited = rule(ruling(fails("B", CANCELLED_KEY, "PK-77", "A"), evidence=("rows",)))
    assert not cited.judge_abstained and not cited.judge_uncited
    assert [r.run_id for r in cited.references] == ["a"]


def test_a_policy_line_the_agent_never_searched_for_cannot_be_what_the_value_should_have_been(tmp_path):
    """`expected` of `policy:<n>` is checked against the lines this judgement was answered with, so a
    number the judge did not look up is a citation of nothing."""
    def rule(*calls):
        model = TestModel([*calls, ModelReply(content=ruling(
            fails("B", CANCELLED_KEY, "PK-77", "policy:1"), evidence=("policy",)))])
        return ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path)], intent=INTENT,
                           policy_lines=POLICY, judge=_judge(model))

    unread = rule(_call("t1", "rows", {"group": "B"}))
    assert unread.judge_uncited and unread.references == []
    read = rule(_call("t1", "policy", {"query": "final month"}))
    assert not read.judge_uncited and [r.run_id for r in read.references] == ["a"]


def test_the_stated_tool_marks_the_values_only_one_state_told_the_user(tmp_path):
    groups = ref.group([refusing_run(tmp_path), cancelling_run(tmp_path)])
    text = J._stated_text(_judge(TestModel([])).case(INTENT, POLICY, groups), "A")
    assert "differs from the other states on: " in text
    assert "2026-09-30 (differs)" in text and "PK-77 (differs)" in text


NOTE_KEY = "cancel_permit.PK-77.note"


def test_the_agent_gets_the_second_pass_over_the_states_its_first_ruling_left_in(tmp_path):
    """D193 sits in `judge_groups`, so the judge that looks is asked twice the way the one-shot judge
    is: a fresh session over the survivors, with the stop rule on the end of its message."""
    model = TestModel([ModelReply(content=ruling(fails("C", NOTE_KEY, "holder moved", "A"))),
                       ModelReply(content=RULING)])
    out = ref.confirm([refusing_run(tmp_path), cancelling_run(tmp_path), noting_run(tmp_path)],
                      intent=INTENT, policy_lines=POLICY, judge=_judge(model))
    assert [r.run_id for r in out.references] == ["a"]
    assert out.judge_passes == 2 and out.judge_residue_resolved
    second = model.calls[-1]["messages"][-1]["content"]
    assert "exactly one of them may remain" in second and "\nC (1 run)" not in second
    assert second.rstrip().endswith("which is worse than saying you could not tell.")


def test_the_prompt_names_its_tools_and_puts_the_stop_rule_last():
    prompt = J.judge_system_prompt()
    for name in ("intent", "policy", "rows", "stated", "constraints", "answer"):
        assert f"- {name} " in prompt
    assert prompt.index("Your tools") < prompt.index("Two worked examples") < prompt.index("The rule for")
    assert prompt.rstrip().endswith("your judgement is lost.")
