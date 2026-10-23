"""The Examiner's goal: what the session is shown, what its tools may do, and how its answer is read.

The model is the harness TestModel with scripted replies; the workdir is a minimal invented
Environment whose Spec, Verifier, Run and agent lines carry sentinels the goal must never show.
"""

from __future__ import annotations

import json

import pytest

from kullback.ai.provider import ModelReply, TestModel
from kullback.examiner import goal as G
from kullback.runner.records import RawPtr, Trace, Turn
from kullback.spec import writer_tools as T
from tests.helpers.scripted import tool_reply

STATE = {"boxes": {"A1": {"box_id": "A1", "slot": "two"}, "B2": {"box_id": "B2", "slot": "four"}}}
POLICY = "# Store policy\n\n## Moving boxes\n\nMove a box only to a free slot.\n"
ASK = "Please move box A1 to slot seven."
PTR = RawPtr(file_hash="h")


def _trace() -> Trace:
    turns = [Turn(idx=0, role="assistant", content="AGENTLINE how can I help", raw_ptr=PTR),
             Turn(idx=1, role="user", content=ASK, raw_ptr=PTR),
             Turn(idx=2, role="assistant", content="AGENTLINE I cannot do that", raw_ptr=PTR),
             Turn(idx=3, role="user", content="ok", raw_ptr=PTR)]
    return Trace(trace_id="rec1", raw_hash="h", ingest_version="1", source="s", turns=turns, raw_ptr=PTR)


def _workdir(root):
    """An Environment on disk with one recording, and a Spec, Verifier, Run and Reference that carry SEEN."""
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": ["rec1"]}))
    (root / "traces").mkdir()
    (root / "traces" / "rec1.json").write_text(_trace().model_dump_json())
    (root / "db.json").write_text(json.dumps(STATE))
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "move_box", "kind": "write"},
                                                     {"name": "list_boxes", "kind": "read"}]))
    (root / "overlays").mkdir()
    (root / "overlays" / "t1.json").write_text(json.dumps({"overlay": {"task_id": "t1", "rows": []}, "values": {}}))
    (root / "env").mkdir()
    (root / "env" / "policy.md").write_text(POLICY)
    (root / "spec").mkdir()
    (root / "spec" / "t1.json").write_text(json.dumps({"task_id": "t1", "intent": {"text": "SEEN"}}))
    (root / "runs" / "t1").mkdir(parents=True)
    (root / "runs" / "t1" / "ref.jsonl").write_text(json.dumps({"type": "stop", "payload": {"sentinel": "SEEN"}}))
    (root / "verifiers").mkdir()
    (root / "verifiers" / "t1.json").write_text(json.dumps({"task_id": "t1", "atoms": [], "note": "SEEN"}))
    (root / "exam").mkdir()
    (root / "exam" / "references.json").write_text(json.dumps({"t1": "SEEN"}))


def answer(**goal) -> ModelReply:
    return ModelReply(content=json.dumps(goal))


MOVE = {"table": "boxes", "row": "A1", "field": "slot", "value": "seven", "user": "move box A1 to slot seven"}


def test_the_session_sees_the_user_words_and_never_the_intent_items_verifier_runs_or_agent_lines(tmp_path):
    _workdir(tmp_path)
    inputs = G.load_goal_inputs(tmp_path, "t1")
    model = TestModel([tool_reply("read_row", table="boxes", key="A1"), answer(required=[MOVE])])
    G.write_goal("t1", inputs, model)
    shown = json.dumps([c["messages"] for c in model.calls])
    assert ASK in shown and "SEEN" not in shown and "AGENTLINE" not in shown


def test_the_inputs_hold_no_intent_and_the_user_turns_hold_no_agent_line(tmp_path):
    _workdir(tmp_path)
    inputs = G.load_goal_inputs(tmp_path, "t1")
    assert "intent" not in G.GoalInputs._fields
    assert inputs.user_turns == {"rec1": [{"turn": 1, "user": ASK}, {"turn": 3, "user": "ok"}]}
    assert inputs.state == STATE and inputs.policy_text == POLICY


def test_the_session_is_given_only_the_read_tools():
    assert [tool["name"] for tool in G.GOAL_TOOLS] == list(G.READS)
    assert "add_items" not in json.dumps(G.GOAL_TOOLS)


def test_a_tool_outside_the_reads_is_answered_as_unknown(tmp_path):
    _workdir(tmp_path)
    model = TestModel([tool_reply("add_items", items=[]), answer(required=[])])
    G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), model)
    assert model.calls[1]["messages"][-1]["content"] == "unknown tool"


def test_a_starting_state_carrying_a_run_is_refused():
    bad = G.GoalInputs(user_turns={}, policy_text=POLICY, state={"events": []}, tools=[])
    with pytest.raises(T.WriterInputError):
        G.write_goal("t1", bad, TestModel([answer()]))


def test_the_prompt_runs_receive_tools_rules_choice_output_then_stop():
    order = [G.WHAT_YOU_RECEIVE, G.TOOLS_TEXT, G.RULES, G.CHOICE_RULE, G.OUTPUT, G.STOP_RULE]
    places = [G.SYSTEM_PROMPT.index(part) for part in order]
    assert places == sorted(places) and G.SYSTEM_PROMPT.rstrip().endswith(G.STOP_RULE.rstrip())


def _inputs() -> G.GoalInputs:
    return G.GoalInputs(user_turns={"rec1": [{"turn": 1, "user": ASK}]}, policy_text=POLICY, state=STATE,
                        tools=[])


def test_a_grounded_goal_is_read_whole_with_no_problems():
    goal = G.parse_goal(json.dumps({"required": [MOVE], "untouched": ["everything else"]}), _inputs())
    assert goal["required"] == [MOVE] and goal["problems"] == [] and goal["cannot_tell"] is False
    assert set(G.KEYS) <= set(goal)


def test_code_names_a_missing_row_an_ungrounded_cell_and_a_quote_no_turn_holds():
    cells = [dict(MOVE, row="Z9"), {"table": "boxes", "row": "B2", "field": "slot", "value": "one"},
             dict(MOVE, user="put it anywhere")]
    problems = G.parse_goal(json.dumps({"required": cells}), _inputs())["problems"]
    assert any("no such row" in p for p in problems)
    assert any("rests on no user words" in p for p in problems)
    assert any("not found in any user turn" in p for p in problems)


def test_a_created_row_is_not_a_missing_row():
    cell = {"table": "boxes", "row": "new:1", "field": "slot", "value": "seven", "policy": "only to a free slot"}
    assert G.parse_goal(json.dumps({"required": [cell]}), _inputs())["problems"] == []


def test_an_answer_with_no_goal_object_is_no_answer_and_not_cannot_tell():
    goal = G.parse_goal("I am not sure.", _inputs())
    assert goal["answered"] is False and goal["cannot_tell"] is False and goal["problems"] == ["no goal object"]


def test_a_restated_answer_is_read_from_its_last_goal_object():
    text = '```json\n{"cannot_tel": false}\n```\nCorrection, the full answer:\n' + json.dumps({"required": [MOVE]})
    goal = G.parse_goal(text, _inputs())
    assert goal["answered"] is True and goal["required"] == [MOVE]


def test_goals_writes_one_file_per_task_and_skips_one_already_written(tmp_path):
    _workdir(tmp_path)
    (tmp_path / "budget.json").write_text(json.dumps({"stages": {}}))
    first = G.goals(tmp_path, ["t1"], TestModel([answer(required=[MOVE])]))
    again = G.goals(tmp_path, ["t1"], TestModel([]))
    saved = json.loads(G.goal_path(tmp_path, "t1").read_text())
    assert first["written"] == 1 and again == dict(again, written=0, skipped=1)
    assert saved["required"] == [MOVE] and saved["task_id"] == "t1"


def test_the_prompt_says_a_value_only_a_tool_sets_is_choice_or_unknown_unless_the_policy_states_it():
    assert "A value only a tool sets" in G.RULES and "is choice or unknown unless the policy states it" in G.RULES
    assert "you also list under unknown is not required" in G.CHOICE_RULE


def test_an_empty_last_answer_gets_one_plain_ask_for_the_goal_object(tmp_path):
    _workdir(tmp_path)
    model = TestModel([tool_reply("read_row", table="boxes", key="A1"), ModelReply(content=""), answer(required=[MOVE])])
    goal = G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), model)
    last = model.calls[-1]["messages"]
    assert goal["asked"] == ["empty"] and goal["required"] == [MOVE] and goal["answered"] is True
    assert last[-1]["content"] == G.EMPTY_PROMPT and last[-2] == {"role": "assistant", "content": G.NO_ANSWER}
    assert model.calls[-1]["config"].tool_choice == "none"


UNSURE = {"what": "the slot may be taken", "table": "boxes", "row": "A1", "field": "slot"}


def test_a_required_cell_also_listed_unknown_goes_back_once_and_a_settled_answer_stands(tmp_path):
    _workdir(tmp_path)
    model = TestModel([answer(required=[MOVE], unknown=[UNSURE]), answer(required=[MOVE])])
    goal = G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), model)
    assert goal["asked"] == ["clash"] and goal["moved"] == 0 and goal["required"] == [MOVE]
    assert model.calls[-1]["messages"][-1]["content"].startswith(G.BACK_PROMPT)


def test_a_clash_still_there_after_the_ask_is_moved_to_unknown_by_code(tmp_path):
    _workdir(tmp_path)
    clash = answer(required=[MOVE], unknown=[UNSURE])
    goal = G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), TestModel([clash, clash]))
    assert goal["moved"] == 1 and goal["required"] == []
    assert goal["unknown"][-1] == {"table": "boxes", "row": "A1", "field": "slot", "what": G.MOVED + G.CLASH}


def test_an_unknown_on_another_row_or_field_is_no_clash():
    other = [dict(UNSURE, row="B2"), dict(UNSURE, field="label"), "a plain unknown"]
    assert G.clashes({"required": [MOVE], "unknown": other}) == []
    assert G.clashes({"required": [MOVE], "unknown": [dict(UNSURE, field="")]}) != []


INDEXED = {"table": "boxes", "row": "A1", "field": "parts[0].slot", "value": "seven", "user": ASK}
ELSEWHERE = {"table": "crates", "row": "new:1", "field": "tool", "value": "move_box", "user": ASK}


def test_the_output_rule_says_a_list_is_whole_or_an_append_never_an_indexed_path():
    assert "never an indexed path" in G.OUTPUT and "names a table the Starting state holds" in G.OUTPUT


def test_an_indexed_path_or_a_table_the_starting_state_lacks_goes_back_once_and_a_fixed_answer_stands(tmp_path):
    _workdir(tmp_path)
    model = TestModel([answer(required=[INDEXED, ELSEWHERE]), answer(required=[MOVE])])
    goal = G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), model)
    asked = model.calls[-1]["messages"][-1]["content"]
    assert goal["asked"] == ["indexed", "table"] and goal["moved"] == 0 and goal["required"] == [MOVE]
    assert asked.startswith(G.BACK_PROMPT) and G.INDEXED in asked and G.NO_TABLE in asked


def test_a_sent_back_cell_still_there_after_the_ask_moves_to_unknown_from_required_or_choice(tmp_path):
    _workdir(tmp_path)
    choice = {"table": "boxes", "row": "B2", "field": "parts[1]", "allowed": "any valid"}
    again = answer(required=[MOVE, INDEXED], choice=[choice])
    goal = G.write_goal("t1", G.load_goal_inputs(tmp_path, "t1"), TestModel([again, again]))
    assert goal["moved"] == 2 and goal["required"] == [MOVE] and goal["choice"] == []
    assert [u["what"] for u in goal["unknown"][-2:]] == [G.MOVED + G.INDEXED] * 2


def test_a_dotted_path_and_a_table_the_schema_declares_are_not_sent_back():
    world = G.I.World(STATE, columns={"crates": ["crate_id"]})
    dotted = dict(MOVE, field="slot.name")
    declared = dict(ELSEWHERE, field="crate_id")
    assert G.sent_back({"required": [dotted, declared], "unknown": []}, world) == []


def test_a_dotted_number_after_a_list_is_an_indexed_path_and_after_a_dict_is_not():
    world = G.I.World({"boxes": {"A1": {"parts": [{"slot": "two"}], "spares": {"7": {"slot": "one"}}}}})
    after_list = dict(MOVE, field="parts.0.slot")
    after_dict = dict(MOVE, field="spares.7.slot")
    sent = G.sent_back({"required": [after_list, after_dict], "unknown": []}, world)
    assert [(c["field"], c["why"]) for c in sent] == [("parts.0.slot", G.INDEXED)]
