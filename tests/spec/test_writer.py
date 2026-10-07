"""The Verifier writer: a scripted model reads the world and adds items, code keeps what it finds and compiles it.

The model is the harness TestModel with scripted replies; the Starting state, tools and facts are the
invented ones of tests/spec/fixtures.py. Verdicts go through the real verdict gates and runner/items.py.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import match_context
from kullback.runner.items import score_items
from kullback.runner.records import Event, Run, Verifier
from kullback.spec import writer as W
from kullback.spec import writer_tools as T
from kullback.spec.schema import SpecIntent, intent_text, load_spec
from tests.spec.fixtures import FACTS, WRITE_TOOLS, fn, reference_run

STATE = {"items": {"A1": {"item_id": "A1", "slot": "two"}, "B2": {"item_id": "B2", "slot": "four"}}}
SECTION = "section:moving items"
POLICY = "# Store policy\n\n## Moving items\n\nMove an item only to a free slot.\n"
TOOLS = [{"name": "list_items", "kind": "read", "args": []},
         {"name": "update_item", "kind": "write", "args": ["item_id", "slot"]},
         {"name": "delete_item", "kind": "write", "args": ["item_id"]}]


def _matches(end_states, context):
    """Any one end state matching passes; else an unsettled one is None; else False."""
    oks = [match_context(state, context).ok for state in end_states]
    return True if True in oks else (None if None in oks else False)


def inputs(facts=FACTS, state=STATE) -> W.WriterInputs:
    intent = SpecIntent(task_id="t1", facts=list(facts), text=intent_text(list(facts)))
    return W.WriterInputs(intent=intent, policy_text=POLICY, sections=T.policy_sections(POLICY), tools=TOOLS,
                          write_tools=set(WRITE_TOOLS) | {"delete_item"}, state=state, constraints=[], canon=fn)


def answer(key: str, items: list) -> ModelReply:
    return ModelReply(content=json.dumps({key: items}))


def add(*items: dict) -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id="a1", name="add_items", arguments={"items": list(items)})])


def call(name: str, **arguments) -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id=f"{name}-1", name=name, arguments=arguments)])


DONE = ModelReply(content="done")
ROW = {"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"}, "expect": {"slot": "seven"},
       "fact_ids": ["f1"]}


def lookup(key: str = "A1") -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id="r1", name="lookup_rows",
                                                  arguments={"table": "items", "key": key})])


def moved_run(slot: str) -> Run:
    run = reference_run()
    for event in run.events:
        if event.payload.get("name") == "update_item":
            event.payload["args"] = {"item_id": "A1", "slot": slot}
    return run


def deleting_run() -> Run:
    run = reference_run()
    extra = [Event(idx=6, type="tool_call", payload={"id": "c2", "name": "delete_item", "args": {"item_id": "B2"},
                                                     "kind": "write"}),
             Event(idx=7, type="tool_result", payload={"id": "c2", "result": {"item_id": "B2"}})]
    return run.model_copy(update={"events": run.events[:-1] + extra + [run.events[-1]]})


def test_policy_sections_are_the_headed_lines_of_two_words_or_more():
    assert T.policy_sections(POLICY) == ["section:store policy", SECTION]


def test_the_session_ends_unfinished_after_the_read_budget():
    model = TestModel([lookup()], loop=True)
    session = T.run_session(model, [{"role": "user", "content": "x"}], STATE, None, max_rounds=2)
    assert session["unfinished"] and session["text"] == "" and session["tool_rounds"] == 3
    assert len(model.calls) == 3


def test_a_tool_the_writer_was_not_given_is_answered_as_unknown():
    call = ModelReply(tool_calls=[ToolCallRequest(id="x", name="read_run", arguments={})])
    model = TestModel([call, answer("demands", [])])
    T.run_session(model, [{"role": "user", "content": "x"}], STATE, None)
    assert model.calls[1]["messages"][-1]["content"] == "unknown tool"


@pytest.mark.parametrize("bad", [reference_run(), Verifier(task_id="t1", atoms=[]),
                                 {"events": []}, {"items": {}, "end_state": {}}])
def test_no_run_reference_verifier_or_end_state_reaches_the_writer(bad):
    with pytest.raises(T.WriterInputError):
        T.refuse_run_inputs(bad)
    with pytest.raises(T.WriterInputError):
        W.write_spec("t1", inputs(state=bad) if isinstance(bad, dict) else inputs()._replace(state=bad),
                     TestModel([DONE]))


def test_a_starting_state_table_named_like_a_run_field_is_still_a_table():
    T.refuse_run_inputs({"events": {"e1": {"id": "e1"}}, "atoms": {}})


def test_a_ruling_excerpt_may_not_be_a_run():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    with pytest.raises(ValidationError):
        W.repair(spec, {"check_ids": ["c0"], "reason": "r", "excerpt": reference_run()}, inputs(),
                 TestModel([answer("checks", [])]))


def test_spoken_text_becomes_one_volunteered_fact_per_sentence():
    intent = W.spoken_intent("t1", "Move item A1 to slot seven. Is that possible?\n\nThanks!")
    assert [fact.text for fact in intent.facts] == ["Move item A1 to slot seven.", "Is that possible?", "Thanks!"]
    assert {fact.stance for fact in intent.facts} == {"volunteered"}
    assert intent.text == intent_text(intent.facts)


def _workdir(root):
    """A minimal Environment on disk, with a Run and a Verifier that carry a sentinel."""
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / "t1.json").write_text(json.dumps({"id": "t1", "run_ids": []}))
    (root / "db.json").write_text(json.dumps(STATE))
    (root / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"},
                                                     {"name": "list_items", "kind": "read"}]))
    (root / "overlays").mkdir()
    (root / "overlays" / "t1.json").write_text(json.dumps({"overlay": {"task_id": "t1", "rows": []}, "values": {}}))
    (root / "env").mkdir()
    (root / "env" / "policy.md").write_text(POLICY)
    (root / "runs" / "t1").mkdir(parents=True)
    (root / "runs" / "t1" / "ref.jsonl").write_text(json.dumps({"type": "stop", "payload": {"sentinel": "SEEN"}}))
    (root / "verifiers").mkdir()
    (root / "verifiers" / "t1.json").write_text(json.dumps({"task_id": "t1", "atoms": [], "note": "SEEN"}))


def test_inputs_come_from_the_environment_files_and_never_from_runs_or_verifiers(tmp_path):
    _workdir(tmp_path)
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert loaded.policy_text == POLICY and loaded.sections == T.policy_sections_of(tmp_path)
    assert loaded.state == STATE and loaded.write_tools == {"update_item"}
    model = TestModel([call("read_row", table="items", key="A1"), DONE])
    W.write_spec("t1", loaded, model)
    assert "SEEN" not in json.dumps(model.calls[0]["messages"])


def test_a_tool_the_schema_records_as_an_action_is_kept_out_of_the_write_tools(tmp_path):
    _workdir(tmp_path)
    (tmp_path / "tool_sigs.json").write_text(json.dumps([{"name": "update_item", "kind": "write"},
                                                         {"name": "close_case", "kind": "write"}]))
    (tmp_path / "schema.json").write_text(json.dumps({"tables": ["actions"], "columns": [
        {"table": "actions", "name": "tool", "class": "hard", "evidence": {"actions_of": ["close_case"]}}]}))
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert loaded.write_tools == {"update_item"} and loaded.action_tools == {"close_case"}


def test_policy_sections_of_joins_the_policy_headings_and_the_compiled_rule_names(tmp_path):
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "policy.md").write_text(POLICY)
    (tmp_path / "constraints_check.json").write_text(json.dumps(
        {"constraints": [{"name": "Moving  Items"}, {"id": "no_double_slot"}, {"text": "unnamed"}]}))
    assert T.policy_sections_of(tmp_path) == ["section:moving items", "section:no_double_slot",
                                              "section:store policy"]


def test_policy_sections_of_reads_the_recorded_prompts_when_no_policy_is_shipped(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "a.json").write_text(json.dumps({"system_prompt": POLICY}))
    (tmp_path / "traces" / "b.json").write_text(json.dumps({"system_prompt": "## Slot limits apply\n"}))
    assert T.policy_sections_of(tmp_path) == ["section:moving items", "section:slot limits apply",
                                              "section:store policy"]


def test_a_cut_body_is_counted_and_the_prompt_says_so():
    long = inputs()._replace(policy_text="x" * (W.BODY_CHARS + 500))
    model = TestModel([DONE])
    written = W.write_spec("t1", long, model)
    content = model.calls[0]["messages"][0]["content"]
    assert written.counts["truncated_chars"] > 500 and content.endswith("read_policy shows any section whole)")
    assert W.write_spec("t1", inputs(), TestModel([DONE])).counts["truncated_chars"] == 0


def test_a_spent_ceiling_makes_the_next_call_the_answer():
    model = TestModel([lookup(), answer("demands", [])])
    session = T.run_session(model, [{"role": "user", "content": "x"}], STATE, W._config(None),
                            price=lambda usage: 1.0, force_answer=True, ceiling_usd=0.5)
    assert [call["config"].tool_choice for call in model.calls] == ["none"]
    assert session["tool_uses"] == [] and "unfinished" in session




def test_the_writer_reads_the_world_adds_items_and_a_refused_value_comes_back_with_its_reason():
    invented = dict(ROW, id="s2", expect={"slot": "nine"})
    model = TestModel([call("find_rows", value="A1"), add(ROW, invented), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert [check.id for check in written.spec.checks] == ["s1"]
    told = model.calls[2]["messages"][-1]["content"]
    assert "s1: kept (row A1)" in told and "s2: refused" in told and "not in the world" in told
    assert json.loads(model.calls[1]["messages"][-1]["content"]) == [{"table": "items", "key": "A1"}]
    check = written.spec.checks[0]
    assert (check.gate, check.weight, check.fact_ids, check.tier) == (True, 1.0, ["f1"], "critical")
    assert written.counts["offered"] == 2 and written.counts["refused"] == 1 and written.spec.gaps == ["f2"]
    assert written.counts["by_kind"] == {"nothing_else": 1, "row_is": 1}
    assert model.calls[0]["tools"] == T.WORLD_TOOLS


def test_a_value_set_and_a_policy_line_item_keep_their_provenance_on_the_check():
    either = dict(ROW, expect={"slot": {"one_of": ["seven", "four"]}})
    confirm = {"id": "e1", "kind": "before", "first": "confirm_turn", "then": "update_item",
               "policy_line": "Move an item only to a free slot."}
    spec = W.write_spec("t1", inputs(), TestModel([add(either, confirm), DONE])).spec
    by_id = {check.id: check for check in spec.checks}
    assert by_id["e1"].policy_line == "Move an item only to a free slot." and by_id["e1"].fact_ids == []
    assert by_id["e1"].because.startswith("section:Moving items: ")
    assert by_id["s1"].demand["expect"] == {"slot": {"one_of": ["seven", "four"]}}


def test_the_compiled_verifier_passes_the_right_run_and_fails_a_wrong_slot_and_a_stray_delete():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    verifier = W.verifier_of(spec, inputs()).verifier
    assert [atom.id for atom in verifier.atoms] == ["s1", "sanity"]
    assert all(atom.kind == "allowed" and atom.gate for atom in verifier.atoms)
    assert verifier.atoms[0].target["from"] == {"fact_ids": ["f1"], "policy_line": None}

    def ok(run):
        context = AtomContext(run)
        return _matches(verifier.expected, context), score_items(verifier, context)

    end = {"items": {"A1": {"item_id": "A1", "slot": "seven"}, "B2": STATE["items"]["B2"]}}
    assert ok(_ended(end)) == (True, {"s1": True, "sanity": True})
    assert ok(_ended({"items": {**end["items"], "A1": {"item_id": "A1", "slot": "nine"}}}))[0] is False
    assert ok(_ended({"items": {"A1": end["items"]["A1"]}})) == (False, {"s1": True, "sanity": False})


def _ended(end: dict) -> Run:
    return Run(run_id="r", task_id="t1", events=[Event(idx=0, type="stop", payload={"start_state": STATE,
                                                                                    "end_state": end})])


def test_items_no_run_can_pass_together_go_back_once_and_a_left_clash_is_named_unsatisfiable():
    other = dict(ROW, id="s2", expect={"slot": "four"})
    fixed = TestModel([add(ROW, other), DONE, call("drop_items", ids=["s2"]), DONE])
    written = W.write_spec("t1", inputs(), fixed)
    assert written.counts["sent_back"] == 1 and written.counts["unsatisfiable"] == []
    assert "s2" in fixed.calls[2]["messages"][-1]["content"] and [c.id for c in written.spec.checks] == ["s1"]
    stuck = W.write_spec("t1", inputs(), TestModel([add(ROW, other), DONE, DONE]))
    assert stuck.counts["unsatisfiable"] == ["s2"]


def test_a_silent_writer_yields_an_empty_spec_with_every_fact_a_gap():
    written = W.write_spec("t1", inputs(), TestModel([ModelReply(content="I cannot decide.")]))
    assert written.spec.checks == [] and written.spec.gaps == ["f1", "f2"]


def test_the_prompt_marks_unwitnessed_facts_and_carries_policy_tools_and_the_item_grammar():
    model = TestModel([DONE])
    W.write_spec("t1", inputs(), model)
    content = model.calls[0]["messages"][0]["content"]
    body = json.loads(content[content.index("\n{") + 1:])
    assert body["intent"][1]["unwitnessed"] is True and body["policy_text"] == POLICY and body["tools"] == TOOLS
    assert body["policy_sections"] == ["Store policy", "Moving items"]
    for kind in ("row_is", "row_new", "row_keeps", "called", "not_called", "before", "said", "judge"):
        assert kind in content
    assert "never seen any agent run" in content and content.index("# Stop") > content.index("# Feedback")


def test_write_verifier_writes_the_spec_and_one_verifier_the_runner_and_the_spec_share(tmp_path):
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    paths = W.write_verifier(spec, tmp_path, inputs())
    assert paths == [tmp_path / "spec" / "t1.json", tmp_path / "spec" / "verifiers" / "t1.json",
                     W.runner_verifier_path(tmp_path, "t1")]
    runner = Verifier.model_validate_json(W.runner_verifier_path(tmp_path, "t1").read_text())
    assert runner == Verifier.model_validate_json(W.spec_verifier_path(tmp_path, "t1").read_text())
    counts = load_spec(tmp_path, "t1").end_state
    assert "reference" not in counts and counts["end_states"] == 1 and counts["cells"] == 1
    assert counts["gate"] == 2 and counts["by_kind"] == {"nothing_else": 1, "row_is": 1}


def test_repair_drops_only_ruled_items_and_adds_under_the_same_rules():
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW, told), DONE])).spec
    ruling = W.Ruling(check_ids=["t1"], reason="the value is not what the user hears", excerpt="x" * 5000)
    model = TestModel([call("drop_items", ids=["t1", "s1"]),
                       add(dict(told, id="t2", values=["slot seven"]), dict(told, id="t3", values=["eleven"])), DONE])
    written = W.repair(spec, ruling, inputs(), model)
    assert [check.id for check in written.spec.checks] == ["s1", "t2"]
    assert written.spec.version == spec.version + 1 and written.counts["refused"] == 1
    shown = model.calls[0]["messages"][0]["content"]
    assert "x" * W.MAX_EXCERPT in shown and "x" * (W.MAX_EXCERPT + 1) not in shown
    assert "s1: not ruled, kept" in model.calls[1]["messages"][-1]["content"]


def test_a_repair_may_not_replace_an_unruled_item_and_adds_at_most_the_cap():
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW, told), DONE])).spec
    ruling = W.Ruling(check_ids=["t1"], reason="a fact is uncovered", item="f2", kind="derivation/uncovered_fact")
    extra = [dict(told, id=f"n{n}") for n in range(W.MAX_ADDS + 1)]
    model = TestModel([add(dict(ROW, expect={"slot": "four"}), *extra), DONE])
    written = W.repair(spec, ruling, inputs(), model)
    kept = {check.id: check for check in written.spec.checks}
    assert kept["s1"] == spec.checks[0] and written.counts["not_ruled"] == 1
    assert written.counts["added"] == W.MAX_ADDS and f"n{W.MAX_ADDS}" not in kept
    shown = model.calls[1]["messages"][-1]["content"]
    assert "s1: refused: not ruled" in shown and f"at most {W.MAX_ADDS} items" in shown
    assert "uncovered_fact" in model.calls[0]["messages"][0]["content"]


def test_a_rebuttal_changes_nothing_and_carries_the_writers_why():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    ruling = W.Ruling(check_ids=["s1"], reason="not asked for", fix="drop s1")
    written = W.repair(spec, ruling, inputs(), TestModel([ModelReply(content="rebut: the user asks for slot seven.")]))
    assert written.spec.checks == spec.checks
    assert written.counts["rebut"] == "the user asks for slot seven."
    assert not (written.counts["changed"] or written.counts["dropped"] or written.counts["added"])


def _ruled_workdir(tmp_path):
    _workdir(tmp_path)
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    spec = W.write_spec("t1", loaded, TestModel([add(ROW), DONE])).spec
    W.write_verifier(spec, tmp_path, loaded)
    return spec


def _bus_events(tmp_path):
    lines = (tmp_path / "bus.jsonl").read_text().splitlines()
    return [json.loads(line)["event"] for line in lines]


def test_repair_for_ruling_saves_the_changed_spec_rewrites_both_verifiers_and_publishes_repaired(tmp_path):
    spec = _ruled_workdir(tmp_path)
    ruling = {"number": 1, "target": "check", "check_ids": ["s1"], "reason": "either slot is fine",
              "code": "other", "excerpt": "policy allows several"}
    model = TestModel([add(dict(ROW, expect={"slot": {"one_of": ["seven", "four"]}})), DONE])
    repaired = W.repair_for_ruling(tmp_path, "t1", ruling, model, 1.0)
    assert repaired.version == spec.version + 1 and load_spec(tmp_path, "t1") == repaired
    runner = W.runner_verifier_path(tmp_path, "t1").read_text()
    assert runner == W.spec_verifier_path(tmp_path, "t1").read_text() and '"one_of"' in runner
    (event,) = _bus_events(tmp_path)
    assert event["name"] == "spec.repaired" and event["payload"]["check_ids"] == ["s1"]


def test_repair_for_ruling_publishes_defended_when_no_item_changes(tmp_path):
    spec = _ruled_workdir(tmp_path)
    ruling = {"number": 2, "target": "check", "check_ids": ["s1"], "reason": "not asked", "code": "other"}
    defended = W.repair_for_ruling(tmp_path, "t1", ruling, TestModel([DONE]))
    assert defended.checks == spec.checks
    (event,) = _bus_events(tmp_path)
    assert event["name"] == "spec.defended" and event["payload"]["check_ids"] == ["s1"]


def test_refusal_counts_keep_the_reasons_words_and_never_the_refused_value():
    store = W.store_of(inputs())
    store.add([{"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"},
                "expect": {"slot": "nine_42"}, "fact_ids": ["f1"]}])
    assert W._store_counts(store)["refused_why"] == {"value is not in the world the policy or": 1}
