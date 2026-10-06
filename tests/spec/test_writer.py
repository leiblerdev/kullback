"""The Verifier writer: a scripted model writes demands, code keeps what is grounded and compiles it.

The model is the harness TestModel with scripted replies; the Starting state, tools and facts are the
invented ones of tests/spec/fixtures.py. Verdicts go through the real check_run.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner.records import Event, Run, Verifier
from kullback.runner.target import check_run
from kullback.spec import writer as W
from kullback.spec import writer_tools as T
from kullback.spec.schema import SpecIntent, intent_text, load_spec
from tests.spec.fixtures import FACTS, WRITE, WRITE_TOOLS, fn, reference_run

STATE = {"items": {"A1": {"item_id": "A1", "slot": "two"}, "B2": {"item_id": "B2", "slot": "four"}}}
SECTION = "section:moving items"
POLICY = "# Store policy\n\n## Moving items\n\nMove an item only to a free slot.\n"
TOOLS = [{"name": "list_items", "kind": "read", "args": []},
         {"name": "update_item", "kind": "write", "args": ["item_id", "slot"]},
         {"name": "delete_item", "kind": "write", "args": ["item_id"]}]
MOVE = dict(WRITE, kind="required", fact_ids=["f1"], because="move item A1 to slot seven")


def inputs(facts=FACTS, state=STATE) -> W.WriterInputs:
    intent = SpecIntent(task_id="t1", facts=list(facts), text=intent_text(list(facts)))
    return W.WriterInputs(intent=intent, policy_text=POLICY, sections=T.policy_sections(POLICY), tools=TOOLS,
                          write_tools=set(WRITE_TOOLS) | {"delete_item"}, state=state, constraints=[], canon=fn)


def answer(key: str, items: list) -> ModelReply:
    return ModelReply(content=json.dumps({key: items}))


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


def test_writer_reads_the_starting_state_then_keeps_only_grounded_witnessed_checks():
    model = TestModel([lookup(), answer("demands", [
        MOVE,
        dict(WRITE, kind="allowed", fact_ids=["f1"], because="it seems sensible"),
        {"kind": "required", "demand": "say", "text": "urgent", "fact_ids": ["f2"],
         "because": "I also want a note saying it was urgent"},
        {"kind": "made_up", "demand": "cap", "count": 1, "because": "move item A1 to slot seven"},
        {"kind": "required", "demand": "cap", "count": 1, "because": "per the moving items rules"},
    ]), answer("demands", [])])
    written = W.write_spec("t1", inputs(), model)
    assert [check.id for check in written.spec.checks] == ["c0", "c4"]
    assert written.counts == {"demands": 5, "kept": 2, "ungrounded": 1, "unwitnessed": 1, "condition_unquoted": 0,
                              "bad_kind": 1, "gaps": 1, "sent_back": 2, "sent_back_kept": 0, "parse_error": "",
                              "truncated_chars": 0, "unsatisfiable_sent_back": 0, "unsatisfiable": [],
                              "write_predicate": 0}
    assert written.spec.gaps == ["f2"]
    assert written.session["tool_uses"] == ["lookup_rows"]
    tool_turn = model.calls[1]["messages"][-1]
    assert json.loads(tool_turn["content"]) == STATE["items"]["A1"]


def test_a_write_naming_only_a_section_goes_back_once_and_its_grounded_correction_keeps_its_check_id():
    by_section = dict(WRITE, id="d1", kind="allowed", fact_ids=["f1"], because="per the moving items rules")
    cap = {"id": "d0", "kind": "required", "demand": "cap", "count": 1, "because": "per the moving items rules"}
    fixed = dict(by_section, because='per the moving items rules: "Move an item only to a free slot."')
    stray = dict(MOVE, id="d7")
    model = TestModel([answer("demands", [cap, by_section]), answer("demands", [fixed, stray])])
    written = W.write_spec("t1", inputs(), model)
    assert [check.id for check in written.spec.checks] == ["c0", "c1"]
    assert written.spec.checks[1].because == fixed["because"]
    assert (written.counts["condition_unquoted"], written.counts["sent_back"], written.counts["sent_back_kept"]) \
        == (1, 1, 1)
    first, second = model.calls[0]["messages"], model.calls[1]["messages"]
    assert second[:1] == first and second[1]["role"] == "assistant" and second[2]["role"] == "user"
    refused = json.loads(second[2]["content"][len(W.FOLLOWUP_PROMPT):])["refused"]
    assert refused == [{"id": "d1", "reason": "condition_unquoted", "rule": W.SENT_BACK_RULES["condition_unquoted"]}]


def test_a_no_write_beside_a_required_write_goes_back_once_and_the_kept_correction_stands():
    no_write = {"id": "d1", "kind": "required", "demand": "no_write", "fact_ids": ["f1"],
                "because": "move item A1 to slot seven"}
    model = TestModel([answer("demands", [dict(MOVE, id="d0"), no_write]), answer("demands", [dict(MOVE, id="d0")])])
    written = W.write_spec("t1", inputs(), model)
    assert [check.id for check in written.spec.checks] == ["c0"]
    assert (written.counts["unsatisfiable_sent_back"], written.counts["unsatisfiable"]) == (2, [])
    refused = json.loads(model.calls[1]["messages"][-1]["content"][len(W.FOLLOWUP_PROMPT):])["refused"]
    assert [(row["id"], row["reason"]) for row in refused] == \
        [("d0", "no_write_beside_required_write"), ("d1", "no_write_beside_required_write")]


def test_a_write_quoting_the_policy_without_naming_a_section_is_still_ungrounded():
    quoted = dict(WRITE, kind="required", fact_ids=["f1"], because="Move an item only to a free slot.")
    written = W.write_spec("t1", inputs(), TestModel([answer("demands", [quoted]), answer("demands", [])]))
    assert written.counts["ungrounded"] == 1 and written.counts["sent_back"] == 1


def test_a_correction_that_still_names_only_a_section_is_dropped():
    by_section = dict(WRITE, id="d0", kind="required", fact_ids=["f1"], because="per the moving items rules")
    model = TestModel([answer("demands", [by_section]), answer("demands", [by_section])])
    written = W.write_spec("t1", inputs(), model)
    assert written.spec.checks == [] and written.counts["sent_back_kept"] == 0 and len(model.calls) == 2


def test_the_write_rule_line_is_short_and_asks_for_the_section_and_its_quoted_sentence():
    (line,) = [row for row in W.SYSTEM_PROMPT.splitlines() if row.startswith("- A write's")]
    assert len(line) < 150 and "names the policy section and quotes the sentence" in line
    assert "a section alone fails" in line


def test_an_accepted_fact_with_a_witness_may_carry_a_check_alone():
    witnessed = [FACTS[0], FACTS[1].model_copy(update={"witnesses": ["rec3"]})]
    note = {"kind": "required", "demand": "say", "text": "urgent", "fact_ids": ["f2"],
            "because": "I also want a note saying it was urgent"}
    written = W.write_spec("t1", inputs(witnessed), TestModel([answer("demands", [MOVE, note])]))
    assert written.counts["kept"] == 2 and written.spec.gaps == []


def test_the_prompt_marks_unwitnessed_facts_and_carries_policy_and_tools():
    model = TestModel([answer("demands", [])])
    W.write_spec("t1", inputs(), model)
    content = model.calls[0]["messages"][0]["content"]
    body = json.loads(content[len(W.SYSTEM_PROMPT) + 1:])
    assert body["intent"][1] == {"id": "f2", "text": FACTS[1].text, "stance": "accepted", "unwitnessed": True}
    assert "unwitnessed" not in body["intent"][0]
    assert body["policy_text"] == POLICY and body["tools"] == TOOLS
    assert model.calls[0]["tools"] == T.READ_TOOLS


def test_a_silent_writer_yields_an_empty_spec_with_every_fact_a_gap():
    written = W.write_spec("t1", inputs(), TestModel([ModelReply(content="I cannot decide.")]))
    assert written.spec.checks == [] and written.spec.gaps == ["f1", "f2"]
    assert written.counts["parse_error"] == "no JSON object"


def test_the_session_ends_unfinished_after_the_read_budget():
    model = TestModel([lookup()], loop=True)
    session = T.run_session(model, [{"role": "user", "content": "x"}], STATE, None, max_rounds=2)
    assert session["unfinished"] and session["text"] == "" and session["tool_rounds"] == 3
    assert len(model.calls) == 3


def test_the_writer_is_made_to_answer_when_its_read_budget_is_spent():
    model = TestModel([lookup()] * 4 + [answer("demands", [MOVE])])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["kept"] == 1 and written.session["tool_rounds"] == 4
    assert [call["config"].tool_choice for call in model.calls] == [None] * 4 + ["none"]


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
                     TestModel([answer("demands", [])]))


def test_a_starting_state_table_named_like_a_run_field_is_still_a_table():
    T.refuse_run_inputs({"events": {"e1": {"id": "e1"}}, "atoms": {}})


def test_the_verifier_passes_the_right_run_and_fails_a_swapped_value_and_a_write_beyond_scope():
    written = W.write_spec("t1", inputs(), TestModel([answer("demands", [MOVE])]))
    assert written.spec.checks[0].demand["entity"] == "A1"  # kept raw: compile canonicalises, as check_run does
    compiled = W.verifier_of(written.spec, inputs())
    assert {atom.kind for atom in compiled.verifier.atoms} == {"required", "forbidden"}
    forbidden = [atom for atom in compiled.verifier.atoms if atom.kind == "forbidden"]
    assert [atom.id for atom in forbidden] == ["forbidden.delete_item"]
    tools = set(WRITE_TOOLS) | {"delete_item"}
    assert check_run(compiled.verifier, reference_run(), fn, write_tools=tools)[0]
    assert not check_run(compiled.verifier, moved_run("nine"), fn, write_tools=tools)[0]
    passed, failing = check_run(compiled.verifier, deleting_run(), fn, write_tools=tools)
    assert not passed and failing == "forbidden.delete_item"


def test_a_write_compile_drops_is_listed_as_a_gap_and_survives_a_repair():
    no_entity = dict(MOVE, entity="")
    written = W.write_spec("t1", inputs(), TestModel([answer("demands", [no_entity])]))
    assert [check.id for check in written.spec.checks] == ["c0"]
    assert written.spec.gaps == ["f2", "c0: write without entity"]
    assert W.verifier_of(written.spec, inputs()).reasons == ("c0: write without entity",)
    ruling = W.Ruling(check_ids=["c0"], reason="no entity")
    repaired, _ = W.apply_repair(written.spec, ruling, [dict(MOVE, id="c0")], inputs())
    assert repaired.gaps == ["f2"]


def test_write_verifier_writes_the_spec_and_one_verifier_the_runner_and_the_spec_share(tmp_path):
    spec = W.write_spec("t1", inputs(), TestModel([answer("demands", [MOVE])])).spec
    paths = W.write_verifier(spec, tmp_path, inputs())
    assert paths == [tmp_path / "spec" / "t1.json", tmp_path / "spec" / "verifiers" / "t1.json",
                     W.runner_verifier_path(tmp_path, "t1")]
    runner = Verifier.model_validate_json(W.runner_verifier_path(tmp_path, "t1").read_text())
    assert runner == Verifier.model_validate_json(W.spec_verifier_path(tmp_path, "t1").read_text())
    assert runner.task_id == "t1" and runner.atoms
    assert load_spec(tmp_path, "t1").end_state["reference"] is None


def test_repair_changes_or_defends_only_ruled_checks_and_every_change_must_ground():
    first = [MOVE, {"kind": "required", "demand": "cap", "count": 1, "because": "per the moving items rules"}]
    spec = W.write_spec("t1", inputs(), TestModel([answer("demands", first)])).spec
    ruling = W.Ruling(check_ids=["c0", "c1"], reason="the cap is too tight", excerpt="x" * 5000)
    model = TestModel([answer("checks", [
        {"id": "c1", "action": "change", "kind": "required", "demand": "cap", "count": 2,
         "because": "per the moving items rules"},
        {"id": "c0", "action": "change", "kind": "required", "demand": "cap", "count": 3, "because": "I think"},
        {"id": "c9", "action": "change", "kind": "required", "demand": "cap", "count": 3,
         "because": "per the moving items rules"},
    ])])
    repaired = W.repair(spec, ruling, inputs(), model)
    assert repaired.counts == {"changed": 1, "defended": 0, "refused": 1, "not_ruled": 1, "parse_error": "",
                               "truncated_chars": 0}
    assert repaired.spec.checks[0] == spec.checks[0]
    assert repaired.spec.checks[1].demand == {"demand": "cap", "count": 2}
    assert repaired.spec.version == spec.version + 1 and len(repaired.spec.checks) == 2
    body = json.loads(model.calls[0]["messages"][0]["content"][len(W.REPAIR_PROMPT) + 1:])
    assert len(body["ruling"]["excerpt"]) == W.MAX_EXCERPT
    assert [check["id"] for check in body["ruled_checks"]] == ["c0", "c1"]


def test_a_defended_check_stands_unchanged():
    spec = W.write_spec("t1", inputs(), TestModel([answer("demands", [MOVE])])).spec
    ruling = W.Ruling(check_ids=["c0"], reason="not asked for")
    model = TestModel([answer("checks", [{"id": "c0", "action": "defend", "because": "move item A1"}])])
    repaired = W.repair(spec, ruling, inputs(), model)
    assert repaired.counts["defended"] == 1 and repaired.spec.checks == spec.checks


def test_a_ruling_excerpt_may_not_be_a_run():
    spec = W.write_spec("t1", inputs(), TestModel([answer("demands", [MOVE])])).spec
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
    model = TestModel([answer("demands", [])])
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
    model = TestModel([answer("demands", [])])
    written = W.write_spec("t1", long, model)
    content = model.calls[0]["messages"][0]["content"]
    assert written.counts["truncated_chars"] > 500 and content.endswith("\n(inputs truncated)")
    assert W.write_spec("t1", inputs(), TestModel([answer("demands", [])])).counts["truncated_chars"] == 0


def test_a_spent_ceiling_makes_the_next_call_the_answer():
    model = TestModel([lookup(), answer("demands", [])])
    session = T.run_session(model, [{"role": "user", "content": "x"}], STATE, W._config(None),
                            price=lambda usage: 1.0, force_answer=True, ceiling_usd=0.5)
    assert [call["config"].tool_choice for call in model.calls] == ["none"]
    assert session["tool_uses"] == [] and "unfinished" in session


def _ruled_workdir(tmp_path):
    _workdir(tmp_path)
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    first = [MOVE, {"kind": "required", "demand": "cap", "count": 1, "because": "per the moving items rules"}]
    spec = W.write_spec("t1", loaded, TestModel([answer("demands", first)])).spec
    W.write_verifier(spec, tmp_path, loaded)
    return spec


def _bus_events(tmp_path):
    lines = (tmp_path / "bus.jsonl").read_text().splitlines()
    return [json.loads(line)["event"] for line in lines]


def test_repair_for_ruling_saves_the_changed_spec_rewrites_both_verifiers_and_publishes_repaired(tmp_path):
    spec = _ruled_workdir(tmp_path)
    ruling = {"number": 1, "target": "check", "check_ids": ["c1"], "reason": "the cap is too tight",
              "code": "check_unpassable", "quoted": ["because c1"], "excerpt": "per the moving items rules"}
    model = TestModel([answer("checks", [{"id": "c1", "action": "change", "kind": "required", "demand": "cap",
                                          "count": 2, "because": "per the moving items rules"}])])
    repaired = W.repair_for_ruling(tmp_path, "t1", ruling, model, 1.0)
    assert repaired.version == spec.version + 1 and load_spec(tmp_path, "t1") == repaired
    runner = W.runner_verifier_path(tmp_path, "t1").read_text()
    assert runner == W.spec_verifier_path(tmp_path, "t1").read_text() and '"count": 2' in runner
    (event,) = _bus_events(tmp_path)
    assert event["name"] == "spec.repaired" and event["payload"]["check_ids"] == ["c1"]
    assert event["payload"]["version"] == repaired.version


def test_repair_for_ruling_publishes_defended_when_every_ruled_check_stands(tmp_path):
    spec = _ruled_workdir(tmp_path)
    before = W.runner_verifier_path(tmp_path, "t1").read_text()
    ruling = {"number": 2, "target": "check", "check_ids": ["c0"], "reason": "not asked", "code": "other"}
    model = TestModel([answer("checks", [{"id": "c0", "action": "defend", "because": "move item A1"}])])
    defended = W.repair_for_ruling(tmp_path, "t1", ruling, model)
    assert defended.checks == spec.checks
    after = W.runner_verifier_path(tmp_path, "t1").read_text()
    assert json.loads(after)["atoms"] == json.loads(before)["atoms"]
    (event,) = _bus_events(tmp_path)
    assert event["name"] == "spec.defended" and event["payload"]["check_ids"] == ["c0"]
