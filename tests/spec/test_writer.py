"""The Verifier writer: a scripted model reads the world and adds items, code keeps what it finds and compiles it.

The model is the harness TestModel with scripted replies; the Starting state, tools and facts are the
invented ones of tests/spec/fixtures.py. Verdicts go through the real verdict.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from kullback.ai.provider import ModelReply, TestModel, ToolCallRequest
from kullback.runner.atom_context import AtomContext
from kullback.runner.expected import match_context
from kullback.runner.records import Event, Run, Verifier
from kullback.runner.verdict import verdict
from kullback.spec import writer as W
from kullback.spec import writer_tools as T
from kullback.spec.intent_tools import mark_side
from kullback.spec.items import AGENT_WEIGHED, SIDE_WEIGHED, TELL_WEIGHED, UNCONFIRMED_WEIGHED
from kullback.spec.schema import FactSource, IntentFact, ReplyQuote, SpecIntent, intent_text, load_spec
from tests.spec.fixtures import FACTS, WRITE_TOOLS, check, fn, reference_run

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
    model = TestModel([call("read_row", table="items", key="A1"), DONE, DONE])
    W.write_spec("t1", loaded, model)
    assert "SEEN" not in json.dumps(model.calls[0]["messages"])


def test_a_spec_written_with_write_tools_and_no_learned_effects_says_none_were_learned(tmp_path):
    _workdir(tmp_path)
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert loaded.effects_missing
    written = W.write_spec("t1", loaded, TestModel([add(ROW), DONE]))
    assert written.counts[W.EFFECTS_COUNT] == W.NO_EFFECTS


def _effects_file(root, effects: dict) -> None:
    from kullback.spec.effects import EFFECTS_FILE

    (root / EFFECTS_FILE).parent.mkdir(parents=True)
    (root / EFFECTS_FILE).write_text(json.dumps({"effects": effects, "signatures": {}}))


def test_a_spec_written_beside_a_learned_effects_file_carries_no_missing_effects_mark(tmp_path):
    _workdir(tmp_path)
    _effects_file(tmp_path, {"update_item": {"writes": [{"table": "items", "field": "slot", "values": ["seven"]}]}})
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert not loaded.effects_missing
    assert W.EFFECTS_COUNT not in W.write_spec("t1", loaded, TestModel([add(ROW), DONE])).counts


def test_a_spec_written_beside_an_effects_file_whose_tools_carry_no_writes_says_none_were_learned(tmp_path):
    _workdir(tmp_path)
    _effects_file(tmp_path, {"update_item": {"writes": []}})
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert loaded.effects_missing
    assert W.write_spec("t1", loaded, TestModel([add(ROW), DONE])).counts[W.EFFECTS_COUNT] == W.NO_EFFECTS


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
    model = TestModel([DONE, DONE])
    written = W.write_spec("t1", long, model)
    content = model.calls[0]["messages"][0]["content"]
    assert written.counts["truncated_chars"] > 500 and content.endswith("read_policy shows any section whole)")
    assert W.write_spec("t1", inputs(), TestModel([DONE, DONE])).counts["truncated_chars"] == 0


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
        return _matches(verifier.expected, context), {item.id: item.holds for item in verdict(run, verifier).items}

    end = {"items": {"A1": {"item_id": "A1", "slot": "seven"}, "B2": STATE["items"]["B2"]}}
    assert ok(_ended(end)) == (True, {"state:items.A1.slot": True, "sanity": True})
    assert ok(_ended({"items": {**end["items"], "A1": {"item_id": "A1", "slot": "nine"}}}))[0] is False
    assert ok(_ended({"items": {"A1": end["items"]["A1"]}})) == (False, {"state:items.A1.slot": True, "sanity": False})


def test_an_added_said_or_judge_item_or_an_older_say_check_is_never_dropped_beside_end_states():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    ruling = W.Ruling(check_ids=[], reason="a fact is uncovered", item="f1", kind="derivation/uncovered_fact")
    told = {"id": "t2", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    declined = {"id": "j1", "kind": "judge", "question": "Did the agent refuse?", "policy_line": "Move an item only to "
                "a free slot.", "anchor": {"answer": "Move an item only to a free slot."}}
    repaired = W.repair(spec, ruling, inputs(), TestModel([add(told, declined), DONE])).spec
    older = check("c9", kind="communicate", demand={"demand": "say", "values": ["seven"]})
    verifier = W.verifier_of(repaired.model_copy(update={"checks": repaired.checks + [older]}), inputs()).verifier
    assert verifier.expected and {"t2", "j1", "sanity"} <= {atom.id for atom in verifier.atoms}
    assert len(verifier.atoms) == 5, "the older say check compiles to its own atom beside the items"


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
    written = W.write_spec("t1", inputs(), TestModel([ModelReply(content="I cannot decide."), DONE]))
    assert written.spec.checks == [] and written.spec.gaps == ["f1", "f2"]


def test_the_prompt_marks_unwitnessed_facts_and_carries_policy_tools_and_the_item_grammar():
    model = TestModel([DONE, DONE])
    W.write_spec("t1", inputs(), model)
    content = model.calls[0]["messages"][0]["content"]
    body = json.loads(content[content.index("\n{") + 1:])
    assert body["intent"][1]["unwitnessed"] is True and body["policy_text"] == POLICY and body["tools"] == TOOLS
    assert body["policy_sections"] == ["Store policy", "Moving items"]
    for kind in ("row_is", "row_new", "row_keeps", "called", "not_called", "before", "said", "judge"):
        assert kind in content
    assert "never seen any agent run" in content and content.index("# Stop") > content.index("# Feedback")


def test_the_facts_view_shows_the_after_pushback_flag():
    flagged = FACTS[0].model_copy(update={"after_pushback": True})
    model = TestModel([DONE, DONE])
    W.write_spec("t1", inputs(facts=[flagged, FACTS[1]]), model)
    content = model.calls[0]["messages"][0]["content"]
    body = json.loads(content[content.index("\n{") + 1:])
    assert body["intent"][0]["after_pushback"] is True and "after_pushback" not in body["intent"][1]


def test_a_weighed_gate_tells_the_writer_why_on_the_kept_line():
    flagged = FACTS[0].model_copy(update={"stance": "fallback", "condition": "agent"})
    store = W.store_of(inputs(facts=[flagged, FACTS[1]]))
    out = store.add([dict(ROW, fact_ids=["f1"])])
    assert "kept" in out and "gate weighed" in out and "cites a reaction" in out
    assert store.items["s1"]["gate"] is False and store.refused == []


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


def test_a_repair_may_not_replace_an_unruled_item_and_a_wholly_refused_answer_is_retried_once():
    told = {"id": "t1", "kind": "said", "values": ["seven"], "fact_ids": ["f1"]}
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW, told), DONE])).spec
    ruling = W.Ruling(check_ids=["t1"], reason="a fact is uncovered", item="f2", kind="derivation/uncovered_fact")
    model = TestModel([add(dict(ROW, expect={"slot": "four"})), DONE, add(dict(told, id="n1")), DONE])
    written = W.repair(spec, ruling, inputs(), model)
    kept = {check.id: check for check in written.spec.checks}
    assert kept["s1"] == spec.checks[0] and written.counts["not_ruled"] == 1
    assert written.counts["retried"] == 1 and written.counts["added"] == 1 and "n1" in kept
    retry = model.calls[2]["messages"][-1]["content"]
    assert "Code refused every item" in retry and "s1: not ruled" in retry
    assert "uncovered_fact" in model.calls[0]["messages"][0]["content"]


def test_a_rebuttal_changes_nothing_and_carries_the_writers_why():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    ruling = W.Ruling(check_ids=["s1"], reason="not asked for", fix="drop s1")
    written = W.repair(spec, ruling, inputs(), TestModel([ModelReply(content="rebut: the user asks for slot seven.")]))
    assert written.spec.checks == spec.checks
    assert written.counts["rebut"] == "the user asks for slot seven."
    assert not (written.counts["changed"] or written.counts["dropped"] or written.counts["added"])


def test_the_repair_session_asks_a_rebuttal_to_quote_the_user_or_the_policy_else_apply():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    model = TestModel([ModelReply(content="rebut: the user asks for slot seven.")])
    W.repair(spec, W.Ruling(check_ids=["s1"], reason="not asked for", fix="drop s1"), inputs(), model)
    prompt = model.calls[0]["messages"][0]["content"]
    assert "quotes those words or that line" in prompt and "With neither, apply the ruling." in prompt
    assert 'A reply that changes nothing always starts with "rebut:"' in prompt


def test_the_repair_session_fixes_what_the_ruling_names_and_never_copies_an_item():
    spec = W.write_spec("t1", inputs(), TestModel([add(ROW), DONE])).spec
    model = TestModel([DONE])
    W.repair(spec, W.Ruling(check_ids=["s1"], reason="wrong slot", fix="expect slot four"), inputs(), model)
    prompt = model.calls[0]["messages"][0]["content"]
    assert "fix what the ruling names" in prompt
    assert "Drop a ruled item only when it asks for something the user never asked for" in prompt
    assert "never add a copy of an item" in prompt


def test_the_writer_session_gates_only_what_the_user_asked_and_the_policy_requires():
    model = TestModel([DONE, DONE])
    W.write_spec("t1", inputs(), model)
    prompt = model.calls[0]["messages"][0]["content"]
    assert "Gate what the user asked for and the policy requires" in prompt
    assert "A fact of condition agent is a reaction" in prompt and "gate the branch the world selects" in prompt
    assert "accept any allowed one or weight it" in prompt
    assert "gates only on a keep fact the user said or the policy line that forbids it; else weight it" in prompt
    assert "one item per ask" in prompt
    assert "A rule the policy does not state is never gated" in " ".join(prompt.split())
    assert "Any other told item (judge, said) gates only citing a told fact" in " ".join(prompt.split())

def test_a_refusal_judge_item_is_still_gated_by_the_prompt_rule():
    assert "A refusal judge item always gates." in W.CHOICE_RULE
    assert "A refusal judge item always gates." in W.SYSTEM_PROMPT

def test_a_value_the_user_asked_to_be_told_is_required_not_only_scored():
    weighted = dict(ROW, gate=False)
    model = TestModel([add(weighted), DONE, add(ROW), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["asked_sent_back"] == 1
    assert [c.id for c in written.spec.checks] == ["s1"]
    assert written.spec.checks[0].gate is True
    sent = json.dumps(model.calls[2]["messages"])
    assert "f1" in sent and "weighted only" in sent


def test_a_volunteered_fact_with_no_item_goes_back_as_uncovered():
    missing = W.asked_missing([], inputs().intent)
    assert missing == [{"id": "f1", "reason": "uncovered"}]
    assert W.asked_missing([dict(ROW, gate=False)], inputs().intent) == [
        {"id": "f1", "reason": "weighted only"}]
    assert W.asked_missing([ROW], inputs().intent) == []


def test_a_reaction_and_an_unfounded_fact_are_never_sent_back_and_after_pushback_alone_is_evidence():
    intent = inputs().intent
    pushed = intent.facts[0].model_copy(update={"after_pushback": True})
    assert W.asked_missing([], intent.model_copy(update={"facts": [pushed, intent.facts[1]]})) == [
        {"id": "f1", "reason": "uncovered"}]
    reaction = intent.facts[0].model_copy(update={"stance": "fallback", "condition": "agent"})
    intent = intent.model_copy(update={"facts": [reaction, intent.facts[1]]})
    assert W.asked_missing([], intent) == []
    stray = intent.facts[0].model_copy(update={"after_pushback": False, "unfounded": True})
    intent = intent.model_copy(update={"facts": [stray, intent.facts[1]]})
    assert W.asked_missing([], intent) == []


def test_a_writer_with_nothing_to_gate_answers_why_not_and_leaves_the_spec():
    model = TestModel([add(dict(ROW, gate=False)), DONE, DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["asked_sent_back"] == 1
    assert written.spec.checks[0].gate is False
    assert "call rebut_fact" in model.calls[2]["messages"][-1]["content"]
    assert written.spec.rebuttals == {} and "rebuttals" not in written.spec.model_dump()


def test_a_rebuttal_is_kept_only_for_a_sent_back_fact_and_only_when_its_why_quotes_the_fact_or_policy():
    store = W.store_of(inputs())
    quoted = "the user said only: move item A1 to slot seven"
    assert "refused: " + T.NOT_SENT_BACK in store.rebut("f1", quoted)
    store.rebuttable = {"f1"}
    assert "refused: " + T.UNQUOTED_WHY in store.rebut("f1", "it needs no gate, trust me")
    assert "refused: no such fact" in store.rebut("f9", quoted)
    assert store.rebut("f1", quoted) == "f1: rebuttal kept" and store.rebuttals == {"f1": quoted}
    assert [row["kind"] for row in store.refused] == ["rebut_fact"] * 3


def test_a_rebuttal_in_the_follow_up_is_recorded_on_the_spec_by_fact_id():
    quoted = "the user said only: move item A1 to slot seven"
    model = TestModel([add(dict(ROW, gate=False)), DONE, call("rebut_fact", fact_id="f1", why=quoted), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.spec.rebuttals == {"f1": quoted} and written.counts["rebutted"] == 1
    assert written.spec.model_dump()["rebuttals"] == {"f1": quoted}


def test_the_prompt_gates_a_told_value_and_a_named_option():
    assert "A value the user" in W.CHOICE_RULE and "anchored on the world row holding it" in W.CHOICE_RULE
    assert "One of several options the world holds that the user named" in " ".join(W.CHOICE_RULE.split())
    assert "Leave text and judge items ungated" not in W.CHOICE_RULE


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


def test_an_unchecked_ask_repair_drops_the_facts_old_rebuttal_and_keeps_a_new_one_that_quotes(tmp_path):
    from kullback.spec.schema import save_spec

    spec = _ruled_workdir(tmp_path)
    save_spec(tmp_path, spec.model_copy(update={"rebuttals": {"f2": "an old why"}}))
    ruling = {"number": 3, "item": "task", "kind": "derivation", "code": "unchecked_ask",
              "verified": [{"fact_id": "f2"}], "reason": "f2 is an ask no gate cites", "fix": "gate it or rebut"}
    assert W.repair_for_ruling(tmp_path, "t1", ruling, TestModel([DONE])).rebuttals == {}
    quoted = "the user wants: a note saying it was urgent"
    rebut = TestModel([call("rebut_fact", fact_id="f2", why=quoted), DONE])
    assert W.repair_for_ruling(tmp_path, "t1", ruling, rebut).rebuttals == {"f2": quoted}
    other = dict(ruling, code="other", verified=[])
    keep = TestModel([call("rebut_fact", fact_id="f2", why="a new why"), DONE, DONE])
    assert W.repair_for_ruling(tmp_path, "t1", other, keep).rebuttals == {"f2": quoted}


def test_refusal_counts_name_a_family_of_code_wording_never_an_empty_key_or_a_value():
    store = W.store_of(inputs())
    store.add([{"id": "s1", "kind": "row_is", "table": "items", "find": {"item_id": "A1"},
                "expect": {"slot": "nine_42"}, "fact_ids": ["f1"]}])
    assert W._store_counts(store)["refused_why"] == {"value not found": 1}


def test_every_reason_code_gives_lands_in_a_named_family_and_an_empty_one_reads_no_reason():
    reasons = ["anchor is {answer, accepted: [...], reject: [...], row}: what a met checkpoint states",
               "anchor.row is {table, find: {field: value}, fields: [...]}: the row you found the answer in",
               "free is {field: one line on why a tool sets it to a value you cannot find}",
               "name the Intent facts the item answers (fact_ids) or quote the policy line (policy_line)",
               "slot: value \"paid by card\" rests only on echo facts (the user said yes to the agent's proposal)"]
    assert [W.reason_family(r) for r in reasons] == ["anchor shape", "anchor shape", "free field", "no provenance",
                                                     "echo value at a gate"]
    assert W.reason_family("") == W.NO_REASON == "no reason"


def test_a_writing_spec_with_no_confirm_item_is_asked_the_fixed_question_and_then_never_again(tmp_path):
    _ruled_workdir(tmp_path)
    confirm = {"id": "b1", "kind": "before", "first": "confirm_turn", "then": "update_item", "gate": True,
               "policy_line": "Move an item only to a free slot."}
    model = TestModel([add(confirm), DONE])
    assert W.ask_confirm(tmp_path, "t1", model) == {"asked": True, "added": 1}
    assert W.CONFIRM_QUESTION in json.dumps(model.calls[0]["messages"])
    assert W.ask_confirm(tmp_path, "t1", TestModel([])) == {"asked": False}, "the Spec now waits for a yes"
    assert W.ask_confirm(tmp_path, "t1", None) == {"asked": False}


def test_every_writer_session_is_saved_whole_and_the_next_session_is_never_shown_it(tmp_path):
    _ruled_workdir(tmp_path)
    ruling = {"number": 1, "check_ids": ["s1"], "reason": "not asked", "code": "unasked_item", "kind": "derivation"}
    model = TestModel([call("read_row", table="items", key="A1"),
                       ModelReply(content="rebut: the user asks for slot seven, said once.")])
    W.repair_for_ruling(tmp_path, "t1", ruling, model)
    first, second = [json.loads(line) for line in W.sessions_path(tmp_path, "t1").read_text().splitlines()]
    assert (first["kind"], first["round"], second["kind"], second["round"]) == ("write", 1, "repair", 1)
    assert second["ruling"]["check_ids"] == ["s1"] and second["counts"]["rebut"].startswith("the user asks")
    (conversation,) = second["transcripts"]
    assert [m["role"] for m in conversation] == ["system", "user", "assistant", "tool", "assistant"][-len(conversation):]
    assert conversation[-3]["tool_calls"][0]["name"] == "read_row" and "A1" in conversation[-2]["content"]
    assert conversation[-1]["content"].startswith("rebut:")
    assert first["transcripts"][0][-1]["content"] == "done"
    shown = json.dumps([c["messages"] for c in model.calls])
    assert "exam/writer" not in shown and '"kind": "write"' not in shown


# --- side facts (D361) -----------------------------------------------------------------------------

TOLD = IntentFact(id="f3", text="Tell me which slot item B2 sits in.", stance="volunteered", kind="told",
                  source=FactSource(recording="rec1", turn=4))
SAID_SLOT = {"id": "t1", "kind": "said", "values": ["seven"], "gate": True}


def _side_facts():
    """A write ask, an accepted fact and a told fact beside them: code marks the told fact side."""
    return mark_side([FACTS[0].model_copy(update={"kind": "do"}), FACTS[1].model_copy(update={"kind": "do"}), TOLD])


def test_a_gate_resting_on_a_side_fact_is_weighed_with_the_reason_kept_on_the_item():
    store = W.store_of(inputs(facts=_side_facts()))
    out = store.add([dict(SAID_SLOT, fact_ids=["f3"])])
    assert "kept" in out and "gate weighed" in out and "side fact" in out and store.refused == []
    assert store.items["t1"]["gate"] is False and store.items["t1"]["weighed"] == SIDE_WEIGHED


def test_a_policy_line_does_not_lift_a_gate_resting_on_a_side_fact():
    store = W.store_of(inputs(facts=_side_facts()))
    store.add([dict(SAID_SLOT, fact_ids=["f3"], policy_line="Move an item only to a free slot.")])
    assert store.items["t1"]["gate"] is False


def test_a_told_item_citing_a_side_fact_beside_the_write_ask_is_weighed():
    store = W.store_of(inputs(facts=_side_facts()))
    out = store.add([dict(SAID_SLOT, fact_ids=["f1", "f3"])])
    assert store.items["t1"]["gate"] is False and "citing only the ask's facts" in out


def test_a_state_item_citing_a_side_fact_beside_the_write_ask_stays_a_gate():
    store = W.store_of(inputs(facts=_side_facts()))
    store.add([dict(ROW, fact_ids=["f1", "f3"])])
    assert store.items["s1"]["gate"] is True and "weighed" not in store.items["s1"]


def test_a_told_fact_in_a_task_with_no_write_ask_still_gates():
    facts = mark_side([TOLD])
    store = W.store_of(inputs(facts=facts))
    store.add([dict(SAID_SLOT, values=["four"], fact_ids=["f3"])])
    assert store.items["t1"]["gate"] is True and not facts[0].side


def test_a_told_gate_on_a_do_fact_with_no_policy_line_is_weighed_as_untold():
    store = W.store_of(inputs(facts=[FACTS[0].model_copy(update={"kind": "do"})]))
    out = store.add([dict(SAID_SLOT, fact_ids=["f1"])])
    assert "gate weighed" in out and "no fact asks to be told" in out
    assert store.items["t1"]["gate"] is False and store.items["t1"]["weighed"] == TELL_WEIGHED


def test_a_told_gate_citing_a_told_fact_or_quoting_a_policy_line_stays_a_gate():
    told = W.store_of(inputs(facts=[FACTS[0].model_copy(update={"kind": "do"}), TOLD]))
    told.add([dict(SAID_SLOT, fact_ids=["f3"])])
    ruled = W.store_of(inputs(facts=[FACTS[0].model_copy(update={"kind": "do"})]))
    ruled.add([dict(SAID_SLOT, fact_ids=["f1"], policy_line="Move an item only to a free slot.")])
    assert told.items["t1"]["gate"] is True and "weighed" not in told.items["t1"]
    assert ruled.items["t1"]["gate"] is True and "weighed" not in ruled.items["t1"]


def test_a_state_gate_on_a_do_fact_with_no_policy_line_is_never_weighed_as_untold():
    store = W.store_of(inputs(facts=[FACTS[0].model_copy(update={"kind": "do"})]))
    store.add([dict(ROW, fact_ids=["f1"])])
    assert store.items[ROW["id"]]["gate"] is True and "weighed" not in store.items[ROW["id"]]


def test_a_side_fact_is_never_sent_back_by_the_asked_followup():
    intent = SpecIntent(task_id="t1", facts=_side_facts())
    assert W.asked_missing([ROW], intent) == []


def test_the_writer_shows_the_side_flag_and_counts_side_facts_and_weighed_gates():
    model = TestModel([add(ROW, dict(SAID_SLOT, fact_ids=["f3"])), DONE])
    written = W.write_spec("t1", inputs(facts=_side_facts()), model)
    content = model.calls[0]["messages"][0]["content"]
    body = json.loads(content[content.index("\n{") + 1:])
    assert body["intent"][2]["side"] is True and "side" not in body["intent"][0]
    assert written.counts["side_facts"] == 1 and written.counts["side_weighed"] == 1
    assert written.counts["asked_sent_back"] == 0
    assert "never gated" in W.CHOICE_RULE


# The writer's files and message body over facts with no kind: the body as the code before D361 wrote it, the
# files as D369 writes them (the said gate cites no told fact and no policy line, so it is weighed).
NO_KIND_HASHES = {
    "spec/t1.json": "7f7100b7af07c4e9d85ce5b5502597ba06064546c4a1162ba523d22d61119dc7",
    "spec/verifiers/t1.json": "c42527b75c96fb36b1d8c9e9ab7d9437e36fbb2643e70fe05b2131abc22d4308",
    "verifiers/t1.json": "c42527b75c96fb36b1d8c9e9ab7d9437e36fbb2643e70fe05b2131abc22d4308",
    "body": "4c4054e384426bc357e4a8de56900537c75443d3576f74d8bcb600a063ca088a",
}


def test_a_recorded_workdir_with_no_told_facts_changes_only_by_weighing_its_untold_said_gate(tmp_path):
    model = TestModel([add(ROW, dict(SAID_SLOT, fact_ids=["f1"])), DONE])
    written = W.write_spec("t1", inputs(), model)
    paths = W.write_verifier(written.spec, tmp_path, inputs())
    got = {path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    content = model.calls[0]["messages"][0]["content"]
    got["body"] = hashlib.sha256(content[content.index("\n{") + 1:].encode()).hexdigest()
    assert got == NO_KIND_HASHES and written.counts["tell_weighed"] == 1
    assert "side_facts" not in written.counts and "side_weighed" not in written.counts


# --- the clash check after the asked turn ----------------------------------------------------------

def test_an_item_the_asked_followup_adds_that_clashes_is_sent_back_once():
    clashing = dict(ROW, id="s2", expect={"slot": "four"})
    model = TestModel([add(dict(ROW, gate=False)), DONE, add(clashing), DONE, call("drop_items", ids=["s1"]), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["asked_sent_back"] == 1 and written.counts["asked_clash_sent_back"] == 1
    assert written.counts["unsatisfiable"] == [] and [c.id for c in written.spec.checks] == ["s2"]
    assert "no run can pass together" in model.calls[4]["messages"][-1]["content"]
    assert len(model.calls) == 6


def test_an_asked_followup_with_no_clash_runs_no_second_send_back():
    model = TestModel([add(dict(ROW, gate=False)), DONE, add(ROW), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["asked_sent_back"] == 1 and "asked_clash_sent_back" not in written.counts
    assert len(model.calls) == 4


def test_a_task_with_no_asked_followup_never_runs_the_second_clash_check():
    model = TestModel([add(ROW), DONE])
    written = W.write_spec("t1", inputs(), model)
    assert written.counts["asked_sent_back"] == 0 and "asked_clash_sent_back" not in written.counts
    assert len(model.calls) == 2


# --- reactions and conditional asks -----------------------------------------------------------------

FALLBACK = IntentFact(id="f2", text="Then put item A1 in slot seven.", stance="fallback", kind="do",
                      source=FactSource(recording="rec1", turn=4), condition="agent",
                      in_reply_to=ReplyQuote(turn=4, quote="I could not do that"))


def _fallback_facts(condition="agent"):
    """The ask, and a request the user made only after the agent's line (its own conduct, or a claim)."""
    return [FACTS[0].model_copy(update={"kind": "do"}), FALLBACK.model_copy(update={"condition": condition})]


@pytest.mark.parametrize("fact_ids", [["f2"], ["f1", "f2"]], ids=["reaction alone", "beside the ask"])
def test_a_gate_citing_a_reaction_is_weighed_with_the_reason_kept_on_the_item_whatever_the_policy_line(fact_ids):
    store = W.store_of(inputs(facts=_fallback_facts()))
    out = store.add([dict(ROW, fact_ids=fact_ids, policy_line="Move an item only to a free slot.")])
    assert "gate weighed" in out and "reaction" in out and store.refused == []
    assert store.items["s1"]["gate"] is False and store.items["s1"]["weighed"] == AGENT_WEIGHED


def test_a_fallback_fact_mined_before_conditions_counts_as_a_reaction():
    store = W.store_of(inputs(facts=_fallback_facts(condition="none")))
    store.add([dict(ROW, fact_ids=["f2"])])
    assert store.items["s1"]["weighed"] == AGENT_WEIGHED


def test_a_gate_on_a_conditional_ask_is_weighed_until_the_writer_confirms_the_claim():
    store = W.store_of(inputs(facts=_fallback_facts(condition="world_or_policy")))
    out = store.add([dict(ROW, fact_ids=["f2"])])
    assert "confirm it yourself" in out and store.items["s1"]["weighed"] == UNCONFIRMED_WEIGHED


@pytest.mark.parametrize("confirm", [{"policy_line": "Move an item only to a free slot."},
                                     {"confirms": {"table": "items", "find": {"item_id": "B2"}}}],
                         ids=["by the policy line", "by a row"])
def test_a_conditional_ask_confirmed_by_a_policy_line_or_one_row_keeps_its_gate(confirm):
    store = W.store_of(inputs(facts=_fallback_facts(condition="world_or_policy")))
    store.add([dict(ROW, fact_ids=["f2"], **confirm)])
    assert store.items["s1"]["gate"] is True and "weighed" not in store.items["s1"]
    if "confirms" in confirm:
        assert store.items["s1"]["confirms"] == {"table": "items", "row": "B2"}


@pytest.mark.parametrize("confirms", [{"table": "items", "find": {"slot": "nine"}}, {"table": "nope", "find": {}},
                                      {"find": {"item_id": "B2"}}])
def test_a_confirms_that_names_no_one_row_is_refused_with_the_reason(confirms):
    store = W.store_of(inputs(facts=_fallback_facts(condition="world_or_policy")))
    store.add([dict(ROW, fact_ids=["f2"], confirms=confirms)])
    assert "s1" not in store.items and "confirms" in store.refused[0]["reason"]


def test_a_fallback_fact_is_never_sent_back_by_the_asked_followup():
    intent = SpecIntent(task_id="t1", facts=_fallback_facts())
    assert W.asked_missing([ROW], intent) == []


def test_the_writer_shows_the_condition_and_quote_and_counts_reactions_and_weighed_gates():
    model = TestModel([add(ROW, dict(ROW, id="s2", fact_ids=["f2"])), DONE])
    written = W.write_spec("t1", inputs(facts=_fallback_facts()), model)
    content = model.calls[0]["messages"][0]["content"]
    shown = json.loads(content[content.index("\n{") + 1:])["intent"][1]
    assert shown["condition"] == "agent" and shown["answers"] == "I could not do that"
    assert written.counts["reaction_facts"] == 1 and written.counts["reaction_weighed"] == 1
    assert "condition agent is a reaction" in W.CHOICE_RULE


# Generated by running the 038b5c1b tree (before fallback facts) on the side-fact Task below.
KINDS_NO_FALLBACK_HASHES = {
    "spec/t1.json": "b8305c3453c6b82fb48ec0533cbf46bbe4202f3df267250ef8ab68c39e187dcf",
    "spec/verifiers/t1.json": "b2fe22a8718b16ed6b1ca5845bbee18e318e21cdeca7a8d4787581b7a5bc365a",
    "verifiers/t1.json": "b2fe22a8718b16ed6b1ca5845bbee18e318e21cdeca7a8d4787581b7a5bc365a",
    "body": "587681b299b463f0034db697438fd4765cf5ecb8b07325dd272c897d52328d55",
}


def test_a_task_with_kinds_and_no_fallback_facts_is_unchanged_byte_for_byte_through_the_writer(tmp_path):
    model = TestModel([add(ROW, dict(SAID_SLOT, fact_ids=["f3"])), DONE])
    written = W.write_spec("t1", inputs(facts=_side_facts()), model)
    paths = W.write_verifier(written.spec, tmp_path, inputs(facts=_side_facts()))
    got = {path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    content = model.calls[0]["messages"][0]["content"]
    got["body"] = hashlib.sha256(content[content.index("\n{") + 1:].encode()).hexdigest()
    assert got == KINDS_NO_FALLBACK_HASHES
    assert "reaction_facts" not in written.counts
