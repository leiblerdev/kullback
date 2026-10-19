"""The Spec writer of a born Task gates what the scenario's end says (D414).

The end lies beside a born Task in born/<task>.json; the writer receives it as end facts, never a user turn.
The models are the harness TestModel with scripted replies, over the invented world of tests/spec/test_writer.py.
"""

from __future__ import annotations

import json

from kullback.agent.bus import Bus
from kullback.ai.provider import TestModel
from kullback.runner.records import write_json
from kullback.spec import events
from kullback.spec import items as I
from kullback.spec import writer as W
from kullback.spec.router import no_verifier_why
from kullback.spec.schema import load_spec, spec_verifier_path
from kullback.spec.stage import write_from_intent
from tests.spec.test_writer import DONE, ROW, STATE, TOOLS, _workdir, add, call, inputs

END = "Item A1 sits in slot seven; the user is told slot seven."
PATH_LINE = "Moves go through update_item and no other tool."


def _born(root, end=END):
    _workdir(root)
    write_json(W.born_path(root, "t1"), {"scenario": "sc_0", "end": end})
    return root


def _ended():
    base = inputs()
    return base._replace(intent=W.with_end(base.intent, END))


def test_a_born_tasks_end_reaches_the_writer_as_end_facts_and_never_as_a_user_turn(tmp_path):
    loaded = W.load_inputs(_born(tmp_path), "t1", inputs().intent)
    ends = [fact for fact in loaded.intent.facts if I.is_end_fact(fact)]
    assert [(fact.id, fact.text) for fact in ends] == [("end1", "Item A1 sits in slot seven;"),
                                                       ("end2", "the user is told slot seven.")]
    assert W.with_end(loaded.intent, END) == loaded.intent, "the end is added once"
    assert [row["id"] for row in W._body(loaded)["intent"] if row.get("end")] == ["end1", "end2"]
    sent = W.writer_messages(loaded)[0]
    assert len(sent) == 1 and json.dumps(END)[1:-1].split(";")[0] in sent[0]["content"]
    assert "Facts marked end are the truth this Task was written with" in W.SYSTEM_PROMPT


def test_a_recorded_task_has_no_end_and_its_intent_is_unchanged(tmp_path):
    _workdir(tmp_path)
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    assert W.scenario_end(tmp_path, "t1") == "" and loaded.intent == inputs().intent


def test_a_writer_leaving_an_end_change_ungated_is_refused_and_one_gating_it_passes():
    told = {"id": "j1", "kind": "said", "values": ["seven"], "gate": True, "fact_ids": ["end2"]}
    left = W.write_spec("t1", _ended(), TestModel([add(ROW, told), DONE, DONE]))
    assert left.counts["asked_sent_back"] == 2 and left.counts[W.UNCHECKED_END] == ["end1", "end2"]
    assert W.unchecked_end(left.spec) == ["end1", "end2"]
    rebut = call("rebut_fact", fact_id="end2", why='the end names it: "the user is told slot seven."')
    gated = W.write_spec("t1", _ended(), TestModel([add(dict(ROW, fact_ids=["f1", "end1"]), told), DONE, rebut, DONE]))
    assert gated.spec.checks[-1].gate is False
    assert W.UNCHECKED_END not in gated.counts and W.unchecked_end(gated.spec) == []


def test_the_spec_stage_sets_a_spec_leaving_an_end_fact_unchecked_aside_with_no_verifier(tmp_path):
    _born(tmp_path)
    bus = Bus(tmp_path / "bus.jsonl", agent="spec")
    write_from_intent(tmp_path, "t1", inputs().intent, TestModel([add(ROW), DONE, DONE]), 1.0, bus)
    spec = load_spec(tmp_path, "t1")
    assert spec.set_aside == W.UNCHECKED_END and not spec_verifier_path(tmp_path, "t1").exists()


def test_the_unchecked_end_reason_keeps_its_code_on_the_bus_and_is_the_tasks_no_verifier_why(tmp_path):
    assert events.code_of(W.UNCHECKED_END) == "unchecked_end" == events.UNCHECKED_END
    _born(tmp_path)
    write_from_intent(tmp_path, "t1", inputs().intent, TestModel([add(ROW), DONE, DONE]), 1.0,
                      Bus(tmp_path / "bus.jsonl", agent="spec"))
    assert no_verifier_why(load_spec(tmp_path, "t1")) == "unchecked_end"
    published = [json.loads(line) for line in (tmp_path / "bus.jsonl").read_text().splitlines()]
    assert any("unchecked_end" in json.dumps(row) for row in published)


def test_a_told_item_citing_only_end_facts_is_weighed_with_the_end_told_note():
    facts = _ended().intent.facts
    world = I.World(STATE, "", facts, TOOLS)
    told = {"id": "j1", "kind": "said", "values": ["seven"], "gate": True, "fact_ids": ["end2"]}
    kept, why = world.check(told, {fact.id: fact for fact in facts})
    assert kept["gate"] is False and kept["weighed"] == I.END_TOLD_WEIGHED
    assert why.startswith("told gate weighed: only the end names it")
    judge = {"id": "j2", "kind": "judge", "question": "Was the user told the item's slot?", "gate": True,
             "fact_ids": ["end2"], "anchor": {"answer": "A1 is in slot two",
                                              "row": {"table": "items", "find": {"item_id": "A1"}, "fields": ["slot"]}}}
    kept, why = world.check(judge, {fact.id: fact for fact in facts})
    assert kept["gate"] is False and kept["weighed"] == I.END_TOLD_WEIGHED, why
    rule = " ".join(W.END_RULE.split())
    assert "once the user confirms, the user will say yes: gate the change" in rule
    assert "never a not_called on it" in rule


def test_a_told_item_citing_an_end_fact_beside_a_user_turn_fact_stays_a_gate():
    facts = _ended().intent.facts
    world = I.World(STATE, "", facts, TOOLS)
    told = {"id": "j1", "kind": "said", "values": ["seven"], "gate": True, "fact_ids": ["f1", "end2"]}
    kept, why = world.check(told, {fact.id: fact for fact in facts})
    assert why == "" and kept["gate"] is True and "weighed" not in kept


def test_a_told_item_citing_only_end_facts_with_its_policy_line_stays_a_gate():
    facts = _ended().intent.facts
    line = "Tell the user the slot an item moves to."
    world = I.World(STATE, line + "\n", facts, TOOLS)
    told = {"id": "j1", "kind": "said", "values": ["seven"], "gate": True, "fact_ids": ["end2"], "policy_line": line}
    kept, why = world.check(told, {fact.id: fact for fact in facts})
    assert why == "" and kept["gate"] is True and "weighed" not in kept


def test_a_called_gate_stands_only_where_its_policy_line_names_the_tool():
    facts = inputs().intent.facts
    world = I.World(STATE, "Move an item only to a free slot.\n" + PATH_LINE + "\n", facts, TOOLS)
    move = {"id": "c1", "kind": "called", "tool": "update_item", "gate": True, "fact_ids": ["f1"]}
    kept, why = world.check(move, {fact.id: fact for fact in facts})
    assert kept["gate"] is False and kept["weighed"] == I.PATH_WEIGHED and "names that tool" in why
    kept, why = world.check(dict(move, policy_line=PATH_LINE), {fact.id: fact for fact in facts})
    assert why == "" and kept["gate"] is True
    assert "never by which lookup tool found them" in " ".join(W.CHOICE_RULE.split())


def test_exclusive_end_writes_are_alternative_end_states_that_never_contradict():
    facts = inputs().intent.facts
    world = I.World(STATE, "", facts, TOOLS)
    kept = [world.check(item, {fact.id: fact for fact in facts})[0]
            for item in (dict(ROW, id="a0", alt=0), dict(ROW, id="a1", alt=1, expect={"slot": "four"}))]
    assert all(kept) and I.contradictions(kept) == []
    assert I.contradictions([dict(kept[0], alt=None), dict(kept[1], alt=None)]) != [], "one end state clashes"
    assert "write each as its own end state (alt 0, 1, ...), never both gated in one" in " ".join(W.END_RULE.split())


def test_a_said_number_compiles_to_the_shape_every_writing_of_it_holds():
    item = {"id": "s1", "kind": "said", "values": ["$3,648.02", "128GB", "12.50", "A1"], "not_values": [],
            "gate": True, "weight": 1.0, "fact_ids": ["f1"]}
    values = I.atom_of(item).target["values"]
    assert values == ["648.02", "128", "12.5", "A1"]
    for text in ("your refund is $3,648.02 for 128 gb at 12.50 each on a1", "3648.02 back, 128gb, 12.5, a1"):
        assert all(value.lower() in text for value in values)


def test_a_born_tasks_intent_says_its_record_facts_the_end_stays_an_end_and_asked_facts_lays_records_never_the_end(
        tmp_path):
    from kullback.hub.package import asked_facts
    from kullback.spec.schema import Spec

    _born(tmp_path)
    write_json(tmp_path / "tasks" / "t1.json", {"id": "t1", "run_ids": ["sc-1"]})
    write_json(tmp_path / "user_rules" / "sc-1.json", {
        "facts": [{"field": "goal", "value": "Move A1.", "span": {"file_hash": "sc_0"}},
                  {"field": "holder_code", "value": "H-77", "context": END},
                  {"field": "end", "value": END, "context": END}],
        "disclosure": [{"field": "holder_code", "on_request": True}]})
    loaded = W.load_inputs(tmp_path, "t1", inputs().intent)
    records = [fact for fact in loaded.intent.facts if fact.source.recording == I.RECORDS_SOURCE]
    assert [(fact.id, fact.text, fact.stance, fact.condition) for fact in records] == [
        ("rec1", "holder_code: H-77", "volunteered", "asked")]
    assert not any(I.is_end_fact(fact) for fact in records)
    assert [fact.id for fact in loaded.intent.facts if I.is_end_fact(fact)] == ["end1", "end2"]
    laid = asked_facts(Spec(task_id="t1", intent=loaded.intent))
    ends = {fact.text for fact in loaded.intent.facts if I.is_end_fact(fact)}
    assert "holder_code: H-77" in laid and not ends & set(laid)


def test_a_recorded_task_gets_no_record_facts(tmp_path):
    _workdir(tmp_path)
    write_json(tmp_path / "user_rules" / "r1.json", {"facts": [{"field": "holder_code", "value": "H-77"}],
                                                     "disclosure": [{"field": "holder_code", "on_request": True}]})
    write_json(tmp_path / "tasks" / "t1.json", {"id": "t1", "run_ids": ["r1"]})
    assert W.load_inputs(tmp_path, "t1", inputs().intent).intent == inputs().intent
