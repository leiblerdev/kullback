"""Every write a born Task's end names is gated by a differing cell, on every named column, and held by no idle Run (D419).

The models are the harness TestModel with scripted replies, over the invented worlds of tests/spec/test_writer.py
and tests/spec/test_stage.py.
"""

from __future__ import annotations

from kullback.agent.bus import Bus
from kullback.ai.provider import ModelReply, TestModel
from kullback.runner.records import write_json
from kullback.spec import stage
from kullback.spec import trust as TR
from kullback.spec import ungated as U
from kullback.spec import writer as W
from kullback.spec.schema import load_spec
from kullback.spec.stage import write_from_intent
from tests.spec import test_stage as S
from tests.spec.fixtures import fn
from tests.spec.test_writer import DONE, ROW, STATE, _workdir, add, inputs

MOVE = "Item A1 moves to slot seven."
AT_START = dict(ROW, expect={"slot": "two"}, fact_ids=["f1", "end1"])
MOVED = dict(ROW, fact_ids=["f1", "end1"])
CANNOT = ModelReply(content="rebut: cannot verify: no other value is named in the world")


def _ended(end: str = MOVE, state: dict = STATE):
    base = inputs(state=state)
    return base._replace(intent=W.with_end(base.intent, end))


def _born(root, end: str = MOVE):
    _workdir(root)
    write_json(W.born_path(root, "t1"), {"scenario": "sc_0", "end": end})
    return root


def test_a_write_gated_to_its_start_value_goes_back_once_and_unrepaired_is_counted_and_marked(tmp_path):
    model = TestModel([add(AT_START), DONE, CANNOT, CANNOT])
    write_from_intent(_born(tmp_path), "t1", inputs().intent, model, 1.0, Bus(tmp_path / "bus.jsonl", agent="spec"))
    spec = load_spec(tmp_path, "t1")
    assert len(model.calls) == 4, "one send back for the ungated write, one for the idle Run holding every gate (D441)"
    assert spec.set_aside is None and spec.writer[U.UNGATED_SENT_BACK] == 1
    assert spec.writer[U.WRITES_UNGATED] == 1 and spec.writer[U.UNGATED_FINDINGS][0]["fact_id"] == "end1"
    assert [check.demand["expect"] for check in spec.checks] == [{"slot": "two"}], "never gated to pass"


def test_a_one_of_holding_the_start_value_or_a_before_item_is_no_gate_of_a_write():
    softened = dict(AT_START, expect={"slot": {"one_of": ["two", "seven"]}})
    before = {"id": "b1", "kind": "before", "first": "confirm_turn", "then": "update_item", "gate": True,
              "fact_ids": ["end1"]}
    for items in ((softened,), (before,)):
        written = W.write_spec("t1", _ended(), TestModel([add(*items), DONE, DONE]))
        assert [row["fact_id"] for row in U.ungated_writes(written.spec, W.world_of(_ended()))] == ["end1"]


def test_a_write_gated_by_a_differing_row_is_is_not_sent_back(tmp_path):
    model = TestModel([add(MOVED), DONE])
    write_from_intent(_born(tmp_path), "t1", inputs().intent, model, 1.0, Bus(tmp_path / "bus.jsonl", agent="spec"))
    spec = load_spec(tmp_path, "t1")
    assert len(model.calls) == 2
    assert not {U.UNGATED_SENT_BACK, U.WRITES_UNGATED, U.END_VALUES_UNGATED, TR.HELD_BY_IDLE} & set(spec.writer)


LOT_STATE = {"items": {"A1": {"item_id": "A1", "slot": "two", "lot": "L-3"},
                       "B2": {"item_id": "B2", "slot": "four", "lot": "L-9"}}}
LOT_END = "Item A1 of lot L-3 moves to slot seven."


def test_an_end_value_of_the_changed_row_no_gate_carries_is_sent_back_then_counted():
    ended = _ended(LOT_END, LOT_STATE)
    written = W.write_spec("t1", ended, TestModel([add(MOVED), DONE]))
    assert U.ungated_values(written.spec, W.world_of(ended)) == [{"fact_id": "end1", "value": "L-3"}]
    left, _ = U.send_back(written, ended, TestModel([CANNOT]))
    assert left.counts[U.END_VALUES_UNGATED] == 1 and U.WRITES_UNGATED not in left.counts
    carried = add(dict(MOVED, expect={"slot": "seven", "lot": "L-3"}))
    fixed, _ = U.send_back(written, ended, TestModel([carried, DONE]))
    assert fixed.counts[U.UNGATED_SENT_BACK] == 1 and U.END_VALUES_UNGATED not in fixed.counts
    assert fixed.spec.checks[0].demand["expect"] == {"slot": "seven", "lot": "L-3"}


QTY_STATE = {"items": {"A1": {"item_id": "A1", "slot": "two", "qty": 2},
                       "B2": {"item_id": "B2", "slot": "four", "qty": 3}}}


def test_a_number_only_another_row_holds_is_no_finding_for_a_row_is_but_is_one_under_a_new_rows_where_key():
    moved = W.write_spec("t1", _ended("Item A1 moves to slot seven with 3 units.", QTY_STATE),
                         TestModel([add(MOVED), DONE]))
    assert U.ungated_values(moved.spec, W.world_of(_ended(state=QTY_STATE))) == []
    new = {"id": "n1", "kind": "row_new", "table": "items", "where": {"slot": "seven", "qty": 2}, "count": 1,
           "gate": True, "fact_ids": ["f1", "end1"]}
    ended = _ended("A new item in slot seven holds 3 units.", QTY_STATE)
    made = W.write_spec("t1", ended, TestModel([add(new), DONE]))
    assert [check.id for check in made.spec.checks] == ["n1"]
    assert U.ungated_values(made.spec, W.world_of(ended)) == [{"fact_id": "end1", "value": "3"}]


def test_the_values_an_end_clause_names_are_its_quoted_values_and_its_tokens_with_a_digit():
    assert U.end_values('Leg 2 of R-17 is "window seat"; refund $120.50.') == ["window seat", "2", "R-17", "120.50"]


def test_a_verifier_held_by_the_idle_run_is_refused_by_the_trust_gate_with_its_reason():
    ended = _ended()
    for item, held in ((AT_START, True), (MOVED, False)):
        spec = W.write_spec("t1", ended, TestModel([add(item), DONE])).spec.model_copy(update={"examined_rounds": [1]})
        verifier = W.verifier_of(spec, ended).verifier
        tier = TR.tier_of_task(spec, verifier, [TR.idle_run("t1", STATE)], canon=fn, write_tools=ended.write_tools)
        assert tier.row[TR.HELD_BY_IDLE] is held
        assert (tier.row["reason"] == TR.HELD_BY_IDLE) is held


def test_the_stage_counts_a_spec_marked_ungated_and_held_by_the_idle_run(tmp_path):
    root = S._workdir(tmp_path)
    write_json(W.born_path(root, "t1"), {"scenario": "sc_0", "end": MOVE})
    at_start = dict(S.ITEM, expect={"slot": "two"}, fact_ids=["f1", "end1"])
    counts = stage.write_specs(root, ["t1"], TestModel(S._mined() + S._items(at_start) + [CANNOT, CANNOT]))
    assert (counts["written"], counts[U.WRITES_UNGATED], counts[TR.HELD_BY_IDLE]) == (1, 1, 1)
    assert load_spec(root, "t1").writer[TR.HELD_BY_IDLE] is True


def test_a_spec_with_no_write_end_is_untouched_by_all_three(tmp_path):
    told = {"id": "j1", "kind": "said", "values": ["seven"], "gate": True, "fact_ids": ["end1"]}
    ended = _ended("The user is told slot seven.")
    written = W.write_spec("t1", ended, TestModel([add(dict(ROW, expect={"slot": "two"}), told), DONE]))
    assert U.write_facts(written.spec, ended.write_tools) == []
    assert U.send_back(written, ended, TestModel([])) == (written, 0.0)
    verifier = W.verifier_of(written.spec, ended).verifier
    assert not TR.held_by_idle(written.spec, verifier, TR.idle_run("t1", STATE), fn, ended.write_tools)
    model = TestModel([add(dict(ROW, expect={"slot": "two"})), DONE, CANNOT])
    _workdir(tmp_path)
    write_from_intent(tmp_path, "t1", inputs().intent, model, 1.0, Bus(tmp_path / "bus.jsonl", agent="spec"))
    assert not {U.UNGATED_SENT_BACK, TR.HELD_BY_IDLE} & set(load_spec(tmp_path, "t1").writer)
    assert len(model.calls) == 3, "only the idle Run's send back (D441): its one gate holds at the start"


def test_the_writer_is_told_each_rule_in_words_once():
    prompt = " ".join(W.SYSTEM_PROMPT.split())
    for line in ("a before or called item never gates a write", "One judge question asks one thing",
                 "Gate only a value the user asked for or the policy demands"):
        assert prompt.count(line) == 1
