from types import SimpleNamespace

import pytest

from kullback.runner.records import Conduct, EndState, Event, ExpectedCell, ItemResult, ValueSource, Verifier
from kullback.spec import mutants as M
from kullback.spec import trust as T
from scripts.mutants import line
from tests.spec.fixtures import WRITE_TOOLS, fn, reference_run

START = {"items": {"A1": {"slot": "two", "label": "red"}, "B2": {"slot": "four", "label": "blue"}},
         "notes": {"n1": {"text": "kept"}}}
END = {"items": {"A1": {"slot": "seven", "label": "red"}, "B2": {"slot": "four", "label": "blue"}},
       "notes": {"n1": {"text": "kept"}}}
SAID = ValueSource(kind="user_turn", ptr={"turn": 0})


def _base(start=START, end=END, model="some/model"):
    """A fresh Run (a live model, not the recorded one) that ends in `end`."""
    run = reference_run()
    stop = Event(idx=len(run.events), type="stop", payload={"start_state": start, "end_state": end})
    return run.model_copy(update={"events": run.events + [stop], "run_id": "fresh-1", "model": model})


def _cell(table="items", row_id="A1", field="slot", value="seven"):
    return ExpectedCell(table=table, row_id=row_id, field=field, value=value, row_source=SAID, source=SAID)


def _verifier(cells=None, allowed=()):
    return Verifier(task_id="t1", expected=[EndState(cells=[_cell()] if cells is None else cells,
                                                     allowed=list(allowed))])


def _report(verifier, base=None, **kw):
    return M.mutant_report(verifier, [base or _base()], canon=fn, tools=WRITE_TOOLS, **kw)


def _by(report, operator):
    return [m for m in report["mutants"] if m["operator"] == operator]


def test_the_value_mutant_takes_the_same_field_of_another_start_row_and_the_undone_one_its_start_value():
    made, _ = M.candidate_mutants(_verifier(), _base(), canon=fn)
    slot = {m.operator: m.run.events[-1].payload["end_state"]["items"]["A1"]["slot"]
            for m in made if (m.table, m.row_id, m.field) == ("items", "A1", "slot")}
    assert slot == {"value": "four", "undone": "two"}


def test_a_verifier_declaring_the_written_cell_kills_value_and_undone_at_that_cell_for_the_right_reason():
    report = _report(_verifier())
    assert report["status"] == "scored" and report["survived"] == 0
    for row in _by(report, "value") + _by(report, "undone"):
        assert row["outcome"] == "killed" and row["failing"] == ["state:items.A1.slot"] and row["right_reason"]


def test_untouched_writes_one_undeclared_row_per_table_and_each_field_left_alone_on_a_written_row():
    report = _report(_verifier())
    untouched = {(m["table"], m["row"], m["field"]) for m in _by(report, "untouched")}
    assert untouched == {("items", "B2", "label"), ("notes", "n1", "text"), ("items", "A1", "label")}
    assert all(m["outcome"] == "killed" and m["failing"] == ["sanity"] and m["right_reason"]
               for m in _by(report, "untouched"))


def test_a_verifier_that_allows_the_written_row_whole_lets_its_untouched_fields_survive():
    loose = _verifier(allowed=[{"table": "items", "row_id": "A1"}])
    report = _report(loose)
    left_alone = [m for m in _by(report, "untouched") if m["row"] == "A1"]
    assert left_alone and all(m["outcome"] == "survived" for m in left_alone)
    assert report["survived"] == len(left_alone)


def _two(members=("nine", "ten")):
    """A Verifier whose cell is a set of two values: no end state can be built from it (D375)."""
    return _verifier(cells=[_cell(value={"one_of": list(members)})])


def test_a_task_with_no_base_built_and_no_fresh_run_passing_is_not_mutated_and_says_no_base():
    report = _report(_two())
    assert report["status"] == M.NO_BASE and "mutants" not in report
    assert line("task_abcd1234", report) == "1234 no base built and no passing fresh run, not scored"


def test_a_passing_replay_of_the_recording_is_never_the_base():
    assert _report(_two(("seven", "eight")), _base(model="recorded"))["status"] == M.NO_BASE


def test_an_unbuildable_verifier_takes_the_newest_passing_fresh_run_as_its_base():
    older, newer = _base(), _base().model_copy(update={"run_id": "fresh-2"})
    failing = _base(end=START).model_copy(update={"run_id": "fresh-3"})
    two = _two(("seven", "eight"))
    assert M.passing_base(two, [older, newer, failing], canon=fn, tools=WRITE_TOOLS).run_id == "fresh-2"
    report = _report(two, None)
    assert report["base"] == "fresh-1" and report["base_kind"] == M.FRESH


def test_the_base_is_built_from_the_verifier_on_the_start_state_with_no_run_passing():
    report = _report(_verifier(cells=[_cell(value="nine")]), _base(model="recorded"))
    assert report["status"] == "scored" and report["base_kind"] == M.CONSTRUCTED and report["base"] == "t1.constructed"
    built = M.constructed_base(_verifier(cells=[_cell(value="nine")]), [_base()], canon=fn)
    assert built.events[-1].payload["end_state"] == {**START, "items": {**START["items"], "A1": {"slot": "nine",
                                                                                                 "label": "red"}}}
    assert built.events[-1].payload["start_state"] == START and not [e for e in built.events if e.type != "stop"]


@pytest.mark.parametrize("cell", [
    _cell(value={"one_of": ["seven", "eight"]}),
    _cell(value={"not": ["two"]}),
    _cell(row_id="Z9"),
], ids=["set of two", "not set", "row not there"])
def test_a_gated_cell_that_is_not_one_literal_value_builds_no_base(cell):
    assert M.constructed_base(_verifier(cells=[cell]), [_base()], canon=fn) is None


def test_a_set_of_one_a_dotted_path_and_a_new_row_are_built_and_a_weighted_cell_is_left_as_it_was():
    cells = [_cell(value={"one_of": ["seven"]}), _cell(row_id="B2", field="meta.size", value=3),
             _cell(row_id="B2", field="label", value="green").model_copy(update={"gate": False})]
    made = Verifier(task_id="t1", expected=[EndState(cells=cells, new_rows=[{"table": "notes",
                                                                             "where": {"text": "new"}, "count": 2}])])
    end = M.constructed_base(made, [_base()], canon=fn).events[-1].payload["end_state"]
    assert end["items"]["A1"]["slot"] == "seven" and end["items"]["B2"] == {"slot": "four", "label": "blue",
                                                                           "meta": {"size": 3}}
    assert sorted(row["text"] for row in end["notes"].values()) == ["kept", "new", "new"]


def test_a_called_gate_failing_on_the_built_base_kills_no_mutant_and_a_loose_cell_still_survives():
    called = Conduct(kind="called", tool="update_item", source=SAID)
    loose = Verifier(task_id="t1", conduct=[called], expected=[EndState(
        cells=[_cell()], allowed=[{"table": "items", "row_id": "A1", "field": "label"}])])
    report = _report(loose, _base(model="recorded"))
    assert report["base_kind"] == M.CONSTRUCTED
    assert all(row["outcome"] == "killed" for row in _by(report, "value") if row["field"] == "slot")
    label = [row for row in report["mutants"] if row["row"] == "A1" and row["field"] == "label"]
    assert label and all(row["outcome"] == "survived" and row["failing"] == [] for row in label)


def test_a_keyed_cell_the_base_kept_is_flipped_and_a_keep_that_holds_kills_it():
    kept = _cell(row_id="B2", value="four")
    keys = M.AskKeys(frozenset({("items", "A1", "slot"), ("items", "B2", "slot")}), frozenset(), frozenset())
    made, _ = M.candidate_mutants(_verifier(cells=[_cell(), kept]), _base(), keys, canon=fn)
    flipped = [m for m in made if (m.operator, m.row_id, m.field) == ("value", "B2", "slot")]
    assert len(flipped) == 1 and flipped[0].run.events[-1].payload["end_state"]["items"]["B2"]["slot"] == "two"
    report = _report(_verifier(cells=[_cell(), kept]), keys=keys)
    assert all(row["outcome"] == "killed" for row in report["mutants"] if row["row"] == "B2" and row["field"] == "slot")


def _ask_spec(demand, stance="volunteered", condition="none"):
    from kullback.spec.schema import FactSource, IntentFact
    from tests.spec.fixtures import check, spec

    fact = IntentFact(id="f1", text="move A1 to slot seven", stance=stance, condition=condition, kind="do",
                      source=FactSource(recording="rec", turn=0))
    the_spec = spec(checks=[check(demand=dict({"fact_ids": ["f1"], "gate": True}, **demand))])
    return the_spec.model_copy(update={"intent": the_spec.intent.model_copy(update={"facts": [fact]})})


def test_ask_keys_take_the_cells_a_gated_item_on_an_ask_holds_and_the_columns_a_called_tool_moves():
    keys = M.ask_keys(_ask_spec({"kind": "row_is", "table": "items", "row": "A1", "expect": {"slot": "seven"}}))
    assert keys.cells == {("items", "A1", "slot")} and not keys.columns and not keys.tables
    called = _ask_spec({"kind": "called", "tool": "update_item"})
    effects = {"update_item": {"columns": [{"table": "items", "column": "slot"}], "makes": ["notes"]}}
    keys = M.ask_keys(called, effects)
    assert keys.columns == {("items", "slot")} and keys.tables == {"notes"}


@pytest.mark.parametrize("demand, stance, condition", [
    ({"kind": "row_is", "table": "items", "row": "A1", "expect": {"slot": "seven"}, "gate": False}, "volunteered", "none"),
    ({"kind": "row_is", "table": "items", "row": "A1", "expect": {"slot": "seven"}}, "fallback", "agent"),
], ids=["weighted item", "reaction"])
def test_a_weighted_item_or_one_on_a_reaction_keys_nothing(demand, stance, condition):
    assert M.ask_keys(_ask_spec(demand, stance, condition)) == M.AskKeys(frozenset(), frozenset(), frozenset())


def test_only_keyed_changes_are_mutated_and_an_unkeyed_write_builds_no_value_or_undone():
    end = {**END, "notes": {"n1": {"text": "edited"}}}
    keys = M.AskKeys(frozenset({("items", "A1", "slot")}), frozenset(), frozenset())
    made, _ = M.candidate_mutants(_verifier(), _base(end=end), keys, canon=fn)
    changed = {(m.operator, m.table) for m in made if m.operator in ("value", "undone")}
    assert changed == {("value", "items"), ("undone", "items")}


def test_an_inserted_row_gets_an_extra_row_copy_and_an_undone_that_removes_it():
    end = {**END, "notes": {"n1": {"text": "kept"}, "n2": {"text": "new"}}}
    verifier = _verifier(cells=[_cell(), _cell("notes", "n2", "text", "new")])
    report = _report(verifier, _base(end=end))
    extra, = _by(report, "extra_row")
    assert extra["table"] == "notes" and extra["row"] == "n2.extra" and extra["outcome"] == "killed"
    assert extra["right_reason"]
    removed = [m for m in _by(report, "undone") if m["row"] == "n2"]
    assert removed and removed[0]["field"] is None and removed[0]["outcome"] == "killed" and removed[0]["right_reason"]


def test_a_change_on_an_exempt_column_is_not_built_and_counted_as_no_effect():
    schema = SimpleNamespace(columns=[SimpleNamespace(table="items", name="label", class_="exempt")])
    made, no_effect = M.candidate_mutants(_verifier(), _base(), canon=fn, schema=schema)
    assert no_effect >= 1 and not any(m.field == "label" for m in made)


def test_the_cap_keeps_every_operator_round_robin_and_the_same_mutants_on_every_call():
    made, _ = M.candidate_mutants(_verifier(), _base(), canon=fn)
    first, again = M.capped(made, "t1", cap=3), M.capped(made, "t1", cap=3)
    assert [m.id for m in first] == [m.id for m in again]
    assert [m.operator for m in first] == ["value", "undone", "untouched"]
    report = _report(_verifier(), cap=3)
    assert report["built"] == 3 and report["dropped"] == report["candidates"] - 3


def test_a_failing_item_on_another_cell_is_a_kill_for_the_wrong_reason():
    made, _ = M.candidate_mutants(_verifier(), _base(), canon=fn)
    value = next(m for m in made if m.operator == "value")
    other = ItemResult(id="state:items.B2.slot", kind="state", gate=True, holds=False)
    elsewhere = ItemResult(id="sanity", kind="sanity", gate=True, holds=False, why="collateral:notes.n1.text")
    assert M.reads_cell(other, value, _verifier()) is False
    assert M.reads_cell(elsewhere, value, _verifier()) is False
    assert M.reads_cell(ItemResult(id="state:items.A1.slot", kind="state", gate=True), value, _verifier())


def test_the_stray_write_stays_the_first_of_the_per_table_stray_writes():
    verifier, reference = _verifier(), _base()
    tables = [table for table, *_ in T.stray_writes(verifier, reference, canon=fn)]
    assert tables == ["items", "notes"]
    first = T.stray_write_run(verifier, reference, canon=fn)
    assert first.events[-1].payload["end_state"]["items"]["B2"]["label"] == "blue (stray)"


def test_the_report_sits_under_exam_mutants_and_its_printed_line_masks_the_task_and_names_no_row(tmp_path):
    report = _report(_verifier())
    assert M.mutants_path(tmp_path, "t1") == tmp_path / "exam" / "mutants" / "t1.json"
    printed = line("task_abcd1234", report)
    assert printed.startswith("1234 built ") and "A1" not in printed


def test_a_new_row_count_whose_where_names_the_mutated_field_kills_for_the_right_reason():
    end = {**END, "notes": {"n1": {"text": "kept"}, "n2": {"text": "new"}}}
    counted = Verifier(task_id="t1", expected=[EndState(cells=[_cell()],
                                                        new_rows=[{"table": "notes", "where": {"text": "new"}}])])
    made, _ = M.candidate_mutants(counted, _base(end=end), canon=fn)
    inserted = next(m for m in made if m.operator == "value" and m.row_id == "n2")
    count = ItemResult(id="new_rows:notes:1", kind="state", gate=True, holds=False)
    assert M.reads_cell(count, inserted, counted)
    assert not M.reads_cell(count, inserted._replace(field="label"), counted)
