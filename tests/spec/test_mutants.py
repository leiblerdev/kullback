from types import SimpleNamespace

from kullback.runner.records import EndState, Event, ExpectedCell, ItemResult, ValueSource, Verifier
from kullback.spec import mutants as M
from kullback.spec import trust as T
from scripts.mutants import line
from tests.spec.fixtures import WRITE_TOOLS, fn, reference_run

START = {"items": {"A1": {"slot": "two", "label": "red"}, "B2": {"slot": "four", "label": "blue"}},
         "notes": {"n1": {"text": "kept"}}}
END = {"items": {"A1": {"slot": "seven", "label": "red"}, "B2": {"slot": "four", "label": "blue"}},
       "notes": {"n1": {"text": "kept"}}}
SAID = ValueSource(kind="user_turn", ptr={"turn": 0})


def _reference(start=START, end=END):
    run = reference_run()
    stop = Event(idx=len(run.events), type="stop", payload={"start_state": start, "end_state": end})
    return run.model_copy(update={"events": run.events + [stop], "run_id": "ref", "model": "recorded"})


def _cell(table="items", row_id="A1", field="slot", value="seven"):
    return ExpectedCell(table=table, row_id=row_id, field=field, value=value, row_source=SAID, source=SAID)


def _verifier(cells=None, allowed=()):
    return Verifier(task_id="t1", expected=[EndState(cells=[_cell()] if cells is None else cells,
                                                     allowed=list(allowed))])


def _report(verifier, reference=None, **kw):
    return M.mutant_report(verifier, reference or _reference(), canon=fn, tools=WRITE_TOOLS, **kw)


def _by(report, operator):
    return [m for m in report["mutants"] if m["operator"] == operator]


def test_the_value_mutant_takes_the_same_field_of_another_start_row_and_the_undone_one_its_start_value():
    made, _ = M.candidate_mutants(_verifier(), _reference(), canon=fn)
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


def test_a_reference_the_verifier_fails_is_not_mutated_and_says_so():
    wrong = _verifier(cells=[_cell(value="nine")])
    report = _report(wrong)
    assert report["status"] == "reference_fails" and "mutants" not in report
    assert line("task_abcd1234", report) == "1234 reference fails, not scored"


def test_an_inserted_row_gets_an_extra_row_copy_and_an_undone_that_removes_it():
    end = {**END, "notes": {"n1": {"text": "kept"}, "n2": {"text": "new"}}}
    verifier = _verifier(cells=[_cell(), _cell("notes", "n2", "text", "new")])
    report = _report(verifier, _reference(end=end))
    extra, = _by(report, "extra_row")
    assert extra["table"] == "notes" and extra["row"] == "n2.extra" and extra["outcome"] == "killed"
    assert extra["right_reason"]
    removed = [m for m in _by(report, "undone") if m["row"] == "n2"]
    assert removed and removed[0]["field"] is None and removed[0]["outcome"] == "killed" and removed[0]["right_reason"]


def test_a_change_on_an_exempt_column_is_not_built_and_counted_as_no_effect():
    schema = SimpleNamespace(columns=[SimpleNamespace(table="items", name="label", class_="exempt")])
    made, no_effect = M.candidate_mutants(_verifier(), _reference(), canon=fn, schema=schema)
    assert no_effect >= 1 and not any(m.field == "label" for m in made)


def test_the_cap_keeps_every_operator_round_robin_and_the_same_mutants_on_every_call():
    made, _ = M.candidate_mutants(_verifier(), _reference(), canon=fn)
    first, again = M.capped(made, "t1", cap=3), M.capped(made, "t1", cap=3)
    assert [m.id for m in first] == [m.id for m in again]
    assert [m.operator for m in first] == ["value", "undone", "untouched"]
    report = _report(_verifier(), cap=3)
    assert report["built"] == 3 and report["dropped"] == report["candidates"] - 3


def test_a_failing_item_on_another_cell_is_a_kill_for_the_wrong_reason():
    made, _ = M.candidate_mutants(_verifier(), _reference(), canon=fn)
    value = next(m for m in made if m.operator == "value")
    other = ItemResult(id="state:items.B2.slot", kind="state", gate=True, holds=False)
    elsewhere = ItemResult(id="sanity", kind="sanity", gate=True, holds=False, why="collateral:notes.n1.text")
    assert M.reads_cell(other, value, _verifier()) is False
    assert M.reads_cell(elsewhere, value, _verifier()) is False
    assert M.reads_cell(ItemResult(id="state:items.A1.slot", kind="state", gate=True), value, _verifier())


def test_the_stray_write_stays_the_first_of_the_per_table_stray_writes():
    verifier, reference = _verifier(), _reference()
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
    made, _ = M.candidate_mutants(counted, _reference(end=end), canon=fn)
    inserted = next(m for m in made if m.operator == "value" and m.row_id == "n2")
    count = ItemResult(id="new_rows:notes:1", kind="state", gate=True, holds=False)
    assert M.reads_cell(count, inserted, counted)
    assert not M.reads_cell(count, inserted._replace(field="label"), counted)
