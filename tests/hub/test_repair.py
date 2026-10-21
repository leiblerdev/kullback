"""The Harbor export's Verifier repairs (D431, D432, D437), on an invented world of shelves and crates."""

from __future__ import annotations

from pathlib import Path

from kullback.hub import repair
from kullback.hub.harbor import unreadable_cells
from kullback.runner.expected import match, path_value
from kullback.runner.records import Event, Run, Verifier, write_json
from kullback.runner.verdict import verdict
from kullback.spec.atoms import called_set_atom
from kullback.spec.effects import EFFECTS_FILE

START = {"shelves": {"s1": {"slots": [{"tag": "red"}], "bins": [{"tag": "a"}, {"tag": "b"}], "free": 4}},
         "crates": {"c1": {"count": 9, "note": "x"}}}


def _verifier(cells: list[dict], atoms: list[dict] = (), allowed: list[dict] = ()) -> Verifier:
    return Verifier.model_validate({"task_id": "t1", "atoms": list(atoms),
                                    "expected": [{"cells": cells, "allowed": list(allowed)}]})


def _called(tool: str) -> dict:
    return {"id": "c1", "kind": "required", "target": {"kind": "called", "tool": tool}}


EFFECTS = {"stack": {"columns": [{"table": "shelves", "column": "free"}, {"table": "crates", "column": "count"}],
                     "makes": [], "writes": [{"table": "shelves", "field": "free", "values": [3]}]},
           "paint": {"columns": [{"table": "crates", "column": "note"}], "makes": [], "writes": []}}


def test_a_cell_on_a_one_member_start_list_is_named_by_index_and_its_value_kept():
    cells = [{"table": "shelves", "row_id": "s1", "field": "slots.tag", "value": "blue"}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    cell = fixed.expected[0].cells[0]
    assert (cell.field, cell.value) == ("slots.0.tag", "blue")
    assert resolved == ["shelves: slots.tag -> slots.0.tag"]
    assert path_value(START["shelves"]["s1"], cell.field) == "red"


def test_one_value_under_a_list_of_several_members_is_left_unreadable():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": "c"}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    assert fixed.expected[0].cells[0].field == "bins.tag" and resolved == []


def test_a_list_value_with_one_entry_per_start_member_becomes_one_indexed_cell_per_member_with_the_weight_split():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": ["c", "d"], "weight": 0.5}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    made = fixed.expected[0].cells
    assert [(cell.field, cell.value, cell.weight, cell.gate) for cell in made] == [
        ("bins.0.tag", "c", 0.25, True), ("bins.1.tag", "d", 0.25, True)]
    assert resolved == ["shelves: bins.tag -> bins.0.tag, bins.1.tag"]
    assert unreadable_cells(fixed, START) == []


def test_members_whose_leaves_are_unchanged_are_matched_in_the_start_rows_order():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": ["b", "a"]}]
    fixed, _ = repair.resolve_cells(_verifier(cells), START)
    end = {"shelves": {"s1": START["shelves"]["s1"]}, "crates": START["crates"]}
    assert [(cell.field, cell.value) for cell in fixed.expected[0].cells] == [("bins.0.tag", "a"), ("bins.1.tag", "b")]
    assert match(fixed.expected[0], START, end).ok is True


def test_a_member_a_required_tool_appends_counts_so_the_longer_list_projects():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": ["a", "b", "c"]}]
    effects = {"stack": {"columns": [{"table": "shelves", "column": "bins", "patterns": ["bins.N"]}], "makes": []}}
    fixed, _ = repair.resolve_cells(_verifier(cells, [_called("stack")]), START, effects)
    assert [cell.field for cell in fixed.expected[0].cells] == ["bins.0.tag", "bins.1.tag", "bins.2.tag"]
    either = {"stack": effects["stack"], "paint": effects["stack"]}
    unrequired, resolved = repair.resolve_cells(_verifier(cells), START, either)
    assert unrequired.expected[0].cells[0].field == "bins.tag" and resolved == []


def test_a_list_value_whose_length_is_not_the_member_count_is_left_unreadable():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": ["a", "b", "c"]},
             {"table": "shelves", "row_id": "s1", "field": "slots.tag", "value": ["x", "y"]}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    assert [cell.field for cell in fixed.expected[0].cells] == ["bins.tag", "slots.tag"] and resolved == []
    assert len(unreadable_cells(fixed, START)) == 2


def test_a_one_member_list_with_one_value_is_still_named_by_index_0_with_the_value_and_weight_kept():
    cells = [{"table": "shelves", "row_id": "s1", "field": "slots.tag", "value": {"one_of": ["blue", "green"]},
              "weight": 0.5}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    cell = fixed.expected[0].cells[0]
    assert (cell.field, cell.value, cell.weight) == ("slots.0.tag", {"one_of": ["blue", "green"]}, 0.5)
    assert resolved == ["shelves: slots.tag -> slots.0.tag"]


def test_a_new_row_where_path_through_a_list_without_an_index_is_named_unreadable():
    verifier = Verifier.model_validate({"task_id": "t1", "expected": [{"cells": [], "new_rows": [
        {"table": "shelves", "where": {"bins.tag": "a", "free": 4}}]}]})
    assert unreadable_cells(verifier, START) == [
        "shelves new row: " + repair.unreadable_path(START["shelves"]["s1"], "bins.tag")]
    readable = Verifier.model_validate({"task_id": "t1", "expected": [{"cells": [], "new_rows": [
        {"table": "shelves", "where": {"bins.0.tag": "a"}}]}]})
    assert unreadable_cells(readable, START) == []


def _run(args: dict) -> Run:
    return Run(run_id="r1", task_id="t1", termination_reason="success", events=[
        Event(idx=0, type="tool_call", payload={"id": "k1", "name": "stack", "args": args}),
        Event(idx=1, type="tool_result", payload={"id": "k1", "result": "ok"})])


def _set_holds(verifier: Verifier, args: dict) -> bool:
    result = verdict(_run(args), verifier, None)
    return next(item.holds for item in result.items if item.id == "c2")


def test_a_called_set_tests_every_member_of_a_list_argument_against_the_set():
    atom = called_set_atom("c2", "stack", {"tags": {"one_of": ["a", "b"]}}, False, ["stack"], "stack a or b")
    fixed, widened = repair.list_sets(Verifier(task_id="t1", atoms=[atom]))
    assert widened == ["c2"]
    assert _set_holds(Verifier(task_id="t1", atoms=[atom]), {"tags": ["a", "b"]}) is False
    assert _set_holds(fixed, {"tags": ["a", "b"]}) is True
    assert _set_holds(fixed, {"tags": ["a", "z"]}) is False
    assert _set_holds(fixed, {"tags": []}) is False
    assert _set_holds(fixed, {"tags": "b"}) is True


def test_a_prohibited_called_set_is_not_widened():
    atom = called_set_atom("c2", "stack", {"tags": {"one_of": ["a"]}}, True, ["stack"], "never stack a")
    fixed, widened = repair.list_sets(Verifier(task_id="t1", atoms=[atom]))
    assert widened == [] and fixed.atoms == [atom]


def test_a_readable_cell_is_never_touched():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    assert fixed == _verifier(cells) and resolved == []


def test_a_required_tool_frees_what_it_always_moves_so_a_right_run_passes_sanity():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
    fixed, added = repair.allow_effects(_verifier(cells, [_called("stack")]), EFFECTS)
    end = {"shelves": {"s1": dict(START["shelves"]["s1"], free=3)}, "crates": {"c1": {"count": 8, "note": "x"}}}
    assert match(_verifier(cells).expected[0], START, end).ok is False
    assert match(fixed.expected[0], START, end).ok is True
    assert added == 1
    assert fixed.expected[0].allowed == [{"table": "crates", "field": "count", "why": "free: learned effect: stack"}]


def test_a_tool_the_verifier_does_not_require_frees_nothing_so_its_write_still_fails_sanity():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
    fixed, added = repair.allow_effects(_verifier(cells, [_called("stack")]), EFFECTS)
    end = {"shelves": {"s1": dict(START["shelves"]["s1"], free=3)}, "crates": {"c1": {"count": 9, "note": "y"}}}
    assert match(fixed.expected[0], START, end).ok is False


def test_the_one_tool_whose_learned_effects_alone_answer_a_declared_row_is_required_without_a_called_item():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
    assert repair.required_tools(_verifier(cells), EFFECTS) == ["stack"]


def test_a_declared_row_two_tools_could_answer_requires_neither():
    cells = [{"table": "crates", "row_id": "c1", "field": "count", "value": 8}]
    both = dict(EFFECTS, paint={"columns": [{"table": "crates", "column": "count"}], "makes": [], "writes": []})
    assert repair.required_tools(_verifier(cells), both) == []


def test_a_column_on_a_declared_row_is_freed_on_that_row_only():
    cells = [{"table": "crates", "row_id": "c1", "field": "note", "value": "y"}]
    fixed, _ = repair.allow_effects(_verifier(cells, [_called("stack")]), EFFECTS)
    assert {"table": "crates", "row_id": "c1", "field": "count", "why": "free: learned effect: stack"} \
        in fixed.expected[0].allowed


def test_the_oracle_tool_that_moves_a_declared_table_is_required_and_one_on_another_table_is_not():
    cells = [{"table": "shelves", "row_id": "s1", "field": "slots", "value": []}]
    oracle = [{"name": "stack", "arguments": {}}, {"name": "paint", "arguments": {}}, {"name": "look", "arguments": {}}]
    assert repair.required_tools(_verifier(cells), EFFECTS, oracle) == ["stack"]


def test_a_column_a_cell_checks_or_a_rule_already_frees_gains_no_rule():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
    verifier = _verifier(cells, [_called("stack")], [{"table": "crates"}])
    fixed, added = repair.allow_effects(verifier, EFFECTS)
    assert added == 0 and fixed == verifier


def test_effects_are_read_from_the_file_when_it_holds_writes(tmp_path: Path):
    write_json(tmp_path / EFFECTS_FILE, {"effects": EFFECTS, "signatures": {}})
    assert repair.tool_effects(tmp_path) == (EFFECTS, repair.FROM_FILE)


def test_a_stale_effects_file_is_learned_again_and_says_so(tmp_path: Path):
    stale = {tool: dict(row, writes=[]) for tool, row in EFFECTS.items()}
    write_json(tmp_path / EFFECTS_FILE, {"effects": stale, "signatures": {}})
    effects, source = repair.tool_effects(tmp_path)
    assert source == repair.LEARNED_STALE and effects == {}
