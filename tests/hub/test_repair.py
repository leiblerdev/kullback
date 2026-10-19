"""The Harbor export's two Verifier repairs (D431, D432), on an invented world of shelves and crates."""

from __future__ import annotations

from pathlib import Path

from kullback.hub import repair
from kullback.runner.expected import match, path_value
from kullback.runner.records import Verifier, write_json
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


def test_a_cell_on_a_list_of_several_members_is_left_unreadable():
    cells = [{"table": "shelves", "row_id": "s1", "field": "bins.tag", "value": "c"}]
    fixed, resolved = repair.resolve_cells(_verifier(cells), START)
    assert fixed.expected[0].cells[0].field == "bins.tag" and resolved == []


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


def test_a_column_on_a_declared_row_is_freed_on_that_row_only():
    cells = [{"table": "crates", "row_id": "c1", "field": "note", "value": "y"}]
    fixed, _ = repair.allow_effects(_verifier(cells, [_called("stack")]), EFFECTS)
    assert {"table": "crates", "row_id": "c1", "field": "count", "why": "free: learned effect: stack"} \
        in fixed.expected[0].allowed


def test_the_oracle_tool_that_moves_a_declared_table_is_required_and_one_on_another_table_is_not():
    cells = [{"table": "shelves", "row_id": "s1", "field": "free", "value": 3}]
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
