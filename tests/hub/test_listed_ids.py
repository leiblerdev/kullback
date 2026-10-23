"""The listed ids a shipped world drops (D429), on invented tables: only where the keyed table is read without doubt."""

from __future__ import annotations

from kullback.hub import listed_ids


def _state(listed: list[str], **extra: dict) -> dict:
    return {"widgets": {"w1": {"label": "plain"}, "w2": {"label": "plain"}},
            "shelves": {"s1": {"holds": listed}}, **extra}


def test_a_column_whose_named_members_all_name_one_table_drops_the_members_naming_no_row():
    found = listed_ids.columns(_state(["w1", "w2", "w9"]))
    assert [(column.name, column.keys, column.missing) for column in found] == [("shelves.holds", "widgets", ["w9"])]
    assert listed_ids.drops(found) == {"shelves": {"holds": ["w9"]}}


def test_a_column_naming_rows_of_two_tables_is_ambiguous_and_drops_nothing():
    found = listed_ids.columns(_state(["w1", "b1", "w9"], bins={"b1": {}}))
    assert [(column.name, column.keys) for column in found] == [("shelves.holds", None)]
    assert listed_ids.drops(found) == {}


def test_a_column_where_fewer_than_half_the_members_name_a_row_is_ambiguous():
    found = listed_ids.columns(_state(["w1", "x7", "x8"]))
    assert [(column.name, column.keys) for column in found] == [("shelves.holds", None)]


def test_a_column_naming_no_row_at_all_is_no_list_of_ids_and_is_not_read():
    assert listed_ids.columns(_state(["red", "blue"])) == []


def test_a_column_whose_every_member_names_a_row_is_left_alone():
    assert listed_ids.columns(_state(["w1", "w2"])) == []


def test_pruning_drops_the_members_from_the_shared_rows_and_from_the_rows_an_overlay_pins():
    rows = {"s1": {"holds": ["w1", "w9"]}, "s2": {"holds": ["w9"]}}
    assert listed_ids.prune_rows(rows, {"holds": ["w9"]}) == 2
    assert rows == {"s1": {"holds": ["w1"]}, "s2": {"holds": []}}
    payload = {"overlay": {"rows": [{"table": "shelves", "id": "s3", "version_hash": "h1"},
                                    {"table": "widgets", "id": "w1", "version_hash": "h2"}]},
               "values": {"h1": {"holds": ["w9", "w2"]}, "h2": {"holds": ["w9"]}}}
    listed_ids.prune_overlay(payload, {"shelves": {"holds": ["w9"]}})
    assert payload["values"] == {"h1": {"holds": ["w2"]}, "h2": {"holds": ["w9"]}}


def test_a_call_reads_an_id_when_any_argument_at_any_depth_equals_it():
    calls = [{"name": "a", "arguments": {"one": "w1"}}, {"name": "b", "arguments": {"many": [{"id": "w9"}]}}]
    assert listed_ids.reads(calls, {"w9"}) and not listed_ids.reads(calls, {"w8"})
    assert not listed_ids.reads([{"name": "a", "arguments": {"text": "about w9"}}], {"w9"})
