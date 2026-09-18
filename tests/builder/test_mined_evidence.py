"""Tests for G32: every mined fact carries the calls that support it (builder/mine.py).

All names are invented (a harbour chandler). Each test gives the miner a name that lies and
calls that tell the truth, then checks the fact's basis, its supporting calls and, where the
calls contradict the name, the recorded contradiction. The table a column names has no
observation test a call could run, so its tests lock the punt: the basis stays the name.

The module skips without the frozen patch that adds "declared" to ClassifiedBy, because the
declared-kind facts only validate with it (docs/frozen-patches/mined-evidence.patch).
"""

from __future__ import annotations

import json
from typing import get_args

import pytest

from conftest import PTR
from kullback.builder.mine import (
    id_supporting_calls,
    mine_schema,
    mine_tools,
    mined_name_counts,
    propose_column_class,
    refuted_ids,
    row_homes,
)
from kullback.report.data import ReportData
from kullback.report.numbers import mined_evidence_counts
from kullback.report.render import _mined_evidence_lines
from kullback.runner.records import ClassifiedBy, ToolCall, Trace

pytestmark = pytest.mark.skipif(
    "declared" not in get_args(ClassifiedBy),
    reason="needs the mined-evidence frozen patch (ClassifiedBy gains declared)",
)


def one_trace(trace_id: str, calls: list[dict], tools_declared=None) -> Trace:
    return Trace(
        trace_id=trace_id,
        raw_hash="h",
        ingest_version="test",
        source="synthetic",
        tools_declared=tools_declared,
        tool_calls=[ToolCall(**{"raw_ptr": PTR, **c}) for c in calls],
        raw_ptr=PTR,
    )


def sig_by_name(sigs, name):
    return next(s for s in sigs if s.name == name)


def col(schema, table, name):
    return next(c for c in schema.columns if c.table == table and c.name == name)


def harbour_corpus() -> list[Trace]:
    """One trace where a read-prefixed tool echoes its arguments back like a create."""
    return [one_trace("t1", [
        {"name": "get_berth", "args": {"berth_tag": "T1"},
         "result": '{"berth_tag": "T1", "colour": "red"}'},
        {"name": "get_berth", "args": {"berth_tag": "T1"},
         "result": '{"berth_tag": "T1", "colour": "red"}'},
        {"name": "list_berths", "args": {},
         "result": '[{"berth_tag": "T1", "colour": "red"}, {"berth_tag": "T2", "colour": "blue"}]'},
        {"name": "get_crate", "args": {"crate": "C1"}, "result": '{"crate": "C1"}'},
    ])]


def test_a_read_prefix_whose_calls_echo_the_arguments_is_an_observed_write():
    sig = sig_by_name(mine_tools(harbour_corpus()), "get_crate")
    assert sig.kind == "write"
    assert sig.classified_by == "observed"
    assert "contradicts the name" in (sig.kind_reason or "")
    assert "t1:3" in (sig.kind_reason or "")
    assert "; basis: observed" in (sig.kind_reason or "")


def test_a_read_the_calls_confirm_stays_on_the_name_and_says_so():
    sig = sig_by_name(mine_tools(harbour_corpus()), "get_berth")
    assert sig.kind == "read"
    assert sig.classified_by == "rule"
    assert "; basis: name, no supporting calls" in (sig.kind_reason or "")


def test_a_declared_kind_beats_contradicting_calls_and_records_it():
    declared = [{"name": "harbormaster", "parameters": {"properties": {}},
                 "annotations": {"readOnlyHint": True}}]
    traces = [one_trace("t1", [
        {"name": "get_berth", "args": {"berth_tag": "T1"},
         "result": '{"berth_tag": "T1", "colour": "red"}'},
        {"name": "harbormaster", "args": {"berth_tag": "T1", "colour": "blue"},
         "result": "done"},
        {"name": "get_berth", "args": {"berth_tag": "T1"},
         "result": '{"berth_tag": "T1", "colour": "blue"}'},
    ], tools_declared=declared)]
    sig = sig_by_name(mine_tools(traces), "harbormaster")
    assert sig.kind == "read"
    assert sig.classified_by == "declared"
    assert "the declared kind stands" in (sig.kind_reason or "")
    assert "; basis: declared by the source, no supporting calls" in (sig.kind_reason or "")


def test_an_id_no_suffix_names_is_observed_with_support():
    schema = mine_schema(harbour_corpus())
    column = col(schema, "berth_tags", "berth_tag")
    assert column.class_ == "hard"
    assert column.evidence["class_basis"] == "observed"
    assert column.evidence["id_fact"] is True
    assert column.evidence["id_basis"] == "observed"
    assert ["t1", 0] in column.evidence["id_support"]
    assert column.evidence["id_support_count"] == 2
    assert column.evidence["id_contradicts_name"] is False
    assert column.evidence["table_basis"] == "name"


def test_id_addressing_calls_are_the_support():
    refs = id_supporting_calls(harbour_corpus())
    assert refs["berth_tag"]["support_count"] == 2
    assert ["t1", 0] in refs["berth_tag"]["supporting_calls"]


def test_a_name_id_that_repeats_is_refuted_and_the_values_decide_its_class():
    traces = [one_trace("t1", [
        {"name": "list_docks", "args": {},
         "result": '[{"dock_id": "D1", "shift": "early"}, {"dock_id": "D1", "shift": "late"}]'},
    ])]
    assert refuted_ids(traces) == {"dock_id"}
    schema = mine_schema(traces)
    column = col(schema, "docks", "dock_id")
    assert column.class_ == "hard"
    assert column.evidence["class_basis"] == "observed"
    assert column.evidence["class_contradicts_name"] is True
    assert column.evidence["id_fact"] is True
    assert column.evidence["id_basis"] == "observed"
    assert column.evidence["id_support"] == [["t1", 0]]
    assert column.evidence["id_contradicts_name"] is True
    assert column.evidence["table_basis"] == "name"


def test_a_timestamp_name_with_plain_values_falls_through_with_the_contradiction_recorded():
    traces = [one_trace("t1", [
        {"name": "list_badges", "args": {},
         "result": '[{"badge_id": "B1", "created_badge": "red"},'
                   ' {"badge_id": "B2", "created_badge": "blue"}]'},
    ])]
    schema = mine_schema(traces)
    column = col(schema, "badges", "created_badge")
    assert column.class_ == "hard"
    assert column.evidence["class_basis"] == "observed"
    assert column.evidence["class_contradicts_name"] is True
    # The direct call has no calls to offer, so the legacy name rule stands there unchanged.
    assert propose_column_class("badges", "created_badge", ["red", "blue"]).column_class == "exempt"


def test_a_timestamp_name_with_timestamps_is_observed_exempt():
    traces = [one_trace("t1", [
        {"name": "list_stamps", "args": {},
         "result": '[{"stamp_id": "S1", "created_stamp": "2025-01-01T10:00"},'
                   ' {"stamp_id": "S2", "created_stamp": "2025-01-02T11:30"}]'},
    ])]
    schema = mine_schema(traces)
    column = col(schema, "stamps", "created_stamp")
    assert column.class_ == "exempt"
    assert column.evidence["class_basis"] == "observed"
    assert column.evidence["class_contradicts_name"] is False


def test_homing_by_the_asked_for_id_is_observed_and_names_the_contradiction():
    traces = [one_trace("t1", [
        {"name": "lookup_customer_parcel", "args": {"customer_id": "C1"},
         "result": '{"customer_id": "C1", "parcel_id": "P1"}'},
    ])]
    homes = row_homes(traces)
    place = homes["lookup_customer_parcel"]["homed"]["customers"]
    assert place["rows"] == 1
    assert place["basis"] == "observed"
    assert place["supporting_calls"] == [["t1", 0]]
    assert place["support_count"] == 1
    assert place["contradicts_name"] is True


def test_homing_by_the_distinct_id_is_observed():
    traces = [one_trace("t1", [
        {"name": "list_moorings", "args": {},
         "result": '[{"dock_id": "D1", "berth_id": "B1"},'
                   ' {"dock_id": "D1", "berth_id": "B2"}]'},
    ])]
    homes = row_homes(traces)
    place = homes["list_moorings"]["homed"]["berths"]
    assert place["rows"] == 2
    assert place["basis"] == "observed"
    assert place["supporting_calls"] == [["t1", 0]]
    assert place["support_count"] == 2
    assert place["contradicts_name"] is True


def test_homing_by_the_only_id_rests_on_the_name():
    traces = [one_trace("t1", [
        {"name": "get_berth", "args": {},
         "result": '{"berth_id": "B7", "colour": "red"}'},
    ])]
    homes = row_homes(traces)
    place = homes["get_berth"]["homed"]["berths"]
    assert place["basis"] == "name"
    assert place["supporting_calls"] == []
    assert place["support_count"] == 0
    assert place["contradicts_name"] is False


def test_the_counts_measure_how_many_facts_rest_on_a_name_alone():
    traces = harbour_corpus()
    sigs = mine_tools(traces)
    schema = mine_schema(traces)
    homes = row_homes(traces)
    counts = mined_name_counts(sigs, schema.columns, homes)
    assert counts["tool_kind"] == {"name": 2, "observed": 1, "declared": 0, "llm": 0, "total": 3}
    assert counts["column_class"] == {"name": 0, "observed": 2, "declared": 0, "llm": 0, "total": 2}
    assert counts["id_column"] == {"name": 0, "observed": 1, "declared": 0, "llm": 0, "total": 1}
    assert counts["table_name"] == {"name": 1, "observed": 0, "declared": 0, "llm": 0, "total": 1}
    assert counts["row_home"]["name"] == 4
    assert counts["row_home"]["observed"] == 0
    assert counts["row_home"]["total"] == 4
    assert counts["row_home"]["unhomed"] == 1


def test_the_report_counts_off_the_same_records():
    traces = harbour_corpus()
    sigs = mine_tools(traces)
    schema = mine_schema(traces)
    homes = row_homes(traces)
    data = ReportData(tool_sigs=sigs, mined_columns=list(schema.columns), row_homes=homes)
    counts = mined_evidence_counts(data)
    assert counts["tool_kind"]["name"] == 2
    assert counts["tool_kind"]["total"] == 3
    assert counts["table_name"]["name"] == 1
    lines = _mined_evidence_lines(data)
    assert any("Tool kinds, per tool: 2 of 3 rest on a name alone." in line for line in lines)
    assert any("Table names" in line and "1 of 1" in line for line in lines)
    assert any("Row homings, per homed row: 4 of 4 rest on a name alone." in line for line in lines)
    assert any("1 rows no rule could home" in line for line in lines)


def test_empty_records_say_so_instead_of_zeroes():
    data = ReportData()
    assert _mined_evidence_lines(data) == [
        "No mined records were read, so name-alone counts are not shown."
    ]


def test_row_homes_json_round_trips_with_the_new_keys():
    homes = row_homes(harbour_corpus())
    back = json.loads(json.dumps(homes))
    assert back["list_berths"]["homed"]["berth_tags"]["basis"] == "name"
    assert back["list_berths"]["homed"]["berth_tags"]["support_count"] == 0
