"""Tests for builder/mine.py tool kinds: recorded effects first, generic verbs after (D68, D70)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import PTR
from kullback.builder.mine import mine_tools, propose_kind
from kullback.runner.records import RawPtr, ToolCall, ToolCallError, Trace


def one_trace(trace_id: str, calls: list[dict]) -> Trace:
    return Trace(
        trace_id=trace_id,
        raw_hash="h",
        ingest_version="test",
        source="synthetic",
        tool_calls=[ToolCall(**{"raw_ptr": PTR, **c}) for c in calls],
        raw_ptr=PTR,
    )


def sig_by_name(sigs, name):
    return next(s for s in sigs if s.name == name)


def change_trace(trace_id: str, writer: str) -> Trace:
    """Two identical reads of one record with a writer between them that names the same id."""
    return one_trace(trace_id, [
        {"name": "show_widget", "args": {"widget_id": "widget_01"},
         "result": json.dumps({"widget_id": "widget_01", "status": "open"})},
        {"name": writer, "args": {"widget_id": "widget_01"},
         "result": json.dumps({"widget_id": "widget_01", "status": "closed"})},
        {"name": "show_widget", "args": {"widget_id": "widget_01"},
         "result": json.dumps({"widget_id": "widget_01", "status": "closed"})},
    ])


def world_traces(path: str) -> list[Trace]:
    """The invented worlds' tau2 simulations as Traces, without depending on ingest."""
    raw = json.loads((Path(__file__).parents[1] / path).read_text(encoding="utf-8"))
    traces = []
    for sim in raw["simulations"]:
        calls: dict[str, ToolCall] = {}
        order: list[str] = []
        for msg in sim["messages"]:
            for raw_call in msg.get("tool_calls") or []:
                calls[raw_call["id"]] = ToolCall(
                    id=raw_call["id"],
                    name=raw_call["name"],
                    args=raw_call.get("arguments") or {},
                    requestor=raw_call.get("requestor") or "assistant",
                    raw_ptr=RawPtr(file_hash="world", sim_index=0, msg_index=0),
                )
                order.append(raw_call["id"])
            if msg.get("role") == "tool" and msg.get("id") in calls:
                call = calls[msg["id"]]
                call.result = msg.get("content")
                if msg.get("error"):
                    call.error = ToolCallError(class_="business_error", payload=msg.get("content"))
        traces.append(Trace(
            trace_id=sim["id"],
            raw_hash="world",
            ingest_version="test",
            source="tau2",
            tool_calls=[calls[i] for i in order],
            raw_ptr=RawPtr(file_hash="world", sim_index=0),
        ))
    return traces


def test_an_observed_effect_makes_any_name_a_high_confidence_write():
    for name in ["tally_widget", "get_widget", "calculate"]:
        proposal = propose_kind(name, has_effect=True)
        assert proposal.kind == "write"
        assert proposal.confidence == "high"
        assert "changed the world" in proposal.reason


def test_a_verb_no_list_has_with_no_evidence_stays_read_low():
    proposal = propose_kind("tally_widget")
    assert proposal.kind == "read"
    assert proposal.confidence == "low"
    assert "no name rule matched" in proposal.reason


@pytest.mark.parametrize("name", [
    "get_widget", "find_widget", "list_widgets", "search_widgets", "read_widget",
    "lookup_widget", "check_widget", "show_widget", "fetch_widget", "count_widgets",
])
def test_generic_read_verbs_are_medium_not_high(name):
    proposal = propose_kind(name)
    assert proposal.kind == "read"
    assert proposal.confidence == "medium"


@pytest.mark.parametrize("name", [
    "create_widget", "add_widget", "update_widget", "set_widget", "delete_widget",
    "remove_widget", "cancel_widget", "modify_widget", "change_widget", "place_widget",
    "book_widget", "renew_widget", "submit_widget", "send_widget", "assign_widget",
    "schedule_widget", "renew_loan", "place_hold",
])
def test_generic_write_verbs_are_medium_not_high(name):
    proposal = propose_kind(name)
    assert proposal.kind == "write"
    assert proposal.confidence == "medium"


def test_first_word_matching_covers_snake_and_camel_case_but_not_longer_words():
    assert propose_kind("getWidget").kind == "read"
    assert propose_kind("renewLoan").kind == "write"
    assert propose_kind("renewLoan").confidence == "medium"
    assert propose_kind("getter_widget").confidence == "low"
    assert propose_kind("renewal_widget").confidence == "low"


def test_old_retail_verbs_no_longer_decide_kind():
    assert propose_kind("return_widget").kind == "read"
    assert propose_kind("return_widget").confidence == "low"
    assert propose_kind("exchange_widget").kind == "read"
    assert propose_kind("exchange_widget").confidence == "low"


def test_calculation_and_handoff_names_stay_generic():
    for name in ["calculate", "think"]:
        proposal = propose_kind(name)
        assert proposal.kind == "generic", name
        assert proposal.confidence != "low", name
    assert propose_kind("transfer_funds").kind != "generic"


def test_a_made_up_write_proven_by_the_calls_is_a_write():
    sig = sig_by_name(mine_tools([change_trace("t1", "tally_widget")]), "tally_widget")
    assert sig.kind == "write"
    assert sig.kind_confidence == "high"
    assert sig.classified_by == "observed"
    assert sig.unclassified is False


def test_a_write_shown_only_by_its_own_answer_is_a_write():
    traces = [one_trace("t1", [
        {"name": "show_widget", "args": {"widget_id": "widget_01"},
         "result": json.dumps({"widget_id": "widget_01", "status": "open"})},
        {"name": "tally_widget", "args": {"widget_id": "widget_01"},
         "result": json.dumps({"widget_id": "widget_01", "status": "closed"})},
    ])]
    sig = sig_by_name(mine_tools(traces), "tally_widget")
    assert sig.kind == "write"
    assert sig.kind_confidence == "high"
    assert sig.classified_by == "observed"


def test_a_read_looking_name_whose_calls_change_state_is_a_write():
    sig = sig_by_name(mine_tools([change_trace("t1", "fetch_widget")]), "fetch_widget")
    assert sig.kind == "write"
    assert sig.kind_confidence == "high"
    assert sig.classified_by == "observed"
    assert "changed the world" in (sig.kind_reason or "")


def test_an_unknown_name_with_no_evidence_stays_read_low_unclassified():
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}])]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.kind == "read"
    assert sig.kind_confidence == "low"
    assert sig.unclassified is True


def test_world_a_tools_cover_more_than_one_kind():
    sigs = mine_tools(world_traces("fixtures/worlds/world_a/traces_small.json"))
    assert len({s.kind for s in sigs}) > 1
    assert sig_by_name(sigs, "update_loan").kind == "write"
    assert sig_by_name(sigs, "get_loan_details").kind == "read"


def test_world_b_tools_cover_more_than_one_kind():
    sigs = mine_tools(world_traces("fixtures/worlds/world_b/traces_small_tau2.json"))
    assert len({s.kind for s in sigs}) > 1
    assert sig_by_name(sigs, "modify_nursery_order").kind == "write"
    assert sig_by_name(sigs, "get_batch_details").kind == "read"
