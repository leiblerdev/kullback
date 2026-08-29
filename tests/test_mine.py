"""Tests for builder/mine.py: ToolSig mining (D68, D70, D72) and EntitySchema mining (D73)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from conftest import PTR
from harness.builder.mine import (
    SCALAR_RESULT_FIELD,
    classify_column,
    classify_kind,
    gate_tools,
    is_scalar_result,
    mine_schema,
    mine_tools,
    propose_column_class,
    propose_kind,
)
from harness.shared.records import RawPtr, ToolCall, ToolCallError, Trace, as_dict

# --- helpers -----------------------------------------------------------------


def traces_from_raw(raw: dict, file_hash: str = "fixture") -> list[Trace]:
    """Minimal tau2 raw -> Trace conversion, so these tests do not depend on ingest.py."""
    traces = []
    for sim_index, sim in enumerate(raw["simulations"]):
        calls: dict[str, ToolCall] = {}
        order: list[str] = []
        for msg_index, msg in enumerate(sim["messages"]):
            for raw_call in msg.get("tool_calls") or []:
                calls[raw_call["id"]] = ToolCall(
                    id=raw_call["id"],
                    name=raw_call["name"],
                    args=raw_call.get("arguments") or {},
                    requestor=raw_call.get("requestor") or "assistant",
                    raw_ptr=RawPtr(file_hash=file_hash, sim_index=sim_index, msg_index=msg_index),
                )
                order.append(raw_call["id"])
            if msg.get("role") == "tool" and msg.get("id") in calls:
                call = calls[msg["id"]]
                call.result = msg.get("content")
                if msg.get("error"):
                    call.error = ToolCallError(class_="business_error", payload=msg.get("content"))
        traces.append(
            Trace(
                trace_id=sim["id"],
                raw_hash=file_hash,
                ingest_version="test",
                source="tau2",
                tool_calls=[calls[i] for i in order],
                raw_ptr=RawPtr(file_hash=file_hash, sim_index=sim_index),
            )
        )
    return traces


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


def json_model(make_test_model, payloads):
    return make_test_model([json.dumps(p) for p in payloads], loop=True)


@pytest.fixture(scope="module")
def fixture_traces():
    path = Path(__file__).parent / "fixtures" / "tau2_retail_small.json"
    return traces_from_raw(json.loads(path.read_text(encoding="utf-8")))


# --- mine_tools over the tau2 fixture ----------------------------------------


def test_mine_tools_finds_every_called_tool(fixture_traces):
    sigs = mine_tools(fixture_traces)
    assert [s.name for s in sigs] == sorted(
        [
            "exchange_delivered_order_items",
            "find_user_id_by_name_zip",
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types",
            "return_delivered_order_items",
        ]
    )
    order = sig_by_name(sigs, "get_order_details")
    assert order.evidence_strength.call_count == 7
    assert order.evidence_strength.error_count == 0
    assert order.evidence_strength.trace_count == 3
    assert order.source == "observed"


def test_mine_tools_result_and_args_schema_from_observed_results(fixture_traces):
    order = sig_by_name(mine_tools(fixture_traces), "get_order_details")
    result_fields = {f.name for f in order.result_schema}
    assert {"order_id", "user_id", "status", "items", "address"} <= result_fields
    status = next(f for f in order.result_schema if f.name == "status")
    assert status.types == ["str"]
    assert status.count == 7
    assert status.optional is False
    assert status.first_seen in {t.trace_id for t in fixture_traces}
    assert status.declared is False

    assert [f.name for f in order.args_fields] == ["order_id"]
    assert order.args_fields[0].optional is False
    assert order.args_schema["properties"]["order_id"]["type"] == ["str"]
    assert order.args_schema["required"] == ["order_id"]


def test_mine_tools_scalar_result_is_marked_apart_from_a_field_named_value(fixture_traces):
    """A bare scalar result is not an object with one field named value: mixing the two up is
    what made every generated calculate body wrap its answer in {"value": ...} and fail replay
    against the real tool's bare number (docs/live-build.md)."""
    find = sig_by_name(mine_tools(fixture_traces), "find_user_id_by_name_zip")
    assert [f.name for f in find.result_schema] == [SCALAR_RESULT_FIELD]
    assert find.result_schema[0].types == ["str"]
    assert is_scalar_result(find)


def test_mine_tools_kind_by_code_rule(fixture_traces):
    sigs = mine_tools(fixture_traces)
    reads = ["get_order_details", "get_user_details", "find_user_id_by_name_zip", "list_all_product_types"]
    for name in reads:
        sig = sig_by_name(sigs, name)
        assert sig.kind == "read"
        assert sig.unclassified is False
        assert sig.classified_by == "rule"
        assert "prefix" in (sig.kind_reason or "")
    for name in ["exchange_delivered_order_items", "return_delivered_order_items"]:
        sig = sig_by_name(sigs, name)
        assert sig.kind == "write"
        assert sig.unclassified is False


def test_mine_tools_output_round_trips_through_json(fixture_traces):
    sigs = mine_tools(fixture_traces)
    back = json.loads(json.dumps([as_dict(s) for s in sigs]))
    by_name = {s["name"]: s for s in back}
    assert set(by_name) == {s.name for s in sigs}
    order = by_name["get_order_details"]
    assert order["kind"] == "read"
    assert order["evidence_strength"]["call_count"] == 7
    assert {f["name"] for f in order["result_schema"]} >= {"order_id", "status", "items"}
    assert by_name["exchange_delivered_order_items"]["kind"] == "write"
    assert order["evidence"] == sig_by_name(sigs, "get_order_details").evidence


# --- D72: union of everything observed ---------------------------------------


def test_result_schema_is_the_union_with_counts_and_seen():
    traces = [
        one_trace("t1", [{"name": "get_thing", "args": {"id": "1"}, "result": '{"a": 1}'}]),
        one_trace(
            "t2",
            [
                {"name": "get_thing", "args": {"id": "2"}, "result": '{"a": 2, "b": "x"}'},
                {"name": "get_thing", "args": {"id": "3"}, "result": '{"a": 3, "b": "y"}'},
            ],
        ),
    ]
    sig = sig_by_name(mine_tools(traces), "get_thing")
    fields = {f.name: f for f in sig.result_schema}
    assert fields["a"].count == 3 and fields["a"].optional is False
    assert fields["b"].count == 2 and fields["b"].optional is True
    assert fields["a"].first_seen == "t1" and fields["a"].last_seen == "t2"
    assert fields["b"].first_seen == "t2"
    assert sig.evidence == ["t1", "t2"]


def test_args_schema_is_the_union_too():
    traces = [
        one_trace(
            "t1",
            [
                {"name": "modify_thing", "args": {"id": "1"}, "result": "ok"},
                {"name": "modify_thing", "args": {"id": "2", "note": "hi"}, "result": "ok"},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "modify_thing")
    fields = {f.name: f for f in sig.args_fields}
    assert fields["id"].optional is False
    assert fields["note"].optional is True
    assert sig.args_schema["required"] == ["id"]


def test_declared_schema_is_recorded_beside_the_observed_one():
    declared = [
        {
            "name": "get_thing",
            "description": "Read one thing.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "verbose": {"type": "boolean"}},
                "required": ["id"],
            },
        }
    ]
    traces = [
        one_trace(
            "t1",
            [{"name": "get_thing", "args": {"id": "1"}, "result": '{"a": 1}'}],
            tools_declared=declared,
        )
    ]
    sig = sig_by_name(mine_tools(traces), "get_thing")
    assert sig.description == "Read one thing."
    fields = {f.name: f for f in sig.args_fields}
    assert fields["id"].declared is True
    assert fields["verbose"].declared is True
    assert fields["verbose"].count == 0
    assert fields["verbose"].optional is True


def test_a_declared_tool_never_called_is_still_mined():
    declared = [{"name": "never_called", "description": "d", "parameters": {"properties": {}}}]
    traces = [one_trace("t1", [{"name": "get_thing", "args": {}, "result": "x"}], tools_declared=declared)]
    sig = sig_by_name(mine_tools(traces), "never_called")
    assert sig.source == "declared"
    assert sig.evidence_strength.call_count == 0


# --- errors ------------------------------------------------------------------


def test_error_shapes_keep_the_verbatim_payload():
    traces = [
        one_trace(
            "t1",
            [
                {
                    "name": "cancel_thing",
                    "args": {"id": "1"},
                    "result": "Error: order not found",
                    "error": ToolCallError(class_="not_found_entity", payload="Error: order not found"),
                },
                {"name": "cancel_thing", "args": {"id": "2"}, "result": '{"status": "cancelled"}'},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "cancel_thing")
    assert sig.evidence_strength.error_count == 1
    assert [(e.class_, e.count) for e in sig.error_shapes] == [("not_found_entity", 1)]
    assert sig.error_shapes[0].sample_payload == "Error: order not found"
    # a failed call contributes no result fields
    assert {f.name for f in sig.result_schema} == {"status"}


# --- D68 and D70: kind classification ----------------------------------------


def test_propose_kind_uses_prefixes_and_says_why():
    assert propose_kind("search_orders").kind == "read"
    assert "prefix" in propose_kind("search_orders").reason
    assert propose_kind("update_user").kind == "write"
    assert propose_kind("update_user").confidence == "high"
    assert propose_kind("frobnicate").kind == "read"
    assert propose_kind("frobnicate").confidence == "low"
    assert "no name rule matched" in propose_kind("frobnicate").reason


def test_calculation_and_handoff_names_are_generic_as_tau2_has_them():
    """tau2's tools.py marks calculate and transfer_to_human_agents GENERIC; compile_env emits the
    kind verbatim, so a read here would emit the wrong ToolType and leave the D70 review open."""
    for name in ["calculate", "transfer_to_human_agents", "think"]:
        proposal = propose_kind(name)
        assert proposal.kind == "generic", name
        assert proposal.confidence != "low", name
    # a transfer that moves money is a write, not a handoff to a person
    assert propose_kind("transfer_funds").kind != "generic"


def test_unknown_name_defaults_to_read_and_unclassified(fixture_traces):
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}])]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.kind == "read"
    assert sig.unclassified is True
    assert sig.kind_confidence == "low"


def test_classify_kind_hook_sees_the_evidence_and_sets_the_class(make_test_model):
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {"id": "1"}, "result": '{"a": 1}'}])]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    model = json_model(make_test_model, [{"kind": "write", "confidence": "high", "reason": "it frobnicates"}])
    proposal = classify_kind(model, sig, {"note": "n"})
    assert (proposal.kind, proposal.confidence, proposal.reason) == ("write", "high", "it frobnicates")
    assert proposal.classified_by == "llm"
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["tool"]["name"] == "frobnicate"
    assert [f["name"] for f in payload["tool"]["result_fields"]] == ["a"]
    assert payload["tool"]["calls"] == 1
    assert payload["evidence"] == {"note": "n"}


def test_mine_tools_with_a_model_applies_the_llm_class(make_test_model):
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {"id": "1"}, "result": "ok"}])]
    model = json_model(make_test_model, [{"kind": "write", "confidence": "high", "reason": "writes"}])
    sig = sig_by_name(mine_tools(traces, model=model), "frobnicate")
    assert sig.kind == "write"
    assert sig.kind_confidence == "high"
    assert sig.classified_by == "llm"
    assert sig.unclassified is False


def test_low_confidence_llm_leaves_the_D70_default(make_test_model):
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}])]
    model = json_model(make_test_model, [{"kind": "write", "confidence": "low", "reason": "guessing"}])
    sig = sig_by_name(mine_tools(traces, model=model), "frobnicate")
    assert sig.kind == "read"
    assert sig.unclassified is True
    assert sig.kind_confidence == "low"
    assert "guessing" in (sig.kind_reason or "")


def test_unparseable_llm_reply_leaves_the_code_proposal(make_test_model):
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}])]
    model = make_test_model(["sorry, I cannot say"], loop=True)
    sig = sig_by_name(mine_tools(traces, model=model), "frobnicate")
    assert sig.kind == "read"
    assert sig.unclassified is True
    assert sig.classified_by == "rule"


def test_observed_effects_override_the_llm(make_test_model):
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "pending"}'},
                {"name": "frobnicate", "args": {"id": "7"}, "result": "done"},
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "gone"}'},
            ],
        )
    ]
    model = json_model(make_test_model, [{"kind": "read", "confidence": "high", "reason": "looks like a read"}])
    sig = sig_by_name(mine_tools(traces, model=model), "frobnicate")
    assert sig.kind == "write"
    assert sig.kind_confidence == "high"
    assert sig.classified_by == "observed"
    assert sig.unclassified is False
    assert [(e.trace_id, e.field) for e in sig.effects_observed] == [("t1", "get_thing.status")]


def test_no_effect_when_the_later_read_is_unchanged():
    """The one condition on its own: the call names the same id, and nothing moved."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "pending"}'},
                {"name": "frobnicate", "args": {"id": "7"}, "result": "done"},
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "pending"}'},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.effects_observed == []
    assert sig.kind == "read"
    assert sig.unclassified is True


def test_no_effect_when_the_call_names_nothing_the_change_touches():
    """The other condition on its own: something moved, but this call names neither the id nor the value."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "pending"}'},
                {"name": "frobnicate", "args": {"id": "9"}, "result": "done"},
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "gone"}'},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.effects_observed == []
    assert sig.kind == "read"


def test_llm_fills_a_missing_result_schema_and_marks_the_source(make_test_model):
    traces = [
        one_trace(
            "t1",
            [
                {
                    "name": "do_thing",
                    "args": {"id": "1"},
                    "result": None,
                    "error": ToolCallError(class_="transient", payload="boom"),
                }
            ],
        )
    ]
    model = json_model(
        make_test_model,
        [
            {"kind": "write", "confidence": "high", "reason": "writes"},
            {"fields": [{"name": "ok", "types": ["bool"]}]},
        ],
    )
    sig = sig_by_name(mine_tools(traces, model=model), "do_thing")
    assert sig.source == "llm"
    assert [(f.name, f.count) for f in sig.result_schema] == [("ok", 0)]


# --- the mine gate -----------------------------------------------------------


def test_gate_needs_three_observed_calls_or_the_llm_flag(fixture_traces):
    gate = gate_tools(mine_tools(fixture_traces))
    assert gate.stage == "mine"
    assert gate.passed is False
    assert gate.metrics["tools"] == 7
    assert gate.metrics["thin"] == 3
    joined = " ".join(gate.failures)
    for name in ["list_all_product_types", "return_delivered_order_items", "exchange_delivered_order_items"]:
        assert name in joined


def test_a_user_requestor_call_is_never_mined_into_a_toolsig_or_schema():
    """Telecom's traces interleave the assistant and the simulated user's own tool calls, run against
    the user's own phone (docs/cross-domain-check.md, Judgement). Retail and airline never carry a
    user-requestor call, so this is new coverage, not a change to their behavior."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"}, "result": '{"order_id": "#W1"}'},
                {"name": "get_order_details", "args": {"order_id": "#W2"}, "result": '{"order_id": "#W2"}'},
                {"name": "get_order_details", "args": {"order_id": "#W3"}, "result": '{"order_id": "#W3"}'},
                {"name": "check_network_status", "args": {}, "result": '{"signal": "5g"}', "requestor": "user"},
            ],
        )
    ]
    sigs = mine_tools(traces)
    assert not any(s.name == "check_network_status" for s in sigs)
    assert sig_by_name(sigs, "get_order_details").evidence_strength.call_count == 3

    schema = mine_schema(traces)
    assert schema.tables == ["orders"]
    assert "signal" not in {c.name for c in schema.columns}

    gate = gate_tools(sigs, traces)
    assert gate.metrics["skipped_user_calls"] == 1
    assert gate.metrics["tools"] == 1


def test_gate_tools_without_traces_reports_zero_skipped_as_before():
    """Retail and airline have no user-requestor calls; the old call shape, with no traces given,
    must keep working unchanged."""
    sigs = mine_tools([one_trace("t1", [{"name": "get_thing", "args": {}, "result": '{"a": 1}'}] * 3)])
    assert gate_tools(sigs).metrics["skipped_user_calls"] == 0


def test_gate_passes_when_every_tool_is_thick_or_flagged(make_test_model):
    traces = [
        one_trace(
            "t1",
            [{"name": "get_thing", "args": {"id": str(i)}, "result": '{"a": 1}'} for i in range(3)]
            + [
                {
                    "name": "do_thing",
                    "args": {},
                    "result": None,
                    "error": ToolCallError(class_="transient", payload="boom"),
                }
            ],
        )
    ]
    model = json_model(
        make_test_model,
        [{"kind": "write", "confidence": "high", "reason": "writes"}, {"fields": [{"name": "ok", "types": ["bool"]}]}],
    )
    gate = gate_tools(mine_tools(traces, model=model))
    assert gate.passed is True
    assert gate.failures == []


# --- D73: columns ------------------------------------------------------------


def test_propose_column_class_rules():
    assert propose_column_class("orders", "created_at", ["2025-06-05T14:01:10"]).column_class == "exempt"
    assert propose_column_class("orders", "retry_count", [1, 2, 3]).column_class == "exempt"
    assert propose_column_class("orders", "order_id", ["#W1", "#W2"]).column_class == "hard"
    assert propose_column_class("orders", "status", ["pending", "delivered", "pending"]).column_class == "hard"
    long = [
        "a paragraph of prose that a person wrote and nobody will match by string equality " + str(i)
        for i in range(6)
    ]
    assert propose_column_class("orders", "note", long).column_class == "semantic"
    proposal = propose_column_class("orders", "note", long)
    assert proposal.confidence in {"low", "medium", "high"}
    assert proposal.evidence["distinct"] == 6


def test_a_nullable_enum_is_still_an_enum():
    proposal = propose_column_class("orders", "cancel_reason", [None, "no longer needed", None, "ordered by mistake"])
    assert proposal.column_class == "hard"
    assert "enum" in proposal.reason
    assert proposal.evidence["types"] == ["NoneType", "str"]


def test_mine_schema_from_the_tau2_db_json(tau2_retail_dir):
    schema = mine_schema([], db_json_path=tau2_retail_dir / "db.json")
    assert schema.tables == ["orders", "products", "users"]
    assert col(schema, "orders", "order_id").class_ == "hard"
    assert col(schema, "orders", "status").class_ == "hard"
    assert col(schema, "orders", "status").classified_by == "rule"
    assert col(schema, "orders", "status").class_rule == "hard"
    assert schema.id_patterns["orders.order_id"] == r"^#W\d{7}$"
    assert col(schema, "users", "email").samples


def test_mine_schema_from_traces_names_tables_after_the_entity(fixture_traces):
    schema = mine_schema(fixture_traces)
    assert {"orders", "users", "products"} <= set(schema.tables)
    assert col(schema, "orders", "status").class_ == "hard"
    assert col(schema, "users", "email")
    assert "orders.order_id" in schema.id_patterns


def test_mine_schema_merges_db_json_and_traces(fixture_traces, tau2_retail_dir):
    schema = mine_schema(fixture_traces, db_json_path=tau2_retail_dir / "db.json")
    assert schema.tables == ["orders", "products", "users"]
    assert col(schema, "orders", "status").evidence["count"] > 1000


def test_classify_column_hook_can_override_the_rule(make_test_model):
    traces = [one_trace("t1", [{"name": "get_note", "args": {}, "result": '{"note_id": "1", "body": "hi"}'}])]
    model = json_model(make_test_model, [{"class": "semantic", "confidence": "high", "reason": "free text"}])
    schema = mine_schema(traces, model=model)
    body = col(schema, "notes", "body")
    assert body.class_ == "semantic"
    assert body.class_rule == "hard"
    assert body.classified_by == "llm"
    assert body.class_reason == "free text"
    assert body.class_confidence == "high"


def test_classify_column_is_called_with_the_proposal_and_evidence(make_test_model):
    model = json_model(make_test_model, [{"class": "exempt", "confidence": "medium", "reason": "a counter"}])
    proposal = propose_column_class("orders", "status", ["a", "b"])
    out = classify_column(model, "orders", "status", proposal, ["a", "b"])
    assert out.column_class == "exempt"
    assert out.confidence == "medium"
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["column"] == "status"
    assert payload["table"] == "orders"
    assert payload["proposed_class"] == "hard"
    assert payload["evidence"]["distinct"] == 2
    assert payload["samples"] == ["a", "b"]


def test_schema_round_trips_through_json(fixture_traces):
    schema = mine_schema(fixture_traces)
    back = json.loads(json.dumps(as_dict(schema)))
    assert back["tables"] == schema.tables
    assert {"orders", "users", "products"} <= set(back["tables"])
    status = next(c for c in back["columns"] if c["table"] == "orders" and c["name"] == "status")
    assert status["class"] == "hard"  # the alias, not the field name
    assert status["classified_by"] == "rule"
    assert back["id_patterns"]["orders.order_id"] == schema.id_patterns["orders.order_id"]


# --- the full raw corpus -----------------------------------------------------


def retail_tool_names() -> set[str]:
    # Kullback keeps vendor/ under the package root; the brain keeps it one level up,
    # at monitoring-tool/vendor. Try the kullback layout first, same as raw_dir in conftest.
    rel = Path("vendor") / "tau2-bench" / "src" / "tau2" / "domains" / "retail" / "tools.py"
    tests_dir = Path(__file__).resolve().parent
    for root in (tests_dir.parent, tests_dir.parents[1]):
        path = root / rel
        if path.is_file():
            break
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    names.add(item.name)
    assert len(names) >= 16
    return names


@pytest.mark.slow
def test_mine_tools_on_the_full_raw_corpus(retail_raw_files):
    # Retail only: raw_dir also holds airline and telecom traces, checked separately.
    traces: list[Trace] = []
    for path in retail_raw_files:
        traces += traces_from_raw(json.loads(path.read_text(encoding="utf-8")), file_hash=path.name)
    assert len(traces) > 100
    sigs = mine_tools(traces)
    mined = {s.name for s in sigs}
    missing = retail_tool_names() - mined
    assert not missing, f"tools in tau2's retail tools.py that the traces never showed: {sorted(missing)}"
    assert gate_tools(sigs).passed is True
    order = sig_by_name(sigs, "get_order_details")
    assert order.evidence_strength.call_count > 1000
    assert {f.name for f in order.result_schema} >= {"order_id", "status", "items"}
    writes = {s.name for s in sigs if s.kind == "write"}
    assert "cancel_pending_order" in writes and "modify_user_address" in writes


@pytest.mark.slow
def test_mine_schema_on_the_full_raw_corpus(raw_dir, tau2_retail_dir):
    traces: list[Trace] = []
    for path in sorted(raw_dir.glob("*.json")):
        traces += traces_from_raw(json.loads(path.read_text(encoding="utf-8")), file_hash=path.name)
    schema = mine_schema(traces, db_json_path=tau2_retail_dir / "db.json")
    # "items" joins the three db.json tables: get_item_details returns rows keyed by item_id
    assert set(schema.tables) >= {"orders", "products", "users"}
    assert col(schema, "orders", "status").class_ == "hard"
    assert set(schema.id_patterns) >= {"orders.order_id", "users.user_id", "products.product_id"}


# --- D67: unknown errors are a flag on the Environment ---

def test_a_tool_whose_errors_are_mostly_unknown_is_flagged():
    from harness.builder.mine import unknown_error_flags
    from harness.shared.records import ErrorShape, ToolSig

    murky = ToolSig(name="charge_card", error_shapes=[
        ErrorShape(class_="unknown", count=7), ErrorShape(class_="business_error", count=3)])
    clear = ToolSig(name="get_order", error_shapes=[ErrorShape(class_="not_found_entity", count=9)])
    quiet = ToolSig(name="list_products")
    flags = unknown_error_flags([murky, clear, quiet])
    assert len(flags) == 1
    assert flags[0].startswith("charge_card: 7 of 10 observed errors are unknown (70%)")


def test_the_unknown_share_threshold_is_a_share_not_a_count():
    from harness.builder.mine import unknown_error_flags
    from harness.shared.records import ErrorShape, ToolSig

    rare = ToolSig(name="t", error_shapes=[
        ErrorShape(class_="unknown", count=1), ErrorShape(class_="transient", count=99)])
    assert unknown_error_flags([rare]) == []
    assert unknown_error_flags([rare], threshold=0.005) != []


# --- D73: re-run evidence overrides the rule and the LLM ---

def test_a_column_that_varies_across_successful_reruns_becomes_exempt():
    from harness.builder.mine import exempt_from_reruns
    from harness.shared.records import Column, EntitySchema

    schema = EntitySchema(tables=["orders"], columns=[
        Column(table="orders", name="status", class_="hard"),
        Column(table="orders", name="updated_at", class_="hard", classified_by="llm"),
    ])
    states = [
        {"orders": {"o1": {"status": "cancelled", "updated_at": "2026-01-01T10:00:00Z"}}},
        {"orders": {"o1": {"status": "cancelled", "updated_at": "2026-01-01T10:00:09Z"}}},
    ]
    out = exempt_from_reruns(schema, states)
    by_name = {c.name: c for c in out.columns}
    assert by_name["updated_at"].class_ == "exempt"
    assert by_name["updated_at"].classified_by == "observed"
    assert by_name["status"].class_ == "hard"


def test_one_rerun_is_not_evidence_of_anything():
    from harness.builder.mine import exempt_from_reruns
    from harness.shared.records import Column, EntitySchema

    schema = EntitySchema(tables=["orders"], columns=[Column(table="orders", name="x", class_="hard")])
    assert exempt_from_reruns(schema, [{"orders": {"o1": {"x": 1}}}]).columns[0].class_ == "hard"


# --- D95: a truncated result is reconstructed, tagged and Assisted ---

def test_a_truncated_result_is_rebuilt_from_the_schema_and_complete_calls():
    """The shape ingest actually produces: the cut JSON string is still in `result` (D95)."""
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_ticket", result_schema=[
        FieldStat(name="id"), FieldStat(name="subject"), FieldStat(name="history")])
    cut = ToolCall(name="get_ticket", args={"id": "t1"}, truncated=True, visible_len=40, cut_marker="...",
                   result='{"id": "t1", "subject": "printer on fire", "hist...', raw_ptr=PTR)
    whole = ToolCall(name="get_ticket", args={"id": "t2"},
                     result='{"id": "t2", "subject": "late delivery", "history": ["opened"]}', raw_ptr=PTR)
    out = reconstruct_truncated(cut, sig, [cut, whole])
    assert out["result"]["id"] == "t1", "the cut row must keep its own id, never a donor's"
    assert out["result"]["subject"] == "printer on fire", "a field the agent saw is not invented"
    assert out["result"]["history"] == ["opened"]
    assert out["reconstructed_fields"] == ["history"]
    assert out["tags"] == ["reconstructed"]
    assert out["assisted"] is True
    assert out["cut_marker"] == "..."
    assert out["visible_len"] == 40


def test_a_truncated_result_already_parsed_is_filled_the_same_way():
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_ticket", result_schema=[
        FieldStat(name="id"), FieldStat(name="subject"), FieldStat(name="history")])
    cut = ToolCall(name="get_ticket", args={"id": "t1"}, truncated=True, visible_len=40,
                   cut_marker="...", result={"id": "t1"}, raw_ptr=PTR)
    whole = ToolCall(name="get_ticket", args={"id": "t2"},
                     result={"id": "t2", "subject": "late delivery", "history": ["opened"]}, raw_ptr=PTR)
    out = reconstruct_truncated(cut, sig, [cut, whole])
    assert out["result"]["id"] == "t1"
    assert out["result"]["subject"] == "late delivery"
    assert sorted(out["reconstructed_fields"]) == ["history", "subject"]


def test_a_truncated_dict_keeps_its_visible_fields_even_when_the_schema_also_saw_lists():
    """Row 10: the shape to reconstruct is this call's own visible content, not the tool's whole
    schema. A cut dict result must not be thrown away for a fabricated list row just because some
    other call to the same tool returned a list."""
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_thing", result_schema=[FieldStat(name="id"), FieldStat(name="[].x")])
    cut = ToolCall(name="get_thing", args={"id": "1"}, truncated=True, visible_len=26, cut_marker='"',
                   result='{"id": "1", "extra": "vis', raw_ptr=PTR)
    donor = ToolCall(name="get_thing", args={"id": "2"}, result=[{"x": 1}], raw_ptr=PTR)
    out = reconstruct_truncated(cut, sig, [cut, donor])
    assert out["result"] == {"id": "1"}, "the visible id must survive, not be replaced by a donor row"
    assert out["reconstructed_fields"] == []


def test_a_complete_result_is_never_reconstructed():
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_ticket", result_schema=[FieldStat(name="id")])
    whole = ToolCall(name="get_ticket", args={}, result={"id": "t1"}, raw_ptr=PTR)
    assert reconstruct_truncated(whole, sig, [whole]) is None


# --- effect attribution: only the tool that explains the change gets the credit (D68, D70) ---


def test_effect_credit_goes_to_the_tool_that_explains_it_not_the_last_candidate():
    """D70: a change 'explained only by this tool'. A prefix-less read that merely names the same id
    after the real write must not become an observed write, and the write must keep its evidence."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "pending"}'},
                {"name": "cancel_pending_order", "args": {"order_id": "#W1", "reason": "no longer needed"},
                 "result": '{"order_id": "#W1", "status": "cancelled"}'},
                {"name": "check_order_eligibility", "args": {"order_id": "#W1"}, "result": '{"eligible": false}'},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "cancelled"}'},
            ],
        )
    ]
    sigs = mine_tools(traces)
    check = sig_by_name(sigs, "check_order_eligibility")
    cancel = sig_by_name(sigs, "cancel_pending_order")
    assert check.effects_observed == []
    assert check.kind == "read"
    assert check.classified_by == "rule"
    assert check.unclassified is True
    assert [e.field for e in cancel.effects_observed] == ["get_order_details.status"]
    assert cancel.classified_by == "observed"


def test_two_writes_are_credited_per_field_by_the_value_they_carry():
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "pending", "address": "a"}'},
                {"name": "modify_pending_order_address", "args": {"order_id": "#W1", "address": "b"},
                 "result": "ok"},
                {"name": "cancel_pending_order", "args": {"order_id": "#W1"}, "result": "ok"},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "cancelled", "address": "b"}'},
            ],
        )
    ]
    effects = {name: [e.field for e in obs] for name, obs in observed_effects(traces).items()}
    assert effects["modify_pending_order_address"] == ["get_order_details.address"]
    assert effects["cancel_pending_order"] == ["get_order_details.status"]


def test_no_actor_when_two_candidates_could_explain_the_same_field():
    """With nothing to tell two writes apart, the field gets no actor rather than the wrong one."""
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "pending"}'},
                {"name": "cancel_pending_order", "args": {"order_id": "#W1"}, "result": "ok"},
                {"name": "return_delivered_order_items", "args": {"order_id": "#W1"}, "result": "ok"},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "cancelled"}'},
            ],
        )
    ]
    assert observed_effects(traces) == {}


def test_a_timestamp_that_moves_on_its_own_is_not_an_effect():
    """D73's exempt columns are not evidence that anything wrote, or every intermediate call becomes
    a high-confidence observed write that neither the LLM nor the rule can correct."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_account", "args": {"user_id": "u1"},
                 "result": '{"user_id": "u1", "balance": 10, "last_checked_at": "2026-01-01T10:00:00Z"}'},
                {"name": "verify_identity", "args": {"user_id": "u1", "pin": "1234"}, "result": '{"ok": true}'},
                {"name": "get_account", "args": {"user_id": "u1"},
                 "result": '{"user_id": "u1", "balance": 10, "last_checked_at": "2026-01-01T10:00:07Z"}'},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "verify_identity")
    assert sig.effects_observed == []
    assert sig.kind == "read"
    # the same trace with a real change beside the timestamp still finds the write
    traces[0].tool_calls[2].result = (
        '{"user_id": "u1", "balance": 20, "last_checked_at": "2026-01-01T10:00:07Z"}')
    sig = sig_by_name(mine_tools(traces), "verify_identity")
    assert [e.field for e in sig.effects_observed] == ["get_account.balance"]


def test_a_generic_tool_is_never_the_actor_of_someone_elses_write():
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "pending"}'},
                {"name": "calculate", "args": {"expression": "#W1"}, "result": "2"},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "status": "cancelled"}'},
            ],
        )
    ]
    assert observed_effects(traces) == {}


def test_a_write_named_only_by_the_new_value_is_credited():
    """The 'names the new value' half of the rule has to work for strings and booleans, not only numbers."""
    from harness.builder.mine import observed_effects

    for new, shown in [("SAVE10", '"SAVE10"'), (5, "5"), (True, "true")]:
        traces = [
            one_trace(
                "t1",
                [
                    {"name": "get_cart", "args": {"cart": "c1"}, "result": '{"cart": "c1", "field": null}'},
                    {"name": "apply_change", "args": {"value": new}, "result": "ok"},
                    {"name": "get_cart", "args": {"cart": "c1"}, "result": '{"cart": "c1", "field": %s}' % shown},
                ],
            )
        ]
        effects = observed_effects(traces)
        assert [e.field for e in effects["apply_change"]] == ["get_cart.field"], new


def test_a_field_that_goes_from_null_to_a_list_is_a_change():
    """tau2 sets exchange_items from null to a list; comparing only the keys both sides share misses it."""
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "exchange_items": null}'},
                {"name": "frobnicate", "args": {"order_id": "#W1"}, "result": "ok"},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "exchange_items": ["i1"]}'},
            ],
        )
    ]
    effects = observed_effects(traces)
    assert [e.field for e in effects["frobnicate"]] == ["get_order_details.exchange_items"]


def test_a_field_that_appears_only_after_the_call_is_a_change():
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_order_details", "args": {"order_id": "#W1"}, "result": '{"order_id": "#W1"}'},
                {"name": "frobnicate", "args": {"order_id": "#W1"}, "result": "ok"},
                {"name": "get_order_details", "args": {"order_id": "#W1"},
                 "result": '{"order_id": "#W1", "cancel_reason": "no longer needed"}'},
            ],
        )
    ]
    effects = observed_effects(traces)
    assert [e.field for e in effects["frobnicate"]] == ["get_order_details.cancel_reason"]


# --- D72: what a truncated or a declared schema does to the union ---


def test_a_truncated_result_does_not_pollute_the_result_union():
    """D95: the schema is what makes reconstruction possible; a cut result must not add a bogus
    'value: str' field nor make every real field optional."""
    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_ticket", "args": {"id": "1"}, "result": '{"id": "1", "subject": "a"}'},
                {"name": "get_ticket", "args": {"id": "2"}, "result": '{"id": "2", "subject": "b"}'},
                {"name": "get_ticket", "args": {"id": "3"}, "result": '{"id": "3", "subj...',
                 "truncated": True, "visible_len": 18, "cut_marker": "..."},
            ],
        )
    ]
    sig = sig_by_name(mine_tools(traces), "get_ticket")
    fields = {f.name: f for f in sig.result_schema}
    assert set(fields) == {"id", "subject"}
    assert fields["id"].optional is False
    assert fields["id"].count == 2
    # the call itself is still evidence that the tool was called, and its arguments still count
    assert sig.evidence_strength.call_count == 3
    assert {f.name for f in sig.args_fields} == {"id"}


def test_a_truncated_result_is_not_effect_evidence():
    from harness.builder.mine import observed_effects

    traces = [
        one_trace(
            "t1",
            [
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "status": "pending"}'},
                {"name": "frobnicate", "args": {"id": "7"}, "result": "ok"},
                {"name": "get_thing", "args": {"id": "7"}, "result": '{"id": "7", "stat...',
                 "truncated": True},
            ],
        )
    ]
    assert observed_effects(traces) == {}


def test_a_list_result_unions_the_types_of_every_item():
    traces = [one_trace("t1", [{"name": "get_rows", "args": {}, "result": '[{"a": 1}, {"a": "x"}]'}])]
    sig = sig_by_name(mine_tools(traces), "get_rows")
    fields = {f.name: f for f in sig.result_schema}
    assert set(fields["[].a"].types) == {"int", "str"}
    assert fields["[].a"].count == 1


def test_a_declared_output_schema_marks_and_adds_result_fields():
    """D72 puts `declared` on result fields, so the review can see where the customer's own
    definition and their logs disagree."""
    declared = [
        {
            "name": "get_thing",
            "parameters": {"properties": {}},
            "output_schema": {"properties": {"a": {"type": "integer"}, "z": {"type": "string"}},
                              "required": ["a", "z"]},
        }
    ]
    traces = [one_trace("t1", [{"name": "get_thing", "args": {}, "result": '{"a": 1}'}], tools_declared=declared)]
    sig = sig_by_name(mine_tools(traces), "get_thing")
    fields = {f.name: f for f in sig.result_schema}
    assert fields["a"].declared is True
    assert fields["a"].count == 1
    assert fields["z"].declared is True
    assert fields["z"].count == 0
    assert fields["z"].types == ["str"]
    assert fields["z"].optional is False


def test_mcp_annotations_are_a_rule_and_reach_the_llm(make_test_model):
    """D68 lists MCP annotations among the context code gathers."""
    from harness.builder.mine import annotations_of

    declared = [{"name": "frobnicate", "description": "d", "parameters": {"properties": {}},
                 "annotations": {"readOnlyHint": False, "destructiveHint": True}}]
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}], tools_declared=declared)]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.kind == "write"
    assert sig.unclassified is False
    assert "destructiveHint" in (sig.kind_reason or "")

    read_only = [{"name": "frobnicate", "parameters": {"properties": {}},
                  "annotations": {"readOnlyHint": True}}]
    traces = [one_trace("t1", [{"name": "frobnicate", "args": {}, "result": "ok"}], tools_declared=read_only)]
    sig = sig_by_name(mine_tools(traces), "frobnicate")
    assert sig.kind == "read"
    assert sig.unclassified is False

    model = json_model(make_test_model, [{"kind": "write", "confidence": "low", "reason": "guessing"}])
    mine_tools(traces, model=model)
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["evidence"]["annotations"] == {"readOnlyHint": True}
    assert annotations_of(read_only[0]) == {"readOnlyHint": True}


def test_a_name_rule_that_disagrees_with_the_annotations_says_so():
    declared = [{"name": "get_thing", "parameters": {"properties": {}},
                 "annotations": {"destructiveHint": True}}]
    traces = [one_trace("t1", [{"name": "get_thing", "args": {}, "result": "ok"}], tools_declared=declared)]
    sig = sig_by_name(mine_tools(traces), "get_thing")
    assert sig.kind == "read"
    assert "disagree" in (sig.kind_reason or "")


# --- D95: reconstruction keeps what the agent saw ---


def test_reconstruction_prefers_a_donor_with_the_same_arguments():
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_ticket", result_schema=[FieldStat(name="id"), FieldStat(name="subject")])
    cut = ToolCall(name="get_ticket", args={"id": "t1"}, truncated=True, result={"id": "t1"}, raw_ptr=PTR)
    other = ToolCall(name="get_ticket", args={"id": "t2"}, result={"id": "t2", "subject": "other ticket"}, raw_ptr=PTR)
    same = ToolCall(
        name="get_ticket", args={"id": "t1"}, result={"id": "t1", "subject": "the real subject"}, raw_ptr=PTR
    )
    out = reconstruct_truncated(cut, sig, [cut, other, same])
    assert out["result"]["subject"] == "the real subject"


def test_reconstruction_of_a_cut_list_result_is_still_a_list():
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="search_orders",
                  result_schema=[FieldStat(name="[].order_id"), FieldStat(name="[].status")])
    cut = ToolCall(name="search_orders", args={"q": "x"}, truncated=True,
                   result='[{"order_id": "#W1", "st...', raw_ptr=PTR)
    whole = ToolCall(name="search_orders", args={"q": "y"},
                     result='[{"order_id": "#W2", "status": "pending"}]', raw_ptr=PTR)
    out = reconstruct_truncated(cut, sig, [cut, whole])
    assert isinstance(out["result"], list)
    assert out["result"][0]["order_id"] == "#W1"
    assert out["result"][0]["status"] == "pending"
    assert out["reconstructed_fields"] == ["[].status"]


def test_reconstruction_never_borrows_another_entitys_id():
    from harness.builder.mine import reconstruct_truncated
    from harness.shared.records import FieldStat, ToolSig

    sig = ToolSig(name="get_ticket", result_schema=[FieldStat(name="ticket_id"), FieldStat(name="subject")])
    cut = ToolCall(name="get_ticket", args={"id": "t1"}, truncated=True, result='{"subj...', raw_ptr=PTR)
    whole = ToolCall(name="get_ticket", args={"id": "t2"},
                     result='{"ticket_id": "t2", "subject": "late delivery"}', raw_ptr=PTR)
    out = reconstruct_truncated(cut, sig, [cut, whole])
    assert "ticket_id" not in out["result"]
    assert out["reconstructed_fields"] == ["subject"]


# --- D73: what a column class may not be settled by ---


def test_a_money_column_that_happens_to_increase_is_not_exempt():
    """Three increasing values are not a counter; a gift-card balance nobody compares is a silent pass."""
    proposal = propose_column_class("gift_cards", "balance", [10.0, 25.5, 100.0])
    assert proposal.column_class != "exempt"
    assert propose_column_class("orders", "amount", [1, 2, 3]).column_class != "exempt"
    counter = propose_column_class("jobs", "attempts", [1, 2, 3, 4, 5])
    assert counter.column_class == "exempt"
    assert counter.confidence == "low", "a counter found by shape alone still goes to review"


def test_business_dates_and_versions_are_not_high_confidence_exempt():
    for name in ["date_of_birth", "time_zone", "delivery_date", "version"]:
        proposal = propose_column_class("users", name, ["1990-01-01", "1985-05-05", "2000-02-02"])
        assert not (proposal.column_class == "exempt" and proposal.confidence == "high"), name
    assert propose_column_class("users", "date_of_birth", ["1990-01-01"]).column_class == "hard"
    # a system timestamp still is one, by name or by the shape of every value
    assert propose_column_class("orders", "updated_at", ["2026-01-01T10:00:00"]).column_class == "exempt"
    assert propose_column_class("orders", "updated_at", ["2026-01-01T10:00:00"]).confidence == "high"
    seen = propose_column_class("orders", "last_seen_time", ["2026-01-01T10:00:00", "2026-01-02T11:00:00"])
    assert seen.column_class == "exempt"
    assert seen.confidence == "medium"


def test_rows_keyed_by_a_plain_id_still_get_a_table():
    """Support, CRM and ticketing APIs key rows by `id` (D52); dropping them mines nothing at all."""
    traces = [
        one_trace("t1", [{"name": "get_ticket", "args": {"id": "t1"},
                          "result": '{"id": "t1", "subject": "x", "status": "open"}'}])
    ]
    schema = mine_schema(traces)
    assert schema.tables == ["tickets"]
    assert {c.name for c in schema.columns} == {"id", "subject", "status"}
    assert col(schema, "tickets", "status").class_ == "hard"
    assert schema.id_patterns["tickets.id"]


def test_an_id_shape_that_appears_late_is_still_matched_by_the_pattern():
    """canon.py fullmatches ids against these patterns, so a pattern read off the first 200 values
    must not reject a real id that comes later."""
    from harness.builder.mine import id_pattern

    values = [f"u_{i}" for i in range(200)] + ["u-x-9"]
    pattern = id_pattern(values)
    assert all(re.fullmatch(pattern, v) for v in values)
    tight = id_pattern([f"u_{i}" for i in range(200)])
    assert tight == r"^[A-Za-z]+_\d+$"


# --- kind and table naming, against the domains the retail rules missed ------
# docs/cross-domain-check.md: every case below is a tool or a table that airline or telecom got
# wrong and retail never exercised. The retail-shaped cases at the end must keep their old answer.


def test_a_tool_that_returns_what_it_was_sent_is_a_write():
    """Airline's `book_reservation`: no write verb in the name and no field it visibly changed.

    A read answers with the world; a create answers with what you handed it. On all three tau2
    domains `book_reservation` is the only tool where most of the result came back out of the
    arguments, at 0.81 against 0.20 for the highest read.
    """
    booked = {"reservation_id": "R1", "user_id": "u_1", "origin": "LAS", "destination": "DEN",
              "cabin": "economy", "insurance": "yes", "total_baggages": 2}
    traces = [one_trace("t1", [
        {"name": "book_reservation",
         "args": {"user_id": "u_1", "origin": "LAS", "destination": "DEN", "cabin": "economy",
                  "insurance": "yes", "total_baggages": 2},
         "result": json.dumps(booked)},
    ])]
    sig = sig_by_name(mine_tools(traces), "book_reservation")
    assert sig.kind == "write"
    assert sig.classified_by == "observed"
    assert "what the call sent" in sig.kind_reason


def test_a_tool_that_answers_with_a_message_about_a_row_it_was_handed_is_a_write():
    """Airline's `send_certificate` and telecom's `send_payment_request`, which return a sentence.

    Neither matches a write verb, neither returns a row, and neither was ever bracketed by two
    identical reads of the thing it touched, so the two stronger signals say nothing about them.
    """
    traces = [one_trace(f"t{i}", [
        {"name": "send_payment_request", "args": {"customer_id": "C1", "bill_id": f"B{i}"},
         "result": "Payment request sent."},
    ]) for i in range(3)]
    sig = sig_by_name(mine_tools(traces), "send_payment_request")
    assert sig.kind == "write"
    assert sig.classified_by == "observed"


def test_a_message_tool_the_corpus_barely_shows_stays_unclassified():
    """The weakest signal is held to the same floor as the mine gate: flag, never synthesize."""
    traces = [one_trace("t1", [
        {"name": "send_payment_request", "args": {"customer_id": "C1", "bill_id": "B1"},
         "result": "Payment request sent."},
    ])]
    sig = sig_by_name(mine_tools(traces), "send_payment_request")
    assert sig.kind == "read"
    assert sig.unclassified is True


def test_a_message_tool_an_unchanged_read_brackets_is_not_called_a_write():
    """Positive evidence of quiet beats the answer-shape guess."""
    calls = []
    for _ in range(3):
        calls += [
            {"name": "get_line", "args": {"line_id": "L1"}, "result": '{"line_id": "L1", "state": "on"}'},
            {"name": "ping_line", "args": {"line_id": "L1"}, "result": "ok"},
            {"name": "get_line", "args": {"line_id": "L1"}, "result": '{"line_id": "L1", "state": "on"}'},
        ]
    sig = sig_by_name(mine_tools([one_trace("t1", calls)]), "ping_line")
    assert sig.kind == "read"


def test_quiet_evidence_needs_a_read_that_asked_about_what_the_call_named():
    """A reservation names its user in its body, and reading it twice says nothing about the user.

    Reading the answer rather than the question is what kept `send_certificate` mined as a read
    even after the quiet rule went in: every airline trace reads a reservation twice, and every
    reservation body carries the `user_id` the certificate was sent to.
    """
    calls = []
    for _ in range(3):
        body = '{"reservation_id": "R1", "user_id": "u_1", "cabin": "economy"}'
        calls += [
            {"name": "get_reservation_details", "args": {"reservation_id": "R1"}, "result": body},
            {"name": "send_certificate", "args": {"user_id": "u_1", "amount": 100},
             "result": "Certificate certificate_1 added to user u_1 with amount 100."},
            {"name": "get_reservation_details", "args": {"reservation_id": "R1"}, "result": body},
        ]
    sig = sig_by_name(mine_tools([one_trace("t1", calls)]), "send_certificate")
    assert sig.kind == "write"


def test_the_table_is_the_noun_before_the_preposition_not_after_it():
    """Telecom filed every Bill under `customers`: `customer` is a token of the tool name, `bill` is not."""
    row = '{"bill_id": "B1", "customer_id": "C1", "total_due": 10.0, "status": "Draft"}'
    traces = [one_trace("t1", [
        {"name": "get_bills_for_customer", "args": {"customer_id": "C1"}, "result": f"[{row}]"},
    ])]
    schema = mine_schema(traces)
    assert "bills" in schema.tables
    assert "customers" not in schema.tables


def test_an_id_column_the_name_rule_cannot_see_is_found_from_the_calls():
    """Airline's `flights` was never recovered despite 338 calls, because ids there end in `_number`.

    The column is used to address a row and its values are distinct in every result that shows
    several rows. That is what an id does, and no verb list or suffix list is consulted to see it.
    """
    rows = json.dumps([
        {"flight_number": "HAT001", "origin": "LAS", "destination": "DEN", "status": "available"},
        {"flight_number": "HAT002", "origin": "LAS", "destination": "DEN", "status": "available"},
    ])
    traces = [one_trace("t1", [
        {"name": "search_direct_flight", "args": {"origin": "LAS", "destination": "DEN"}, "result": rows},
        {"name": "get_flight_status", "args": {"flight_number": "HAT001"}, "result": '"delayed"'},
    ])]
    schema = mine_schema(traces)
    assert "flights" in schema.tables
    assert schema.id_patterns.get("flights.flight_number")


def test_a_column_shared_by_every_row_of_a_result_is_not_an_id():
    """`origin` is passed as an argument too, and it is the same on every row a search returns.

    Being addressed is not enough on its own, or every filter a search takes would name a table.
    """
    rows = json.dumps([
        {"flight_number": "HAT001", "origin": "LAS", "destination": "DEN"},
        {"flight_number": "HAT002", "origin": "LAS", "destination": "DEN"},
    ])
    traces = [one_trace("t1", [
        {"name": "search_direct_flight", "args": {"origin": "LAS", "destination": "DEN"}, "result": rows},
        {"name": "get_flight_status", "args": {"flight_number": "HAT001"}, "result": '"delayed"'},
    ])]
    from harness.builder.mine import id_columns

    found = id_columns(traces)
    assert "flight_number" in found
    assert "origin" not in found and "destination" not in found


def test_the_retail_shaped_names_keep_their_old_answer():
    """Nothing above may move retail: its verbs, its `_id` columns and its object nouns are unchanged."""
    assert propose_kind("get_order_details").kind == "read"
    assert propose_kind("cancel_pending_order").kind == "write"
    assert propose_kind("calculate").kind == "generic"
    traces = [one_trace("t1", [
        {"name": "get_order_details", "args": {"order_id": "#W1"},
         "result": '{"order_id": "#W1", "user_id": "u_1", "status": "pending"}'},
    ])]
    schema = mine_schema(traces)
    assert schema.tables == ["orders"]
