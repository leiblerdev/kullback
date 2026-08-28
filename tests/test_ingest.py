"""Tests for builder/ingest.py: raw store, tau2 derivation, grader stripping, error and truncation marking, the ingest gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.builder import ingest
from harness.shared.provider import TestModel
from harness.shared.records import Trace

# --- helpers ---------------------------------------------------------------


def tau2_file(simulations, tasks=None, info=None) -> dict:
    """A minimal file in the tau2 native shape."""
    return {
        "timestamp": "2025-06-05T14:00:00",
        "info": info if info is not None else {"environment_info": {"domain_name": "retail"}},
        "tasks": tasks or [],
        "simulations": simulations,
    }


def assistant_msg(idx, tool_calls=None, content=None, ts=None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "turn_idx": idx,
        "timestamp": ts,
        "cost": 0.0,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def tool_msg(idx, call_id, content, error=False, ts=None) -> dict:
    return {
        "role": "tool",
        "id": call_id,
        "content": content,
        "requestor": "assistant",
        "error": error,
        "turn_idx": idx,
        "timestamp": ts,
    }


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


@pytest.fixture
def small_file(workdir, tmp_path):
    """One simulation, one successful call, one failing call, one truncated call."""
    calls = [{"id": "c1", "name": "get_user", "arguments": {"uid": "u1"}, "requestor": "assistant"}]
    sim = {
        "id": "sim-1",
        "task_id": "7",
        "trial": 0,
        "seed": 1,
        "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0, "action_checks": [{"a": 1}], "nl_assertions": ["x"]},
        "messages": [
            assistant_msg(0, content="hello", ts="2025-06-05T14:00:00"),
            {"role": "user", "content": "hi", "tool_calls": None, "turn_idx": 1},
            assistant_msg(2, tool_calls=calls, ts="2025-06-05T14:00:00"),
            tool_msg(3, "c1", '{"uid": "u1", "name": "Mei"}', ts="2025-06-05T14:00:00.250000"),
            assistant_msg(
                4,
                tool_calls=[{"id": "c2", "name": "cancel_order", "arguments": {"oid": "o1"}, "requestor": "assistant"}],
            ),
            tool_msg(5, "c2", "Error: Non-pending order cannot be cancelled", error=True),
            assistant_msg(
                6,
                tool_calls=[{"id": "c3", "name": "list_orders", "arguments": {}, "requestor": "assistant"}],
            ),
            tool_msg(7, "c3", "order o1, order o2, order o3..."),
        ],
    }
    task = {"id": "7", "description": {}, "evaluation_criteria": {"actions": [{"action_id": "7_0"}]}}
    return write_json(tmp_path / "small.json", tau2_file([sim], [task]))


# --- store_raw -------------------------------------------------------------


def test_store_raw_copies_byte_for_byte(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    stored = Path(raw.path)
    assert stored.read_bytes() == small_file.read_bytes()
    assert stored.parent == workdir / "raw"
    assert stored.name == raw.raw_hash + ".json"
    assert raw.bytes == len(small_file.read_bytes())
    assert raw.format_detected == "tau2_native"


def test_store_raw_is_content_addressed_and_idempotent(small_file, workdir, tmp_path):
    first = ingest.store_raw(small_file, workdir)
    copy = write_json(tmp_path / "copy.json", json.loads(small_file.read_text()))
    second = ingest.store_raw(copy, workdir)
    assert first.raw_hash == second.raw_hash
    assert len(list((workdir / "raw").iterdir())) == 1


def test_store_raw_labels_an_unknown_format_and_the_rejection_happens_at_derive(workdir, tmp_path):
    """store_raw keeps every byte whatever the format; it is derive_traces that refuses, with a reason."""
    path = write_json(tmp_path / "odd.json", {"hello": "world"})
    raw = ingest.store_raw(path, workdir)
    assert raw.format_detected == "unknown"
    assert Path(raw.path).read_bytes() == path.read_bytes()
    with pytest.raises(ValueError) as caught:
        ingest.derive_traces(raw.raw_hash, workdir)
    assert raw.raw_hash in str(caught.value)
    assert not (workdir / "traces").exists()


# --- format_detect ---------------------------------------------------------


def test_format_detect_tau2_and_otel():
    assert ingest.format_detect({"simulations": [{"messages": []}]}) == "tau2_native"
    otel = [{"name": "gen_ai.client.inference.operation.details", "attributes": {"gen_ai.system": "anthropic"}}]
    assert ingest.format_detect(otel) == "otel_genai"
    assert ingest.format_detect({"resourceSpans": []}) == "otel_genai"
    assert ingest.format_detect({"nothing": 1}) == "unknown"


def test_otel_mapper_is_a_stub(workdir, tmp_path):
    otel = [{"name": "gen_ai.client.inference.operation.details", "attributes": {"gen_ai.system": "anthropic"}}]
    raw = ingest.store_raw(write_json(tmp_path / "otel.json", otel), workdir)
    assert raw.format_detected == "otel_genai"
    with pytest.raises(NotImplementedError, match="OpenTelemetry"):
        ingest.derive_traces(raw.raw_hash, workdir)


def test_derive_traces_rejects_unknown_format(workdir, tmp_path):
    raw = ingest.store_raw(write_json(tmp_path / "odd.json", {"hello": "world"}), workdir)
    with pytest.raises(ValueError, match="unknown"):
        ingest.derive_traces(raw.raw_hash, workdir)


# --- derive_traces on the small file ---------------------------------------


def test_derive_traces_shape(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert len(traces) == 1
    trace = traces[0]
    assert isinstance(trace, Trace)
    assert trace.trace_id == "sim-1"
    assert trace.raw_hash == raw.raw_hash
    assert trace.source == "tau2_native"
    assert trace.ingest_version == ingest.INGEST_VERSION
    assert len(trace.turns) == 8
    assert [c.name for c in trace.tool_calls] == ["get_user", "cancel_order", "list_orders"]


def test_every_field_carries_a_raw_pointer(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    assert trace.raw_ptr is not None and trace.raw_ptr.file_hash == raw.raw_hash
    assert trace.raw_ptr.sim_index == 0
    for turn in trace.turns:
        assert turn.raw_ptr is not None
        assert turn.raw_ptr.file_hash == raw.raw_hash
        assert turn.raw_ptr.msg_index == turn.idx
    for call in trace.tool_calls:
        assert call.raw_ptr is not None
        assert call.raw_ptr.file_hash == raw.raw_hash
        assert call.raw_ptr.msg_index is not None


def test_results_and_latency_are_attached(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    first = trace.tool_calls[0]
    assert first.result == {"uid": "u1", "name": "Mei"}
    assert first.error is None
    assert first.latency_ms == pytest.approx(250.0)
    assert first.args == {"uid": "u1"}
    assert first.requestor == "assistant"


def test_error_call_keeps_verbatim_payload_and_class(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    failing = trace.tool_calls[1]
    assert failing.error is not None
    assert failing.error.class_ == "business_error"
    assert failing.error.payload == "Error: Non-pending order cannot be cancelled"
    assert failing.error.encoding == "text"
    assert failing.error.classified_by == "rule"
    assert failing.result is None


def test_truncated_result_is_marked(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    cut = trace.tool_calls[2]
    assert cut.truncated is True
    assert cut.visible_len == len("order o1, order o2, order o3...")
    assert cut.cut_marker == "..."


def test_untruncated_result_is_not_marked(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    assert trace.tool_calls[0].truncated is False
    assert trace.tool_calls[0].visible_len is None


# --- grader stripping (D66) ------------------------------------------------


def test_grader_fields_go_to_the_sidecar_and_not_into_the_trace(small_file, workdir):
    """The negative half scans what the rest of the pipeline actually reads: the files under traces/."""
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    ingest.write_traces([trace], workdir)
    written = [p.read_text(encoding="utf-8") for p in (workdir / "traces").glob("*.json")]
    assert written
    for word in ("reward_info", "action_checks", "nl_assertions", "evaluation_criteria", "reward"):
        assert word not in json.dumps(trace.model_dump(mode="json", by_alias=True))
        assert all(word not in blob for blob in written)
    sidecar = json.loads(ingest.grader_file(trace, workdir).read_text(encoding="utf-8"))
    assert sidecar["trace_id"] == "sim-1"
    assert sidecar["raw_hash"] == raw.raw_hash
    assert sidecar["fields"]["reward_info"]["reward"] == 1.0
    assert sidecar["fields"]["task_id"] == "7"
    assert sidecar["fields"]["trial"] == 0
    assert sidecar["fields"]["evaluation_criteria"]["actions"][0]["action_id"] == "7_0"


def test_grader_sidecar_written_even_when_the_source_has_no_grader_fields(workdir, tmp_path):
    sim = {"id": "plain", "messages": [assistant_msg(0, content="hi")]}
    raw = ingest.store_raw(write_json(tmp_path / "plain.json", tau2_file([sim])), workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    sidecar = json.loads(ingest.grader_file(trace, workdir).read_text(encoding="utf-8"))
    assert sidecar["fields"] == {}


# --- hashes stable ---------------------------------------------------------


def test_hashes_stable_across_two_runs(small_file, workdir, tmp_path):
    raw = ingest.store_raw(small_file, workdir)
    first = ingest.derive_traces(raw.raw_hash, workdir)
    other = tmp_path / "work2"
    other.mkdir()
    again_raw = ingest.store_raw(small_file, other)
    second = ingest.derive_traces(again_raw.raw_hash, other)
    assert again_raw.raw_hash == raw.raw_hash
    assert [t.hash for t in first] == [t.hash for t in second]
    assert first[0].hash != ""


def test_trace_hash_covers_the_content(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    changed = trace.model_copy(deep=True)
    changed.tool_calls[0].args = {"uid": "u2"}
    assert ingest.trace_hash(changed) != trace.hash
    assert ingest.trace_hash(trace) == trace.hash


# --- error classification rules --------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Error: User not found", "not_found_entity"),
        ("Error: Order not found", "not_found_entity"),
        ("Error: Non-delivered order cannot be exchanged", "business_error"),
        ("Error: Payment method should be the original payment method", "business_error"),
        ("Error: Insufficient gift card balance to pay for the price difference", "business_error"),
        ("Unknown tool: frobnicate", "tool_not_found"),
        ("Tool 'x' not found", "tool_not_found"),
        ("Invalid arguments: zip must be a string", "invalid_arguments"),
        ("get_user() missing 1 required positional argument: 'uid'", "invalid_arguments"),
        ("Permission denied for user u1", "permission_denied"),
        ("403 Forbidden", "permission_denied"),
        ("Rate limit exceeded, retry later", "transient"),
        ("Request timed out", "transient"),
        ("The request was cancelled by the caller", "cancelled"),
        ("wobble", "unknown"),
    ],
)
def test_classify_error_rules(text, expected):
    err = ingest.classify_error(text)
    assert err.class_ == expected
    assert err.payload == text
    assert err.classified_by == "rule"


def test_classify_error_keeps_json_encoding():
    err = ingest.classify_error('{"code": "not_found", "message": "no such order"}')
    assert err.encoding == "json"
    assert err.class_ == "not_found_entity"


def test_classify_error_uses_a_structured_code_when_the_source_has_one():
    err = ingest.classify_error("boom", structured={"code": "permission_denied"})
    assert err.class_ == "permission_denied"
    assert err.classified_by == "code"


# --- truncation rules ------------------------------------------------------


@pytest.mark.parametrize(
    "text, marker",
    [
        ("a long list of things...", "..."),
        ("a long list…", "…"),
        ("rows here [truncated]", "[truncated]"),
        ("rows here ... (truncated)", "... (truncated)"),
        ("rows here <truncated>", "<truncated>"),
        ("rows here [output truncated]", "[output truncated]"),
    ],
)
def test_detect_truncation_markers(text, marker):
    truncated, visible_len, found = ingest.detect_truncation(text)
    assert truncated is True
    assert found == marker
    assert visible_len == len(text)


def test_detect_truncation_leaves_clean_text_alone():
    assert ingest.detect_truncation('{"ok": true}') == (False, None, None)
    assert ingest.detect_truncation(None) == (False, None, None)


# --- the ingest gate -------------------------------------------------------


def test_gate_passes_on_the_small_file(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is True
    assert gate.stage == "ingest"
    assert gate.metrics["traces"] == 1
    assert gate.metrics["tool_calls"] == 3
    assert gate.metrics["errors"] == 1
    assert gate.metrics["truncated"] == 1
    assert gate.failures == []


def test_gate_fails_when_a_call_has_no_result_and_no_error(workdir, tmp_path):
    calls = [{"id": "c9", "name": "get_user", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "dangling", "messages": [assistant_msg(0, tool_calls=calls)]}
    raw = ingest.store_raw(write_json(tmp_path / "dangling.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is False
    assert gate.metrics["unresolved"] == 1
    assert any("c9" in failure for failure in gate.failures)


def test_gate_fails_when_a_grader_sidecar_is_missing(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    ingest.grader_file(traces[0], workdir).unlink()
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is False
    assert any("grader" in failure for failure in gate.failures)


# --- whole-file entry point ------------------------------------------------


def test_ingest_file_writes_traces_and_prints_counts(small_file, workdir, capsys):
    summary = ingest.ingest_file(small_file, workdir)
    printed = capsys.readouterr().out.strip()
    assert printed == (
        "ingest tau2_native: 1 runs, 3 tool calls, 1 errors, 1 truncated, 0 rejected, gate pass"
    )
    assert summary["runs"] == 1
    assert summary["tool_calls"] == 3
    assert summary["gate"]["pass"] is True
    written = json.loads((workdir / "traces" / (summary["trace_hashes"][0] + ".json")).read_text(encoding="utf-8"))
    assert written["trace_id"] == "sim-1"
    assert written["hash"] == summary["trace_hashes"][0]


def test_ingest_file_is_repeatable(small_file, workdir):
    first = ingest.ingest_file(small_file, workdir)
    second = ingest.ingest_file(small_file, workdir)
    assert first["trace_hashes"] == second["trace_hashes"]
    assert first["raw_hash"] == second["raw_hash"]


# --- the real tau2 fixture -------------------------------------------------


def test_tau2_fixture_ingests(tau2_small_path, tau2_small, workdir, capsys):
    summary = ingest.ingest_file(tau2_small_path, workdir)
    capsys.readouterr()
    assert summary["runs"] == len(tau2_small["simulations"]) == 3
    assert summary["tool_calls"] > 0
    assert summary["gate"]["pass"] is True


def test_tau2_fixture_strips_grader_fields(tau2_small_path, tau2_small, workdir):
    raw = ingest.store_raw(tau2_small_path, workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    for trace in traces:
        blob = json.dumps(trace.model_dump(mode="json", by_alias=True))
        for word in ("reward_info", "action_checks", "nl_assertions", "evaluation_criteria"):
            assert word not in blob
        sidecar = ingest.grader_file(trace, workdir)
        assert sidecar.is_file()
        fields = json.loads(sidecar.read_text(encoding="utf-8"))["fields"]
        assert "reward_info" in fields and "task_id" in fields


def test_tau2_fixture_carries_policy_and_pairs_every_call(tau2_small_path, workdir):
    raw = ingest.store_raw(tau2_small_path, workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert all(t.system_prompt and "retail agent policy" in t.system_prompt.lower() for t in traces)
    for trace in traces:
        for call in trace.tool_calls:
            assert call.result is not None or call.error is not None


def test_tau2_fixture_hashes_stable_across_two_runs(tau2_small_path, workdir, tmp_path):
    other = tmp_path / "work2"
    other.mkdir()
    first = ingest.ingest_file(tau2_small_path, workdir)
    second = ingest.ingest_file(tau2_small_path, other)
    assert first["trace_hashes"] == second["trace_hashes"]


# --- one raw file per trace, files named by content (ingest-1, ingest-13) ---


def test_two_files_reusing_one_simulation_id_do_not_overwrite_each_other(workdir, tmp_path):
    """A second customer file that reuses a simulation id must not silently take over the first one's trace."""
    def one(reward, note):
        sim = {"id": "sim-1", "task_id": "7", "reward_info": {"reward": reward},
               "messages": [assistant_msg(0, content=note)]}
        return write_json(tmp_path / f"file-{reward}.json", tau2_file([sim]))

    first = ingest.ingest_file(one(1.0, "file A"), workdir)
    second = ingest.ingest_file(one(0.0, "file B"), workdir)

    assert first["raw_hash"] != second["raw_hash"]
    assert first["trace_hashes"] != second["trace_hashes"]
    assert len(list((workdir / "traces").glob("*.json"))) == 2
    assert len(list((workdir / "grader").glob("*.json"))) == 2
    for summary, reward in ((first, 1.0), (second, 0.0)):
        sidecar = json.loads((workdir / "grader" / (summary["trace_hashes"][0] + ".json")).read_text(encoding="utf-8"))
        assert sidecar["raw_hash"] == summary["raw_hash"]
        assert sidecar["fields"]["reward_info"]["reward"] == reward


def test_gate_fails_when_a_sidecar_belongs_to_another_raw_file(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    sidecar = ingest.grader_file(trace, workdir)
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    body["raw_hash"] = "0" * 64
    sidecar.write_text(json.dumps(body), encoding="utf-8")
    gate = ingest.gate_ingest([trace], workdir)
    assert gate.passed is False
    assert any("belongs to another trace" in failure for failure in gate.failures)


def test_a_simulation_id_with_path_separators_cannot_escape_the_folders(workdir, tmp_path):
    sim = {"id": "../escaped", "messages": [assistant_msg(0, content="hi")]}
    raw = ingest.store_raw(write_json(tmp_path / "escape.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    written = ingest.write_traces(traces, workdir)
    assert [p.parent for p in written] == [workdir / "traces"]
    assert ingest.grader_file(traces[0], workdir).parent == workdir / "grader"
    assert not (workdir / "escaped.json").exists()
    assert traces[0].trace_id == "../escaped"  # the customer's id is kept, it is only not used as a file name


# --- results that do not parse (ingest-2) ----------------------------------


CUT_JSON = '{"orders": [{"order_id": "#W1", "status": "pending"}, {"order_id": "#W2", "sta'


def test_a_json_result_cut_off_without_a_marker_is_marked_and_fails_the_gate(workdir, tmp_path):
    calls = [{"id": "c1", "name": "list_orders", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "cut", "messages": [assistant_msg(0, tool_calls=calls), tool_msg(1, "c1", CUT_JSON)]}
    raw = ingest.store_raw(write_json(tmp_path / "cut.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    call = traces[0].tool_calls[0]
    assert call.truncated is True
    assert call.cut_marker == ingest.UNPARSED_JSON_MARKER
    assert call.result == CUT_JSON  # kept verbatim, never a half-parsed object
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is False
    assert gate.metrics["unparseable"] == 1
    assert any("does not parse" in failure for failure in gate.failures)


def test_a_json_result_cut_off_with_a_marker_also_fails_the_gate(workdir, tmp_path):
    calls = [{"id": "c1", "name": "list_orders", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "cutm", "messages": [assistant_msg(0, tool_calls=calls),
                                      tool_msg(1, "c1", CUT_JSON + "... (truncated)")]}
    raw = ingest.store_raw(write_json(tmp_path / "cutm.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert traces[0].tool_calls[0].cut_marker == "... (truncated)"
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is False
    assert gate.metrics["unparseable"] == 1


def test_a_whole_json_result_is_neither_truncated_nor_unparseable(small_file, workdir):
    raw = ingest.store_raw(small_file, workdir)
    trace = ingest.derive_traces(raw.raw_hash, workdir)[0]
    assert ingest.unparsed_json(trace.tool_calls[0].result) is False
    assert ingest.gate_ingest([trace], workdir).metrics["unparseable"] == 0


# --- an empty simulations list (ingest-3) ----------------------------------


def test_an_empty_simulations_list_yields_no_phantom_run(workdir, tmp_path):
    raw = ingest.store_raw(write_json(tmp_path / "empty.json", tau2_file([])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert traces == []
    gate = ingest.gate_ingest(traces, workdir, raw_hash=raw.raw_hash)
    assert gate.passed is False
    assert gate.metrics["rejected"] == 1
    assert any("empty simulations list" in failure for failure in gate.failures)


# --- error rule precedence (ingest-4) --------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # a number that happens to be 500 is not an HTTP status
        ("Error: gift card balance 500 is insufficient for this order", "business_error"),
        ("Error: refund of 1200 exceeds the 500 limit for this policy", "business_error"),
        # "tool" somewhere in the sentence does not make a missing entity a missing tool
        ("Error: Order not found. Use the lookup tool to search by email", "not_found_entity"),
        # a business refusal phrased with "invalid" is still a business refusal
        ("Error: Invalid order status: non-pending order cannot be cancelled", "business_error"),
        # the real HTTP shapes still class as transient
        ("HTTP 503 Service Unavailable", "transient"),
        ("502 Bad Gateway", "transient"),
        ("status code: 500", "transient"),
        # and the tool rule still fires when the tool itself is the thing missing
        ("Unknown tool: frobnicate", "tool_not_found"),
        ("Tool 'x' not found", "tool_not_found"),
    ],
)
def test_classify_error_precedence(text, expected):
    assert ingest.classify_error(text).class_ == expected


# --- the ingest version is the code hash (ingest-6) ------------------------


def test_ingest_version_is_the_hash_of_the_ingest_source(small_file, workdir):
    """D66 keys re-derivation on (raw hash, ingest code hash), so a changed mapper must change the version."""
    import hashlib

    source = Path(ingest.__file__).read_bytes()
    assert ingest.INGEST_VERSION == hashlib.sha256(source).hexdigest()[:16]
    assert ingest.INGEST_VERSION != "1"
    trace = ingest.derive_traces(ingest.store_raw(small_file, workdir).raw_hash, workdir)[0]
    assert trace.ingest_version == ingest.INGEST_VERSION
    assert ingest.trace_hash(trace.model_copy(update={"ingest_version": "other"})) != trace.hash


# --- the classification sources: code, rule, llm (ingest-7) ----------------


def test_a_structured_error_body_is_classified_by_code_through_derive(workdir, tmp_path):
    calls = [{"id": "c1", "name": "get_user", "arguments": {}, "requestor": "assistant"}]
    body = json.dumps({"code": "permission_denied", "message": "boom"})
    sim = {"id": "structured", "messages": [assistant_msg(0, tool_calls=calls),
                                            tool_msg(1, "c1", body, error=True)]}
    raw = ingest.store_raw(write_json(tmp_path / "structured.json", tau2_file([sim])), workdir)
    error = ingest.derive_traces(raw.raw_hash, workdir)[0].tool_calls[0].error
    assert error.class_ == "permission_denied"
    assert error.classified_by == "code"
    assert error.encoding == "json"
    assert error.payload == body


def test_the_llm_second_pass_only_sees_what_the_rules_left_unknown(workdir, tmp_path):
    calls = [{"id": "c1", "name": "a", "arguments": {}, "requestor": "assistant"},
             {"id": "c2", "name": "b", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "llm", "messages": [
        assistant_msg(0, tool_calls=calls),
        tool_msg(1, "c1", "Error: wobble", error=True),
        tool_msg(2, "c2", "Error: User not found", error=True),
    ]}
    raw = ingest.store_raw(write_json(tmp_path / "llm.json", tau2_file([sim])), workdir)
    model = TestModel(['{"class": "transient", "reason": "the backend was flapping"}'])

    without = ingest.derive_traces(raw.raw_hash, workdir)
    assert [c.error.class_ for c in without[0].tool_calls] == ["unknown", "not_found_entity"]
    assert [c.error.classified_by for c in without[0].tool_calls] == ["rule", "rule"]

    traces = ingest.derive_traces(raw.raw_hash, workdir, model=model)
    first, second = traces[0].tool_calls
    assert (first.error.class_, first.error.classified_by) == ("transient", "llm")
    assert (second.error.class_, second.error.classified_by) == ("not_found_entity", "rule")
    assert len(model.calls) == 1  # the rules settled the second one, so it never reached the model
    assert "wobble" in json.dumps(model.calls[0]["messages"])
    assert traces[0].hash == ingest.trace_hash(traces[0])


def test_an_unusable_llm_reply_leaves_the_rule_class_standing(workdir, tmp_path):
    calls = [{"id": "c1", "name": "a", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "llm2", "messages": [assistant_msg(0, tool_calls=calls),
                                      tool_msg(1, "c1", "Error: wobble", error=True)]}
    raw = ingest.store_raw(write_json(tmp_path / "llm2.json", tau2_file([sim])), workdir)
    for reply in ("no json here", '{"class": "not_a_class"}', '{"class": "unknown"}'):
        error = ingest.derive_traces(raw.raw_hash, workdir, model=TestModel([reply]))[0].tool_calls[0].error
        assert (error.class_, error.classified_by) == ("unknown", "rule")


# --- formats we have not mapped yet (ingest-8) -----------------------------


def test_claude_code_jsonl_is_labelled_and_says_it_is_not_mapped(workdir, tmp_path):
    path = tmp_path / "cc.jsonl"
    path.write_text(
        '{"type":"user","message":{"role":"user","content":"hi"}}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1"}]}}\n',
        encoding="utf-8",
    )
    raw = ingest.store_raw(path, workdir)
    assert raw.format_detected == "claude_code_jsonl"
    with pytest.raises(NotImplementedError, match="claude_code_jsonl"):
        ingest.derive_traces(raw.raw_hash, workdir)


def test_a_json_array_of_records_is_not_read_as_jsonl():
    assert ingest.format_detect([{"type": "user", "message": {}}]) == "unknown"
    assert ingest.format_detect([{"type": "user", "message": {}}], jsonl=True) == "claude_code_jsonl"


# --- a tool result nobody asked for (ingest-9) -----------------------------


def test_a_tool_result_with_no_matching_call_fails_the_gate(workdir, tmp_path):
    sim = {"id": "orphan", "messages": [assistant_msg(0), tool_msg(1, "ghost", "{}")]}
    raw = ingest.store_raw(write_json(tmp_path / "orphan.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert traces[0].tool_calls == []
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is False
    assert gate.metrics["orphan_results"] == 1
    assert any("ghost" in failure and "answers no recorded call" in failure for failure in gate.failures)


# --- a tool that answered with null (ingest-10) ----------------------------


def test_a_tool_that_answered_null_counts_as_resolved(workdir, tmp_path):
    calls = [{"id": "c1", "name": "get_user", "arguments": {}, "requestor": "assistant"}]
    sim = {"id": "nullish", "messages": [assistant_msg(0, tool_calls=calls), tool_msg(1, "c1", None)]}
    raw = ingest.store_raw(write_json(tmp_path / "nullish.json", tau2_file([sim])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    call = traces[0].tool_calls[0]
    assert call.result is None and call.error is None
    gate = ingest.gate_ingest(traces, workdir)
    assert gate.passed is True
    assert gate.metrics["unresolved"] == 0


# --- a byte order mark (ingest-11) -----------------------------------------


def test_a_file_saved_with_a_utf8_bom_still_ingests(workdir, tmp_path):
    sim = {"id": "bom", "messages": [assistant_msg(0, content="hi")]}
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(tau2_file([sim])).encode("utf-8"))
    raw = ingest.store_raw(path, workdir)
    assert raw.format_detected == "tau2_native"
    assert Path(raw.path).read_bytes() == path.read_bytes()  # the stored bytes keep the BOM
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert [t.trace_id for t in traces] == ["bom"]


# --- one broken simulation does not cost the file (ingest-12) --------------


def test_a_message_with_an_unknown_role_rejects_that_trace_with_a_reason(workdir, tmp_path):
    good = {"id": "good", "messages": [assistant_msg(0, content="hi")]}
    bad = {"id": "bad", "messages": [{"role": "function", "content": "x", "tool_calls": None}]}
    raw = ingest.store_raw(write_json(tmp_path / "roles.json", tau2_file([good, bad])), workdir)
    traces = ingest.derive_traces(raw.raw_hash, workdir)
    assert [t.trace_id for t in traces] == ["good"]
    rejects = ingest.read_rejects(workdir, raw.raw_hash)
    assert [r["trace_id"] for r in rejects] == ["bad"]
    assert rejects[0]["sim_index"] == 1 and "refuse" in rejects[0]["reason"]
    gate = ingest.gate_ingest(traces, workdir, raw_hash=raw.raw_hash)
    assert gate.passed is False
    assert gate.metrics["rejected"] == 1
    assert any("bad" in failure for failure in gate.failures)


def test_rejects_do_not_linger_when_the_file_is_ingested_again(workdir, tmp_path):
    bad = {"id": "bad", "messages": [{"role": "function", "content": "x"}]}
    broken = write_json(tmp_path / "broken.json", tau2_file([bad]))
    raw = ingest.store_raw(broken, workdir)
    ingest.derive_traces(raw.raw_hash, workdir)
    assert ingest.read_rejects(workdir, raw.raw_hash)

    fixed = {"id": "bad", "messages": [assistant_msg(0, content="x")]}
    other = ingest.store_raw(write_json(tmp_path / "fixed.json", tau2_file([fixed])), workdir)
    ingest.derive_traces(other.raw_hash, workdir)
    assert ingest.read_rejects(workdir, other.raw_hash) == []
