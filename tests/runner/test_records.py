"""Every record round-trips through JSON, keeps its reserved-word aliases, and hashes by content."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest

from kullback.runner import records as records_module
from kullback.runner.records import (
    ALL_RECORDS,
    Atom,
    Column,
    Cost,
    Environment,
    Event,
    Finding,
    GateResult,
    Intent,
    IntentSpan,
    Probe,
    ProbePool,
    RawPtr,
    Record,
    Refusal,
    RoundRecord,
    Run,
    Task,
    TaskOverlay,
    ToolCall,
    ToolCallError,
    ToolSig,
    Trace,
    Usage,
    Verdict,
    Verifier,
    VerifierHistory,
    VerifierVersion,
    apply_intent,
    as_dict,
    canonical_json,
    content_hash,
)


def _minimal(model: type[Record]) -> Record:
    """Build the record with only its required fields, using placeholder values."""
    values = {}
    for name, field in model.model_fields.items():
        if field.is_required():
            values[field.alias or name] = _placeholder(name, field.annotation)
    return model(**values)


def _placeholder(name: str, annotation):
    if get_origin(annotation) is Literal:
        return get_args(annotation)[0]
    if annotation is bool:
        return True
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if isinstance(annotation, type) and issubclass(annotation, Record):
        # A required field can itself be a record, as `raw_ptr: RawPtr` is on Turn, ToolCall and
        # Trace (D66); it gets the same minimal treatment as the record under test.
        return _minimal(annotation)
    return f"x-{name}"


@pytest.mark.parametrize("model", ALL_RECORDS, ids=lambda m: m.__name__)
def test_every_record_round_trips(model: type[Record]):
    record = _minimal(model)
    payload = json.loads(json.dumps(as_dict(record)))
    again = model.model_validate(payload)
    assert as_dict(again) == payload
    assert content_hash(again) == content_hash(record)


def test_reserved_word_aliases_survive_json():
    verdict = Verdict(
        run_id="r1",
        env_id="e1",
        **{"pass": False, "class": "fail"},
        failing_atom="a3",
        cause="candidate",
    )
    payload = as_dict(verdict)
    assert payload["pass"] is False
    assert payload["class"] == "fail"
    assert "passed" not in payload and "class_" not in payload
    assert Verdict.model_validate(payload).passed is False

    gate = GateResult(stage="ingest", **{"pass": True})
    assert as_dict(gate)["pass"] is True

    error = ToolCallError(**{"class": "not_found_entity"}, payload="no such order", encoding="text")
    assert as_dict(error)["class"] == "not_found_entity"
    assert ToolCallError.model_validate(as_dict(error)).class_ == "not_found_entity"

    column = Column(table="orders", name="status", **{"class": "hard"})
    assert as_dict(column)["class"] == "hard"

    # The field name works too, so module code can write Verdict(passed=..., class_=...).
    assert as_dict(Verdict(run_id="r", passed=True, class_="pass"))["pass"] is True


def test_a_full_run_round_trips_with_nested_records():
    run = Run(
        run_id="run-1",
        env_id="env-1",
        task_id="task-1",
        trace_id="trace-1",
        model="anthropic/claude",
        seed=7,
        events=[
            Event(idx=0, type="user_turn", payload={"content": "cancel my order"}, route=None),
            Event(
                idx=1,
                type="tool_call",
                payload={"name": "cancel_pending_order", "args": {"order_id": "#W1"}},
                route="recording",
                cache_key="k1",
            ),
        ],
        route_counts={"code": 0, "recording": 1, "llm": 0},
        assisted=False,
        end_state_hash="deadbeef",
    )
    again = Run.model_validate(json.loads(json.dumps(as_dict(run))))
    assert again == run
    assert again.events[1].route == "recording"


def test_content_hash_is_stable_and_order_independent():
    a = Task(id="t1", category_id="c1", run_ids=["r1", "r2"], intent="cancel a late order")
    b = Task.model_validate({"category_id": "c1", "id": "t1", "intent": "cancel a late order", "run_ids": ["r1", "r2"]})
    assert content_hash(a) == content_hash(b)
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash(a) != content_hash(a.model_copy(update={"unguarded": True}))
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_defaults_carry_the_decisions_they_encode():
    # D70: a tool with no evidence is read, and flagged unclassified.
    sig = ToolSig(name="get_order_details")
    assert sig.kind == "read" and sig.unclassified is True and sig.kind_confidence == "low"
    # D97: the three sub-versions sit on Environment and are copied onto Verdict.
    env = Environment(env_id="e1", schema_version="s1", tools_version="t1", policy_version="p1")
    verdict = Verdict(
        run_id="r1",
        env_id=env.env_id,
        schema_version=env.schema_version,
        tools_version=env.tools_version,
        policy_version=env.policy_version,
        passed=True,
        class_="pass",
    )
    assert (verdict.schema_version, verdict.tools_version, verdict.policy_version) == ("s1", "t1", "p1")
    # D81: a Task is guarded unless it is marked otherwise.
    assert Task(id="t1").unguarded is False
    # D88: cause is optional and takes only the four values.
    with pytest.raises(ValueError):
        Verdict(run_id="r", passed=False, class_="fail", cause="model_was_lazy")


def test_raw_ptr_and_spans_point_back_at_the_raw_file():
    ptr = RawPtr(file_hash="abc123", sim_index=2, msg_index=17)
    call = ToolCall(id="c1", name="get_user_details", args={"user_id": "u1"}, raw_ptr=ptr)
    atom = Atom(id="a1", kind="required", spans=[ptr], provenance="user_stated")
    verifier = Verifier(task_id="t1", atoms=[atom], verifier_version="v1")
    payload = as_dict(verifier)
    assert payload["atoms"][0]["spans"][0] == {"file_hash": "abc123", "sim_index": 2,
                                               "msg_index": 17, "section": None}
    assert Verifier.model_validate(payload) == verifier
    assert ToolCall.model_validate(as_dict(call)).raw_ptr == ptr
    # A field taken from the export's info block is not inside the transcript, so it cites a
    # section instead of a message (D66).
    info = RawPtr(file_hash="abc123", sim_index=2, section="info.environment_info")
    assert RawPtr.model_validate(as_dict(info)) == info and info.msg_index is None


def test_overlay_rows_given_as_dicts_become_overlay_rows():
    overlay = TaskOverlay(task_id="t1", rows=[{"table": "orders", "id": "#W1", "version_hash": "h1"}])
    assert overlay.rows[0].table == "orders"
    assert TaskOverlay.model_validate(as_dict(overlay)) == overlay


# --- invariants a wrong record must not survive ---


def test_a_verdict_whose_flag_and_class_disagree_is_refused():
    """The flag and the class are two names for one outcome (design section 5)."""
    with pytest.raises(ValueError):
        Verdict(run_id="r", passed=True, class_="fail")
    with pytest.raises(ValueError):
        Verdict(run_id="r", passed=False, class_="pass")
    with pytest.raises(ValueError):
        Verdict(run_id="r", **{"pass": True, "class": "env_error"})
    # A stored file with the same disagreement is refused at load, not quietly accepted.
    with pytest.raises(ValueError):
        Verdict.model_validate({"run_id": "r", "pass": True, "class": "fail"})
    # The four consistent shapes all build.
    assert Verdict(run_id="r", passed=True, class_="pass").passed is True
    for klass in ("fail", "transferred_without_acting", "env_error"):
        assert Verdict(run_id="r", passed=False, class_=klass).class_ == klass


def test_versions_a_verdict_never_copied_are_absent_not_zero():
    """A placeholder '0' passes the regrade gate's presence check, so there is no placeholder:
    a Verdict that never copied its Environment and Runner versions leaves them None (D97)."""
    bare = Verdict(run_id="r", passed=True, class_="pass")
    for field in ("env_id", "schema_version", "tools_version", "policy_version",
                  "verifier_version", "verdict_version", "runner_version"):
        assert getattr(bare, field) is None, field
    filled = Verdict(
        run_id="r", env_id="e1", schema_version="s1", tools_version="t1", policy_version="p1",
        verifier_version="vf1", verdict_version="vd1", runner_version="rv1",
        passed=True, class_="pass",
    )
    assert filled.schema_version == "s1" and filled.runner_version == "rv1"


def test_an_unknown_field_fails_at_load_instead_of_being_dropped():
    """A misspelled or renamed key in a stored file must not vanish on the way in."""
    with pytest.raises(ValueError):
        Verdict.model_validate({"run_id": "r", "pass": True, "class": "pass", "failing_atoms": "a1"})
    with pytest.raises(ValueError):
        Task.model_validate({"id": "t1", "run_id": ["r1"]})
    good = Verdict.model_validate({"run_id": "r", "pass": True, "class": "pass", "failing_atom": None})
    assert good.failing_atom is None


def test_the_run_record_refuses_a_footer_key_that_is_not_a_run_field():
    """The Start and End state live on the stop event now, so Run forbids extras like every record."""
    with pytest.raises(ValueError):
        Run.model_validate({"run_id": "r1", "start_state": {"orders": {}}, "end_state": {"orders": {}}})
    assert Run.model_validate({"run_id": "r1"}).run_id == "r1"


# --- pointers back to the raw file (D66, D67) ---


def test_an_error_carries_its_pointer_and_a_binary_payload_survives():
    ptr = RawPtr(file_hash="abc", sim_index=1, msg_index=2)
    error = ToolCallError(**{"class": "unknown"}, payload=b"\xff\xfe raw", encoding="bytes", raw_ptr=ptr)
    payload = as_dict(error)
    assert payload["raw_ptr"]["file_hash"] == "abc"
    assert payload["encoding"] == "base64"
    import base64

    assert base64.b64decode(payload["payload"]) == b"\xff\xfe raw"
    assert ToolCallError.model_validate(payload) == error
    # Text payloads are left exactly as the customer wrote them.
    text = ToolCallError(**{"class": "not_found_entity"}, payload="Error: no such order")
    assert as_dict(text)["payload"] == "Error: no such order"
    assert as_dict(text)["encoding"] == "text"


def test_a_trace_can_point_at_where_its_prompt_and_tools_came_from():
    ptr = RawPtr(file_hash="abc", sim_index=0)
    trace = Trace(trace_id="t", raw_hash="abc", ingest_version="1", source="tau2",
                  system_prompt="be brief", system_prompt_ptr=ptr,
                  tools_declared=[{"name": "get"}], tools_declared_ptr=ptr, raw_ptr=ptr)
    assert Trace.model_validate(as_dict(trace)) == trace


# --- content addressing and token counts ---


def test_content_hash_of_a_set_ignores_iteration_order_and_refuses_an_unhashable_value():
    """str() followed the interpreter's hash randomization, so the same set hashed differently
    in every run and content addressing broke for any caller that passed one."""
    assert content_hash({"a", "b", "c"}) == content_hash({"c", "b", "a"})
    assert content_hash({"a", "b"}) != content_hash({"a", "c"})
    assert content_hash(("a", "b")) == content_hash(["a", "b"])
    # Anything with no stable form raises rather than hashing on its memory address.
    with pytest.raises(TypeError):
        content_hash({"f": lambda: None})


def test_content_hash_of_a_set_is_the_same_in_a_fresh_interpreter():
    import subprocess
    import sys

    code = "from kullback.runner.records import content_hash; print(content_hash({'b','a','c','d','e'}))"
    digests = set()
    for seed in ("1", "2", "3"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": seed, "PATH": os.environ.get("PATH", ""),
                                  "PYTHONPATH": str(Path(records_module.__file__).parents[3])})
        assert out.returncode == 0, out.stderr
        digests.add(out.stdout.strip())
    assert len(digests) == 1, digests


def test_token_counts_and_costs_are_never_negative():
    """A negative count makes call_cost negative, which would lower the spend ceiling's total."""
    for field in ("input", "output", "cache_read", "cache_write"):
        with pytest.raises(ValueError):
            Usage(**{field: -5})
    with pytest.raises(ValueError):
        Cost(usd=-0.5)
    with pytest.raises(ValueError):
        Cost(wall_ms=-1.0)
    assert Usage(input=0).input == 0


# --- the Examiner's records (phase 5) ---


def _examiner_records() -> list[Record]:
    run = Run(run_id="probe-t1-1", task_id="t1", events=[Event(idx=0, type="stop", payload={"reason": "success"})])
    verifier = Verifier(task_id="t1", atoms=[Atom(id="w0", kind="required", provenance="user_stated",
                                                  target={"tool": "cancel", "entity": "order", "field": "status",
                                                          "value": "cancelled"})])
    probe = Probe(probe_id="probe-t1-1", task_id="t1", bug_class="extra_field_acceptance", verifier_hash="v1",
                  round=1, scored_pass=False, run=run)
    version = VerifierVersion(task_id="t1", content_hash=content_hash(verifier), verifier_version="1", by="derive",
                              accepted=True, verifier=verifier)
    return [probe, ProbePool(task_id="t1", probes=[probe]), version,
            VerifierHistory(task_id="t1", versions=[version]),
            Refusal(task_id="t1", reason="no frontier Run finished", round=1, admitted=True, finished_runs=[]),
            Finding(finding_id="f1", task_id="t1", kind="assisted_tool", text="the tool never fails", run_id="probe-t1-1",
                    tool="cancel", suggested="compile_tool", round=1),
            RoundRecord(round=1, counts={"trusted": 1, "fidelity": 1}, exit="done")]


def test_the_examiner_records_round_trip_through_as_dict_and_hash_by_content():
    for record in _examiner_records():
        payload = json.loads(json.dumps(as_dict(record)))
        again = type(record).model_validate(payload)
        assert again == record, type(record).__name__
        assert content_hash(again) == content_hash(record)
        # The hash reads content, not identity: a copy with a field changed hashes differently.
        first = next(iter(payload))
        changed = type(record).model_validate(dict(payload, **{first: "other" if isinstance(payload[first], str)
                                                             else payload[first]}))
        assert (content_hash(changed) == content_hash(record)) == (payload[first] == as_dict(changed)[first])
    pool = _examiner_records()[1]
    assert pool.probes[0].run.events[0].payload == {"reason": "success"}
    assert {type(r) for r in _examiner_records()} <= set(ALL_RECORDS)


def test_the_intent_record_and_apply_intent_live_in_records_and_the_builder_re_exports_them():
    """The Intent moved into records.py in phase 5 because the Examiner reads it; the Builder keeps
    naming it under its old path, and the names it re-exports are the records' own classes and
    function, so an Intent built on either side is one record."""
    from kullback.builder import intent as builder_intent

    assert builder_intent.Intent is Intent and builder_intent.IntentSpan is IntentSpan
    assert builder_intent.apply_intent is apply_intent
    intent = Intent(task_id="t1", text="cancel the order", grounded=True,
                    spans=[IntentSpan(phrase="the order", trace_id="tr1", source="user_utterance", text="the order")])
    task = Task(id="t1", run_ids=["tr1"])
    applied = apply_intent(task, intent)
    assert applied.intent == "cancel the order" and applied.name == "cancel the order"
    theirs = builder_intent.Intent.model_validate(as_dict(intent))
    assert as_dict(theirs) == as_dict(intent)
    assert as_dict(builder_intent.apply_intent(task, theirs)) == as_dict(applied)
    assert Intent.model_validate(as_dict(theirs)) == intent


def test_a_probe_pool_forbids_unknown_fields():
    with pytest.raises(ValueError):
        ProbePool.model_validate({"task_id": "t1", "probes": [], "closed": True})
    with pytest.raises(ValueError):
        Probe.model_validate({"probe_id": "p", "task_id": "t1", "bug_class": "other", "verifier_hash": "v",
                              "run": {"run_id": "p"}, "passed": False})
    assert ProbePool.model_validate({"task_id": "t1"}).probes == []
