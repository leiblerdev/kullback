from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kullback.runner.records import RawPtr, Task, ToolCall, Trace, Turn, UserRules
from kullback.user.population import RecordingStanding, build_population, pick_user
from kullback.user.signals import EndEvidence, ModelLabel, mine_signal


def _ptr():
    return RawPtr(file_hash="bb", sim_index=2)


def _turn(idx, role, content):
    return Turn(idx=idx, role=role, content=content, tool_call_ids=[], raw_ptr=_ptr())


def _call(name="lookup", answered=True, truncated=False):
    if answered:
        return ToolCall(
            name=name,
            args={},
            result="ok",
            has_result=True,
            resolved=True,
            truncated=truncated,
            raw_ptr=_ptr(),
        )
    return ToolCall(
        name=name,
        args={},
        result=None,
        has_result=None,
        resolved=False,
        truncated=truncated,
        raw_ptr=_ptr(),
    )


def _trace(tid, turns, calls=()):
    return Trace(
        trace_id=tid,
        raw_hash="h",
        ingest_version="v1",
        source="s",
        turns=list(turns),
        tool_calls=list(calls),
        raw_ptr=_ptr(),
    )


def _label(kind, idx, content, quote):
    start = content.index(quote)
    return {
        "kind": kind,
        "span": {"turn_idx": idx, "start": start, "end": start + len(quote), "quote": quote},
    }


def test_exact_counts_and_last_speaker():
    trace = _trace(
        "t",
        [
            _turn(0, "system", "sys"),
            _turn(1, "user", "hello"),
            _turn(2, "assistant", "hi"),
            _turn(3, "user", "bye"),
            _turn(4, "tool", "out"),
        ],
    )
    sig = mine_signal(trace)
    assert (sig.system_turns, sig.user_turns, sig.assistant_turns, sig.tool_turns) == (1, 2, 1, 1)
    assert sig.last_speaker == "tool"
    assert sig.trace_id == "t"


def test_empty_trace_signal():
    sig = mine_signal(_trace("e", [], []))
    assert (sig.system_turns, sig.user_turns, sig.assistant_turns, sig.tool_turns) == (0, 0, 0, 0)
    assert sig.last_speaker is None
    assert sig.repeated_user_utterances == 0
    assert sig.transfer_call is False
    assert (sig.unanswered_calls, sig.truncated_calls) == (0, 0)
    assert sig.end_kind == "unknown"
    assert sig.ending_actor == "unknown"
    assert sig.why_left is None
    assert sig.stated_frustration is None
    assert sig.dropped == ()
    assert sig.contradictions == 0


def test_end_evidence_stored():
    trace = _trace("t", [_turn(0, "user", "hi")])
    sig = mine_signal(trace, end={"end_kind": "walked_away", "ending_actor": "user"})
    assert sig.end_kind == "walked_away"
    assert sig.ending_actor == "user"
    sig2 = mine_signal(trace, end=EndEvidence(end_kind="agent_closed", ending_actor="assistant"))
    assert sig2.end_kind == "agent_closed"
    assert sig2.ending_actor == "assistant"


def test_duplicate_turn_indices_reject_evidence():
    trace = _trace(
        "t",
        [_turn(0, "user", "a"), _turn(1, "user", "first"), _turn(1, "assistant", "second")],
    )
    sig = mine_signal(trace, labels=[_label("why_left", 1, "first", "first")])
    assert sig.why_left is None
    assert [(d.kind, d.reason) for d in sig.dropped] == [("why_left", "ambiguous_turn")]
    assert (sig.user_turns, sig.assistant_turns) == (2, 1)


def test_transfer_tool_set_not_heuristic():
    trace = _trace("t", [_turn(0, "user", "hi")], [_call(name="transfer_room")])
    assert mine_signal(trace, transfer_tools={"transfer_call"}).transfer_call is False
    assert mine_signal(trace, transfer_tools={"transfer_room"}).transfer_call is True
    assert mine_signal(trace, transfer_tools=set()).transfer_call is False
    assert mine_signal(trace, transfer_tools={"Transfer_Room"}).transfer_call is False


def test_repeated_exact_asks():
    trace = _trace(
        "t",
        [
            _turn(0, "user", "a"),
            _turn(1, "assistant", "a"),
            _turn(2, "user", "b"),
            _turn(3, "user", "a"),
            _turn(4, "assistant", "a"),
            _turn(5, "user", "B "),
            _turn(6, "user", "a"),
            _turn(7, "user", None),
        ],
    )
    assert mine_signal(trace).repeated_user_utterances == 2


def test_incomplete_recordings_produce_signals():
    trace = _trace(
        "t",
        [_turn(0, "user", "hi"), _turn(1, "assistant", "hello")],
        [_call(name="a"), _call(name="b", answered=False), _call(name="c", truncated=True)],
    )
    sig = mine_signal(trace)
    assert (sig.unanswered_calls, sig.truncated_calls) == (1, 1)
    assert (sig.user_turns, sig.assistant_turns) == (1, 1)
    assert sig.last_speaker == "assistant"


def test_valid_why_and_frustration_stored():
    content = "I am leaving now, this is hopeless"
    trace = _trace("t", [_turn(0, "assistant", "hi"), _turn(1, "user", content)])
    sig = mine_signal(
        trace,
        end={"end_kind": "walked_away", "ending_actor": "user"},
        labels=[
            _label("why_left", 1, content, "leaving now"),
            _label("stated_frustration", 1, content, "hopeless"),
        ],
    )
    assert sig.why_left.quote == "leaving now"
    assert sig.why_left.turn_idx == 1
    assert sig.why_left.source.file_hash == "bb"
    assert sig.stated_frustration.quote == "hopeless"
    assert sig.dropped == ()
    assert sig.contradictions == 0


def test_unknown_turn_dropped():
    trace = _trace("t", [_turn(0, "user", "hi")])
    sig = mine_signal(
        trace, labels=[{"kind": "why_left", "span": {"turn_idx": 9, "start": 0, "end": 1, "quote": "h"}}]
    )
    assert sig.why_left is None
    assert [(d.kind, d.reason) for d in sig.dropped] == [("why_left", "unknown_turn")]


def test_bad_bounds_dropped():
    trace = _trace("t", [_turn(0, "user", "hi")])
    sig = mine_signal(
        trace,
        labels=[
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 99, "quote": "hi"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 1, "end": 1, "quote": "h"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": -1, "end": 1, "quote": "h"}},
        ],
    )
    assert sig.why_left is None
    assert [d.reason for d in sig.dropped] == ["bad_bounds", "bad_bounds", "bad_bounds"]


def test_bool_offsets_dropped():
    trace = _trace("t", [_turn(0, "user", "xab")])
    first = {"kind": "why_left", "span": {"turn_idx": 0, "start": True, "end": 3, "quote": "ab"}}
    second = {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": True, "quote": "x"}}
    sig = mine_signal(trace, labels=[first, second])
    assert sig.why_left is None
    assert [d.reason for d in sig.dropped] == ["bad_bounds", "bad_bounds"]
    assert sig.contradictions == 0
    assert first["span"]["start"] is True
    assert second["span"]["end"] is True


def test_char_offsets_accepted_byte_offsets_dropped():
    content = "sält Ω ok"
    trace = _trace("t", [_turn(0, "user", content)])
    sig = mine_signal(trace, labels=[_label("why_left", 0, content, "Ω")])
    assert sig.why_left.quote == "Ω"
    byte_style = {"kind": "why_left", "span": {"turn_idx": 0, "start": 6, "end": 7, "quote": "Ω"}}
    sig2 = mine_signal(trace, labels=[byte_style])
    assert sig2.why_left is None
    assert [(d.kind, d.reason) for d in sig2.dropped] == [("why_left", "nonverbatim_quote")]


def test_nonverbatim_quote_dropped():
    trace = _trace("t", [_turn(0, "user", "leaving now")])
    sig = mine_signal(
        trace,
        labels=[{"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 7, "quote": "Leaving"}}],
    )
    assert sig.why_left is None
    assert [(d.kind, d.reason) for d in sig.dropped] == [("why_left", "nonverbatim_quote")]


def test_assistant_and_tool_spans_dropped():
    trace = _trace(
        "t",
        [_turn(0, "assistant", "done now"), _turn(1, "tool", "done now"), _turn(2, "user", "hi")],
    )
    sig = mine_signal(
        trace,
        labels=[
            _label("why_left", 0, "done now", "done now"),
            _label("stated_frustration", 1, "done now", "done now"),
        ],
    )
    assert sig.why_left is None
    assert sig.stated_frustration is None
    assert [d.reason for d in sig.dropped] == ["non_user_turn", "non_user_turn"]


def test_empty_and_whitespace_quotes_dropped():
    trace = _trace("t", [_turn(0, "user", "a   b")])
    sig = mine_signal(
        trace,
        labels=[
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 0, "quote": ""}},
            _label("stated_frustration", 0, "a   b", "   "),
        ],
    )
    assert sig.why_left is None
    assert sig.stated_frustration is None
    assert [d.reason for d in sig.dropped] == ["bad_bounds", "empty_quote"]


def test_structural_override_and_unsupported_rejected():
    trace = _trace("t", [_turn(0, "user", "hi")])
    override = dict(_label("why_left", 0, "hi", "hi"))
    override["last_speaker"] = "user"
    with pytest.raises(ValidationError):
        mine_signal(trace, labels=[override])
    with pytest.raises(ValidationError):
        mine_signal(
            trace,
            labels=[{"kind": "mood", "span": {"turn_idx": 0, "start": 0, "end": 2, "quote": "hi"}}],
        )
    with pytest.raises(ValidationError):
        mine_signal(trace, end={"end_kind": "x", "ending_actor": "user", "trace_id": "t"})
    with pytest.raises(ValidationError):
        mine_signal(trace, end={"end_kind": "x", "ending_actor": "robot"})


def test_contradiction_precedence_and_counter():
    content = "I am leaving now"
    assistant_trace = _trace("a", [_turn(0, "user", content), _turn(1, "assistant", "bye")])
    sig = mine_signal(
        assistant_trace,
        end={"end_kind": "agent_closed", "ending_actor": "assistant"},
        labels=[_label("why_left", 0, content, "leaving now")],
    )
    assert sig.why_left is None
    assert [(d.kind, d.reason) for d in sig.dropped] == [("why_left", "contradiction")]
    assert sig.contradictions == 1
    assert (sig.user_turns, sig.assistant_turns) == (1, 1)
    assert sig.ending_actor == "assistant"
    env_trace = _trace("e", [_turn(0, "user", content)])
    sig_env = mine_signal(
        env_trace,
        end={"end_kind": "expired", "ending_actor": "environment"},
        labels=[_label("why_left", 0, content, "leaving now")],
    )
    assert sig_env.why_left is None
    assert sig_env.contradictions == 1
    unknown_trace = _trace("u", [_turn(0, "user", content)])
    sig_unknown = mine_signal(unknown_trace, labels=[_label("why_left", 0, content, "leaving now")])
    assert sig_unknown.why_left.quote == "leaving now"
    assert sig_unknown.end_kind == "unknown"
    assert sig_unknown.contradictions == 0


def test_frustration_coexists_with_assistant_end():
    content = "this is hopeless, fine"
    trace = _trace("t", [_turn(0, "user", content), _turn(1, "assistant", "closed")])
    sig = mine_signal(
        trace,
        end={"end_kind": "agent_closed", "ending_actor": "assistant"},
        labels=[_label("stated_frustration", 0, content, "hopeless")],
    )
    assert sig.stated_frustration.quote == "hopeless"
    assert sig.contradictions == 0


def test_duplicate_label_dropped():
    content = "I am leaving now"
    trace = _trace("t", [_turn(0, "user", content)])
    sig = mine_signal(
        trace,
        end={"end_kind": "walked_away", "ending_actor": "user"},
        labels=[
            _label("why_left", 0, content, "leaving now"),
            _label("why_left", 0, content, "leaving"),
        ],
    )
    assert sig.why_left.quote == "leaving now"
    assert [(d.kind, d.reason) for d in sig.dropped] == [("why_left", "duplicate")]


def test_detached_and_deterministic():
    content = "I am leaving now"
    trace = _trace("t", [_turn(0, "user", content)])
    labels = [_label("why_left", 0, content, "leaving now")]
    first = mine_signal(trace, labels=labels)
    assert mine_signal(trace, labels=labels) == first
    trace.turns[0].content = "mutated text here"
    trace.turns[0].raw_ptr.file_hash = "zz"
    assert first.why_left.quote == "leaving now"
    assert first.why_left.source.file_hash == "bb"
    assert labels == [_label("why_left", 0, content, "leaving now")]


def test_model_label_instance_accepted():
    content = "I am leaving now"
    trace = _trace("t", [_turn(0, "user", content)])
    label = ModelLabel.model_validate(_label("why_left", 0, content, "leaving now"))
    sig = mine_signal(
        trace, end={"end_kind": "walked_away", "ending_actor": "user"}, labels=[label]
    )
    assert sig.why_left.quote == "leaving now"


def test_signals_never_change_population_selection():
    args = {"city": "X"}
    first_call = ToolCall(
        name="set_addr", args=dict(args), result="ok", has_result=True, resolved=True, raw_ptr=_ptr()
    )
    second_call = ToolCall(
        name="set_addr", args=dict(args), result="ok", has_result=True, resolved=True, raw_ptr=_ptr()
    )
    traces = {
        "r1": _trace("r1", [_turn(0, "user", "please set it")], [first_call]),
        "r2": _trace("r2", [_turn(0, "user", "please set it")], [second_call]),
    }
    task = Task(id="t9", run_ids=["r1", "r2"])
    standings = {
        "r1": RecordingStanding(task_eligible=True, complete=True, replay_confirmed=True),
        "r2": RecordingStanding(task_eligible=True, complete=True, replay_confirmed=True),
    }
    rules = {"r1": UserRules(), "r2": UserRules()}
    pop = build_population(
        task,
        traces,
        reference_id="r1",
        standings=standings,
        rules=rules,
        held_out_ids=set(),
        write_tools=["set_addr"],
    )
    before = pick_user(pop, seed=1, salt="sig-salt")
    for trace in traces.values():
        mine_signal(
            trace,
            end={"end_kind": "walked_away", "ending_actor": "user"},
            labels=[_label("why_left", 0, "please set it", "set it")],
        )
    after = pick_user(pop, seed=1, salt="sig-salt")
    assert before == after
    assert pop.support_count == 2


def test_strict_offset_types_dropped():
    trace = _trace("t", [_turn(0, "user", "xab")])
    str_start = {"kind": "why_left", "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"}}
    float_end = {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 3.0, "quote": "xab"}}
    str_idx = {
        "kind": "stated_frustration",
        "span": {"turn_idx": "0", "start": 0, "end": 3, "quote": "xab"},
    }
    bool_idx = {"kind": "why_left", "span": {"turn_idx": True, "start": 0, "end": 1, "quote": "x"}}
    sig = mine_signal(trace, labels=[str_start, float_end, str_idx, bool_idx])
    assert sig.why_left is None
    assert sig.stated_frustration is None
    assert [(d.kind, d.reason) for d in sig.dropped] == [
        ("why_left", "bad_bounds"),
        ("why_left", "bad_bounds"),
        ("stated_frustration", "unknown_turn"),
        ("why_left", "unknown_turn"),
    ]
    assert sig.contradictions == 0
    assert str_start["span"]["start"] == "1"
    assert float_end["span"]["end"] == 3.0
    assert str_idx["span"]["turn_idx"] == "0"


def test_direct_construction_rejects_nonstrict_offsets():
    with pytest.raises(ValidationError):
        ModelLabel(
            kind="why_left", span={"turn_idx": 0, "start": "1", "end": 2, "quote": "hi"}
        )
    with pytest.raises(ValidationError):
        ModelLabel(
            kind="why_left", span={"turn_idx": 0, "start": 0, "end": 2.0, "quote": "hi"}
        )
    with pytest.raises(ValidationError):
        ModelLabel(
            kind="why_left", span={"turn_idx": 0, "start": True, "end": 2, "quote": "hi"}
        )
    with pytest.raises(ValidationError):
        ModelLabel(
            kind="why_left", span={"turn_idx": 1.0, "start": 0, "end": 2, "quote": "hi"}
        )


def test_mixed_override_and_bad_offset_rejects():
    trace = _trace("t", [_turn(0, "user", "xab")])
    mixed = {
        "kind": "why_left",
        "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"},
        "last_speaker": "user",
    }
    with pytest.raises(ValidationError):
        mine_signal(trace, labels=[mixed])
    bad_kind = {
        "kind": "mood",
        "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"},
    }
    with pytest.raises(ValidationError):
        mine_signal(trace, labels=[bad_kind])


def test_accepted_label_retains_offsets():
    content = "set it, set it"
    trace = _trace("t", [_turn(0, "user", content)])
    second = content.index("set it", 1)
    label = {
        "kind": "why_left",
        "span": {"turn_idx": 0, "start": second, "end": second + len("set it"), "quote": "set it"},
    }
    sig = mine_signal(trace, labels=[label])
    assert sig.why_left.quote == "set it"
    assert sig.why_left.start == second
    assert sig.why_left.end == second + len("set it")
    assert content[sig.why_left.start:sig.why_left.end] == "set it"
    assert (sig.why_left.start, sig.why_left.end) != (0, len("set it"))


def test_signal_module_boundary():
    text = Path("kullback/user/signals.py").read_text(encoding="utf-8")
    assert "kullback.builder" not in text
    assert "kullback.examiner" not in text
    assert "kullback.rounds" not in text
    assert "verdict" not in text
    assert "kullback.gates" not in text
    assert "end_of_run" not in text
