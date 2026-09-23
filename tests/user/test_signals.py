from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn
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

    empty = mine_signal(_trace("e", [], []))
    assert (empty.system_turns, empty.user_turns, empty.assistant_turns, empty.tool_turns) == (
        0, 0, 0, 0)
    assert empty.last_speaker is None
    assert empty.repeated_user_utterances == 0
    assert empty.transfer_call is False
    assert (empty.unanswered_calls, empty.truncated_calls) == (0, 0)
    assert empty.end_kind == "unknown"
    assert empty.ending_actor == "unknown"
    assert empty.why_left is None
    assert empty.stated_frustration is None
    assert empty.dropped == ()
    assert empty.contradictions == 0


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


@pytest.mark.parametrize("as_model", [False, True], ids=["dict_labels", "model_labels"])
def test_valid_why_and_frustration_stored(as_model):
    """Spans are kept at their own character offsets, from dicts or ModelLabel instances."""
    content = "sält: leaving now, leaving now, this is hopeless Ω"
    trace = _trace("t", [_turn(0, "assistant", "hi"), _turn(1, "user", content)])
    second = content.index("leaving now", content.index("leaving now") + 1)
    why = {
        "kind": "why_left",
        "span": {"turn_idx": 1, "start": second, "end": second + len("leaving now"),
                 "quote": "leaving now"},
    }
    labels = [why, _label("stated_frustration", 1, content, "hopeless Ω")]
    if as_model:
        labels = [ModelLabel.model_validate(label) for label in labels]
    sig = mine_signal(
        trace,
        end={"end_kind": "walked_away", "ending_actor": "user"},
        labels=labels,
    )
    assert sig.why_left.quote == "leaving now"
    assert sig.why_left.turn_idx == 1
    assert sig.why_left.source.file_hash == "bb"
    assert sig.why_left.start == second
    assert sig.why_left.end == second + len("leaving now")
    assert content[sig.why_left.start:sig.why_left.end] == "leaving now"
    assert sig.why_left.start != content.index("leaving now")
    assert sig.stated_frustration.quote == "hopeless Ω"
    assert sig.dropped == ()
    assert sig.contradictions == 0


def _drop_inputs(case):
    if case == "unknown_turn":
        trace = _trace("t", [_turn(0, "user", "hi")])
        labels = [
            {"kind": "why_left", "span": {"turn_idx": 9, "start": 0, "end": 1, "quote": "h"}}
        ]
        return trace, labels, [("why_left", "unknown_turn")]
    if case == "bad_bounds":
        trace = _trace("t", [_turn(0, "user", "hi")])
        labels = [
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 99, "quote": "hi"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 1, "end": 1, "quote": "h"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": -1, "end": 1, "quote": "h"}},
        ]
        return trace, labels, [("why_left", "bad_bounds")] * 3
    if case == "nonverbatim_quote":
        trace = _trace("t", [_turn(0, "user", "leaving now")])
        labels = [
            {
                "kind": "why_left",
                "span": {"turn_idx": 0, "start": 0, "end": 7, "quote": "Leaving"},
            }
        ]
        return trace, labels, [("why_left", "nonverbatim_quote")]
    if case == "non_user_turn":
        trace = _trace(
            "t",
            [_turn(0, "assistant", "done now"), _turn(1, "tool", "done now"), _turn(2, "user", "hi")],
        )
        labels = [
            _label("why_left", 0, "done now", "done now"),
            _label("stated_frustration", 1, "done now", "done now"),
        ]
        return trace, labels, [("why_left", "non_user_turn"), ("stated_frustration", "non_user_turn")]
    if case == "byte_offsets":
        trace = _trace("t", [_turn(0, "user", "sält Ω ok")])
        labels = [{"kind": "why_left", "span": {"turn_idx": 0, "start": 6, "end": 7, "quote": "Ω"}}]
        return trace, labels, [("why_left", "nonverbatim_quote")]
    if case == "bool_offsets":
        trace = _trace("t", [_turn(0, "user", "xab")])
        labels = [
            {"kind": "why_left", "span": {"turn_idx": 0, "start": True, "end": 3, "quote": "ab"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": True, "quote": "x"}},
        ]
        return trace, labels, [("why_left", "bad_bounds")] * 2
    if case == "strict_offset_types":
        trace = _trace("t", [_turn(0, "user", "xab")])
        labels = [
            {"kind": "why_left", "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"}},
            {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 3.0, "quote": "xab"}},
            {"kind": "stated_frustration",
             "span": {"turn_idx": "0", "start": 0, "end": 3, "quote": "xab"}},
            {"kind": "why_left", "span": {"turn_idx": True, "start": 0, "end": 1, "quote": "x"}},
        ]
        return trace, labels, [
            ("why_left", "bad_bounds"),
            ("why_left", "bad_bounds"),
            ("stated_frustration", "unknown_turn"),
            ("why_left", "unknown_turn"),
        ]
    trace = _trace("t", [_turn(0, "user", "a   b")])
    labels = [
        {"kind": "why_left", "span": {"turn_idx": 0, "start": 0, "end": 0, "quote": ""}},
        _label("stated_frustration", 0, "a   b", "   "),
    ]
    return trace, labels, [("why_left", "bad_bounds"), ("stated_frustration", "empty_quote")]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("unknown_turn", id="unknown_turn"),
        pytest.param("bad_bounds", id="bad_bounds"),
        pytest.param("nonverbatim_quote", id="nonverbatim_quote"),
        pytest.param("non_user_turn", id="non_user_turn"),
        pytest.param("empty_quote", id="empty_quote"),
        pytest.param("byte_offsets", id="byte_offsets"),
        pytest.param("bool_offsets", id="bool_offsets"),
        pytest.param("strict_offset_types", id="strict_offset_types"),
    ],
)
def test_unusable_spans_dropped(case):
    """Each unusable span is dropped with its reason, and the caller's labels stay as given."""
    trace, labels, want = _drop_inputs(case)
    given = repr(copy.deepcopy(labels))
    sig = mine_signal(trace, labels=labels)
    assert sig.why_left is None
    assert sig.stated_frustration is None
    assert [(d.kind, d.reason) for d in sig.dropped] == want
    assert sig.contradictions == 0
    assert repr(labels) == given


def _malformed_shape(case):
    if case.startswith("construct_"):
        span = {
            "construct_str_start": {"turn_idx": 0, "start": "1", "end": 2, "quote": "hi"},
            "construct_float_end": {"turn_idx": 0, "start": 0, "end": 2.0, "quote": "hi"},
            "construct_bool_start": {"turn_idx": 0, "start": True, "end": 2, "quote": "hi"},
            "construct_float_idx": {"turn_idx": 1.0, "start": 0, "end": 2, "quote": "hi"},
        }[case]
        return lambda: ModelLabel(kind="why_left", span=span)
    trace, kwargs = _mined(case)
    return lambda: mine_signal(trace, **kwargs)


def _mined(case):
    hi = _trace("t", [_turn(0, "user", "hi")])
    if case == "override":
        override = dict(_label("why_left", 0, "hi", "hi"))
        override["last_speaker"] = "user"
        return hi, {"labels": [override]}
    if case == "bad_kind":
        bad = {"kind": "mood", "span": {"turn_idx": 0, "start": 0, "end": 2, "quote": "hi"}}
        return hi, {"labels": [bad]}
    if case == "bad_end_extra":
        return hi, {"end": {"end_kind": "x", "ending_actor": "user", "trace_id": "t"}}
    if case == "bad_end_actor":
        return hi, {"end": {"end_kind": "x", "ending_actor": "robot"}}
    xab = _trace("t", [_turn(0, "user", "xab")])
    if case == "mixed_override":
        mixed = {
            "kind": "why_left",
            "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"},
            "last_speaker": "user",
        }
        return xab, {"labels": [mixed]}
    bad_kind = {
        "kind": "mood",
        "span": {"turn_idx": 0, "start": "1", "end": 3, "quote": "ab"},
    }
    return xab, {"labels": [bad_kind]}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("override", id="override"),
        pytest.param("bad_kind", id="bad_kind"),
        pytest.param("bad_end_extra", id="bad_end_extra"),
        pytest.param("bad_end_actor", id="bad_end_actor"),
        pytest.param("mixed_override", id="mixed_override"),
        pytest.param("mixed_bad_kind", id="mixed_bad_kind"),
        pytest.param("construct_str_start", id="construct_str_start"),
        pytest.param("construct_float_end", id="construct_float_end"),
        pytest.param("construct_bool_start", id="construct_bool_start"),
        pytest.param("construct_float_idx", id="construct_float_idx"),
    ],
)
def test_malformed_label_and_end_shapes_rejected(case):
    """The miner and direct ModelLabel construction both refuse nonstrict shapes."""
    with pytest.raises(ValidationError):
        _malformed_shape(case)()


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
    frustrated = "this is hopeless, fine"
    sig_frustrated = mine_signal(
        _trace("f", [_turn(0, "user", frustrated), _turn(1, "assistant", "closed")]),
        end={"end_kind": "agent_closed", "ending_actor": "assistant"},
        labels=[_label("stated_frustration", 0, frustrated, "hopeless")],
    )
    assert sig_frustrated.stated_frustration.quote == "hopeless"
    assert sig_frustrated.contradictions == 0


def _adjudication_case(case):
    content = "I am leaving now"
    first = _label("why_left", 0, content, "leaving now")
    second = _label("why_left", 0, content, "leaving")
    if case == "tuple_bad_first":
        bad = {
            "kind": "why_left",
            "span": {"turn_idx": 0, "start": 0, "end": 99, "quote": content},
        }
        return (bad, first), None, "leaving now", [("why_left", "bad_bounds")]
    if case == "reversed_backward":
        return [second, first], None, "leaving", [("why_left", "duplicate")]
    end = None
    if case == "list_duplicates_with_end":
        end = {"end_kind": "walked_away", "ending_actor": "user"}
    return [first, second], end, "leaving now", [("why_left", "duplicate")]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("tuple_bad_first", id="tuple_bad_first"),
        pytest.param("list_duplicates", id="list_duplicates"),
        pytest.param("list_duplicates_with_end", id="list_duplicates_with_end"),
        pytest.param("reversed_backward", id="reversed_backward"),
    ],
)
def test_first_valid_label_wins(case):
    trace = _trace("t", [_turn(0, "user", "I am leaving now")])
    labels, end, want_quote, want_dropped = _adjudication_case(case)
    if end is None:
        sig = mine_signal(trace, labels=labels)
    else:
        sig = mine_signal(trace, end=end, labels=labels)
    assert sig.why_left.quote == want_quote
    assert [(d.kind, d.reason) for d in sig.dropped] == want_dropped


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


def _container_case(case):
    content = "I am leaving now"
    trace = _trace("t", [_turn(0, "user", content)])
    good = ModelLabel.model_validate(_label("why_left", 0, content, "leaving now"))
    if case == "empty_set":
        return trace, set()
    if case == "set_of_good":
        return trace, {good}
    if case == "frozenset_of_good":
        return trace, frozenset([good])
    if case == "dict_labels":
        return trace, {"why_left": _label("why_left", 0, content, "leaving now")}
    if case == "dict_from_label":
        return trace, dict(_label("why_left", 0, content, "leaving now"))
    if case == "poison_set":
        poison = _trace(
            "t",
            [_turn(0, "user", "a"), _turn(1, "user", "first"), _turn(1, "assistant", "second")],
        )
        return poison, {good}
    if case == "str_labels":
        return trace, "why_left"
    if case == "bytes_labels":
        return trace, b"why_left"
    return trace, bytearray(b"why_left")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("empty_set", id="empty_set"),
        pytest.param("set_of_good", id="set_of_good"),
        pytest.param("frozenset_of_good", id="frozenset_of_good"),
        pytest.param("dict_labels", id="dict_labels"),
        pytest.param("dict_from_label", id="dict_from_label"),
        pytest.param("poison_set", id="poison_set"),
        pytest.param("str_labels", id="str_labels"),
        pytest.param("bytes_labels", id="bytes_labels"),
        pytest.param("bytearray_labels", id="bytearray_labels"),
    ],
)
def test_labels_container_types_rejected_before_adjudication(case):
    trace, labels = _container_case(case)
    with pytest.raises(TypeError):
        mine_signal(trace, labels=labels)
