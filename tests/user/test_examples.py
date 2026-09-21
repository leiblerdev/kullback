from __future__ import annotations

from collections import UserDict
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from kullback.runner.records import RawPtr, Trace, Turn
from kullback.user import examples
from kullback.user.examples import (
    ExampleCandidate,
    ExampleSpan,
    ExampleWithheld,
    assemble_candidates,
    trace_content_hash,
)


def _ptr():
    return RawPtr(file_hash="cc", sim_index=3)


def _turn(idx, role, content):
    return Turn(idx=idx, role=role, content=content, tool_call_ids=[], raw_ptr=_ptr())


def _trace(tid, turns):
    return Trace(
        trace_id=tid,
        raw_hash="h",
        ingest_version="v1",
        source="s",
        turns=list(turns),
        tool_calls=[],
        raw_ptr=_ptr(),
    )


def _span(trace, situation, idx, content, quote):
    start = content.index(quote)
    return ExampleSpan(
        recording_id=trace.trace_id,
        situation=situation,
        turn_idx=idx,
        start=start,
        end=start + len(quote),
        trace_hash=trace_content_hash(trace),
    )


def _raw_span(trace, situation, idx, content, quote):
    start = content.index(quote)
    return {
        "recording_id": trace.trace_id,
        "situation": situation,
        "turn_idx": idx,
        "start": start,
        "end": start + len(quote),
        "trace_hash": trace_content_hash(trace),
    }


def _span_at(trace, situation, idx, start, end):
    return ExampleSpan(
        recording_id=trace.trace_id,
        situation=situation,
        turn_idx=idx,
        start=start,
        end=end,
        trace_hash=trace_content_hash(trace),
    )


def _run(traces, spans, held=(), known=None):
    return assemble_candidates(
        traces, list(spans), held_out_ids=set(held), known_values_by_id=dict(known or {})
    )


def _reasons(dropped):
    return [(w.recording_id, w.reason) for w in dropped]


def test_all_five_situations_supplied():
    situations = ["unknown_fact", "company_only", "agent_refuses", "confirmation", "giving_up"]
    traces = {}
    spans = []
    known = {}
    for pos, situation in enumerate(situations):
        rid = "r" + str(pos)
        content = "excerpt for " + rid
        traces[rid] = _trace(rid, [_turn(0, "assistant", content)])
        spans.append(_span(traces[rid], situation, 0, content, content))
        known[rid] = []
    cands, dropped = _run(traces, spans, known=known)
    assert dropped == ()
    assert len(cands) == 5
    assert sorted(c.situation for c in cands) == sorted(situations)
    for cand in cands:
        assert cand.masked_excerpt == "excerpt for " + cand.recording_id


@pytest.mark.parametrize(
    "offsets",
    [
        pytest.param({"turn_idx": True, "start": 0, "end": 1}, id="bool_idx"),
        pytest.param({"turn_idx": 1.0, "start": 0, "end": 1}, id="float_idx"),
        pytest.param({"turn_idx": "0", "start": 0, "end": 1}, id="str_idx"),
        pytest.param({"turn_idx": 0, "start": "0", "end": 1}, id="str_start"),
        pytest.param({"turn_idx": 0, "start": 0, "end": 1.5}, id="float_end"),
    ],
)
def test_strict_offsets_reject_bool_float_string(offsets):
    with pytest.raises(ValidationError):
        ExampleSpan(recording_id="r", situation="confirmation", trace_hash="0" * 64, **offsets)


def test_raw_bool_float_string_offsets_withheld():
    content = "hello"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": []}
    digest = trace_content_hash(traces["r"])
    bad_idx = [
        {"recording_id": "r", "situation": "confirmation", "turn_idx": True, "start": 0, "end": 1, "trace_hash": digest},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 1.0, "start": 0, "end": 1, "trace_hash": digest},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": "0", "start": 0, "end": 1, "trace_hash": digest},
    ]
    cands, dropped = _run(traces, bad_idx, known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "unknown_turn")]
    bad_bounds = [
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": "0", "end": 1, "trace_hash": digest},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1.5, "trace_hash": digest},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": False, "end": 1, "trace_hash": digest},
    ]
    cands, dropped = _run(traces, bad_bounds, known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "bad_bounds")]


def test_malformed_spans_withheld():
    content = "hello"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": []}
    raws = [
        {"recording_id": "r", "situation": "nope", "turn_idx": 0, "start": 0, "end": 1},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "extra": 1},
        {"situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "task_id": "t"},
        42,
    ]
    cands, dropped = _run(traces, raws, known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "malformed_span"), ("unknown", "malformed_span")]


class _Watch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touched = []

    def __contains__(self, key):
        self.touched.append(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self.touched.append(key)
        return super().get(key, default)


class _Boom:
    def __iter__(self):
        raise AssertionError("held inventory read")

    def __len__(self):
        raise AssertionError("held inventory read")


def test_held_out_poison_never_read():
    ok_content = "ok 111 here"
    held_content = "held 999 here"
    traces = _Watch(
        {
            "ok": _trace("ok", [_turn(0, "user", ok_content)]),
            "held": _trace("held", [_turn(0, "user", held_content)]),
        }
    )
    known = _Watch({"ok": ["111"], "held": _Boom()})
    spans = [
        _span(traces["ok"], "confirmation", 0, ok_content, ok_content),
        _span(traces["held"], "confirmation", 0, held_content, held_content),
    ]
    cands, dropped = assemble_candidates(
        traces, spans, held_out_ids={"held"}, known_values_by_id=known
    )
    assert [c.masked_excerpt for c in cands] == ["ok <MASKED> here"]
    assert _reasons(dropped) == [("held", "held_out")]
    assert "held" not in traces.touched
    assert "held" not in known.touched
    assert "ok" in traces.touched
    assert "ok" in known.touched


def _prerequisite_case(case):
    content = "hi there"
    if case == "missing_trace":
        phantom = _trace("ghost", [_turn(0, "user", "hi")])
        span = _span(phantom, "confirmation", 0, "hi", "hi")
        return {}, [span], {}, [("ghost", "missing_trace")]
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    if case == "missing_inventory":
        span = _span(traces["r"], "confirmation", 0, content, content)
        return traces, [span], {}, [("r", "unknown_mask_inventory")]
    other = {"r": _trace("other", [_turn(0, "user", content)])}
    span = ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=len(content),
        trace_hash=trace_content_hash(other["r"]),
    )
    return other, [span], {"r": []}, [("r", "trace_mismatch")]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("missing_trace", id="missing_trace"),
        pytest.param("missing_inventory", id="missing_inventory"),
        pytest.param("trace_mismatch", id="trace_mismatch"),
    ],
)
def test_missing_prerequisites_withheld(case):
    traces, spans, known, want = _prerequisite_case(case)
    cands, dropped = _run(traces, spans, known=known)
    assert cands == ()
    assert _reasons(dropped) == want


def test_unknown_and_ambiguous_turn():
    traces = {"r": _trace("r", [_turn(0, "user", "hello")])}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 9, "hello", "hell")], known={"r": []})
    assert cands == ()
    assert _reasons(dropped) == [("r", "unknown_turn")]
    dup = {"r": _trace("r", [_turn(1, "user", "first"), _turn(1, "assistant", "second")])}
    cands, dropped = _run(dup, [_span(dup["r"], "confirmation", 1, "first", "first")], known={"r": []})
    assert cands == ()
    assert _reasons(dropped) == [("r", "ambiguous_turn")]


def test_bad_bounds_and_empty():
    content = "hello"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": []}
    bounds = [
        _span_at(traces["r"], "confirmation", 0, 2, 2),
        _span_at(traces["r"], "confirmation", 0, 3, 2),
        _span_at(traces["r"], "confirmation", 0, -1, 2),
        _span_at(traces["r"], "confirmation", 0, 0, 99),
    ]
    cands, dropped = _run(traces, bounds, known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "bad_bounds")]
    none_trace = {"r": _trace("r", [_turn(0, "user", None)])}
    ghost = _span_at(none_trace["r"], "confirmation", 0, 0, 1)
    cands, dropped = _run(none_trace, [ghost], known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "bad_bounds")]


def test_full_and_user_spoken_masking():
    content = "call me at 555-0100"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": ["555-0100"]}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["call me at <MASKED>"]
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, content, "555-0100")], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED>"]


def test_short_single_char_and_repeated_masking():
    content = "banana banana"
    traces = {"r": _trace("r", [_turn(3, "assistant", content)])}
    known = {"r": ["a"]}
    cands, dropped = _run(traces, [_span(traces["r"], "company_only", 3, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["b<MASKED>n<MASKED>n<MASKED> b<MASKED>n<MASKED>n<MASKED>"]


def test_overlapping_longest_first_no_leftover():
    content = "xxabcdefxx"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    known = {"r": ["cde", "abcdef"]}
    cands, dropped = _run(traces, [_span(traces["r"], "unknown_fact", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["xx<MASKED>xx"]
    assert cands[0].masked_excerpt.count("<MASKED>") == 1


def test_substring_overmask_short_common():
    content = "hello"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": ["he"]}
    cands, dropped = _run(traces, [_span(traces["r"], "giving_up", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED>llo"]


def test_adjacent_values():
    content = "abcd"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    known = {"r": ["ab", "cd"]}
    cands, dropped = _run(traces, [_span(traces["r"], "agent_refuses", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED><MASKED>"]


def test_unicode_and_metachar_literal_masking():
    uni = "says 日本語 ok"
    traces = {"r": _trace("r", [_turn(0, "user", uni)])}
    known = {"r": ["日本語"]}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, uni, uni)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["says <MASKED> ok"]
    meta = "key a.c*(x)[y]\\z end"
    traces = {"r": _trace("r", [_turn(1, "assistant", meta)])}
    known = {"r": ["a.c*(x)[y]\\z"]}
    cands, dropped = _run(traces, [_span(traces["r"], "company_only", 1, meta, meta)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["key <MASKED> end"]


def test_case_sensitive_matching():
    content = "secret Secret SECRET"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": ["Secret"]}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["secret <MASKED> SECRET"]


def test_value_outside_excerpt_ignored():
    content = "aaa BBB aaa"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    known = {"r": ["BBB"]}
    tail = _span_at(traces["r"], "confirmation", 0, 8, 11)
    cands, dropped = _run(traces, [tail], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["aaa"]


@pytest.mark.parametrize(
    ("start", "end", "known_values"),
    [
        pytest.param(8, 14, ["WORLD"], id="start"),
        pytest.param(0, 8, ["WORLD"], id="end"),
        pytest.param(1, 3, ["hello WORLD bye"], id="covering"),
    ],
)
def test_partial_boundary_withheld(start, end, known_values):
    content = "hello WORLD bye"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    cut = _span_at(traces["r"], "confirmation", 0, start, end)
    cands, dropped = _run(traces, [cut], known={"r": known_values})
    assert cands == ()
    assert _reasons(dropped) == [("r", "partial_boundary")]


def test_token_collision_withheld():
    content = "see MASK here"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": ["MASK"]}
    )
    assert cands == ()
    assert _reasons(dropped) == [("r", "token_collision")]
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": ["<MASKED>"]}
    )
    assert cands == ()
    assert _reasons(dropped) == [("r", "token_collision")]
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": [">"]}
    )
    assert cands == ()
    assert _reasons(dropped) == [("r", "token_collision")]


def test_empty_inventory_passes_through():
    content = "plain text 123"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    cands, dropped = _run(traces, [_span(traces["r"], "unknown_fact", 0, content, content)], known={"r": []})
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == [content]


def test_bad_inventory_withheld_without_raw():
    content = "has zxq-secret-9 inside"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    span = _span(traces["r"], "confirmation", 0, content, content)
    for bad in (["ok", 7], ["ok", ""], "ok-string", None, {"k": "v"}):
        cands, dropped = _run(traces, [span], known={"r": bad})
        assert cands == ()
        assert _reasons(dropped) == [("r", "bad_inventory")]
        for w in dropped:
            assert "zxq-secret-9" not in w.reason
            assert "ok" not in w.reason


def test_no_raw_values_anywhere():
    secret = "zxq-secret-9"
    content = "wrap " + secret + " twice " + secret
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": [secret]}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, content, content)], known=known)
    assert dropped == ()
    assert secret not in cands[0].masked_excerpt
    assert secret not in repr(cands[0])
    assert set(ExampleCandidate.model_fields) == {
        "recording_id",
        "situation",
        "masked_excerpt",
        "turn_idx",
        "start",
        "end",
        "trace_hash",
        "source",
    }
    phantom = _trace("ghost", [_turn(0, "user", "x")])
    cands, dropped = _run(
        traces, [_span(phantom, "confirmation", 0, "x", "x")], known={"ghost": [secret]}
    )
    assert cands == ()
    for w in dropped:
        assert secret not in w.reason


def test_only_own_inventory_used():
    traces = {
        "r1": _trace("r1", [_turn(0, "user", "alpha beta")]),
        "r2": _trace("r2", [_turn(0, "user", "beta gamma")]),
    }
    known = {"r1": ["alpha"], "r2": ["gamma"]}
    spans = [
        _span(traces["r1"], "confirmation", 0, "alpha beta", "alpha beta"),
        _span(traces["r2"], "confirmation", 0, "beta gamma", "beta gamma"),
    ]
    cands, dropped = _run(traces, spans, known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED> beta", "beta <MASKED>"]


def test_deterministic_ordering():
    traces = {
        "b": _trace("b", [_turn(0, "user", "bee")]),
        "a": _trace("a", [_turn(0, "user", "aye")]),
        "c": _trace("c", [_turn(0, "user", "cee")]),
    }
    known = {"a": [], "b": [], "c": []}
    spans = [
        _span(traces["b"], "confirmation", 0, "bee", "bee"),
        _span(traces["a"], "confirmation", 0, "aye", "aye"),
        _span(traces["c"], "agent_refuses", 0, "cee", "cee"),
    ]
    cands, dropped = _run(traces, spans, known=known)
    assert dropped == ()
    assert [(c.situation, c.recording_id) for c in cands] == [
        ("agent_refuses", "c"),
        ("confirmation", "a"),
        ("confirmation", "b"),
    ]


def test_duplicate_identical_deduplicated():
    content = "dup here"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": []}
    span = _span(traces["r"], "confirmation", 0, content, content)
    twin = _raw_span(traces["r"], "confirmation", 0, content, content)
    cands, dropped = _run(traces, [span, span, twin], known=known)
    assert dropped == ()
    assert len(cands) == 1
    assert cands[0].masked_excerpt == content


def test_stable_under_reorder():
    traces = {
        "a": _trace("a", [_turn(0, "user", "aye 1")]),
        "b": _trace("b", [_turn(0, "user", "bee 2")]),
    }
    known = {"a": ["1"], "b": ["2"]}
    phantom = _trace("ghost", [_turn(0, "user", "x")])
    spans = [
        _span(traces["a"], "confirmation", 0, "aye 1", "aye 1"),
        _span(traces["b"], "giving_up", 0, "bee 2", "bee 2"),
        _span(phantom, "confirmation", 0, "x", "x"),
    ]
    first = _run(traces, spans, known=known)
    flipped = assemble_candidates(
        dict(reversed(list(traces.items()))),
        list(reversed(spans)),
        held_out_ids=set(),
        known_values_by_id=dict(reversed(list(known.items()))),
    )
    assert first == flipped


def test_mutation_isolation():
    content = "call 555-0100 now"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": ["555-0100"]}
    spans = [_span(traces["r"], "confirmation", 0, content, content)]
    cands, dropped = _run(traces, spans, known=known)
    assert [c.masked_excerpt for c in cands] == ["call <MASKED> now"]
    assert dropped == ()
    traces["r"].turns.append(_turn(9, "user", "late"))
    known["r"].append("late")
    spans.append(_span(traces["r"], "confirmation", 9, "late", "late"))
    assert [c.masked_excerpt for c in cands] == ["call <MASKED> now"]
    assert dropped == ()
    again, redropped = _run(traces, spans[:1], known={"r": ["555-0100"]})
    assert again == ()
    assert _reasons(redropped) == [("r", "stale_trace")]


def test_provenance_detached():
    first = _turn(0, "user", "hi 555 there")
    traces = {"r": _trace("r", [first])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, "hi 555 there", "hi 555 there")], known={"r": ["555"]}
    )
    assert dropped == ()
    assert cands[0].source == first.raw_ptr
    assert cands[0].source is not first.raw_ptr
    assert cands[0].trace_hash == trace_content_hash(traces["r"])
    assert (cands[0].recording_id, cands[0].turn_idx, cands[0].start, cands[0].end) == ("r", 0, 0, 12)


def test_no_task_builder_promotion_surface():
    text = Path("kullback/user/examples.py").read_text(encoding="utf-8")
    assert "kullback.builder" not in text
    assert "kullback.examiner" not in text
    assert "kullback.rounds" not in text
    assert "Task" not in text
    assert "promote" not in text
    assert "activate" not in text
    assert not hasattr(examples, "promote_view")
    assert callable(examples.assemble_candidates)
    assert set(ExampleWithheld.model_fields) == {"recording_id", "reason"}


def test_strict_identity_rejects_bytes():
    digest = "0" * 64
    with pytest.raises(ValidationError):
        ExampleSpan(recording_id=b"r", situation="confirmation", turn_idx=0, start=0, end=1, trace_hash=digest)
    with pytest.raises(ValidationError):
        ExampleSpan(recording_id=7, situation="confirmation", turn_idx=0, start=0, end=1, trace_hash=digest)


def test_bytes_identity_never_touches_held():
    held_content = "private-held-content"
    traces = _Watch({"held": _trace("held", [_turn(0, "user", held_content)])})
    known = _Watch({"held": ["private"]})
    raw = {
        "recording_id": b"held",
        "situation": "confirmation",
        "turn_idx": 0,
        "start": 0,
        "end": 7,
    }
    cands, dropped = assemble_candidates(
        traces, [raw], held_out_ids={"held"}, known_values_by_id=known
    )
    assert cands == ()
    assert _reasons(dropped) == [("unknown", "malformed_span")]
    assert traces.touched == []
    assert known.touched == []
    assert held_content not in repr((cands, dropped))


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("mapping_proxy", id="mapping_proxy"),
        pytest.param("user_dict", id="user_dict"),
    ],
)
def test_mapping_held_out_without_access(kind):
    held_content = "private-held-content"
    traces = _Watch({"held": _trace("held", [_turn(0, "user", held_content)])})
    known = _Watch({"held": ["private"]})
    span = {
        "recording_id": "held",
        "situation": "confirmation",
        "turn_idx": 0,
        "start": 0,
        "end": 7,
    }
    raw = MappingProxyType(span) if kind == "mapping_proxy" else UserDict(span)
    cands, dropped = assemble_candidates(
        traces, [raw], held_out_ids={"held"}, known_values_by_id=known
    )
    assert cands == ()
    assert _reasons(dropped) == [("held", "held_out")]
    assert traces.touched == []
    assert known.touched == []
    assert held_content not in repr((cands, dropped))


def test_plain_dict_held_out_invalid_offsets_still_held_out():
    held_content = "private-held-content"
    traces = _Watch({"held": _trace("held", [_turn(0, "user", held_content)])})
    known = _Watch({"held": ["private"]})
    raws = [
        {
            "recording_id": "held",
            "situation": "confirmation",
            "turn_idx": 99,
            "start": 0,
            "end": 1,
        },
        {
            "recording_id": "held",
            "situation": "confirmation",
            "turn_idx": 0,
            "start": 2,
            "end": 2,
        },
    ]
    cands, dropped = assemble_candidates(
        traces, raws, held_out_ids={"held"}, known_values_by_id=known
    )
    assert cands == ()
    assert _reasons(dropped) == [("held", "held_out")]
    assert traces.touched == []
    assert known.touched == []


def test_typed_span_held_out_without_access():
    held_content = "private-held-content"
    traces = _Watch({"held": _trace("held", [_turn(0, "user", held_content)])})
    known = _Watch({"held": ["private"]})
    span = _span(traces["held"], "confirmation", 0, held_content, held_content)
    cands, dropped = assemble_candidates(
        traces, [span], held_out_ids={"held"}, known_values_by_id=known
    )
    assert cands == ()
    assert _reasons(dropped) == [("held", "held_out")]
    assert traces.touched == []
    assert known.touched == []


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("mapping_proxy", id="mapping_proxy"),
        pytest.param("user_dict", id="user_dict"),
    ],
)
def test_mapping_allowed_id_works(kind):
    content = "ok public here"
    traces = {"ok": _trace("ok", [_turn(0, "user", content)])}
    span = {
        "recording_id": "ok",
        "situation": "confirmation",
        "turn_idx": 0,
        "start": 0,
        "end": len(content),
        "trace_hash": trace_content_hash(traces["ok"]),
    }
    raw = MappingProxyType(span) if kind == "mapping_proxy" else UserDict(span)
    cands, dropped = assemble_candidates(
        traces, [raw], held_out_ids={"held"}, known_values_by_id={"ok": []}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == [content]


def test_mixed_identities_only_allowed_candidate():
    held_content = "private-held-content"
    ok_content = "ok public here"
    traces = _Watch(
        {
            "ok": _trace("ok", [_turn(0, "user", ok_content)]),
            "held": _trace("held", [_turn(0, "user", held_content)]),
        }
    )
    known = _Watch({"ok": [], "held": ["private"]})
    spans = [
        {
            "recording_id": b"held",
            "situation": "confirmation",
            "turn_idx": 0,
            "start": 0,
            "end": 7,
        },
        MappingProxyType(_raw_span(traces["held"], "confirmation", 0, held_content, held_content)),
        UserDict(_raw_span(traces["held"], "confirmation", 0, held_content, held_content)),
        _span(traces["held"], "confirmation", 0, held_content, held_content),
        _span(traces["ok"], "confirmation", 0, ok_content, ok_content),
    ]
    cands, dropped = assemble_candidates(
        traces, spans, held_out_ids={"held"}, known_values_by_id=known
    )
    assert [c.masked_excerpt for c in cands] == [ok_content]
    assert _reasons(dropped) == [("held", "held_out"), ("unknown", "malformed_span")]
    assert "held" not in traces.touched
    assert "held" not in known.touched
    assert held_content not in repr((cands, dropped))


@pytest.mark.parametrize(
    ("content", "situation", "role", "orders"),
    [
        pytest.param("abc", "confirmation", "user", (["ab", "bc"], ["bc", "ab"]), id="abc"),
        pytest.param("ababa", "unknown_fact", "assistant", (["aba", "bab"], ["bab", "aba"]), id="ababa"),
    ],
)
def test_overlapping_orders_fully_masked(content, situation, role, orders):
    traces = {"r": _trace("r", [_turn(0, role, content)])}
    span = _span(traces["r"], situation, 0, content, content)
    for inventory in orders:
        cands, dropped = _run(traces, [span], known={"r": list(inventory)})
        assert dropped == ()
        assert [c.masked_excerpt for c in cands] == ["<MASKED>"]


def test_repeated_self_overlap_no_leak():
    content = "aaa"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": ["aa"]}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED>"]


def test_nested_long_short_single_token():
    content = "zzabcdefzz"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "company_only", 0, content, content)], known={"r": ["abcdef", "cde", "bcd"]}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["zz<MASKED>zz"]
    assert cands[0].masked_excerpt.count("<MASKED>") == 1


def test_adjacent_same_value_stays_split():
    content = "abab"
    traces = {"r": _trace("r", [_turn(0, "assistant", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "agent_refuses", 0, content, content)], known={"r": ["ab"]}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED><MASKED>"]


def test_unicode_overlap_no_leak():
    content = "a日本b"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": ["a日本", "日本b"]}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["<MASKED>"]


def test_overlap_inside_excerpt_only_covered_part():
    content = "xxabcxx"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": ["ab", "bc"]}
    cands, dropped = _run(traces, [_span(traces["r"], "confirmation", 0, content, content)], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["xx<MASKED>xx"]
    mid = _span_at(traces["r"], "confirmation", 0, 1, 6)
    cands, dropped = _run(traces, [mid], known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["x<MASKED>x"]
    tight = _span_at(traces["r"], "confirmation", 0, 3, 5)
    cands, dropped = _run(traces, [tight], known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "partial_boundary")]


def test_residual_synthesized_by_token_withheld():
    content = "xb"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, content, content)], known={"r": ["x", "<MASKED>b"]}
    )
    assert cands == ()
    assert _reasons(dropped) == [("r", "residual_known_value")]


def test_stale_same_length_text_withheld():
    actual = _trace("r", [_turn(0, "user", "abcd")])
    traces = {"r": actual}
    other = _trace("r", [_turn(0, "user", "abce")])
    stale = ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=4,
        trace_hash=trace_content_hash(other),
    )
    cands, dropped = _run(traces, [stale], known={"r": []})
    assert cands == ()
    assert _reasons(dropped) == [("r", "stale_trace")]
    assert "abce" not in repr(dropped)
    assert "abcd" not in repr(dropped)


def test_changed_raw_ptr_withheld():
    actual = _trace("r", [_turn(0, "user", "hello")])
    traces = {"r": actual}
    moved = _trace("r", [_turn(0, "user", "hello")])
    moved.turns[0].raw_ptr = RawPtr(file_hash="dd", sim_index=9)
    stale = ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=5,
        trace_hash=trace_content_hash(moved),
    )
    cands, dropped = _run(traces, [stale], known={"r": []})
    assert cands == ()
    assert _reasons(dropped) == [("r", "stale_trace")]


def test_copied_trace_hash_field_not_trusted():
    actual = _trace("r", [_turn(0, "user", "changed content here")])
    original = _trace("r", [_turn(0, "user", "original content here")])
    digest = trace_content_hash(original)
    actual.hash = digest
    stale = ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=7,
        trace_hash=digest,
    )
    traces = {"r": actual}
    cands, dropped = _run(traces, [stale], known={"r": []})
    assert cands == ()
    assert _reasons(dropped) == [("r", "stale_trace")]


def test_reconstructed_equal_trace_accepted():
    first = _trace("r", [_turn(0, "user", "same words")])
    second = _trace("r", [_turn(0, "user", "same words")])
    assert trace_content_hash(first) == trace_content_hash(second)
    traces = {"r": second}
    cands, dropped = _run(
        traces, [_span(first, "confirmation", 0, "same words", "same words")], known={"r": []}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["same words"]
    assert cands[0].trace_hash == trace_content_hash(second)


def test_malformed_fingerprint_withheld():
    content = "hello"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    known = {"r": []}
    good = trace_content_hash(traces["r"])
    raws = [
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "trace_hash": "zzz"},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "trace_hash": "A" * 64},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "trace_hash": ""},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1, "trace_hash": 123},
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1},
    ]
    cands, dropped = _run(traces, raws, known=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "malformed_span")]
    with pytest.raises(ValidationError):
        ExampleSpan(recording_id="r", situation="confirmation", turn_idx=0, start=0, end=1, trace_hash="zzz")
    assert good == trace_content_hash(traces["r"])


def test_stale_check_before_inventory():
    actual = _trace("r", [_turn(0, "user", "abcd")])
    traces = _Watch({"r": actual})
    other = _trace("r", [_turn(0, "user", "abce")])
    stale = ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=4,
        trace_hash=trace_content_hash(other),
    )
    known = _Watch({})
    cands, dropped = assemble_candidates(
        traces, [stale], held_out_ids=set(), known_values_by_id=known
    )
    assert cands == ()
    assert _reasons(dropped) == [("r", "stale_trace")]
    assert known.touched == []


def test_span_candidate_field_sets_include_revision():
    assert set(ExampleSpan.model_fields) == {
        "recording_id",
        "situation",
        "turn_idx",
        "start",
        "end",
        "trace_hash",
    }
    assert set(ExampleCandidate.model_fields) == {
        "recording_id",
        "situation",
        "masked_excerpt",
        "turn_idx",
        "start",
        "end",
        "trace_hash",
        "source",
    }
