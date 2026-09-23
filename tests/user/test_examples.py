from __future__ import annotations

from collections import UserDict
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from kullback.runner.records import RawPtr, Trace, Turn
from kullback.user.examples import (
    ExampleCandidate,
    ExampleSpan,
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


def _as_kind(span, kind):
    if kind == "typed":
        return ExampleSpan(**span)
    return MappingProxyType(span) if kind == "mapping_proxy" else UserDict(span)


@pytest.mark.parametrize("kind", ["typed", "mapping_proxy", "user_dict"])
def test_each_of_the_five_situations_yields_a_candidate(kind):
    situations = ["unknown_fact", "company_only", "agent_refuses", "confirmation", "giving_up"]
    traces = {}
    spans = []
    known = {}
    for pos, situation in enumerate(situations):
        rid = "r" + str(pos)
        content = "excerpt for " + rid
        traces[rid] = _trace(rid, [_turn(0, "assistant", content)])
        spans.append(_as_kind(_raw_span(traces[rid], situation, 0, content, content), kind))
        known[rid] = []
    cands, dropped = _run(traces, spans, held={"held"}, known=known)
    assert dropped == ()
    assert len(cands) == 5
    assert sorted(c.situation for c in cands) == sorted(situations)
    for cand in cands:
        assert cand.masked_excerpt == "excerpt for " + cand.recording_id


def _hello():
    return {"r": _trace("r", [_turn(0, "user", "hello")])}


def _raw(digest=None, **fields):
    span = {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1}
    if digest is not None:
        span["trace_hash"] = digest
    span.update(fields)
    return span


def _typed(**fields):
    return {**_raw("0" * 64), **fields}


def _withheld(traces, spans, known, want, leaks=(), typed=()):
    return traces, list(spans), known, want, leaks, typed


def _malformed_shape():
    raws = [
        _raw(situation="nope"),
        {"recording_id": "r", "situation": "confirmation", "turn_idx": 0, "start": 0},
        _raw(extra=1),
        {"situation": "confirmation", "turn_idx": 0, "start": 0, "end": 1},
        _raw(task_id="t"),
        42,
    ]
    typed = [_typed(recording_id=b"r"), _typed(recording_id=7)]
    return _withheld(_hello(), raws, {"r": []}, [("r", "malformed_span"), ("unknown", "malformed_span")], typed=typed)


def _turn_idx_not_int():
    traces = _hello()
    digest = trace_content_hash(traces["r"])
    raws = [_raw(digest, turn_idx=True), _raw(digest, turn_idx=1.0), _raw(digest, turn_idx="0")]
    typed = [_typed(turn_idx=True), _typed(turn_idx=1.0), _typed(turn_idx="0")]
    return _withheld(traces, raws, {"r": []}, [("r", "unknown_turn")], typed=typed)


def _bounds_not_int():
    traces = _hello()
    digest = trace_content_hash(traces["r"])
    raws = [_raw(digest, start="0"), _raw(digest, end=1.5), _raw(digest, start=False)]
    typed = [_typed(start="0"), _typed(end=1.5)]
    return _withheld(traces, raws, {"r": []}, [("r", "bad_bounds")], typed=typed)


def _prerequisite(case):
    def build():
        content = "hi there"
        if case == "missing_trace":
            phantom = _trace("ghost", [_turn(0, "user", "hi")])
            return _withheld({}, [_span(phantom, "confirmation", 0, "hi", "hi")], {}, [("ghost", "missing_trace")])
        traces = {"r": _trace("r", [_turn(0, "user", content)])}
        if case == "missing_inventory":
            span = _span(traces["r"], "confirmation", 0, content, content)
            return _withheld(traces, [span], {}, [("r", "unknown_mask_inventory")])
        other = {"r": _trace("other", [_turn(0, "user", content)])}
        span = ExampleSpan(
            recording_id="r",
            situation="confirmation",
            turn_idx=0,
            start=0,
            end=len(content),
            trace_hash=trace_content_hash(other["r"]),
        )
        return _withheld(other, [span], {"r": []}, [("r", "trace_mismatch")])

    return build


def _phantom_never_leaks_its_inventory():
    secret = "zxq-secret-9"
    content = "wrap " + secret + " twice " + secret
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    phantom = _trace("ghost", [_turn(0, "user", "x")])
    span = _span(phantom, "confirmation", 0, "x", "x")
    return _withheld(traces, [span], {"ghost": [secret]}, [("ghost", "missing_trace")], leaks=[secret])


def _unknown_turn():
    traces = _hello()
    return _withheld(traces, [_span(traces["r"], "confirmation", 9, "hello", "hell")], {"r": []}, [("r", "unknown_turn")])


def _ambiguous_turn():
    dup = {"r": _trace("r", [_turn(1, "user", "first"), _turn(1, "assistant", "second")])}
    span = _span(dup["r"], "confirmation", 1, "first", "first")
    return _withheld(dup, [span], {"r": []}, [("r", "ambiguous_turn")])


def _bad_bounds():
    traces = _hello()
    bounds = [(2, 2), (3, 2), (-1, 2), (0, 99)]
    spans = [_span_at(traces["r"], "confirmation", 0, start, end) for start, end in bounds]
    return _withheld(traces, spans, {"r": []}, [("r", "bad_bounds")])


def _empty_turn():
    none_trace = {"r": _trace("r", [_turn(0, "user", None)])}
    ghost = _span_at(none_trace["r"], "confirmation", 0, 0, 1)
    return _withheld(none_trace, [ghost], {"r": []}, [("r", "bad_bounds")])


def _partial_boundary(start, end, known_values, content="hello WORLD bye"):
    def build():
        traces = {"r": _trace("r", [_turn(0, "user", content)])}
        cut = _span_at(traces["r"], "confirmation", 0, start, end)
        return _withheld(traces, [cut], {"r": known_values}, [("r", "partial_boundary")])

    return build


def _one_turn_whole(content, known_values, want, leaks=()):
    def build():
        traces = {"r": _trace("r", [_turn(0, "user", content)])}
        span = _span(traces["r"], "confirmation", 0, content, content)
        return _withheld(traces, [span], {"r": known_values}, [("r", want)], leaks=leaks)

    return build


def _malformed_fingerprint():
    raws = [_raw("zzz"), _raw("A" * 64), _raw(""), _raw(123), _raw()]
    return _withheld(_hello(), raws, {"r": []}, [("r", "malformed_span")], typed=[_typed(trace_hash="zzz")])


_BAD_INVENTORY_LEAKS = ("zxq-secret-9", "ok")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_malformed_shape, id="malformed_shape"),
        pytest.param(_turn_idx_not_int, id="turn_idx_not_int"),
        pytest.param(_bounds_not_int, id="bounds_not_int"),
        pytest.param(_prerequisite("missing_trace"), id="missing_trace"),
        pytest.param(_prerequisite("missing_inventory"), id="missing_inventory"),
        pytest.param(_prerequisite("trace_mismatch"), id="trace_mismatch"),
        pytest.param(_phantom_never_leaks_its_inventory, id="missing_trace_keeps_inventory_private"),
        pytest.param(_unknown_turn, id="unknown_turn"),
        pytest.param(_ambiguous_turn, id="ambiguous_turn"),
        pytest.param(_bad_bounds, id="bad_bounds"),
        pytest.param(_empty_turn, id="empty_turn"),
        pytest.param(_partial_boundary(8, 14, ["WORLD"]), id="partial_boundary_start"),
        pytest.param(_partial_boundary(0, 8, ["WORLD"]), id="partial_boundary_end"),
        pytest.param(_partial_boundary(1, 3, ["hello WORLD bye"]), id="partial_boundary_covering"),
        pytest.param(_partial_boundary(3, 5, ["ab", "bc"], "xxabcxx"), id="partial_boundary_inside_overlap"),
        pytest.param(_one_turn_whole("see MASK here", ["MASK"], "token_collision"), id="token_collision_word"),
        pytest.param(_one_turn_whole("see MASK here", ["<MASKED>"], "token_collision"), id="token_collision_token"),
        pytest.param(_one_turn_whole("see MASK here", [">"], "token_collision"), id="token_collision_char"),
        pytest.param(_one_turn_whole("xb", ["x", "<MASKED>b"], "residual_known_value"), id="residual_known_value"),
        *[
            pytest.param(
                _one_turn_whole("has zxq-secret-9 inside", bad, "bad_inventory", leaks=_BAD_INVENTORY_LEAKS),
                id="bad_inventory_" + name,
            )
            for name, bad in [
                ("non_string", ["ok", 7]),
                ("empty_value", ["ok", ""]),
                ("string", "ok-string"),
                ("none", None),
                ("mapping", {"k": "v"}),
            ]
        ],
        pytest.param(_malformed_fingerprint, id="malformed_fingerprint"),
    ],
)
def test_a_malformed_span_is_withheld_with_its_reason(case):
    traces, spans, known, want, leaks, typed = case()
    cands, dropped = _run(traces, spans, known=known)
    assert cands == ()
    assert _reasons(dropped) == want
    for leak in leaks:
        assert leak not in repr(dropped)
    for fields in typed:
        with pytest.raises(ValidationError):
            ExampleSpan(**fields)


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


HELD_CONTENT = "private-held-content"
OK_CONTENT = "ok 111 here"


def _held_raw(**fields):
    return {"recording_id": "held", "situation": "confirmation", "turn_idx": 0, "start": 0, "end": 7, **fields}


def _held_spans(case, traces):
    held_typed = _span(traces["held"], "confirmation", 0, HELD_CONTENT, HELD_CONTENT)
    held_raw = _raw_span(traces["held"], "confirmation", 0, HELD_CONTENT, HELD_CONTENT)
    if case == "typed_beside_allowed":
        return [_span(traces["ok"], "confirmation", 0, OK_CONTENT, OK_CONTENT), held_typed]
    if case == "bytes_identity":
        return [_held_raw(recording_id=b"held")]
    if case == "mapping_proxy":
        return [MappingProxyType(_held_raw())]
    if case == "user_dict":
        return [UserDict(_held_raw())]
    if case == "plain_dict_invalid_offsets":
        return [_held_raw(turn_idx=99, end=1), _held_raw(start=2, end=2)]
    if case == "typed":
        return [held_typed]
    return [
        _held_raw(recording_id=b"held"),
        MappingProxyType(held_raw),
        UserDict(held_raw),
        held_typed,
        _span(traces["ok"], "confirmation", 0, OK_CONTENT, OK_CONTENT),
    ]


@pytest.mark.parametrize(
    ("case", "with_ok", "want_excerpts", "want_reasons"),
    [
        pytest.param("typed_beside_allowed", True, ["ok <MASKED> here"], [("held", "held_out")], id="typed_beside_allowed"),
        pytest.param("bytes_identity", False, [], [("unknown", "malformed_span")], id="bytes_identity"),
        pytest.param("mapping_proxy", False, [], [("held", "held_out")], id="mapping_proxy"),
        pytest.param("user_dict", False, [], [("held", "held_out")], id="user_dict"),
        pytest.param("plain_dict_invalid_offsets", False, [], [("held", "held_out")], id="plain_dict_invalid_offsets"),
        pytest.param("typed", False, [], [("held", "held_out")], id="typed"),
        pytest.param(
            "mixed",
            True,
            ["ok <MASKED> here"],
            [("held", "held_out"), ("unknown", "malformed_span")],
            id="mixed_identities",
        ),
    ],
)
def test_a_held_out_recording_is_never_read(case, with_ok, want_excerpts, want_reasons):
    world = {"held": _trace("held", [_turn(0, "user", HELD_CONTENT)])}
    inventory = {"held": _Boom()}
    if with_ok:
        world["ok"] = _trace("ok", [_turn(0, "user", OK_CONTENT)])
        inventory["ok"] = ["111"]
    traces = _Watch(world)
    known = _Watch(inventory)
    spans = _held_spans(case, traces)
    cands, dropped = assemble_candidates(traces, spans, held_out_ids={"held"}, known_values_by_id=known)
    assert [c.masked_excerpt for c in cands] == want_excerpts
    assert _reasons(dropped) == want_reasons
    touched = {"ok"} if with_ok else set()
    assert set(traces.touched) == touched
    assert set(known.touched) == touched
    assert HELD_CONTENT not in repr((cands, dropped))


SECRET = "zxq-secret-9"


@pytest.mark.parametrize(
    ("world", "spans", "known", "want"),
    [
        pytest.param(
            {"r": (0, "user", "call me at 555-0100")},
            [("r", 0, "call me at 555-0100")],
            {"r": ["555-0100"]},
            ["call me at <MASKED>"],
            id="full_excerpt",
        ),
        pytest.param(
            {"r": (0, "user", "call me at 555-0100")},
            [("r", 0, "555-0100")],
            {"r": ["555-0100"]},
            ["<MASKED>"],
            id="user_spoken_value",
        ),
        pytest.param(
            {"r": (3, "assistant", "banana banana")},
            [("r", 3, "banana banana")],
            {"r": ["a"]},
            ["b<MASKED>n<MASKED>n<MASKED> b<MASKED>n<MASKED>n<MASKED>"],
            id="single_char_repeated",
        ),
        pytest.param({"r": (0, "user", "hello")}, [("r", 0, "hello")], {"r": ["he"]}, ["<MASKED>llo"], id="substring"),
        pytest.param({"r": (0, "user", "abcd")}, [("r", 0, "abcd")], {"r": ["ab", "cd"]}, ["<MASKED><MASKED>"], id="adjacent"),
        pytest.param(
            {"r": (0, "user", "says 日本語 ok")},
            [("r", 0, "says 日本語 ok")],
            {"r": ["日本語"]},
            ["says <MASKED> ok"],
            id="unicode",
        ),
        pytest.param(
            {"r": (1, "assistant", "key a.c*(x)[y]\\z end")},
            [("r", 1, "key a.c*(x)[y]\\z end")],
            {"r": ["a.c*(x)[y]\\z"]},
            ["key <MASKED> end"],
            id="regex_metachars_literal",
        ),
        pytest.param(
            {"r": (0, "user", "secret Secret SECRET")},
            [("r", 0, "secret Secret SECRET")],
            {"r": ["Secret"]},
            ["secret <MASKED> SECRET"],
            id="case_sensitive",
        ),
        pytest.param({"r": (0, "user", "aaa BBB aaa")}, [("r", 0, (8, 11))], {"r": ["BBB"]}, ["aaa"], id="value_outside_excerpt"),
        pytest.param({"r": (0, "user", "plain text 123")}, [("r", 0, "plain text 123")], {"r": []}, ["plain text 123"], id="empty_inventory"),
        pytest.param(
            {"r": (0, "user", "wrap " + SECRET + " twice " + SECRET)},
            [("r", 0, "wrap " + SECRET + " twice " + SECRET)],
            {"r": [SECRET]},
            ["wrap <MASKED> twice <MASKED>"],
            id="repeated_secret",
        ),
        pytest.param(
            {"r1": (0, "user", "alpha beta"), "r2": (0, "user", "beta gamma")},
            [("r1", 0, "alpha beta"), ("r2", 0, "beta gamma")],
            {"r1": ["alpha"], "r2": ["gamma"]},
            ["<MASKED> beta", "beta <MASKED>"],
            id="only_own_inventory",
        ),
    ],
)
def test_every_known_value_is_masked_and_no_raw_value_escapes(world, spans, known, want):
    traces = {rid: _trace(rid, [_turn(idx, role, content)]) for rid, (idx, role, content) in world.items()}
    made = []
    for rid, idx, quote in spans:
        content = world[rid][2]
        if isinstance(quote, tuple):
            made.append(_span_at(traces[rid], "confirmation", idx, *quote))
        else:
            made.append(_span(traces[rid], "confirmation", idx, content, quote))
    cands, dropped = _run(traces, made, known=known)
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == want
    for cand in cands:
        for value in known[cand.recording_id]:
            assert value not in cand.masked_excerpt
            if len(value) >= 3:
                assert value not in repr(cand)
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


def test_candidates_are_deterministic_deduplicated_and_order_free():
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

    content = "dup here"
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    span = _span(traces["r"], "confirmation", 0, content, content)
    twin = _raw_span(traces["r"], "confirmation", 0, content, content)
    cands, dropped = _run(traces, [span, span, twin], known={"r": []})
    assert dropped == ()
    assert len(cands) == 1
    assert cands[0].masked_excerpt == content

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

    first_turn = _turn(0, "user", "hi 555 there")
    traces = {"r": _trace("r", [first_turn])}
    cands, dropped = _run(
        traces, [_span(traces["r"], "confirmation", 0, "hi 555 there", "hi 555 there")], known={"r": ["555"]}
    )
    assert dropped == ()
    assert cands[0].source == first_turn.raw_ptr
    assert cands[0].source is not first_turn.raw_ptr
    assert cands[0].trace_hash == trace_content_hash(traces["r"])
    assert (cands[0].recording_id, cands[0].turn_idx, cands[0].start, cands[0].end) == ("r", 0, 0, 12)


@pytest.mark.parametrize(
    ("content", "inventory", "bounds", "want"),
    [
        pytest.param("abc", ["ab", "bc"], None, "<MASKED>", id="abc"),
        pytest.param("ababa", ["aba", "bab"], None, "<MASKED>", id="ababa"),
        pytest.param("xxabcdefxx", ["cde", "abcdef"], None, "xx<MASKED>xx", id="longest_first"),
        pytest.param("aaa", ["aa"], None, "<MASKED>", id="repeated_self_overlap"),
        pytest.param("zzabcdefzz", ["abcdef", "cde", "bcd"], None, "zz<MASKED>zz", id="nested_single_token"),
        pytest.param("abab", ["ab"], None, "<MASKED><MASKED>", id="adjacent_same_value_stays_split"),
        pytest.param("a日本b", ["a日本", "日本b"], None, "<MASKED>", id="unicode"),
        pytest.param("xxabcxx", ["ab", "bc"], None, "xx<MASKED>xx", id="inside_excerpt"),
        pytest.param("xxabcxx", ["ab", "bc"], (1, 6), "x<MASKED>x", id="inside_part_of_excerpt"),
    ],
)
def test_overlapping_values_are_masked_whole_in_any_order(content, inventory, bounds, want):
    traces = {"r": _trace("r", [_turn(0, "user", content)])}
    if bounds is None:
        span = _span(traces["r"], "confirmation", 0, content, content)
    else:
        span = _span_at(traces["r"], "confirmation", 0, *bounds)
    for order in (list(inventory), list(reversed(inventory))):
        cands, dropped = _run(traces, [span], known={"r": order})
        assert dropped == ()
        assert [c.masked_excerpt for c in cands] == [want]


def _stale_span(source, end):
    return ExampleSpan(
        recording_id="r",
        situation="confirmation",
        turn_idx=0,
        start=0,
        end=end,
        trace_hash=trace_content_hash(source),
    )


def _copied_hash_field():
    actual = _trace("r", [_turn(0, "user", "changed content here")])
    original = _trace("r", [_turn(0, "user", "original content here")])
    actual.hash = trace_content_hash(original)
    return actual, _stale_span(original, 7), ()


def _same_length_text():
    other = _trace("r", [_turn(0, "user", "abce")])
    return _trace("r", [_turn(0, "user", "abcd")]), _stale_span(other, 4), ("abce", "abcd")


def _changed_raw_ptr():
    moved = _trace("r", [_turn(0, "user", "hello")])
    moved.turns[0].raw_ptr = RawPtr(file_hash="dd", sim_index=9)
    return _trace("r", [_turn(0, "user", "hello")]), _stale_span(moved, 5), ()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_copied_hash_field, id="copied_hash_field"),
        pytest.param(_same_length_text, id="same_length_text"),
        pytest.param(_changed_raw_ptr, id="changed_raw_ptr"),
    ],
)
def test_a_span_over_a_changed_recording_is_withheld_as_stale(case):
    actual, stale, leaks = case()
    traces = _Watch({"r": actual})
    known = _Watch({"r": []})
    cands, dropped = assemble_candidates(traces, [stale], held_out_ids=set(), known_values_by_id=known)
    assert cands == ()
    assert _reasons(dropped) == [("r", "stale_trace")]
    assert known.touched == []
    for leak in leaks:
        assert leak not in repr(dropped)

    first = _trace("r", [_turn(0, "user", "same words")])
    second = _trace("r", [_turn(0, "user", "same words")])
    assert trace_content_hash(first) == trace_content_hash(second)
    cands, dropped = _run(
        {"r": second}, [_span(first, "confirmation", 0, "same words", "same words")], known={"r": []}
    )
    assert dropped == ()
    assert [c.masked_excerpt for c in cands] == ["same words"]
    assert cands[0].trace_hash == trace_content_hash(second)
