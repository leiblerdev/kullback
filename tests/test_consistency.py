from __future__ import annotations

import pytest

from kullback.consistency import (
    BUDGET_LIMITED,
    NOT_FAILING,
    SHORTEST_SUBSEQUENCE,
    ShrinkResult,
    shrink_sequence,
)


class Bench:
    def __init__(self):
        self.store = {}

    def run(self, ops):
        self.store = {}
        broken = False
        for op in ops:
            if op[0] == "put":
                if op[1] in self.store:
                    broken = True
                self.store[op[1]] = op[2]
            elif op[0] == "get":
                self.store.get(op[1])
        return broken


def bench_fails(ops):
    return Bench().run(list(ops))


def test_status_constants_use_the_named_values():
    assert NOT_FAILING == "not_failing"
    assert SHORTEST_SUBSEQUENCE == "shortest_subsequence"
    assert BUDGET_LIMITED == "budget_limited"


def test_original_not_failing_returns_full_sequence_with_one_evaluation():
    seen = []
    seq = [1, 2, 3]

    def fails(candidate):
        seen.append(list(candidate))
        return False

    result = shrink_sequence(seq, fails, max_evaluations=10)
    assert isinstance(result, ShrinkResult)
    assert result.status == NOT_FAILING
    assert result.subsequence == (1, 2, 3)
    assert result.indices == (0, 1, 2)
    assert result.evaluations == 1
    assert seen == [[1, 2, 3]]


def test_original_is_probed_first_with_the_full_sequence():
    seen = []
    seq = ["a", "b", "c"]

    def fails(candidate):
        seen.append(list(candidate))
        return True

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert seen[0] == ["a", "b", "c"]
    assert result.evaluations == len(seen)


def test_single_failing_call_shrinks_to_itself():
    result = shrink_sequence(["only"], lambda s: list(s) == ["only"], max_evaluations=10)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.subsequence == ("only",)
    assert result.indices == (0,)
    assert result.evaluations == 2


def test_pairwise_dependency_through_stateful_processor():
    seq = [("put", "k", 1), ("get", "k"), ("put", "j", 2), ("put", "k", 3)]
    result = shrink_sequence(seq, bench_fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 3)
    assert result.subsequence == (("put", "k", 1), ("put", "k", 3))
    assert Bench().run(result.subsequence) is True


def test_pairwise_result_is_proven_shortest():
    seq = [("put", "k", 1), ("get", "k"), ("put", "j", 2), ("put", "k", 3)]
    result = shrink_sequence(seq, bench_fails, max_evaluations=100)
    assert result.evaluations == 9
    assert result.status == SHORTEST_SUBSEQUENCE


def test_non_monotonic_case_beats_deletion_minimal_greedy():
    seq = ["a", "b", "c", "d"]

    def fails(candidate):
        items = list(candidate)
        return items == ["a", "b", "c", "d"] or items == ["c"]

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.subsequence == ("c",)
    assert result.indices == (2,)
    assert result.evaluations == 5


def test_deterministic_tie_break_picks_lowest_index_order():
    seq = ["p", "q"]

    def fails(candidate):
        items = list(candidate)
        return "p" in items or "q" in items

    first = shrink_sequence(seq, fails, max_evaluations=100)
    second = shrink_sequence(list(reversed(seq)), fails, max_evaluations=100)
    assert first.indices == (0,)
    assert first.subsequence == ("p",)
    assert second.indices == (0,)
    assert second.subsequence == ("q",)


def test_duplicate_equal_calls_keep_distinct_positions():
    seq = [{"op": "tick"}, {"op": "tick"}]
    assert seq[0] == seq[1]
    result = shrink_sequence(seq, lambda s: len(list(s)) == 2, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.subsequence == ({"op": "tick"}, {"op": "tick"})


def test_empty_failing_sequence_is_shortest_with_one_evaluation():
    result = shrink_sequence([], lambda s: True, max_evaluations=10)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.subsequence == ()
    assert result.indices == ()
    assert result.evaluations == 1


def test_empty_passing_sequence_is_not_failing():
    result = shrink_sequence([], lambda s: False, max_evaluations=10)
    assert result.status == NOT_FAILING
    assert result.subsequence == ()
    assert result.evaluations == 1


def test_budget_of_one_returns_original_as_budget_limited():
    result = shrink_sequence([1, 2, 3], lambda s: True, max_evaluations=1)
    assert result.status == BUDGET_LIMITED
    assert result.subsequence == (1, 2, 3)
    assert result.indices == (0, 1, 2)
    assert result.evaluations == 1


def test_exact_budget_boundary_flips_status():
    seq = ["a", "b", "c", "d"]

    def fails(candidate):
        items = list(candidate)
        return items == ["a", "b", "c", "d"] or items == ["c"]

    exact = shrink_sequence(seq, fails, max_evaluations=5)
    assert exact.status == SHORTEST_SUBSEQUENCE
    assert exact.evaluations == 5
    assert exact.subsequence == ("c",)
    short = shrink_sequence(seq, fails, max_evaluations=4)
    assert short.status == BUDGET_LIMITED
    assert short.evaluations == 4
    assert short.subsequence == ("a", "b", "c", "d")


def test_no_evaluation_happens_after_the_cap():
    calls = []
    seq = ["a", "b", "c", "d", "e"]

    def fails(candidate):
        calls.append(list(candidate))
        return list(candidate) == seq

    result = shrink_sequence(seq, fails, max_evaluations=3)
    assert len(calls) == 3
    assert result.evaluations == 3
    assert result.status == BUDGET_LIMITED


def test_predicate_exception_on_short_candidate_propagates():
    seq = ["x"]

    def fails(candidate):
        if len(candidate) == 0:
            raise RuntimeError("probe blew up")
        return True

    with pytest.raises(RuntimeError):
        shrink_sequence(seq, fails, max_evaluations=10)


def test_predicate_exception_on_original_propagates():
    def fails(candidate):
        raise RuntimeError("always blows up")

    with pytest.raises(RuntimeError):
        shrink_sequence([1, 2], fails, max_evaluations=10)


@pytest.mark.parametrize("budget", [0, -1, -100])
def test_non_positive_budget_raises_value_error(budget):
    with pytest.raises(ValueError):
        shrink_sequence([1], lambda s: True, max_evaluations=budget)


@pytest.mark.parametrize("budget", [True, False, 1.5, "3", None, (2,)])
def test_non_int_budget_raises_type_error(budget):
    with pytest.raises(TypeError):
        shrink_sequence([1], lambda s: True, max_evaluations=budget)


def test_missing_budget_is_rejected():
    with pytest.raises(TypeError):
        shrink_sequence([1], lambda s: True)


def test_positional_budget_is_rejected():
    with pytest.raises(TypeError):
        shrink_sequence([1], lambda s: True, 10)


def test_non_callable_predicate_is_rejected():
    with pytest.raises(TypeError):
        shrink_sequence([1], "not-callable", max_evaluations=10)


def test_mutating_predicate_cannot_corrupt_caller_data():
    seq = [{"v": 1}, {"v": 2}]
    before = [{"v": 1}, {"v": 2}]

    def fails(candidate):
        candidate.append("junk")
        for item in candidate:
            if isinstance(item, dict):
                item["v"] = 999
        return True

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert seq == before
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.subsequence == ()


def test_probes_are_independent_objects():
    log = []
    seq = [{"v": 1}, {"v": 2}]

    def fails(candidate):
        log.append(candidate)
        return len(candidate) > 0

    shrink_sequence(seq, fails, max_evaluations=100)
    assert len(log) == 3
    assert log[0] is not log[2]
    assert all(entry is not seq for entry in log)
    log[0][0]["v"] = 999
    log[0].append("junk")
    assert log[2] == [{"v": 1}]


def test_output_is_detached_from_later_caller_mutation():
    seq = [{"v": 1}, {"v": 2}]
    result = shrink_sequence(seq, lambda s: True, max_evaluations=1)
    seq[0]["v"] = 999
    seq.append({"v": 3})
    assert result.subsequence == ({"v": 1}, {"v": 2})
    assert result.indices == (0, 1)


def test_mutating_result_items_does_not_leak_into_next_call():
    seq = [{"v": 1}, {"v": 2}]
    first = shrink_sequence(seq, lambda s: True, max_evaluations=1)
    first.subsequence[0]["v"] = 999
    second = shrink_sequence(seq, lambda s: True, max_evaluations=1)
    assert second.subsequence == ({"v": 1}, {"v": 2})
    assert second.subsequence[0] == {"v": 1}


def test_shared_identity_across_entries_is_preserved():
    shared = {"value": 1}
    seq = [shared, shared]
    assert seq[0] is seq[1]

    def fails(candidate):
        items = list(candidate)
        return len(items) == 2 and items[0] is items[1]

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert len(result.subsequence) == 2
    assert result.subsequence[0] is result.subsequence[1]
    assert list(result.subsequence) == [{"value": 1}, {"value": 1}]
    assert fails(list(result.subsequence)) is True


def test_shortest_proper_subsequence_needs_alias_identity():
    shared = {"value": 1}
    other = {"value": 1}
    assert shared == other
    assert shared is not other
    seq = [shared, shared, other]

    def fails(candidate):
        items = list(candidate)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i] is items[j]:
                    return True
        return False

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.evaluations == 6
    assert result.subsequence[0] is result.subsequence[1]
    assert fails(list(result.subsequence)) is True


def test_nested_shared_reference_is_preserved():
    inner = {"v": 1}
    seq = [{"a": inner}, {"b": inner}]

    def fails(candidate):
        items = list(candidate)
        return len(items) == 2 and items[0]["a"] is items[1]["b"]

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.subsequence[0]["a"] is result.subsequence[1]["b"]
    assert list(result.subsequence) == [{"a": {"v": 1}}, {"b": {"v": 1}}]
    assert fails(list(result.subsequence)) is True


def test_cyclic_alias_is_preserved():
    node: dict = {}
    node["me"] = node
    seq = [node, node]
    assert seq[0] is seq[1]

    def fails(candidate):
        items = list(candidate)
        return len(items) == 2 and items[0] is items[1] and items[0]["me"] is items[0]

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.subsequence[0] is result.subsequence[1]
    assert result.subsequence[0]["me"] is result.subsequence[0]
    assert fails(list(result.subsequence)) is True


def test_mutating_shared_probe_stays_isolated():
    shared = {"v": 1}
    seq = [shared, shared]
    seen: list = []

    def fails(candidate):
        items = list(candidate)
        if len(items) == 2:
            alias = items[0] is items[1]
            seen.append((alias, items[0]["v"]))
            items[0]["v"] = 999
            items.append("junk")
            return alias and seen[-1][1] == 1
        if len(items) > 0:
            seen.append((None, items[0]["v"]))
            items[0]["v"] = 999
            items.append("junk")
        return False

    result = shrink_sequence(seq, fails, max_evaluations=100)
    assert seq[0] == {"v": 1}
    assert seq[1] == {"v": 1}
    assert seq[0] is seq[1]
    assert len(seen) > 0
    assert all(value == 1 for _, value in seen)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.subsequence[0] is result.subsequence[1]
    assert result.subsequence[0] == {"v": 1}
