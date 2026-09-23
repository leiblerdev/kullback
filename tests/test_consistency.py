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


def test_the_original_is_probed_first_and_a_passing_original_ends_the_search():
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

    failing_seen = []

    def always_fails(candidate):
        failing_seen.append(list(candidate))
        return True

    failing = shrink_sequence(["a", "b", "c"], always_fails, max_evaluations=100)
    assert failing_seen[0] == ["a", "b", "c"]
    assert failing.evaluations == len(failing_seen)


def test_duplicate_equal_calls_keep_distinct_positions():
    seq = [{"op": "tick"}, {"op": "tick"}]
    assert seq[0] == seq[1]
    result = shrink_sequence(seq, lambda s: len(list(s)) == 2, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 1)
    assert result.subsequence == ({"op": "tick"}, {"op": "tick"})


def test_pairwise_dependency_shrinks_to_shortest():
    seq = [("put", "k", 1), ("get", "k"), ("put", "j", 2), ("put", "k", 3)]
    result = shrink_sequence(seq, bench_fails, max_evaluations=100)
    assert result.status == SHORTEST_SUBSEQUENCE
    assert result.indices == (0, 3)
    assert result.subsequence == (("put", "k", 1), ("put", "k", 3))
    assert result.evaluations == 9
    assert Bench().run(result.subsequence) is True


@pytest.mark.parametrize(
    "budget,status,expected,indices,evaluations",
    [
        (100, SHORTEST_SUBSEQUENCE, ("c",), (2,), 5),
        (5, SHORTEST_SUBSEQUENCE, ("c",), (2,), 5),
        (4, BUDGET_LIMITED, ("a", "b", "c", "d"), (0, 1, 2, 3), 4),
        (1, BUDGET_LIMITED, ("a", "b", "c", "d"), (0, 1, 2, 3), 1),
    ],
    ids=["ample", "exact", "short", "one"],
)
def test_non_monotonic_budget_boundary(budget, status, expected, indices, evaluations):
    seq = ["a", "b", "c", "d"]

    def fails(candidate):
        items = list(candidate)
        return items == ["a", "b", "c", "d"] or items == ["c"]

    result = shrink_sequence(seq, fails, max_evaluations=budget)
    assert result.status == status
    assert result.subsequence == expected
    assert result.indices == indices
    assert result.evaluations == evaluations


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


@pytest.mark.parametrize(
    "fails,budget,error",
    [
        *[(lambda s: True, budget, ValueError) for budget in (0, -1, -100)],
        *[(lambda s: True, budget, TypeError) for budget in (True, False, 1.5, "3", None, (2,))],
        ("not-callable", 10, TypeError),
    ],
)
def test_a_budget_that_is_not_a_positive_int_or_a_predicate_that_cannot_be_called_is_refused(fails, budget, error):
    with pytest.raises(error):
        shrink_sequence([1], fails, max_evaluations=budget)


def test_a_predicate_mutating_its_probe_corrupts_neither_the_caller_data_nor_other_probes():
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

    log = []

    def logs(candidate):
        log.append(candidate)
        return len(candidate) > 0

    shrink_sequence(seq, logs, max_evaluations=100)
    assert len(log) == 3
    assert log[0] is not log[2]
    assert all(entry is not seq for entry in log)
    log[0][0]["v"] = 999
    log[0].append("junk")
    assert log[2] == [{"v": 1}]

    shared = {"v": 1}
    aliased = [shared, shared]
    seen: list = []

    def mutates_aliases(candidate):
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

    isolated = shrink_sequence(aliased, mutates_aliases, max_evaluations=100)
    assert aliased[0] == {"v": 1}
    assert aliased[1] == {"v": 1}
    assert aliased[0] is aliased[1]
    assert len(seen) > 0
    assert all(value == 1 for _, value in seen)
    assert isolated.status == SHORTEST_SUBSEQUENCE
    assert isolated.indices == (0, 1)
    assert isolated.subsequence[0] is isolated.subsequence[1]
    assert isolated.subsequence[0] == {"v": 1}


def test_the_result_is_a_copy_neither_side_can_mutate():
    seq = [{"v": 1}, {"v": 2}]
    result = shrink_sequence(seq, lambda s: True, max_evaluations=1)
    seq[0]["v"] = 999
    seq.append({"v": 3})
    assert result.subsequence == ({"v": 1}, {"v": 2})
    assert result.indices == (0, 1)

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

    other = {"value": 1}
    assert shared == other
    assert shared is not other

    def any_alias(candidate):
        items = list(candidate)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i] is items[j]:
                    return True
        return False

    proper = shrink_sequence([shared, shared, other], any_alias, max_evaluations=100)
    assert proper.status == SHORTEST_SUBSEQUENCE
    assert proper.indices == (0, 1)
    assert proper.evaluations == 6
    assert proper.subsequence[0] is proper.subsequence[1]
    assert any_alias(list(proper.subsequence)) is True

    inner = {"v": 1}

    def nested_alias(candidate):
        items = list(candidate)
        return len(items) == 2 and items[0]["a"] is items[1]["b"]

    nested = shrink_sequence([{"a": inner}, {"b": inner}], nested_alias, max_evaluations=100)
    assert nested.status == SHORTEST_SUBSEQUENCE
    assert nested.indices == (0, 1)
    assert nested.subsequence[0]["a"] is nested.subsequence[1]["b"]
    assert list(nested.subsequence) == [{"a": {"v": 1}}, {"b": {"v": 1}}]
    assert nested_alias(list(nested.subsequence)) is True


