from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

NOT_FAILING = "not_failing"
SHORTEST_SUBSEQUENCE = "shortest_subsequence"
BUDGET_LIMITED = "budget_limited"


@dataclass(frozen=True)
class ShrinkResult:
    subsequence: tuple
    indices: tuple
    evaluations: int
    status: str


def _check_budget(max_evaluations: int) -> None:
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise TypeError("max_evaluations must be an int")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")


def _detached(items: list) -> tuple:
    return tuple(copy.deepcopy(items))


def _candidate(snapshot: list, indices: tuple) -> list:
    return copy.deepcopy([snapshot[i] for i in indices])


def shrink_sequence(
    sequence: Sequence[Any],
    fails: Callable[[list[Any]], Any],
    *,
    max_evaluations: int,
) -> ShrinkResult:
    _check_budget(max_evaluations)
    if not callable(fails):
        raise TypeError("fails must be callable")
    snapshot = copy.deepcopy(list(sequence))
    total = len(snapshot)
    full = tuple(range(total))
    evaluations = 1
    if not fails(_candidate(snapshot, full)):
        return ShrinkResult(
            subsequence=_detached(snapshot),
            indices=full,
            evaluations=evaluations,
            status=NOT_FAILING,
        )
    best = full
    for length in range(total):
        for combo in itertools.combinations(range(total), length):
            if evaluations >= max_evaluations:
                return ShrinkResult(
                    subsequence=_detached([snapshot[i] for i in best]),
                    indices=best,
                    evaluations=evaluations,
                    status=BUDGET_LIMITED,
                )
            evaluations += 1
            if fails(_candidate(snapshot, combo)):
                return ShrinkResult(
                    subsequence=_detached([snapshot[i] for i in combo]),
                    indices=combo,
                    evaluations=evaluations,
                    status=SHORTEST_SUBSEQUENCE,
                )
    return ShrinkResult(
        subsequence=_detached([snapshot[i] for i in best]),
        indices=best,
        evaluations=evaluations,
        status=SHORTEST_SUBSEQUENCE,
    )
