"""A refusal for a wrong Reference stands even when a frontier Run finished; every other refusal keeps D128."""

from __future__ import annotations

import pytest

from gates.examiner_fixtures import TASK, replay_row, reroll_row
from kullback.gates import trust as T

FINISHED = ({TASK: {"tr1": replay_row("tr1", True)}}, {TASK: [reroll_row("reroll-t1-0", "success")]})


def test_a_reference_wrong_refusal_is_admitted_although_a_frontier_run_finished():
    ruling = T.refuse_gate({TASK: {"task_id": TASK, "reason": T.REFERENCE_WRONG}}, *FINISHED)
    assert ruling.passed and ruling.metrics == {"refused": [TASK], "rejected": []}


@pytest.mark.parametrize("reason", ["no frontier Run reaches the End state", "outcome_not_in_state", ""])
def test_a_refusal_with_any_other_reason_and_a_finished_run_is_still_rejected(reason):
    ruling = T.refuse_gate({TASK: {"task_id": TASK, "reason": reason}}, *FINISHED)
    assert not ruling.passed and ruling.metrics == {"refused": [], "rejected": [TASK]}
