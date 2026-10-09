"""The set-aside rule the trust gates read: the mark, or the writer-round cap with a blocking ruling open (D331, D350)."""

from __future__ import annotations

from kullback.spec import router as R
from kullback.spec.rulings import Ruling
from tests.spec.fixtures import spec


def test_set_aside_is_the_mark_or_the_round_cap_with_a_ruling_open():
    assert R.is_set_aside(spec().model_copy(update={"set_aside": R.NO_AGREEMENT}))
    assert R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 1}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 0}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP - 1, "rulings_open": 1}))


def test_the_bus_driven_router_is_retired():
    assert not hasattr(R, "Router") and R.ROUNDS_CAP == 3


def _ruling(number, answered_in=None):
    answer = {"action": "rebutted", "why": "the user asked", "round": answered_in} if answered_in else None
    return Ruling(task_id="t1", number=number, round=1, item="s1", kind="derivation", code="unasked_item",
                  reason="the user never asked for this", fix="drop it", blocking=True, answer=answer)


def test_the_cap_counts_writer_rounds_so_an_examiner_only_round_leaves_the_writer_a_round():
    examined_thrice = spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 1})
    one_answer = [_ruling(1, answered_in=2), _ruling(2, answered_in=2), _ruling(3)]
    assert R.writer_rounds(one_answer) == 2 and not R.is_set_aside(examined_thrice, one_answer)
    two_answers = one_answer + [_ruling(4, answered_in=4)]
    assert R.writer_rounds(two_answers) == R.ROUNDS_CAP and R.is_set_aside(examined_thrice, two_answers)
