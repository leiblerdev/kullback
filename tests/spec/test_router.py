"""The set-aside rule the trust gates read: the mark, or the writer-round cap with a blocking ruling open (D331, D350)."""

from __future__ import annotations

import pytest

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


def test_the_round_cap_setting_moves_where_an_open_ruling_sets_the_task_aside(monkeypatch):
    three_rounds = spec().model_copy(update={"round": 3, "rulings_open": 1})
    four_rounds = spec().model_copy(update={"round": 4, "rulings_open": 1})
    assert R.rounds_cap() == R.ROUNDS_CAP and R.is_set_aside(three_rounds)
    monkeypatch.setenv(R.ROUNDS_CAP_ENV, "4")
    assert R.rounds_cap() == 4
    assert not R.is_set_aside(three_rounds) and R.is_set_aside(four_rounds)


def test_a_round_cap_setting_that_is_not_a_whole_number_of_at_least_one_is_refused(monkeypatch):
    for raw in ("0", "-2", "four", "3.5"):
        monkeypatch.setenv(R.ROUNDS_CAP_ENV, raw)
        with pytest.raises(ValueError):
            R.rounds_cap()


def test_a_ruling_kept_open_and_answered_again_counts_every_round_it_was_answered_in(tmp_path):
    from kullback.spec import rulings as RL

    RL.save_ruling(tmp_path, _ruling(1))
    for number in (2, 3, 4):
        answered = RL.answer(tmp_path, "t1", 1, "rebutted", "The policy asks it.", number)
    assert RL.answer_rounds(answered) == [2, 3, 4] and R.writer_rounds([answered]) == 4
