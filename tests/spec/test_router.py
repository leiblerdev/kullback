"""The set-aside rule the trust gates read: the mark, or the round cap with a blocking ruling open (D331)."""

from __future__ import annotations

from kullback.spec import router as R
from tests.spec.fixtures import spec


def test_set_aside_is_the_mark_or_the_round_cap_with_a_ruling_open():
    assert R.is_set_aside(spec().model_copy(update={"set_aside": R.NO_AGREEMENT}))
    assert R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 1}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP, "rulings_open": 0}))
    assert not R.is_set_aside(spec().model_copy(update={"round": R.ROUNDS_CAP - 1, "rulings_open": 1}))


def test_the_bus_driven_router_is_retired():
    assert not hasattr(R, "Router") and R.ROUNDS_CAP == 2
