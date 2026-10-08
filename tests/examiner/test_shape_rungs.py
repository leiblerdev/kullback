"""Which shape rung a Reference stores when a loose and a tight source both hold on it.

`no_column` holds on any row that lacks the column and then accepts any value; `prior_result` asks
that the value was read off an earlier result. The locker room of test_derive_provenance.py again:
locker L7 has no row in the Starting state, so the write names a row without the column.
"""

from __future__ import annotations

import ast
import re

import pytest

from kullback import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules
from tests.test_derive_provenance import (
    ASSIGN,
    ROSTER,
    WRITE_TOOLS,
    assignment_events,
    atom,
    call,
    derive,
    result,
    run,
    says,
    user,
)


def new_locker_events(badge: str = "QX41", read: bool = True) -> list[dict]:
    """The desk reads L9's badge code and assigns it to L7, a locker the Starting state has no row for."""
    reading = [call(ROSTER, {"locker_id": "L9"}, kind="read", cid="c0"),
               result({"locker_id": "L9", "badge_code": "QX41"}, cid="c0")] if read else []
    return [user("Please assign the new locker L7 to me, under the badge code of locker L9.")] + reading + [
        call(ASSIGN, {"locker_id": "L7", "badge_code": badge}, cid="c1"),
        result({"locker_id": "L7", "assigned": True}, cid="c1"),
        says("Locker L7 is yours."),
    ]


def stored(order: str, monkeypatch, events: list[dict]) -> tuple[str, tuple]:
    """The source that answered on the Reference, and the sources the stored rule accepts."""
    monkeypatch.setattr(V, "SHAPE_RUNG_ORDER", order)
    shape = atom(derive(run("ref", events)), "w0.badge_code")
    sources = re.search(r"^_SOURCES = (.*)$", shape.predicate_src, re.M).group(1)
    return S.atom_payload(shape).get("shape_source"), ast.literal_eval(sources)


def passes_a_value_never_read(order: str, monkeypatch) -> bool:
    monkeypatch.setattr(V, "SHAPE_RUNG_ORDER", order)
    verifier = derive(run("ref", new_locker_events()))
    unread = run("other", new_locker_events(badge="ZZ99", read=False))
    return S.check_run(verifier, unread, CanonRules(), write_tools=WRITE_TOOLS)[0]


def test_today_s_order_stores_the_loose_rung_that_passes_a_value_never_read(monkeypatch):
    assert stored("row_first", monkeypatch, new_locker_events()) == ("no_column", ("own_row", "no_column"))
    assert passes_a_value_never_read("row_first", monkeypatch) is True


@pytest.mark.parametrize("order", ["tightest_first", "tightest_first_strict"])
def test_tightest_first_stores_the_read_rung_and_turns_a_value_never_read_away(monkeypatch, order):
    assert stored(order, monkeypatch, new_locker_events()) == ("prior_result", ("own_row", "prior_result"))
    assert passes_a_value_never_read(order, monkeypatch) is False


def test_tightest_first_keeps_the_row_rule_where_the_row_holds_the_value(monkeypatch):
    """D190's row rule is untouched: its own row answers first and the row and its fallback are one demand."""
    assert stored("tightest_first", monkeypatch, assignment_events()) == ("own_row", ("own_row", "no_column"))


def test_the_strict_order_stores_no_loose_rung_while_the_read_also_holds(monkeypatch):
    """The row answered, but the value was also read first, so the stored rule demands the read."""
    assert stored("tightest_first_strict", monkeypatch, assignment_events()) == ("own_row", ("own_row", "prior_result"))


@pytest.mark.parametrize("order", sorted(V.SHAPE_RUNG_ORDERS))
def test_every_order_still_stores_the_loose_rung_where_nothing_tighter_holds(monkeypatch, order):
    """A value a result held only under another column has only the loose rung to stand on."""
    events = new_locker_events()
    events[2] = result({"locker_id": "L9", "spare_badge": "QX41"}, cid="c0")
    assert stored(order, monkeypatch, events) == ("no_column", ("own_row", "no_column"))


def test_the_default_order_asks_the_tightest_source_first():
    assert V.SHAPE_RUNG_ORDER == "tightest_first"
