"""Tests for state.py: the StateView the Router and the Simulated user read the world through."""

from __future__ import annotations

from kullback.runner.state import StateView


def test_state_view_any_value_reads_overlay_then_shared():
    """The Simulated user's reader (D77) goes through this same overlay-then-db lookup."""
    state = StateView(
        shared={"users": {"u1": {"email": "old@b.com", "zip": "10001"}}},
        overlay={"users": {"u1": {"email": "new@b.com"}}},
    )
    assert state.any_value("email") == "new@b.com"
    assert state.any_value("zip") == "10001"
    assert state.any_value("nothing") is None
    assert state.row("users", "u1") == {"email": "new@b.com", "zip": "10001"}


def test_state_view_any_value_reads_a_field_the_world_holds_nested():
    """D77: tau2 keeps a user's zip under address and the card's last four under payment_methods."""
    state = StateView(shared={"users": {"u1": {
        "address": {"city": "Denver", "zip": "80279"},
        "payment_methods": {"credit_card_1": {"source": "credit_card", "last_four": "9212"}},
    }}})
    assert state.any_value("zip") == "80279"
    assert state.any_value("last_four") == "9212"


def test_state_view_any_value_does_not_hand_over_another_rows_fact():
    """D41, D77: a fact taken from some other customer's row is an invented fact for this Task."""
    two = {"users": {"u1": {"email": "one@x.com"}, "u2": {"email": "two@x.com"}}}
    assert StateView(shared=two).any_value("email") is None
    scoped = StateView(shared=two, overlay={"users": {"u2": {"email": "two@x.com"}}})
    assert scoped.any_value("email") == "two@x.com"
