"""The prefix property with no constant and no provider in it (G24)."""

from __future__ import annotations

import pytest

from kullback.agent.prefix_check import first_prefix_break


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def said(text: str) -> dict:
    return {"role": "user", "content": text}


def replied(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": []}


@pytest.mark.parametrize(
    "requests",
    [
        pytest.param(
            [
                [system("head"), said("one")],
                [system("head"), said("one"), replied("two")],
                [system("head"), said("one"), replied("two"), said("three")],
            ],
            id="appended",
        ),
        pytest.param([[system("head"), said("one")], [system("head"), said("one")]], id="identical"),
        pytest.param(
            [
                [{"role": "system", "content": "head"}, {"content": "one", "role": "user"}],
                [system("head"), said("one"), replied("two")],
            ],
            id="key-order",
        ),
        pytest.param([], id="no-requests"),
        pytest.param([[system("head")]], id="one-request"),
    ],
)
def test_an_appended_conversation_passes(requests):
    assert first_prefix_break(requests) is None


@pytest.mark.parametrize(
    ("requests", "turn"),
    [
        pytest.param(
            [
                [system("head"), said("one")],
                [system("head"), said("one"), replied("two")],
                [system("head, extended"), said("one"), replied("two")],
            ],
            2,
            id="moved-head",
        ),
        pytest.param(
            [
                [system("head"), said("one")],
                [system("head"), said("one"), replied("two")],
                [system("head"), replied("two"), said("one")],
            ],
            2,
            id="reordered",
        ),
        pytest.param(
            [[system("head"), said("one"), replied("two")], [system("head"), said("one")]], 1, id="shortened"
        ),
    ],
)
def test_a_broken_prefix_fails_and_names_the_turn(requests, turn):
    assert first_prefix_break(requests) == turn
