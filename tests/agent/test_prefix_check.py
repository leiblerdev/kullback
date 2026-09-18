"""The prefix property with no constant and no provider in it (G24)."""

from __future__ import annotations

from kullback.agent.prefix_check import first_prefix_break


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def said(text: str) -> dict:
    return {"role": "user", "content": text}


def replied(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": []}


def test_an_appended_conversation_passes():
    first = [system("head"), said("one")]
    second = [system("head"), said("one"), replied("two")]
    third = [*second, said("three")]
    assert first_prefix_break([first, second, third]) is None


def test_a_moved_head_fails_and_names_the_turn():
    first = [system("head"), said("one")]
    second = [system("head"), said("one"), replied("two")]
    moved = [system("head, extended"), said("one"), replied("two")]
    assert first_prefix_break([first, second, moved]) == 2


def test_reordered_history_fails_and_names_the_turn():
    first = [system("head"), said("one")]
    second = [system("head"), said("one"), replied("two")]
    reordered = [system("head"), replied("two"), said("one")]
    assert first_prefix_break([first, second, reordered]) == 2


def test_a_shortened_request_fails_and_names_the_turn():
    first = [system("head"), said("one"), replied("two")]
    cut = [system("head"), said("one")]
    assert first_prefix_break([first, cut]) == 1


def test_an_identical_request_counts_as_an_extension():
    first = [system("head"), said("one")]
    assert first_prefix_break([first, list(first)]) is None


def test_key_order_does_not_break_the_prefix():
    first = [{"role": "system", "content": "head"}, {"content": "one", "role": "user"}]
    second = [system("head"), said("one"), replied("two")]
    assert first_prefix_break([first, second]) is None


def test_no_requests_or_one_request_passes():
    assert first_prefix_break([]) is None
    assert first_prefix_break([[system("head")]]) is None
