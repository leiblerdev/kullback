"""The Builder's prompt: GEPA order, general examples, the stop rule last."""

from __future__ import annotations

from kullback.builder import prompt as prompt_mod

FORBIDDEN = ("corpus", "publisher", "bench")


def test_sections_are_in_gepa_order_with_the_stop_rule_last():
    names = [name for name, _ in prompt_mod.sections()]
    assert names == ["receive", "tools", "examples", "choice", "feedback", "stop"]
    assert names[-1] == "stop"


def test_every_tool_has_one_example_call():
    tools = prompt_mod.TOOLS
    for name in ("read", "write", "edit", "bash", "grep", "find", "ls", "web_search",
                 "ingest", "derive_world", "grow", "replay", "run", "status", "examine"):
        assert f"{name}:" in tools


def test_sections_name_no_corpus():
    text = " ".join(text for _, text in prompt_mod.sections()).lower()
    for word in FORBIDDEN:
        assert word not in text


def test_sections_hold_no_dash_outside_ascii():
    for _, text in prompt_mod.sections():
        assert "\u2014" not in text and "\u2013" not in text


def test_findings_name_a_path_never_a_verb():
    assert "never names a verb" in prompt_mod.EXAMPLES
    assert "path" in prompt_mod.FEEDBACK and "change" in prompt_mod.FEEDBACK


def test_the_stop_rule_ends_a_stalled_line_of_work():
    assert "two consecutive edits" in prompt_mod.STOP
    assert "what remains" in prompt_mod.STOP
