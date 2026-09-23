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


def test_receive_says_the_bodies_start_as_stubs_written_from_their_calls():
    assert "stubs under tools/" in prompt_mod.RECEIVE
    assert "NotImplementedError" in prompt_mod.RECEIVE
    assert "calls/<tool>.jsonl" in prompt_mod.RECEIVE


def test_receive_points_at_the_tool_file_header_and_names_every_provided_helper():
    from kullback.gates.confinement import PROVIDED_HELPERS

    assert "header of every tool file names the imports and helpers" in prompt_mod.RECEIVE
    assert "names the confinement gate refuses" in prompt_mod.RECEIVE
    assert all(helper in prompt_mod.RECEIVE for helper in PROVIDED_HELPERS)


def test_choice_orders_the_loop_replay_then_examine_then_run():
    choice = prompt_mod.CHOICE
    replay, examine, run = (choice.index(word) for word in ("replay until", "examine to derive", "run to play"))
    assert replay < examine < run
    assert "examine again for findings" in choice


def test_the_replay_and_run_examples_show_the_every_task_form():
    assert "or {} to replay every Task" in prompt_mod.TOOLS
    assert "or {} for every open Task with a confirmed Reference" in prompt_mod.TOOLS
