"""The Builder's prompt: GEPA order, general examples, the stop rule last."""

from __future__ import annotations

from kullback.builder import prompt as prompt_mod

FORBIDDEN = ("corpus", "publisher", "bench")


def test_sections_are_in_gepa_order_with_the_stop_rule_last():
    names = [name for name, _ in prompt_mod.sections()]
    assert names == ["receive", "tools", "examples", "choice", "feedback", "stop"]
    assert names[-1] == "stop"


def _builder_tool_names(tmp_path):
    from kullback.builder.domain_tools import domain_tools
    from kullback.builder.session import BUILDER_BASE_TOOLS

    return set(BUILDER_BASE_TOOLS) | {tool.name for tool in domain_tools(workdir=tmp_path)}


def test_every_tool_has_one_example_call(tmp_path):
    lines = prompt_mod.TOOLS.splitlines()[1:]
    named = [line.split(":", 1)[0] for line in lines]
    assert sorted(named) == sorted(_builder_tool_names(tmp_path))


def test_the_builder_is_not_handed_web_search_or_shown_it(tmp_path):
    assert "web_search" not in _builder_tool_names(tmp_path)
    assert "web_search" not in prompt_mod.TOOLS


def test_the_examples_name_the_refusal_sentence_and_the_body_fault():
    examples = prompt_mod.EXAMPLES
    assert "raise ValueError(" in examples
    assert "word for word" in examples
    assert "the replay compares that message" in examples
    assert "KeyError or an IndexError is a body fault at its line, never a refusal" in examples


def test_the_examples_show_rulings_and_inspect_before_one_edit():
    examples = prompt_mod.EXAMPLES
    assert "Call rulings on the tool file to see every failing row grouped" in examples
    assert "inspect the large file for its shape before reading it" in examples
    assert "one hypothesis and make one edit" in examples


def test_the_choice_rule_names_valueerror_refusals_inspect_and_rulings():
    choice = prompt_mod.CHOICE
    assert "only with ValueError and the recorded message" in choice
    assert "Use inspect for the shape of a large file" in choice
    assert "Use rulings to see every failing row at once instead of one write per guess" in choice


def test_feedback_says_the_write_result_names_body_faults_and_differing_refusals():
    feedback = prompt_mod.FEEDBACK
    assert "body fault with its line" in feedback
    assert "refusal whose message differs" in feedback


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
