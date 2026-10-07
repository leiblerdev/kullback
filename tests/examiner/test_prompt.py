"""The Examiner's prompt reads in GEPA order and stops last, with general examples only."""

from __future__ import annotations

from kullback.examiner import prompt as P


def test_sections_read_in_gepa_order_with_the_stop_rule_last():
    names = [name for name, _ in P.sections()]
    assert names == ["receives", "tools", "examples", "choice", "feedback", "stop"]


def test_stop_rule_ends_with_one_line_and_no_tool_call():
    text = P.stop_section()
    assert "one line and no tool call" in text


def test_stop_rule_ends_when_every_task_is_examined_and_every_answer_ruled():
    text = P.stop_section()
    assert "every Task is examined and every answered ruling is closed or kept open" in text


def test_prompt_names_files_with_finding_and_has_no_dashes():
    from kullback.examiner.domain_tools import TOOL_NAMES

    whole = P.render(P.sections(TOOL_NAMES)) + P.opening("Examine.", "rulings", ["runs/", "replays.json"])
    assert "env/tools/<name>.py" in whole
    assert "\u2014" not in whole and "\u2013" not in whole


def test_receives_names_the_root_as_dot_and_says_the_examiner_never_edits():
    text = P.receives_section()
    assert 'Your root is "."' in text
    assert "You give feedback and never edit" in text
    assert P.WRITABLE_DIRS == ()


def test_the_opening_lists_the_root_entries_and_the_rulings_after_the_ask():
    text = P.opening("Examine.", "t1: runs: runs/t1/a.jsonl", ["history.json", "runs/", "tasks/"])
    assert text.startswith("Examine.\n\n")
    assert "The root holds: history.json, runs/, tasks/." in text
    assert "t1: runs: runs/t1/a.jsonl" in text


def test_the_system_prompt_is_the_same_whatever_the_rulings_and_the_root_hold():
    """The rulings and the root listing differ per session; in the system prompt they re-wrote the
    whole cached prefix of every new Examiner session, so they live in the opening message."""
    first = P.opening("Examine.", "t1: runs: a", ["runs/"])
    second = P.opening("Examine.", "t2: runs: b", ["runs/", "tasks/"])
    assert first != second
    assert all("t1: runs" not in text and "runs/, tasks/" not in text for _, text in P.sections())


def test_the_prompt_asks_for_rulings_with_a_fix_and_world_lookups_of_every_anchor():
    choice, tools = P.choice_section(), P.tools_section()
    assert "Look up in the world every anchor" in choice and "never from what a Run did" in choice
    assert '"fix":' in tools and '"blocking": true' in tools
    assert "finding:" not in tools and "no_finding:" not in tools


def test_round_two_asks_to_close_or_keep_open_and_round_one_does_not():
    assert "Close each answered ruling" in P.opening("Examine.", "t1", round_number=2)
    assert "Close each answered ruling" not in P.opening("Examine.", "t1", round_number=1)


def test_the_finding_tool_takes_body_edits_only():
    from kullback.examiner.domain_tools import FindingArgs
    from kullback.examiner.exam_files import EDIT_KINDS

    described = FindingArgs.model_fields["edits"].description
    assert EDIT_KINDS == ("body",) and "kind: body," in described and "ruling" in described
