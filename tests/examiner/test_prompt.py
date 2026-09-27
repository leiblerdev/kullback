"""The Examiner's prompt reads in GEPA order and stops last, with general examples only."""

from __future__ import annotations

from kullback.examiner import prompt as P


def test_sections_read_in_gepa_order_with_the_stop_rule_last():
    names = [name for name, _ in P.sections()]
    assert names == ["receives", "tools", "examples", "choice", "feedback", "stop"]


def test_stop_rule_ends_with_one_line_and_no_tool_call():
    text = P.stop_section()
    assert "one line and no tool call" in text


def test_stop_rule_forbids_stopping_with_an_unchecked_trusted_reference():
    text = P.stop_section()
    assert "every trusted Task's Reference is checked" in text
    assert "check_reference has been called on its Task this session" in text


def test_prompt_names_files_never_verbs_and_no_dashes():
    whole = P.render(P.sections()) + P.opening("Examine.", "rulings", ["runs/", "replays.json"])
    assert "env/tools/<name>.py" in whole
    assert "never a verb" in whole
    assert "\u2014" not in whole and "\u2013" not in whole


def test_receives_names_the_root_as_dot_and_its_first_write_dirs():
    text = P.receives_section()
    assert 'Your root is "."' in text
    assert "never a host path" in text
    assert "on your first write under them: verifiers/, probes/" in text


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


def test_the_prompt_says_a_dry_run_is_no_refusal_and_counts_only_edit_refusals():
    """Both halves of the stop rule stand in the prompt: trials are for trying, edits count."""
    assert "A dry run verdict is not a refusal" in P.tools_section()
    choice = P.choice_section()
    assert "Two refused edit_verifier proposals" in choice, choice
    assert "not a refusal and counts toward neither" in choice, choice


def test_the_intent_prompt_keeps_the_gepa_order_and_the_runs_prompt_unchanged():
    assert [name for name, _ in P.sections("intent")] == [name for name, _ in P.sections()]
    assert P.sections("runs") == P.sections()


def test_the_intent_prompt_says_the_examiner_judges_and_quotes_its_evidence():
    tools, choice = P.intent_tools_section(), P.intent_choice_section()
    assert "rule:" in tools and "in quotation marks" in tools and "is refused" in tools
    assert "You hold no edit tool" in choice
    assert "never from which side has more Runs" in choice
