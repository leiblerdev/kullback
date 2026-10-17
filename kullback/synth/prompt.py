"""The Examiner's scenario review prompt in GEPA order: receives, tools, examples, choice, feedback,
stop (D373).

The same section order as the Examiner's own prompt (examiner/prompt.py). General only: no corpus,
publisher, tool, table or field name appears here, and every example is over placeholders, one per
outcome. The review is blind: the receives section says it never sees where a scenario came from or how
often the recordings show it. The stop rule is last.
"""

from __future__ import annotations

import textwrap
from typing import Iterable

from kullback.synth.skills import SCENARIO_SKILL

# Every prompt line is folded under this many characters.
WIDTH = 140
TOOLS = ("read_scenario", "list_scenarios", "search", "read_page", "lookup_rows", "search_rows", "find_tools",
         "rule_scenario", "fix_scenario")
# The shared web tools (D377): named here by name only, their example calls are in their descriptions.
WEB = ("search", "read_page")


def receives_section() -> str:
    return ("You review scenarios. A scenario is one situation that one customer of a business meets, in two to "
            "four sentences, with the rows of the world it is bound to, what the world lacks for it (requires) "
            "and the policy lines it touches. You judge whether it occurs in this kind of business and whether "
            "this world can serve it. You are blind to where a scenario came from and to how often the recordings "
            "show it: some come from real conversations, some from a web search, some from lines a customer "
            "typed, and the view you read hides which. Judge each on the evidence alone. Each scenario has a "
            'view under views/; your root is ".". The policy, the tool list and the one-line description of '
            "the business sit beside this text. You give verdicts and never edit a scenario; to change one you "
            "send it back to the generator with the reason.")


_EXAMPLES = {
    "read_scenario": 'read_scenario: the view of one scenario, e.g. {"id": "sc_<id>"}.',
    "list_scenarios": 'list_scenarios: the scenarios with one status, e.g. {"status": "born"}.',
    "lookup_rows": 'lookup_rows: rows of the world by table and key, e.g. {"table": "<table>", "key": ""}.',
    "search_rows": ('search_rows: row keys whose field equals a value, e.g. {"table": "<table>", "field": '
                    '"<field>", "value": "<value>"}.'),
    "find_tools": 'find_tools: the tools whose name or description match a few words, e.g. {"query": "<change>"}.',
    "rule_scenario": ('rule_scenario: file the verdict on one scenario, e.g. {"id": "sc_<id>", "occurs": "yes", '
                      '"occurs_url": "https://<page you read>", "occurs_span": "<its words, copied exactly>", '
                      '"feasible": "yes", "checked": ["tool:<tool name>", "row:<table>:<row key>"], "missing": "", '
                      '"text": "<why, in one or two sentences>"}. occurs and feasible are yes, no or unsure. occurs '
                      "yes needs a page you read with read_page in this session and a span of its text; a search "
                      "result alone is no cite, and a policy line is no cite. feasible yes needs checked: each "
                      "tool it needs and each bound row you looked up; a scenario whose requires is not empty is "
                      "never feasible yes. feasible no needs missing: the tool or the kind of row this world "
                      'lacks. A scenario that says again what another says takes "repeat_of": "sc_<other id>" '
                      "and ends struck."),
    "fix_scenario": ('fix_scenario: send a scenario back to the generator once, e.g. {"id": "sc_<id>", "quote": '
                     '"<the wrong words of the statement, copied exactly>", "why": "<why they are wrong>", '
                     '"change": "<what they should say instead, or what to look up>"}. A second request on one '
                     "scenario ends it unsure, for a person."),
}


def held_line(names: Iterable[str]) -> str:
    """The tools a session holds besides its own: the base tools, and the shared web tools where it has them."""
    web = [name for name in names if name in WEB]
    shared = f"; and the web tools {' and '.join(web)}, each with its example call in its description" if web else ""
    return f"Tools, one example call each. Besides these you hold read, ls, grep and find over your root{shared}."


def tools_section(names: Iterable[str] = TOOLS) -> str:
    names = list(names)
    return held_line(names) + "\n" + "\n".join(_EXAMPLES[name] for name in names if name in _EXAMPLES)


def examples_section() -> str:
    return ("Examples of verdicts, over placeholders.\n"
            "passes: <a customer> wants to <change> a <thing> before <a step>; a page you read says customers "
            "ask this (its span cited), the bound rows are in that step, and a tool changes it (checked names both).\n"
            "struck: <a customer> asks for <something no business of this kind offers>; nothing you read shows "
            "it (occurs no, with the page and span you read, or with nothing).\n"
            "infeasible: <a customer> wants <a remedy> the policy allows, but no tool makes it and no row holds "
            "it, or requires names what is missing (feasible no, missing names the tool or the kind of row).\n"
            "unsure: the evidence is split, or the world may serve it only through a path you cannot confirm.\n"
            "repeat: the scenario says what another scenario already says in other words (repeat_of names it).\n"
            "send back: the words say a value the bound rows do not hold, or join two situations; quote those "
            "words, say why, and say what they should say instead or what to look up.")


def choice_section() -> str:
    return ("Choosing. Rule on every scenario in the opening message. Read its view before any verdict. List the "
            "scenarios by status: one that says what another already says is a repeat, ruled with repeat_of. Search, "
            "then read the page before you cite it; cite only words of a page you read. Look the bound rows up "
            "and find the tools before you rule feasible, and name the rows and tools you checked: a tool alone "
            "is not enough when no row could reach the step. A send back quotes the words. "
            "Send a scenario back only when its words, not the world, are what is wrong; a scenario this world "
            "cannot serve is a verdict of feasible no, never a send back and never a drop. Rule unsure only "
            "when the evidence leaves the question open; a person reads every unsure.\n" + SCENARIO_SKILL)


def feedback_section() -> str:
    return ("Feedback. A filed verdict answers with what follows from it: occurs yes and feasible yes passes; "
            "any unsure goes to a person; feasible no files a finding for the Builder naming what is missing; "
            "occurs no strikes the scenario. A refused call says what was missing. A scenario sent back comes "
            "back rewritten in a later session.")


def stop_section() -> str:
    return ("Stopping. Answer with one line and no tool call when every scenario in the opening message has a "
            "verdict or was sent back. Say how many you ruled per outcome.")


def wrap(text: str, width: int = WIDTH) -> str:
    """The text with every line folded under `width` characters, words kept whole."""
    return "\n".join(textwrap.fill(line, width, break_long_words=False, break_on_hyphens=False) or line
                     for line in str(text).splitlines())


def sections(names: Iterable[str] = TOOLS) -> list[tuple[str, str]]:
    """Every section in GEPA order, the stop rule last. The same text in every session."""
    return [(name, wrap(text)) for name, text in (
        ("receives", receives_section()), ("tools", tools_section(names)), ("examples", examples_section()),
        ("choice", choice_section()), ("feedback", feedback_section()), ("stop", stop_section()))]


def environment_text(policy: str, tools: Iterable[dict], description: str) -> str:
    """The policy whole (D342), the tool list and the one-line description, as one section."""
    listed = "\n".join(f"- {row.get('name')} ({row.get('kind')}): {row.get('description') or 'no description'}"
                       for row in tools)
    return (f"The business, in one line: {description or 'not given'}\n\nThe tools:\n{listed or 'none recorded'}"
            f"\n\nThe policy (the whole text).\n{policy or 'no policy recorded'}")


def opening(lines: str, entries: Iterable[str] = ()) -> str:
    """The opening message: the ask, what the root holds, the scenarios with their view paths."""
    listed = ", ".join(entries) or "nothing yet"
    return (f"Review these scenarios.\n\nThe root holds: {listed}.\n"
            f"The scenarios, each with its view; open those paths rather than searching:\n{lines}")


__all__ = ["TOOLS", "WEB", "choice_section", "environment_text", "examples_section", "feedback_section", "held_line", "opening",
           "receives_section", "sections", "stop_section", "tools_section", "wrap"]
