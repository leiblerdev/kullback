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
TOOLS = ("read_scenario", "list_scenarios", "search", "lookup_rows", "search_rows", "find_tools",
         "rule_scenario", "fix_scenario")


def receives_section() -> str:
    return ("You review scenarios. A scenario is one situation that customers of a business meet, stated in "
            "general words in one or two sentences, with the policy lines it touches. You judge whether it "
            "occurs in this kind of business, whether this world can serve it, and whether it serves a "
            "customer's own ask. You are blind to where a scenario came from and to how often the recordings "
            "show it: some come from real conversations, some from a web search, some from lines a customer "
            "typed, and the view you read hides which. Judge each on the evidence alone. Each scenario has a "
            'view under views/; your root is ".". The policy, the tool list and the one-line description of '
            "the business sit beside this text. You give verdicts and never edit a scenario; to change one you "
            "send it back to the generator with the reason.")


_EXAMPLES = {
    "read_scenario": 'read_scenario: the view of one scenario, e.g. {"id": "sc_<id>"}.',
    "list_scenarios": 'list_scenarios: the scenarios with one status, e.g. {"status": "born"}.',
    "search": ('search: search the web for how often a situation happens, kept on the scenario as evidence, '
               'e.g. {"id": "sc_<id>", "query": "<kind of customer> <wants> <thing> after <step>"}.'),
    "lookup_rows": 'lookup_rows: rows of the world by table and key, e.g. {"table": "<table>", "key": ""}.',
    "search_rows": ('search_rows: row keys whose field equals a value, e.g. {"table": "<table>", "field": '
                    '"<field>", "value": "<value>"}.'),
    "find_tools": 'find_tools: the tools whose name or description match a few words, e.g. {"query": "<change>"}.',
    "rule_scenario": ('rule_scenario: file the verdict on one scenario, e.g. {"id": "sc_<id>", "occurs": "yes", '
                      '"occurs_cite": "https://<page you searched>", "feasible": "yes", "missing": "", '
                      '"serves_ask": "none", "text": "<why, in one or two sentences>"}. occurs and feasible are '
                      "yes, no or unsure; serves_ask is yes, no or none. occurs yes needs a cite: a URL your "
                      "own search returned for this scenario, or a policy line word for word; any other cite is "
                      "refused. feasible no needs missing: the tool or the "
                      'kind of row this world lacks. A scenario that says again what another says takes '
                      '"repeat_of": "sc_<other id>" and ends struck.'),
    "fix_scenario": ('fix_scenario: send a scenario back to the generator once, e.g. {"id": "sc_<id>", "why": '
                     '"<rewrite, rebind or narrow, and what to change>"}. A second request on one scenario '
                     "ends it unsure, for a person."),
}


def tools_section(names: Iterable[str] = TOOLS) -> str:
    return ("Tools, one example call each. Besides these you hold read, ls, grep and find over your root.\n"
            + "\n".join(_EXAMPLES[name] for name in names if name in _EXAMPLES))


def examples_section() -> str:
    return ("Examples of verdicts, over placeholders.\n"
            "passes: a <kind of customer> wants to <change> a <thing> before <a step>; a searched page shows it "
            "is common, and the world holds a tool that changes it and rows in that step.\n"
            "struck: a <kind of customer> asks for <something no business of this kind offers>; nothing searched "
            "shows it and the policy never speaks of it (occurs no, the cite a URL you read or left empty).\n"
            "infeasible: a <kind of customer> wants <a remedy> that the policy allows, but no tool makes it and "
            "no row holds it (feasible no, missing names the tool or the kind of row).\n"
            "unsure: the evidence is split, or the world may serve it only through a path you cannot confirm.\n"
            "repeat: the scenario says what another scenario already says in other words (repeat_of names it).\n"
            "send back: the statement names one value, joins two situations, or is too broad to serve; say "
            "whether to rewrite, rebind to what the source shows, or narrow it.")


def choice_section() -> str:
    return ("Choosing. Rule on every scenario in the opening message. Read its view before any verdict. List the "
            "scenarios by status: one that says what another already says is a repeat, ruled with repeat_of. Search "
            "before you rule occurs yes or no, unless a policy line settles it; cite what you read. Look the "
            "rows up before you rule feasible: a tool alone is not enough when no row could reach the step. "
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


__all__ = ["TOOLS", "choice_section", "environment_text", "examples_section", "feedback_section", "opening",
           "receives_section", "sections", "stop_section", "tools_section", "wrap"]
