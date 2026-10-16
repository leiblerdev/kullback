"""The synthetic generator's mining prompt in GEPA order: receives, tools, examples, choice, feedback,
stop (D372).

The same section order as the Examiner's prompt and the scenario review's (prompt.py). General only: no
corpus, publisher, tool, table or field name appears here, and every example is over placeholders, one
per shape. What one session mines (a batch of opening asks, the web, a batch of typed lines, or the
scenarios sent back) is said in its opening message, never in the sections, so every session reads the
same sections. The stop rule is last.
"""

from __future__ import annotations

from typing import Iterable

from kullback.synth.prompt import held_line, wrap

TOOLS = ("list_openings", "list_scenarios", "search", "read_page", "write_scenario", "rewrite_scenario")

RULE = ("A scenario is one or two sentences in the third person saying what happens, in general words: the kind "
        "of person, the kind of thing and the step it has reached. Never name a system, a tool, a button, an id, "
        "a number, a name, a place or any other value taken from a message; say the kind of thing instead.")


def receives_section() -> str:
    return ("You write scenarios for one business. " + RULE + " Each session mines one source, named in the "
            "opening message: the opening asks of real conversations, numbered; the web, searched and read; "
            "lines a person typed, numbered, each naming a situation in their own words; or scenarios a reviewer "
            "sent back with the reason. The policy, the tool list and the one-line description of the business "
            'sit beside this text. Your root is the scenario store, "."; read it, never edit it.')


_EXAMPLES = {
    "list_openings": "list_openings: the numbered items of this session, opening asks or typed lines, e.g. {}.",
    "list_scenarios": "list_scenarios: the scenarios already written, so you never write one twice, e.g. {}.",
    "write_scenario": ('write_scenario: write one scenario, e.g. {"statement": "<the scenario>", "source": "<the '
                       'source of this session>", "policy_lines": ["<a policy line, word for word>"], "openings": '
                       '[0], "url": "", "span": ""}. For opening asks list every number it groups; for typed lines '
                       "the one number it rewrites; for a page the URL you read and the span of it the scenario rests "
                       "on, word for word."),
    "rewrite_scenario": ('rewrite_scenario: rewrite one scenario sent back, e.g. {"id": "sc_<id>", "statement": '
                         '"<the scenario, rewritten>"}.'),
}


def tools_section(names: Iterable[str] = TOOLS) -> str:
    names = list(names)
    return held_line(names) + "\n" + "\n".join(_EXAMPLES[name] for name in names if name in _EXAMPLES)


def examples_section() -> str:
    return ("Examples of scenarios, over placeholders, one per shape.\n"
            "a <kind of customer> wants to <change> a <thing> that has already <reached a later step>\n"
            "a <kind of customer> asks for <something> that the rules allow only <under a condition>\n"
            "a <kind of customer> reports that <a thing> <went wrong> and wants <a remedy>")


def choice_section() -> str:
    return ("Choosing. List the scenarios already written before you write, and write none twice. Write one "
            "scenario per distinct situation. Quote a policy line only word for word. From opening asks, group "
            "the asks that share a situation into one scenario. From the web, search for what customers of this "
            "kind of business ask for, how its workflows go and what goes wrong, read the pages, and write only "
            "what a page shows, with the span it rests on; never search for an academic paper, a benchmark, a "
            "dataset or a code repository. From typed lines, write each line as a scenario of the situation it "
            "names, never the line as typed. From scenarios sent back, follow the reason: rewrite the words, "
            "rebind to what the source shows, or narrow it.")


def feedback_section() -> str:
    return ("Feedback. A written scenario answers with its id. A refused write says why: a tool name, an id shaped "
            "value, a policy line the policy does not hold, a span the page does not hold, a line copied as typed. "
            "Write it again once, following the reason; a second refusal of the same scenario drops it.")


def stop_section() -> str:
    return ("Stopping. Answer with one line and no tool call when every item of this session is written or "
            "dropped. Say how many you wrote.")


def sections(names: Iterable[str] = TOOLS) -> list[tuple[str, str]]:
    """Every section in GEPA order, the stop rule last."""
    return [(name, wrap(text)) for name, text in (
        ("receives", receives_section()), ("tools", tools_section(names)), ("examples", examples_section()),
        ("choice", choice_section()), ("feedback", feedback_section()), ("stop", stop_section()))]


OPENINGS = {
    "recordings": "Mine this batch of opening asks of real conversations: source recordings, {count} items.",
    "ask": "Mine this batch of lines a person typed: source ask, {count} items.",
    "search": "Mine the web for this business: source search. Search, read the pages, write what they show.",
    "fix": "Rewrite the scenarios a reviewer sent back, each with its reason and what it was written from:\n{items}",
}


def opening(kind: str, entries: Iterable[str] = (), count: int = 0, items: str = "") -> str:
    """The opening message of one session: what it mines and what the root holds."""
    listed = ", ".join(entries) or "nothing yet"
    return wrap(OPENINGS[kind].format(count=count, items=items) + f"\n\nThe root holds: {listed}.")


__all__ = ["RULE", "TOOLS", "opening", "sections"]
