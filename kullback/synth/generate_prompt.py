"""The scenario generator prompt in GEPA order: receives, tools, examples, choice, feedback,
stop (D372).

The same section order as the Examiner's prompt and the scenario review's (prompt.py). General only: no
corpus, publisher, tool, table or field name appears here, and every example is over placeholders, one
per shape. What one session generates from (a batch of opening asks, the web, a batch of typed lines, or the
scenarios sent back) is said in its opening message, never in the sections, so every session reads the
same sections. The stop rule is last.
"""

from __future__ import annotations

from typing import Iterable

from kullback.synth.prompt import _EXAMPLES as _WORLD_EXAMPLES
from kullback.synth.prompt import held_line, wrap

TOOLS = ("list_openings", "list_scenarios", "search", "read_page", "lookup_rows", "search_rows", "find_tools",
         "write_scenario", "rewrite_scenario")

RULE = ("A scenario is two to four sentences in the third person saying exactly what happens to one customer of "
        "this world: who they are as the bound rows name them, what they have (the order or reservation, the "
        "product with its options, the amounts, the dates, the places), what they ask, and what they bring with "
        "them (what they know, what they get wrong, what they want instead). Every fact comes from a row you "
        "looked up and bound, or from the policy; say the values, never an id and never a tool or system name, "
        "and write a date in words. Bind to rows that fit the situation, whichever customer holds them. Write "
        "requires only when a search shows no row of the world fits; then say in words what the world lacks, and "
        "the scenario is still written. Customers and rows used in earlier sessions may be used again; only the "
        "scenarios of this session need different ones, and list_scenarios marks them. Beside the statement write two parts apart. The ask is what "
        "the customer says to open the conversation, in the first person and their own words: who they are as the "
        "bound rows name them, what they have, what they want and what they bring with them, including what they get "
        "wrong; never what the agent will do and never how it ends. The end is what a correct agent leaves the world "
        "and the customer with, in the third person: the rows changed and their new values, or that nothing changed "
        "and why, quoting the policy line. The ask and the end share no sentence.")


def receives_section() -> str:
    return ("You write scenarios for one business. " + RULE + " Each session generates from one source, named in the "
            "opening message: situations of the plan, each with the opening asks of real conversations that show "
            "it and its variants, numbered; the web, searched and read; lines a person typed, numbered, each "
            "naming a situation in their own words; gaps in the store, numbered, each a kind of situation "
            "and a table no scenario is bound to yet or a policy line no scenario turns on; or scenarios a reviewer "
            "sent back with the reason. The policy, "
            "the tool list and the one-line description of the business sit beside this text. Your root is the "
            'scenario store, "."; read it, never edit it.')


_EXAMPLES = {
    "list_openings": "list_openings: the numbered items of this session, situations or typed lines, e.g. {}.",
    "list_scenarios": ("list_scenarios: the scenarios already written, so you never write one twice; this_session "
                       "marks the ones written in this session, e.g. {}."),
    **{name: _WORLD_EXAMPLES[name] for name in ("lookup_rows", "search_rows", "find_tools")},
    "write_scenario": ('write_scenario: write one scenario, e.g. {"statement": "<the scenario>", "source": "<the '
                       'source of this session>", "situation": "si_<id>", "variant": "", "binding": {"<table>": '
                       '"<row key>"}, "requires": [], "policy_lines": ["<a policy line, word for word>"], '
                       '"openings": [], "url": "", "span": "", "ask": "<the opening words, first person>", '
                       '"end": "<what a correct agent leaves behind>"}. For a situation give its id and the variant line '
                       "word for word, or empty for the situation itself; for a typed line the one number it "
                       "rewrites; for a gap the one number it fills and, for a policy line, the kind in one word; "
                       "for a page the URL you read and the span it rests on, word for word. binding "
                       "names each row you looked up by table and key; requires says in words what the world "
                       "lacks."),
    "rewrite_scenario": ('rewrite_scenario: rewrite one scenario sent back, e.g. {"id": "sc_<id>", "statement": '
                         '"<the scenario, rewritten>", "binding": {"<table>": "<row key>"}, "ask": "<the opening '
                         'words>", "end": "<what a correct agent leaves behind>"}; leave binding, requires, ask or '
                         "end out to keep them."),
}


def tools_section(names: Iterable[str] = TOOLS) -> str:
    names = list(names)
    return held_line(names) + "\n" + "\n".join(_EXAMPLES[name] for name in names if name in _EXAMPLES)


def examples_section() -> str:
    return ("Examples of scenarios, over placeholders, one per shape, each with its ask and its end.\n"
            "<name from the bound row> bought <the product with its option values> on <the date> for <the amount>. "
            "It has already <reached a later step>, and they want to <change> it to <the other option>. They do "
            "not know the rules allow that only before <the step>.\n"
            "ask: Hi, I am <name>. I bought <the product> on <the date> and I want it changed to <the other "
            "option>, today if you can.\n"
            "end: Nothing changed, since <the policy line, word for word>; <name> was told so and offered <what "
            "the policy allows>.\n"
            "<name from the bound row> holds <the booking with its places and dates> and asks for <a remedy> the "
            "policy promises; the world holds no way to make it, so requires says <what is missing>.\n"
            "ask: I have <the booking> from <the place> on <the date>, and I was promised <the remedy>.\n"
            "end: <name> is told <the remedy> is due under <the policy line>; no row changed, since the world "
            "holds no way to make it.\n"
            "<name from the bound row> reports that <the item with its options> <went wrong> and gives <a wrong "
            "detail> at first; they want <a remedy> worth <the amount>.\n"
            "ask: My <item> <went wrong>. It was the <wrong detail> one, and I want <the remedy>.\n"
            "end: <the row> now holds <the new value>, and <name> has <the remedy> worth <the amount>.")


def choice_section() -> str:
    return ("Choosing. List the scenarios already written before you write, and write none twice. Look the rows "
            "up before you write and bind every row the scenario names. Write each scenario as soon as its rows "
            "are looked up; never look everything up first. Quote a policy line only word for word. "
            "From situations of the plan, write one scenario for each situation and one for each of its variants. "
            "From the web, search for what customers of this kind of business ask for, how its workflows go and "
            "what goes wrong, read the pages, and write only what a page shows, with the span it rests on, bound "
            "to rows that fit it; never search for an academic paper, a benchmark, a dataset or a code repository. "
            "From typed lines, write each line as a scenario of the situation it names, never the line as typed. "
            "From gaps, write one specific scenario per gap that could reach this business's support in the real "
            "world, bound to real rows you looked up, quoting the policy line a line gap names; where the world "
            "holds no rows for it, write it with requires naming what is missing. A search result says how many "
            "keys match and how many it does not show, and a key a table does not hold is answered with the "
            "table's key shape and sample keys: search by a field rather than guess a key. "
            "From scenarios sent back, follow the reason: change the quoted words as asked, keep the binding or "
            "change it to rows that fit.")


def feedback_section() -> str:
    return ("Feedback. A written scenario answers with its id. A refused write says why: a tool name, an id shaped "
            "value (naming the word that matched; a value of a bound row is no id), a policy line the policy does "
            "not hold (quoting the nearest policy line), a span the page does not hold (quoting the nearest line), a line copied as typed, "
            "a binding the world does not hold (naming the table and key), a scenario bound to nothing with "
            "nothing required, a situation and variant already written (naming the scenario), a statement that "
            "says what a written scenario says (naming it), an ask and an end that share a sentence, a tool name "
            "in the ask or the end, an ask without an end or an end without an ask. Write it again "
            "once, following the reason; a second refusal of the same scenario drops it. Each typed line belongs "
            "to one scenario: a write picking a line already rewritten is refused naming the scenario that holds "
            "it.")


def stop_section() -> str:
    return ("Stopping. Answer with one line and no tool call when every item of this session is written or "
            "dropped. Say how many you wrote.")


# --- the plan session (D380) -----------------------------------------------------------------

PLAN_RULE = ("A situation is one sentence in general words saying one thing that happens: the kind of customer, "
             "the thing and the step it has reached. It names no system, tool or id, and it never hedges: no "
             "\"sometimes\" and no \"or\"; what changes what the agent must do is a variant of it, one line each.")

_PLAN_EXAMPLES = {
    "list_openings": "list_openings: the numbered opening asks of this session, e.g. {}.",
    "list_plan": "list_plan: the situations of the plan so far, with their ids and counts, e.g. {}.",
    "write_situation": ('write_situation: write one situation, e.g. {"statement": "<the situation>", "kind": '
                        '"<one word>", "openings": [0, 4], "variants": ["<a variant>"]}.'),
    "attach_openings": ('attach_openings: add openings to a situation the plan holds, e.g. {"id": "si_<id>", '
                        '"openings": [7]}.'),
    "skip_opening": ('skip_opening: mark items that are no opening ask, e.g. {"openings": [9], "why": "<only '
                     'identification, no ask>"}.'),
}


_SEARCH_PLAN_EXAMPLES = {
    "list_plan": _PLAN_EXAMPLES["list_plan"],
    "write_situation": ('write_situation: write one situation from a page you read, e.g. {"statement": "<the '
                        'situation>", "kind": "<one of the plan\'s kinds>", "url": "<the URL you read>", "span": "<the '
                        'span it rests on, word for word>"}.'),
}


def plan_search_texts(names: list[str]) -> tuple[tuple[str, str], ...]:
    """The plan_search session's sections (D410): situations from the web, each on a page span, in GEPA order."""
    tools = held_line(names) + "\n" + "\n".join(_SEARCH_PLAN_EXAMPLES[n] for n in names if n in _SEARCH_PLAN_EXAMPLES)
    receives = ("You add situations to the plan of one business's scenarios, from the web. You search for what "
                "customers of this kind of business ask for, how its workflows go and what goes wrong, read the "
                "pages, and write each situation a page shows with the span it rests on. " + PLAN_RULE + " The "
                "policy, the tool list and the one-line description of the business sit beside this text. Your "
                'root is the scenario store, "."; read it, never edit it.')
    choice = ("Choosing. List the plan before you write and write nothing it already says. Fill the kinds the "
              "opening names as thinnest first. Each situation is something a real customer of this business "
              "could bring to its support and that this business's policy and world could handle or refuse. "
              "Write a situation only where the tool list holds an action for it or the policy says to refuse it. "
              "Give a kind the plan holds, never a new word for one it holds. "
              "Search widely: different customers, different steps of the workflow, different things gone wrong. "
              "Never search for an academic paper, a benchmark, a dataset or a code repository.")
    feedback = ("Feedback. A written situation answers with its id. A refused call says why: a page you never read; "
                "a span the page does not hold, quoting the nearest line; a kind the plan does not hold, naming the "
                "kinds; a kind sharing its stem with a kind of the plan, naming it; a statement that "
                "nearly repeats a situation of the plan, naming it; an empty statement, a tool name, an id shaped "
                "value or a hedge. Follow the reason and call again.")
    stop = ("Stopping. Answer with one line and no tool call when you have written the number of situations the "
            "opening asks for, or the web shows no more. Say how many you wrote.")
    examples = ("Examples over placeholders.\n"
                "a situation: a <kind of customer> wants to <change> a <thing> that has already <reached a later "
                "step>\nthe span it rests on: <the sentence of the page that shows it, word for word>")
    return (("receives", receives), ("tools", tools), ("examples", examples), ("choice", choice),
            ("feedback", feedback), ("stop", stop))


def plan_receives_section() -> str:
    return ("You plan the scenarios of one business before any is written. You read the opening asks of real "
            "conversations, numbered, and put every item in exactly one place: a situation you write, a situation "
            "the plan already holds, or the skipped list with the reason. " + PLAN_RULE + " The policy, the tool "
            'list and the one-line description of the business sit beside this text. Your root is the scenario '
            'store, "."; read it, never edit it.')


def plan_examples_section() -> str:
    return ("Examples over placeholders.\n"
            "a situation: a <kind of customer> wants to <change> a <thing> that has already <reached a later step>\n"
            "a variant of it: the <thing> has <reached a step where only a remedy is left>, so the agent offers "
            "<the remedy>")


def plan_choice_section() -> str:
    return ("Choosing. Read every item before you write. Write one situation per distinct thing that happens; the "
            "same thing at another step or with another remedy is a variant, never a second situation and never a "
            "hedge. Reuse a kind the plan holds rather than a new word for it. Attach an item to a situation the plan already holds rather than writing it again. Skip only "
            "an item that is no opening ask: a turn from the middle of a conversation, identification only, no "
            "ask. A situation the policy handles that no item shows is still worth one line, with no openings.")


def plan_feedback_section() -> str:
    return ("Feedback. A written situation answers with its id. A refused call says why: an item another "
            "situation holds, named with its holder; an empty statement; a tool name or an id shaped value, "
            "naming the word; a hedged statement, naming the hedging word; a kind sharing its stem with a kind of "
            "the plan, naming it. Follow the reason and call again.")


def plan_stop_section() -> str:
    return ("Stopping. Answer with one line and no tool call when every numbered item sits in exactly one "
            "situation or is skipped. Say how many situations you wrote and how many items you skipped.")


def _plan_texts(names: list[str]) -> tuple[tuple[str, str], ...]:
    tools = held_line(names) + "\n" + "\n".join(_PLAN_EXAMPLES[name] for name in names if name in _PLAN_EXAMPLES)
    return (("receives", plan_receives_section()), ("tools", tools), ("examples", plan_examples_section()),
            ("choice", plan_choice_section()), ("feedback", plan_feedback_section()), ("stop", plan_stop_section()))


def sections(names: Iterable[str] = TOOLS, kind: str = "") -> list[tuple[str, str]]:
    """Every section in GEPA order, the stop rule last; the plan session reads its own texts."""
    if kind == "plan":
        return [(name, wrap(text)) for name, text in _plan_texts(list(names))]
    if kind == "plan_search":
        return [(name, wrap(text)) for name, text in plan_search_texts(list(names))]
    return [(name, wrap(text)) for name, text in (
        ("receives", receives_section()), ("tools", tools_section(names)), ("examples", examples_section()),
        ("choice", choice_section()), ("feedback", feedback_section()), ("stop", stop_section()))]


OPENINGS = {
    "plan": ("Plan the situations this batch of opening asks shows: {count} items. Every numbered item ends in "
             "exactly one situation or skipped; write a situation for what the plan does not hold, attach to "
             "what it holds.\n\nThe plan so far:\n{items}"),
    "recordings": ("Generate scenarios from this batch of situations of the plan: source recordings, {count} items. "
                   "Write one scenario per situation and one per variant, each bound to rows of the world."),
    "ask": "Generate scenarios from this batch of lines a person typed: source ask, {count} items.",
    "gaps": ("Generate scenarios for the gaps in the store: source gaps, {count} items. Each item is a gap in the "
             "store; write one specific scenario for it, bound to real rows (look them up), that could reach this "
             "business's support in the real world; if the world has no rows for it, write it with requires naming "
             "what is missing. Every gap scenario has an ask and an end. A policy line gap's kind is one of the "
             "plan's kinds: {items}."),
    "search": ("Generate scenarios from the web for this business: source search. Search, read the pages, write what "
               "they show. Give each scenario the kind of situation it is, one of the plan's kinds: {items}."),
    "plan_search": ("Add {count} situations to the plan from the web. The kinds of the plan with how many situations "
                    "each holds, fewest first: {items}. Fill the thinnest kinds first.\n\nThe plan so far is "
                    "what list_plan shows."),
    "fix": "Rewrite the scenarios a reviewer sent back, each with its reason and what it was written from:\n{items}",
}


def plan_listing(rows: Iterable[dict]) -> str:
    """The plan so far as the opening message lists it: id, statement, how many openings."""
    lines = [f"{row['id']}: {row['statement']} ({row['openings']} openings)" for row in rows]
    return "\n".join(lines) or "no situation yet"


def opening(kind: str, entries: Iterable[str] = (), count: int = 0, items: str = "") -> str:
    """The opening message of one session: what it generates from and what the root holds."""
    listed = ", ".join(entries) or "nothing yet"
    return wrap(OPENINGS[kind].format(count=count, items=items) + f"\n\nThe root holds: {listed}.")


__all__ = ["plan_search_texts", "PLAN_RULE", "RULE", "TOOLS", "opening", "plan_listing", "sections"]
