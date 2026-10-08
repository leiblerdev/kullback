"""The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D320, D331).

The Examiner gives feedback and never edits: it reads the Intent, the policy and the Spec and files
rulings the writer applies or rebuts, then closes or keeps them open. General only: no corpus,
publisher, tool, table or column name appears here. The stop rule is last.
"""

from __future__ import annotations

from typing import Iterable

from kullback.examiner.skills import REVIEW_SKILL
from kullback.spec.rulings import RULING_KINDS

#: The Examiner writes nothing under its root (D320).
WRITABLE_DIRS: tuple[str, ...] = ()
#: The tools a review session holds; the full examine session adds the Environment ones.
REVIEW_TOOLS = ("rule", "close", "verify", "lookup_rows", "search_rows")


def receives_section() -> str:
    """What the session receives, the part that is the same in every session: its root as "."."""
    return ("You examine Specs. A Spec is the Verifier of one Task written from the user's Intent and the policy: "
            "items (end-state checks on rows and fields, checks on the calls made, and judge items, yes or no "
            "questions a model answers about what the agent told the user, each with an anchor value and the "
            "evidence it reads), each item citing the Intent facts or the policy line it comes from. Each Task "
            "has a view: the Intent facts, the items with their provenance, the Verifier's compiled items, the "
            "facts no item covers, and every ruling so far with the writer's answer. The policy and the tool "
            'list sit beside the views. Your root is ".", the exam directory: every path is relative to it. '
            "You give feedback and never edit: you file rulings, the writer applies or rebuts them. The Tasks "
            "and the round are in the opening message.")


def received(lines: str, entries: Iterable[str] = (), notes: str = "", round_number: int = 1) -> str:
    """This session's own facts: the round, what the root holds, the Tasks with their views, the Builder's
    notes. Said in the opening message, never the system prompt."""
    listed = ", ".join(entries) or "nothing yet"
    text = (f"Round {round_number}. The root holds: {listed}.\n"
            f"The Tasks, each with its view; open those paths rather than searching:\n{lines}")
    if round_number > 1:
        text += ("\nThe writer has answered the rulings of the last round. Close each answered ruling the answer "
                 "settles and keep open, with why, each it does not; file a new ruling only for what the "
                 "writer's change broke.")
    if notes:
        text += f"\nThe Builder's notes, each saying a Task is not verifiable as written:\n{notes}"
    return text


def opening(ask: str, lines: str, entries: Iterable[str] = (), notes: str = "", round_number: int = 1) -> str:
    """The opening message: the ask, then this session's facts."""
    return f"{ask}\n\n{received(lines, entries, notes, round_number)}"


_TOOL_EXAMPLES = {
    "rule": ('rule: file one ruling on one item, e.g. {"task_id": "t1", "item": "c2", "kind": "derivation", '
             '"code": "unasked_item", "reason": "No Intent fact or policy line asks for the second write.", '
             '"fix": "Drop c2 or cite the fact that asks for it.", "blocking": true}. item is a check, atom or '
             "fact id from the view, or task for the Task as a whole. The codes per kind: "
             + "; ".join(f"{kind}: {', '.join(codes)}" for kind, codes in RULING_KINDS.items())
             + ". A fails_reference ruling names wrong_side, reference or verifier."),
    "close": ('close: close an answered ruling or keep it open, e.g. {"task_id": "t1", "number": 1, '
              '"keep_open": false, "why": "The writer dropped the write no fact asked for."}.'),
    "verify": ('verify: run the do-nothing Run, a stray write and the Reference\'s end state through the '
               'Verifier, e.g. {"task_id": "t1"}; a passes_do_nothing or fails_reference ruling cites it.'),
    "lookup_rows": ('lookup_rows: read a row of the Task\'s world, the one the writer read, e.g. '
                    '{"task_id": "t1", "table": "<table>", "key": "<row key>"}.'),
    "search_rows": ('search_rows: find row keys whose field equals a value, e.g. {"task_id": "t1", '
                    '"table": "<table>", "field": "<field>", "value": "<value>"}.'),
    "finding": ('finding: what is wrong with the Environment, for the Builder, e.g. {"task_id": "t1", "kind": '
                '"fidelity", "text": "what differs", "rows": [...], "path": "env/tools/<name>.py", "edits": '
                '[{"kind": "body", "path": "env/tools/<name>.py", "call_id": "<id>", "column": "<column>", '
                '"recorded": "<value>", "replayed": "<value>", "why": "<why>"}]}.'),
    "no_finding": 'no_finding: a review that found nothing wrong, e.g. {"task_id": "t1", "reason": "<the reason>"}.',
    "check_reference": 'check_reference: the evidence on a Task\'s Reference in one view, e.g. {"task_id": "t1"}.',
}


def tools_section(tool_names: Iterable[str] = REVIEW_TOOLS) -> str:
    """The tools the session holds, one example call each."""
    lines = [_TOOL_EXAMPLES[name] for name in tool_names if name in _TOOL_EXAMPLES]
    return "Tools, one example call each. Besides these you hold read, ls, grep and find.\n" + "\n".join(lines)


def examples_section() -> str:
    """General examples of what a ruling says, one per kind."""
    return ("Examples of rulings.\n"
            "derivation: an item checks something no Intent fact or policy line asks for; an Intent fact has no "
            "item; an item fixes one value where the policy allows several (fix: a value set); the instruction "
            "forbids something and no item checks it; an item forbids something nothing in the Intent or policy "
            "forbids.\n"
            "judge: the question is not answerable yes or no; it asks about taste; its anchor is not the value "
            "the world holds or goes against the policy (look the value up first); its evidence cannot answer it; "
            "it leads or is ambiguous.\n"
            "code: the check reads the wrong table or field; it does not run; verify shows the do-nothing Run "
            "passes it; verify shows the Reference fails it, and you rule which side is wrong from the Intent.\n"
            "scope: the request is not doable under the policy (fix: a refusal, the sanity item plus a judge item "
            "citing the policy line); the Intent is too vague to check; the instruction lacks a fact the Task "
            "needs.")


def choice_section() -> str:
    """The choice rule: what to read, what blocks, then the order."""
    return ("Choosing. Read each Task's view, then the policy where an item cites it. Look up in the world every "
            "anchor and every value an item fixes before you accept it. Call verify once per Task before any "
            "code ruling. Block when a correct solve could fail or a wrong one pass; file a note when the item "
            "is weak but the verdict would stand. One ruling per problem, on the item it is about. Decide from "
            "the Intent and the policy, never from what a Run did.\n" + REVIEW_SKILL)


def feedback_section() -> str:
    """The shape of the feedback."""
    return ("Feedback. A filed ruling answers with its number; a refusal says what was missing. The writer "
            "answers each ruling next round, applied or rebutted with why. Two rounds at most: a blocking ruling "
            "still open after the last round holds the Task untrusted.")


def stop_section() -> str:
    """The stop rule, last: one line and no tool call when nothing is left to do."""
    return ("Stopping. Answer with one line and no tool call when every Task is examined and every answered "
            "ruling is closed or kept open. Say how many rulings you filed per Task and how many block.")


def sections(tool_names: Iterable[str] = REVIEW_TOOLS) -> list[tuple[str, str]]:
    """Every prompt section in GEPA order, the stop rule last. The same text in every session."""
    return [("receives", receives_section()),
            ("tools", tools_section(tool_names)),
            ("examples", examples_section()),
            ("choice", choice_section()),
            ("feedback", feedback_section()),
            ("stop", stop_section())]


def render(sections_list: list[tuple[str, str]]) -> str:
    """The sections joined for a reader that wants one text."""
    return "\n\n".join(text for _, text in sections_list)


__all__ = ["REVIEW_TOOLS", "WRITABLE_DIRS", "choice_section", "examples_section", "feedback_section", "opening",
           "receives_section", "received", "render", "sections", "stop_section", "tools_section"]
