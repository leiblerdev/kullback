"""The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D320, D331).

The Examiner gives feedback and never edits: it reads the user turns, the Intent, the policy and the
Spec and files rulings the writer applies or rebuts, then closes or keeps them open. General only: no corpus,
publisher, tool, table or column name appears here. The stop rule is last.
"""

from __future__ import annotations

from typing import Iterable

from kullback.examiner.skills import REVIEW_SKILL
from kullback.spec.rulings import RULING_KINDS

#: The Examiner writes nothing under its root (D320).
WRITABLE_DIRS: tuple[str, ...] = ()
#: The tools a review session holds; the full examine session adds the Environment ones.
REVIEW_TOOLS = ("rule", "close", "verify", "lookup_rows", "search_rows", "find_tools")


def receives_section() -> str:
    """What the session receives, the part that is the same in every session: its root as "."."""
    return ("You examine Specs. A Spec is the Verifier of one Task written from the user's Intent and the policy: "
            "items (end-state checks on rows and fields, checks on the calls made, and judge items, yes or no "
            "questions a model answers about what the agent told the user, each with an anchor value and the "
            "evidence it reads), each item citing the Intent facts or the policy line it comes from. Each Task "
            "has a view: the user turns of every recording (each with its index and the agent's last line "
            "before it), the Intent facts with the recording and turn each was read from, its stance (a "
            "fallback fact was asked only after the agent refused, could not, or proposed another way, and "
            "names the agent line it answers: it is scored, never gated) and its "
            "echo and side flags (a side fact asks only to be told something beside the Task's ask: it is scored, "
            "never gated, so a ruling never asks to gate it), the items with their provenance, the Verifier's compiled items, the facts no item "
            "covers, and every ruling so far with the writer's answer. The policy and the tool "
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
        text += ("\nThe writer has answered the rulings of the last round. First question the goal again as in "
                 "round one; then close each answered ruling the answer settles and keep open, with why, each it "
                 "does not; file a new ruling only for what the writer's change broke or a goal the Intent still "
                 "gets wrong.")
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
             + ". A fails_reference ruling names wrong_side, reference or verifier, and ref_step, a step the verify answer "
             "shows, or never for a write or call the Reference never made."),
    "close": ('close: close an answered ruling or keep it open, e.g. {"task_id": "t1", "number": 1, '
              '"keep_open": false, "why": "The writer dropped the write no fact asked for."}.'),
    "verify": ('verify: run the do-nothing Run, a stray write and the Reference\'s end state through the '
               'Verifier, e.g. {"task_id": "t1"}; a passes_do_nothing or fails_reference ruling cites it. The answer '
               "carries the Reference steps (user turns, tool calls, replies); a fails_reference ruling names the step."),
    "lookup_rows": ('lookup_rows: read a row of the Task\'s world, the one the writer read, e.g. '
                    '{"task_id": "t1", "table": "<table>", "key": "<row key>"}.'),
    "search_rows": ('search_rows: find row keys whose field equals a value, e.g. {"task_id": "t1", '
                    '"table": "<table>", "field": "<field>", "value": "<value>"}.'),
    "find_tools": ('find_tools: the tools whose name or description match a few words, e.g. '
                   '{"query": "change after payment"}.'),
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
            "passes it; verify shows the Reference fails it, and you first read its steps, then rule which side "
            "is wrong and at which step, from the user turns and the policy.\n"
            "scope: the request is not doable under the policy (fix: a refusal, the sanity item plus a judge item "
            "citing the policy line); the Intent is too vague to check; the instruction lacks a fact the Task "
            "needs.\n"
            "intent_unfounded: an Intent fact no user turn says, or one the user only accepted after the agent "
            "refused, failed or proposed it (fix: drop the fact from every gate).\n"
            "gated_against_ask: a gate works against what the user volunteered while the policy and the world "
            "allow a path to it, such as a refusal gated where the policy offers a paid way.\n"
            "not_enough_truth: the turns never show what the user wants at the point the Verifier must decide "
            "(fix: keep untrusted).")


def choice_section() -> str:
    """The choice rule: what to read, what blocks, then the order."""
    return ("Choosing. Question the goal first, on every Task, before the items. From the user's own requests, "
            "as made before the agent refused, failed or proposed anything, and from the policy alone, say in one "
            "or two lines what a correct agent leaves behind. Hold the Intent against that: a fact the user only "
            "accepted after the agent's proposal, refusal or failure is not the user's wish (rule "
            "intent_unfounded only where a gate, or an item that decides the verdict, rests on it; where nothing "
            "does, say so in your review text and file nothing); a request the user made that no fact or item carries is an uncovered ask (rule "
            "missing_fact, or uncovered_fact where a fact has no item); where the turns never show what the user "
            "would want once the policy allows the request, rule not_enough_truth. Users ask for things the world "
            "or the policy may not allow; serve the ask only as far as they allow. Only then check the items "
            "against the Intent and the policy. The whole policy is in this prompt: read the part that governs "
            "the ask before you accept or rule. Look up in the world every anchor and every value an "
            "item fixes (lookup_rows, search_rows) before you accept it. Call verify once per Task before any code ruling. "
            "When verify shows the Reference failing a check, investigate before ruling: read the Reference steps in "
            "the verify answer, find what the agent did at the step the failing check reads, hold it against the "
            "user's own requests and the policy, then name the side and the step. Before you name a side, ask whether "
            "a right solve of the user's own requests, under the policy, would pass this check: if not, the check "
            "is wrong (wrong_side verifier), whatever the Reference did. A Reference passing is never a "
            "reason to change a check: a wrong Reference ends the Task with no Verifier, an honest outcome. "
            "Block when a correct solve could fail or a wrong one pass; file a note when the item is weak but the "
            "verdict would stand. One ruling per problem, on the item it is about. Decide from what the user said "
            "(the user turns), the world and the policy, never from what a Run did.\n" + REVIEW_SKILL)


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
