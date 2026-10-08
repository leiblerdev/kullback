"""The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D320).

The Examiner reads and reviews: it files one review per Task, a finding with the rows and the edit
or no_finding with its reason. It writes no Verifier and runs nothing; the Spec applies the edits.
General only: no corpus, publisher, tool, table or column name appears here. The stop rule is last.
"""

from __future__ import annotations

from typing import Iterable

from kullback.examiner.skills import REVIEW_SKILL

#: The Examiner writes nothing under its root (D320).
WRITABLE_DIRS: tuple[str, ...] = ()


def receives_section() -> str:
    """What the session receives, the part that is the same in every session: its root as ".".

    What the root holds and the Tasks differ per session, so they arrive in the opening message,
    after the system prompt, where they leave the cached prefix of tools and system intact.
    """
    return ("You review Tasks whose Verifier the Spec wrote: the expected end states with each cell's "
            "source, the forbidden list, the conduct rules and the fact and question checks. Each Task has "
            "a spec view: the Intent facts, the checks with their because, and one verdict row per Run. "
            'Your root is ".", the exam directory: every path you pass is relative to it, never a host '
            "path. It holds the Runs as JSONL, the replay and re-roll rows, the References, the task "
            "status, the Verifiers, the task, intent and user-rule records, and one spoken file per Task. "
            "You read and review; you write nothing and run nothing. What the root holds and the Tasks "
            "are in the opening message.")


def received(rulings: str, entries: Iterable[str] = (), notes: str = "") -> str:
    """This session's own facts: what the root held when it opened, the Tasks with their paths, and the
    Builder's notes. Said in the opening message, never the system prompt."""
    listed = ", ".join(entries) or "nothing yet"
    text = (f"The root holds: {listed}.\n"
            "The Tasks to review, each with the paths of its Runs and its Verifier; open those paths "
            f"rather than searching for them:\n{rulings}")
    if notes:
        text += ("\nThe Builder's notes, each saying a Task is not verifiable as written; rule on "
                 f"every open one:\n{notes}")
    return text


def opening(ask: str, rulings: str, entries: Iterable[str] = (), notes: str = "") -> str:
    """The opening message: the ask, then this session's facts."""
    return f"{ask}\n\n{received(rulings, entries, notes)}"


def tools_section() -> str:
    """The tools with one example call each."""
    return ("Tools, one example call each. Besides these you hold read, ls, grep and find.\n"
            "finding: file what is wrong with a Task, "
            'e.g. {"task_id": "t1", "kind": "fidelity", "text": "what differs and why", "rows": [...], '
            '"edits": [{"kind": "atoms", "task_id": "t1", "drop": ["atom-3"], "add": [], '
            '"why": "the check demands what no fact says"}]}. An edit is the diff: atoms (task_id, drop '
            "and add; an added atom is a fact told or a question asked, never a write), text (path "
            "intents/<task>.json or tasks/<task>.json, where the verbatim current text, found exactly "
            "once, and replace the new text) or body (path env/tools/<name>.py, the call, the column and "
            "both values). Atoms and text edits go to the Spec, body edits to the Builder. Add "
            'note_ruling ("builder_right" or "builder_wrong") when the finding answers a Builder note.\n'
            "no_finding: file a review that found nothing wrong, "
            'e.g. {"task_id": "t1", "reason": "every cell has a source and the Runs agree with the facts"}.\n'
            "rule: file one ruling on a Task, "
            'e.g. {"task_id": "t1", "target": "check", "check_ids": ["c2"], "code": "check_unpassable", '
            '"reason": "the check demands a second write, but its because says only \\"<the words>\\""}; '
            "target is run, check, intent or environment, and the reason quotes verbatim, in quotation "
            "marks and at least twelve characters, a because of the Spec or a turn of the Run.\n"
            "check_reference: the evidence on a Task's Reference in one view, "
            'e.g. {"task_id": "t1"}.\n'
            "reject_reference: rule on a Reference Run whose End state is wrong for the Intent, "
            'e.g. {"task_id": "t1", "run_id": "r1", "why": "the user said <their words> and it wrote nothing"}.')


def examples_section() -> str:
    """General examples of what a review finds and the call that files it."""
    return ("Examples of a review and the call that files it.\n"
            "A check demands what no fact of the Intent says: finding with an atoms edit dropping it.\n"
            "An Intent fact misquotes the user's turn: finding with a text edit quoting both.\n"
            "A replayed call differs from the recorded one on a column: finding with a body edit naming "
            "the call, the column and both values.\n"
            "Every cell has a source, the checks match the facts and the Runs agree: no_finding with "
            "the reason.")


def choice_section() -> str:
    """The choice rule: one review per Task, then the order."""
    return ("Choosing. File one review per Task: a finding with what is wrong, why, the rows and the edit, "
            "or no_finding with its reason. A review without rows is not a finding, and a finding names "
            "the file and what differs, never a verb. Read each Task's spec "
            "view first and its Verifier's cells and sources; act first on Tasks where Runs and checks "
            "disagree or a cell has no source. Decide which side is wrong from the Intent and the policy, "
            "never from which side has more Runs. A Builder note is ruled before anything else on its "
            "Task. Prefer an edit whenever you can name the values or quote the text.\n" + REVIEW_SKILL)


def feedback_section() -> str:
    """The shape of the feedback."""
    return ("Feedback. A filed review answers with what it recorded; a refusal says what was missing. "
            "Code routes each edit: atoms and text to the Spec, which rewrites the Verifier and runs the "
            "gates, body to the Builder. A Task the gates still refuse at the round ceiling is set aside "
            "as pending.")


def stop_section() -> str:
    """The stop rule, last: one line and no tool call when nothing is left to do."""
    return ("Stopping. Answer with one line and no tool call when every Task has one review filed and "
            "every Builder note is ruled. Say which Tasks got a finding, with which edit kinds, and which "
            "got no_finding.")


def sections() -> list[tuple[str, str]]:
    """Every prompt section in GEPA order, the stop rule last. The same text in every session."""
    return [("receives", receives_section()),
            ("tools", tools_section()),
            ("examples", examples_section()),
            ("choice", choice_section()),
            ("feedback", feedback_section()),
            ("stop", stop_section())]


def render(sections_list: list[tuple[str, str]]) -> str:
    """The sections joined for a reader that wants one text."""
    return "\n\n".join(text for _, text in sections_list)


__all__ = ["WRITABLE_DIRS", "choice_section", "examples_section", "feedback_section", "opening",
           "receives_section", "received", "render", "sections", "stop_section", "tools_section"]
