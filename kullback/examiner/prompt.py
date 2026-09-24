"""The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop.

General only: no corpus, publisher, tool, table or column name appears here. The stop
rule is last: one line and no tool call when every Task is trusted or refused, or when
the rulings after the proposals are the ones already answered.
"""

from __future__ import annotations

from typing import Iterable

#: Where the session's writes land; each directory appears on its first write.
WRITABLE_DIRS = ("verifiers", "probes")


def receives_section(writable: Iterable[str] = WRITABLE_DIRS) -> str:
    """What the session receives, the part that is the same in every session: its root as ".".

    The root is never named by a host path: every path the session passes is relative to it. What
    the root holds and the rulings differ per session, so they arrive in the opening message
    (`received`), after the system prompt, where they leave the cached prefix of tools and system
    intact from one session to the next (prompt-caching.md: keep the system prompt frozen).
    """
    later = ", ".join(f"{name}/" for name in writable)
    return ("You receive the rulings of a derivation: one Verifier per Task from its References "
            "through the suite, with the rows that still differ on each ruling. "
            'Your root is ".", the exam directory: every path you pass is relative to it, never '
            "a host path. The read surface under it holds the Runs as JSONL, the replay "
            "and re-roll rows, the References, the task status, the Verifiers, the probe pools, "
            "the task, intent and user-rule records, and one spoken file per Task with the user "
            "turns of its References.\n"
            f"These appear on your first write under them: {later}.\n"
            "What the root holds and the rulings to answer are in the opening message.")


def received(rulings: str, entries: Iterable[str] = (), notes: str = "") -> str:
    """This session's own facts: what the root held when it opened (directories end with "/"),
    the rulings per Task with their paths, and the Builder's notes. Said in the opening message,
    never the system prompt."""
    listed = ", ".join(entries) or "nothing yet"
    text = (f"The root holds: {listed}.\n"
            "The rulings to answer, each Task with the paths of its Runs and its Verifier; open "
            f"those paths rather than searching for them:\n{rulings}")
    if notes:
        text += ("\nThe Builder's notes, each saying a Task is not verifiable as written; rule on "
                 f"every open one:\n{notes}")
    return text


def opening(ask: str, rulings: str, entries: Iterable[str] = (), notes: str = "") -> str:
    """The opening message: the ask, then this session's facts."""
    return f"{ask}\n\n{received(rulings, entries, notes)}"


def tools_section() -> str:
    """The tools with one example call each."""
    return ("Tools, one example call each.\n"
            "edit_verifier: write a new Verifier version off the current one, "
            'e.g. {"task_id": "t1", "edit": [{"id": "atom-2", "payload": {"count": 2}}], '
            '"drop": ["atom-3"], "add": [], "reason": "the atom rejects the second path"}; '
            "use edit for a change to an existing atom (only the fields that change, the kind "
            "stays) and drop with add only to remove an atom or bring a new one; the suite, "
            "the pool and the loosening gate rule on the write.\n"
            "probe: score a hand-written Run against the current Verifier, "
            'e.g. {"task_id": "t1", "bug_class": "loose check", "note": "what it changes", '
            '"events": [...]}; the Run stays in the pool.\n'
            "finding: file what is wrong on the Builder side, "
            'e.g. {"task_id": "t1", "kind": "fidelity", "text": "what differs", '
            '"rows": [...], "path": "env/tools/<name>.py", "change": "what should differ"}.\n'
            "reroll: buy more frontier Runs of a Task, "
            'e.g. {"task_id": "t1", "count": 2}.')


def examples_section() -> str:
    """General examples of a ruling and the call that answers it."""
    return ("Examples of a ruling and the call that answers it.\n"
            "A Verifier that failed the suite on one check is answered with edit_verifier "
            "changing the atom the ruling names in place, with the reason saying why.\n"
            "A body whose replay differs on recorded calls is answered with a finding naming "
            "env/tools/<name>.py in path and the differing columns in change, with the replay "
            "rows in rows.\n"
            "A Task no frontier Run finished is answered with a finding suggesting refusal, "
            "with the termination reasons in rows.")


def choice_section() -> str:
    """The choice rule: what to act on first, and what ends a line of proposals."""
    return ("Choosing. Act first on Tasks with a confirmed Reference whose Verifier failed the "
            "suite, then on Tasks with no Verdict. Search before reading: ground a claim about "
            "many Tasks with a search, then read the records it names. A finding names the file "
            "and what differs, never a verb. Two rejected proposals on one check end with a "
            "finding, not a third. A Builder note says a Task is not verifiable as written: rule "
            "on it before anything else on that Task, with edit_verifier when you agree, a probe, "
            "or a finding with note_ruling saying whether the Builder is right and why. The Task "
            "cannot be refused while its note is open; the ruling goes back to the Builder.")


def feedback_section() -> str:
    """The shape of the feedback: rulings with the rows that still differ."""
    return ("Feedback. A write under verifiers/ answers with one ruling per gate and the rows "
            "that still differ on each; a refusal says which bound the write hit. Read the rows "
            "before proposing again: they name the call, the recorded value, our value and the "
            "first differing column.")


def stop_section() -> str:
    """The stop rule, last: one line and no tool call when nothing is left to do."""
    return ("Stopping. Answer with one line and no tool call when every Task is trusted or "
            "refused, or when the rulings after your proposals are the ones you already "
            "answered, and every Builder note is ruled. Say which Tasks remain and what you filed for each.")


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


__all__ = ["WRITABLE_DIRS", "choice_section", "examples_section", "feedback_section", "opening", "receives_section",
           "received", "render", "sections", "stop_section", "tools_section"]
