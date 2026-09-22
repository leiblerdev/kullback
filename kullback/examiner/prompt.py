"""The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop.

General only: no corpus, publisher, tool, table or column name appears here. The stop
rule is last: one line and no tool call when every Task is trusted or refused, or when
the rulings after the proposals are the ones already answered.
"""

from __future__ import annotations


def receives_section(rulings: str, root: str) -> str:
    """What the session receives: the derivation's rulings per Task with rows, and its root."""
    return ("You receive the rulings of a derivation: one Verifier per Task from its References "
            "through the suite, with the rows that still differ on each ruling. "
            f"Your root is {root}. The read surface under it holds the Runs as JSONL, the replay "
            "and re-roll rows, the References, the task status, the Verifiers, the probe pools, "
            "the task, intent and user-rule records, and one spoken file per Task with the user "
            "turns of its References.\n"
            f"The rulings to answer:\n{rulings}")


def tools_section() -> str:
    """The tools with one example call each."""
    return ("Tools, one example call each.\n"
            "propose_verifier: propose a new Verifier version off the current one, "
            'e.g. {"task_id": "t1", "drop": ["atom-3"], "add": [], '
            '"reason": "the atom rejects the second path"}; the suite, the pool and the '
            "loosening gate rule on the write.\n"
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
            "A Verifier that failed the suite on one check is answered with propose_verifier "
            "dropping or loosening the atom the ruling names, with the reason saying why.\n"
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
            "finding, not a third.")


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
            "answered. Say which Tasks remain and what you filed for each.")


def sections(rulings: str, root: str) -> list[tuple[str, str]]:
    """Every prompt section in GEPA order, the stop rule last."""
    return [("receives", receives_section(rulings, root)),
            ("tools", tools_section()),
            ("examples", examples_section()),
            ("choice", choice_section()),
            ("feedback", feedback_section()),
            ("stop", stop_section())]


def render(sections_list: list[tuple[str, str]]) -> str:
    """The sections joined for a reader that wants one text."""
    return "\n\n".join(text for _, text in sections_list)


__all__ = ["choice_section", "examples_section", "feedback_section", "receives_section",
           "render", "sections", "stop_section", "tools_section"]
