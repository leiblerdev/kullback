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
            "try_atoms: try an edit before writing it: same arguments as edit_verifier, nothing "
            "is written, and the answer is the same rulings with the failing atoms named. "
            "A dry run verdict is not a refusal. "
            "probe: score a hand-written Run against the current Verifier, "
            'e.g. {"task_id": "t1", "bug_class": "loose check", "note": "what it changes", '
            '"events": [...]}; the Run stays in the pool.\n'
            "finding: file what is wrong on the Builder side, "
            'e.g. {"task_id": "t1", "kind": "fidelity", "text": "what differs", '
            '"rows": [...], "path": "env/tools/<name>.py", "change": "what should differ"}.\n'
            "reroll: buy more frontier Runs of a Task, "
            'e.g. {"task_id": "t1", "count": 2}.\n'
            "check_reference: the evidence on a Task's Reference in one view, "
            'e.g. {"task_id": "t1"}.\n'
            "reject_reference: reject a Reference whose End state is wrong for the Intent, "
            'e.g. {"task_id": "t1", "run_id": "r1", "why": "the user asked for a change and it wrote nothing"}; '
            "every Run of that End state goes with it and the Reference is chosen again; a Task with "
            "none left waits in the Reference pool for a later Run.")


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
            "and what differs, never a verb. Two refused edit_verifier proposals on one check end with a "
            "finding, not a third. A try_atoms dry run is not a refusal and counts toward neither. "
            "A Builder note says a Task is not verifiable as written: rule "
            "on it before anything else on that Task, with edit_verifier when you agree, a probe, "
            "or a finding with note_ruling saying whether the Builder is right and why. The Task "
            "cannot be refused while its note is open; the ruling goes back to the Builder. "
            "Call check_reference before editing atoms when most re-rolls disagree with the "
            "Reference, when the Reference wrote nothing, or when a note is open. Judge the view "
            "against the Intent and the policy; when the End state is not what they ask for, "
            "reject_reference instead of fitting the atoms to it.")


def feedback_section() -> str:
    """The shape of the feedback: rulings with the rows that still differ."""
    return ("Feedback. A write under verifiers/ answers with one ruling per gate and the rows "
            "that still differ on each; a refusal says which bound the write hit. Read the rows "
            "before proposing again: they name the call, the recorded value, our value and the "
            "first differing column.")


def stop_section() -> str:
    """The stop rule, last: one line and no tool call when nothing is left to do."""
    return ("Stopping. Answer with one line and no tool call when every Task is trusted or "
            "refused, every trusted Task's Reference is checked, and every Builder note is "
            "ruled; or when the rulings after your proposals are the ones you already "
            "answered. A Reference is checked once check_reference has been called on its "
            "Task this session. Say which Tasks remain, what you filed for each, and which "
            "trusted Tasks are still unchecked.")


# --- under --verifier-from intent: the Spec writes the Verifier, the Examiner judges only ---


def intent_receives_section() -> str:
    """What an intent-mode session receives: a Spec per Task and the Runs scored against it."""
    return ("You receive Tasks whose Verifier the Spec wrote from the user's words. Each Task has a spec "
            "view: the Intent facts, each with its stance (volunteered or accepted), the checks, each with "
            "its because and tier, and one verdict row per Run naming the checks it failed and the values "
            'they expect. Your root is ".", the exam directory: every path you pass is relative to it, '
            "never a host path. It holds the Runs as JSONL and one spoken file per Task. You judge; code "
            "moves on what you rule. What the root holds and the Tasks are in the opening message.")


def intent_tools_section() -> str:
    """The intent-mode tools with one example call each: `rule` is the one move."""
    return ("Tools, one example call each.\n"
            "rule: file one ruling on a Task, "
            'e.g. {"task_id": "t1", "target": "check", "check_ids": ["c2"], "code": "check_unpassable", '
            '"reason": "the check demands '
            'a second write, but its because says only \\"<the words of the because>\\""}; target is run (the '
            "Run did the wrong thing), check (a check demands what the Intent does not), intent (a fact is "
            "missing or wrong), or environment (the Environment cannot support the Task). The reason must "
            "copy verbatim, in quotation marks and at least twelve characters, a because of the Spec or a "
            "turn of the Run in run_id; a reason without a quote is refused. Put a long quote in excerpt. "
            "code names the kind of reason: check_unpassable, run_contradicts_intent, reference_wrong, "
            "environment_defect or other.\n"
            "check_reference: the evidence on a Task's Reference in one view, "
            'e.g. {"task_id": "t1"}.\n'
            "reject_reference: rule on a Reference Run whose End state is wrong for the Intent, "
            'e.g. {"task_id": "t1", "run_id": "r1", "why": "the user said <their words> and it wrote nothing"}; '
            "it files rule with target run, so the why quotes like any reason.\n"
            "finding: file what is wrong on the Builder side, "
            'e.g. {"task_id": "t1", "kind": "fidelity", "text": "what differs", "rows": [...], '
            '"path": "env/tools/<name>.py", "change": "what should differ"}.\n'
            "reroll: buy more frontier Runs of a Task, "
            'e.g. {"task_id": "t1", "count": 2}.')


def intent_examples_section() -> str:
    """General examples of a disagreement and the ruling that answers it."""
    return ("Examples of a disagreement and the ruling that answers it.\n"
            "Runs fail a check whose because quotes a fact the user never asked for: rule target check "
            "on that check, quoting the because.\n"
            "One Run fails a check the others pass, and its turns show it did something the user did "
            "not ask: rule target run on that Run, quoting the turn.\n"
            "Every Run fails a check because the user's turns ask for something the facts left out: "
            "rule target intent, quoting the turn.\n"
            "The Runs cannot reach the End state because a tool refuses what the policy allows: rule "
            "target environment, quoting the turn that shows it.")


def intent_choice_section() -> str:
    """The intent-mode choice rule: judge disagreements, one ruling per wrong thing."""
    return ("Choosing. You hold no edit tool: the Spec repairs its checks, the Runner re-rolls, the "
            "Builder fixes the Environment. Read each Task's spec view first. Act on the Tasks where "
            "Runs and checks disagree: some Runs fail a check. Decide which side is wrong from the Intent "
            "and the policy, never from which side has more Runs. A Task where every Run passes and the "
            "checks match the facts needs no ruling. File one ruling per wrong thing, with the check ids "
            "or the run id it is about. A ruling already filed on a check is not filed again this session. "
            "A refused rule says what its reason lacked: quote the evidence and file it once more, or drop it.")


def intent_feedback_section() -> str:
    """The shape of the feedback in intent mode."""
    return ("Feedback. A filed rule answers with its number and the sources it quoted; a refusal says "
            "what was missing. Code routes each ruling: a Run to a re-roll, a check or the Intent to "
            "the Spec's repair, the Environment to the Builder. Two rounds on a Task without agreement "
            "set it aside.")


def intent_stop_section() -> str:
    """The intent-mode stop rule, last."""
    return ("Stopping. Answer with one line and no tool call when every Task with a disagreement has a "
            "ruling or needs none, and every Builder note is ruled. Say which Tasks you ruled on, with "
            "which target, and which needed none.")


def sections(verifier_from: str = "runs") -> list[tuple[str, str]]:
    """Every prompt section in GEPA order, the stop rule last. The same text in every session of a mode."""
    if verifier_from == "intent":
        return [("receives", intent_receives_section()),
                ("tools", intent_tools_section()),
                ("examples", intent_examples_section()),
                ("choice", intent_choice_section()),
                ("feedback", intent_feedback_section()),
                ("stop", intent_stop_section())]
    return [("receives", receives_section()),
            ("tools", tools_section()),
            ("examples", examples_section()),
            ("choice", choice_section()),
            ("feedback", feedback_section()),
            ("stop", stop_section())]


def render(sections_list: list[tuple[str, str]]) -> str:
    """The sections joined for a reader that wants one text."""
    return "\n\n".join(text for _, text in sections_list)


__all__ = ["WRITABLE_DIRS", "choice_section", "examples_section", "feedback_section", "intent_choice_section",
           "intent_examples_section", "intent_feedback_section", "intent_receives_section", "intent_stop_section",
           "intent_tools_section", "opening", "receives_section",
           "received", "render", "sections", "stop_section", "tools_section"]
