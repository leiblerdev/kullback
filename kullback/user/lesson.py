"""What the fidelity score teaches the agent user's context next round, and when to stop paying (D214 rule 6).

The Builder is handed a lesson per tool and the Examiner a finding per Task; this is the same loop
closed round the third agent. The score already names, per Task, the facts the driver missed, the
facts it added that the recorded turn did not carry, the record facts it spoke and the ends it got
wrong, so the lesson is those four lists in one sentence each, filed under the Task and put in front
of the next round's turns as the `user_lessons` section (context.py).

The stall rule is the other half. A Task whose agent score does not beat the rules after three
rounds of lessons has been told three times and has not learned; paying a model per turn on it after
that buys nothing, so the Task drops to the rule-driven user until its facts or its persona change.
That change is a content key over those two and nothing else (context.content_key), so a new round,
a new lesson or a new Candidate model does not restart the clock, and a Task that was re-mined does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from kullback.runner.records import Record, read_json, write_json
from kullback.user.fidelity import AGENT_DRIVER, RULES_DRIVER, TaskScore

FILE_NAME = "user_lessons.json"
FORMAT = 1
# How many rounds of lessons a Task gets before it stops paying for the agent user.
STALL_ROUNDS = 3
# How many names one line of a lesson carries. A lesson that lists forty fields is a lesson nobody
# reads, and the fields are ranked by nothing, so the cut is the first few in the score's own order.
NAMED = 6


class Lesson(Record):
    """One round's reading of one Task, in the four things the score can name."""

    task_id: str = ""
    round: int = 0
    key: str = ""              # the content key of the facts and the persona this was learned on
    agent: Optional[float] = None
    rules: Optional[float] = None
    missed: list[str] = []
    invented: list[str] = []
    record_spoken: list[str] = []
    ends_wrong: int = 0

    def beat_the_rules(self) -> bool:
        return self.agent is not None and self.rules is not None and self.agent > self.rules

    def lines(self) -> list[str]:
        """The lesson as the next round's context reads it, one line per thing that went wrong."""
        out: list[str] = []
        if self.missed:
            out.append("You were asked for these and did not say them: "
                       + ", ".join(self.missed[:NAMED]) + ".")
        if self.invented:
            out.append("You said these where the conversation did not: "
                       + ", ".join(self.invented[:NAMED]) + ". Answer the question you were asked.")
        if self.record_spoken:
            out.append("You read out values that are not yours to give: "
                       + ", ".join(self.record_spoken[:NAMED]) + ". Point at where they live instead.")
        if self.ends_wrong:
            out.append(f"You got the ending wrong on {self.ends_wrong} turn(s): you stopped where the "
                       "conversation went on, or went on where it had stopped.")
        return out


def lesson_from(agent: Optional[TaskScore], rules: Optional[TaskScore], *, task_id: str = "",
                round: int = 0, key: str = "") -> Lesson:
    """One round's lesson for one Task, off the two scores the round computed."""
    source = agent or rules
    return Lesson(task_id=task_id or (source.task_id if source else ""), round=round, key=key,
                  agent=agent.score if agent is not None else None,
                  rules=rules.score if rules is not None else None,
                  missed=list(source.missed) if source else [],
                  invented=list(source.invented) if source else [],
                  record_spoken=list(source.record_spoken) if source else [],
                  ends_wrong=source.ends_wrong if source else 0)


def lines_for(lessons: Iterable[Lesson], task_id: str, key: str = "") -> list[str]:
    """The lines the next round puts in front of this Task's turns: its own lessons, newest last.

    A lesson learned on facts or a persona this Task no longer has is dropped rather than shown: it
    would be telling the model about someone else.
    """
    rows = [lesson for lesson in lessons or ()
            if lesson.task_id == task_id and (not key or not lesson.key or lesson.key == key)]
    rows.sort(key=lambda lesson: lesson.round)
    out: list[str] = []
    for lesson in rows[-STALL_ROUNDS:]:
        for line in lesson.lines():
            if line not in out:
                out.append(line)
    return out


def stalled(lessons: Iterable[Lesson], task_id: str, key: str = "") -> bool:
    """This Task stops paying for the agent user (D214 rule 6).

    Three rounds of lessons on the same facts and persona in which the agent did not beat the rules.
    A round where the agent was never scored does not count against it: it was not asked.
    """
    rows = [lesson for lesson in lessons or ()
            if lesson.task_id == task_id and lesson.agent is not None
            and (not key or not lesson.key or lesson.key == key)]
    if any(lesson.beat_the_rules() for lesson in rows):
        return False
    return len(rows) >= STALL_ROUNDS


def drives(lessons: Iterable[Lesson], task_id: str, agent: Optional[float], rules: Optional[float],
           key: str = "") -> bool:
    """Whether the agent user drives this Task's Runs this round (D214 rule 3).

    It drives when its offline fidelity beats the rule-driven user's on the same recorded turns, and
    it does not drive a Task the stall rule has closed, however this round's numbers came out: the
    stall is what stops a Task that keeps almost winning from being paid for forever.
    """
    if agent is None or rules is None:
        return False
    if stalled(lessons, task_id, key):
        return False
    return agent > rules


def write_lessons(workdir: Any, lessons: Iterable[Lesson]) -> Path:
    body = {"format": FORMAT, "lessons": [lesson.model_dump(mode="json") for lesson in lessons or ()]}
    return write_json(Path(workdir) / FILE_NAME, body)


def load_lessons(workdir: Any) -> list[Lesson]:
    body = read_json(Path(workdir) / FILE_NAME, {}) or {}
    if not isinstance(body, dict) or body.get("format") != FORMAT:
        return []
    out: list[Lesson] = []
    for row in body.get("lessons") or ():
        try:
            out.append(Lesson.model_validate(row))
        except ValueError:
            continue
    return out


def append_round(workdir: Any, lessons: Sequence[Lesson]) -> Path:
    """This round's lessons after the ones already on disk, so the stall rule can count rounds."""
    return write_lessons(workdir, [*load_lessons(workdir), *lessons])


def counts(lessons: Iterable[Lesson], round: int) -> dict:
    """What a round line says about the lesson loop: how many Tasks learned, how many stalled."""
    rows = [lesson for lesson in lessons or () if lesson.round == round]
    tasks = {lesson.task_id for lesson in lessons or ()}
    return {
        "user_lessons_written": len(rows),
        "user_tasks_beating_rules": sum(1 for lesson in rows if lesson.beat_the_rules()),
        "user_tasks_stalled": sum(1 for task in tasks if stalled(lessons, task)),
        "drivers": {RULES_DRIVER: sum(1 for lesson in rows if not lesson.beat_the_rules()),
                    AGENT_DRIVER: sum(1 for lesson in rows if lesson.beat_the_rules())},
    }
