"""The Simulated user as an agent of the harness, beside the Builder and the Examiner (D214).

The other half of an Environment is the person on the other end of the conversation, and it was the
one part of the harness nobody had made an agent: a set of rules read off one trace, which answers
what its cues match and runs out of scenario on the rest. It is a package now, on the same shared
agent brain (`kullback.agent`) the other two sit on, with its own curated context, its own tools,
its own guards, its own fidelity score and its own lesson loop.

  rules.py       the rule-driven user (D44, D77, D210), moved here whole. It is the floor: the agent
                 drives a Task only where it beats this offline, and this answers every beat a guard
                 drops the agent's turn on
  vocabulary.py  what a fact is and the generic core, which both halves read
  context.py     what is curated for one Task, all mined, each item a tagged prompt section
  skills.py      the system prompt in this harness's order
  tools.py       my_facts, my_goal, what_i_said, end_run, and consult where a recording justifies it
  extension.py   the setup the harness loads, and the two refusals that are code and not prompt
  agent.py       one turn per model call, guarded, with the rules underneath
  guards.py      the five checks after the model, and the end protocol
  fidelity.py    how close a driver's turns are to the recorded ones, per Task and per corpus
  lesson.py      what the score teaches the next round, and when to stop paying

The package imports the agent core, the provider layer, the gates' path rules and the Runner's
records, and never the Builder or the Examiner: the Builder derives the vocabulary and drives the
Runs, and a package the Builder depends on may not depend back.
"""

from __future__ import annotations

from kullback.user.agent import AgentUser, run_user
from kullback.user.context import Persona, TaskContext, curate
from kullback.user.fidelity import TaskScore, TurnScore, score_driver, score_turn
from kullback.user.guards import EndProtocol, GuardOutcome, Guards
from kullback.user.lesson import Lesson, drives, stalled
from kullback.user.rules import SimulatedUser, derive_user_rules, end_of_run, ends_by_kind

__all__ = [
    "AgentUser",
    "EndProtocol",
    "GuardOutcome",
    "Guards",
    "Lesson",
    "Persona",
    "SimulatedUser",
    "TaskContext",
    "TaskScore",
    "TurnScore",
    "curate",
    "derive_user_rules",
    "drives",
    "end_of_run",
    "ends_by_kind",
    "run_user",
    "score_driver",
    "score_turn",
    "stalled",
]
