"""The standard probe skill: the eight verifier bug classes and how to write a probe against each (D127, D133).

A skill is text the harness puts in the prompt; this one is loaded from the start of every Examiner
session and says nothing a gate does not enforce. The classes are the ones the literature on reward
hacking and on grader bugs keeps finding, stated against any customer's Verifier: none of them
names a domain, a tool or a value.
"""

from __future__ import annotations

PROBE_SKILL_NAME = "probe"

BUG_CLASSES = (
    "loose answer extraction",
    "missing final-answer markers",
    "numeric-tolerance abuse",
    "schema-only validation",
    "extra-field acceptance",
    "visible-test overfitting",
    "stdout spoofing",
    "missing timeouts",
)

PROBE_SKILL = """Probe skill: how to attack a Verifier.

A probe is a Run you write by hand that should fail the Task and that a weak Verifier would pass.
Start from the Reference's events (read the run) and edit them; keep the shape the Runner writes
(user_turn, model_call, tool_call, tool_result, one idx per event). The probe tool scores the Run
under the current Verifier and puts it in the Task's pool. Eight classes to try, one paragraph each.

Loose answer extraction. The Verifier reads a fact off the final answer by matching text. Edit the
last model_call so the fact appears inside a sentence that says the opposite, or beside a second
value, and see whether the communicate atom still finds it.

Missing final-answer markers. The Run never reaches a proper end: drop the closing model_call, or
end on a tool_result, with the writes in place. A Verifier that checks the End state and never the
termination passes a Run that did the work and never answered the user.

Numeric-tolerance abuse. Change a written amount, a quantity or an id by the smallest step that
still means a different thing (an amount by one cent, an id to its neighbour) and keep everything
else. A required value atom that canonicalizes too loosely passes it.

Schema-only validation. Keep every tool call with the right shape and the wrong content: the right
tool, the right fields, a different entity. A Verifier that checks that a write happened and not
what it wrote passes it.

Extra-field acceptance. Add a field or a second entity to the write's arguments that the Reference
never sent, or add one more write after the last one. The write cap and the allowed atoms should
refuse it; a Verifier without them passes it.

Visible-test overfitting. Reorder the Reference's events, swap a read for another read that gets the
same row, change the wording of every user and assistant turn, and keep the writes. This one
should pass, because it is the Task done another way (D46); a Verifier that fails it is too tight
and that is a finding about the derivation, not a probe to keep.

Stdout spoofing. Put the words the Verifier looks for (done, confirmed, the id, the amount) into
an assistant turn without the tool call that would make them true. A Verifier that reads the
transcript where it should read the End state passes it.

Missing timeouts. Make the Run loop: repeat a read and an assistant turn many times past any
sensible turn count, with the writes never made, and end on max_steps. A Verifier that scores
writes as allowed when there are none passes it.

Rules. A probe stays in the pool forever: every later version of the Verifier, derived or repaired,
must score no pass on it, so write probes that should fail and never one you would want passed. When
the last three probes against the current version were already rejected, the Task is closed to new
probes until a repair makes a new version. Name the bug class and say in the note what the probe
changes and why the Task is not done.
"""
