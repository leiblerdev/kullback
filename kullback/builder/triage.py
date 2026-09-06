"""The standard triage skill: how the Builder works a red light to green, read off where it kept failing (D150).

A skill is text the harness puts in the prompt, listed in the `<skills>` block and loaded from the
start of every Builder session, the way the Examiner's probe skill is; skills.py beside this file is
the other kind, the workdir SKILL.md files a model may rewrite under the skill gate (D130). It says nothing a gate does not enforce and names no domain, tool
or value. Build 12's model arm wrote the four habits here: repairs fired from the grouped status
without a zoom, a tool that stayed assisted after one recompile was never called again and was
named as "existing" in three closing lines, one Intent was repaired four times with hints that
restated the failure instead of reading the Runs, and "nothing changed" was read as done.
"""

from __future__ import annotations

TRIAGE_SKILL_NAME = "triage"

TRIAGE_SKILL = """Triage skill: how to work one red light to green, and when to leave it.

Zoom before you repair. The grouped status says which gate and which target; the zoom says why.
Call status(target="<tool or Task>") before every repair and put the failure's own words in the
hint: the exception line, the columns that differ, the phrase no Run evidences. A hint written
from the grouped line alone ("match the recording") tells the compiler nothing it did not know.

Rank by Tasks blocked. A tool body that crashes or differs on re-play blocks every Task whose
Reference calls it; one Task's Intent blocks that Task. Work the tools first, largest count
first, and only then the Intents. The count is in the status line: a gate with many failures
names how many and the first as an example.

Read the first line of every result. A repair result opens with the ruling of the target you
named. `cleared the gates` or `grounded:` means that one is done. `still assisted:` or `still
refused:` means it is not, and the text after the colon is the next hint's subject. The gate line
under it is the whole gate and says nothing about your target.

Stay on a tool until it clears or three hints have failed. A tool many Tasks call is not
"existing" or "remaining": after `still assisted:` the next call is the same tool with a hint
that answers the new failure line, never the same hint again. Only after three failed hints do
you leave it, and then the closing line quotes the last failure so the round report carries it.

An Intent gets two attempts. The first hint comes from the zoom: which phrase no Run evidences
and what every Run does say, in the Runs' own words. If `still refused:` comes back, zoom again
and write the second hint from what changed. After a second refusal, leave that Task and say so;
a Task refused twice is the round report's, not a third hint's.

`nothing changed` is a fact about your call, not about the build. It means the artifact you
targeted is byte-identical to before: the model wrote the same body or the same Intent. Change
what the hint says, or the target, before calling again; the same call gives the same answer.

Close only when the list is exhausted. Before answering with no tool call, check that every tool
still assisted has had three hints and every refused Intent two. The closing line names each
remaining red light with the last hint tried and the failure it left.
"""
