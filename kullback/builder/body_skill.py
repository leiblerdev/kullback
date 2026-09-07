"""The body skill: how a tool body is written so it clears the gates, read off where bodies kept failing (D168).

A skill is text the harness puts in the prompt and a gate enforces; this one is in the system prefix
of every body-writing call (`compile_env._stable_system`), the way the triage skill is in every
Builder session (D150). It names no domain, tool or value. It was written from four builds over
three customers' traces, body by body: a body that raised TypeError for an argument the recorded
call never passed; a body that read a column no table holds; a body that answered null where the
recording held a value from a related row; an appended history entry with the wrong sign, then the
wrong keys, then an index past the end; a status text with the segments in the body's own sequence;
a search that answered an empty list where the recording listed rows; a body that bound a
placeholder name copied out of a refused call's payload; a body that answered every argument set
the same way. Each line below is one of those, turned into the habit that avoids it.
"""

from __future__ import annotations

BODY_SKILL_NAME = "body"

BODY_SKILL = """Body skill: how to write a body that replays the recording, and what to do when it does not.

The signature is the recorded call's. The method receives exactly the arguments the recorded calls
show, no more. What a call does not pass, the body finds in the world: the one row the Starting
state holds for the subject, the row the last write touched, the row a foreign key names. It never
adds a parameter, and it never calls a helper of its own with an argument the call did not carry.

Only the listed columns exist. A row has the columns the tables above list and no other; reading
one that is not listed raises AttributeError on every call. A value you need that is not a column
of this row lives on a related row (follow the id) or inside a nested column (the tables say which
and show the form that reads it). Never invent a column and never default one to None: a field the
recording fills is looked up, and a null where the recording holds a value fails replay_fidelity.

Copy the shape of what is already there. An entry appended to a list column has the keys, in the same
sequence, the sign convention and the id form of the entries the recording shows beside it. A number
the recording derives (a fee, a total, a refund) is derived the same way: read two recorded calls
and take the rule that gives both, never a constant that fits one.

Assemble text the recorded way. A result that is one string built from several fields is built with
its segments in the sequence and wording the recorded results show, every segment the world's values call for
and none the recording never shows. Derive the format from the union of the examples, not from the
first.

An empty answer is a wrong lookup before it is an empty world. When the recording lists rows and the
body lists none, the key the body used is wrong (a nested key, a date, a normalised id) far more
often than the world is empty. Check how the recorded rows are keyed before answering [].

Answer by the arguments, then by the world. Different arguments that the recording answered
differently must get different answers from the body (non_trivial); identical arguments the
recording answered differently were answered from a world an earlier write changed, and the body
reads that state rather than a constant.

Names in a refused call's payload are not names. A recorded error that names a tool, a function or
a placeholder is a message to raise, not a symbol to bind; the confined gate refuses a body that
names anything the module does not define.

Ids and values come out of the world. A literal copied from a recorded call (an id, a code, a row's
value) is refused by memorised_values however well it replays; look the row up instead. Words that
are the customer's vocabulary (an enum member, a status word, a key name) are fine.

When a gate fails, read its first line before changing anything. `hard columns differ` names the
column and both values: fix that column's rule. `expected a result, got <Error>` is a crash on a
recorded call: fix the lookup that raised. `expected error <class>, got None` means the body
accepted what the recording refused: add the check with the customer's own message. `answers every
call the same way` means the body ignored the arguments. A second attempt that repeats the first
attempt's failure has not read the line."""
