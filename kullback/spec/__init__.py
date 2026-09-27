"""The Spec: one file per Task holding the Intent and the Verifier written from it (DESIGN.md rule 1).

Code only, no model call. `schema` holds the records, `ground` the because and coverage rules
(trust gate 1), `compile` turns checks into Verifier atoms with the existing builders, and
`canfail` asks whether the compiled Verifier can fail at all (trust gate 2).
"""

# Where a Task's Verifier comes from: the Spec written from the Intent, or the derivation from Runs.
VERIFIER_FROM = ("intent", "runs")
