"""The Spec: one file per Task holding the Intent and the Verifier written from it (DESIGN.md rule 1).

Code only, no model call. `schema` holds the records, `ground` the because and coverage rules
(trust gate 1), `compile` turns checks into Verifier atoms with the existing builders, and
`canfail` scores one Run under the Spec's Verifier (`judge_run`).
"""
