"""The Examiner: one Verifier per Task from the Builder's References, probes against it, repairs the gates
rule on, refusals the refuse gate admits, and findings for the Builder (D123, D127, D128, D133).

`derive.py` is the derivation (moved from builder/verifier.py, behaviour unchanged), `reference.py`
which Runs it may derive from (D111), `stage.py` the old derive_verifier stage body run outside the
pipeline, `plan.py` the session's store, `tools.py` the seven tools, `skills.py` the probe skill,
`extension.py` the extension on the agent core with its hooks and context guards, `agent.py` the
session driver. The Examiner never reads a tool body, the Starting state or the Environment, and
never writes them; the Builder never writes a Verifier or a probe."""
