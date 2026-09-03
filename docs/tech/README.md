# Tech

What we build. Currently minimal: the product is validated by hand first.

## Scope

1. **Website**: landing page with a sharp value prop (see `website.md`).
2. **SDK wrapper / log-drain**: capture traces (see `sdk-wrapper.md`).

Everything else (routing, evals, model optimization, hosting) comes later, gated on validation.

## Files

- `website.md`: landing page + value prop
- `sdk-wrapper.md`: thin wrapper / log-drain plan
- `progress.md`: log
- `rebuild-phases.md`: the rebuild of the harness in seven phases; each phase leaves a note here when it lands, phases 1 to 5 and phase 7 landed 2026-09-02, phase 6 remains
- `phase-1-the-move.md`: the `kullback` package at the repository root, the import-linter contract, RunnerVersion over the whole runner package, and the byte-identical check with its ingest_version proviso
- `phase-2-ai-and-agent.md`: the provider-neutral stream and the agent core (messages, tools, events, the stateless loop, the harness, the session tree, extensions), and why each is shaped as it is
- `phase-3-gates.md`: the `kullback.gates` package (the D79 suite, the artifact gates, the fidelity bar, the confinement gates, the scorecard), the ruling record, the registry, the gates hash beside RunnerVersion, and the byte-identical check
- `phase-4-builder-extension.md`: the pipeline as a scheduler over one declared graph, the stages as tools with pydantic arguments, the gate rulings in the `tool_result` hook and the raising `tool_call` hook over `gates/` and `runner/`, the code driver behind `kullback build` and the model-driven path behind `--agent`, the twelve rulings brought into the registry, and the byte-identical check with the one renamed stage
- `phase-5-examiner.md`: the Examiner as the second extension with `derive_verifier` moved out of the Builder, the rounds `kullback build` runs (one stream, four beats, one agent at a time), the two drivers and what `--agent` changes, the findings drained into the Builder as follow-ups, the counts on `round_end` and where each comes from, the three exits with `stall_rounds` and the per-agent allowance, the CLI lines, the TUI rows and the report's trusted, refused and false-rejection lines, and the byte-identical check
- `test-audit-2026-09-02.md`: the audit of the suite the rebuild left behind, what came out per package (1,643 tests to 1,625) and why, the two high findings and their fixes, and the mocks, names and findings deliberately left open
- `phase-7-context-tools.md`: the context estimate and the 40% line, the session record behind a harness, the forget, recall, load and unload tools with every refusal rule, the code floor and how its fallback is recorded, the three arms, the counters, and what phases 5 and 6 must call for the guards to bite
