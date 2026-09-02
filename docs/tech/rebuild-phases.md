# The rebuild, in phases

Decided in the grill of 2026-09-02 (decision log D120 to D130, ADR-0007). The shape follows huggingface/tau: a provider layer, an agent core that knows nothing about the application, and applications as extensions on the core. Each phase is small, gets its own note in this directory when it lands (`phase-N-<name>.md`: what it added, why, what the next phase builds on), leaves the tests green, and leaves the build's artifacts byte-identical until the phase that changes them on purpose.

## Target layout

```
kullback/
  ai/         providers, retry, pricing, the memo cache        imports nothing of ours
  agent/      messages, events, tools, loop, harness, session   imports ai
  runner/     records, canon, confinement, the frozen loop,     imports ai
              route, verdict, validate, budget, replay, regrade
  gates/      every accept-or-reject check, no model call       imports runner
  builder/    the Environment agent (extension on agent)        imports agent, gates, runner
  examiner/   the Verifier and probe agent (extension on agent) imports agent, gates, runner
  cli.py      frontend: consumes events, calls entrypoints
  tui/        frontend: same
  report.py   frontend: the customer-facing report
```

No `src/`. The dependency direction is an import-linter contract in CI.

## Phase 1: the move

No behavior change. Rename distribution, package and CLI from `harness` to `kullback`; split `src/harness/{builder,runner,shared}` into the layout above; add the import-linter contract; every test passes; `kullback build` on the fixture workdir produces byte-identical artifacts to `harness build` (compare `bodies.json`, `constraints.json`, `intents/`, `gates.json`, the Runs).

File map (today's path, then where it goes):

- `shared/provider.py`, `shared/pricing.py` to `ai/`
- `shared/records.py`, `shared/canon.py`, `shared/confinement.py`, `shared/budget.py` to `runner/` (confinement is a primitive both a gate and `verdict.py` call; the gate over tool bodies lands in `gates/` in phase 3)
- `runner/loop.py`, `route.py`, `verdict.py`, `validate.py`, `atom_context.py`, `boundary.py`, `gate_support.py`, `replay.py`, `regrade.py`, `state.py`, `judge.py`, `scorecard.py` stay in `runner/`
- `runner/pipeline.py` to `builder/` (only `build.py` uses it; it becomes the DAG scheduler in phase 4)
- `shared/parallel.py`, `shared/search.py` to `builder/` (their only callers)
- `builder/*` stays, including `verifier.py` until phase 3
- `shared/report.py` to `kullback/report.py`; `cli.py`, `tui.py` to `kullback/cli.py`, `kullback/tui/`
- `examiner/` is created empty with an `__init__.py` and a one-line docstring saying phase 5 fills it

`RunnerVersion` becomes the hash of the `runner/` package. Tests move to mirror the packages. `mutants/` and `scripts/` update their import paths.

## Phase 2: `ai` and `agent`

`ai/` gains a provider-neutral stream (tau's `AssistantMessageEvent` shape) over today's `Model.query`, keeping `MemoModel`, `RecordedModel` and `TestModel`. `agent/` is new: `messages.py` (`UserMessage`, `AssistantMessage`, `ToolResultMessage`, `ToolCall`), `tools.py` (`AgentTool` as pydantic args and result models plus an async executor; the JSON schema is derived from the args model, arguments are validated before the executor runs, results before they enter the transcript), `events.py` (the typed union: agent, turn, message, tool execution, stage, round, error), `loop.py` (stateless, sequential tool execution, drains steering after each tool batch and follow-ups when the run would otherwise stop), `harness.py` (owns the transcript, `prompt`, `continue_`, `steer`, `follow_up`, `cancel`, `subscribe`), `session/` (append-only JSONL tree with `id`, `parent_id`, `leaf`, entry types including `CompactionEntry`, `ToolSetChangeEntry`, `SkillChangeEntry`; active context is the root-to-leaf replay), `extensions.py` (`setup(api)`, `register_tool`, `add_prompt_section`, `on(event)`, `tool_call` and `tool_result` hooks where a raising `tool_call` hook blocks, `send_message(deliver_as="steer"|"follow_up")`), a fake provider, and tests that assert on events. Nothing uses it yet.

## Phase 3: `gates`

`builder/verifier.py`'s D79 suite, `runner/validate.py`'s check list, the fidelity bar and the confinement gate on tool bodies move to `gates/`. Each gate is a function over an artifact (or, for the skill gate, over two rounds of the memory tree) returning a ruling record; shaped so it can run inside a `tool_result` hook, but still called by the pipeline in this phase. Artifacts byte-identical. The gates' hash is recorded next to `RunnerVersion`.

## Phase 4: the Builder as an extension, targets over the DAG

`builder/pipeline.py` becomes the scheduler: stages declare inputs and outputs, a stage starts when everything it reads is complete, independent stages run side by side, each emits `stage_start` and `stage_end` with its name. Stages become tools with pydantic args (`recluster()` with fixed config, `grow(table, count)` under D107, `compile_tool(name)`, `replay(task)`, `reroll(task)`, `build(target)`); a tool whose inputs are stale resolves upstream first. Gates run in `tool_result` hooks and append their ruling to the result. A raising `tool_call` hook blocks any write into `gates/` or `runner/`. The Builder runs with one message, `build(target="environment")`, repair verbs off; the comparison against phase 3's artifacts is the phase's test.

## Phase 5: the Examiner

`derive_verifier` leaves the Builder. The Examiner is a second extension with its own session, prompt and skills; it reads traces, Intents, re-rolls and the Runner's Runs, never tool bodies. The probe pool per Task (monotone) and the loosening gate (one-directional, with a pool that grows only from frontier re-rolls of any round and live production Runs, D133) land in `gates/`; when the gate rejects a repair the Examiner may spend allowance on more re-rolls of that Task, and probing a Task stops after three consecutive probes fail. The Examiner's standard probe skill carries the fuzzing paper's bug classes. Per Task the false-rejection number (held-out re-rolls the required atoms wrongly fail) is reported next to the trusted-Verifier count. Rounds and `round_end` with the counts; the three exits (done, stalled with `stall_rounds`, ceiling); turn-taking on one event stream; the Examiner's findings reach the Builder as follow-ups; the refuse verb over re-rolls.

## Phase 6: repair, ratchet, lesson, skills

The five repair verbs (recompile, grow, rewrite-skill, refuse-Task, escalate). The ratchet and the lesson as `tool_result` handlers on `compile_tool`. Stage prompts move out of `build.py` into `skills/<name>/SKILL.md` in the workdir, listed by name in the system prompt and loaded on demand; a skill edit is a node in the memory tree with its content hash, every artifact records the skill hash it was compiled under, and the skill gate accepts it by a paired sequential test (D132): the same artifacts compiled under both hashes, paired gate differences accumulated across rounds, alpha 0.05, tentative until decisive, promoted or reverted when decisive, the trunk parent advancing only on promotion, and promoted skills re-checked on later rounds and demoted the same way. The gate is code over gate counts, never an LLM judgment. Per-round allowance per agent for replay and re-rolls, sized from round one.

## Phase 7: context tools and the floor

`forget(entry_ids, note)`, `recall(entry_id)`, `load` and `unload` of tools and skills, validated against the active path (never `session_info`, an unacted gate ruling, the current turn's results, an entry an open finding or unfinished repair refers to, or the recent tool-output payloads, D131). A `recall` lands at the end of context, marked. A soft cap of 20 loaded tools, shown to the model. The 40% line: when the model has not kept the context under it, code compacts tau's way and records the fallback as its own entry. The context estimate and the line in every tool result (no precedent; part of the experiment). First numbers per agent per build: forget and load call counts, fallback count, gate pass per round against context fill; arms: tools on, tools off (code only), and a file-shaped variant of the same tools.
