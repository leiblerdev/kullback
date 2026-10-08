# Modules

Every `.py` file under `kullback/`, one row each, grouped by package. The line says what the module does, read off its docstring and its top-level names. Package boundaries and the build flow are on [the architecture page](../architecture.md).

## kullback (top level)

The top-level modules sit outside the import-linter layers contract. The last column of the boundaries table on the architecture page says who reads each one.

| Module | What it does |
|---|---|
| `kullback/__init__.py` | Package marker with the one-line description of the Harness. |
| `kullback/cli.py` | The Typer app behind the `kullback` command: every subcommand, each importing its module only when it runs. |
| `kullback/claims.py` | Checks what a Run's transcript says it did against what its state received, and the partial completion number (D223). |
| `kullback/consistency.py` | Shrinks a failing call sequence to its shortest failing subsequence, for the laws checker. |
| `kullback/container_cheats.py` | The cheat suite and birth scan for container Tasks: faked tests, faked exit codes, leaked ground truth. Not yet called from a command. |
| `kullback/container_grader.py` | Grades an exported end state as an opaque tar inside a fresh, locked-down container. |
| `kullback/container_world.py` | The disposable container backend: one fresh container per world, commands, limits, export. |
| `kullback/derive.py` | Derives one Task's Verifier from its Reference and stored re-runs (D42, D43, D91); the Reference stage's survivor choice and `synthesise` use it, the Examiner never writes with it (D320). |
| `kullback/difficulty.py` | Every Task's objective difficulty and the buckets a round reports trusted coverage over (D209). |
| `kullback/domain.py` | Reads a domain's public material into task archetypes and coverage gaps, with a guarded fetcher (D225). |
| `kullback/graph.py` | The tool-call dependency graph the recordings show, and walks over it (D224). |
| `kullback/laws.py` | Laws that hold for any tool, checked by generated call sequences with no model (D258). |
| `kullback/round_snapshot.py` | One round's Task table, written once and never rewritten (D218). |
| `kullback/sampling.py` | One keyed draw for every per-Task or per-Run sample, stable when counts move (D212). |
| `kullback/store.py` | `WorkdirStore`: the durable, locked read and write of a declared workdir artifact. |
| `kullback/store_specs.py` | Declares every single-file workdir artifact once, with its path, owner and shape check (D267). |
| `kullback/synthesise.py` | Turns a graph walk into a synthetic Task: bind arguments, run in the rebuilt world, derive and store apart (D224). |
| `kullback/variants.py` | Second paths synthesised from a Run's own calls, code only (D199). |

## kullback/ai

| Module | What it does |
|---|---|
| `kullback/ai/__init__.py` | The provider layer's public names: one contract over every model. |
| `kullback/ai/_provider_events.py` | The raw events one adapter reports before anything is assembled. |
| `kullback/ai/_sse.py` | The shared streaming POST: retry, cancel, one SSE line at a time through an adapter's parser. |
| `kullback/ai/anthropic.py` | The Anthropic Messages API, streamed. |
| `kullback/ai/cache.py` | Prompt caching for every adapter: request fingerprint, TTL, break points. |
| `kullback/ai/events.py` | The canonical stream events of one assistant message, whoever produced it. |
| `kullback/ai/http.py` | Shared HTTP plumbing: read budget, client, request body hash, request id. |
| `kullback/ai/http_errors.py` | The provider errors and the safe detail a message may carry. |
| `kullback/ai/messages.py` | The transcript's message types, defined here because the stream assembles them (D121). |
| `kullback/ai/model_catalog.py` | Which models an account can reach, read off the models.dev snapshot. |
| `kullback/ai/model_limits.py` | Context window and request rules per model, which decide when the core compacts. |
| `kullback/ai/openai_compatible.py` | OpenAI-shaped endpoints streamed: chat completions and the Responses API. |
| `kullback/ai/pricing.py` | Live prices from a snapshotted models.dev catalog, with local overlays (D116). |
| `kullback/ai/provider.py` | The `ModelProvider` contract, the `Model` handle, live-call switches and the `TestModel`, `RecordedModel` and `MemoModel` replay handles. |
| `kullback/ai/replay.py` | The replay seam: a stored reply answered through the same contract as a live endpoint. |
| `kullback/ai/retry.py` | When a failed provider call is retried and how long the caller waits. |
| `kullback/ai/sigv4.py` | AWS Signature Version 4 for one request, standard library only (Bedrock). |
| `kullback/ai/stream.py` | Turns a provider stream or a whole reply into canonical events. |
| `kullback/ai/tool_call_ids.py` | Cleans tool call ids for a provider without merging two distinct ids. |
| `kullback/ai/usage.py` | The `Usage` token record and the per-provider readers that fill it (D76). |

## kullback/agent

| Module | What it does |
|---|---|
| `kullback/agent/__init__.py` | The shared agent core's public names; it knows nothing about the application. |
| `kullback/agent/base_tools.py` | The base tools bound to a root: read, write, edit, grep, find, ls, inspect, web_search, bash. |
| `kullback/agent/bus.py` | One append-only, sequence-numbered event log per workdir. |
| `kullback/agent/compaction.py` | Folds the older prefix of a transcript into one summary, recent turns kept. |
| `kullback/agent/context.py` | Context accounting: what the active context costs and where the compaction line is. |
| `kullback/agent/events.py` | The typed events one run of the loop emits, plus stage and round events. |
| `kullback/agent/extensions.py` | The `setup(api)` extension API through which an application registers tools, prompt sections and hooks. |
| `kullback/agent/harness.py` | The stateful harness: transcript, tool registry, queues, hooks, subscribers. |
| `kullback/agent/loop.py` | The stateless agent loop: one assistant message, its tool calls, the queues, until the model stops. |
| `kullback/agent/messages.py` | Re-exports the message types from `kullback.ai.messages`. |
| `kullback/agent/prefix_check.py` | Checks that each request is a byte prefix extension of the last, so caches hold (G24). |
| `kullback/agent/provider.py` | The provider contract as the loop sees it, and `provider_for` a model handle. |
| `kullback/agent/reading.py` | Outline-first reading helpers: outline, page, locate, part, select, around (G21). |
| `kullback/agent/session/__init__.py` | The session: an append-only JSONL tree and its active path. |
| `kullback/agent/session/entries.py` | The session entry types, one pydantic model each. |
| `kullback/agent/session/store.py` | The JSONL session store: load, append, branch, active path with compactions applied. |
| `kullback/agent/skills.py` | Skills as named texts put in the system prompt, loaded from a directory. |
| `kullback/agent/tool_history.py` | Repairs a transcript whose tool calls and results do not line up. |
| `kullback/agent/tools.py` | `AgentTool` and `ToolRegistry`: pydantic args in, pydantic result out, rulings attached. |
| `kullback/agent/types.py` | JSON-like type aliases for tool arguments, results and payloads. |

## kullback/runner

| Module | What it does |
|---|---|
| `kullback/runner/__init__.py` | Package marker for the Runner. |
| `kullback/runner/arith.py` | The one code-owned arithmetic evaluator handed to model-written tool bodies. |
| `kullback/runner/atom_context.py` | What an atom predicate may look at, and the confinement gate that certifies it. |
| `kullback/runner/boundary.py` | The D89 and D91 import boundary as an AST scan, and the `runner_version` hash. |
| `kullback/runner/budget.py` | Cost and token accounting per call, stage and build, the context cap (D65) and spend ceiling (D86). |
| `kullback/runner/cache_view.py` | What the prompt cache did for one build, per stage, read off the feed. |
| `kullback/runner/canon.py` | The canonicalizer (D39), equality by column class (D73) and the equivalence table (D84). |
| `kullback/runner/confinement.py` | The AST check that a model-written predicate cannot reach outside its case. |
| `kullback/runner/feed.py` | The live feed of one build, `events.jsonl`, in the order things happened. |
| `kullback/runner/gate_support.py` | The `GateResult` helper, field access and canonical equality every gate uses. |
| `kullback/runner/heartbeat.py` | One small file per running build, for the screen's session list. |
| `kullback/runner/judge.py` | The agentic judge (D92): read-only state views, pre-run checks, two judges and a disagreement queue. |
| `kullback/runner/loop.py` | One Run, turn by turn: query the model, route tool calls, one JSONL line per event (D90). |
| `kullback/runner/parallel.py` | A bounded worker pool shared by the Builder and the Examiner (D118, D163). |
| `kullback/runner/real_tools.py` | Tools run for real in a container world, beside the imitating routes (D262). |
| `kullback/runner/records.py` | Every record passed between modules, as pydantic models that round-trip through JSON. |
| `kullback/runner/regrade.py` | Re-scores stored Runs against a new Verifier or Environment without re-running them (D97). |
| `kullback/runner/replay.py` | Replays a recorded Trace through the rebuilt Environment, scored call by call. |
| `kullback/runner/route.py` | Answers one tool call: real, then code, then the recording, then an LLM stand-in. |
| `kullback/runner/state.py` | A Task's Starting state, its overlay and the write path into a toolkit's db. |
| `kullback/runner/target.py` | One scorer for a Verifier atom, shared by the Verdict and the gates (G3). |
| `kullback/runner/tool.py` | The Runner as a tool: run, replay, replay_all, probe and reroll over a built Environment. |
| `kullback/runner/transaction.py` | Snapshot and restore of a toolkit's db around one tool call. |
| `kullback/runner/validate.py` | Re-exports what certifies the Runner: its import boundary and its version. |
| `kullback/runner/verdict.py` | Decides one Run's pass or fail from its End state, in code only (D43, D46, D94). |
| `kullback/runner/world/__init__.py` | The world a Run executes in. |
| `kullback/runner/world/birth.py` | Birth checks for a Task: the scripted walk passes, do-nothing fails, arguments trace. |
| `kullback/runner/world/clock.py` | The world clock a tool body reads, taken from the recording (D283). |
| `kullback/runner/world/environment.py` | `BuiltEnvironment`: a workdir or fetched package read without the Builder (G1). |
| `kullback/runner/world/episode.py` | `Episode`: reset, step and a reward by code over one Task (G1). |
| `kullback/runner/world/law_world.py` | Adapts one Task of a built Environment to the laws protocol (D258). |
| `kullback/runner/world/loading.py` | Loading toolkits, overlays and seeds, moved down out of the Builder (G1). |
| `kullback/runner/world/recorded.py` | What one recorded call witnessed for the tool context: new ids, times, held-out values. |
| `kullback/runner/world/solve.py` | The solve-rate table: a cheap model over the Tasks through the episode interface. |

## kullback/gates

| Module | What it does |
|---|---|
| `kullback/gates/__init__.py` | The gate registry: `Ruling`, `GateSpec`, which gates run over which path (D122). |
| `kullback/gates/artifacts.py` | The per-artifact gates: ingest, mine, compiled body evidence, policy, environment, user rules, leaks, verifier, regrade. |
| `kullback/gates/bindings.py` | Maps a written path to the gates that run on it and the rows each refusal names. |
| `kullback/gates/confinement.py` | Refuses a model-written tool body or predicate before it runs anywhere in process. |
| `kullback/gates/counts.py` | The workdir counts read off rulings: trusted share, fidelity rate, goal met. |
| `kullback/gates/fidelity.py` | The replay fidelity bar at call, Reference and oracle grain (D39, D51, D80, D108). |
| `kullback/gates/hook.py` | The `tool_result` hook that runs the path-bound gates on an agent's writes. |
| `kullback/gates/ledger.py` | `GateLedger`: `gates.json` under one lock, shared by both agents (D122, D128). |
| `kullback/gates/loosening.py` | One-directional loosening over the legitimate pool and the false-rejection number (D127, D133). |
| `kullback/gates/probes.py` | The monotone probe pool (D127) and the stop rule (D133). |
| `kullback/gates/scorecard.py` | The D62 scorecard: tool fidelity, Task coverage, user fact consistency, Verdict agreement. |
| `kullback/gates/stages.py` | Rulings over each Builder stage's artifact: cluster, compile, intent, readers, rerolls, export, vocabulary. |
| `kullback/gates/tool_runs.py` | The eight rulings over what a generated tool body did in the sandbox. |
| `kullback/gates/trust.py` | The refuse ruling (D128) and what a trusted Verifier is (D126). |
| `kullback/gates/trust.py` | The Reference pool: Tasks waiting on a Reference, untrusted until a later Run earns one. |
| `kullback/gates/verifier_suite.py` | The nine D79 checks over a Verifier. |

## kullback/user

| Module | What it does |
|---|---|
| `kullback/user/__init__.py` | The Simulated user as an agent beside the Builder and the Examiner (D214). |
| `kullback/user/account.py` | The agent user's two readings: choices from recordings, tagged by source, and its own account, read only, cut to its rows and to columns recorded users stated. |
| `kullback/user/agent.py` | `AgentUser`: a model-driven Simulated user, one turn per call, the rules as its floor. |
| `kullback/user/context.py` | The mined, per-Task context the agent user is given: persona, facts, consultations. |
| `kullback/user/ends.py` | How a Run ended and whether a write moved the world. |
| `kullback/user/examples.py` | Assembles masked example spans from Traces as candidate user examples, withholding held-out ones. Not yet called from the package. |
| `kullback/user/extension.py` | The agent user as an extension on the core. |
| `kullback/user/factory.py` | The one place a Simulated user is built, for every caller (G5). |
| `kullback/user/fidelity.py` | User fidelity: how close a driver's turns are to the recorded ones (D214 rule 5). |
| `kullback/user/guards.py` | Code guards after every model turn: grounding, no invented values, account values only when asked, end protocol (D214 rule 4). |
| `kullback/user/lesson.py` | What the fidelity score teaches the next round's user context (D214 rule 6). |
| `kullback/user/population.py` | Which recordings of a Task may seed a user, and a seeded pick among them. Not yet called from the package. |
| `kullback/user/rules.py` | Derives the rule-driven user's rules from one trace (D44, D77). |
| `kullback/user/signals.py` | Mines how a recording ended from its turns, with model labels accepted or dropped. Not yet called from the package. |
| `kullback/user/simulated.py` | `SimulatedUser`: replies as the recorded user from the rules and the world. |
| `kullback/user/skills.py` | The agent user's system prompt, in the harness's section order. |
| `kullback/user/tools.py` | The agent user's own tools: my_facts, my_goal, what_i_said, end_run, consult where the recording looked something up, and my_choices and my_account where the caller built them. |
| `kullback/user/value_strip.py` | Takes system-known values out of an Intent or a user answer, so the user says only what a user said (D196, D210). |
| `kullback/user/vocabulary.py` | `FieldSpec` and `Vocabulary`: how a fact is stated, asked for and stored (D115). |

## kullback/examiner

| Module | What it does |
|---|---|
| `kullback/examiner/__init__.py` | The Examiner: reads and reviews, one review per Task, findings with rows and the edit; it writes no Verifier (D320). |
| `kullback/examiner/domain_tools.py` | The Examiner's tools: rule, finding (with edits and note_ruling), no_finding (a review that found nothing, with its reason, in `examiner/reviews.json`), check_reference, reject_reference. None writes a Verifier or runs a Run (D320). |
| `kullback/examiner/exam_files.py` | The Examiner's root `exam/`: the read surface it is given and its findings file. |
| `kullback/examiner/findings.py` | Findings the round's records file by themselves before the model speaks (D170). |
| `kullback/examiner/judge.py` | The reference judge as a bounded agent, one-shot judge as fallback (D12, D92). |
| `kullback/examiner/lifecycle.py` | Retires a derived artefact whose source was withdrawn (D208). |
| `kullback/examiner/plan.py` | `ExaminerPlan`: what one Examiner session is given and the store its gates rule over. |
| `kullback/examiner/prompt.py` | The Examiner's review prompt in GEPA order: one review per Task, writes nothing, runs nothing. |
| `kullback/examiner/reference.py` | Chooses References by the D111 rule and checks constraints against them. |
| `kullback/examiner/reference_check.py` | `check_reference`: the evidence on a Task's Reference in one view, code only, for the Examiner to judge. |
| `kullback/examiner/reference_check.py` | `reject_reference`: excludes a wrong Reference's End state, re-picks, and pools the Task in `exam/reference_pool.json` when none is left. |
| `kullback/examiner/rule_tool.py` | The `rule` tool: a ruling must quote a because or a Run turn, lands whole in exam/rulings/ and on the bus as a code; the per-Task spec view with the Verifier's expected cells and their sources. |
| `kullback/examiner/session.py` | The Examiner extension and `examine`, called by the Builder. |
| `kullback/examiner/skills.py` | The review skill: what one review per Task reads and files (D320). |
| `kullback/examiner/stage.py` | The Reference stage: chooses References by the D111 rule and writes `references.json`, `task_status.json`, `constraints_check.json`; under D320 it writes no Verifier (every Task is the Spec's). |

## kullback/spec

| Module | What it does |
|---|---|
| `kullback/spec/__init__.py` | The Spec package: the Spec writes and repairs every Verifier (D320). |
| `kullback/spec/schema.py` | `IntentFact`, `SpecIntent`, `Check`, `Spec` and `tier_of`; one file per Task at `workdir/spec/<task_id>.json`. |
| `kullback/spec/ground.py` | Trust gate 1: `valid_because` (an Intent quote of 12 characters or a named policy section), `coverage`, `ground_spec`. |
| `kullback/spec/compile.py` | `compile_spec`: checks to Verifier atoms with the existing builders, no new kinds, ungrounded checks dropped and counted. |
| `kullback/spec/canfail.py` | Trust gate 2: `can_fail` scores the D79 empty, wrong entity and swapped value Runs against the Spec's Verifier. |
| `kullback/spec/events.py` | The Spec and ruling events on the workdir bus: spec.written, spec.repaired, spec.defended, spec.repair_deferred, ruling.filed, task.set_aside; ids, counts and reason codes from `REASON_CODES`, never free text. |
| `kullback/spec/router.py` | `Router`, a bus subscriber that moves each ruling once by target (run re-roll, check or intent to `writer.repair_for_ruling`, environment Builder note), counts rounds and sets a Task aside past `ROUNDS_CAP`; `is_set_aside` is the one set-aside rule. |
| `kullback/spec/trust.py` | The four gates in order (grounded, can fail, an independent Run passes, no open ruling) and `tier_of_task`: trusted, replay_only, untrusted, set_aside, each row naming the failing gate and the Runs scored; gates 2 and 3 score the Verifier file the Spec writer wrote (`spec/verifiers/<task_id>.json`), and a Task without one is untrusted; the replay prefix is read off the Runner's replay index. |
| `kullback/spec/split.py` | The Reference class of a Task from the benchmark sidecar (right, wrong, mixed, unknown), measurement only; scripts/measure/trust_split.py imports it. |
| `kullback/spec/report.py` | The per-build tier report (`tiers.json`): counts per tier, trusted split by Reference class (`split.py`) when the sidecar is there, one table line in docs/builds/README.md's format. |
| `kullback/spec/writer.py` | The Verifier writer (intentv r1): checks with becauses and fact ids from the Intent, the recorded policy, the tool schemas and the Starting state; `write_verifier` saves the Spec and its whole Verifier (expected, forbidden, conduct from `end_state.py`, fact and question atoms), the Runner's file the same bytes; `repair` answers a ruling on the ruled checks only; `repair_for_ruling` is the router's entry point, which saves, rewrites the Verifier files and publishes spec.repaired or spec.defended. |
| `kullback/spec/writer_tools.py` | The writer's two read-only Starting state lookups, the bounded session (4 read rounds, a spend ceiling, a forced final answer), `refuse_run_inputs`, which keeps every Run, Reference, Verifier and End state out, and `policy_sections_of`, the one list of policy names a because may cite (policy headings plus compiled rule names). |
| `kullback/spec/must_not.py` | The forbidden side of the writer's Verifier: scope atoms for uncovered write tools and the compiled policy prohibitions; `tier_of` by atom kind. `kullback/examiner/must_not.py` re-exports it. |
| `kullback/spec/atoms.py` | `shape_atom` and `no_write_atom`, the two generated Hard rules, moved from the Examiner so the Spec never reads it; `derive` re-exports them. |
| `kullback/spec/text.py` | `STOPWORDS`, `APOSTROPHE_RE`, `TOKEN_RE` and `normalise`, moved from the Builder so the Spec never reads it; `cluster` and `intent` re-export them. |
| `kullback/spec/intent.py` | The intent miner: `mine_intent` runs one agent session per Task on the Examiner's model path, capped in calls and USD, and writes the SpecIntent part of `workdir/spec/<task_id>.json`. |
| `kullback/spec/stage.py` | `write_specs`: the build's Spec stage; for each named Task without a Spec and Spec Verifier, mines the Intent, writes the Spec and both Verifier files and publishes spec.written, priced under the `spec` stage, a failing Task counted and skipped; a written Spec whose Task gained a Reference since has its gates rewritten by code (`refresh_gates`). |
| `kullback/spec/end_state.py` | A Task's end-state gates by code from its Reference Run (D315, D320): `reference_run`, `gates_of` (expected end states with alternatives, forbidden list) and `conduct_of` (confirmation, refusal, hand-off from grounded demands, each with its source). |
| `kullback/spec/review.py` | The Spec answers a review (D320): atoms and text edits applied by code, the Verifier rewritten, the gates run; one round per call on the Spec, pending at the ceiling; `route_findings` is the Builder examine tool's seam. |
| `kullback/spec/intent_tools.py` | The miner's view and tools: user turns of every recording with the agent's line before, never a tool call or result; `add_facts` checks each fact's words against its turn; witnesses and `counts` (rule 5) in code. |

## kullback/builder

| Module | What it does |
|---|---|
| `kullback/builder/__init__.py` | Package marker for the Builder. |
| `kullback/builder/body_skill.py` | The body skill: how a tool body is written so it clears the gates (D168). |
| `kullback/builder/cache_reach.py` | Hashes a module's import closure so code-keyed caches invalidate (G29). |
| `kullback/builder/run_user.py` | The Simulated user the run tool meets: the rule-driven user, or the agent user over it with a model, from the Reference's user rules (D214). |
| `kullback/builder/cluster.py` | Groups Runs into Categories by write-tool set, then Tasks by intent (D83). |
| `kullback/builder/compile_env.py` | Builds the world: `db.json` by inverse replay, per-Task overlays, the tau2 shape, each body through its gates. |
| `kullback/builder/domain_tools.py` | The Builder's tools: ingest, derive_world, grow, replay, rulings, run, status, examine, note_task. |
| `kullback/builder/effects.py` | What a write changed on rows its arguments never named, read off the recording (D215). |
| `kullback/builder/env_files.py` | The Environment on disk under `env/`: one file per tool body, rendered modules, read surface. |
| `kullback/builder/ingest.py` | Stores customer files byte for byte and derives Traces with raw pointers (D66, D67, D95). |
| `kullback/builder/intent.py` | Writes a Task's Intent and refuses any phrase without a span in every member Run (D47, D83). |
| `kullback/builder/lesson.py` | What a stalled tool body is told: differing leaves, relations, unreached lines (D211). |
| `kullback/builder/memory.py` | The Builder's version tree and the cross-customer lessons file with anonymization (D64, D87). |
| `kullback/builder/mine.py` | Mines ToolSigs and the EntitySchema out of ingested traces (D68, D70, D72, D73). |
| `kullback/builder/parallel.py` | Re-exports the worker pool that now lives in `kullback.runner.parallel`. |
| `kullback/builder/policy.py` | Turns policy sentences into before-write Constraint predicates (D43, D76). |
| `kullback/builder/prefixes.py` | Selects a verified trace prefix at ingest, or says why the trace is withheld. |
| `kullback/builder/prompt.py` | The Builder's prompt sections in GEPA order. |
| `kullback/builder/readers.py` | Readers for prose results: proposed as code and gated on the recording (D164). |
| `kullback/builder/registry.py` | Task registries: the container and verifier behind a recorded task id (D263). |
| `kullback/builder/replay_evidence.py` | Which recorded calls a replay failed on, and the sentences they become. |
| `kullback/builder/sandbox.py` | Runs a generated body only in a subprocess, and the gates over it. |
| `kullback/builder/search.py` | Web search and fetch for domain vocabulary, with memo and test backends (D115). |
| `kullback/builder/session.py` | The Builder extension and `build`, which drives one session to its stop. |
| `kullback/builder/skills.py` | Workdir skills the model may rewrite, versioned by content hash (D130). |
| `kullback/builder/sources/__init__.py` | The intake seam: adapter registry and format detection. |
| `kullback/builder/sources/claude_code_jsonl.py` | Detector for the Claude Code JSONL shape; no mapper yet. |
| `kullback/builder/sources/otel_genai.py` | Detector for the OpenTelemetry GenAI shape; no mapper yet. |
| `kullback/builder/sources/tau2_native.py` | Adapter for tau2 native exports. |
| `kullback/builder/sources/terminus_2.py` | Adapter for terminus-2 terminal recordings, one shell tool. |
| `kullback/builder/synth.py` | Grows the Starting state past the ids the traces named (D40, D107). |
| `kullback/builder/templates.py` | Aligns a reader for homed prose results out of the tool's own results (D176, D180). |
| `kullback/builder/triage.py` | The triage skill: how the Builder works a red light to green (D150). |
| `kullback/builder/user_sim.py` | Re-exports the rule-driven Simulated user from `kullback.user.rules` (D214). |
| `kullback/builder/vocabulary.py` | Derives the facts users state and the words agents ask for them with (D115). |
| `kullback/builder/world_tools.py` | The first pass as two functions: `ingest_files` and `derive_world`. |

## kullback/hub

| Module | What it does |
|---|---|
| `kullback/hub/__init__.py` | Publishing an Environment: package, card, fetch (D221). |
| `kullback/hub/card.py` | Renders the Environment card and the organisation card from a manifest. |
| `kullback/hub/client.py` | The five-call dataset host surface and its Hugging Face adapter. |
| `kullback/hub/package.py` | Exports a workdir as a self-contained package with a manifest and leak scan. |
| `kullback/hub/publish.py` | Uploads the package as one tagged commit, and fetches one back verified. |

## kullback/report

| Module | What it does |
|---|---|
| `kullback/report/__init__.py` | The Markdown report: built or not, then the numbers per Task, then a suggestion (D85). |
| `kullback/report/data.py` | The shapes the report reads. |
| `kullback/report/load.py` | Loads every record of one workdir off disk, deciding nothing. |
| `kullback/report/numbers.py` | The numbers beside each Task, counted off stored Verdicts. |
| `kullback/report/pipeline.py` | One row per recorded stage with the gate that failed. |
| `kullback/report/render.py` | Renders the report one section at a time and writes it. |

## kullback/tui

| Module | What it does |
|---|---|
| `kullback/tui/__init__.py` | The terminal screen: a live board, transcript and commands for one build. |
| `kullback/tui/diagrams.py` | Pure text rendering of the pipeline, the loop and the layering. |

## scripts/measure

| Module | What it does |
|---|---|
| `scripts/measure/trust_split.py` | Every build's trusted Tasks split by whether the Reference is right, read afterwards off the round file, the status rows, the references record and the benchmark sidecars (which the build never reads). |
