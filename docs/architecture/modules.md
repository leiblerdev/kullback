# Modules

One line per `.py` file under `kullback/`, read off its docstring, grouped by package. What each package
owns and how they connect is on [the architecture page](../architecture.md).

### kullback (top level)

| Module | What it does |
|---|---|
| `__init__.py` | The Leibler Harness: the Builder makes an Environment from traces, the Runner re-executes Runs and computes Verdicts. |
| `claims.py` | What a Run's transcript claims it did, against what its state actually received (D223). |
| `cli.py` | The Typer app behind the `kullback` command; each subcommand imports its module only when it runs. |
| `consistency.py` | Shrinks a failing call sequence to its shortest failing subsequence, for the laws checker. |
| `curriculum.py` | Shared curriculum records and counting rules for the training agent. |
| `derive.py` | Derive one Task's Verifier from its Reference and k re-runs read off disk (D42, D43, D91). |
| `difficulty.py` | Every Task's objective difficulty, and the buckets a round reports trusted coverage over (D209). |
| `domain.py` | What the domain's own public material says people ask for, read as task archetypes (D225). |
| `graph.py` | The tool-call dependency graph the recordings show, and the walks taken over it (D224). |
| `laws.py` | Environment consistency: laws that hold for any tool, checked off the recorded path. |
| `processes.py` | A process pool for independent CPU-bound work, the trust ruling's Tasks and a Harbor export's (D401). |
| `round_snapshot.py` | One round, one Task table, written once and never rewritten (D218). |
| `sampling.py` | One keyed draw for every sample the harness takes per Task or per Run (D212). |
| `store_specs.py` | Every single-file workdir artifact the package writes today, declared once (D267, registry step only). |
| `store.py` | `WorkdirStore`: the durable, locked read and write of a declared workdir artifact. |
| `synthesise.py` | Synthetic Tasks: a walk over the mined graph, run in the rebuilt world, derived and reported apart (D224). |
| `variants.py` | Second paths synthesised from a Run's own calls, code only (D199). |

### kullback/agent

| Module | What it does |
|---|---|
| `agent/__init__.py` | The shared agent core, shaped after tau_agent; it knows nothing about the application. |
| `agent/base_tools.py` | The base tools every extension may have: read, write, edit, grep, find, ls, inspect, web_search, bash. |
| `agent/bus.py` | One bus per workdir: the append-only, sequence-numbered log every harness and every tool writes to. |
| `agent/compaction.py` | Compaction, tau's way, at our line: the older prefix becomes one summary, recent turns stand. |
| `agent/context.py` | Context accounting: what the active context costs, where the line is, and the record it is kept on. |
| `agent/events.py` | The typed events one run of the loop emits. |
| `agent/extensions.py` | Extensions: how an application sits on the core without the core knowing it. |
| `agent/harness.py` | The harness: the one stateful object, owning the transcript, the tools, the queues and the subscribers. |
| `agent/loop.py` | The loop: one assistant message, its tool calls in order, the queues, until the model stops. |
| `agent/messages.py` | The transcript's message types, defined in `kullback.ai.messages` and owned here by re-export. |
| `agent/prefix_check.py` | Whether one conversation's requests grow by appending, and nothing else (G24). |
| `agent/provider.py` | The provider contract the loop streams through: tau_agent/provider.py's `ModelProvider`. |
| `agent/session/__init__.py` | The session: an append-only JSONL tree of everything a harness did, and the active path through it. |
| `agent/session/entries.py` | The entry types of the session tree, one pydantic model each, discriminated by `type`. |
| `agent/session/store.py` | The JSONL store: load, append, branch, and the active path with compactions applied. |
| `agent/skills.py` | Skills: texts that teach one way of working, put in front of the model as prompt sections. |
| `agent/tool_history.py` | Provider-safe repair of a transcript whose tool calls and results do not line up. |
| `agent/tools.py` | Tools the loop can call: pydantic arguments in, pydantic result out, validated on both sides. |
| `agent/types.py` | The JSON-like aliases tool arguments, tool results and event payloads are written in. |
| `agent/web_tools.py` | Web search and page reading as agent tools, the same two for every role (D377). |

### kullback/ai

| Module | What it does |
|---|---|
| `ai/__init__.py` | The provider layer: one contract over every model, the streams it answers in, pricing, usage. |
| `ai/_provider_events.py` | What one adapter reports while a provider answers, before anything is assembled. |
| `ai/_sse.py` | The streaming POST every adapter shares: retry, cancel, and one SSE line at a time. |
| `ai/anthropic.py` | Anthropic's Messages API, streamed. |
| `ai/cache.py` | Prompt caching, written once for every adapter: the request fingerprint, the TTL, the break points. |
| `ai/events.py` | The canonical events one assistant message is streamed as, whoever produced it. |
| `ai/http_errors.py` | The errors a provider answers with, and the safe detail a message may carry. |
| `ai/http.py` | The HTTP plumbing every adapter shares: the read budget, the client, and what one post carries. |
| `ai/limits.py` | One rate limiter per provider key, shared by every caller on that key. |
| `ai/messages.py` | The typed messages of a transcript: what the model said, what it was told, what a tool returned. |
| `ai/model_catalog.py` | What models an account can reach, and what each of them takes. |
| `ai/model_limits.py` | How much context one model takes, which is what decides when the core compacts. |
| `ai/openai_compatible.py` | The OpenAI-shaped endpoints, streamed: chat completions and the Responses API. |
| `ai/pricing.py` | The live price source: models.dev's catalog, snapshotted to disk (D116). |
| `ai/provider.py` | The provider contract every model answers through, and the model handle the Harness holds. |
| `ai/replay.py` | The replay seam: a stored reply answered through the same contract a live endpoint answers. |
| `ai/retry.py` | When a failed provider call is tried again, and how long the caller waits first. |
| `ai/sigv4.py` | AWS Signature Version 4 for one HTTP request, from the standard library alone. |
| `ai/stream.py` | One assistant message as canonical events, from a provider stream or from a whole reply. |
| `ai/tool_call_ids.py` | Tool call ids a provider will accept, without letting two distinct ids become one. |
| `ai/usage.py` | The token count one model call returns, the one record the provider layer owns. |

### kullback/builder

| Module | What it does |
|---|---|
| `builder/__init__.py` | Builder modules: traces in, Environment, Tasks, References and re-rolls out. |
| `builder/cache_reach.py` | The import closure of a builder module as one hash (G29). |
| `builder/cluster.py` | Groups Runs into Tasks by the user's request alone, then labels each Run with its confirmed write set (D83, D313). |
| `builder/compile_env.py` | Builds the world: `db.json` by inverse replay, per-Task overlays, each tool body through its gates. |
| `builder/domain_tools.py` | The Builder's domain tools: ingest, derive_world, grow, replay, rulings, run, status, examine, note_task. |
| `builder/effects.py` | What a write changed on rows its own arguments never named, read off the recording (D215). |
| `builder/env_files.py` | The Builder's Environment on disk: one file per tool body, rendered modules, the read surface. |
| `builder/ingest.py` | Stores customer files byte for byte and derives Traces with raw pointers (D66, D67, D95). |
| `builder/lesson.py` | What a stalled tool body is told: differing leaves, relations, unreached lines (D211). |
| `builder/mine.py` | Mines the customer's tools (ToolSig) and world (EntitySchema) out of ingested traces (D68, D70, D72, D73). |
| `builder/parallel.py` | The Builder's name for the worker pool, which moved to the Runner when the Examiner needed it too. |
| `builder/policy.py` | Turns policy sentences into before-write Constraint predicates (D43, D76). |
| `builder/prefixes.py` | Selects a verified trace prefix at ingest, or says why the trace is withheld. |
| `builder/prompt.py` | The autonomous Builder's prompt sections, in GEPA order, stop rule last. |
| `builder/readers.py` | Readers: the columns a requestor's prose results describe, proposed as code and gated on the recording. |
| `builder/registry.py` | A task registry: the starting container and verifier behind a recorded task id. |
| `builder/replay_evidence.py` | Replay evidence: which recorded calls a replay failed on, and the sentences they become. |
| `builder/run_user.py` | The agent user a fresh Run meets, over its rule floor, built from the workdir (D214). |
| `builder/sandbox.py` | Runs a generated tool body only in a subprocess, and the five gates that check it before it is trusted. |
| `builder/search.py` | Web search and page fetch for the Builder (D115). |
| `builder/session.py` | The autonomous Builder as an extension on the core. |
| `builder/skills.py` | Phase 6 skills: the stage prompts the model may rewrite, versioned by content hash (D130). |
| `builder/sources/__init__.py` | One intake seam: every trace format enters through a registered adapter. |
| `builder/sources/claude_code_jsonl.py` | The Claude Code JSONL adapter: one session transcript mapped to one Trace. |
| `builder/sources/otel_genai.py` | The OpenTelemetry GenAI adapter: model-call and tool-execution spans mapped to a Trace. |
| `builder/sources/tau2_native.py` | The first adapter behind the intake seam: the export this harness reads today. |
| `builder/sources/terminus_2.py` | The second adapter behind the intake seam: terminal agent recordings in terminus-2 shape. |
| `builder/synth.py` | Grow the Starting state past the ids the traces named (D40, D107). |
| `builder/templates.py` | The reader a homed prose result never got, aligned out of the tool's own recorded results. |
| `builder/tool_effects.py` | The build's tool effects and call signatures, learned once and stored for the Spec stage (spec/effects.py). |
| `builder/user_sim.py` | The rule-driven Simulated user, which now lives in `kullback.user.rules` (D214). |
| `builder/vocabulary.py` | The facts a customer's users state and the words agents ask for them with (D115). |
| `builder/world_tools.py` | The Builder's first pass as two tools: ingest and derive_world (stream 5a). |

### kullback/examiner

| Module | What it does |
|---|---|
| `examiner/__init__.py` | The Examiner: reads and rules, one review per Task; it writes no Verifier (D320). |
| `examiner/audit.py` | A Reference audit pass over the trusted Tasks the main session never checked. |
| `examiner/domain_tools.py` | The Examiner's domain tools over its root: the ruling tools, finding, no_finding, check_reference. |
| `examiner/exam_files.py` | The Examiner's root under the workdir: what the model session may read (D123). |
| `examiner/findings.py` | The findings the round's records file by themselves, before a model chooses what to say (D170). |
| `examiner/goal.py` | The Examiner's own expected end state for one Task (the goal), from other evidence than the writer. |
| `examiner/judge.py` | The reference judge as an agent with a bounded look, and the one-shot judge as its fallback (D12, D92). |
| `examiner/lifecycle.py` | When a derived artefact stops being live: its source was withdrawn (D208). |
| `examiner/must_not.py` | Re-exports `kullback.spec.must_not`. |
| `examiner/plan.py` | The Examiner's plan: what one session over one workdir is given, and the store its gates rule over. |
| `examiner/prompt.py` | The Examiner's prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D320, D331). |
| `examiner/reference_check.py` | Is the Task's Reference right? A view the Examiner judges, and the move that acts when it is not. |
| `examiner/reference.py` | References by the D111 rule, and the constraints checked against them before they gate anything. |
| `examiner/rule_tool.py` | The Examiner's ruling tools: rule, close, verify and the world lookups (D320, D331). |
| `examiner/session.py` | The Examiner as an extension on the core, called by the Builder (D79, D111, D122-D124, D230, D320). |
| `examiner/skills.py` | The review skill: what one review of a Spec reads and rules (D320, D331). |
| `examiner/stage.py` | The Reference stage: chooses References (D111) and writes `references.json`, `task_status.json`, `constraints_check.json`. |

### kullback/gates

| Module | What it does |
|---|---|
| `gates/__init__.py` | Every accept-or-reject check over what the Builder or the Examiner made, and no model call (D122). |
| `gates/artifacts.py` | The gates over each build artifact, each a function returning a GateResult. |
| `gates/bindings.py` | Which gates run when an agent writes a file, and the rows each refusal names. |
| `gates/confinement.py` | Refuses a model-written tool body or predicate before it runs anywhere in process. |
| `gates/counts.py` | The workdir counts as code: every count comes from a gate ruling. |
| `gates/fidelity.py` | The replay fidelity bar, at the three grains it is held to (D39, D51, D80, D108). |
| `gates/hook.py` | The `tool_result` hook that runs the path-bound gates over an agent's writes. |
| `gates/ledger.py` | gates.json under one lock: the one ledger both agents record their rulings through (D122, D128). |
| `gates/loosening.py` | One-directional loosening over an expandable legitimate pool (D127, D133), and the false-rejection number. |
| `gates/probes.py` | The monotone probe pool (D127) and the stop rule (D133), as code over a Task's pool and its Verifier. |
| `gates/scorecard.py` | The D62 scorecard: tool fidelity, Task coverage, user fact consistency, Verdict agreement. |
| `gates/stages.py` | The rulings over each Builder stage's artifact. |
| `gates/tool_runs.py` | The eight rulings over what a generated tool body did when the sandbox ran it. |
| `gates/trust.py` | The refuse ruling (the solvability judge as code, D128) and what a trusted Verifier is (D126, D319). |
| `gates/verifier_suite.py` | The nine D79 checks over a Verifier, and the scorer they share with the Examiner (D79, D119, D122). |

### kullback/hub

| Module | What it does |
|---|---|
| `hub/__init__.py` | Publishing an Environment: the package a build leaves, its card, and the fetch that brings it back (D221). |
| `hub/card.py` | The cards: one per Environment, one for the organisation, and the template that documents both (D221). |
| `hub/client.py` | The whole surface a dataset host is spoken to through, and the one adapter that speaks it (D221). |
| `hub/harbor_world.py` | One Kullback Task as it runs inside a Harbor container: the tools, the call log, the Verifier (D363). |
| `hub/harbor.py` | An Environment as a Harbor dataset: one Harbor task directory per Kullback Task (D363). |
| `hub/listed_ids.py` | The listed ids a shipped world drops: members of an id list that name no row of the table they key (D429). |
| `hub/package.py` | The Environment package: the rebuilt world and the graders, with a manifest and a leak scan (D221). |
| `hub/publish.py` | Publishing an Environment and fetching one back (D221). |
| `hub/repair.py` | Three repairs the Harbor export makes to a written Verifier, by code, before it is laid (D431, D432, D437). |

### kullback/judge

| Module | What it does |
|---|---|
| `judge/__init__.py` | Judge items: one model call per Verifier item (D328). |
| `judge/items.py` | Judge items (D328): one model call per item, score 0 or 1, on evidence cut from the Run's event log. |
| `judge/shape.py` | The shape of a judge item (D328), with no model in reach: the Verdict reads it, so no provider here. |

### kullback/report

| Module | What it does |
|---|---|
| `report/__init__.py` | The Markdown report: built or not, then the numbers per Task, then a suggestion (D85). |
| `report/data.py` | The shapes the report reads: one object holding every record a workdir yields. |
| `report/load.py` | One workdir as the report reads it: every record off disk, nothing decided. |
| `report/numbers.py` | The numbers beside each Task, counted off the stored Verdicts and never decided. |
| `report/pipeline.py` | The pipeline as the run recorded it: one row per stage, with the gate that failed. |
| `report/render.py` | The report as Markdown, one section at a time, in the order the design fixes. |

### kullback/runner

| Module | What it does |
|---|---|
| `runner/__init__.py` | Package marker for the Runner. |
| `runner/arith.py` | One code-owned arithmetic evaluator, handed to every model-written tool body by name. |
| `runner/atom_context.py` | What an atom predicate may look at, and the confinement gate that certifies it. |
| `runner/boundary.py` | The D89 and D91 import boundary as a static AST scan. |
| `runner/budget.py` | Cost and token accounting: per call, per stage, per build, plus the D65 context cap and the D86 spend ceiling. |
| `runner/cache_view.py` | What the provider's prompt cache did for one build, per stage, read off the feed (`kullback budget cache`). |
| `runner/canon.py` | The canonicalizer (D39), equality by column class (D73) and the equivalence table (D84). |
| `runner/code_hash.py` | One content hash of the code that executes and scores a Run, read off the files' bytes at import. |
| `runner/confinement.py` | The AST check that a model-written predicate cannot reach outside its case. |
| `runner/expected.py` | A Task's expected end states and forbidden list from a Reference Run, by code (D315, D316). |
| `runner/feed.py` | The live feed of one build: what it is doing right now, in the order it did it. |
| `runner/gate_support.py` | The `GateResult` helper, field access and canonical equality every gate uses. |
| `runner/heartbeat.py` | Which builds are running, for the screen's session list. |
| `runner/judge.py` | The agentic judge (D92): read-only state views, pre-run checks, two judges, a disagreement queue. |
| `runner/loop.py` | One Run, one turn at a time: query the model, route its tool calls, write one JSONL line per event (D90). |
| `runner/parallel.py` | A bounded worker pool for independent work, the Builder's model calls and the Examiner's Tasks (D118, D163). |
| `runner/real_tools.py` | A tool run for real in a sandbox, beside the routes that imitate one (D262). |
| `runner/records.py` | Every record the Harness passes between modules, as Pydantic v2 models that round-trip through JSON. |
| `runner/regrade.py` | Re-score stored Runs against a new Verifier or Environment version without re-executing them (D97, tau3 --fresh-tasks). |
| `runner/replay.py` | Replay a recorded Trace through the rebuilt Environment: the Reference Run, scored call by call. |
| `runner/route.py` | Answers one tool call: real first, then code, the recording table, an LLM stand-in (D262, D45, D49, D74). |
| `runner/state.py` | The Task's Starting state, its overlay over the shared world, and the write path into a generated toolkit's own db (D74, D46). |
| `runner/target.py` | One scorer for a Verifier atom: the structured target as the only meaning (G3). |
| `runner/tool.py` | The runner as a tool: run, replay and reroll over a built Environment. |
| `runner/transaction.py` | One tool call as a transaction on a toolkit's db: snapshot before the body runs, restore if it raises. |
| `runner/validate.py` | What certifies the Runner itself: its import boundary and its version. |
| `runner/verdict.py` | Decide one Run's pass or fail from its End state against a Task's Verifier, in code only (D43, D46, D94). |
| `runner/world/__init__.py` | The world a Run executes in: loading from rows and overlays, the laws, the starting states. |
| `runner/world/birth.py` | Checks at birth for a Task, by code alone: a scripted walk passes, do-nothing fails. |
| `runner/world/clock.py` | The world clock: the one time a tool body can see, read off the recording (D283). |
| `runner/world/environment.py` | A built Environment on disk, workdir or fetched package, read without the Builder (G1). |
| `runner/world/episode.py` | Anyone's handle on a Run in a built Environment: reset, step, a reward by code (G1). |
| `runner/world/law_world.py` | One Task's built Environment as a world the D258 laws are checked against. |
| `runner/world/loading.py` | What a Run needs from the Builder, owned below it (G1). |
| `runner/world/recorded.py` | What one recorded call witnessed for the tool context, read off the recording alone. |
| `runner/world/solve.py` | The solve-rate stage: a cheap model over the Tasks, through the interface (G1). |

### kullback/spec

| Module | What it does |
|---|---|
| `spec/__init__.py` | The Spec: one file per Task holding the Intent and the Verifier written from it. |
| `spec/actions.py` | Tools the world records as actions, not writes (D308). |
| `spec/atoms.py` | The two generated Hard rules a Verifier is written in, whoever writes it (moved from kullback/derive.py). |
| `spec/canfail.py` | How the Spec's Verifier scores one Run: the Verdict for a gated Verifier, else the atoms alone. |
| `spec/coherence.py` | Did the recorded user hold to their own opening asks? Answered by the miner, checked by code. |
| `spec/compile.py` | A Spec's checks to a Verifier with the existing atom builders, no new kinds. |
| `spec/effects.py` | What each tool always does, learned once per Environment build from the corpus's recordings. |
| `spec/events.py` | The Spec's and the rulings' events on the workdir bus. |
| `spec/ground.py` | Trust gate 1: every check rests on the Intent or the policy, every Intent fact has a check or a gap. |
| `spec/intent_tools.py` | The intent miner's view of a Task: the user turns of every recording, and nothing else. |
| `spec/intent.py` | The intent miner: a Task's SpecIntent from the user's words only, provenance on every fact. |
| `spec/items.py` | The Verifier's items, written by the Spec writer from the Intent, the policy and the world. |
| `spec/judged.py` | The judge items' answers that verify and trust read (D346, D347): the Verdict never calls a model. |
| `spec/must_not.py` | The must-not side of an Intent-written Verifier: what the agent must not do. |
| `spec/mutant_rulings.py` | Surviving mutants become the Examiner's rulings: which field or row is unchecked, never a value (D362). |
| `spec/mutants.py` | Code mutants of the Task's base end state, keyed to the ask: each is that state with one change, each should fail. |
| `spec/report.py` | The per-build tier report: trusted, untrusted by reason, the two flags, trusted by Reference class. |
| `spec/review.py` | The writer's round of the review loop: each open ruling applied or rebutted (D331). |
| `spec/rounds.py` | The review loop: Examiner round, writer round, Examiner round, at most the round cap (D331, D359). |
| `spec/router.py` | The set-aside rule of the review loop (D331). |
| `spec/rulings.py` | The Examiner's rulings on a Spec: feedback the writer applies or rebuts, never an edit (D331). |
| `spec/schema.py` | The Spec records and their one file on disk: workdir/spec/<task_id>.json. |
| `spec/split.py` | The Reference class of a Task from the benchmark's own rewards: measurement only, never read by trust. |
| `spec/stage.py` | The Spec stage of a build: each named Task's Intent mined, its Spec and Verifier written, the bus told. |
| `spec/text.py` | The word-level text rules the Builder and the Spec both read (moved from builder/cluster.py and intent.py). |
| `spec/trust.py` | Trust on the Spec's Verifier: constructed Runs fail and no ruling is open, by code (D333). |
| `spec/ungated.py` | Every write a born Task's scenario end names is gated by a differing cell, on every named column (D419). |
| `spec/witness.py` | The Task's Reference as a witness Run, for the reference_passes flag only; the writer never reads it. |
| `spec/writer_tools.py` | The writer's only tools: world search, the policy, and the one tool that adds items; the bounded session. |
| `spec/writer.py` | The Verifier writer: items from the Intent, the policy and the world, never from a Reference. |

### kullback/synth

| Module | What it does |
|---|---|
| `synth/__init__.py` | The scenario layer of the synthetic generator (D372 to D374). |
| `synth/generate_prompt.py` | The scenario generator prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D372). |
| `synth/generate_tools.py` | The scenario generator session: a role on the shared agent harness and event stream (D372). |
| `synth/generate.py` | The scenario generator: scenarios from the built Environment, its recordings, a web search and typed lines (D372). |
| `synth/plan.py` | The plan of situations the recordings' openings show, written before any scenario (D380). |
| `synth/prompt.py` | The Examiner's scenario review prompt in GEPA order: receives, tools, examples, choice, feedback, stop (D373). |
| `synth/records.py` | The scenario record and its store (D372). |
| `synth/report.py` | The synth report: every scenario count under its own heading, never in the build table (D372, D373). |
| `synth/review.py` | The Examiner's scenario review: a role on the shared agent harness and event stream (D373). |
| `synth/rule.py` | The world a synthetic generator session reads, and the statement rule its tools hold to (D372). |
| `synth/search.py` | The scenario layer's seam onto the harness's one web reader (D374, D115) and the shared web tools (D377). |
| `synth/skills.py` | The scenario review skill: the order one review of a scenario reads in (D373, D382). |
| `synth/slice.py` | The slice of the world one scenario is written onto, picked by code before any session (D420). |
| `synth/tasks.py` | Tasks born of passed scenarios, through the mechanism every recorded Task already goes through (D389). |
| `synth/world_tools.py` | The world tools every scenario session holds: rows by key, rows by a field's value, tools by words (D381). |

### kullback/trainer

| Module | What it does |
|---|---|
| `trainer/__init__.py` | Rollouts the training agent learns from: one Task played k times under the rule user. |
| `trainer/act.py` | Local backends for acting on training: asks, smoke runs and env checks. |
| `trainer/observe.py` | Local backends for observing training runs, read off files on disk. |
| `trainer/prompt.py` | The training agent's prompt in fixed order: role, tools, loop, numbers, stop. |
| `trainer/remote.py` | Remote backends for the training agent tools: shell, GPU status, run control, sync. |
| `trainer/rollout.py` | One Task played through the Episode interface under the rule user, grouped in k. |
| `trainer/session.py` | The training agent as an extension on the core, run over a workdir. |
| `trainer/tools.py` | The training agent's tools: thin calls over the trainer backends. |

### kullback/tui

| Module | What it does |
|---|---|
| `tui/__init__.py` | kullback's terminal view: one screen that shows a build while it is happening. |
| `tui/diagrams.py` | Diagrammatic views of the kullback build: the pipeline as a flow, the loop as beats, the layering as a stack. |

### kullback/user

| Module | What it does |
|---|---|
| `user/__init__.py` | The Simulated user as an agent of the harness, beside the Builder and the Examiner (D214). |
| `user/account.py` | What the agent user may look up about itself beyond its facts: its choices and its own account. |
| `user/agent.py` | `AgentUser`: one Simulated user driven by a model, one turn per call, with the rules as its floor. |
| `user/context.py` | The context curated for the agent user, per Task, all of it mined (D214). |
| `user/ends.py` | Run end and transcript readers: how a Run ended, and whether a write call moved the world. |
| `user/extension.py` | The agent user as an extension on the shared agent core (D214, ADR-0007). |
| `user/factory.py` | The one place a Simulated user is built, whoever asks for it (G5). |
| `user/fidelity.py` | User fidelity: how close a driver's turns are to the ones the recording holds (D214 rule 5). |
| `user/guards.py` | The guards that run in code after the model, on every turn, never waived (D214 rule 4). |
| `user/lesson.py` | What the fidelity score teaches the agent user's context next round, and when to stop paying (D214 rule 6). |
| `user/lookup.py` | The user's fact store and the one lookup both Simulated users answer from (D332). |
| `user/rules.py` | Derives the rule-driven user's rules from one trace (D44, D77). |
| `user/simulated.py` | The rule-driven Simulated user: it replies as the recorded user, from the rules and the world. |
| `user/skills.py` | The agent user's system prompt, in the order every agent of this harness states one. |
| `user/tools.py` | The agent user's own tools: what a person on the phone can reach, and nothing else (D214 rule 2). |
| `user/value_strip.py` | Takes system-known values out of an Intent or a user answer (D196, D210). |
| `user/vocabulary.py` | What a fact is, in the shape the Simulated user reads it (D115, D214). |
