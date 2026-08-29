# Monitoring tool: deferred work

These are the items I pushed past the first build. Each line says what it buys and what gates it. The date is when I deferred it.

## After the first Replica clears Gate A

- **Output perturbation** (2026-08-26). The Environment can serve slightly modified tool outputs (formatting, order, harmless value drift) for the same call, so a Candidate is tested against variation, not one frozen recording. Buys robustness signal; costs a second fidelity mode (Gate A must still hold on the unperturbed path). Never on for routing-plan Verdicts until validated.
- **Process reward for the path** (2026-08-26). Verdicts stay End state only. A process reward model over Steps (which Steps were necessary, which were wasted, where the Candidate went wrong) is the diagnostic layer for failed Runs and the reward layer for later post-training. Needs: Step labels from the frontier re-rolls, agreement study against human annotators (TRAIL 11% localization and kappa 0.55 are the numbers to beat). Not a Verdict input.
- **Human audit rate validation** (2026-08-26). The 10% audit of Verifiers and judge rulings has no published comparator. Measure on the first customer: agreement rate, hours per 100 Verifiers, which atom types disagree most. Publish; then tune the rate per Task from observed disagreement rather than a fixed 10%.

## After the Environment generator is stable

- **Post-train on generated Environments and watch for hill-climbing** (2026-08-26). Generate Scenarios from a customer's Environment, post-train a Student on them, evaluate on the held-out Replicas and on public benchmarks with the same domain (tau2 retail, airline, telecom; AgentDojo workspace). The question is whether gains on our Environments transfer. Gated on: Scenario generation (ADR-0005), replay fidelity on two customers, a 20% Replica holdout that was never used for Scenario seeds.
- **Simulated user from production traces** (2026-08-26). Replace hand-written personas with a simulator conditioned on the customer's real user turns (phrasing, information they volunteer versus withhold, when they get frustrated). Fidelity target: recorded-turn replay agreement and the simulator error rate on held-out Runs, both published per customer. Split per D44 (2026-08-27): fact consistency on the recorded user's facts (build with the Environment, before any Verdict, since D43 atoms depend on it) and style representativeness (paired Turing test, needs enough multi-turn Runs per Task to hold out). No longer gated on Gate A.

## Verifier and Environment width

- **Synthetic row generation for reads** (2026-08-26, D40). Learn the shape and distribution of each table from the rows the traces show (fields, types, value ranges, foreign-key relationships, entity counts) and generate tagged synthetic rows for entities the traces never touched. Representativeness checks are a Hard constraint (D41): a generated row must be indistinguishable in shape from observed rows and must not contradict any observed fact. Buys: off-path reads return plausible data instead of "not found". Gated on: Gate A on writes, and the assisted-share measurement that shows which tables need it.

- **Snapshot rung of the trust ladder** (2026-08-26). Generator input for a customer DB snapshot or staging copy, running inside the customer's boundary (ADR-0006). Buys a complete Environment for evaluation and Scenario training. Gated on: a customer who has climbed the earlier rungs.
- **Schema and tool-definition ingestion as a first-class input** (2026-08-26). Traces-only Environments have holes (see `eval-design.md`, "Holes in a traces-only Environment"). Tool definitions are usually already in the traces (the `tools` parameter of each LLM call, or MCP `tools/list`); parse them on ingest. DDL or OpenAPI is the first thing to ask a customer for after traces.

- **Assisted-vs-real Verdict agreement** (2026-08-27, D49). On the first customer that supplies a snapshot (ADR-0006 rung 6), re-run the previously assisted Runs against the real Environment and measure how often the assisted Verdict matched. If agreement is high, revisit whether assisted Verdicts can be a counted tier.

- **Harden Tasks (TauForge stage 4)** (2026-08-27, D54). After Environment, seeds and generated Tasks are stable: hardening taps (traps, perturbations, adversarial personas) grounded in the customer's traces. Not before Gate A.
- **Builder assets across customers** (2026-08-27, D54). Accumulate logs, skills, past builds, utility scripts, a knowledge graph of tool and schema patterns, personas; each must stay grounded in observed traces (D41).

- **Ingestion for the customer's real traces** (2026-08-27, D56). The founder will upload real traces soon; ingestion must accept their format as-is (R16 minimal fields, R23 formats), normalize tool errors (D45), group Runs, and report what the traces do and do not contain (tool definitions, errors per tool, untruncated results) before any build.

- **Regrade stored Runs after an Environment fix** (2026-08-27, R22 section 1.6). tau3 1.0.1 added `evaluate-trajs --fresh-tasks`; re-scoring moved +0.47 to +9.02 points with no downward movement. Every Run we execute is stored so a Verifier or Environment fix re-scores it without re-running the model.
- **Hacker pass over the Verifier pool** (2026-08-27, R22 item 11, 2606.08960). Now D79 check 6 inside the build, one model Run per Task under `--probe-limit` (D108); the pass over a whole pool with a stronger attacker is still this item. An agent told to reach a pass without doing the work, against each Task's Verifier; a hit means the Verifier is too narrow or leaks. Cheap on compiled Hard constraints and DB-diff atoms; fits step 8 validation, before hardening (D54 stage 4).
- **Per-Task policy coverage number** (2026-08-27, R22 item 10, 2608.06329). Count which policy items the customer's traces exercise ("your traces cover 6 of 40 policy items") so the report says what was not tested.

- **Self-generated trace setup** (2026-08-27, D60). Chatwoot or Zammad in Docker, an existing MCP server (Zammad 52 tools), two external agent harnesses and models, tau2's user simulator, injected real HTTP errors. Output in tau2 native, OTel and MCP formats. Labeled self-generated everywhere; never the only tuning input for the Builder.

## Components named and deferred (2026-08-27, D69)

I chose which of the missing components go into the first build (clusters, intent writer, canonicalizer, report, cost and budget accounting, provider adapter, Builder memory and self-modification). The rest:

- **Filter and Screen** (eval-design step 2, Match then Appeal). Which Runs are worth re-executing. The first build runs everything; needed at the first customer with more Runs than budget.
- **Reference confirmation as a module** (step 5, D57: Hard constraints on the Reference, outcome signals, k re-rolls, judge that abstains to a human). The replay half exists since D108 (`runner/replay.py`, Gate A per Trace by code); outcome signals and the abstaining judge are still this item.
- **Dispute path** (step 11). The tool-equipped judge for End states outside required and allowed atoms. Until it exists such Runs are reported as "not gradeable" (D49 wording).
- **Statistics module** (step 12). Paired non-inferiority CI, pass^k, the Task floor. Not needed for one Task and one Candidate; needed the moment the report says "cheaper model is good enough".
- **Sandbox for model-written tool code**. `compile_env.py` executes code an LLM wrote. Subprocess with no network and resource limits at minimum; container (OpenEnv's Docker shape) when Environments run for customers. Must exist before any customer trace is compiled.
- **Anonymization and boundary control**. Clio-style privacy pass on traces and Intents; what leaves the customer's boundary (ADR-0002, rung 6). Trace intake question 13 asks it; no component answers it.
- **Secrets handling**. Keys for the self-generated helpdesk setup (D60) and customer staging endpoints. Environment variables and a `.env` outside git until a customer needs more.
- **Cache store** beyond content-addressed files on disk (harness-design section 8).
- **Review UI** for the setup review and the blind audit (D48). Founder: "maybe later". Files on disk plus the report for the first customer; a UI when reviewers are not us.
- **Scenario generation and seed augmentation** (D54 stages 2 and 3). Already gated in "Post-train on generated Environments" above; listed here so the module gap is explicit.
- **Dataset export for the Student**. The tree of Runs with Verdicts as a post-training set (harness glossary, second half).
- **Deploy and route**. The routing half of the glossary Harness.
- **Production monitoring**. Comparing the routed model's live traffic against the Environment's predictions; the "monitoring" in the product name; nothing designed yet. Opened up on 2026-08-29, see "Execution and production monitoring" at the end of this file.

## Mining (2026-08-27)

- ~~**Parallel model calls in the Builder**~~ Done as D118 the same evening (`shared/parallel.py`, `--workers`, default 8).
- **DAG execution of the stages, for real** (founder, 2026-08-29: "also a dag graph based execution? would be good right?", then, on the recommendation to measure first: "it should be dag not on paper but quite good"; decided, not yet built). The pipeline declares the graph (Stage inputs, outputs and input_paths, topological order, mermaid in state.json) and runs it in a line. To build: a scheduler that starts a stage the moment every artifact it reads is complete, so compile_tools, compile_policy, judge_lessons, vocabulary and intent run side by side (none reads another's output), rerolls starts when bodies, user_rules and environment are in, derive_verifier when everything is. What has to hold: gates.json, budget.json and state.json are written from several stages at once (one lock each, as budget already has); the on_event stream carries the stage name on every event so a live view can interleave them; a failed gate's rollback edge (design section 8) still rolls back only the stages downstream of it and lets independent ones finish; the content-addressed cache key of a stage does not change (the graph, not the schedule, is what a hash names); the ceiling stops every running stage, not just the one that crossed it. The engineer's note, for the record: with eight workers inside each stage the saving is bounded by the longest of the five independent stages, a few minutes on a fifteen-minute build, so the value is the loop (a stage re-runs as soon as its edited input lands) more than this build's clock.
- **Workers: how many, and how they are launched** (founder, 2026-08-29: "add in todo ... about the workers and how it is launching the workers"). D118 launches one `ThreadPoolExecutor` per stage call, sized `min(workers, items)`, threads in the build's own process, and the pool dies with the stage; `--workers` is a flat count with no view of the provider's rate limit, the sandbox subprocesses each thread spawns (a tool body probe is a subprocess with a 30 s timeout, so eight threads can hold eight of them), or the DAG above, under which two stages' pools would add up. To decide: one shared budget of threads for the whole build or a pool per stage; whether re-rolls (long Runs, a few calls a minute each) and tool bodies (short calls with a subprocess each) want different counts; a worker count read from the provider's 429s (back off the pool, not just the call); whether a worker is ever a process rather than a thread (the sandboxes already are, the model calls need not be); and what the live view shows per worker (which item, which stage, how long). Until then: 8 on the CLI, 1 in code, and budget.json's `wall_ms` is latency summed per call, not elapsed time.
- **Luna was unpriced on every live build so far** (found 2026-08-29): budget.json says `usd 0.0`, `unpriced_calls 3396`, because the hand table never had a price for `openai/gpt-5.6-luna` and the models.dev snapshot (D116) did not exist until fetched by hand that evening. Rebuilt budgets price it from the snapshot; the live-build doc's cost lines before this date are token counts, not dollars.
- **Verified synthetic data, for when the product is more than an evaluation platform** (founder, 2026-08-29: "we also need synthetic (verified data) when we move from being just an evaluation platform"). The Runs that pass a code Verifier in a built Environment are training data with a verified label: the transcript, the tool calls, the End state and the Verdict, all replayable. Export them (and the failing ones, labelled) in a shape a fine-tuning pipeline reads (OpenAI chat JSONL, tau2 format from D88's adapter, or whatever the customer's trainer takes), with the Verifier's atoms and the Environment version attached so a label can be traced to the code that awarded it. FinetuneDB-style products do the log, curate, fine-tune loop over answers; the adjacency here is Runs verified by behaviour in a world, not judged by a rubric. Not designed; needs the Runner's Runs to carry the Environment hash they ran in (D69) and a decision on whether re-rolls (D112) count as synthetic data or only as References.
- **GLM 5.3's environment-generation design, adopted as a target** (founder, 2026-08-29: "we also need to use this design for glm 5.3 for environment generation"). What was described: tasks that mirror expert work rather than puzzle problems, with the agent given the resources a person has (compute, storage, docs) and a long-horizon goal (find the bottleneck, run experiments, keep correctness); research agents convert task patterns from real workflows into runnable long-horizon environments; a judge agent checks each environment is solvable so no bad training signal enters; a verifier is generated per environment without access to the reference solution; solver trajectories close reward shortcuts and trivial solutions; every verifier passes three checks before use (the known-correct solution is rewarded, a run that did nothing is not, a run that left the task unfinished is not); high volume at high quality. Where the Harness already is: environments from real traces (the Builder), re-rolls of the frontier as solvability evidence (D112), the D79 suite with the oracle check (2) and the empty-Run check (3), the loophole probe as one solver trajectory against shortcuts (check 6). What is missing, in order: an unfinished-Run check in D79 (the Reference cut before its last write must score below the oracle; today only the code-built wrong Run and the probe play that role); a solvability judge that reads the re-rolls per Task and refuses a Task no frontier Run finishes, rather than a gate that only counts finished Runs corpus-wide; many solver trajectories per Task feeding the probe, not one; long-horizon Tasks with resources beyond tool calls (files, docs, a compute budget), which the trace model does not carry yet; volume, which needs the loop item above and the DAG. One honest disagreement to settle in the grill: their verifier is written without the reference solution, ours is derived from it (D111), and the two guard against different failures (theirs against a verifier that only recognises one path, ours against a verifier grounded in nothing); an unseen-reference derivation with the Reference used only as the oracle check is the version that keeps both.
- **The Builder as a loop that runs its own Environment and fixes what it finds** (founder, 2026-08-29, restated: "there has to be a loop which runs so that the harness can steer the creation of the environment"; grill started the same evening). D63 principle 4, D64 and design section 8 already decide this in principle (the model proposes edits to its prompts, checklists, gates and seed corpus, an evaluator outside the loop accepts them, Builder history is a tree); no code runs it. Today the pipeline is one pass with bounded repair inside single stages, and `--iterate` only resumes the cache. The founder wants a loop over the whole thing: 1. build the Environment, 2. augment it (grow the world, add the rows and error paths the traces only hint at), 3. generate the Tasks, 4. harden the Tasks (probe them, find the loophole, the unreachable Reference, the Verifier an empty Run passes, and fix the cause), then round again until the Environment is very good. The model inside the loop should be able to change its own system prompts, checklists and seed corpus (principle 8 in the design), run the Environment, read where a Run went wrong and repair the artifact, with every change gated by code as now. Open questions for the grill: what "good" is measured by per round (D79 suite pass rate, re-roll agreement, assisted count, tau2 reward agreement while the scaffold exists), what stops the loop (a round that improves nothing, a spend ceiling), what the model may and may not rewrite (never a gate, D110), and how a rewritten prompt is versioned so a regrade can name it (D69 content hashes).
- **Mine user behaviour, not only tools** (founder, 2026-08-27). `mine.py` today produces `ToolSig` and `EntitySchema` from tool calls; the user turns carry a second body of knowledge: what facts users volunteer before being asked versus only on request, how many turns they tolerate before frustration or walk-away, how they phrase the same request, how often they change their mind mid-Run, which questions they refuse. Output would be a per-customer `UserBehaviour` record feeding `user_sim.py` (D44 style representativeness), the disclosure rules in `UserRules`, and the report ("your users volunteer the order id 80% of the time"). Related: "Simulated user from production traces" above. Gated on: enough multi-turn Runs per Task to measure it (D44 held-out requirement).

- **Experiment: choose k, the frontier re-run count** (2026-08-27, D78). tau2 retail first: 10 re-runs per Task, required and allowed sets from the first k for k = 2 to 10, the k at which sets stop changing per Task, Verdict agreement with tau2 reward at each k, cost per Task. Compare fixed k with a stop-when-stable rule (start 3, stop after 2 quiet additions, ceiling 10). Output: one table, the default k in config, A30 confirmed or refuted.

- **OpenEnv wrapper** (2026-08-27, D90). `reset() / step() / state()` over `route.py` and the Task's Starting state, about a hundred lines. Gated on the tau2 slice passing its gates. The loop is already a one-turn function with a step-by-step test, so this is packaging, not redesign.

## From the first build and offline slice (2026-08-28)

This comes from the first offline slice (2026-08-28) and the build workflow's review and fix passes: 756 tests pass, and I made no model calls anywhere.

- **Cluster threshold** (D97 revision by slice). ~~Decide: raise the default to 0.6 or replace token Jaccard.~~ Done, and it was the second of the two: token Jaccard is now weighted by corpus idf, argument keys are dropped, linkage is complete, and the default is 0.4. Pair F1 is 0.717 at the default and 0.685 to 0.719 across 0.3 to 0.6, against 0.276 to 0.720 for the unweighted measure, so the result no longer turns on the threshold. Recorded as D100. The 79.8% ceiling for any tool-set clustering is unchanged (only 91 of 114 tau2 task_ids have all four trials writing through the same tools).
- **`propose_kind` cannot return `generic`**. ~~Either a third kind or an explicit rule that pure functions and hand-offs are reads.~~ Done: a name rule returns `generic` at medium confidence, classified rather than unclassified, and a generic tool is never credited with an observed effect. tau2's `calculate` and `transfer_to_human_agents` are the two it picks out. Recorded as D98.
- **Error payload prefix**. tau2's harness prepends `Error: ` to every tool error; a replay compares the raw payload against the tool's own message and would count 10 of 10 as mismatches without stripping it. Canon needs a per-source error-prefix rule.
- **End-state reference row**. Gate A must compare against the row a write returned, not the last row read; last-sighting gave one false mismatch in 95 (a stale gift-card balance).
- **`tools_declared` is empty on every tau2 trace**; the export carries no tools list, so every ToolSig is `source: observed`. The declared-vs-observed merge is untested on real data.
- **`build` and `run_batch` are written**, in `builder/build.py` rather than `runner/pipeline.py`: the stage graph has to import the Builder, and the Runner may not (D89). `harness build` and `harness run` reach them. They still need a live Model and provider config to do anything, so they are untested end to end against a real key.
- **Model-written tool code runs in-process** (`compile_env.load_toolkit`). The subprocess Sandbox exists for compile checks only. A static confinement check now refuses a body that imports outside the allowlist, names a denied builtin or touches a dunder attribute before anything is executed, and the same check guards atom predicates in `verdict.py` and constraint predicates in `validate.py`. That is a name check, not a sandbox: it is what stands in for one until the subprocess Sandbox carries the Runner's tool route too. Still gated as "Sandbox for model-written tool code" above; must land before any customer trace is compiled.
- **Size**. `src/` is 8,911 lines against the design's 2,700 to 3,700. Every module says what it carries beyond its band (overlays, repair loop, judge, memory tree, provider adapters). Decide whether the bands or the code change.
- **Could not run offline**: compile gates on LLM-written tool bodies, per-tool replay fidelity on 30 held-out calls, the five files loading in tau2's own harness, `intent`, and section 11 steps 5 to 8 (Verifier, Candidate runs, Verdict vs tau2 reward, k experiment).

## Asked for on 2026-08-28 (evening)

- Overfitting check (done 2026-08-28, results in `docs/cross-domain-check.md`; fixes listed below). Everything measured so far is tau2 retail: ingest shape, ToolSig matching against retail tools.py, the D73 and D99 id and time exemptions, the D98 generic tool names, the D100 cluster similarity and its 0.4 default (fit to retail task labels), the tau2 `Error: ` prefix handling. "Held out" in the README and on the website means runs held out, not domains. Run the same code with zero retuning on tau2 airline and telecom (both on Sierra's bucket, both vendored with real tools and db) and report the same gates side by side. Any constant that has to move to pass a new domain is a constant that was overfit; record it in the decision log rather than tune it quietly. Next after that: a domain that is not tau2 at all.
- Trace formats. Ingest reads only the Sierra tau2 shape; the OpenTelemetry GenAI reader is a stub that raises NotImplementedError. Add readers, each yielding the same records with raw_ptr into the original file, for: Langfuse (trace and observation export, GENERATION, SPAN and TOOL observations), OpenTelemetry GenAI semantic conventions (both the event dialect and the attribute dialect), OpenInference (Arize Phoenix), LangSmith run exports, and plain OpenAI and Anthropic message logs (request and response pairs, which is what most customers actually have). Public benchmark formats too: tau-bench v1, BFCL multi-turn, AgentBench, WebArena and SWE-bench style trajectories, AppWorld and ToolSandbox. One fixture file per format under tests/fixtures, built from a real export, and the ingest gate (counts against a hand count) run per format. The rest of the Builder must not know which reader produced the records.
- Synthetic data generation. The harness now builds and verifies the environment; the next output is synthetic data from it: new Tasks in the clusters the traces cover (and the gaps next to them), runs of a strong model in the environment, filtered by the Verifier to passing trajectories, and the report saying how many were kept and why the rest were dropped. Design questions to settle before code: what counts as a new Task versus a rewording (D100 similarity applies), how far from the observed distribution a synthetic Task may sit, and how a person audits a sample. Founder's words: "we need to have synthetic data generation now as well."
- Judge stays an agent. D92 is implemented (judge.py: read-only tools over Starting and End state, at least one check before a verdict, two judges, disagreement queue). The simplification pass must not reduce it to a transcript-reading prompt.
- The win that matters (plan in `docs/training-plan.md`, 2026-08-28: baselines on the real tau2 environment, build from the seen 80%, train the 2B model by rejection-sampling fine-tuning, on-policy distillation and RL with the Verifier as reward, plain scripts not a harness yet, score on the real held-out Tasks against the same training on the real environment). Founder's words: "the major win for this design would be to show that it can generate good environment and the models post trained on this environment are quite performant." Two claims, two measurements. Environment quality: the gates above, on held-out runs and on held-out domains, plus a person's audit of a sample of Tasks and verdicts. Post-training effect: take a small open model, post-train it on trajectories that passed the Verifier in a generated environment, and measure it on the real held-out Tasks (tau2 airline and telecom first, a customer domain after) against the same model untrained and against the frontier model that produced the traces. Report the numbers either way; this is the experiment the blog post is built on.
- Any model, one id. Founder: "did you add the model providers so that we can literally choose any model like open code and pi?" Today: adapters for anthropic and openai, and any OpenAI-compatible endpoint if you pass a base URL; prices are a hand-kept table. Not yet: a provider registry the way OpenCode (models.dev) and Pi keep one, where `provider/model` resolves to base URL, auth env var, context window and prices for every listed provider (OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, xAI, OpenRouter, Together, Fireworks, Ollama and local). Plan: vendor the models.dev JSON as the registry with a refresh script, resolve ids through it in `model_for`, keep the hand table only as an override, and report any model missing from the registry as unpriced as today.

## From the cross-domain check (2026-08-28)

Airline and telecom were run through the offline slice with nothing retuned. The numbers are in `docs/cross-domain-check.md`. Five places turned out to be shaped by retail, each with its fix. Write the test against airline and telecom first, then retail must still pass unchanged.

Four are done (2026-08-29, D101 to D104), and the check itself is now `scripts/xdomain_check.py` rather than scratch scripts outside the repository. Kind is exact on all three domains and every table is recovered on all three; retail did not move. The three still open are marked below.

- ~~Read `ToolCall.requestor` in mine, cluster and compile_env.~~ Done (2026-08-28). Telecom's mined tools fell from 38 to 21 and its clustering ceiling rose from 0.561 to 0.658. The 8 that remain are not a filter failure: the assistant in that export really does call the user's phone tools, 53 times for `check_network_status`. Old note: Only the agent's calls define the Environment; user-side tools (telecom's `user_tools.py`, 29 names) belong to the user simulator. This is the largest single cause of telecom's numbers: 38 mined tools against 13 real, `grant_app_permission` in a Category write signature, cluster F1 0.207, 240 of 356 Gate A calls with no tool to hit.
- ~~Retire `WRITE_PREFIXES` and `READ_PREFIXES` as the primary kind signal.~~ Done, D101. Three observed signals decide and the name rule is the fallback: a changed field (D68), a result that is mostly what the call sent, and a message answer about a row the tool was handed with no read ever showing it unmoved. Kind is now exact on all three domains. Old note: The observed-effect classifier (D68) recovered 5 of 6 telecom writes and 2 of airline's that the prefix list missed; make it primary and keep a small verb list only as the fallback. Airline's `book_reservation` and `send_certificate` and telecom's `send_payment_request` are real writes currently mined as read.
- ~~Widen `_is_id()` beyond `_id` suffixes.~~ Done, D102: a column the calls address whose values are distinct per row. Airline recovers `flights`. Old note: any column unique per row and referenced by other rows is an id. Airline's `flight_number` defeats the current rule and the `flights` table is never recovered despite 338 calls, which is every Gate A loss on airline.
- ~~Fix `_table_of()`'s tie-break.~~ Done, D103: the noun before the first preposition, then the id distinct across the rows it came back with. Telecom recovers `bills`. Old note: prefer the id whose singular matches the tool's object noun, fall back to the id unique within the result. Telecom files every `Bill` row under `customers` because `customer` is a token of `get_bills_for_customer` and `bill` is not.
- Accept write results that are confirmation strings or partial dicts. Read the End state from the next read of the same row instead of from the write's return value. Telecom's writes return "Roaming enabled successfully" or an id-less summary, so the End-state check finds nothing there on either database.
- Support per-task initialization actions as a Starting-state overlay. All 2,285 telecom tasks apply real tool calls to the shared database before the conversation starts, invisible to the trajectory; the one shared trace-reconstructible world is the wrong model for that domain.
- ~~`norm()` in the comparison scripts.~~ Done in `scripts/xdomain_check.py`, along with two conventions the old scratch scripts had wrong: null and absent are the same field, and a table's row key is not one of the row's fields.

- **Still open**: write results that are confirmation strings (read the End state from the next read of the same row); per-task initialization actions as a Starting-state overlay that carries values, not only a version hash (`OverlayRow` holds a `version_hash` today, so an overlay can detect a version but not reconstruct one).

## Prompt caching (2026-08-28)

My words: "you also need to explore how to cache prompts as well please or else the cost would be too much." The exploration is in `docs/prompt-caching.md`. Short version: the provider already sets Anthropic cache points and prices cache reads and writes, but the Builder's system prompts are 45 to 90 tokens, under the 1,024 token minimum, and the variable evidence sits in the user message, so nothing is cached there today. The Runner's candidate loop and the judges cache correctly.

- Move the stable bulk (schema, tool list, policy text, emit rules) into the system message in `compile_env` and `policy`; the tool or rule and the case follow in the user message.
- Retries append the failure as a new turn instead of rewriting the user message (`body_messages`, `policy` rewrite).
- Request memo on disk under `workdir/model_cache/`, keyed by model, messages, tools and config; counted as `memo_hits` in `budget.json`; on for Builder stages and judges, off for live candidate runs.
- `prompt_cache_key` per build and stage on OpenAI.
- Per-stage hit rate and dollars saved in the report.
- Measure on the first live build before and after; the Builder's read share is the number to beat.


## Execution and production monitoring (2026-08-29)

Founder's words: "we need execution monitoring and production monitoring as well haha, we need to mine those from the traces right ?", with https://trymaitai.com/ as the reference. Maitai organizes production traffic by Application and Intent, runs what it calls Sentinels (automated quality checks) against live traffic, and feeds what those catch back into fine-tuning. It takes traffic through a base-URL redirect on an OpenAI-compatible SDK or as request and response pairs.

The shape is close enough to ours that the interesting question is not what to build but who writes the checks. Theirs are configured. Ours should be mined from the customer's own traces, which is the claim the Verifier already makes offline; this is that claim pointed at live traffic instead of at a replay. Answering the founder's question directly: yes, nearly all of it is already mined, and most of the work is wiring what the Builder produces to a live stream rather than inventing new checks.

Both items extend the "Production monitoring" line in the D69 deferred components above. Neither may become a second grading path: a production fault is a flag for review, never a Verdict (D49 wording).

### Execution monitoring (inside one Run, while it happens)

What the Builder already produces that a live check can read, with nothing new to mine:

- `ToolSig.args_schema` (D72): an argument outside the observed schema, or a required argument missing. Already the `mine_gate` check, run against a live call instead of a recorded one.
- Error classes and per-tool error rates (D67): an error class this tool has never returned, or a rate outside the band the corpus shows.
- Hard constraints compiled from the policy (D43): a policy line violated mid-Run, and the sequence constraints for orderings the traces never show.
- `UserRules` disclosure rules: the agent asking for a fact the user already volunteered, which is the D44 fact-consistency check pointed at a live turn.
- Observed step counts per Task: a Run past the band its Task's References sit in, which is the cheap loop and stall detector.
- The Category write signature (D83): a Run writing through tools no confirmed Reference for that Task ever wrote through.

Gated on: a live traffic source, which is the SDK wrapper or log drain in `tech/docs/sdk-wrapper.md` and does not exist yet. Not gated on Gate A, because none of these compare against a rebuilt Environment; they compare against what the traces showed.

### Production monitoring (across Runs, over time)

- **Intent drift**. Live Runs clustered with the same D100 similarity against the build's Categories and Tasks. A Run that joins nothing is an intent the Environment was never built for, and it is the signal that says when to rebuild.
- **Tool contract drift**. Result schema, error classes and encodings compared against the mined `ToolSig`. The customer changed their API and the Environment is now stale; today nothing would notice until a rebuild.
- **Routing quality**. The routed model's live outcome against the Environment's prediction for that Task, which is the original line above and the only one that needs Gate A first.
- **Live coverage**. What share of live traffic falls in Tasks that have a confirmed Reference and a passing Verifier, extending the per-Task policy coverage line ("your traces cover 6 of 40 policy items") from the build corpus to live traffic.
- **Feed the loop**. Live Runs that trip a check are the next build's traces, and the ones that pass are candidates for the synthetic data set. This closes the loop the training plan describes and is why the mining has to be shared rather than reimplemented per surface.

Gated on the same traffic source, plus two questions that are already open and unanswered: anonymization and boundary control (what may leave the customer's boundary, ADR-0002 and trace intake question 13), and retention. A live stream makes both urgent in a way a one-off trace upload does not.

Not decided, and worth deciding before code: whether a check runs inline (able to block or correct a Run, which is what Maitai's fine-tuning loop implies) or strictly after the fact. Inline means we are in the customer's request path, which is a different product and a different trust conversation from reading their traces.

## After the first live build (2026-08-29)

The build ran end to end; docs/live-build.md has what broke and the fidelity table. Left open from it:

- **Price `openai/gpt-5.6-*`.** No row in `budget.PRICES`, so `usd` stays 0.00 and `--ceiling-usd`
  raises `UnpricedModel`. The number has to come from a person.
- **Rebuild and remeasure.** ~~Every fix below and the D106 mining fixes landed after the build
  that measured 64.3 percent.~~ Second build: 88.2 percent on 297 calls, four of six causes closed
  (docs/live-build.md, second build). Still open there: `calculate` boxes its scalar (19 calls) and
  `get_item_details` was built on the pre-D106 schema because the `mine` cache never moved (D109);
  the third build measures D106.
- **Second path needs k re-runs.** ~~D79 check 5 takes the Task's second confirmed Trace (D108); a
  Task with one Trace never passes the suite until the frontier re-runs of D78 exist.~~ D112: the
  `rerolls` stage runs the frontier three times per Task (`--rerolls`) and the finished ones enter
  the D111 rule beside the recordings, so a single-recording Task has a second path when a re-roll
  agrees with it. The k experiment (D78) still decides whether three is the number.
- **Constraint vocabulary gate.** Two of the four miscompiled retail constraints (D113) list tool
  names the corpus never shows. `policy.compile_rule` could refuse a predicate whose string
  literals name a tool or field absent from the mined signatures and schema before spending a
  rewrite on it; today the corpus-rate demotion in `reference.py` catches them after the fact.
- **Overlay conflicts on airline and telecom.** Done as a `tau2_export` gate (D113); read the
  conflict counts on the rebuilt airline and telecom workdirs and check that the pre-write version
  is the one the export kept.
- **Loophole probe cost.** One Run per Task by the model, up to six turns; on 205 Tasks that is
  the slowest thing in a build, so `--probe-limit` exists and the third build ran 40. Measure what
  the probe catches at 40 before deciding whether every Task gets one on every build or only on a
  Verifier that changed.
- **Stage side-effect files on a cache hit.** A cached stage returns its artifacts but does not
  rewrite the files it wrote the first time (`synthetic.json`, `runs/`, `probes/`); a workdir that
  lost them shows a built Environment with no replay on disk. Either files become artifacts or a
  cache hit restores them from the stage's own record.
- **Leak check hits.** The offline fixture's first Verifier trips D79 check 7 (a system-derived
  constant appears in the Intent or the user rules). Read the hits on the third live build before
  deciding whether the Intent writer or the check is wrong.
- **A UI over the Builder and Runner.** Asked 2026-08-29: "we need a good ui for the same as
  well so that it can talk to the underlying implementation." The TUI shows one build; the ask is
  a UI that drives builds, runs and reports through the same entry points the CLI uses, with no
  logic of its own (design section 3).
- **Telecom shows one customer.** 456 traces, one customer row, one device, one plan. Growing it
  is copying (docs/synthetic-rows.md, last section). Either the corpus needs traces over more
  customers or telecom's Starting state comes from the customer's snapshot (ADR-0006), not traces.
- **Tool parity for synthetic rows.** Replay recorded calls with a synthetic id substituted and
  require the same result schema and the same success or error branch (docs/synthetic-rows.md,
  practice 6). Belongs in `scripts/env_fidelity.py`.
- **Free text in synthetic rows.** Product names repeat because nothing invents them; either an
  LM-written list per free-text column (tau-bench's own method) or a customer-supplied list. A
  zip drawn with its city is the same kind of fix (a joint draw of the address record).
- **Airline and telecom live builds.** Retail is one domain. The confinement prompt, the scalar
  result rule and the error prefix rule were each written from one corpus.
- **`compile_policy` cost.** 123 of 174 calls in the build were policy sentences, one call each.
  It was the slowest stage by far. Batching sentences per call is an obvious cut; whether it
  changes the constraints it produces is the question to measure first.

- ~~Simulated user vocabulary is a closed retail list~~ Done as D115 (2026-08-29): `builder/vocabulary.py` derives the fields from the corpus, `shared/search.py` (TinyFish, Firecrawl keyless) adds ask wording. Left: `user_sim._row_value` still maps a few column aliases by hand (`payment_method` to `source`, `card_last4` to `last_four`); measure on a live build how many aliases the web adds and whether a re-roll's asks are matched more often than before.
- **tau2 export as an optional adapter** (D90, audit). `emit_tau2_shape` and `tau2_files` run on every build and the `tau2_export` gate can fail a build over an overlay conflict the Runner does not have. Keep the compiled representation format-free and put the tau2 rendering behind an adapter beside future ones.
- **Name prefix rule as a low-confidence prior** (audit). `READ_PREFIXES`, `WRITE_PREFIXES` and `GENERIC_NAME` in `mine.py` are the fallback after D101's three observed signals; a tool with too few calls still gets "medium" from the name alone. Make the name rule "low" so the observed signals always outrank it.
- **The Runner does not route an assisted tool to the recording** (D49, D114). `Environment.assisted_tools` is now filled, and nothing in `route.py` reads it: an assisted body answers every call, and a seed Trace that calls one diverges on replay. D49 says recorded calls are answered from the recording and the rest by the stand-in; wire it, then measure how many of the six retail assisted tools' Tasks come back.
- **Rebuild retail, airline and telecom with D114** and read: the refusal probe's failures per write tool, `after_write_skipped` per tool, the assisted list on `environment.json`, `constraints_check.json` demoted rules against the founder's count of 15, and `scripts/reference_agreement.py` per domain. Airline stopped at the build_environment gate on `_ids_in`; it has not run past the environment stage yet.
- **Simplification proposals not applied** (parallel pass, 2026-08-29; the two applied were `cluster.py`'s duplicated `is_assistant_call` and `mine.py`'s self-alias). Near-duplicates left alone because they differ in behaviour: `_get` in `gate_support.py` and `memory.py` (the first also matches `class_` to `class`), `_append` in `judge.py` and `canon.py` (`default=str` against `sort_keys=True`), `_runs` in `mine.py` and `synth.py` (ASCII-only letters against Unicode letters), the `a_run`/`a_verdict` test factories in `test_validate.py` and `test_report.py` (different contracts under one name), and the module splits section 10 of the design already lists.

## First live build (2026-08-29)

Nothing in the Builder has ever run against a real model. Every stage above `state` is written and
unit tested against `TestModel`, and four things are untested for that reason alone, all of them in
section 11 steps 5 to 8: whether the model writes tool bodies that compile and pass the sandbox
gates, whether those bodies replay the corpus per tool at the fidelity the gate demands, whether
`compile_policy` and `judge_lessons` produce anything a Verifier can use, and whether the whole
graph gets through on a budget worth paying. Mutation testing cannot answer any of them, because
the thing under test is what the model writes, not what we wrote.

To run one:

1. `cp .env.example .env` and fill in `OPENAI_API_KEY`. `.env` is gitignored.
2. `uv run harness ingest <export>.json --workdir .work` (or hand the file to `/build --file`).
3. `uv run harness tui --workdir .work --model openai/gpt-5.6-luna --ceiling-usd 25`
4. `/keys` to confirm the shell sees the key, then `/build`.

The ceiling is the thing to set first. `budget.Ceiling` stops the graph before a stage rather than
during it, so a low ceiling on the first run costs one stage's worth of tokens to find out that a
prompt is wrong. Raise it once `compile_tools` has got through once.

Remaining, not done:

- Mutation survivors above 200 per module: `report` 681, `provider` 498, `verifier` 476, `memory`
  343, `judge` 263. `gate_support._passed` is closed. The next holes by value are
  `sandbox._classify_exception` (39), `verdict.verdict` (37), `validate._run_predicate` (30),
  `validate.candidate_runs_gate` (28), `verifier._hard_holds` (20), `boundary.runner_version` (17).
- The mutmut results for `mine.py`, `sandbox.py` and `compile_env.py` are stale after D101 to D104
  and need a re-run before their counts mean anything.
- Cross-domain, still open: write results that are only a confirmation string (read the End state
  from the next read of the same row); per-task initialization actions as a Starting-state overlay
  that carries values, since `OverlayRow` holds only a `version_hash` today.

## Remove the benchmark agreement lines (D112)

The scorecard's per-domain agreement with tau2's reward (D111) is scaffolding for improving the Reference rule on the benchmarks. Once the rule holds across retail, airline and telecom, delete the lines and the sidecar read so nothing can be tuned to them. Founder: "later on we delete it and make sure we are not overfitting to the score card."
- Every Run's `stop` event carries the whole Starting state (`loop.py` `start_state`); with the grown retail world that is 3 MB per re-roll, 1.8 GB per build, and `runs.json` mirrors it. The Environment id already names that state; the atom context and `policy.py` should load it from the Environment and a Run should carry only what differs. Not done because three readers depend on the payload shape.
