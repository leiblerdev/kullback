# Monitoring tool: deferred work

Items the design has explicitly pushed past the first build. Each line says what it buys and what gates it. Dates are when the item was deferred.

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
- **Hacker pass over the Verifier pool** (2026-08-27, R22 item 11, 2606.08960). An agent told to reach a pass without doing the work, against each Task's Verifier; a hit means the Verifier is too narrow or leaks. Cheap on compiled Hard constraints and DB-diff atoms; fits step 8 validation, before hardening (D54 stage 4).
- **Per-Task policy coverage number** (2026-08-27, R22 item 10, 2608.06329). Count which policy items the customer's traces exercise ("your traces cover 6 of 40 policy items") so the report says what was not tested.

- **Self-generated trace setup** (2026-08-27, D60). Chatwoot or Zammad in Docker, an existing MCP server (Zammad 52 tools), two external agent harnesses and models, tau2's user simulator, injected real HTTP errors. Output in tau2 native, OTel and MCP formats. Labeled self-generated everywhere; never the only tuning input for the Builder.

## Components named and deferred (2026-08-27, D69)

The founder chose which of the missing components go into the first build (clusters, intent writer, canonicalizer, report, cost and budget accounting, provider adapter, Builder memory and self-modification). The rest:

- **Filter and Screen** (eval-design step 2, Match then Appeal). Which Runs are worth re-executing. The first build runs everything; needed at the first customer with more Runs than budget.
- **Reference confirmation as a module** (step 5, D57: Hard constraints on the Reference, outcome signals, k re-rolls, judge that abstains to a human). The tau2 slice uses tau2's own references; needed before the first customer Verifier, because D43 atoms derive from confirmed References only.
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
- **Production monitoring**. Comparing the routed model's live traffic against the Environment's predictions; the "monitoring" in the product name; nothing designed yet.

## Mining (2026-08-27)

- **Mine user behaviour, not only tools** (founder, 2026-08-27). `mine.py` today produces `ToolSig` and `EntitySchema` from tool calls; the user turns carry a second body of knowledge: what facts users volunteer before being asked versus only on request, how many turns they tolerate before frustration or walk-away, how they phrase the same request, how often they change their mind mid-Run, which questions they refuse. Output would be a per-customer `UserBehaviour` record feeding `user_sim.py` (D44 style representativeness), the disclosure rules in `UserRules`, and the report ("your users volunteer the order id 80% of the time"). Related: "Simulated user from production traces" above. Gated on: enough multi-turn Runs per Task to measure it (D44 held-out requirement).

- **Experiment: choose k, the frontier re-run count** (2026-08-27, D78). tau2 retail first: 10 re-runs per Task, required and allowed sets from the first k for k = 2 to 10, the k at which sets stop changing per Task, Verdict agreement with tau2 reward at each k, cost per Task. Compare fixed k with a stop-when-stable rule (start 3, stop after 2 quiet additions, ceiling 10). Output: one table, the default k in config, A30 confirmed or refuted.

- **OpenEnv wrapper** (2026-08-27, D90). `reset() / step() / state()` over `route.py` and the Task's Starting state, about a hundred lines. Gated on the tau2 slice passing its gates. The loop is already a one-turn function with a step-by-step test, so this is packaging, not redesign.

## From the first build and offline slice (2026-08-28)

Source: `harness/SLICE_RESULTS.md` and the build workflow's review and fix passes. 756 tests pass; no model calls were made anywhere.

- **Cluster threshold** (D97 revision by slice). Shipped `DEFAULT_THRESHOLD = 0.3` gives 74 Tasks from 456 Runs with purity 0.53 and one Task of 58 Runs (F1 0.276 against tau2's task_id groups). At 0.6 the same code gives F1 0.720 and purity 0.87. Ceiling for any tool-set clustering is 79.8% (only 91 of 114 tau2 task_ids have all four trials writing through the same tools). Decide: raise the default to 0.6 or replace token Jaccard.
- **`propose_kind` cannot return `generic`**. tau2's `calculate` and `transfer_to_human_agents` fall to D70's read plus `unclassified`. Either a third kind or an explicit rule that pure functions and hand-offs are reads.
- **Error payload prefix**. tau2's harness prepends `Error: ` to every tool error; a replay compares the raw payload against the tool's own message and would count 10 of 10 as mismatches without stripping it. Canon needs a per-source error-prefix rule.
- **End-state reference row**. Gate A must compare against the row a write returned, not the last row read; last-sighting gave one false mismatch in 95 (a stale gift-card balance).
- **`tools_declared` is empty on every tau2 trace**; the export carries no tools list, so every ToolSig is `source: observed`. The declared-vs-observed merge is untested on real data.
- **`pipeline.build` and `run_batch` are not written**. `harness build` and `harness run` exit 2. They need a live Model (tool bodies, policy predicates, kind classification) and provider config; write them when a key is in `harness/.env`.
- **Model-written tool code runs in-process** (`compile_env.load_toolkit`). The subprocess Sandbox exists for compile checks only. Still gated as "Sandbox for model-written tool code" above; must land before any customer trace is compiled.
- **Size**. `src/` is 8,911 lines against the design's 2,700 to 3,700. Every module says what it carries beyond its band (overlays, repair loop, judge, memory tree, provider adapters). Decide whether the bands or the code change.
- **Could not run offline**: compile gates on LLM-written tool bodies, per-tool replay fidelity on 30 held-out calls, the five files loading in tau2's own harness, `intent`, and section 11 steps 5 to 8 (Verifier, Candidate runs, Verdict vs tau2 reward, k experiment).
