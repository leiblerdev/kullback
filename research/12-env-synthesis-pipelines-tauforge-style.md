# End-to-end synthesis of verifiable agent environments (policy + DB + tools + tasks + verifiers): state of the art, Aug 2026

Source: research agent, 2026-08-26. WebSearch budget was exhausted; arXiv/GitHub fetched directly.

## 0. Naming note: "TauForge" is not public

No arXiv, GitHub, or API hit exists for "TauForge", "tau-forge", or "τ-Forge" (arXiv title search for "Forge" + agent returns 25 papers, none of them: https://export.arxiv.org/api/query?search_query=ti:Forge+AND+%28abs:agent+OR+abs:environment%29; GitHub repo search: https://api.github.com/search/repositories?q=tauforge+OR+tau-forge). The two-phase design you describe (phase 1: policy, DB, tools grounded in usage; phase 2: seeds, scenarios, personas, traps, tasks, rubrics, verifiable rewards) is however closely matched by the public systems below, especially LOGIGEN, AutoForge, ScaleEnv, EnvScaler, AgentScaler, and Envs-FORGE (a different "Forge"). "EnvRigger" does exist: it is the synthesis agent inside EnvHarness (Aug 2026).

## 1. Pipelines that synthesize the whole stack (tau-bench style)

### LOGIGEN (Feb 2026), https://arxiv.org/abs/2603.00540
- Inputs: minimal seed domain knowledge, expanded by an "Architect" agent into a wiki policy P (business rules, access privileges, transaction flows, data structures).
- DB/data: policy is hard-compiled into a normalized relational schema plus SQL triggers: "BEFORE INSERT/UPDATE triggers validate preconditions and raise a POLICY_VIOLATION error"; AFTER triggers implement side effects (audit rows, inventory deduction). A Verifier Agent checks semantic consistency and physical executability. Initial state uses "boundary-adjacent seeding": numeric constraints seeded at N-1, discrete states "exactly one valid action away from triggering a rule violation". Resource injection adds trade-offs, distractors, substitutes, noise; cast assembly instantiates four user archetypes (Mismatch, Entangled, Rookie, Edge).
- Tools: atomic tools over the DB; tool descriptions expose trigger logic as preconditions and side effects.
- Tasks: an Explorer agent does Client/Consultant dual-role exploration (consultant reads state and policy to propose feasible options; client picks goals) and produces a spoiler-free instruction with tool names and IDs stripped. Three tiers: L1 51 tasks, L2 6,161, L3 3,788; >20k tasks over >2,000 policy environments, 8 domains.
- Verifiable reward: target state comes from actually executing the explored path under the compiled constraints ("freedom from ground-truth drift"). Reward = I[DIFF(s_final, s_target)=0] on canonicalized relational sets minus technical keys; dense shaped reward from state-proximity deltas and a penalty for trigger-caught policy violations; TA-GRPO uses turn-level advantages.
- Validation: only trajectories that reach s_target are kept for SFT; no human audit reported.
- Numbers: Qwen3-32B 40.7% to 79.5% on τ²-bench (RL); 8B 30.8% to 71.8%; beats AgentScaler-8B (49.4%) by 22.4 points.

### AutoForge (Dec 2025), https://arxiv.org/abs/2512.22857
- Inputs: tool description documentation only.
- DB: state as key-value attributes, names generated first, values deferred to task instantiation. Tools: LLM-written Python conforming to that state. Dependency graph edges where "the output of one tool is likely to be a valid input for another"; random walks give tool sequences; sequences are merged and augmented with reasoning nodes/edges into a DAG blueprint.
- Task instantiation: generate values, execute the gold sequence topologically to get the golden state, then write a minimal natural instruction.
- Reward: 1 iff final state equals golden state; state-based rather than sequence-based "since multiple valid sequences may exist". DAPO dynamic sampling drops all-correct/all-wrong groups. ERPO adds LLM-as-judge masking of trajectories that failed due to simulated-user errors.
- Human: tool docs, manual tuning of user-simulator prompts.
- Numbers: 10 envs, 1,078 hard tasks, Qwen3-Thinking-235B as synthesizer, GPT-4.1 as user sim, 64 GPUs; 73.1 τ-bench Retail, 76.3 τ²-Telecom.

### ScaleEnv (Feb 2026), https://arxiv.org/abs/2602.06820
- Inputs: a domain keyword ("Job Seeking").
- Tools first, then DB: tool schema with pre/post conditions is generated top-down; a Database Agent "reverse-engineer[s] the database structure" from the tools. Code Agent implements tools while a Test Agent "synthesizes unit test cases and corresponding matched database instances"; outcomes are Success, Anticipated Rejection, or Unexpected Failure, the last triggering debug-until-pass. A Tool Dependency Agent builds the graph.
- Tasks: sample an executable seed chain C1 as the reference solution; populate state to support it plus "distractors... functionally orthogonal to the ground-truth trajectory"; persona and instruction are "strictly grounded in C1". Difficulty via dependency-aware BFS expansion gated by an LLM on structural complexity and an oracle-agent feasibility score; minimum 20 tools in the visible set.
- Reward: rule-based diff of final DB vs ground truth with three column classes: exempt (generated IDs), hard (exact), semantic (fuzzy). Explicitly chosen to avoid LLM-judge reward hacking.
- Cost: ~546k tokens per domain foundation, ~93.2k tokens per task; 16 domains, ~50 tools and 5-20 tables each, 2,560 tasks. Qwen3-8B +12.5 absolute on τ²-bench.

### EnvScaler (Jan 2026), https://arxiv.org/abs/2601.05808
- Inputs: themes reverse-inferred from existing task datasets (API-Bank, ToolACE) rather than a toolset.
- Env: Python class per environment (state attributes, rules, tool methods) via logic planning, program modeling, program assembly with AST syntax checks.
- Tool correctness: a dual-agent loop where a testing agent fires positive and negative calls without seeing code, and a checking agent inspects code, results and state deltas; "the average judging pass rate serves as the quantitative metric"; envs below 0.85 over 100 rounds are discarded.
- Tasks: initial state generated first, then a challenging task from that state. Reward: task decomposed into conditions, each gets an LLM-generated terminal-state check function; reward = fraction passed (tolerates alternative paths).
- Numbers: 191 envs, 18.6 tools and 21.4 state categories each, ~7k scenarios; Qwen3-8B +3.16 τ-bench, +11.67 ACEBench-Agent from SFT, more with RL.

### AgentScaler (Sep 2025), https://arxiv.org/abs/2509.13311
- 30k+ APIs clustered by parameter similarity and Louvain into 1,000+ domains; each function tagged read or write over a shared per-domain schema; tools materialized as Python over that DB. Tasks by directed walks over the dependency graph with actual execution. Three-stage filter: validity, final-DB-state alignment with the golden state, exact function-call match for read-only tasks. Trajectories with intermediate tool errors are kept if the final state is right.

### Envs-FORGE (Aug 2026), https://arxiv.org/abs/2608.14312
- The clearest published "hardening by pass rate" loop. Seeds are (instruction, fixtures, oracle solution, tests, Docker env). Per seed: estimate pass rate from policy rollouts; project six actions (increase complexity Δ=-0.25, reduce +0.25, diversify 0; in-depth or in-breadth with γ=0.65); solve a MILP maximizing frontier utility exp(-(p-0.5)²/2·0.2²) subject to skill coverage; rewrite instruction, fixtures, oracle, tests and Docker jointly. Contract: "Every tested threshold, tie-break, filename, output key, or fixture value must be stated in the instruction rather than hidden in the verifier." Accept only if static checks pass and the oracle scores reward 1.
- Cost: 22.7k-28.8k tokens and 1.9-2.9 attempts per accepted environment; 100 verified envs per run. Qwen3.5-35B + GRPO: tb-core 40.0 to 49.2, τ²-bench 23.0 to 29.4, SWE-bench Verified 73.4 to 77.1.

### EnvFactory (May 2026), https://arxiv.org/abs/2605.18703
- Search Agent mines API docs and usage examples for coverage gaps; Code Agent derives a stateful schema as Pydantic models with load/dump for session isolation; tools wrapped as MCP. Test Agent checks four criteria: metadata consistency, import/execute, expected behavior, correct state transitions. 85 verified envs from 842 tools. Query refinement adds implicit references, action compression, ambiguity, goal expansion. Reward R = 0.5·trajectory match + 0.5·final-state equivalence minus a length penalty; ablations show either alone is worse. ~20 GPU-hours per 1,000 SFT trajectories.

### Agent-World (Apr 2026), https://arxiv.org/abs/2604.18292
- Real grounding: MCP servers from Smithery, tool docs, and industrial PRDs; a deep-research agent mines real-world databases from the web, then complexifies them. Tools coded and cross-validated (>50% test accuracy). Tasks via graph random walks (rubric LLM-judge reward) or programmatic Python solutions with paired validation scripts (executable reward). Pass@5 filter keeps tasks with at least 2 of 5 successes. Self-evolution: evaluate, diagnose, target, retrain; hardening by longer walks, weaker dependencies, and rewriting to hide tool names. 1,978 envs, 19,822 tools; +3.3 to +6.8 per round.

### AgentOmnia (Jul 2026), https://arxiv.org/abs/2607.23124
- 5,018 envs, 255k tools, 52k tasks (DAG, program, solver-based). Rubrics derived from task, initial state and validated trajectory, scored without requiring reference call order. Failures become PRDs (also accepted from human product managers) that condition the next synthesis round; rollback curriculum restarts from golden-trajectory prefixes when a task group all-fails. Challenging-subset pass rate 9.16% to 37.11%.

### EnvHarness / EnvRigger (Aug 2026), https://arxiv.org/abs/2608.19880
- Different design point: the verifier is frozen ("success is the benchmark's own verdict, so reshaping reward cannot move the eval metric") and EnvRigger synthesizes wrappers: Stage (initial-state mutations), Contract (action/transition/observation rewrites), Chain (env composition). Loop: observe rollouts, diagnose systemic flaws (loops, long-observation parsing, misread tool constraints), write hooks, validate with fresh rollouts and ACCEPT/REFINE/REJECT toward a target success band; unsolvability signals (timeouts, SR=0 on the action axis) force loosening. Accepts optional human-specified weaknesses. Gains: ALFWorld OOD +9.0, SWE-bench Verified +2.7 with 9.8% fewer steps; scaling from 47.67 to 54.79 at 300 envs.

## 2. Adjacent pipelines worth stealing from

- APIGen-MT (Salesforce, Apr 2025), https://arxiv.org/abs/2504.03601: blueprint = (instruction, ground-truth actions, expected outputs) from samplers over τ-bench APIs, policy text, PersonaHub personas, DB rows. Three gates: format/execution/policy (policies "translated into Python unit tests"), LLM committee on the DB diff_patch (correctness, completeness, satisfaction, creativity, majority vote), semantic review with feedback loops. Task config success 70% vs 28% without feedback; trajectory simulation success 67%; simulated human with best-of-4 plus self-critique; keep only r=1 trajectories (state equals a_gt and response contains o_gt); reverse recombination of validated blueprints sharing a persona. 5k trajectories; xLAM-2-70b 56.2% τ-bench.
- CUA-Gym (May 2026), https://arxiv.org/abs/2605.25624: Generator writes initial_setup.py and golden_patch.py, Discriminator writes reward.py "from the task description alone" behind an information barrier; forbidden-pattern scan; must satisfy r(s_init) < r(s_gold); LLM majority voting on consistency, executability, hack risk, difficulty; teacher rollout (Claude Sonnet 4.6) retains only solved tasks. 32,112 verified tuples, 110 mocks.
- Simia (Nov 2025), https://arxiv.org/abs/2511.01824: no DB or tools; an LLM simulates observations from tool specs, policy and one seed trajectory; o4-mini judges success. Cheap (5k seeds to 90k trajectories, 58.9 τ²-bench for 32B) but the reward is an LLM verdict, not verifiable.
- AutoEnv (Nov 2025), https://arxiv.org/abs/2511.19304: three-layer env DSL, coding agent with self-repair, then execution test, level validators (goal reachability), and differential testing (if GPT-4o-mini beats DeepSeek-V3.1, the reward is unreliable). 65% overall yield, $4.12 per env; human theme review lifted yield 60% to 80%.
- SPADE (Aug 2026), https://arxiv.org/abs/2608.19197: designer generates Gym-style envs and tool-use tasks; deterministic reset gate over seeds plus an LLM check that every success criterion is reachable; designer reward = hint-regret and a flat-top band on win rate [0.4, 0.6].
- RLAnything (Feb 2026), https://arxiv.org/abs/2602.02488: task mutations accepted only if new accuracy lands between 0.2 and 0.8 relative to the old.
- ASTRA (Jan 2026), https://arxiv.org/abs/2601.21558: 1,585 MCP servers, sub-envs accepted only if executing the generated code yields the target answer; F1 reward of solved-tasks vs tool calls; 20% injected tool failures.
- AReaL-SEA (Jan 2026), https://arxiv.org/abs/2601.22607: per-instance executable checkers plus a trajectory-verification agent that attributes failures to Task vs Trajectory and routes each to a reflection loop; removing validation agents drops pass¹ 56.0 to 50.0, removing self-evolution to 44.0. Telecom 53.7 to 98.3.
- Toucan (Oct 2025), https://arxiv.org/abs/2510.01179: 2,800 MCP servers filtered to 871 accessible then 495 healthy via generated test questions; six Likert quality dimensions incl. verifiability and stability.
- SWE-smith (Apr 2025), https://arxiv.org/abs/2504.21798: keep only patches that break passing tests; total cost $1,360 for 50k instances (2.54 cents each), ~20 human hours; difficulty from expert-model resolve rate (58.6/41.0/17.0%). R2E-Gym, https://arxiv.org/abs/2504.07164: hybrid execution + execution-free verifiers, 51% SWE-bench Verified.
- Reward-hack and judge checks: Hack-Verifiable Terminal Bench plants honeypot solutions/tests with inotify watchers, hack rates 12-60% across frontier models (https://arxiv.org/abs/2608.22103); RubricForge induces rubrics from 85 labeled τ-bench trajectories and halves false-pass rate 0.173 to 0.115 vs G-Eval (https://arxiv.org/abs/2608.13564); AgenticAI-Supervisor found constraint misrepresentation in ~40% of positively rewarded episodes under outcome-only rewards, fixed with side-effect entity-count checks (https://arxiv.org/abs/2607.05773).
- Infrastructure only, no generation: OpenEnv (step/reset/state, Docker on HF Spaces, `openenv init`/`push`/`import` from Verifiers, https://github.com/meta-pytorch/OpenEnv); Prime Intellect Environments Hub on the verifiers library (https://www.primeintellect.ai/blog/environments); InternAgentHarness four-layer interface with rejection sampling at score 0.8 and failure-to-task loop (https://arxiv.org/abs/2508.08636). Procedural generator+verifier suites: Reasoning Gym 100+ generators (https://arxiv.org/abs/2505.24760), Enigmata 36 tasks (https://arxiv.org/abs/2505.19914), SynLogic 35 tasks (https://arxiv.org/abs/2505.19641). AutoRule extracts rules from preference reasoning as auxiliary verifiable reward (https://arxiv.org/abs/2506.15651). AgentSynth chains persona-conditioned subtasks at $0.60/trajectory with an 88%-accurate screenshot judge (https://arxiv.org/abs/2506.14205). Text2World checks LLM-written PDDL by execution (https://arxiv.org/abs/2502.13092). ToolUniverse standardizes 2,700 tools with tool-creation/testing agents (https://arxiv.org/abs/2509.23426).

## 3. Cross-cutting patterns

1. Ground truth by execution, never by generation: every credible system replays a reference path in the real engine to produce s_target (LOGIGEN, AutoForge, ScaleEnv, AgentScaler, Envs-FORGE, CUA-Gym). Simia is the exception and pays with an LLM-judged reward.
2. Policy as code: LOGIGEN's SQL triggers and APIGen-MT's policy unit tests make "policy compliance" a mechanical check and make traps (boundary states) free.
3. Reward = state diff with column classes (ScaleEnv exempt/hard/semantic; LOGIGEN excludes technical keys) plus an output check (APIGen-MT o_gt), optionally mixed with trajectory match (EnvFactory 0.5/0.5).
4. Verifier validation triad: oracle passes (Envs-FORGE, LOGIGEN), null state fails (CUA-Gym r(s_init) < r(s_gold)), independent judge agreement (APIGen-MT committee, CUA-Gym voting, Agent-World Pass@5). Only AutoEnv and AutoForge report differential tests between models.
5. Difficulty calibration by pass rate is now standard: target 0.5 (Envs-FORGE), band [0.4, 0.6] (SPADE), [0.2, 0.8] (RLAnything), 2/5 solvable (Agent-World), DAPO dropping 0/8 and 8/8 groups (AutoForge).
6. Human input is thin and front-loaded: seed docs, personas, taxonomy (Agent-World 3 annotators), simulator prompt tuning (AutoForge), PRDs (AgentOmnia), theme review (AutoEnv +20 points yield). Nobody reports human audit rates of generated verifiers; the closest are RubricForge's 173 labeled trajectories and SWE-smith's 20 hours.
7. Reported costs: $4.12/env (AutoEnv), ~25k tokens/env (Envs-FORGE), ~546k tokens/domain + 93k/task (ScaleEnv), 2.54 cents/instance (SWE-smith), $0.60/trajectory (AgentSynth).

## 4. A minimal version for a two-person team, grounded in a customer's agent traces

Goal: one customer domain, 1 policy doc, 1 DB, 10-25 tools, 150-300 verified tasks in two weeks. Skip anything that does not raise verifier trust.

Phase 1, environment from traces (days 1-4)
1. Mine the traces, not the docs. From N production traces extract the observed tool set, argument distributions, entity ID formats, error messages, and the write-set per tool (which fields change). This is the grounding AgentScaler and EnvFactory approximate with public APIs; you have the real thing.
2. Schema by reverse-engineering from tools (ScaleEnv order): tables = entities touched by writes; columns = fields observed in tool outputs. Populate with anonymized real rows plus LLM-generated rows in the same distributions. Keep 5-20 tables.
3. Tools as Python over SQLite (one file per tool, Pydantic in/out, EnvFactory's load/dump state snapshot for isolation). Generate from trace-observed signatures. Correctness gate: replay 30-50 real trace tool calls through your tools and require the returned fields to match the recorded outputs on the hard columns (this beats EnvScaler's dual-agent judging because you have recordings).
4. Policy as constraints (LOGIGEN): turn every "must/never" sentence in the policy doc into either a BEFORE trigger raising POLICY_VIOLATION or a pure Python predicate; write one positive and one negative unit test per rule (APIGen-MT). Rules you cannot compile go into a short list for the LLM judge, nothing else.
5. User simulator: reuse the customer's real opening messages as personas; add best-of-4 self-critique only if drift shows up (APIGen-MT).

Phase 2, tasks and verifiable rewards (days 5-10)
6. Seed abstract tasks from trace intents (cluster first user turns into 15-30 intents). Persona x scenario x trap sampling: scenario = a concrete DB row set; trap = LOGIGEN boundary seeding (one step from a rule) or a distractor entity (ScaleEnv). Sample 10-20 per intent.
7. Reference solution by execution: an oracle agent (frontier model with policy and full schema visible) solves each task in the real env; record the tool sequence and snapshot s_target. Discard tasks the oracle cannot solve in 3 tries (Envs-FORGE acceptance rule).
8. Verifier per task, in this exact form:
   - state check: DB diff s_final vs s_target over the write-set tables only, with three column classes (exempt generated IDs/timestamps, hard exact, semantic fuzzy);
   - action check: set of required write tool calls with argument constraints (order-free, so alternative paths pass);
   - communicate-info check: substring or normalized-number match on outputs the user must be told (APIGen-MT o_gt);
   - policy check: the compiled triggers already fired; count violations as failure;
   - side-effect check: entity counts vs baseline to catch spurious creations (AgenticAI-Supervisor).
   No NL rubric rewards in the training signal. Use a rubric only as an offline audit.
9. Validate each verifier before it enters the pool (the part most teams skip):
   a. oracle trajectory scores 1;
   b. null agent (no calls, polite reply) scores 0 and s_init scores below s_gold (CUA-Gym);
   c. a wrong-but-plausible path (oracle told to skip the trap) scores 0; if it scores 1 the trap is decorative and the task is dropped or rewritten;
   d. a second oracle run with a different model; if both pass with different tool orders and the verifier accepts both, alternative-path tolerance is confirmed;
   e. leakage: grep the instruction for tool names, internal IDs and verifier constants (LOGIGEN spoiler-free rule, Envs-FORGE contract that every tested threshold appears in the instruction);
   f. human audit of a 10% sample (about 20-30 tasks): read instruction plus verifier, label "would a competent human agree". Target >90% agreement; if lower, fix the generator prompt, not the individual tasks.
10. Difficulty calibration: run your training policy 8 times per task; drop pass rate 8/8 and 0/8 (DAPO, AutoForge); keep the rest, and every week regenerate variants of tasks in the 0.4-0.6 band with one added constraint (Envs-FORGE in-depth) and re-run steps 7-9. Failed-rollout diagnosis in prose (EnvRigger style) is a good use of the second person's time: read 20 failures, name the systemic flaw, ask the generator for 30 tasks that exercise it.

What to skip
- Skip LLM-simulated tools (Simia) and LLM-judged rewards; both ship broken agents silently (RubricForge false-pass 11-17%).
- Skip graph random walks and thousands of domains; one grounded domain with 200 audited tasks outperformed 5x more environments in EnvFactory's results.
- Skip a MILP curriculum; a pass-rate band filter and one hardening rewrite per week gets most of the benefit.
- Skip trajectory-order matching; keep it order-free or you will punish valid paths.
- Skip building the harness: use OpenEnv or Verifiers for step/reset/state and Docker packaging.

Expected budget: roughly 1-3M tokens for phase 1, ~100k tokens per accepted task (ScaleEnv's 93k is a fair estimate), and about 20-30 human hours, mostly on tool replay checks and the 10% audit.