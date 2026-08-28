# Automated generation of executable environments from production agent traces

Research sweep, 2026-08-26. Source: web research agent. Topic 9. (Search budget exhausted mid-task; sources via direct arXiv/repo fetch.)

## 1. Landscape

### 1.1 Hand-built, real-code environments (the fidelity ceiling)
- **tau-bench / tau2-bench.** Domain = JSON database (initial state) + Python tool set + policy markdown + tasks JSON; reset = reload JSON. tau2 **compositional task generation** (telecom): each atomic subtask is a triple (init function, solution function, assertion function); subtasks grouped into mutually exclusive groups; composites pick at most one per group; valid iff init + solution satisfies all assertions. 15 groups gave 2,285 candidates, 114 subsampled ([repo](https://github.com/sierra-research/tau2-bench), [arXiv 2506.07982](https://arxiv.org/html/2506.07982)).
- **AppWorld.** 9 apps, 457 APIs, 60K lines of real code, SQLite (101 tables, ~360K rows). Hash-based DB diff; pass iff expected ⊆ diff ⊆ expected ∪ allowed. Time frozen with freezegun; reset = fresh copy of Base DB ([arXiv 2407.18901](https://arxiv.org/html/2407.18901)).
- **ToolSandbox.** Stateful tools over a world state; hardest: state dependency, canonicalization, insufficient information ([arXiv 2408.04682](https://arxiv.org/abs/2408.04682)).
- **Meta ARE / Gaia2.** Apps are stateful objects with tools tagged read or write; state = app states + simulated time + notifications; deterministic given fixed start. Scenario = initial state + user message + event DAG + oracle write actions. Verifier: hard checks on IDs, soft LLM checks on free text, causality, timing. 98% agreement, 0.99 precision on 450 labeled trajectories vs 72% / 0.53 for a naive LLM judge. Scenarios at 0% or 100% pass flagged broken. Judge hacking observed during RL; added global sanity checks ([arXiv 2509.17158](https://arxiv.org/html/2509.17158), [repo](https://github.com/facebookresearch/meta-agents-research-environments)).
- **MCPMark.** 127 CRUD tasks with curated initial state and expert-refined verification scripts (3-5 expert-hours/task). >50% of failures are "implicit": agent thinks it finished, state is wrong ([arXiv 2509.24002](https://arxiv.org/html/2509.24002)).
- **MCP-Universe.** Format / static / dynamic evaluators ([arXiv 2508.14704](https://arxiv.org/abs/2508.14704)). **Toolathlon.** Locally containerized replicas (Poste.io, Canvas, WooCommerce) per task with init scripts ([arXiv 2510.25726](https://arxiv.org/html/2510.25726)). **ClawsBench** (mock Gmail/Slack/Calendar with deterministic snapshot/restore) ([arXiv 2604.05172](https://arxiv.org/abs/2604.05172)). **REAL** ([arXiv 2504.11543](https://arxiv.org/abs/2504.11543)).

### 1.2 LLM-synthesized executable environments (code generated, then executed)
- **EnvScaler (Jan 2026).** Closest match to "generate an executable tool sandbox automatically". SkelBuilder: infer environment descriptions from a task corpus, three-stage LLM pipeline (logic planning of state/rules/tools; program modeling with state as class attributes and tools as methods; assembly with AST checks). Dual-agent QA loop: a testing agent fires random positive/negative tool calls for 100 rounds, a checking agent inspects code and state deltas; environments under 0.85 pass rate dropped (~28% removed). ScenGenerator generates initial state first, derives tasks from it, emits per-checkpoint terminal-state validation functions. ~$1.02 per environment, $0.064 per scenario; 191 environments, ~7K scenarios ([arXiv 2601.05808](https://arxiv.org/html/2601.05808)).
- **APIGen-MT.** Blueprints validated by executing against the real environment, policy unit tests, LLM committee; acceptance 28% to 70% with feedback. Phase 2 simulated human; keep trajectories whose final state matches (67% yield) ([arXiv 2504.03601](https://arxiv.org/html/2504.03601)).
- **AutoEnv (Nov 2025).** YAML DSL to code with self-repair; verification incl. differential testing (a weaker model beating a stronger one flags a broken reward); 65% validity at $4.12/env, 80% with human review of descriptions ([arXiv 2511.19304](https://arxiv.org/html/2511.19304)).
- **CUA-Gym (May 2026)**: Generator builds states, Discriminator writes reward functions; 32,112 verified tuples ([arXiv 2605.25624](https://arxiv.org/abs/2605.25624)). **GUI-GENESIS** ([arXiv 2602.14093](https://arxiv.org/abs/2602.14093)). **AutoWebWorld** (FSM websites, $0.04/trajectory) ([arXiv 2602.14296](https://arxiv.org/abs/2602.14296)). **ShopGym / ShopArena (May 2026)**: live storefronts to sandboxes via an anonymized intermediate spec ([arXiv 2605.16116](https://arxiv.org/html/2605.16116)). **E-Bench**: graph-guided DB filling, DB-diff grading ([arXiv 2607.23722](https://arxiv.org/abs/2607.23722)). **Envs-FORGE**, **Meta-Task** (execute and verify inside a real container before acceptance).
- Earlier: EnvGen ([arXiv 2403.12014](https://arxiv.org/abs/2403.12014)); Text2World shows LLMs are weak at correct PDDL world models ([arXiv 2502.13092](https://arxiv.org/abs/2502.13092)).

### 1.3 LLM-simulated tools (model plays the environment)
- **Simia-SFT / Simia-RL (Nov 2025)**: simulator prompted with tool specs, a reference trajectory, history; no database; consistency left to the context window; synthetic-only matched real data on tau2 but authors note simulator misalignment ([arXiv 2511.01824](https://arxiv.org/html/2511.01824)).
- **DreamGym (Meta)**: experience model over abstract textual state with retrieved real transitions; cross-domain transfer failed ([arXiv 2511.03773](https://arxiv.org/html/2511.03773)).
- **Kimi K2**: 3,000+ real MCP tools plus 20,000+ synthetic, stateful simulator, persona users, rubric tasks; real sandboxes reserved for coding ([arXiv 2507.20534](https://arxiv.org/html/2507.20534)). **Toucan-1.5M** executes ~500 real MCP servers ([arXiv 2510.01179](https://arxiv.org/abs/2510.01179)).
- Learned simulators: **MirrorAPI** (fidelity via BLEU/cosine vs real; failure responses highly templated) ([arXiv 2503.20527](https://arxiv.org/html/2503.20527)); **Qwen-AgentWorld** (Jun 2026) language world models with fidelity refinement, AgentWorldModel-1K on OpenEnv ([arXiv 2606.24597](https://arxiv.org/abs/2606.24597)); **WebWorld** Factuality and Turing scores; admits sycophancy, "overly optimistic outcomes that cater to the agent's action" ([arXiv 2602.14721](https://arxiv.org/html/2602.14721)).
- Cached replay: **CacheRL (Jun 2026)** three-tier fuzzy cache with cache-tier-aware rewards ([arXiv 2606.14179](https://arxiv.org/abs/2606.14179)); StableToolBench cache + LLM fallback.

### 1.4 Infrastructure standards
- **OpenEnv** (Meta-PyTorch + HF, Oct 2025): Gymnasium-style `reset()/step()/state()`, typed dataclasses, Docker, HF Spaces ([repo](https://github.com/meta-pytorch/OpenEnv), [catalog](https://huggingface.co/docs/openenv/environments)).
- **Prime Intellect Environments Hub** + `verifiers` ([blog](https://www.primeintellect.ai/blog/environments)). **AgentGym / AgentGym-RL** ([arXiv 2509.08755](https://arxiv.org/abs/2509.08755)).

## 2. Environment definitions compared

| System | State | Tool impl | Reset | Task instance | Verifier |
|---|---|---|---|---|---|
| tau2 | JSON DB | real Python | reload JSON | init fn + goal + assertions | DB assertions + actions + communicate_info |
| AppWorld | SQLite | real code | copy Base DB, frozen clock | task + expected/allowed change sets | hash DB diff, expected ⊆ diff ⊆ allowed |
| ARE/Gaia2 | app objects + clock + events | real Python | deterministic replay | initial state + event DAG + oracle writes | write-action matching, hard + soft, timing |
| MCPMark | live services seeded | real MCP | re-seed | curated state + script | expert-refined script |
| EnvScaler | Python class attrs | LLM-generated code | re-instantiate | generated state + checklist | generated boolean terminal-state functions |
| APIGen-MT | real env DB | real | env reset | blueprint | execution + policy tests + state/output match |
| Simia / DreamGym / K2 | text in context | LLM | none | seed trajectories or rubrics | LLM judge on rubric |
| CacheRL | recorded responses | cache | trivial | recorded task | hybrid reward |

## 3. Known failure modes
1. **LLM-simulated tools hallucinate and drift.** ToolEmu: only 68.8% of emulator-identified failures were valid real-world failures ([arXiv 2309.15817](https://arxiv.org/abs/2309.15817)). OccuBench: "strong agents are not necessarily strong environment simulators" ([arXiv 2604.10866](https://arxiv.org/abs/2604.10866)).
2. **User simulator noise**: tau2 error rates 47% airline, 40% retail, 16% telecom (constrained tool interface).
3. **Generated environments buggy or trivially easy**: EnvScaler discards ~28%; AutoEnv 65% validity; ARE flags 0%/100% scenarios.
4. **Leaky or hackable verifiers**: METR 30.4% reward-hacking on RE-Bench with visible scoring code vs 0.7% on HCAST ([METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/)); ImpossibleBench; Anthropic: reward hacking generalizes to misalignment ([arXiv 2511.18397](https://arxiv.org/abs/2511.18397)).
5. **Silent wrong end states**: >50% of MCPMark failures implicit; DB-diff grading catches them, output grading does not.
6. **Collateral damage** only caught with allowed-change sets (AppWorld).
7. **Cache-only replay brittle off-path** (CacheRL).
8. **Sim-to-real transfer domain-bound** (DreamGym).

## 4. How fidelity is measured today
Response-level BLEU/cosine (MirrorAPI); state-level Factuality + Turing (WebWorld); structural (ShopGym); behavioral (twin vs live success correlates per model); verifier-level agreement with human labels (ARE 98% / 0.99) and differential testing (AutoEnv). A dedicated "environment fidelity" metric for tool sandboxes is still missing from the literature.

## 5. Recommended architecture: environment from a customer's traces

**Target:** OpenEnv-compatible, Docker-packaged, EnvScaler-style code environment (real Python tool implementations over an explicit state store), AppWorld-style DB-diff grading, seeded from traces, validated by replay. Do not ship an LLM-as-environment for grading; LLM simulation only as a labelled fallback tier.

**Inferred automatically from traces**
1. Tool inventory and schemas from system prompt and tool definitions; classify read vs write by whether later reads change after the call.
2. Entity and state schema by mining tool-result payloads: recurring keys, ID formats, foreign keys (IDs appearing in one tool's result and another's arguments), enums, timestamps. Emit a typed relational schema (SQLite).
3. Tool semantics as generated code constrained by observed (args, result, later state) triples; each write must reproduce every recorded post-state; each read must return recorded results from reconstructed state; reject implementations failing any recorded call.
4. Initial state per task by inverse replay: union of everything observed, then apply inverse operations of recorded writes so forward replay reproduces recorded results.
5. Error and edge behavior from recorded error results; failure responses are highly templated.
6. Task text from the first user turn; candidate verifier = DB diff between reconstructed initial and recorded final state, split into expected and allowed (AppWorld), plus communicate_info.
7. Compositional expansion tau2-style; drop composites where a weak model scores 100% or a strong model 0%.
8. Simulated user only when the trace shows multi-turn input; constrain with a tool interface over its own state; best-of-N self-critique.

**Needs the customer**
1. Unobserved branches (reads never performed, writes never issued): API docs / OpenAPI, or a staging endpoint.
2. Invariants and policies (refund limits, auth rules) as unit tests on state; customer confirms.
3. 20-50 labeled trajectories to score the verifier (ARE-style agreement); <~95% agreement should not gate anything.
4. Allowed-change set per task (side effects tolerated).
5. Anonymization sign-off; strip PII before code generation.

**Fidelity gates (all four per environment)**
1. Replay fidelity: replay each source trace verbatim; 100% exact match on read results and post-write state per tool; report per-tool coverage.
2. Off-path fidelity: frontier model in synthesized env vs customer staging on the same tasks; compare responses (BLEU/cosine, success and error separately) and state deltas.
3. Behavioral fidelity: per-model success on twin vs live correlates; near-100% everywhere means too easy.
4. Verifier robustness: differential testing, conflicting-spec probes, hidden verifier code, "no mutation outside the tool API" check.

**Opinionated choices**: real code over LLM simulation wherever state matters; three-tier fallback (exact cache, generated code, LLM simulator with schema-conforming output) with the answering tier recorded; grade on end state plus communicated facts; freeze the clock; regenerate tasks compositionally rather than reversing tool sequences; budget $1-5 LLM cost per environment; expect to discard 20-35% of candidates.
