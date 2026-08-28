# LLM model routing, cost/quality optimization, and how suitability is evaluated

Research sweep, 2026-08-26. Source: web research agent. Topic 2 of 6.

## 1. Query-level routers and cascades: the academic foundations

**RouteLLM (LMSYS, ICLR 2025).** Trains binary strong/weak routers on Chatbot Arena human preference pairs plus augmented labels. Four router types: similarity-weighted Elo, matrix factorization, BERT classifier, causal LLM classifier. At 95% of GPT-4 quality it cut cost by over 85% on MT Bench, 45% on MMLU, 35% on GSM8K, matched Martian and Unify while being over 40% cheaper ([arXiv 2406.18665](https://arxiv.org/abs/2406.18665), [LMSYS blog](https://www.lmsys.org/blog/2024-07-01-routellm/)). Evaluation method: sweep the router threshold, plot quality vs fraction of strong-model calls.

**FrugalGPT (Stanford, 2023).** Cascade: query cheap model first, a learned scoring function judges answer reliability, escalate on low score. Up to 98% cost reduction at GPT-4 accuracy ([arXiv 2305.05176](https://arxiv.org/abs/2305.05176)). Suitability is judged per answer, post hoc, by a trained verifier.

**Hybrid LLM (Microsoft, ICLR 2024).** A BERT-scale router predicts query difficulty with a tunable quality knob; up to 40% fewer large-model calls at no quality drop ([Microsoft Research](https://www.microsoft.com/en-us/research/publication/hybrid-llm-cost-efficient-and-quality-aware-query-routing/)). BEST-Route (2025) also chooses how many samples to draw ([GitHub](https://github.com/microsoft/best-route-llm)).

**MixLLM (NAACL 2025).** Contextual bandit over query embeddings; 97.25% of GPT-4 quality at 24.18% of cost ([ACL Anthology](https://aclanthology.org/2025.naacl-long.545/)).

**Arch-Router (Katanemo, 2025).** A 1.5B model maps queries to user-defined domain/action labels rather than predicting quality; the operator binds labels to models ([arXiv 2506.16655](https://arxiv.org/pdf/2506.16655)).

**Cascades with confidence (2025).** Token-level confidence fails on open-ended generation; "Semantic Agreement" uses agreement among multiple cheap-model samples as the deferral signal ([arXiv 2509.21837](https://arxiv.org/pdf/2509.21837)). The 2026 survey "Dynamic Model Routing and Cascading" concludes routers win when model capabilities differ by domain and cascades win when models are quality tiers of the same profile ([arXiv 2603.04445](https://arxiv.org/pdf/2603.04445)).

## 2. Router benchmarks and evaluation critiques

**RouterBench (Martian, 2024).** 405k inference outcomes. Oracle routing crushes any single model (MMLU oracle 0.957 at $0.297 vs GPT-4 0.828 at $4.086); but predictive KNN/MLP routers did not significantly beat a random "zero router" on most datasets, while cascades with a near-perfect judge approached oracle ([arXiv 2403.12031](https://arxiv.org/html/2403.12031)). Takeaway: the headroom is real, the prediction problem is hard, the judge is the bottleneck.

**RouterEval, RouterArena, Who Routes the Router** criticize limited task diversity and oversimplified metrics ([arXiv 2503.10657](https://arxiv.org/abs/2503.10657), [RouterArena](https://arxiv.org/html/2510.00202v1)). "Towards Fair and Comprehensive Evaluation of Routers" (2026) notes routing breaks when both models converge on the same wrong answer ([arXiv 2602.11877](https://arxiv.org/abs/2602.11877)).

**Judge artifacts.** "Unsolvability Ceiling in Multi-LLM Routing": LLM-judge and exact-match labels diverge by 13 pp on MMLU, inflating apparent router gains; recommends dual-judge validation ([arXiv 2605.07395](https://arxiv.org/pdf/2605.07395)). Known judge biases: position (~40% inconsistency for GPT-4), verbosity (~15% inflation), self-enhancement (5-7%) ([Judging the Judges](https://arxiv.org/pdf/2604.23178)).

## 3. Commercial products

- **Martian**: proprietary "model mapping"; no public method.
- **OpenRouter auto (Not Diamond)**: classifier assigns one of ~30 task types, then ranks models by community spend share for that task; no quality evaluation ([docs](https://openrouter.ai/docs/guides/routing/routers/auto-router)).
- **Not Diamond**: trains a custom router from your (prompt, candidate responses, eval score) triples ([docs](https://docs.notdiamond.ai/docs/router-training-quickstart)).
- **Unify**: neural scoring function predicts per-prompt quality, combined with live latency/cost benchmarks.
- **Requesty**: stacked rules, embeddings, classifiers; 16 ms overhead ([blog](https://www.requesty.ai/blog/agentic-routing-benchmarked)).
- **Portkey / LiteLLM**: no quality prediction; ops-signal routing ([Portkey](https://portkey.ai/docs/product/ai-gateway/load-balancing), [LiteLLM](https://docs.litellm.ai/docs/routing)).

Net: none publishes agent-step-level evaluation.

## 4. Vendor guidance on model selection

**Anthropic** recommends starting with Haiku 4.5 and upgrading on measured gaps, or starting with Opus 5 and downgrading; states that tuning the effort parameter "is often a better lever than switching models" ([Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model)). The cost-optimization guide (Jul-Aug 2026 measurements): the **advisor** pattern (cheap executor escalates to frontier advisor) pays only when the executor actually consults; at low effort the consult rate can collapse to near zero and score below the executor alone. The **orchestrator** pattern saved money only for routine work with a cost tail and corpora larger than one context window; on hard BrowseComp, Fable alone matched accuracy at 22-30% lower cost. Prescribed method: sweep effort first, then price the stronger model alone at low effort as the baseline any multi-model config must beat ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). Haiku 4.5: 73.3% SWE-bench Verified at one-third the cost of Sonnet 4 ([Anthropic](https://www.anthropic.com/news/claude-haiku-4-5)).

**OpenAI** describes nano-class models as for "classification, data extraction, ranking, and sub-agents" and "not designed for complex agentic tasks"; advises testing one reasoning level lower ([model guidance](https://developers.openai.com/api/docs/guides/latest-model)).

## 5. Agent-level routing (per step, not per task), 2025-2026

- **The Replay Gap (CMU, Aug 2026)**: static replay of logged trajectories with a substituted model "scores a state that never occurs." Branching rollouts on SWE-bench Verified: 61-94% of post-fork actions are rewritten, only 3-8% of replayed states remain valid for early swaps (85-86% for late downgrades), replay mispredicted all 5 success-critical outcomes, a constant "always fail" baseline beat the log-stitching evaluator. Even same-model FP8 replays diverged 90-96%. Recommendations: hand off downward late, escalate early, price a model's "step appetite" ([arXiv 2608.08239](https://arxiv.org/html/2608.08239)).
- **TwinRouterBench (May 2026)**: 970 router-visible step prefixes across SWE-bench, BFCL, mtRAG, QMSum; labels built by execution-verified greedy downgrade (lock earlier steps, try cheaper tier, accept if the trajectory still resolves). 71% of steps have a verified low-tier label, but SWE-bench skews high-tier. A logistic-regression router on frozen embeddings cut live SWE-bench API cost 53.1% at equal resolve rate. Prompting Opus 4.6 as a router failed. Their cost metric charges failed trajectories full cost plus a penalty ([arXiv 2605.18859](https://arxiv.org/html/2605.18859v1)).
- **StepWise (Apr 2026)**: step-level small/large routing for computer-use agents; failures concentrate in visual reasoning, multi-step planning, ambiguous instructions ([arXiv 2604.27151](https://arxiv.org/pdf/2604.27151)).
- **R2V Agent (May 2026)**: trains the SLM to ask for help at step level ([arXiv 2605.16604](https://arxiv.org/pdf/2605.16604)).
- **xRouter (Salesforce)**, **Agent-as-a-Router / CodeRouterBench** (per-task oracle 57% vs 43.83% best single), **BoundaryRouter** ([arXiv 2510.08439](https://arxiv.org/abs/2510.08439), [arXiv 2606.22902](https://arxiv.org/html/2606.22902v1), [arXiv 2605.07180](https://arxiv.org/html/2605.07180)).

## 6. Small models on tool calling

Function-calling specialists close the gap on single-turn calls but not multi-turn. xLAM-2-3b: 88.2% non-live AST but 55.6% multi-turn ([TinyLLM](https://arxiv.org/pdf/2511.22138)). Qwen-3 4B hit 96% on simple BFCL, but SLMs hallucinated argument formats "semantically correct but syntactically mismatched" ([arXiv 2608.22472](https://arxiv.org/html/2608.22472)). CONFETTI: parameter validity 72.4% (Sonnet 3.5) vs 66.1% (Haiku 3.5) ([arXiv 2506.01859](https://arxiv.org/pdf/2506.01859)). On tau2-bench, an 8B agent-tuned model reached 49.3 vs ~90 for 2026 frontier ([Simia](https://arxiv.org/pdf/2511.01824)).

## 7. Agent distillation

FireAct: 500 GPT-4 trajectories gave Llama2-7B +77% on HotpotQA ([arXiv 2310.05915](https://arxiv.org/abs/2310.05915)). KAIST "Agent Distillation" (NeurIPS 2025) ([arXiv 2505.17612](https://arxiv.org/abs/2505.17612)). SCoRe: 7B matched 72B on math/factual but only ~75% of teacher on deep-search agent tasks ([arXiv 2509.14257](https://arxiv.org/html/2509.14257v3)). NVIDIA "SLMs are the Future of Agentic AI" estimates 40-70% of agent LLM calls could move to SLMs, and proposes pipeline S1-S6: log calls, scrub PII, cluster tasks, pick SLM, LoRA fine-tune, iterate ([arXiv 2506.02153](https://arxiv.org/html/2506.02153)).

## 8. Where small models fail inside agent loops

2% per-step error yields 33% failure over 20 dependent steps ([Adaline](https://labs.adaline.ai/p/long-horizon-ai-agents-planning-ceiling)). HORIZON taxonomy: planning errors arise early and propagate; process-level failures are 72.5% of failures ([arXiv 2604.11978](https://arxiv.org/html/2604.11978v1)). Agents are most vulnerable "when failures lack explicit error signals or when recovery requires longer alternative tool-use paths" ([PlanBench-XL](https://arxiv.org/html/2606.22388v1)). Advisor consult-rate collapse is a documented "does not know it is stuck" failure.

## 9. 2026 pricing tiers (per 1M tokens, list, 2026-08-26)

Claude Fable 5 $10/$50; Opus 5 $5/$25; Sonnet 5 $2/$10; Haiku 4.5 $1/$5; GPT-5.5 $5/$30; GPT-5.4 $2.50/$15; GPT-5.4 mini $0.75/$4.50; GPT-5.4 nano $0.20/$1.25; Gemini 3 Pro $2/$12; Gemini 3.5 Flash-Lite $0.30/$2.50; DeepSeek V3.2 $0.28/$0.42; Qwen3.5 Flash $0.10/$0.40 ([BenchLM](https://benchlm.ai/llm-pricing)). Gap: frontier-to-small is ~10x, frontier-to-nano 40-50x, frontier-to-commodity-open ~100x. Anthropic says prompt caching (2.5-3.7x on agent loops) is a bigger lever than model switching for most workloads.

## 10. Evidence of per-task heterogeneity

RouterBench oracle picks cheap models often. CodeRouterBench: Opus best on average but GLM-5 beats it by 86% on algorithm design. TwinRouterBench: 71% of steps low-tier-safe overall, code repair skews high. Anthropic: delegation paid on easy BrowseComp and lost on hard.

## What this means for a tool that reads a company's agent traces and reports which task clusters a cheaper model can already handle

1. **Cluster on step role, not only on task.** The unit of savings is often the step: tool-result parsing, file reads, formatting, single tool calls, classification. Cluster by (tool called, prompt template, position in trajectory, output shape).
2. **Do not label safety by replaying logs with a cheap model.** Off-policy substitution is structurally unsound for multi-step agents (3-8% of replayed states stay valid for early swaps). Static replay is acceptable only for terminal or near-terminal steps and for stateless steps (extraction, classification, summarization of a fixed input). For everything else use TwinRouterBench's method: branch live from the logged prefix, run the cheap model forward, verify the outcome.
3. **Prefer downgrade-late, escalate-early framing.**
4. **Anchor labels on outcome checks, and run two judges.** Publish the discordance rate per cluster. Never train the router on the same judge you report with.
5. **Charge failures at full cost plus penalty.** Report "net savings at equal trajectory success," not token savings.
6. **Cascade vs router:** cascade first for reversible, verifiable steps; router for the rest.
7. **Report confidence as intervals with the label source attached:** (a) execution-verified, (b) judge-verified, (c) heuristic.
8. **Compare against effort reduction and caching before model swaps.**
9. **Watch the failure modes small models specifically show.** Flag recovery-from-error, loop detection, long dependent chains, free-form argument construction as high-risk; single-tool, schema-constrained, extraction, classification, summarization as low-risk. Measure argument validity rate separately.
10. **Account for step appetite and consult rate.** Count tokens per trajectory, not per call.
11. **Expect drift and re-verify on every model release.**
