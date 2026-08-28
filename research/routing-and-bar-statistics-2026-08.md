# LLM model routing and cost optimization: literature sweep

Compiled 2026-08-26. Scope: deciding which model handles which request or task so that quality stays at a fixed bar while cost drops. Sections 1 to 5 map to the brief; the final section is a recommended statistical protocol.

Notation used throughout:
- "Strong" = frontier or expensive model, "weak" = cheap candidate.
- PGR = performance gap recovered = (router_perf - weak_perf) / (strong_perf - weak_perf).
- CPT(x%) = fraction of calls to the strong model needed to recover x% of the gap.
- AIQ = area under the normalized cost-quality curve (RouterBench).
- pass^k = probability all k independent trials of a task succeed, averaged over tasks (tau-bench).

---

## 1. Routing and cascade systems

### FrugalGPT (Chen, Zaharia, Zou, 2023)
- Year: 2023 (arXiv 2305.05176).
- Method: three levers: prompt adaptation, LLM approximation (caching, fine-tuning), and LLM cascade. The cascade queries models in a learned order; a DistilBERT regression "scorer" predicts reliability of each answer; if score exceeds that tier's threshold the answer is returned, otherwise escalate. Model order and thresholds are found by a constrained optimization: maximize accuracy subject to average cost <= budget (mixed-integer, solved with pruning and interpolation).
- Quality bar: task accuracy on labeled sets (HEADLINES, OVERRULING, COQA) must match or exceed GPT-4 accuracy at the given budget.
- Results: HEADLINES 98.3% cost reduction at GPT-4 parity; OVERRULING 73.3% saving with +1% accuracy; COQA 59.2% saving. Learned cascade on HEADLINES: GPT-J (threshold 0.96) then J1-L (0.37) then GPT-4.
- Limitations: needs labeled training data from the same distribution; thresholds are task-specific; no guarantee per query; the scorer itself must be trained per task.
- URL: https://arxiv.org/abs/2305.05176

### Large Language Model Cascades with Mixture of Thought Representations (Yue et al, ICLR 2024)
- Year: 2023/2024 (arXiv 2310.03094).
- Method: weak model (GPT-3.5) answers with several samples in two representations (Chain-of-Thought and Program-of-Thought). Answer consistency (vote agreement across samples, and agreement between CoT and PoT answers) is the difficulty signal; inconsistent answers escalate to GPT-4. No trained router.
- Quality bar: accuracy on six reasoning datasets (GSM8K, MATH-like, causal reasoning) comparable to GPT-4 alone.
- Results: comparable accuracy at about 40% of GPT-4-only cost.
- Limitations: consistency thresholds are tuned per dataset; sampling multiple weak answers itself costs tokens; only works where answers are discrete and comparable.
- URL: https://arxiv.org/abs/2310.03094

### AutoMix (Aggarwal, Madaan et al, NeurIPS 2024)
- Year: 2023/2024 (arXiv 2310.12963).
- Method: small model generates, then few-shot self-verifies its own answer (entailment-style prompt); because self-verification is noisy, a POMDP router uses the verification signal as a partial observation to decide whether to escalate. Metric introduced: IBC (incremental benefit per cost).
- Quality bar: task accuracy vs. cost, five datasets, five model pairs.
- Results: over 50% cost reduction at comparable performance.
- Limitations: self-verification is weakly calibrated; the POMDP needs a small labeled set to fit; gains depend on how correlated verifier errors are with model errors.
- URL: https://arxiv.org/abs/2310.12963

### Zooter (Lu et al, NAACL 2024)
- Year: 2023/2024 (arXiv 2311.08692).
- Method: use an off-the-shelf reward model to score every candidate model's outputs on training queries, distill those rewards into a lightweight routing function (query -> model), with tag-based label smoothing to reduce reward noise. Inference cost is just the router.
- Quality bar: reward-model score and benchmark accuracy across 26 subsets; compared against best single model and against reward-model reranking of all outputs.
- Results: beats best single model on average, ranks first on 44% of subsets, approaches full reranking at a fraction of cost.
- Limitations: quality is defined by a reward model, so inherits its biases; open-source model pool of similar size, no cost dimension.
- URL: https://arxiv.org/abs/2311.08692

### Routoo (Mohammadshahi et al, Leeroo, 2024)
- Year: 2024 (arXiv 2401.13979).
- Method: a lightweight LLM "performance predictor" estimates each candidate's expected score on a prompt without running it; a cost-aware selector picks the cheapest model whose predicted score satisfies the constraint.
- Quality bar: MMLU-style accuracy relative to a reference model.
- Results: matches Mixtral 8x7B at one third lower cost; with GPT-4 in the pool, nearly matches GPT-4 at half the cost, exceeds it at 25% less cost.
- Limitations: predictor trained on benchmark data; transfer to production prompts unverified.
- URL: https://arxiv.org/abs/2401.13979

### RouterBench (Hu et al, Martian, ICML 2024)
- Year: 2024 (arXiv 2403.12031).
- Method: benchmark of 405k precomputed outputs from 11 LLMs on 8 datasets (MMLU, MT-Bench, MBPP, HellaSwag, Winogrande, GSM8K, ARC, RAG). Defines the Zero router (probabilistic mixture of models on the non-decreasing convex hull of single-model cost/quality points), the Oracle router (best model per query, cheapest on ties), and AIQ (area under the normalized cost-quality curve).
- Quality bar: per-dataset accuracy; a router is judged by its whole cost-quality curve, not a point.
- Results: KNN and MLP predictive routers reach best-single-model quality at similar or lower cost, but none significantly beat the Zero router. Cascading routers help only when the judge error rate is below about 0.1 and degrade rapidly above about 0.2.
- Limitations: no latency; benchmark-only prompts; oracle assumes at least one model is correct.
- URL: https://arxiv.org/abs/2403.12031

### Hybrid LLM (Ding et al, Microsoft, ICLR 2024)
- Year: 2024 (arXiv 2404.14618).
- Method: train a DeBERTa router to predict whether the small model's answer is "good enough". Quality gap H(x) = q(S(x)) - q(L(x)) with q = BART score against a reference; because generation is stochastic, sample 10 responses per model per query and use soft labels Pr[H(x) >= 0] (probabilistic router) or Pr[H(x) >= -t] (transformed router, with slack t chosen by grid search to sharpen the label distribution). At test time route to the small model when predicted probability exceeds a threshold that is tuned to a target quality drop.
- Quality bar: average BART score of the routed system must be within a stated tolerance (0% or 1%) of the large model. Threshold picked on 500 validation queries by grid search; it transferred to test.
- Results: about 20% fewer large-model calls at zero quality drop for close model pairs; 20 to 22% at <= 1% drop for Llama-2 13B vs GPT-3.5; 40% at about 10% drop for wide gaps.
- Limitations: pairwise only; no task labels; depends on BART score being a faithful quality metric; OOD generalization untested.
- URL: https://arxiv.org/abs/2404.14618

### Language Model Cascades: Token-level Uncertainty and Beyond (Gupta et al, Google, 2024)
- Year: 2024 (arXiv 2404.10136).
- Method: for generative tasks, simple aggregates of token log-probs are poor deferral signals; learn a post-hoc deferral rule from token-level uncertainty quantiles plus embeddings. FLAN-T5 pairs.
- Quality bar: task accuracy vs. deferral rate curves.
- Results: learned deferral beats mean or min log-prob aggregation across NLP benchmarks.
- Limitations: white-box (needs token probabilities); small model pairs.
- URL: https://arxiv.org/abs/2404.10136

### C2MAB-V (Dai et al, 2024)
- Year: 2024 (arXiv 2405.16587).
- Method: online combinatorial multi-armed bandit that picks a set of LLMs per query under a cost budget, with three reward models for different collaboration types (AWC, AEC, SSC). Relaxes the NP-hard selection to continuous space, rounds on the cloud, updates online. Regret and constraint-violation bounds.
- Quality bar: bandit reward (accuracy, efficiency, similarity) per task type.
- Results: nine LLMs, three scenarios; beats fixed selection baselines in reward per cost.
- Limitations: task type must be known a priori; feedback needed online.
- URL: https://arxiv.org/abs/2405.16587

### RouteLLM (Ong et al, LMSYS/Berkeley, 2024)
- Year: 2024 (arXiv 2406.18665).
- Method: binary strong/weak routing trained on Chatbot Arena human preference pairs. Four routers: similarity-weighted Bradley-Terry ranking, matrix factorization, BERT classifier, Llama-3-8B causal classifier. Data augmentation with (a) golden-label MMLU comparisons (~1.5k) and (b) ~120k Nectar prompts labeled by GPT-4 as judge (~$700). Router outputs P(strong wins | q); route to strong if P >= alpha.
- Quality bar: PGR and CPT. The user picks a target PGR (they report CPT(50%) and CPT(80%)) and sets alpha on a validation split to hit it. Quality metrics: MT-Bench GPT-4 judge score, MMLU accuracy, GSM8K accuracy.
- Results: MT-Bench 3.66x cheaper at 95% of GPT-4 quality, 2.49x at 80%; MF router CPT(50%) = 13.4% strong calls; MMLU harder (CPT(50%) about 35%); GSM8K CPT(50%) about 34%. Routers transfer when the model pair is swapped.
- Limitations: benchmarks differ from production traffic; binary only; router-to-router variance on the same data is unexplained; latency and the cost of the router itself.
- URL: https://arxiv.org/abs/2406.18665 and https://www.lmsys.org/blog/2024-07-01-routellm/

### GraphRouter (Feng et al, UIUC, ICLR 2025)
- Year: 2024 (arXiv 2410.03834).
- Method: heterogeneous graph with task nodes (GPT-4o-written task descriptions embedded with BERT), query nodes, and LLM nodes; edge prediction estimates effect and cost of each (query, LLM) pair. Reward = alpha*performance - beta*cost, with three settings (performance-first, balanced, cost-first). New LLMs added via few-shot without retraining.
- Quality bar: per-task metric (F1 or accuracy) on Alpaca, GSM8K, SQuAD, Multi-News (600 samples each), reported as fraction of oracle reward.
- Results: at least 12.3% higher reward than FrugalGPT and C2MAB-V; reaches about 89% of optimal; 9.5% better effect on unseen LLMs.
- Limitations: exploratory; task node requires an explicit task label per query (this is a task-level routing design).
- URL: https://arxiv.org/abs/2410.03834

### Not Diamond (commercial, 2024 to 2026)
- Year: 2024 onward.
- Method: custom router trained on the user's own evaluation data: CSV of prompt, per-model response, per-model score (any metric). Minimum 15 samples, up to 10k per job. Learns a "meta-model" predicting which candidate scores best. Not Diamond Code (2026) routes coding agents per turn: predicts future reward and cost for each (model, reasoning effort) using session state, token counts, task complexity and KV-cache warmth, and optimizes for full-session outcome.
- Quality bar: Pareto plots of benchmark score vs. cost; per-turn reward prediction. No published CIs or hypothesis tests.
- Results (vendor reported): Poly-SWE-Bench Verified 39% cost reduction at approximately Opus 4.8 Xhigh quality; LongCodeQA 61% reduction; cross-provider routing 66% savings with +3.6% score; enterprise SRE benchmark +39% accuracy.
- Limitations: proprietary; RouterArena found NotDiamond accurate but expensive relative to academic routers.
- URL: https://www.notdiamond.ai/blog/not-diamond-code-intelligent-model-routing-for-coding-agents and https://docs.notdiamond.ai/docs/router-training-quickstart

### Universal Model Routing / UniRoute (Jitkrittum et al, Google, 2025)
- Year: 2025 (arXiv 2502.08773).
- Method: each LLM represented as a feature vector of its accuracy on a set of representative prompts (or on K prompt clusters). Route by comparing the query's cluster to the LLM's per-cluster error profile, minus a cost penalty. Two instantiations: cluster-based (k-means on prompt embeddings) and a learned cluster map. Excess-risk bound. Handles new, unseen LLMs at test time without retraining.
- Quality bar: accuracy under a cost weight; deferral curves.
- Results: routes among 30+ unseen LLMs, beats zero-shot and single-model baselines.
- Limitations: needs a representative prompt set labeled for every new LLM; clusters must be stable across time.
- URL: https://arxiv.org/abs/2502.08773

### MixLLM (Wang et al, NAACL 2025)
- Year: 2025 (arXiv 2502.18482).
- Method: contextual bandit routing. Query embeddings enhanced with InsTag tags manually grouped into coarse domains; per-LLM random-forest regressors predict quality and cost; a meta decision maker trades quality, cost and latency (exponential penalty past a waiting-time threshold); online policy-gradient updates from binary user feedback.
- Quality bar: percentage of GPT-4 quality retained (judge score) at a cost fraction.
- Results: 97.25% of GPT-4 quality at 24.18% of cost; with Llama 3.1 in the pool, 98.55% at 16.79%.
- Limitations: 5.44% drop on out-of-domain queries; needs feedback data.
- URL: https://arxiv.org/abs/2502.18482

### Arch-Router (Katanemo, 2025)
- Year: 2025 (arXiv 2506.16655).
- Method: 1.5B generative router that maps a query to a user-written Domain-Action taxonomy ("legal / summarization"); the developer assigns a model per route. Policies are in-context so new routes need no retraining.
- Quality bar: routing accuracy against human-labeled domain/action (turn, span, conversation level), not answer correctness. Any quality of the downstream model is the developer's responsibility.
- Results: 93.17% routing accuracy, +7.71 points over proprietary LLM routers, 51 ms vs 1450 ms for Claude Sonnet 3.7 as router. Datasets: CLINC-150, MANtIS, SGD, LMSYS-1M.
- Limitations: does not measure whether the chosen model is good enough; relies on precise policy descriptions.
- URL: https://arxiv.org/abs/2506.16655

### Router-R1 (UIUC, NeurIPS 2025) and R2-Router (2026)
- Router-R1: the router is itself an LLM trained with RL to interleave "think" and "route" actions over multiple rounds, rewarded on format, final outcome and a cost term; conditions only on model descriptors (price, latency, example performance) so it generalizes to unseen pools. R2-Router (arXiv 2602.02823) jointly selects model and an output-length budget, finding that a strong model with a short budget often beats a weak model at equal cost; reports 4 to 5x lower cost at state-of-the-art accuracy.
- URLs: https://arxiv.org/abs/2506.09033 , https://arxiv.org/abs/2602.02823

### Avengers-Pro (Zhang et al, DAI 2025)
- Year: 2025 (arXiv 2508.12631).
- Method: task-level routing by clustering. Embed queries with Qwen3-embedding-8B (4096-d), k-means with k = 60; per cluster compute each model's min-max-normalized accuracy p and cost q; score x = alpha*p + (1-alpha)*(1-q); route each new query to its nearest cluster's top-scoring model. 70/30 split: 70% to fit clusters and per-cluster stats, 30% held out.
- Quality bar: average accuracy across six benchmarks relative to GPT-5-medium.
- Results: +7% accuracy at equal cost; equal accuracy at 27% lower cost; 90% of accuracy at 63% lower cost. LLMRouterBench later found it sits on the Pareto frontier (near-zero ParetoDist).
- Limitations: 2,603 total queries; per-cluster stats are noisy when clusters are small; must re-estimate when models change.
- URL: https://arxiv.org/abs/2508.12631

### LLM Routing with Dueling Feedback (2025)
- Year: 2025 (arXiv 2510.00841).
- Method: contextual dueling bandit that learns from pairwise preference feedback instead of absolute scores; Category-Calibrated Fine-Tuning builds model embeddings from offline data; Feel-Good Thompson Sampling. Evaluated on RouterBench and MixInstruct.
- Relevance: shows routing can be learned from win/loss labels, which are what LLM judges produce most reliably.
- URL: https://arxiv.org/abs/2510.00841

### RouterArena (2025)
- Year: 2025 (arXiv 2510.00202).
- Method: black-box benchmark, 8,400 queries, 7 categories, 5 dimensions: accuracy, cost, optimality, robustness (paraphrase noise), latency. Optimality = choosing the cheapest model that is still correct (optimal selection ratio, optimal accuracy ratio, optimal cost ratio).
- Findings: GPT-5-as-router and NotDiamond are accurate but expensive; CARROT and GraphRouter are cheaper at competitive accuracy; every router is far from oracle "primarily because they are inefficient at recognizing when smaller, cheaper models are sufficient". No router wins on all metrics.
- URL: https://arxiv.org/abs/2510.00202

### LLMRouterBench (2026)
- Year: 2026 (arXiv 2601.07206).
- Method: 400k+ instances, 21 datasets, two pools (20 small open models; 13 flagship models). Metrics: Gain@Random, Gain@BestSingle, Gap@Oracle, PerfGain, CostSave, ParetoDist.
- Findings: leading academic routers (EmbedLLM, GraphRouter, MODEL-SAT, Avengers) are nearly indistinguishable; several routers fail to beat the best single model (OpenRouter auto-router at -24.7%); best routers get about +4% accuracy with 31.7% cost saving. Gap to oracle is driven by "model-recall failures": on queries only 1 to 3 models solve, routers pick a correct model only 23 to 24% of the time. Embedding choice barely matters; larger pools show diminishing returns.
- URL: https://arxiv.org/abs/2601.07206

### Unsolvability ceiling, routing plateau, and "how much of the gap is real" (2026)
- Unsolvability Ceiling (arXiv 2605.07395): oracle upper bounds ignore queries no model solves; router quality should be measured against the achievable ceiling.
- The Routing Plateau (arXiv 2606.07587): routers stall because of label noise, query ambiguity (several models tie) and inter-model correlation.
- How Much of the Routing Gap Is Real? (arXiv 2607.03436): re-labels RouterBench-style data with k >= 20 draws per (query, model) and decomposes router-to-oracle gap into reproducible specialist advantage plus single-draw noise. Noise share: GSM8K 12% of a 3.3 point gap, MATH-500 36% of 10.1 points, GPQA 13% of 42.8 points; noise reaches 43% on thin-support competition math. Correlated pools behave like 2 to 3 independent models. Recommends k >= 20 draws per cell, reporting gaps against both expected oracle and reproducible ceiling, and effective pool size.
- From Sampled Outcomes to Capability Distributions (arXiv 2606.06924): train routers on per-model success probabilities, not single binary outcomes; improves generalization on MMLU, DROP, GPQA.
- URLs: https://arxiv.org/abs/2605.07395 , https://arxiv.org/abs/2606.07587 , https://arxiv.org/abs/2607.03436 , https://arxiv.org/abs/2606.06924

### Calibrated and conformal cascades (2025 to 2026)
- UCCI (arXiv 2605.18796): uncertainty = 1 - mean token margin (top1 - top2 prob); isotonic regression on a 22,500-example calibration set (30% of 75k) maps it to calibrated error probability (ECE 0.12 -> 0.03); threshold chosen on a 15k validation split by minimizing cost subject to an accuracy target. 4B vs 12B on NER: 31% cost saving (95% CI 27 to 35%) at micro-F1 0.91; beats entropy thresholding (+11% cost), conformal (+5%), FrugalGPT-style (+8%). Assumes large-model accuracy is invariant to which queries escalate (0.004 gap observed).
- Conformal Cascade (arXiv 2607.25018): split-conformal prediction sets per tier; accept when the set has exactly one answer, else defer. Marginal finite-sample guarantee 1 - K*alpha (worst case) or 1 - alpha under selection preservation; recommends n >= 200 calibration examples per tier; wins on 49 of 72 model-benchmark pairs; commits about 96% of queries to tier 1 on easy benchmarks and 0% on hard ones. Multiple-choice only; breaks under distribution shift.
- RouteNLP (arXiv 2604.23577): conformal risk control with alpha = 0.05, 500 calibration examples per task and tier (250 gave 5.8% violations, 1,000 gave 3.9%); 40 to 85% cost reduction at 96 to 100% retained quality on structured tasks; 8-week pilot 58% cost drop at 91% quality acceptance; violations rose to 8.1% under domain shift. Closed loop: cluster escalation failures and distill them into the cheap model (cost ratio 0.203 -> 0.159 vs 0.184 for random distillation).
- Cost-Saving Cascades with Early Abstention (arXiv 2502.09054): adds an abstain threshold below the deferral threshold; 13% cost and 5% error reduction on average across six benchmarks at 4.1% more abstentions.
- URLs: https://arxiv.org/abs/2605.18796 , https://arxiv.org/abs/2607.25018 , https://arxiv.org/abs/2604.23577 , https://arxiv.org/abs/2502.09054

### Survey
- Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey (arXiv 2603.04445, 2026): six paradigms (difficulty-aware, preference-aligned, clustering-based, RL, uncertainty-based, cascading); metrics used across the field: routing accuracy, task metric, win rate, AUC over cost budgets, latency, throughput. Notes that verbalized confidence "consistently exhibits low alignment" with correctness, and that agentic multi-stage routing is under-covered. Open problems: generalization to new models, multi-stage cascades, multimodality.
- URL: https://arxiv.org/abs/2603.04445

---

## 2. How the quality bar is defined and measured

Summary of conventions across the works above:

| Work | Quality signal | Bar / threshold rule | Cost metric | Router calibration |
|---|---|---|---|---|
| FrugalGPT | labeled accuracy | maximize accuracy s.t. avg cost <= budget | $ per query (API list price) | scorer thresholds tuned per tier on train set |
| Hybrid LLM | BART score vs reference, 10 samples/query | <= 1% (or 0%) average quality drop, threshold on 500 validation queries | fraction of large-model calls | soft labels Pr[H >= -t] |
| RouteLLM | human preference wins; MT-Bench GPT-4 judge; MMLU/GSM8K accuracy | pick alpha to hit target PGR (50%, 80%, 95%) | fraction of strong calls, converted to $ | P(strong wins) from BT / MF / BERT |
| RouterBench | per-dataset accuracy | whole cost-quality curve (AIQ), Zero-router baseline | $ per query | n/a; notes cascades need judge error < 0.1 |
| MixLLM | judge score as % of GPT-4 | fixed latency constraint, quality-cost weights | % of GPT-4 cost | confidence-weighted online updates |
| Avengers-Pro | benchmark accuracy | alpha sweeps the Pareto curve | normalized $ | per-cluster min-max stats |
| UCCI / RouteNLP / Conformal Cascade | task accuracy or BERTScore | accuracy target or risk alpha = 0.05 with finite-sample guarantee | measured $ | isotonic regression / conformal quantile |
| Not Diamond | user-supplied score per model | Pareto plots; no formal test | $ | proprietary |
| Agent Capsules | LLM judge 0 to 1 | rolling mean over 10 runs >= quality floor; escalate after 2 consecutive misses | tokens | judge MDD measured (0.030 Opus, 0.065 GPT-4o over 7 runs) |
| Compiling Agentic Workflows | 1 to 5 judge rubric, two judges | 87 to 98% of frontier baseline; Wilcoxon / Mann-Whitney, Holm-Bonferroni, bootstrap CIs (10k), n = 200 per condition | $ per conversation | n/a |

Observations:
- The most common bar is "x% of the strong model's score" (PGR 95%, or 97 to 99% of GPT-4 quality). Almost no paper before 2026 attached a confidence interval to that claim; UCCI (95% CI on savings) and the Compiling Agentic Workflows paper are the exceptions.
- Threshold choice is nearly always a grid search on a held-out split against a target drop; the conformal line of work replaces this with a quantile that carries a marginal guarantee.
- Cost is reported as fraction of strong-model calls (RouteLLM, Hybrid LLM), $ per query at list price (FrugalGPT, Switchcraft, RouterArena), or tokens. Router inference cost is usually ignored except by RouterArena and Arch-Router.
- Calibration: token-margin plus isotonic regression gave ECE 0.03 (UCCI). Verbalized confidence is unreliable (survey). Self-verification is noisy (AutoMix). RouterBench shows cascades are very sensitive to judge error rate: useful below 0.1, harmful above 0.2.
- Single-draw labels overstate the oracle: with stochastic decoding, 12 to 36% of the apparent router-to-oracle gap is noise (arXiv 2607.03436). Label each (query, model) with k >= 20 draws when the task is stochastic.

---

## 3. Task-level vs request-level routing

### Task-level designs in the routing literature
- GraphRouter: explicit task nodes with LLM-written descriptions; routing conditions on task identity.
- UniRoute: cluster-based routing; each LLM is a vector of per-cluster error rates; new query -> cluster -> cheapest LLM whose cluster error is acceptable.
- Avengers-Pro: k-means (k = 60) on Qwen3-embedding-8B vectors; per-cluster model scoreboard. This is the cleanest published "cluster then assign a model per cluster" recipe and is Pareto-optimal on LLMRouterBench.
- MixLLM: InsTag tags manually merged into domains; domain is a feature to the quality regressors.
- Arch-Router: human-defined domain/action taxonomy; the router only classifies, the model per route is a config choice.
- Agent-as-a-Router / CodeRouterBench (arXiv 2606.22902): 10 coding "dimensions" (code gen, bug fixing, refactoring, test generation, data science, agentic programming, etc.). Adding per-dimension performance statistics to the router improved routing by 15.3%; the bottleneck was "information deficit rather than reasoning failure". Cheap models won test generation and multi-language; premium models were reserved for algorithm design.
- Topaz (arXiv 2604.03527): skill-based profiling of models from benchmarks, budget-constrained assignment of subtasks, with natural-language explanations of each routing decision.
- Switchcraft (arXiv 2605.07112): per-request DistilBERT classifier over tool-calling requests; found that the "task" signal (single vs multi-turn vs parallel calls) was the main driver of which model was needed.

### Clustering pipelines for discovering task types
- Anthropic Clio (arXiv 2412.13678, Dec 2024): Claude extracts facets per conversation (task, request, language); embed facet summaries with all-mpnet-base-v2; k-means into base clusters; recursive k-means plus Claude prompting to build a hierarchy; Claude names clusters; minimum cluster size thresholds on unique accounts and conversations; audit pass. Reconstructed a synthetic ground-truth topic distribution with 94% accuracy (random 5%).
- Kura (jxnl / 567-labs, 2025): open reproduction. LLM summary of each conversation into a short intent, embed, cluster (k-means by default), MetaClusterModel with max_clusters (default 10) builds the hierarchy, JSONL checkpoints. Docs explicitly suggest using the resulting clusters to build production query classifiers.
- OpenClio (Phylliida): another open reproduction.
- PostHog LLM analytics clustering (2025/2026): traces rendered as ASCII trees, hourly sampling, GPT-4.1-nano structured summaries, embeddings, UMAP + HDBSCAN, LLM labels clusters. Lesson: structured summaries cluster better than free text.
- Dial-In LLM (arXiv 2412.09049) and Lifecycle-Aware Clustering (arXiv 2601.04388): LLM-in-the-loop intent clustering for support dialogues; compare k-means vs HDBSCAN by silhouette; UMAP to 15 dims then HDBSCAN is a recurring recipe.

### Practical signals for task identity in agent traffic (from the papers and industry posts)
- System-prompt hash or template id (strongest and cheapest signal; used implicitly by Arch-Router policies and Not Diamond session state).
- Tool-set signature (the set of tools exposed in the call): Switchcraft and AgentFloor both show tool-call shape (single, chained, conditional, parallel) predicts difficulty better than topic.
- Embedding of an LLM-written one-line intent (Clio / Kura), then k-means (k in the tens) or UMAP + HDBSCAN.
- Position in the agent loop (planning turn vs execution turn vs verification turn): TwinRouterBench and Not Diamond Code route on this.

Trade-off: task-level routing amortizes the labeling cost (one decision per cluster, many requests) and can be validated with a proper sample per cluster; request-level routing needs a calibrated per-request score and, per RouterArena and LLMRouterBench, is where routers currently fail (they cannot tell when the cheap model is sufficient). A reasonable hybrid is: route by cluster, and inside a cluster use a cheap request-level fallback signal (schema check, self-consistency, token margin) to escalate the residual.

---

## 4. Statistical methodology for "is model B good enough on task T"

### Anthropic, Adding Error Bars to Evals (Miller, Nov 2024, arXiv 2411.00640)
- Treat an eval as a sample from a super-population of questions. SE = sqrt(Var(s)/n); for binary scores sqrt(p(1-p)/n).
- Clustered SEs when questions share a passage or document: on DROP the clustered SE was 3.05x the naive one.
- Resample K answers per question: conditional variance falls by 1/K; for binary uniformly-hard questions K = 2 removes one third of variance, K = 6 removes 5/9. Do not lower temperature to fake precision.
- Paired differences: SE_paired = sqrt(Var(s_A - s_B)/n); with correlation 0.5 between models the variance is about one third lower than unpaired. Always compare models on the same questions.
- Power: n = (z_{alpha/2} + z_beta)^2 (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2. Worked example: detecting a 3 point difference with 80% power at alpha 0.05 needs about 969 questions (unpaired binary near 50%).
- Recommendations: report SEs, adjust for clustering, resample, pair, report correlations and CIs, run a power analysis first.
- URL: https://arxiv.org/abs/2411.00640

### Hamel Husain and Shreya Shankar, LLM Evals FAQ (2025, updated 2026)
- Start with at least 100 traces for error analysis; stop adding categories when about 20 consecutive traces yield nothing new (saturation). 100+ fresh traces per review cycle every 2 to 4 weeks.
- Binary pass/fail labels, not Likert. Measure judge quality with TPR and TNR on a held-out human-labeled set, then correct the judge's estimated failure rate for its error rates. Track CIs on production metrics; investigate when the lower bound crosses the threshold. Cohen's kappa for inter-annotator agreement.
- Rule of thumb quoted in the same ecosystem: 100 examples at 80% pass gives a 95% CI of roughly 72 to 88.
- URL: https://hamel.dev/blog/posts/evals-faq/

### Eval set sizing (two-proportion power math)
- Unpaired, baseline 80%, alpha 0.05, power 0.8: 2 point delta needs about 6,300 per arm; 4 points about 1,580; 7 points about 480; 12 points about 165.
- Paired design (same questions through both models, McNemar on discordant pairs) needs roughly an order of magnitude fewer examples. Use Wilson or Agresti-Coull intervals rather than Wald near 0 or 1. Sequential testing with O'Brien-Fleming boundaries allows early stopping.
- URL: https://dev.to/gabrielanhaia/eval-set-sizing-the-statistical-power-math-behind-llm-ab-tests-4gpc

### How to Correctly Report LLM-as-a-Judge Evaluations (arXiv 2511.21140)
- Estimate judge TPR and FPR on a human-labeled subset; correct the observed pass rate: p_true = (p_obs - FPR) / (TPR - FPR). Report Agresti-Coull intervals. Judge noise both inflates required n and can bias differences (a judge that prefers long answers inflates the model that writes longer). Provide correction formulas and show underpowered studies mislead.
- URL: https://arxiv.org/abs/2511.21140

### Eugene Yan
- Judge-human agreement is 70 to 85% on well-defined tasks (factuality, format, code correctness) and often below 60% on subjective tasks, so pairwise or judge-based bars must be validated per task. Forcing a winner on near-identical outputs adds noise; allow ties.
- URL: https://eugeneyan.com/writing/product-evals/ and https://eugeneyan.com/writing/eval-process/

### Pairwise win rates and Bradley-Terry (Chatbot Arena)
- Bootstrap the BT fit (resample pairs with replacement, thousands of times) for CIs; budget at least 200 to 500 judged pairs per comparison; control position bias by swapping order; expect length and self-preference bias. Conformal Elo (arXiv 2606.13221) gives calibrated rating intervals.
- URL: https://www.lmsys.org/blog/2023-12-07-leaderboard/

### Non-inferiority testing
- The right frame for "B is not worse than A by more than a margin". Two one-sided tests (TOST) with only the lower bound is a non-inferiority test: declare B acceptable if the lower CI bound of (score_B - score_A) is above -delta. One arXiv study explicitly used non-inferiority on automated eval scores (arXiv 2409.03500). Standard references: VSNi explainer, r-statistics TOST calculator.
- URLs: https://vsni.co.uk/neither-better-nor-worse-equivalence-non-inferiority-and-non-superiority-tests-2/ , https://r-statistics.co/tools/equivalence-noninferiority-calculator.html

### pass^k and stochastic labels
- tau-bench (arXiv 2406.12045): pass^k = E_task[ p_task^k ]. GPT-4o about 61% pass^1 on retail and 35% on airline, but pass^8 falls to about 25% on retail. Reliability, not average accuracy, is the bar for agents.
- Routing gap paper (arXiv 2607.03436): with T > 0, label each (query, model) with k >= 20 draws (k >= 30 in sparse strata) to separate reproducible capability from sampling noise.
- Agent Capsules: a point-estimate gate flips on judge noise; gate on a rolling mean over 10 runs and require 2 consecutive misses before escalating.

---

## 5. Small models inside agent loops: where they fail

### tau-bench (Yao et al, Sierra, 2024)
- Function-calling agents on retail and airline domains with simulated users, graded on final database state. GPT-4o pass^1 about 61% retail, 35% airline; pass^8 about 25% retail. Failures: wrong action selection and wrong arguments dominate; complex reasoning over the database, following ad-hoc policy, and compound requests are the hardest. Native function calling beat ReAct and Act for every model that supports it. Small and open models (GPT-3.5, Mistral, Llama 3 70B at the time) scored far lower, primarily on argument correctness and policy compliance.
- URL: https://arxiv.org/abs/2406.12045

### Insurance underwriting agents (arXiv 2602.00456)
- Even top models made at least one tool-call error in 32% of conversations despite full tool metadata.
- URL: https://arxiv.org/abs/2602.00456

### AgentFloor (May 2026, arXiv 2605.00334)
- 30 deterministic tasks in six tiers: A0 instruction following, A single tool call, B two-tool chain, C conditional branching, D multi-source synthesis with conflict recovery, E long-horizon planning under persistent constraints. 16 open models 0.27B to 32B plus GPT-5; 16,542 scored runs.
- Pass rates (4B / 8B / 14B / 32B / GPT-5): A0 92/75/88/76/80; A 76/84/92/96/98; B 72/84/84/72/82; C 20/56/16/36/51; D 36/16/4/9/42; E 0/16/4/0/10.
- Reading: single and two-step tool use is essentially solved by 8B to 14B open models (within a few points of GPT-5). The cliff is at conditional branching (tier C) and multi-source conflict recovery (tier D); long-horizon constraint tracking (tier E) is unsolved by everyone zero-shot ("no zero-shot configuration clears even the 60% bar" on C, D, E).
- Failure taxonomy: F1 hallucinated tools, F2 malformed calls, F4 step budget exhausted, F5 early resignation, F5b planning without executing, F6 wrong tool, F7 partial completion. Small models fail with F1/F2/F4 (format and resource); GPT-5 fails tier E with F5/F5b (gives up). Interventions (submission prompt, doubled budget, reasoning toggle, phase decomposition) help one model and are null or harmful on others.
- Design principle stated: small open models for the broad base of routine actions, frontier for the narrow set of deep planning and control tasks.
- URL: https://arxiv.org/abs/2605.00334

### Switchcraft (May 2026, arXiv 2605.07112)
- 157k tool-calling examples from BFCL v3, ConFETTI, Glaive v2, xLAM-60K, Hermes v1; eight models; AST correctness with five bias fixes. GPT-5.3-chat 82.29% at $0.0431/query; Qwen-3.5-9B 72.40% at $0.0021; DistilBERT router 82.94% at $0.0068 (84% cheaper, about $3,630 saved per million queries). Small open models lose on format violations (markdown-wrapped or malformed JSON), argument hallucination, and refusing to call tools. Bigger is not always better on tool use, and "cheap" models can cost more via reasoning tokens. Router weakness: multi-turn context beyond a 512-token encoder window. Oracle gap 6.45 points.
- URL: https://arxiv.org/abs/2605.07112

### BFCL v3/v4 (Patil et al, ICML 2025)
- Single-turn calls are near-saturated for strong models; multi-turn memory, dynamic decision making and long-horizon reasoning remain open. Leaderboard (July 2026) still shows large open models near the top on AST accuracy, confirming tool-call formatting is not size-gated the way planning is.
- URL: https://gorilla.cs.berkeley.edu/leaderboard.html

### Can Small Agents Collaborate to Beat a Single LLM? (arXiv 2601.11327, 2026)
- Orchestrator plus web-search, code-exec and file-inspection sub-agents, Qwen3 1.7B to 32B, on GAIA, GPQA, AIME, MuSiQue, HLE. An 8B multi-agent system matched or beat a 32B single agent on most benchmarks (AIME 55% vs 45%) and was 4.2x faster. Orchestrator reasoning gave +3.7 to +36.7 points; sub-agent reasoning gave little. "Planner-limited rather than executor-limited." HLE remained unsolved for all sizes.
- URL: https://arxiv.org/abs/2601.11327

### Compiling Agentic Workflows into LLM Weights (arXiv 2605.22502, May 2026)
- Fine-tune 3B to 8B Qwen models on 2k to 6k synthetic conversations of a procedural workflow (travel booking, Zoom support, insurance claims). Judge: Claude Sonnet 4.5 (validated with GPT-4.1), five criteria 1 to 5, n = 200 scenarios per condition, Wilcoxon signed-rank, Holm-Bonferroni, Cohen's d with 10k bootstrap CIs. 87 to 98% of the frontier in-context baseline, 128 to 462x cheaper per conversation ($0.0003 to $0.001 vs $0.10 to $0.33). Gaps remain in information accuracy (world knowledge), not procedure following. Low-rank fine-tuning failed; full fine-tuning was required.
- URL: https://arxiv.org/abs/2605.22502

### Agent Capsules (arXiv 2605.00410, May 2026)
- Runtime that merges agents into fewer calls and gates every mode switch on rolling-mean judge quality (window 10, escalate after 2 consecutive below-floor readings, de-escalate after 5 above). 51% fewer input tokens vs a LangGraph 14-agent pipeline at +0.02 quality; 19 to 68% fewer tokens vs DSPy variants. Naive merging silently loses tools and compresses prompts.
- URL: https://arxiv.org/abs/2605.00410

### Not Diamond Code, TwinRouterBench, Learning Agent Routing From Early Experience (2026)
- Not Diamond Code: per-turn routing among Haiku 4.5, Sonnet 4.6, Opus 4.8 at several reasoning efforts; 39% cost cut on Poly-SWE-Bench Verified and 61% on LongCodeQA at approximately Opus quality; cache-aware.
- TwinRouterBench (arXiv 2605.18859): static plus live evaluation of per-step agentic routing; early reasoning and retrieval steps tolerate cheap models, planning and verification steps benefit most from strong models; 30 to 50% savings at comparable accuracy.
- Learning Agent Routing From Early Experience (arXiv 2605.07180): watch the first steps of an agent trajectory, then decide whether to continue on the cheap model or switch; evaluated on AIME 2025 and other reasoning sets.
- URLs above.

### Industry posts (Anthropic, MindStudio, Caylent, others, 2025 to 2026)
- Anthropic positions Haiku 4.5 (73.3% SWE-bench Verified vs Sonnet 4.5 77.2%, about one third the price) for "plan (heavy) -> execute (many Haiku) -> verify (tests) -> review (heavy)". MindStudio claims 5 to 10x token cost reduction with Opus planning and Haiku executing (2k to 3k Opus tokens vs 15k to 25k Haiku tokens per refactor) but reports no measured quality numbers; the only failure mode named is cheap models on tasks needing architectural judgment or ambiguity resolution, caught by the orchestrator's review. Cascade posts report 50 to 70% of classification-heavy traffic settling on the cheap tier when gated by schema and guardrail checks.
- URLs: https://www.anthropic.com/news/claude-haiku-4-5 , https://www.mindstudio.ai/blog/smart-orchestrator-cheaper-sub-agent-models-claude-code

### Consolidated picture of where small models break in agent loops
1. Tool selection and single-call argument formatting: mostly fine for 8B+ open models and all commercial "mini/flash/haiku" tiers; residual failures are format (JSON wrappers), argument hallucination, and refusal to call (Switchcraft, AgentFloor A/B).
2. Conditional branching on intermediate results and recovery from conflicting sources: sharp drop for small models (AgentFloor C/D: 4 to 36% vs GPT-5 42 to 51%).
3. Long-horizon constraint tracking and multi-step planning: nobody is good zero-shot; frontier models fail by resignation, small models by budget exhaustion and hallucinated tools (AgentFloor E; tau-bench pass^8).
4. Policy compliance and compound requests in conversational agents: the main tau-bench failure class; strongly size-dependent.
5. Long-context comprehension: Not Diamond's LongCodeQA and Switchcraft's 512-token router limit both flag context length as a separate axis from reasoning.
6. Planning vs execution split: multiple independent results (small-agents paper, AgentFloor, TwinRouterBench, Anthropic guidance) converge on "planner-limited": spend on the planner and verifier, economize on executors.

---

## Synthesis: recommended protocol for declaring a cheaper model "clears the bar" on a task

The decision is a non-inferiority claim on a task cluster, made with paired, resampled, judge-corrected data. Concrete defaults below; adjust the margin to the business.

### Step 0. Define the task unit
- Cluster production traffic Clio/Kura style: LLM one-line intent summary -> embedding -> k-means (k in the 20 to 60 range for a mid-size app, as in Avengers-Pro) or UMAP(15d) + HDBSCAN (min cluster size 50). Also record system-prompt hash, tool-set signature, and loop position (plan / execute / verify). A "task" is a cluster; do not certify a model on a cluster with fewer than 200 production requests per week (the decision will not pay back the labeling).

### Step 1. Define the score and validate the judge
- Prefer a deterministic scorer (tests pass, schema valid, tool call AST match, final-state check as in tau-bench). Otherwise a binary LLM judge with a rubric; pairwise (A vs B, order-swapped, ties allowed) for open-ended text.
- Label 100 to 200 items per cluster by a human; compute judge TPR and TNR; require both >= 0.85 (Yan's 70 to 85% agreement band means many subjective clusters will fail this and need a pairwise judge or a human panel instead). Correct pass rates with p_true = (p_obs - FPR) / (TPR - FPR) (arXiv 2511.21140). RouterBench: if judge error is above 0.1, cascades and gates degrade; above 0.2 do not gate automatically.

### Step 2. Choose the bar
- State it as a non-inferiority margin delta on the corrected score, not as "x% of the strong model". Suggested defaults: delta = 2 points for deterministic pass/fail on critical clusters (code, tool actions with side effects), 3 points for standard clusters, 5 points for low-stakes drafting. For pairwise judging: win-or-tie rate of B vs A >= 0.5 - delta with delta = 0.05.
- For agentic steps use pass^k, not pass^1: require pass^3 (or pass^5 for irreversible actions) of B within delta of A. A model that matches on pass^1 and loses 10 points on pass^5 has not cleared the bar (tau-bench).

### Step 3. Sample size
- Paired design on identical prompts, K = 3 samples per prompt per model at production temperature (Miller: K = 2 already removes one third of within-question variance; the routing-gap paper wants k >= 20 for oracle labeling, which is only needed when building a per-query router, not for a cluster-level decision).
- Power calculation (Miller formula, paired, binary). Rules of thumb with baseline pass rate around 80%, alpha = 0.05 one-sided, power 0.8:
  - delta = 5 points: about 250 paired prompts (unpaired would need about 900).
  - delta = 3 points: about 600 paired prompts (unpaired about 1,600, close to Miller's 969 at 50% base rate).
  - delta = 2 points: about 1,300 paired prompts (unpaired about 6,300, per the sizing table).
  - Correlation between models on the same prompts of 0.5 is what makes pairing pay; measure it on the pilot and recompute.
- Pilot with 100 prompts x 3 samples first (Husain's 100-trace minimum); if the paired difference is already below -delta at the lower 90% bound, stop early (sequential boundary), otherwise continue to the planned n.
- Cluster-robust SEs if prompts share a document or session (Miller found 3x inflation on DROP; agent sessions are clusters).

### Step 4. Decision rule
- Compute the paired difference d = score_B - score_A per prompt (averaged over K samples), its cluster-robust SE, and a 10,000-resample bootstrap 90% two-sided CI (equivalently a one-sided 95% bound). Declare B clears the bar on cluster T if the lower bound of d exceeds -delta AND the corrected absolute pass rate of B is above the cluster's floor (for example 0.90). Report both intervals and the effective sample size.
- Optional escalation gate inside the cluster: keep a cheap request-level signal (schema check, self-consistency across 2 samples, or token-margin score calibrated with isotonic regression on at least 500 labeled items, alpha = 0.05 conformal quantile; UCCI, RouteNLP). Only accept this gate if the calibration set's empirical violation rate is inside its Clopper-Pearson interval around alpha.

### Step 5. Cost accounting
- Cost per request = input + output + reasoning tokens at list price, plus router and gate overhead, plus expected escalation cost (escalation rate x strong cost). Switchcraft and R2-Router both show "cheaper per token" models can cost more via longer outputs; measure tokens, do not assume. Report savings with a bootstrap CI (UCCI reported 31% [27, 35]).

### Step 6. Monitoring after the switch
- Gate on rolling-mean quality over the last 10 to 20 judged samples with two consecutive misses required before reverting (Agent Capsules), and a weekly 100-trace review per cluster (Husain). Recalibrate any conformal or isotonic threshold when cluster composition drifts (RouteNLP saw violations rise from 5% to 8.1% under domain shift). Re-run the full non-inferiority test when either model version changes.

### What not to do
- Do not use single-draw labels to pick a model on a stochastic task (12 to 43% of the apparent gap is noise).
- Do not declare success from "97% of GPT-4 quality" without an interval; 100 items at 80% has a 16-point wide CI.
- Do not rely on verbalized confidence or unvalidated self-verification for the escalation gate.
- Do not route agent planning or verification steps to the cheap tier on the basis of execution-step results; certify plan, execute and verify positions as separate clusters.

---

## Source URLs (all items)
- FrugalGPT https://arxiv.org/abs/2305.05176
- MoT cascades https://arxiv.org/abs/2310.03094
- AutoMix https://arxiv.org/abs/2310.12963
- Zooter https://arxiv.org/abs/2311.08692
- Routoo https://arxiv.org/abs/2401.13979
- RouterBench https://arxiv.org/abs/2403.12031
- Hybrid LLM https://arxiv.org/abs/2404.14618
- LM cascades token uncertainty https://arxiv.org/abs/2404.10136
- C2MAB-V https://arxiv.org/abs/2405.16587
- tau-bench https://arxiv.org/abs/2406.12045
- RouteLLM https://arxiv.org/abs/2406.18665 , https://www.lmsys.org/blog/2024-07-01-routellm/
- GraphRouter https://arxiv.org/abs/2410.03834
- Adding Error Bars to Evals https://arxiv.org/abs/2411.00640
- Clio https://arxiv.org/abs/2412.13678 , https://www.anthropic.com/research/clio
- Dial-In LLM intent clustering https://arxiv.org/abs/2412.09049
- UniRoute https://arxiv.org/abs/2502.08773
- Early abstention cascades https://arxiv.org/abs/2502.09054
- MixLLM https://arxiv.org/abs/2502.18482
- Router-R1 https://arxiv.org/abs/2506.09033
- Arch-Router https://arxiv.org/abs/2506.16655
- Avengers-Pro https://arxiv.org/abs/2508.12631
- RouterArena https://arxiv.org/abs/2510.00202
- Dueling feedback routing https://arxiv.org/abs/2510.00841
- Reporting LLM-as-a-judge https://arxiv.org/abs/2511.21140
- Small agents vs large LLM https://arxiv.org/abs/2601.11327
- LLMRouterBench https://arxiv.org/abs/2601.07206
- R2-Router https://arxiv.org/abs/2602.02823
- Insurance underwriting agents https://arxiv.org/abs/2602.00456
- Routing survey https://arxiv.org/abs/2603.04445
- Topaz https://arxiv.org/abs/2604.03527
- RouteNLP https://arxiv.org/abs/2604.23577
- AgentFloor https://arxiv.org/abs/2605.00334
- Agent Capsules https://arxiv.org/abs/2605.00410
- Switchcraft https://arxiv.org/abs/2605.07112
- Learning Agent Routing From Early Experience https://arxiv.org/abs/2605.07180
- Unsolvability ceiling https://arxiv.org/abs/2605.07395
- Log analysis for agent evals https://arxiv.org/abs/2605.08545
- TwinRouterBench https://arxiv.org/abs/2605.18859
- UCCI https://arxiv.org/abs/2605.18796
- Compiling agentic workflows https://arxiv.org/abs/2605.22502
- Capability distributions supervision https://arxiv.org/abs/2606.06924
- Routing plateau https://arxiv.org/abs/2606.07587
- Agent-as-a-Router https://arxiv.org/abs/2606.22902
- Routing gap decomposition https://arxiv.org/abs/2607.03436
- Conformal Cascade https://arxiv.org/abs/2607.25018
- Not Diamond https://www.notdiamond.ai/blog/not-diamond-code-intelligent-model-routing-for-coding-agents , https://docs.notdiamond.ai/docs/router-training-quickstart
- Kura https://usekura.xyz/ , https://github.com/jxnl/kura
- PostHog clustering https://posthog.com/blog/llm-analytics-clustering-how-it-works
- Hamel Husain evals FAQ https://hamel.dev/blog/posts/evals-faq/
- Eugene Yan https://eugeneyan.com/writing/product-evals/
- Eval set sizing https://dev.to/gabrielanhaia/eval-set-sizing-the-statistical-power-math-behind-llm-ab-tests-4gpc
- Chatbot Arena BT bootstrap https://www.lmsys.org/blog/2023-12-07-leaderboard/
- TOST / non-inferiority https://vsni.co.uk/neither-better-nor-worse-equivalence-non-inferiority-and-non-superiority-tests-2/
- BFCL https://gorilla.cs.berkeley.edu/leaderboard.html
- Anthropic Haiku 4.5 https://www.anthropic.com/news/claude-haiku-4-5
- MindStudio orchestrator post https://www.mindstudio.ai/blog/smart-orchestrator-cheaper-sub-agent-models-claude-code
