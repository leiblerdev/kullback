# Step-level agent evaluation and the statistics of "model B matches model A"

Research sweep, 2026-08-26. Source: web research agent. Topic 6 of 6.

## 1. Process reward models for agents and step-level labels

**How step labels are made.** Math-Shepherd labels a step by rolling out completions and estimating the probability of reaching the correct answer ([arXiv 2312.08935](https://www.emergentmind.com/papers/2312.08935)). Qwen "Lessons of Developing PRMs": MC labels underperform because completion models reach correct answers through wrong steps. ProcessBench F1: MC labels 40.1, LLM-judge labels 46.5, human PRM800K labels 56.5. Fix: consensus filtering, discarding ~60% of MC data ([arXiv 2501.07301](https://arxiv.org/html/2501.07301)). ProcessBench: most trained PRMs underperform prompted critic models ([arXiv 2412.06559](https://arxiv.org/abs/2412.06559)).

**Transfer to agents.** AgentPRM (Cornell, 2025): step-level Q from MC rollouts; 3B Llama 88.1% on ALFWorld vs GPT-4o 65.7%; with too few rollouts the PRM is reward-hacked ([arXiv 2502.10325](https://arxiv.org/html/2502.10325)). AgentPRM (Fudan, 2025): "promise and progress", TD plus GAE ([arXiv 2511.08325](https://arxiv.org/abs/2511.08325)). Web-Shepherd: step PRM beats GPT-4o as judge by ~30 points on WebRewardBench ([arXiv 2505.15277](https://arxiv.org/abs/2505.15277)). Agent Q ([arXiv 2408.07199](https://arxiv.org/abs/2408.07199)). SPA-RL ([arXiv 2505.20732](https://arxiv.org/abs/2505.20732)).

**Tool-use transfer.** ToolPRMBench (Jan 2026): math PRMs and Web-Shepherd score near chance (~50%) on tool-use step cases, frontier LLM judges 73-75% (GPT-5 74.4, Claude 4.5 Haiku 75.1). "PRMs trained for math reasoning or web navigation do not directly transfer to tool-using process evaluation" ([arXiv 2601.12294](https://arxiv.org/html/2601.12294)). Bottom line: a prompted frontier judge is the safer step grader for tool agents, and it is still only ~75% accurate pairwise.

## 2. Teacher-forced step evaluation vs end-to-end success

**Offline-online gap.** Mind2Web step success rate against reference actions; known failure modes include picking a functionally identical element ([review](https://liner.com/review/mind2web-towards-a-generalist-agent-for-the-web)). WebCanvas replaces static step matching with "key nodes" that must be hit regardless of path; best agent 23.1% task success vs 48.8% key-node completion ([arXiv 2406.12373](https://arxiv.org/abs/2406.12373)). Reference-based action matching is conservative because it "assumes a single correct action sequence" ([OTAP arXiv 2607.17082](https://arxiv.org/html/2607.17082); [TRACE arXiv 2510.02837](https://arxiv.org/abs/2510.02837)).

**Exposure bias and compounding errors.** "Better prediction on expert trajectories need not translate to better downstream performance once the student acts on its own induced contexts" ([Revisiting DAgger arXiv 2605.12913](https://arxiv.org/abs/2605.12913)). "Prefix trap": conditioning on teacher prefixes is only informative for early, low-shift prefixes ([arXiv 2607.04763](https://arxiv.org/abs/2607.04763)). Expert reference trajectories contain failures ([EEF arXiv 2504.13145](https://arxiv.org/abs/2504.13145)).

**Which steps matter.** Verified Critical Step Optimization: a critical step is one where swapping the action flips the outcome, verified by branch rollouts; only ~16% of steps need supervision ([arXiv 2602.03412](https://arxiv.org/abs/2602.03412)). **The Intervention Paradox**: a critic with AUROC 0.94 at predicting failure still degraded agents by 0-26 points when used to intervene, because it disrupted trajectories that would have recovered; all regressions came from early-step disruptions ([arXiv 2602.03338](https://arxiv.org/html/2602.03338)). 58.8% of transition failures recovered ([arXiv 2602.17037](https://arxiv.org/pdf/2602.17037)). AgentEval: step-level grading catches 2.17x more failures than end-to-end checks (recall 0.89 vs 0.41), kappa 0.84 ([arXiv 2604.23581](https://arxiv.org/abs/2604.23581)). TraceProbe: successful runs differ widely in wasted work ([arXiv 2607.06184](https://arxiv.org/abs/2607.06184)).

Net: step-level accuracy under teacher forcing is neither necessary nor sufficient for task success.

## 3. Grading the reasoning text

Turpin et al.: CoT explanations omit biasing features ([arXiv 2305.04388](https://arxiv.org/abs/2305.04388)). Anthropic 2025: CoTs verbalize a used hint often below 20%; reward hacking almost never verbalized ([arXiv 2505.05410](https://arxiv.org/abs/2505.05410)). Post-hoc rationalization 13% (GPT-4o-mini), 7% (Haiku 3.5), 0.04-2% frontier ([arXiv 2503.08679](https://arxiv.org/abs/2503.08679)). In agents, as training proceeds the model becomes less likely to revise its action in response to its own CoT ([arXiv 2606.26935](https://arxiv.org/abs/2606.26935)). OpenAI: penalizing "bad thoughts" produces obfuscated reward hacking ([arXiv 2503.11926](https://arxiv.org/abs/2503.11926)); multi-lab position paper on monitorability ([arXiv 2507.11473](https://arxiv.org/html/2507.11473)). No lab recommends grading reasoning text for quality as a proxy for capability.

## 4. Statistics of comparing two models on shared tasks

**Anthropic error-bars paper** (Nov 2024): report SEM and 95% CIs; cluster SEs on the randomization unit (up to 3x naive); resample K answers per question (K=2 cuts variance one third, K=4 halves it); paired differences because per-question scores correlate 0.3-0.7; power formula gives ~969 questions to detect a 3-point gap at 80% power ([blog](https://www.anthropic.com/research/statistical-approach-to-model-evals); [arXiv 2411.00640](https://arxiv.org/html/2411.00640)). evalci packages this; 3 of 8 adjacent MMLU gaps vanish after correction ([arXiv 2607.04429](https://arxiv.org/abs/2607.04429)). "Resolution Diagnostics for Paired LLM Evaluation": the unpaired-times-(1-rho) shortcut is off by ~2x ([arXiv 2605.30315](https://arxiv.org/abs/2605.30315)).

**Reliability, not just mean.** tau-bench pass^k drops below 25% at k=8 for GPT-4o retail despite >60% pass@1. "Beyond pass@1": capability and reliability rankings invert at long horizons ([arXiv 2603.29231](https://arxiv.org/abs/2603.29231)). HAL: 21,730 rollouts, ~$40k ([arXiv 2510.11977](https://arxiv.org/pdf/2510.11977)).

**Noise sources.** Accuracy swings up to 15% across 10 "deterministic" runs ([arXiv 2408.04667](https://arxiv.org/abs/2408.04667)); BF16 adds up to 9% std on AIME ([arXiv 2506.09501](https://arxiv.org/html/2506.09501v2)). Prompt paraphrases change rankings, though much of that shrinks under semantic grading ([arXiv 2509.01790](https://arxiv.org/abs/2509.01790)). Contamination: o3-mini 76% on SWE-bench Verified file identification without needed context ([arXiv 2506.12286](https://arxiv.org/abs/2506.12286)).

**Non-inferiority.** TOST / non-inferiority (null: candidate is worse by more than margin delta; one-sided test) is standard in clinical statistics and used in ML deployment gating ([Suitability Filter arXiv 2505.22356](https://arxiv.org/abs/2505.22356)). No agent-eval paper adopts it explicitly; the Anthropic paired-difference machinery is the natural plug-in.

## 5. Using the frontier model's trajectory as reference

Reference-guided judging improves agreement with humans and curbs self-preference ([arXiv 2408.09235](https://arxiv.org/html/2408.09235)); reference-free judges over-credit wrong answers ([arXiv 2607.12885](https://arxiv.org/html/2607.12885)). But when the reference is wrong, "grading reliability drops sharply under swapped references" ([arXiv 2601.07506](https://arxiv.org/pdf/2601.07506)). Expert trajectories contain failures; a frontier model's own actions can be reward-hacked and unverbalized. So the frontier trajectory is a floor (one known-valid path) not a ceiling, and any step disagreement is "different," not "wrong," until a rollout or checklist says otherwise.

## Recommended design

**Grading unit: outcome first, steps as diagnostics, never step-match as the score.**
1. Primary metric per task: verifiable outcome (state check, tests, checklist of required end-conditions), graded by rules plus a checklist judge, not by matching the frontier trajectory. Run the cheaper model on its own prefixes (online).
2. Secondary, blocking: WebCanvas-style key nodes (2-4 milestones any valid path must hit) and hard-constraint violations (destructive action, policy breach, hallucinated tool result). Derive key nodes from the frontier trajectory but validate each with at least one alternative path.
3. Per-step signal for debugging: branch rollouts, not a judge. At a disagreement step, continue the cheaper model from the frontier action and vice versa, k=3-5; a step is "critical" only if outcomes flip. Expect 15-20% of steps to matter.
4. Step judge, when used: prompted frontier model with tool metadata and checklist, pairwise; ~75% agreement is the ceiling.

**Reasoning text: do not grade it for quality.** Grade only (a) hard violations visible in reasoning (intent to bypass a check, fabricated tool output), as a monitor, and (b) whether the stated plan matches the taken action, as a consistency flag.

**Statistical protocol for "cheaper model clears the bar on cluster C".**
1. Pre-register a non-inferiority margin delta per cluster (3-5 points absolute on outcome pass rate; zero tolerance for hard-constraint violations).
2. Both models run the same task set, K=4 seeds per task at production temperature; 2-3 prompt paraphrases on a subset.
3. Per-task paired difference d_i = mean_k(B) - mean_k(A). Cluster SEs by task family. Paired bootstrap or permutation over tasks.
4. Declare non-inferior if the lower bound of the one-sided 95% CI on mean(d) exceeds -delta. Report the CI.
5. Sample size: delta=5 points, 80% power needs ~150-250 independent tasks per cluster; delta=3 needs 400-700. With 20-40 tasks per cluster you can only resolve 10-15 point margins. Pool small clusters with a hierarchical model and report shrinkage estimates.
6. Reliability gate: require pass^4 of B within delta_r of A.
7. Multiple clusters: Holm correction if the claim is "B clears every cluster"; none if each cluster is an independent decision.
8. Hold out 20% of tasks never used during iteration; refresh tasks periodically and check the frontier reference still passes them.
9. Log cost and latency with the same CIs; the claim is "non-inferior at X% of cost."

One-line summary: score outcomes online with checklists and key nodes, use branch rollouts (not judges) to find the few steps that matter, ignore reasoning quality except as a safety monitor, and declare parity only with a pre-registered margin, paired clustered CIs, K=4 seeds, and a pass^k reliability gate, which realistically needs 150+ tasks per cluster for a 5 point margin.
