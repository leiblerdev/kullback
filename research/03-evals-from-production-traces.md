# Building replayable eval sets from production agent traces

Research sweep, 2026-08-26. Source: web research agent. Topic 3 of 6.

## 1. The replay problem and how people handle it

**Naive replay scores the wrong world.** "The Replay Gap" (CMU, Aug 2026): forking SWE-bench Verified trajectories and swapping models, 74-77% of early swaps diverge at the very first post-fork action, only 3-8% of the original post-fork states are still valid. Even temperature-0 same-model controls diverged on 50-96% of forks depending on quantization. Recommendation: abandon log-based replay for mid-trajectory model switching, use branching live rollouts. Late forks are much safer than early forks ([arXiv 2608.08239](https://arxiv.org/html/2608.08239)).

Strategies, cheapest to most faithful:

**a) Single-step replay with frozen context (teacher forcing).** Mind2Web evaluates "each step independently based on the ground-truth action history" with Element Accuracy, Step Success Rate, Operation F1; task success only if all steps succeed ([arXiv 2401.01614](https://arxiv.org/pdf/2401.01614)). Langfuse "N+1 evaluation" stores the prefix as the dataset item ([Langfuse](https://langfuse.com/blog/2025-10-09-evaluating-multi-turn-conversations)). Braintrust: single-step evaluation "isolates specific decisions to test tool selection, argument construction, or retrieval relevance" ([Braintrust](https://www.braintrust.dev/articles/how-to-eval)). Hamel Husain: focus on "the first upstream failure" ([hamel.dev](https://hamel.dev/blog/posts/evals-faq/)). Tradeoff: measures next-action quality on the old model's state distribution.

**b) Branching and forking from a trace.** Re-execute the prefix in a fresh container, let the candidate continue. Replay Gap got 99.99% return-code agreement over 11,702 replayed prefix actions, so prefix re-execution is reliable; divergence is in what comes after. SWE-Replay branches at critical intermediate steps ([arXiv 2601.22129](https://arxiv.org/abs/2601.22129)). Sandbox vendors expose snapshot/fork (E2B, Daytona, Modal, Vercel) ([rywalker](https://rywalker.com/research/ai-agent-sandboxes)).

**c) Deterministic environments.** REAL ([arXiv 2504.11543](https://arxiv.org/pdf/2504.11543)); WebStep "bifurcation analysis" ([arXiv 2606.15673](https://arxiv.org/html/2606.15673)); Anthropic: each trial starts from a clean isolated environment ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

**d) Tool mocking and VCR-style cassettes.** langchain-replay warns timestamps/UUIDs baked into prompts break replay ([GitHub](https://github.com/sixty-north/langchain-replay)). agent-vcr records MCP JSON-RPC with five match strategies (exact, method, method_and_params, subset, sequential) and diffs recordings ([GitHub](https://github.com/Jarvis2021/agent-vcr)). Catch: a cassette only has responses for calls the original model made.

**e) Counterfactual / off-policy evaluation.** Causal Agent Replay (CMU, Jun 2026) resamples individual steps, rolls forward many times, reports distributional effects with CIs; "point of commitment" rule ([arXiv 2606.08275](https://arxiv.org/html/2606.08275v1)).

**f) First-divergence and first-critical-error metrics.** Replay Gap "First Divergence Index". TrajDebug/TrajErrBench defines the critical error as "the earliest error step causally linked to the task failure"; failed trajectories average 7.62 local errors but only one critical error, 61.9% of non-critical errors are later repaired; best automated attribution 34% ([arXiv 2608.06346](https://arxiv.org/html/2608.06346v1)). Who&When: step-level attribution 14.2% ([arXiv 2505.00212](https://arxiv.org/abs/2505.00212)). Implication: step-level "first error" labels are expensive and noisy; humans for the gold set.

**g) Simulated users.** tau-bench/tau2 LLM user; simulators show "benevolence bias"; tau2 solo vs interactive up to 25-point drops ([arXiv 2604.21480](https://arxiv.org/pdf/2604.21480)). LangSmith `run_multiturn_simulation` with `fixed_responses` ([docs](https://docs.langchain.com/langsmith/multi-turn-simulation)).

## 2. Vendor approaches

- **LangSmith**: runs/threads to datasets, automation rules, annotation queues; `agentevals` strict/unordered/subset/superset trajectory match with exact/ignore/subset/superset tool-arg matching ([docs](https://docs.langchain.com/langsmith/manage-datasets-in-application), [agentevals](https://github.com/langchain-ai/agentevals)).
- **Braintrust**: trace to dataset; Topics classify every trace daily by Task, Sentiment, Issues ([Braintrust](https://www.braintrust.dev/articles/continuous-evaluation-ai-agents-trace-classifications-2026)).
- **Arize Phoenix**: datasets from traces; `run_experiment(dataset, task, evaluators)`.
- **Langfuse**: add observation to dataset; multi-turn/session dataset items still an open request ([discussion](https://github.com/orgs/langfuse/discussions/4208)).
- **OpenAI Evals API**: `data_source_config` type `logs`; graders string_check, text_similarity, label_model, score_model, python; tool calls not covered ([reference](https://developers.openai.com/api/reference/resources/evals/methods/create)).
- **Anthropic**: task, trial, harness, grader, suite vocabulary; pass@k vs pass^k; prefer state/outcome checks then grade the transcript; start with 20 to 50 tasks drawn from real failures ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **W&B Weave**: calls to dataset; sessions/turns/steps/tools first-class.
- **Humanloop**: acqui-hired by Anthropic Aug 2025; wound down.
- **Hamel Husain / Shreya Shankar**: error analysis first (open coding, axial coding, count), binary labels not Likert, review 100+ traces and stop when ~20 more yield no new category, validate judges by TPR/TNR on 100+ held-out labels ([hamel.dev](https://hamel.dev/blog/posts/evals-faq/)). EvalGen "criteria drift" ([arXiv 2404.12272](https://arxiv.org/abs/2404.12272)). Eugene Yan: 50:50 pass/fail annotated set ([eugeneyan](https://eugeneyan.com/writing/eval-process/)).

## 3. Task clustering and sample size

Clio: extract facets per conversation with a small model, embed, k-means, name clusters with a model, recursive hierarchy; drop clusters below minimum unique-account thresholds; 94% reconstruction accuracy on synthetic ([arXiv 2412.13678](https://arxiv.org/html/2412.13678)). Practitioner adaptation for agent traces: facets = task summary, tool-call sequence, failure fingerprint; ~$0.0075 per trace ([saulius.io](https://saulius.io/blog/hierarchical-clustering-agent-traces-unknown-failure-modes)).

Sample size: Anthropic "Adding Error Bars to Evals": SE = sqrt(p(1-p)/n); clustered SEs up to 3x naive; paired per-item differences (~one third variance reduction); detecting a 3-point gap at 80% power needs ~969 items; K=2 to 6 trials per item cuts variance by one to two thirds ([arXiv 2411.00640](https://arxiv.org/html/2411.00640v1)). With 30 traces per cluster, a 50% pass rate has a 9-point SE. June 2026 audit of BFCL, tau2, LiveMCPBench, MCP-Atlas found 18.5% evaluator-human misalignment; an LLM-judge benchmark swung 57.9 to 76.8% across 23 identical runs ([arXiv 2607.02577](https://arxiv.org/html/2607.02577v1)).

## 4. Data issues

- **PII**: redact at the span processor; keyed-hash pseudonymization so the same identifier maps to the same token across a trace ([dev.to](https://dev.to/gabrielanhaia/redacting-pii-in-llm-traces-without-losing-debuggability-2jll)). OTel GenAI conventions capture no content by default; `gen_ai.input.messages` / `gen_ai.output.messages` when enabled ([OTel](https://opentelemetry.io/blog/2026/genai-observability/)).
- **Prompt caching**: Anthropic caches require byte-identical prefixes, invalidated in tools then system then messages order ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). Replaying N steps of one trace is cache-friendly; pseudonymized prompts or re-serialized schemas zero the hit rate.
- **Tool schemas**: replay needs the exact tool list and JSON schemas the model saw ([MCP spec](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)).
- **Non-deterministic tools**: search, time, remote APIs make recordings stale.
- **Detecting replayability**: no paper defines this; signals are MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), whether the trace contains the full tool schema and system prompt, whether every tool result is present, whether prefix re-execution reproduces recorded results.
- **Multi-model traces**: record which model produced each step and fork only at same-model boundaries.

## 5. Recommended pipeline

1. **Capture complete, content-enabled traces.** Per step: model id, full system prompt, tool list with schemas, input messages, output (incl. tool calls), tool results, environment identifiers. Without tool schema and system prompt the step is not replayable; discard.
2. **Pseudonymize at the span processor, before storage.** Freeze the redacted trace as the eval artifact.
3. **Cluster Clio-style.** Facets (task summary, tool sequence, outcome, failure fingerprint), embed, over-segment, name, 3-level hierarchy, drop clusters under a minimum unique-user threshold. Sample stratified by cluster.
4. **Error analysis by hand first.** 100+ traces, binary pass/fail, label the first critical error step with 2-3 annotators.
5. **Classify each trace's replay tier and build three eval layers.**
   - Tier A: step-level frozen-context items. Item = prefix up to step k, expected = acceptable next action(s); graders = tool-name match with subset arg matching plus a rubric judge for free text.
   - Tier B: forked rollouts for traces whose tools are read-only or idempotent and whose environment can be snapshotted. Always include a same-model control arm at each fork.
   - Tier C: full re-runs with a simulated user seeded from real opening messages, for conversational agents.
   Traces with destructive, non-idempotent or open-world tools and no snapshot get Tier A only.
6. **Cassette non-deterministic tools; live-run deterministic ones.** Log every cache miss as a divergence event.
7. **Grade with code first, judge second, validate the judge.** Measure judge TPR/TNR against human labels.
8. **Report per-cluster verdicts with error bars.** K = 3 to 5 trials per item, paired differences, clustered SEs.
9. **Close the loop.** Automation rules promote new failing traces; refresh with fresh traces per cycle.

**Hard tradeoffs.** Tier A is cheap and reproducible but measures next-action quality on the old model's state distribution; it will overrate a candidate good at imitating the prefix and underrate one that would have taken a better path. Tier B is the only design that answers "would the other model have succeeded," but costs a container per fork and needs environment snapshots most production systems do not keep. Opinionated resolution: ship Tier A for every cluster this week, invest sandbox-snapshot infrastructure in the two or three highest-value clusters for Tier B, and never publish a per-task verdict from fewer than about 50 items and 3 trials without its confidence interval.
