# Tool/function-call correctness metrics: definitions, decompositions, recommended suite

Research sweep, 2026-08-26. Source: web research agent. Topic 5 of 6.

## 1. Benchmark-side definitions

### BFCL
- **AST evaluation** ([blog 8](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)): function name must match; all `required` params present and only documented params allowed (catches hallucinated names); type checks (int accepted for float in Python); lists order-sensitive; strings case-insensitive with whitespace and `,./-_*^` removed; dict key order ignored; optional params marked `""` may be omitted.
- **Possible-answer lists**: each parameter's `possible_answer` is a list of acceptable values. Error codes: `wrong_func_name`, `wrong_count`, `missing_required`, `missing_optional`, `unexpected_param`, `type_error:*`, `value_error:*` ([ast_checker.py](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py)).
- **Parallel / multiple**: order-insensitive, all-or-nothing; `wrong_count` fails on extra or missing calls.
- **Relevance vs irrelevance** ([blog 12](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html)): irrelevance = must output no call; relevance = must output some call, correctness not checked because "there could be infinitely many correct function calls".
- **Multi-turn v3** ([blog 13](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)): state-based check compares backend state after each turn; response-based check compares execution path to "minimal viable execution result paths" with subset matching. Pass only if both pass in all turns; force-fail on zero calls or >20 steps. Augmented categories: missing params (should ask), missing functions (should say no tool fits), long context.
- **v4**: search query arguments not graded; only the final answer ([blog 15](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)).

### ToolBench / ToolEval, StableToolBench
- Pass rate (final answer resolves instruction, or honest refusal for unsolvable); win rate via pairwise judge on richness, factuality, reasoning, milestones, exploration, cost; majority vote ([arXiv 2307.16789](https://arxiv.org/html/2307.16789)). StableToolBench: Solvable Pass/Win Rate ([arXiv 2403.07714](https://arxiv.org/abs/2403.07714)).

### NESTFUL
- Partial sequence match (F1 over API names, F1 over `arg = val` slots), full sequence match, win rate (executed with variable substitution) ([arXiv 2409.03797](https://arxiv.org/html/2409.03797v3)). GPT-4o: 28% full sequence, 60% win rate.

### ToolSandbox (Apple)
- Milestones in a DAG with 0-1 similarity (exact, ROUGE-L, AST); geometric means so any zero nullifies; minefields are forbidden events, `score = score_M+ x 1(score_M- = 0)` ([arXiv 2408.04682](https://arxiv.org/html/2408.04682v2)).

### API-Bank, Seal-Tools, ToolACE, HammerBench, ACEBench, tau-bench
- **API-Bank**: correct when name and params both match ([arXiv 2304.08244](https://arxiv.org/pdf/2304.08244)).
- **Seal-Tools**: Format ACC, Tool P/R/F1, Parameter P/R/F1 (name + param name + value); 7% omitted required, 9% overfilled ([arXiv 2405.08355](https://arxiv.org/html/2405.08355)).
- **ToolACE**: rule layer (name in list, required present, regex format) and model layer flagging values "not mentioned in either the user query or the system prompt" (argument hallucination) ([arXiv 2409.00920](https://arxiv.org/html/2409.00920)).
- **HammerBench**: Function Name Accuracy, Parameter Hallucination Rate (PHR), Parameter Missing Rate (PMR), Success Rate, Progress Rate; ROUGE-L thresholds plus LLM re-assessment ([arXiv 2412.16516](https://arxiv.org/html/2412.16516v2)).
- **ACEBench**: Normal (AST), Special (must ask or refuse), Agent (state and process accuracy) ([arXiv 2501.12851](https://arxiv.org/pdf/2501.12851)).
- **tau-bench**: final DB state vs goal; pass^k. Proxy-state work shows state-match and trajectory-match disagree materially ([arXiv 2602.16246](https://arxiv.org/pdf/2602.16246)).

### MCP-based evals
- **MCP-Bench** (ICLR 2026): Tool Name Validity Rate, Schema Compliance Rate, Execution Success Rate; LLM judge on completion, grounding, appropriateness, parameter accuracy, dependency awareness, efficiency, axis-order shuffled ([arXiv 2508.20453](https://arxiv.org/html/2508.20453)).
- **MCPEval**: name/parameter/order matching weighted 0.4/0.4/0.2; strict vs flexible (param similarity >= 0.6) ([arXiv 2507.12806](https://arxiv.org/html/2507.12806)).
- **MCP-Universe**: execution-based only; format 4.8%, static 38.1%, dynamic live ground truth 57.1% ([arXiv 2508.14704](https://arxiv.org/html/2508.14704)).

## 2. Vendor/framework implementations

- **LangSmith agentevals**: `trajectory_match_mode` strict / unordered / subset / superset; `tool_args_match_mode` exact / ignore / subset / superset; `tool_args_match_overrides` per tool (mode, exact-field list, or comparator) ([GitHub](https://github.com/langchain-ai/agentevals)).
- **DeepEval ToolCorrectnessMetric**: correctly used / total; name-only, INPUT_PARAMETERS, OUTPUT; `should_consider_ordering`, `should_exact_match` ([docs](https://deepeval.com/docs/metrics-tool-correctness)).
- **Ragas**: ToolCallAccuracy = argument accuracy x sequence aligned; ToolCallF1 ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/)).
- **Phoenix/Arize**: Tool Selection judge (best tool, exists, necessity, hallucinated, parallel, no-call) and Tool Invocation judge; recommends LLM judge for "CDG Airport" vs "Charles De Gaulle Airport" ([blog](https://arize.com/blog/how-to-evaluate-tool-calling-agents/)).
- **Braintrust autoevals**: JSONDiff (Levenshtein for strings, NumericDiff), ExactMatch, ListContains, ValidJSON ([SCORERS.md](https://github.com/braintrustdata/autoevals/blob/main/SCORERS.md)).
- **Galileo**: Tool Selection Quality (boolean, multi-sample consensus, incl. should-not-call), Tool Error, Action Completion ([docs](https://docs.galileo.ai/concepts/metrics/agentic/tool-selection-quality)).
- **Databricks**: MLflow 3 `make_judge` with `{{ trace }}` ([blog](https://www.databricks.com/blog/building-custom-llm-judges-ai-agent-accuracy)).
- **Google Vertex / ADK**: trajectory_exact_match, in_order_match, any_order_match, precision, recall, single_tool_use; ADK `tool_trajectory_avg_score`, `rubric_based_tool_use_quality_v1`, `final_response_match_v2` ([docs](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/evaluate), [adk.dev](https://adk.dev/evaluate/criteria)).

## 3. Cross-cutting issues

- **Semantically valid argument variants.** Exact match is the default nearly everywhere. Mitigations: possible-answer lists (BFCL), normalization, ROUGE-L then LLM re-check (HammerBench), AlignScore for strings ([ACL 2025](https://aclanthology.org/2025.emnlp-main.1242.pdf)), two-tier AST/EM then LLM judge ([arXiv 2605.15104](https://arxiv.org/pdf/2605.15104)), execution/state outcome as ground truth, or not grading the argument (BFCL v4 search). Judge reliability caveat on tool traces: substring kappa 0.049, single GPT-4o-mini judge kappa 0.567 ([arXiv 2604.16706](https://arxiv.org/html/2604.16706)).
- **Order.** Strict (Ragas, LangSmith strict, Vertex exact/in-order), unordered multiset (BFCL parallel, Vertex any-order), DAG-constrained (ToolSandbox, NESTFUL).
- **Extra / redundant calls.** Allowed by subset matching; penalized by precision, F1, ToolScan "Repeated API Calls", RelyToolBench utility `R_task - P_tool - P_hallucination` ([arXiv 2412.04141](https://arxiv.org/html/2412.04141)), MAST "Step repetition" 15.7% ([arXiv 2503.13657](https://arxiv.org/html/2503.13657)).
- **Missing calls.** Recall, ToolScan "Insufficient API Calls" (most common error), AgentHallu "Missing Tool", under-calling FN ([arXiv 2605.00737](https://arxiv.org/pdf/2605.00737)).
- **When not to call.** BFCL irrelevance and v3 missing-params/functions, ToolSandbox minefields, ACEBench Special, Phoenix "should have answered directly", Galileo call-necessity; confusion-matrix framing over-calling FP / under-calling FN / abstention TN.

## 4. Error decomposition and failure taxonomies

- **ToolScan**: Insufficient API Calls, Incorrect Argument Value, Incorrect Argument Name, Incorrect Argument Type, Repeated API Calls, Incorrect Function Name, Invalid Format ([arXiv 2411.13547](https://arxiv.org/html/2411.13547)).
- **RelyToolBench**: selection hallucination (type, timing) vs usage hallucination (format, content); Reliable Pass Rate = Pass Rate - Task Hallucination Rate.
- **AgentHallu**: 14 subtypes; tool-use step localization accuracy only 11.6% ([arXiv 2601.06818](https://arxiv.org/html/2601.06818v1)).
- **MAST**: 14 modes in 3 categories (system design 43.8%, inter-agent misalignment 32.35%, task verification 23.5%).
- **TRAIL**: reasoning / system execution / planning-coordination; best model 11% joint ([arXiv 2505.08638](https://arxiv.org/html/2505.08638)).
- **AgentErrorTaxonomy / AgentDebug**: memory, reflection, planning, action, system; failures cluster in steps 6-15 and cascade ([arXiv 2509.25370](https://arxiv.org/html/2509.25370)).
- **Who&When**: best 53.5% agent-level, 14.2% step-level ([arXiv 2505.00212](https://arxiv.org/abs/2505.00212)). AgentProp-Bench: parameter-level error propagates to wrong final answer with p ~ 0.62.

## 5. Recommended metric suite: candidate next action vs reference frontier action

Score each (context, reference_action, candidate_action) triple on four orthogonal axes, record separately, then aggregate. Treat the frontier action as a strong prior, not an oracle.

**Axis 0. Action type (no-call gate).** Classify both as `call`, `answer`, `ask_clarification`, `refuse`. 4x4 confusion matrix; report over-call rate, under-call rate, abstention agreement. Stop scoring when types disagree unless a state-based check is available.

**Axis 1. Tool selection.** Single calls: exact name after normalization. Parallel sets: unordered multiset, precision/recall/F1 plus all-or-nothing flag. Codes: `wrong_func_name`, `hallucinated_tool`, `extra_call`, `missing_call`, `repeated_call`. Hard-fail on repeated identical (name, args) pairs.

**Axis 2. Arguments (only when selection matched).** Per-parameter typed scoring, geometric mean per call:
- Schema layer (deterministic, first): required present, no unexpected names, type validity. Report PMR and PHR.
- Enumerated / numeric / boolean / id / date: exact after normalization; possible-answer list where available.
- Structured containers: order-sensitive only if the schema says so; per-tool overrides.
- Free-text params (queries, messages, code): Tier A cheap similarity (normalized Levenshtein or token F1, per-tool threshold); Tier B LLM judge with tool schema, user turn, reference value, one binary question: "would this value make the tool return results that satisfy the same user intent as the reference?", 3 samples majority vote, validated on a 200-item human slice; Tier C where the tool is executable and idempotent: execute both and compare result sets (Jaccard).
- Grounding check: flag any value appearing in neither user context nor prior tool results. Report argument-hallucination rate independent of match.

**Axis 3. Ordering and dependency.** Exact / in-order / any-order flags, precision/recall, dependency-aware check (a call consuming a prior result must follow it). For a single next action: "is the candidate the reference call or any call in the minimal-viable set whose preconditions are satisfied."

**Aggregation and reporting.**
- Headline: strict next-action accuracy = type match AND selection match AND all params >= threshold.
- Secondary: lenient accuracy with Tier B/C free text and admissible-set selection.
- Always publish the decomposition: over-call, under-call, wrong tool, hallucinated tool, extra/repeated call, missing required param, hallucinated param name, wrong typed value, free-text mismatch, ungrounded value.
- Where a sandbox exists, add a state-based check and report agreement between state-based and trajectory-based verdicts; disagreement estimates how often the frontier reference was not the only valid answer.
- Never let the LLM-judge tier raise a call that failed the deterministic schema layer.
