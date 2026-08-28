# How major agent benchmarks grade multi-step trajectories

Research sweep, 2026-08-26. Source: web research agent. Topic 1 of 6.

## 1. Per-benchmark grading mechanics

### tau-bench (Sierra, 2024)
- **Unit:** final database state plus required information in agent messages. "Compares the database state at the end of a conversation with the annotated goal state" ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045)). Reward is binary 0/1 per trial.
- **Scoring:** DB-state hash comparison; `communicate_info` strings must appear as substrings in agent messages. pass^k = all k trials succeed, contrasted with pass@k = at least one succeeds. GPT-4o "drops to ~25% on pass^8 in tau-retail, a staggering 60% drop compared to its pass^1 score" ([Sierra blog](https://sierra.ai/blog/benchmarking-ai-agents)).
- **Argument matching:** not path-based. Any tool sequence producing the same end state passes.
- **Non-determinism:** user is an LLM simulator with lexically varied instructions; pass^k is the explicit answer to variance.
- **Criticisms:** ABC paper found "TAU-bench counts empty responses as successful" ([arXiv 2507.02825](https://arxiv.org/abs/2507.02825)). "Lost in Simulation" measured agent success varying up to 9 points depending on which LLM plays the user ([arXiv 2601.17087](https://arxiv.org/abs/2601.17087)). In tau-airline the simulator went off-instruction in 11 of 50 conversations ([arXiv 2506.07982](https://arxiv.org/pdf/2506.07982)). Amazon's tau2-bench-verified documents policy-compliance, DB-accuracy, logical-consistency and evaluation-ambiguity errors in gold tasks ([amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified)).

### tau2-bench and tau3-bench (Sierra, 2025-2026)
- **Unit:** composite; per-task `reward_basis` picks from DB, ENV_ASSERTION, ACTION, COMMUNICATE, NL_ASSERTION. "The final reward is the product of the components listed in `evaluation_criteria.reward_basis`" ([tau2 evaluation.md](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)).
- **Scoring:** DB check = hash of predicted DB vs target DB, where target = fresh env plus replay of reference actions. ACTION check = "for every entry in `actions`, did the agent produce a matching tool call", but "any sequence of tool calls that produces an equivalent DB end state passes the DB check". NL assertions use an LLM judge (marked WIP). Telecom domain uses only assertion functions on world state ([arXiv 2506.07982](https://arxiv.org/html/2506.07982v1)).
- **Multiple valid paths:** tasks are generated compositionally from atomic subtasks with init/solution/assertion functions, so correctness is verified by assertions rather than paths.
- **Successors:** tau3-bench adds a banking knowledge (RAG) domain, full-duplex voice, and "75+ task fixes: removed incorrect expected actions and clarified ambiguous instructions" ([tau2-bench README](https://github.com/sierra-research/tau2-bench), [Sierra tau3 blog](https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice)).

### BFCL v3 and v4 (Berkeley)
- **Single-turn AST matching:** function name must match; strict typing (booleans not strings, lists ordered, dict key order irrelevant); strings compared case-insensitively after normalisation; a **possible-answers list** allows multiple acceptable values per parameter (`{"location": ["New York City", "NYC"]}`); optional params may be omitted only if the docs mark them optional; parallel/multiple categories are all-or-nothing, order-agnostic. Executable evaluation has exact, real-time (within 20%) and structural match modes ([BFCL blog 8](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)).
- **v3 multi-turn:** two checks per turn. State-based: "compare the backend system's state after all function calls are executed at the end of each turn." Response-based: "the model result is considered correct if it contains the ground truth as a subset, even if it contains additional function calls or takes a different trajectory." Turn is force-failed after 20 steps. Acknowledged limits: state checks cannot see read-only calls; response checks can penalise legitimate exploration ([BFCL blog 13](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)).
- **v4 agentic:** web search = 100 multi-hop questions graded by normalised exact match on the final answer, with randomised simulated HTTP failures ([BFCL blog 15](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)); memory = answer correctness with no dialogue history ([BFCL blog 16](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)); format sensitivity = 26 prompt variations on 200 entries ([BFCL blog 17](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)). Overall = Agentic 40% + Multi-turn 30% + Live 10% + Non-live 10% + Hallucination 10% ([leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)).

### AgentBench (2023)
- **Unit:** per-environment outcome. OS: verification scripts; DB: SQL result or table hash; KG: F1; card game: win rate; ALFWorld: success; WebShop: reward; Mind2Web: step success rate/element accuracy. Failure taxonomy: context limit, invalid format, invalid action, task limit exceeded ([ar5iv 2308.03688](https://ar5iv.labs.arxiv.org/html/2308.03688)).

### AgentBoard (ICLR 2024)
- **Unit:** trajectory progress. Progress rate `r_t = max_{i<=t} (1/K) sum_k f(s_i, g_k)` over K human-annotated subgoals, with `f` a regex/state match returning {0,1}. Grounding accuracy = fraction of valid actions. Stated limitation: "reliance on human-annotated subgoals" ([arXiv 2401.13178](https://arxiv.org/html/2401.13178)).

### ToolBench / ToolEval and StableToolBench
- **Unit:** final answer plus action sequence, judged by an LLM. Pass rate = "proportion of successfully completing an instruction within limited OpenAI API calls"; win rate = pairwise preference by ChatGPT; each judged multiple times; reported 87.1% / 80.3% agreement with humans ([ToolEval README](https://github.com/OpenBMB/ToolBench/blob/master/toolbench/tooleval/README.md)).
- **Criticisms:** only 44.4% of real API calls succeeded; GPT-3.5 judge "cannot reliably determine whether a task is solvable"; StableToolBench introduced a cached/simulated API server and Solvable Pass/Win Rate with solvability decided by a majority vote of judges ([arXiv 2403.07714](https://arxiv.org/html/2403.07714)).

### ToolSandbox (Apple, 2024)
- **Unit:** trajectory snapshots matched to a milestone DAG. Each milestone has column-wise similarity functions (exact match for binary state, ROUGE-L for text, AST match for tool traces), aggregated by geometric mean; the evaluator finds "the highest averaged similarity score among all possible mappings between turns and milestones, given that the resulting chronological milestone sequence is a topological sort of the DAG." Minefields are forbidden events: `score = score_M+ x I(score_M- = 0)`. Limits: milestone authoring "hinders scalability" ([arXiv 2408.04682](https://arxiv.org/html/2408.04682)).

### NESTFUL (IBM, EMNLP 2025)
- **Unit:** predicted API sequence with `$var_i.field$` references. Metrics: F1 on function names, F1 on parameters, partial sequence match, full sequence match, and win rate (executed sequence equals gold answer). Argument values string-exact ([arXiv 2409.03797](https://arxiv.org/html/2409.03797v2)). Best full-sequence match was ~25-28% ([ACL Anthology](https://aclanthology.org/2025.emnlp-main.1702/)).

### API-Bank (2023)
- **Unit:** per-call. Accuracy = exact match of API name and parameters against annotation (not execution); ROUGE-L for the post-call response ([ACL 2023](https://aclanthology.org/2023.emnlp-main.187.pdf)).

### GAIA (2023) and Gaia2/ARE (Meta, 2025-2026)
- **GAIA:** quasi-exact match on a single final answer after type-dependent normalisation. "GAIA does not evaluate the trace leading to the answer... different paths could lead to the correct answer and there is no obvious and simple ways to grade those" ([ar5iv 2311.12983](https://ar5iv.labs.arxiv.org/html/2311.12983)).
- **Gaia2:** 1,120 scenarios in an asynchronous mobile environment; the ARE verifier "evaluates every state-changing write action against oracle annotations", with **hard (exact) checks for rigid fields like IDs and soft (LLM-judge) checks for free-form arguments**, causality via dependency DAGs, timing via tolerance windows, plus task-agnostic style checks against verifier hacking; agreement 0.98 with 450 human-labelled trajectories ([arXiv 2509.17158](https://arxiv.org/html/2509.17158)).

### WebArena (2023)
- **Unit:** final answer or final page/DB state. Information-seeking: `exact_match`, `must_include`, `fuzzy_match` (GPT-4 semantic equivalence). Navigation/config: `program_html` with a locator plus required contents ([ar5iv 2307.13854](https://ar5iv.labs.arxiv.org/html/2307.13854)).
- **Criticism:** Berkeley RDI reached ~100% by reading the config file with reference answers; weak substring matching; unvalidated LLM judge; WebArena and OSWorld both call `eval()` on agent-controlled strings ([RDI blog](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)).

### TheAgentCompany (CMU, 2024-2025)
- **Unit:** checkpoints. `S_full` = 1 only if all checkpoints pass; `S_partial = 0.5 x (points/total) + 0.5 x S_full`. Most checkpoints deterministic; 51 of 175 tasks (29%) use an LLM with rubrics or reference outputs, with deterministic screening first ([arXiv 2412.14161](https://arxiv.org/html/2412.14161v2)).

### SWE-bench, Verified, Pro
- **Unit:** repository end state via tests; resolved only if all FAIL_TO_PASS and PASS_TO_PASS tests pass ([swebench.com](https://www.swebench.com/SWE-bench/guides/evaluation/)).
- **Criticisms:** solution leakage in ~32.67% of successful patches and 12.5-22% of passing patches logically wrong ([SWE-Bench+](https://arxiv.org/pdf/2410.06992)); 6-7 point leaderboard inflation ([arXiv 2506.09289](https://arxiv.org/pdf/2506.09289)). OpenAI audited 138 o3 failures: 59.4% had material test/description issues, and stopped reporting it ([OpenAI](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)). SWE-bench Pro: top pass@1 ~23% ([arXiv 2509.16941](https://arxiv.org/pdf/2509.16941)).

### Terminal-Bench 2.0 (ICLR 2026)
- **Unit:** container end state via test scripts; binary pass/fail; pass@1 averaged over trials ([arXiv 2601.11868](https://arxiv.org/html/2601.11868v1), [tbench.ai](https://www.tbench.ai/news/announcement-2-0)).

### Other 2025-2026 successors
- **MCPMark:** curated initial state, tool loop, programmatic verifier; reports pass@1, pass@4 and pass^4 (gpt-5-medium: 52.56% pass@1, 33.86% pass^4) ([arXiv 2509.24002](https://arxiv.org/abs/2509.24002)).
- **Toolathlon:** final-environment-state verification scripts over MCP tools; flags difficulty with "tasks with multiple valid solution paths" ([arXiv 2510.25726](https://arxiv.org/pdf/2510.25726)).
- **HAL:** standardised harness, 21,730 rollouts; LLM-aided log inspection found agents "searching for the benchmark on HuggingFace instead of solving a task" ([arXiv 2510.11977](https://arxiv.org/abs/2510.11977)).
- **ClawTrack / Claw-Eval (2026):** Task Score (outcome) plus Process Score per turn on goal alignment, efficiency, information utilisation, result verification, using 12,541 rubric items; "reliable pass" requires both thresholds; judge-human Pearson r = 0.912 ([arXiv 2607.28037](https://arxiv.org/html/2607.28037)).
- **TRACE (2025):** reference-free LLM judging over an evidence bank because "annotating all valid ground-truth trajectories is prohibitively expensive"; agents with equal accuracy differ sharply in efficiency and hallucination ([arXiv 2510.02837](https://arxiv.org/html/2510.02837v3)).
- **AgentLens (2026):** coding-agent trajectories reviewed by combining "formal verification, where an objective check exists, with LLM-written trajectory reviews and side-by-side comparisons" ([arXiv 2607.06624](https://arxiv.org/abs/2607.06624)).

## 2. Surveys and meta-evaluation
- Yehudai et al. 2025: evaluation by capability (planning, tool use, reflection, memory); distinguish final-response from stepwise/trajectory evaluation ([arXiv 2503.16416](https://arxiv.org/pdf/2503.16416)).
- Mohammadi et al. 2025: graph-based trajectory matching (Node F1 for tool selection, Edge F1 for invocation order) ([arXiv 2507.21504](https://arxiv.org/html/2507.21504v1)).
- Kapoor et al. "AI Agents That Matter": cost as first-class metric, missing holdouts breed overfitting ([arXiv 2407.01502](https://arxiv.org/abs/2407.01502)).
- ABC checklist: task validity and outcome validity; 7 of 10 benchmarks violated each; prefer state checks, validate LLM judges, report pass^k ([arXiv 2507.02825](https://arxiv.org/abs/2507.02825)).
- "Towards a Science of AI Agent Reliability" (ICML 2026): capability gains "only yielded small improvements in reliability" ([arXiv 2602.16666](https://arxiv.org/abs/2602.16666)).
- Reliability@k critique: pass@k often misapplied with n = number of unit tests rather than independent rollouts ([arXiv 2608.14711](https://arxiv.org/html/2608.14711)).

## 3. Cross-cutting patterns
1. **End state beats path.** tau, BFCL v3 state checks, WebArena, SWE-bench, Terminal-Bench, MCPMark, Toolathlon all grade the world, not the transcript. Path-based graders (API-Bank, NESTFUL full-sequence, ADK `tool_trajectory_avg_score` at threshold 1.0) are the strictest and least tolerant of alternative valid paths ([adk.dev](https://adk.dev/evaluate/)).
2. **Argument matching is tiered.** Exact for IDs, normalised for strings, possible-answer lists (BFCL), subset containment (BFCL v3, openevals subset/superset modes), LLM soft checks for free text (Gaia2). openevals exposes this as `tool_args_match_mode` in {exact, ignore, subset, superset} plus per-tool overrides ([openevals](https://github.com/langchain-ai/openevals)).
3. **Read-only steps are invisible to state graders**, which is why milestone/checkpoint schemes exist.
4. **Non-determinism is handled by repetition, not by widening the reference:** pass^k (tau), pass^4 (MCPMark), trials averaging (Terminal-Bench).
5. **LLM judges are used as a fallback with validation** and are the softest attack surface.

## 4. Implications for an eval built from production traces with the frontier trajectory as ground truth

1. **Do not grade path equality. Grade a derived end state plus a small set of hard invariants.** For each production trace, compute the write-effect set from the frontier trajectory (every side-effecting call with its arguments, plus the final answer) and treat that as the target. Read-only calls in the frontier trace should not be required.
2. **Split arguments into hard and soft keys per tool, the Gaia2 way.** IDs, enums, amounts, dates: exact after normalisation. Free-text fields: LLM soft check with the user request as context. Ship this as a per-tool matcher table.
3. **Use subset containment with an ordering DAG, not strict sequence.** Required calls must appear, causal dependencies must be respected, extra calls are allowed unless they hit a minefield (a forbidden side effect the frontier never issued).
4. **The frontier trajectory is a noisy label, so budget for label QA.** Run pass^k style consistency on the frontier model itself (k=3 to 5 re-rolls per trace) and keep only traces where its end state is stable; disagreeing traces are the ambiguous-task set, not ground truth.
5. **Report pass^k on the candidate, computed with independent rollouts.**
6. **Add a process score for the cases end state cannot see** (goal alignment, unnecessary calls, verification-before-commit, recovery after tool errors). Validate the judge against a few hundred human-labelled steps first.
7. **Isolate the grader from the agent.** Never pass the reference into the candidate's context, never `eval()` agent strings, sanitise agent output before an LLM judge.
8. **Track cost and steps alongside correctness.** "Matches the frontier" at 3x the tokens is a different decision from matching at parity.
9. **Hold out a rotating slice of production traces.** A fixed set will be overfit within weeks.
