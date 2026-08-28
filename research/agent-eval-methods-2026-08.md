# Evaluating LLM agents from recorded traces: step level and trajectory level

Literature sweep, 2026-08-26. Sources: roughly 45 web searches and 50 page fetches covering arXiv, benchmark repos, vendor docs (Anthropic, OpenAI, Google, LangChain, Langfuse, Arize, Braintrust, DeepEval, Patronus, NVIDIA) and practitioner blogs (Hamel Husain, Eugene Yan). Where a paper's full text could not be fetched, the entry says so and relies on the abstract plus secondary sources.

Conventions used below. "Step" means one (observation, reasoning, tool call or final message) unit inside a trace. "Trajectory" means the full recorded episode. "Reference-based" means a gold action, gold trajectory, or gold end state exists; "reference-free" means the judge reasons only from context. "Teacher forcing" means the model is scored on each recorded prefix, with the recorded (not the model's own) history supplied.

---

## 0. Shared vocabulary (from Anthropic, Jan 2026)

Anthropic's "Demystifying evals for AI agents" fixes the terms most 2026 writing now uses:

- Task: one test case with inputs and success criteria.
- Trial: one stochastic run of a task.
- Transcript / trace / trajectory: "the complete record of a trial, including outputs, tool calls, reasoning, intermediate results, and any other interactions."
- Outcome: final environment state after the trial (a reservation exists in the DB, not "the agent said it booked").
- Grader: logic that scores a trial; a task may combine several graders by weighting, hard requirements, or hybrids.
- Grader types: code-based (string/regex/unit tests/static analysis: fast, objective, brittle to valid variation), model-based (flexible, non-deterministic, must be calibrated against humans), human (gold standard, does not scale).
- pass@k = P(at least one of k trials succeeds); pass^k = P(all k succeed). Use pass@k for exploratory settings, pass^k where reliability matters.
- Rule of thumb: "It's often better to grade what the agent produced, not the path it took" to avoid brittle tests, but transcript grading still matters for tool-call sequences, token efficiency and reasoning quality. Start with 20 to 50 tasks, ensure every task is passable by an agent that follows instructions, include negative cases (behaviour must be absent), and verify graders against a reference solution.

URL: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

---

## 1. Tool-calling accuracy benchmarks: what exactly is matched

### Quick comparison

| Benchmark | Year | Unit scored | What is compared | Argument matching | Order / parallel handling | Judge |
|---|---|---|---|---|---|---|
| BFCL v1/v2 | 2024 | single turn | function name + args vs possible-answer list | type-aware; strings normalised; lists ordered | multiple/parallel calls matched order-insensitively, all-or-nothing | AST (code) |
| BFCL v3 | 2024 | multi-turn, per turn | backend state after turn (writes) + response subset match (reads) | state hash / subset | extra exploratory calls allowed | code |
| BFCL v4 | 2025 | agentic (web search, memory, format sensitivity) | as above + error recovery | as above | as above | code |
| ToolBench / ToolEval | 2023 | whole trajectory | Solved / Unsolved; pairwise preference | none (judge reads answer) | n/a | ChatGPT judge |
| API-Bank | 2023 | single call / dialogue | API call exact match + ROUGE-L on reply | exact | n/a | code |
| ToolTalk | 2023 | per assistant turn (teacher forced) | set of tool calls vs gold; success = recall 1 and no incorrect actions | per-tool equivalence fn; free text via embedding cosine > 0.9; non-action tools by execution result | sets (order-insensitive) | code |
| NexusRaven | 2023 | single query | parsed call(s) vs gold | dict of args, P/R/F1 | single, parallel, nested | code |
| tau-bench | 2024 | episode | final DB state vs goal state, required outputs in reply, pass^k | n/a (state) | any tool sequence that reaches state passes | code |
| tau2-bench | 2025 | episode (dual control) | product of DB check x communicate check (x action check, rare) | substring for communicate_info | reference actions only diagnostic | code + optional LLM NL assertions |
| ToolSandbox | 2024 | episode with simulated user | milestones and minefields on any trajectory | per milestone similarity | order via milestone DAG | code + LLM similarity |
| AgentDojo | 2024 | episode | utility fn on env state + output; targeted attack success | n/a | n/a | code |
| ACEBench | 2025 | single call / episode | AST + candidate answer pool; end-to-end state; process accuracy n/m | exact vs candidates | process accuracy is sequence alignment | code + GPT-4o user sim |
| AppWorld | 2024 | episode | avg 8 state-based unit tests incl. collateral damage | n/a | multiple valid solutions | code |
| MCPMark / MCP-Universe | 2025 | episode | programmatic verification script on sandboxed final state | n/a | n/a | code |
| Vertex AI / agentevals / DeepEval | 2025 | trajectory | exact, in-order, unordered, subset, superset, precision, recall | exact / ignore / subset / superset / per-tool override | mode-dependent | code, optional LLM |

### 1.1 Berkeley Function Calling Leaderboard (BFCL) v1 to v4 (Patil et al., 2024; ICML 2025)

Summary. BFCL is the de facto standard for function-calling accuracy. v1/v2 score single-turn calls with an AST matcher; v3 (Sept 2024) adds multi-turn, multi-step tasks with a stateful backend; v4 (2025) adds agentic tracks (web search with injected HTTP errors, memory, format sensitivity). It deliberately avoids LLM judges, so results are reproducible.

Exact AST matching rules (v1/v2 blog):
- Function name must match (dots and underscores normalised because some APIs forbid dots).
- All parameters marked `required` must be present; any parameter not in the schema counts as a hallucination and fails.
- Types: booleans must be real booleans (string "true" fails); Python allows int-to-float widening, Java/JS require exact types.
- Strings: case-insensitive, whitespace and the punctuation set `,./-_*^` stripped before comparison; value must equal one of the listed possible answers.
- Lists/tuples: order matters and elements must match exactly; if order is irrelevant the ground truth lists all permutations.
- Dicts: keys and values checked, key order ignored.
- Optional parameters: may be omitted only if the function doc marks them optional with a default; otherwise the value must be supplied and correct.
- Multiple and parallel calls: each possible answer is associated with its function doc; model outputs are matched against them with "the order of model outputs relative to possible answers is not required", scored all-or-nothing.
- Relevance detection: two categories test that the model emits no call when no function fits (chatting capability, function relevance detection).
- Executable eval (v1/v2): non-REST functions are run and compared by exact match, real-time match (numerical within 20 percent), or structural match (types, list length, dict keys); REST calls check effective execution, response type and JSON key consistency rather than static values.

Multi-turn (v3): 1,000 entries (200 base, 200 missing parameters, 200 missing functions, 200 long context, 200 composite). Two checks per turn: state-based evaluation compares the backend state after each turn for write/delete operations; response-based evaluation uses subset matching, i.e. "the model's execution path must contain all ground truth function calls as a minimum, but additional exploratory steps are permitted."

Limitations. Possible-answer lists must be hand-enumerated, so semantically equivalent but textually different arguments ("CDG" vs "Charles de Gaulle") fail unless listed; all-or-nothing scoring gives no partial credit; single-turn AST cannot reward recovery behaviour; multi-turn categories run on the model's own history so an early error cascades.

URLs: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html , https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html , https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html , https://proceedings.mlr.press/v267/patil25a.html

### 1.2 ToolBench / ToolEval (Qin et al., ToolLLM, 2023; ICLR 2024)

Summary. 16k RapidAPI tools; solutions produced via DFSDT. ToolEval uses ChatGPT as the judge. Pass Rate = proportion of instructions the model completes within a fixed API-call budget (the judge labels an answer Solved, Unsolved, or Unsure). Win Rate = percentage of pairwise comparisons in which the candidate's answer (an action sequence plus final answer) is preferred over a ChatGPT-ReACT reference, using predefined criteria delivered as a prompt; each pair is judged several times and averaged. Reported evaluator agreement with humans: 87.1 percent on pass rate, 80.3 percent on win rate. Later versions of the leaderboard report Solvable Pass Rate and Solvable Win Rate, restricting to queries verified as solvable because many RapidAPI endpoints are dead or unreliable.

Limitations. Judge is a closed model and its criteria are a prompt, so scores drift with the judge version; no argument-level matching at all; live APIs are non-deterministic, which is why the "solvable" filter was needed.

URLs: https://github.com/OpenBMB/ToolBench/blob/master/toolbench/tooleval/README.md , https://openbmb.github.io/ToolBench/

### 1.3 API-Bank (Li et al., 2023, EMNLP)

Summary. 73 APIs, 314 annotated dialogues, three ability levels: Call, Retrieve+Call, Plan+Retrieve+Call. Metric: exact-match accuracy of the API call (name and arguments) given the dialogue prefix and, for the response turn, ROUGE-L between generated and reference reply. Retrieval precision/recall and end-to-end success reported for the harder levels. Limitations: exact match on arguments, single gold call per turn, ROUGE-L is a weak proxy for reply quality.

URL: https://arxiv.org/abs/2304.08244

### 1.4 ToolTalk (Farn and Shin, Microsoft, 2023)

Summary. 78 conversations (50 hard, 28 easy), 28 tools in 7 plugins, each with a Python simulator, a database, an "is action" flag (has side effects) and a per-tool equivalence function. This is the cleanest early example of turn-level teacher forcing: for every prefix ending in a user utterance, the model is given the ground-truth tool calls, results and assistant replies of all earlier turns, then predicts tool calls for the current turn, each executed in the simulator until it emits a reply; predictions are then discarded and the next turn starts from the gold history.

Exact metrics (Algorithm 2 in the paper): each gold call g may match at most one predicted call p via f_tool(p, g). M = matched predictions, P = all predictions, G = all gold calls, A = predicted action calls, I = actions that match no gold call and executed without error.
- precision = |M| / |P|; recall = |M| / |G|; incorrect action rate = |I| / |A|
- success = (M == G) and (I == empty), i.e. perfect recall and zero incorrect actions.
Argument matching: for action tools a hand-written comparator per tool (e.g. recipient sets compared as sets, order ignored); free-text arguments (message bodies) compared by DistilBERT sentence embedding cosine similarity > 0.9; optional arguments are ignored if the gold call leaves them empty; non-action (search) tools are compared by execution result rather than arguments. Error taxonomy from the analysis: premature tool calls (hallucinating arguments the user never gave), faulty planning (omitting or wrong tools), incorrect invocation of the correct tool.

Limitations. Only successful conversations are annotated; one gold plan per turn; embedding threshold is arbitrary; teacher forcing means recovery from the model's own mistakes is never tested.

URL: https://arxiv.org/abs/2311.10775

### 1.5 NexusRaven / Nexus Function Calling Benchmark (Nexusflow, 2023)

Summary. Nine real-API task sets (eight public, one held out to detect overfitting) across single, parallel and nested calls. Evaluation parses the model output into (function name, argument dict, optional id) and computes precision, recall and F1 against gold annotations; nested calls require the argument to itself be a correct call. Limitation: exact argument values, small task sets, no state.

URL: https://github.com/nexusflowai/NexusRaven-V2

### 1.6 tau-bench (Yao et al., Sierra, 2024) and pass^k

Summary. Retail and airline domains with a database, policy document, and an LLM-simulated user. Success is determined by "comparing the database state at the end of a conversation with the annotated goal state", plus a check that required information appears in the agent's reply. Because the check is on state, any tool sequence that reaches the goal state passes.

Metric. pass^k = P(all k independent trials of a task succeed) = p^k under independence. A 90 percent pass@1 agent has pass^8 of about 43 percent. GPT-4o at launch: under 50 percent pass@1, pass^8 under 25 percent in retail.

Limitations. Simulated user is itself an LLM and a source of noise; tasks were hand-written; no partial credit; action-level behaviour is invisible to the score.

URL: https://arxiv.org/abs/2406.12045

### 1.7 tau2-bench (Sierra, 2025) and tau3-bench (2026)

Summary. tau2 adds a telecom domain modelled as a Dec-POMDP where the simulated user also holds tools (dual control), a compositional task generator, and a user simulator constrained by tools and state. Reward, per the repo docs: `reward = product of components in reward_basis`. DB check: the reference actions are replayed on a fresh environment to compute a target DB, and the agent's final DB is compared by hash ("any sequence of tool calls that produces an equivalent DB end state passes"). Communicate check: every string in `communicate_info` must appear as a substring of an agent message. Action check: match against reference tool calls with arguments; only counted when `ACTION` is in `reward_basis` (about nine banking_knowledge tasks), otherwise purely diagnostic, broken down into READ vs WRITE via `partial_action_reward`. NL assertions: an LLM checks natural-language assertions (marked experimental). pass^k is the headline reliability number. tau3 (2026) adds a banking knowledge-retrieval domain, full-duplex voice, and 75+ task fixes; scores are not comparable with the frozen tau1 board.

Limitations. Substring checks are brittle; NL assertions reintroduce an LLM judge; the simulated user leaks noise into pass^k.

URLs: https://arxiv.org/abs/2506.07982 , https://github.com/sierra-research/tau2-bench , https://benchmarkingagents.com/tau3-bench/

### 1.8 ToolSandbox (Apple, 2024)

Summary. Stateful tool execution with implicit state dependencies (e.g. must enable WiFi before search), a built-in LLM user simulator for on-policy conversational evaluation, and a milestone / minefield scheme: milestones are required intermediate or final events (a tool call with specific arguments, a state change, or a reply) organised as a DAG that fixes required ordering; minefields are events that must never occur. Each milestone is matched to the closest event in the trajectory with a similarity measure, so evaluation works "over an arbitrary trajectory" rather than against one gold path. Categories such as State Dependency, Canonicalization (converting user phrasing into API-valid values) and Insufficient Information were the hardest.

Limitations. Milestones are human-authored per scenario; similarity thresholds are tunable; user simulator noise.

URLs: https://arxiv.org/abs/2408.04682 , https://github.com/apple/ToolSandbox

### 1.9 AgentDojo (Debenedetti et al., NeurIPS 2024)

Summary. 97 user tasks and 629 security cases across banking, Slack, workspace and travel environments. Each user task has a utility function evaluated on the final environment state and the model output; each injection task has a security function checking whether the attacker goal was achieved. Reported: benign utility, utility under attack, targeted attack success rate. Claude 3.5 Sonnet reached 78 percent benign utility; GPT-4o dropped from 69 to 50 percent utility under attack.

Limitations. Utility functions are hand-coded per task; no partial credit; attack set is static and now widely used for defence tuning.

URL: https://arxiv.org/abs/2406.13352

### 1.10 ACEBench (USTC and Huawei, 2025, EMNLP Findings)

Summary. Three data types. Normal: AST parse of the generated call, function and parameter verification against a candidate answer pool ("matching any candidate constitutes correctness"), binary score. Special: imperfect instructions (missing required parameter, wrong parameter value, no matching function); score 1 if the model correctly identifies the issue. Agent: multi-turn interaction with GPT-4o as user simulator; end-to-end accuracy = final instance attributes equal target (binary); process accuracy = n/m where m is the length of the ideal call sequence and n the number of aligned steps. Error taxonomy on Normal data: function name, parameter type, parameter value, output format, missing parameter (parameter value errors dominate).

Limitations. Candidate pools still enumerate answers; process accuracy assumes a single ideal sequence; user simulator is a closed model.

URL: https://arxiv.org/abs/2501.12851

### 1.11 AppWorld (Trivedi et al., ACL 2024)

Summary. 750 tasks over 9 apps and 457 APIs with about 100 simulated users. Evaluation uses on average 8 (max 22) state-based unit tests per task that check database entries and changes, explicitly allowing different valid solutions and flagging collateral damage (unexpected changes). This is the reference design for "grade the outcome, not the path, but also check side effects."

URL: https://arxiv.org/abs/2407.18901

### 1.12 MCP-era benchmarks: MCPMark, MCP-Bench, MCP-Universe, MCP-AgentBench (2025)

Summary. MCPMark: 127 expert-and-agent co-created tasks, each with a curated initial state in a sandbox and a programmatic verification script over the final environment; explicit state tracking for reproducibility. MCP-Universe connects to real running MCP servers and stresses long-horizon interaction and unknown tool discovery. MCP-Bench (Accenture) evaluates discovery, selection and use end-to-end. Limitations: real servers drift; verification scripts are per-task code.

URLs: https://arxiv.org/abs/2509.24002 , https://github.com/Accenture/mcp-bench

### 1.13 Gaia2 / ARE (Meta, 2025)

Summary. 1,120 scenarios in a smartphone-like environment that runs asynchronously with injected events and delays; verifiers are part of the scenario graph so checks can run after write actions and stop early on failure. Oracle events are provided for a development split. Adds ambiguity handling, agent-to-agent collaboration and temporal constraints as scored dimensions.

URL: https://arxiv.org/abs/2509.17158

### 1.14 Proxy State-Based Evaluation (PayPal AI, Feb 2026)

Summary. Keeps final-state evaluation without a deterministic backend: a scenario specifies user goal, user/system facts, expected final state and expected behaviour; an LLM state tracker infers a structured proxy state from the full trace; LLM judges verify goal completion and detect tool or user hallucinations against scenario constraints. Human-LLM judge agreement above 90 percent; produces stable model rankings. Limitation: the state tracker is itself an LLM, so errors are correlated with the agent's own.

URL: https://arxiv.org/abs/2602.16246

### 1.15 Vendor trajectory metrics (Google Vertex AI, LangChain agentevals, DeepEval, Arize Phoenix, Langfuse)

Google Vertex AI Gen AI evaluation service defines: trajectory_exact_match (1 iff same tool calls in the same order), trajectory_in_order_match (1 iff all reference calls appear in order, extras allowed), trajectory_any_order_match, trajectory_precision (predicted calls that appear in reference / predicted calls), trajectory_recall (reference calls that appear in predicted / reference calls), trajectory_single_tool_use, plus response_match_score (ROUGE-1) and final_response_match_v2 (LLM judge). URL: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-agents

LangChain agentevals: `create_trajectory_match_evaluator` with `trajectory_match_mode` in {strict, unordered, subset, superset}; `tool_args_match_mode` in {exact (default), ignore, subset, superset}; `tool_args_match_overrides` maps tool name to a mode, a list of fields, or a custom comparator; LLM-as-judge trajectory evaluators with and without a reference; graph trajectory evaluators for LangGraph node paths. URL: https://github.com/langchain-ai/agentevals

DeepEval: ToolCorrectness (deterministic vs expected_tools, configurable on ordering and frequency), ArgumentCorrectness (LLM, reference-free, or percentage-of-parameters-correct), Task Completion (reference-free LLM infers the goal and judges), Step Efficiency (redundant calls / total), Plan Quality and Plan Adherence (LLM). URL: https://deepeval.com/docs/metrics-tool-correctness

Arize Phoenix: two reference-free judges, Tool Selection (right tool, parallel tools, or correctly no tool) and Tool Invocation (all required parameters present, no hallucinated or intent-inconsistent arguments); a reference-based custom judge for semantic equivalence ("CDG Airport" equals "Charles De Gaulle Airport"). Documented pitfalls: models assuming the training-cutoff year, judges too strict on equivalence, missing tools making correct arguments impossible. URL: https://arize.com/blog/how-to-evaluate-tool-calling-agents/

Langfuse: recommends checking tool `name`, `arguments`, `type`, `id`, `index` against a schema, counting unnecessary calls, loops and retries, recovery after failed calls, and turning failing production traces into dataset items. URL: https://langfuse.com/resources/engineering/ai-agent-evaluation

---

## 2. Step-level next-action prediction with teacher forcing

### 2.1 Who does it

- Mind2Web (Deng et al., 2023). Each step is evaluated independently with the ground-truth action history provided ("with the assumption that the model successfully completes all previous steps"). Metrics: Element Accuracy (selected element in the set of acceptable elements), Operation F1 (token-level F1 on the operation plus value), Step Success Rate (element and operation both correct), Task Success Rate (all steps correct). URL: https://arxiv.org/abs/2306.06070
- WebLINX (Lu et al., 2024). Turn-level scoring on 2,337 demonstrations: Intent Match (binary), Element IoU for click/textinput/submit, chrF F1 for text; micro-averaged into a WebLINX score. History = last five actions and utterances from the recorded demo. URL: https://arxiv.org/abs/2402.05930
- AndroidControl / AitW style offline evaluation. Screenshot plus gold history, relaxed action accuracy (a click is correct if inside the target element's box). Reported that "human validation scores typically outperform complete action matching scores due to the multiple valid action alternatives" (e.g. system back vs app back). URLs: https://arxiv.org/pdf/2307.10088 , https://arxiv.org/html/2502.06395
- ToolTalk (above): turn-level teacher forcing with gold tool calls and results for prior turns.
- ToolPRMBench (2026): "offline sampling constrains models to follow golden trajectory prefixes, isolating single-step errors", contrasted with online sampling from full rollouts. URL: https://arxiv.org/abs/2601.12294
- OpenCUA / AgentNetBench (2025): annotates multiple plausible actions per step precisely because single gold actions under-count. URL: https://arxiv.org/pdf/2508.09123
- LangChain run-level evals: about half of the recommended test cases are single-step ("Did the agent call the right tool at this step? Did it pass the correct arguments?"), isolated with `interrupt_before`. URL: https://www.langchain.com/resources/agent-evals
- OpenAI evaluation best practices: "evaluate each workflow step independently", separately testing instruction following, tool selection, argument precision and handoffs. URL: https://developers.openai.com/api/docs/guides/evaluation-best-practices

### 2.2 Known problems

1. Exposure bias / compounding error. Teacher-forced accuracy is measured only on states that lie on the reference trajectory. At deployment the model conditions on its own history, so a step-level score never measures recovery. STeCa (ACL Findings 2025) frames this for LLM agents: "suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories"; it estimates a Monte Carlo step reward r_step(s_{t-1}, a_t) = E[r_o] over N rollouts from the state, flags a step as deviated when its reward falls below the expert action's, and measures trajectory deviation with normalised DTW. URL: https://aclanthology.org/2025.findings-acl.604.pdf
2. Multiple valid actions. One gold action per step systematically under-scores agents (AitW, OpenCUA, Braintrust: "several routes may complete the same task correctly, so the scorer should allow an acceptable step range and multiple valid paths"). Fixes in use: acceptable-element sets (Mind2Web), candidate answer pools (ACEBench, BFCL), milestones instead of paths (ToolSandbox), state checks instead of action checks (tau, AppWorld), multi-annotated steps (AgentNetBench).
3. Offline versus online gap. Offline step accuracy and online task success are only loosely correlated: STEP reports offline-driven gains of 21.1 points on the OSWorld train split translating to 7.0 points overall; "An Illusion of Progress" shows agents claiming about 90 percent on WebVoyager achieve 61 percent (Operator) on Online-Mind2Web. AgentRewardBench shows rule-based checks reach only 55.9 percent recall on WebArena, so the reference itself is often wrong.
4. Off-trajectory divergence when substituting models. "The Replay Gap" (Aug 2026) measured what happens when you score a different model on recorded prefixes: swapping the model at step k rewrites 61 to 94 percent of post-fork actions relative to same-model controls; 74 to 77 percent of early swaps diverge at the very first action, leaving only about 3 percent of replayed states valid; replay mispredicted every success-relevant outcome. Conclusion: prefix-based scoring is a proxy for "would this model have emitted the recorded action", not for "would this model have finished the task". URL: https://arxiv.org/abs/2608.08239
5. Neutral and exploratory steps. AgentProcessBench (2026) found judges cannot reliably separate harmless exploratory steps from harmful ones, and that over-predicting "correct" (positive label bias) is the dominant judge failure, so binary step labels are too coarse; it uses a ternary +1 / 0 / -1 scheme with an error-propagation rule (dependent steps after an error stay -1 until the agent recovers). URL: https://arxiv.org/html/2603.14465v1
6. Long-context degradation. Step-level judge accuracy falls 10 to 21 points on long trajectories and first-error localisation degrades faster than overall step accuracy (AgentProcessBench, Plan-RewardBench).

---

## 3. LLM-as-judge for agent trajectories

### 3.1 Foundational judge findings (chat era, still cited for agents)

Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., NeurIPS 2023). GPT-4 vs human agreement about 85 percent, above human-human 81 percent, but documents position bias (GPT-3.5 inconsistent in about 50 percent of swapped pairs, Claude v1 about 70 percent), verbosity bias (over 90 percent preference for padded answers in a repetition attack), self-enhancement bias (10 to 25 percent higher win rate for own outputs) and weak math/reasoning grading. Mitigations: swap positions and only count consistent verdicts, few-shot calibration, chain-of-thought and reference-guided grading. URL: https://arxiv.org/abs/2306.05685

Self-Preference Bias in LLM-as-a-Judge (Wataoka et al., 2024). Introduces a quantitative self-preference metric; finds judges assign higher scores to lower-perplexity text "regardless of whether the outputs were self-generated", i.e. familiarity drives the bias. URL: https://arxiv.org/abs/2410.21819

Justice or Prejudice? (CALM, 2024). Twelve bias types measured by principle-guided perturbation with robustness rate and consistency metrics; concludes "significant biases persist in certain specific tasks." URL: https://arxiv.org/abs/2410.02736

JudgeBench (ICLR 2025). Converts hard datasets into response pairs with objective correctness labels; accuracy on pairs; strong judges such as GPT-4o are "just slightly better than random guessing", so preference-alignment benchmarks overstate judge reliability on correctness. URL: https://arxiv.org/abs/2410.12784

Prometheus 2 (EMNLP 2024). Open 7B / 8x7B evaluator supporting direct assessment (1 to 5 with a custom rubric and optional reference) and pairwise ranking; Pearson 0.6 to 0.7 with GPT-4 on Likert benchmarks, 72 to 85 percent agreement on pairwise sets; removing the reference answer causes the largest degradation. URL: https://arxiv.org/abs/2405.01535

Eugene Yan, "Evaluating the Effectiveness of LLM-Evaluators" (2024). Synthesis of about two dozen papers: pairwise is more stable than pointwise for subjective criteria, pointwise is better for objective checks; fine-tuned evaluators behave like task-specific classifiers and generalise poorly; recommends classification metrics (recall and precision on defects, Cohen's kappa) over correlation, because correlation "doesn't account for chance agreement" and hides recall on bad outputs. URL: https://eugeneyan.com/writing/llm-evaluators/

### 3.2 Checklists and rubrics

TICK (Cook et al., 2024). LLM-generated, instruction-specific YES/NO checklists; exact agreement between judge and human preferences rises from 46.4 to 52.2 percent versus direct scoring; giving the checklists to humans raises inter-annotator agreement from 0.194 to 0.256; STICK (self-TICK) improves generation via self-refinement (+7.8 on LiveBench reasoning). URL: https://arxiv.org/abs/2410.03608

HealthBench (OpenAI, 2025). 5,000 conversations, 48,562 physician-written criteria, each with a point value in [-10, 10] (negative for undesirable behaviour). GPT-4.1 grades each criterion met / not met; score = points earned / max possible, clipped at 0. Meta-evaluation on about 61k physician-labelled criterion judgments gives macro F1 0.71 for the grader, roughly physician level, with a later analysis attributing the ceiling to about 22.5 percent physician disagreement. URL: https://openai.com/index/healthbench/

Rubrics as Rewards (Gunjal et al., 2025) and the rubric-generation family (OpenRubrics, RubricHub, EvoRubrics, Auto-Rubric, ARCO for multi-step agents, 2025 to 2026). Prompt-specific rubrics decomposed into weighted criteria scored by an LLM judge; rubric-structured rewards "yield better alignment for smaller judges and reduce performance variance across judge scales." Methods for producing rubrics: LLM synthesis from the query (optionally with a reference), mining human documents, or inducing from preference data. Open issue: quality control of generated rubrics and reward hacking of static rubrics. URLs: https://arxiv.org/abs/2507.17746 , https://arxiv.org/abs/2510.07743

### 3.3 Agent-specific judges and judge benchmarks

Agent-as-a-Judge (Zhuge et al., Meta/KAUST, 2024; ICML 2025). A judge that has tools: graph (project structure), locate, read (33 formats), search, retrieve (trajectory segments), ask (is requirement satisfied), plus planning and memory that ablations showed hurt. DevAI: 55 AI-development tasks with 365 hierarchical requirements; each requirement judged yes/no, either independently or with dependencies. Metrics: alignment rate with human consensus (90.44 percent vs 60.38 percent for LLM-as-a-Judge in black-box mode), judge shift (0.27 percent vs 31.42 percent), PR curves because failing requirements dominate. Gray-box (with trajectory) beats black-box (artifacts only). Cost: 2.29 percent of human time, 2.64 percent of cost. Limitation: human evaluators disagreed 10 to 30 percent, setting a ceiling; error propagation between modules. URL: https://arxiv.org/abs/2410.10934

TRAIL (Patronus AI, 2025). 148 human-annotated OpenTelemetry traces (118 GAIA, 30 SWE-bench Lite), 1,987 spans, 841 errors (5.68 per trace). Taxonomy: Reasoning (hallucination, information processing, decision making, output generation), System Execution (configuration, API/system, resource management), Planning and Coordination (context management, task management). Judge metrics: category F1 (error type classification), location accuracy (which span), joint accuracy (category and span both right), plus Pearson correlation with human trace-level scores for reliability, security, instruction adherence and plan optimality. Best model (Gemini 2.5 Pro) scored 11 percent joint accuracy. Annotation cost about 110 to 120 minutes per trace. Limitations: text-only, long tail of rare categories. A 2026 follow-up ("Holistic Evaluation and Failure Diagnosis of AI Agents") decomposes judging into per-span verdicts with rationales and gets 38 percent relative category-F1 gain, 3.5x localisation and 12.5x joint accuracy with the same frontier model, showing judge scaffolding matters more than judge model. URLs: https://arxiv.org/abs/2505.08638 , https://arxiv.org/abs/2605.14865

Who&When (Zhang et al., ICML 2025 spotlight). 184 failure logs from 127 multi-agent systems (CaptainAgent on GAIA/AssistantBench, plus hand-built Magnetic-One), three-round expert annotation of the responsible agent and the decisive error step. Methods: all-at-once (whole log, one pass), step-by-step (incremental, stop at first error), binary search. Metrics: agent-level accuracy, step-level accuracy, step accuracy with tolerance of plus or minus 1 to 5. Best: 53.5 percent agent-level, 14.2 percent step-level; step-by-step is better at steps (25.5 vs 12.5 percent) but worse at agents; some methods are below random; ground truth in the prompt helps modestly. URL: https://arxiv.org/abs/2505.00212

Causal Agent Replay (CAR, CMU, June 2026). Treats attribution as intervention rather than reading: model the run as an SCM (state, stochastic action, observation, outcome), apply do-operations (do_resample, do_action, do_observation, do_context, do_policy), re-execute forward under the same policy many times, and measure the outcome-distribution shift; a point-of-commitment rule picks the latest step whose effect excludes zero, and Monte Carlo Shapley splits credit. Handles non-determinism by treating the model call as the one irreducible random input and reporting replay fidelity as an action-match rate. Real tools with side effects are explicitly out of scope (mocked tools only). URL: https://arxiv.org/abs/2606.08275

AgentRewardBench (McGill, COLM 2025). 1,302 web-agent trajectories over 5 benchmarks and 4 agents, 6 expert annotators (89.3 percent agreement on success) labelling success, side effects and repetition loops. Twelve judges scored by precision (primary, because judges feed rejection fine-tuning), recall and F1. Findings: screenshot-only input beat accessibility-tree-only and beat both combined; no judge wins everywhere; rule-based benchmark checks reach 55.9 percent recall on WebArena, i.e. they reject many genuinely successful runs. URL: https://arxiv.org/abs/2504.08942

WebJudge / Online-Mind2Web (OSU and Berkeley, 2025). Judge identifies key task points, filters the crucial screenshots, then judges; 85.7 percent agreement with humans and a 3.8 point average gap to human success rates. URL: https://arxiv.org/abs/2504.01382

Plan-RewardBench (2026). Pairwise trajectory preference over tool environments, four families (safety refusal, tool irrelevance, complex planning, error recovery); each pair judged in both orders. Best evaluator about 70 percent; accuracy collapses beyond 32k tokens, pairwise judges worst. Named biases: effort bias (rewarding unnecessary tool calls), stale constraint blindness (crediting an outdated plan after the user changed requirements), compliance inertia (missing a safety violation after benign turns). Human audit kappa above 0.7. URL: https://arxiv.org/html/2604.08178v2

BabelJudge (2026). Trajectory-length bias (longer trajectories rated higher at constant quality), position effects inside trajectories, and reliability variation across languages; recommends normalising trajectory presentation, multiple judges, and human validation before deployment. URL: https://arxiv.org/pdf/2606.22329

Judge's Verdict (2025/2026). 54 judges scored on RAG and agentic pipeline outputs: filter on Pearson r >= 0.80, then Cohen's kappa with a z-score against the human-human baseline (kappa 0.801). 23 models are "human-like" (|z| < 1, kappa 0.753 to 0.806), 4 "super-consistent" (kappa 0.804 to 0.813). URL: https://arxiv.org/html/2510.09738v1

False success detection (2026). Agents claim completion while the environment is incomplete in 45 to 48 percent of single-control tau tasks and 75.8 percent of self-reporting coding agents; LLM judges reached AUROC 0.54 to 0.65 because they key on confident language, while a TF-IDF detector reached 0.83 to 0.95 at 3,300x lower latency. Recommendation: cheap calibrated detectors for triage, state verification for truth. URL: https://arxiv.org/abs/2606.09863

Success provenance (AcquaBench, 2026). Matched CLEAN / GOLD / SHAM sources show that outcome-only scores can move 19 to 26 points purely because the answer leaked into a retrievable source, so a correct final answer does not prove the trajectory was legitimate. URL: https://arxiv.org/html/2607.24054v1

StepShield (2026). 9,429 code-agent trajectories; Early Intervention Rate = fraction of detected rogue trajectories where the alert fires within k steps of the divergence point. Regex guardrails hit 86 percent recall but fire on benign code 75 percent of the time with random timing (EIR 0.23 vs 0.24), a "forensics trap." URL: https://arxiv.org/abs/2601.22136

Judgment Labs "Agent Judge" (2026, vendor). Argues fixed-prompt judges fail on long traces because they cannot fit hundreds of tool calls in context, cannot verify state changes, and grade against stale rubrics; proposes an agentic judge with search over trace slices, verification against APIs and audit logs, and continuously rebuilt rubrics. URL: https://www.judgmentlabs.ai/blogs/agent-judge-solving-long-context-evaluations

Anthropic Bloom and Petri (Dec 2025). Bloom turns a behaviour spec into a suite via understanding, ideation, rollout (simulated user and tools), judgment (judge plus meta-judge); separated a seeded model organism from production in 9 of 10 cases. Petri is the broad, many-behaviour auditor. Both are transcript-level judges with a simulated environment rather than trace graders for production. URL: https://www.anthropic.com/research/bloom

### 3.4 Process reward models for agents (step scores learned rather than prompted)

AgentPRM (Fudan, Nov 2025). Redefines the step reward as promise (proximity to goal) and progress rather than correctness; labels via TD estimation with GAE instead of Monte Carlo rollouts; over 8x more compute-efficient; improves with test-time compute on WebShop and browser tasks. URL: https://arxiv.org/abs/2511.08325

ToolPRMBench (Jan 2026). 987 samples (542 train, 445 test) from ToolTalk, GTA, BFCL and ToolSandbox; each case = history, correct action, plausible incorrect alternative, tool metadata; offline (golden prefix) and online (free rollout) sampling; labels verified by three-LLM majority vote with human review of borderline cases (96 percent agreement). Metrics: step-level pairwise discrimination accuracy and first-error identification. Best: Claude 4.5 Haiku 75.1 percent, GPT-5 74.4 percent, ToolPRM-GRPO 78.6 percent among open models. URL: https://arxiv.org/abs/2601.12294

AgentProcessBench (Mar 2026). Ternary step labels with error propagation; StepAcc and FirstErrAcc; positive-label bias and neutral-step confusion are the main judge failures; Gemini 3 Flash 81.6 percent StepAcc vs Qwen3-30B 68.5 percent; inter-annotator agreement 89.1 percent; text-only. URL: https://arxiv.org/html/2603.14465v1

AgentEval DAG (2026). Step-level tool-selection (1 to 5 LLM score), argument-correctness and execution-quality metrics over a DAG of step dependencies, with explicit error-propagation tracking so upstream failures are not hidden by an eventual success. URL: https://arxiv.org/pdf/2604.23581

Related step-level scoring for training: divergence-point preference learning (2026) aligns a failed and a successful trajectory (state signature Jaccard >= 0.4 or shared message prefix), takes the first differing action as a chosen/rejected pair, and filters with annotations; on tau2-bench "chosen reward positivity" predicted good checkpoints better than accuracy or margin. URL: https://arxiv.org/html/2606.23112

### 3.5 Judge calibration against human labels (practitioner consensus)

- Hamel Husain (Evals FAQ, 2025 to 2026): judges should be binary pass/fail on one named failure mode, not Likert dashboards ("Helpfulness 4.2"); validate on a held-out human-labelled set and report TPR and TNR; label 100+ fresh traces per cycle; for agents, first ask "did we meet the user's goal", then use transition failure matrices and focus on the first upstream failure because later errors cascade; do error analysis (open coding, axial coding into a taxonomy, iterate until about 20 new traces add no category). URL: https://hamel.dev/blog/posts/evals-faq/
- Eugene Yan, "Product Evals in Three Simple Steps" (Nov 2025): label a small set, align the LLM evaluator to it, then run the harness on every config change; AlignEval tooling. URL: https://eugeneyan.com/writing/product-evals/
- Cohen's kappa thresholds in circulation for 2026 production use: judge-to-human kappa above 0.6 acceptable, above 0.8 strong; re-sample monthly for judge drift; always swap positions for pairwise; never let a model judge only its own family's outputs. URLs: https://arxiv.org/html/2510.09738v1 , https://galileo.ai/blog/calibrate-llm-judge-human-annotations
- OpenAI evaluation best practices (2026): start with the strongest available grader, validate against human labels, "use pairwise comparison or pass/fail for more reliability", chain-of-thought in the grader, task-specific rubrics that name tool selection, argument extraction and handoffs, mix production data with expert-curated cases, and grow the set over time to avoid overfitting. URL: https://developers.openai.com/api/docs/guides/evaluation-best-practices

---

## 4. Turning production logs into eval datasets; replay and side effects

### 4.1 Mining and curating traces

Fireworks, "Turning Production Logs into Evaluation Datasets" (2025). Pipeline: ingest traces from Langfuse/Braintrust, embed (512-d), UMAP to 5-d for clustering and 2-d for viewing, HDBSCAN clusters (example: 100 support queries into 5 clusters of 14 to 29), then stratified sampling of one representative trace per cluster to get coverage without redundancy. Golden labels are left to a later labelling step. URL: https://fireworks.ai/blog/Turning-Production-Logs-into-Evaluation-Datasets

Langfuse golden-dataset guidance: build from real traces first, synthetic second; maintain with schema validation, dedup and item versioning; the standard loop is "collect failing production traces, turn them into dataset items, reproduce the failure in an experiment, fix it, and keep the item as a permanent regression test." URLs: https://langfuse.com/resources/engineering/golden-dataset-evaluation , https://langfuse.com/resources/engineering/ai-agent-evaluation

LangChain run / trace / thread framework (2025 to 2026): run-level (single tool call or LLM call), trace-level (one full agent turn: final response, trajectory quality, state changes; "score what matters: the outcome and the quality of the decisions that produced it"; reserve strict ordering for safety-critical sequences), thread-level (multi-turn intent and outcome). Production traces feed Insights, which shape datasets, which power evals; one-click trace-to-dataset. Their 2026 State of Agent Engineering survey (1,340 respondents): 89 percent have observability, 52 percent run offline evals, 37 percent online evals; quality is the top production blocker (32 percent). URLs: https://www.langchain.com/resources/agent-evals , https://www.langchain.com/state-of-agent-engineering

OpenAI trace grading and agent evals (2025 to 2026): traces from the Agents SDK are graded in the dashboard with structured criteria "to identify workflow-level issues", filterable by model, date range and tool calls; recommended progression is traces first (debugging), then datasets (repeatable benchmarking); Evals 2026 supports multi-turn traces, tool-call grading (right tool, right arguments) and final-answer grading in one YAML suite with model, exact-match and Python graders. URLs: https://developers.openai.com/api/docs/guides/trace-grading , https://developers.openai.com/api/docs/guides/agent-evals

Anthropic (Jan 2026): mine the bug tracker and support queue; start from the manual checks already done before each release; keep the first set to 20 to 50 tasks.

Hamel Husain: sample real traces, write open-ended notes, cluster into a failure taxonomy, count frequency; the taxonomy drives which judges to build. Test sets for CI are "small (100+ examples) and purpose-built"; production monitoring uses reference-free judges on sampled live traces.

AlphaEval (2026): 94 production tasks from seven companies, evaluating whole agent products (Claude Code, Codex) with a requirement-to-benchmark construction framework mixing LLM judges, reference metrics, formal verification, rubrics and UI tests; shows product-level differences invisible at model level. URL: https://arxiv.org/abs/2604.12162

### 4.2 Using a stronger model's recorded output as the reference (distillation-style eval)

NVIDIA Data Flywheel Blueprint (2025): the teacher's production prompt/response logs become the evaluation dataset with no manual labels; metrics are `function_name_and_args_accuracy` (does the candidate's call match the teacher's call) and `tool_calling_correctness` via NeMo Evaluator's LLM-as-judge for semantic equivalence; a fine-tuned Llama 3.2 1B reached "98 percent of the tool-calling accuracy of the original 70B model" and promotion thresholds are left to the operator. URL: https://developer.nvidia.com/blog/build-efficient-ai-agents-through-model-distillation-with-nvidias-data-flywheel-blueprint/

Agent distillation papers (2025 to 2026, e.g. "Distilling LLM Agent into Small Models with Retrieval and Code Tools", SOD step-wise on-policy distillation) use the teacher's trajectories as SFT targets and report step agreement with the teacher as an intermediate metric, but always validate on environment success because teacher agreement is not success. URLs: https://arxiv.org/abs/2505.17612 , https://arxiv.org/html/2605.07725v1

Known pitfalls of teacher-as-reference: the teacher's own errors become the gold (needs SME review of a sample, per AWS FMEval and Label Studio guidance); self-preference and perplexity bias when the same family judges (Wataoka et al.); answer flipping across teacher versions destabilises the reference; agreement with the teacher on a prefix says nothing about the student's trajectory once it diverges (Replay Gap); success provenance must be audited (AcquaBench). Practical mitigations seen in the field: judge from a different model family than the agent, record the teacher's reasoning and use it as rubric context rather than as the answer, prefer state or milestone references over action references, and hold out a human-labelled slice to measure the teacher-reference's own precision.

### 4.3 Record and replay, mocking observations, non-determinism and side effects

langchain-replay (sixty-north, 2025). Records the LLM's decisions (tool name, arguments, text) rather than HTTP traffic; on replay it yields recorded decisions while actually executing tools, so tool code paths are exercised without API cost; recorded tool inputs are dispatched verbatim, so a drifted decision fails the test. Documented failure modes: timestamps or UUIDs baked into prompts, stale temporary paths, assertions on values that should change; fixes are design-level (stable inputs, no per-run values in LLM-visible context). URL: https://github.com/sixty-north/langchain-replay

Deterministic replay writeups (Helicone session replay, agiflow, TianPan, Sakura Sky, 2025 to 2026). Common pattern: log every LLM call, tool response and timestamp; replay 1,000+ traces against a new prompt or model in minutes at near-zero cost by serving recorded tool observations; mock environments extend this by generating plausible responses for off-trace actions and by injecting failures (timeouts, 429s, malformed payloads) absent from happy-path logs. URL: https://docs.helicone.ai/guides/cookbooks/replay-session

The Replay Gap (Aug 2026). The quantitative warning against the above when the policy changes: replay assumes an open loop, but the trajectory is a closed loop through the environment; the fix is branched live rollouts (fork at step k, rebuild the environment, continue each branch, compare to same-model control forks to isolate serving noise). Late forks and downgrades diverge less than early forks and upgrades. URL: https://arxiv.org/abs/2608.08239

Causal Agent Replay (June 2026): the same closed-loop logic used for attribution; mocked reproducible tools only, real side effects out of scope.

AgentRR, "Get Experience from Practice: LLM Agents with Record & Replay" (2025). Records interaction traces and internal decisions, summarises them into structured "experiences" (workflow plus constraints), and replays them to guide later similar tasks; motivated by reliability, privacy and cost rather than evaluation, but the trace schema is reusable for eval. URL: https://arxiv.org/abs/2505.17716

Environment snapshots as the answer to side effects: tau-bench replays reference actions on a fresh DB to compute the target state; AppWorld and MCPMark start every task from a curated initial state in a sandbox and verify the final state programmatically; Braintrust recommends versioning each environment with browser config and fixture/snapshot id and replaying failed live-site tasks against the last pinned version to separate agent regressions from site drift; Proxy State-Based Evaluation removes the deterministic backend entirely by inferring a proxy state with an LLM.

Trajectory graphs for pre-execution diagnosis (2026): deduplicate actions into nodes with observations as edges, train a GNN to classify steps into six error types (illegal action, repeated action, incorrect target, precondition not met, condition met but action not taken, none), 5 to 10 points above text classifiers with fewer samples. URL: https://arxiv.org/html/2607.27443v1

---

## 5. 2025 to 2026 specifically: benchmarks, vendor guidance, reliability science

- Anthropic, Demystifying evals for AI agents (Jan 2026): definitions above; outcome grading first; pass^k for reliability; balanced positive/negative task sets. URL: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Towards a Science of AI Agent Reliability (Feb 2026, ICML 2026): twelve metrics across consistency, robustness, predictability and safety computed from repeated trials of 15 models on two benchmarks; capability gains "have only yielded small improvements in reliability." URL: https://arxiv.org/abs/2602.16666
- Towards More Standardized AI Evaluation: From Models to Agents (Feb 2026): reframes evaluation as a measurement discipline for non-deterministic systems; static benchmarks and aggregate scores "introduce silent failure modes." URL: https://arxiv.org/abs/2602.18029
- OpenAI Evals 2026: first-class multi-turn traces, tool-call grading, YAML suites, trace grading in the dashboard. URLs above.
- LangChain State of Agent Engineering 2026: numbers above.
- BFCL v4 agentic tracks; tau2-bench (2025) and tau3-bench (2026); MCPMark, MCP-Universe, MCP-Bench, MCP-AgentBench (2025); Gaia2/ARE (Sept 2025); AlphaEval (2026); Online-Mind2Web and WebJudge (2025).
- Judge and PRM benchmarks: TRAIL (May 2025), Who&When (May 2025), AgentRewardBench (2025), AgentPRM (Nov 2025), ToolPRMBench (Jan 2026), StepShield (Jan 2026), AgentProcessBench (Mar 2026), Plan-RewardBench (Apr 2026), Holistic Evaluation and Failure Diagnosis (May 2026), Causal Agent Replay (June 2026), false success characterisation (June 2026), BabelJudge (June 2026), AcquaBench success provenance (July 2026), The Replay Gap (Aug 2026).
- Anthropic Bloom and Petri (Dec 2025) for behavioural transcript audits.
- Hamel Husain Evals FAQ (updated Jan 2026) and Evals Skills for Coding Agents; Eugene Yan Product Evals (Nov 2025) and "An LLM-as-Judge Won't Save The Product, Fixing Your Process Will."

---

## Synthesis

### What the field agrees on for scoring a single agent step

1. Decompose the step into separately scored parts: (a) did the agent act or not (tool call vs reply vs clarification), (b) which tool, (c) arguments, (d) the resulting state or observation, (e) the reasoning or message. BFCL, ToolTalk, Mind2Web, Braintrust, DeepEval, Arize, OpenAI and AgentEval all score these independently and report them independently.
2. Tool name is matched exactly (after normalisation); arguments are matched by type-aware, normalised comparison against a set of acceptable values, never by raw string equality. Free-text arguments get semantic matching (embedding threshold in ToolTalk, LLM equivalence in Arize and Vertex). Required parameters must be present; hallucinated parameters fail the step. Lists are ordered unless declared otherwise; dict key order is ignored. Parallel calls are matched as a set, all-or-nothing.
3. Prefer a set of acceptable actions, a milestone, or a resulting state over a single gold action, because single-gold scoring under-counts correct behaviour by double-digit percentages (AitW, AgentRewardBench, OpenCUA, ACEBench candidate pools, ToolSandbox milestones).
4. Distinguish action tools (side effects) from read tools. An incorrect write is scored separately (ToolTalk incorrect action rate, tau2 READ/WRITE breakdown, AppWorld collateral damage) and usually dominates the verdict.
5. Step labels should allow a neutral class. Exploratory or redundant steps are not errors but are not progress either (AgentProcessBench ternary labels, BFCL v3 subset matching, AgentBoard progress rate that only rises when a subgoal is reached).
6. Judge scores must be binary or pairwise, criterion-specific, produced with the judge in a different family from the agent where possible, swapped for position, and validated on a held-out human-labelled set with TPR/TNR or kappa (target kappa above 0.6, ideally 0.8) before use. Rubrics and checklists raise agreement measurably (TICK +5.8 points, HealthBench macro F1 0.71 at physician level).
7. Teacher-forced step accuracy is a diagnostic, not a success estimate. It measures "would this model have emitted an acceptable action on this recorded prefix" and is valid only on the recorded distribution; it cannot measure recovery or off-trajectory behaviour, and it should be paired with at least one outcome-level signal (state check, milestone, pass^k on live or branched rollouts).
8. Locate the first error, not every error. Who&When, TRAIL, AgentProcessBench (FirstErrAcc), Hamel's "first upstream failure" and StepShield's divergence point all converge on the first decisive step as the unit that matters for both debugging and training labels, with a tolerance window of a few steps.

### The five biggest open problems

1. Step localisation by LLM judges is near random. Best step-level attribution on Who&When is 14.2 percent, TRAIL joint accuracy 11 percent, and first-error accuracy degrades fastest with trajectory length. Scaffolded per-span judges and causal replay help, but no method yet produces step labels reliable enough to train on without human review.
2. The replay gap. Any evaluation that scores a new model or prompt on recorded prefixes assumes an open loop. Measured divergence after a model swap is 61 to 94 percent of post-fork actions and about 3 percent of replayed states remain valid. Cheap replay and faithful evaluation are currently in tension; branched live rollouts are correct but require re-executable, side-effect-free environments that production systems rarely have.
3. Reference quality and provenance. Rule-based references under-report success (55.9 percent recall on WebArena), single-gold actions penalise valid alternatives, teacher-model references import the teacher's mistakes and self-preference bias, and correct outcomes can be caused by leaked information rather than competence (AcquaBench). There is no accepted protocol for auditing a reference set's own precision and recall.
4. Judges on long, stateful traces. Accuracy collapses beyond roughly 32k tokens, judges reward effort and length, miss stale constraints and safety violations after benign turns, and cannot verify state they cannot access; false-success claims fool LLM judges (AUROC 0.54 to 0.65) while trivial lexical detectors do better. Agentic judges with state verification exist as products and prototypes, not as validated standards.
5. Reliability, not accuracy, is the unmet requirement. pass^k, consistency, robustness and predictability metrics show capability gains translating into small reliability gains; evals from production still cover a minority of teams (52 percent offline, 37 percent online), and there is no shared standard for how many trials, which seeds, which environment snapshots and which judge calibration evidence must accompany a reported agent score.
