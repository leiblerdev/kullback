# LLM-as-a-judge and reward models for agent trajectories (2024-2026)

Research sweep, 2026-08-26. Source: web research agent. Topic 4 of 6.

## 1. Agent-specific judge frameworks and benchmarks

**Agent-as-a-Judge (Zhuge et al., ICML 2025).** Gives the judge tools (graph, locate, read, search, retrieve, ask, memory, planning) so it inspects the artifact and action log rather than reading a transcript. On DevAI the agent judge reached 90.44% alignment with human consensus vs 60.38% for a plain LLM judge; individual humans 76-93%. Ablation: "ask only" 65.0%, +graph 76.0%, +read 82.2%, +locate 90.4%. Cost 2.28% of human evaluation ([arXiv 2410.10934](https://arxiv.org/abs/2410.10934)). Lesson: the alignment gain came from giving the judge tools to verify, not a better prompt.

**AgentRewardBench (COLM 2025).** 1,302 web-agent trajectories, expert-annotated for success, side effects, repetition; inter-annotator agreement 89.3%. 12 LLM judges tested; **no judge exceeded ~70% precision** (Claude 3.7 Sonnet 68.8%, GPT-4o 69.8%). Rule-based evaluators had 83.8% precision but only 55.9% recall. Feeding both screenshots and a11y trees hurt precision. Judges systematically overestimate success and "often accept flawed agent reasoning without verification" ([arXiv 2504.08942](https://arxiv.org/abs/2504.08942)).

**TRAJECT-Bench (Oct 2025).** 1,228 executable tools, 5,670 queries. Diagnostics: tool selection, argument correctness, dependency/order satisfaction. Failure modes: similar-tool confusion, parameter-blind selection ([arXiv 2510.04550](https://arxiv.org/abs/2510.04550)).

**Step-level agent benchmarks (2026).** AgentProcessBench: 8,509 human-labeled steps with a ternary label (+1 correct/effective, 0 neutral/exploratory, -1 incorrect) and an error-propagation rule. Human kappa 0.767. Best judge 81.6% step accuracy but only 65.8% first-error accuracy; all judges biased toward positive labels ([arXiv 2603.14465](https://arxiv.org/html/2603.14465v1)). AgentProp-Bench: substring judging at chance (kappa 0.049), best single LLM judge kappa 0.567 vs human-human 0.835; parameter errors propagate to wrong final answers with p ~ 0.62; several agents fabricate tool executions ([arXiv 2604.16706](https://arxiv.org/abs/2604.16706)). WebStep: process metrics separate agents whose success rates are indistinguishable ([arXiv 2606.15673](https://arxiv.org/abs/2606.15673)).

**Error localization is hard.** TRAIL (Patronus): 148 traces, 841 errors; best model 11% ([arXiv 2505.08638](https://arxiv.org/abs/2505.08638)). Online Agent-as-a-Judge (2026): judge generates situations inside the environment; human agreement 70% vs 33% offline ([arXiv 2606.08200](https://arxiv.org/html/2606.08200v1)).

## 2. Process reward models: from math to agents

- PRM800K / Let's Verify Step by Step ([arXiv 2305.20050](https://arxiv.org/abs/2305.20050)); Math-Shepherd MC labels ([arXiv 2312.08935](https://arxiv.org/abs/2312.08935)).
- Qwen "Lessons" (2025): MC-estimated labels are inferior to LLM-judge or human labels; Best-of-N is biased; fix is consensus filtering ([arXiv 2501.07301](https://arxiv.org/abs/2501.07301)).
- ThinkPRM / GenPRM: generative verifiers ([arXiv 2504.16828](https://arxiv.org/abs/2504.16828)). ReasonEval: validity and redundancy; answer accuracy gains do not imply better step quality ([arXiv 2404.05692](https://arxiv.org/abs/2404.05692)).
- Agents: Agent Q ([arXiv 2408.07199](https://arxiv.org/abs/2408.07199)); AgentPRM (Fudan) "promise and progress" ([arXiv 2511.08325](https://arxiv.org/abs/2511.08325)); DataPRM ternary reward distinguishing recoverable from irrecoverable errors, because binary PRMs penalize legitimate exploration ([arXiv 2604.24198](https://arxiv.org/abs/2604.24198)). Lesson: agent PRMs need a neutral/exploratory class.

## 3. Judge reliability, biases, calibration

- MT-Bench: GPT-4 >80% agreement with humans; position, verbosity, self-enhancement biases ([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)).
- Position bias is systematic and driven by the quality gap between candidates ([arXiv 2406.07791](https://arxiv.org/abs/2406.07791)); even rubric option ordering induces bias ([arXiv 2602.02219](https://arxiv.org/pdf/2602.02219)).
- "Judging the Judges": only the largest judges align reasonably; leniency bias ([arXiv 2406.12624](https://arxiv.org/abs/2406.12624)). "Reliability without Validity" (2026, 541K judgments): kappa 33-41 points below percent agreement; rankings shift up to 14 places ([arXiv 2606.19544](https://arxiv.org/abs/2606.19544)).
- Self-preference: LLMs rate lower-perplexity text higher ([arXiv 2410.21819](https://arxiv.org/abs/2410.21819)); self-recognition correlates with self-preference ([arXiv 2404.13076](https://arxiv.org/pdf/2404.13076)).
- Length-controlled AlpacaEval ([arXiv 2404.04475](https://arxiv.org/abs/2404.04475)). CALM 12-bias taxonomy incl. "fallacy-oversight" (ignoring logical errors when the final answer looks right) ([arXiv 2410.02736](https://arxiv.org/html/2410.02736)).
- Pairwise is more discriminative but flips in ~35% of cases under distractor features vs 9% for absolute scores ([arXiv 2504.14716](https://arxiv.org/abs/2504.14716)).
- Clear criteria matter most; CoT in the judge adds little when criteria are clear; sampling improves alignment ([arXiv 2506.13639](https://arxiv.org/html/2506.13639v1)). Criteria drift (EvalGen) ([arXiv 2404.12272](https://arxiv.org/abs/2404.12272)).

## 4. Rubric, checklist, and reference-guided grading

- OpenAI RBR: binary propositions ([arXiv 2411.01111](https://arxiv.org/abs/2411.01111)). HealthBench per-conversation rubrics ([OpenAI](https://openai.com/index/healthbench/)). PaperBench hierarchical rubrics, SimpleJudge F1 0.83 ([arXiv 2504.01848](https://arxiv.org/pdf/2504.01848)). OpenAI grader API ([docs](https://developers.openai.com/api/docs/guides/graders)).
- Rubrics as Rewards (Scale): rubrics reduce variance across judge sizes ([arXiv 2507.17746](https://arxiv.org/abs/2507.17746)). Checklists beat reward models (RLCF) ([arXiv 2507.18624](https://arxiv.org/abs/2507.18624)). Rubicon veto rubrics ([arXiv 2508.12790](https://arxiv.org/abs/2508.12790)).
- Rubric reward hacking (2026): policies satisfy compound criteria partially; gains vanish when re-judged by a stronger cross-family panel ([arXiv 2605.12474](https://arxiv.org/abs/2605.12474)).

## 5. Judging reasoning quality and CoT faithfulness

**There is no validated method for judging from text alone whether a reasoning step caused the action or is post-hoc rationalization.**
- Lanham 2023, Turpin 2023: CoT is often post-hoc ([arXiv 2307.13702](https://arxiv.org/pdf/2307.13702)).
- Anthropic 2025: Claude 3.7 Sonnet verbalized hints 25% of the time, R1 39%; reward hacks acknowledged in CoT <2% ([Anthropic](https://www.anthropic.com/research/reasoning-models-dont-say-think)).
- "CoT in the wild": post-hoc rationalization up to 13% for production models ([arXiv 2503.08679](https://arxiv.org/abs/2503.08679)). FaithCoT-Bench ([arXiv 2510.04040](https://arxiv.org/abs/2510.04040)).
- **"Gaming the Judge" (2026)**: manipulating only the reasoning text (actions and observations unchanged) inflated judge false positives by up to 90% across 800 web trajectories. Recommendation: verify reasoning claims against observable evidence ([arXiv 2601.14691](https://arxiv.org/abs/2601.14691)).
- OpenAI CoT monitoring: monitors catch intents action-only monitors miss, but optimizing against the monitor produces obfuscated hacking ([OpenAI](https://openai.com/index/chain-of-thought-monitoring/)).

What is validated: step validity/redundancy on math, and step effectiveness relative to observations on agent tasks, at ~65-80% judge accuracy at locating the first error.

## 6. Vendor and lab guidance, concrete metrics

- **Anthropic (Jan 2026)**: "It's often better to grade what the agent produced, not the path it took"; exact tool sequences are "too rigid"; give the judge an "Unknown" escape; isolated judge per rubric dimension; a good task is one where two experts independently agree ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Use a different model to grade than to generate ([docs](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)).
- **Google Vertex**: trajectory_exact_match, in_order_match, any_order_match, precision, recall, single_tool_use ([docs](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-agents)).
- **LangSmith agentevals**: `trajectory_match_mode` strict/unordered/subset/superset; `tool_args_match_mode` exact/ignore/subset/superset/per-tool overrides ([GitHub](https://github.com/langchain-ai/agentevals)).
- **Arize Phoenix**: judge receives tool calls, user input, tool schema, optional reference; outputs correct/incorrect; templates for Tool Selection, Tool Invocation, Tool Response Handling ([docs](https://arize.com/docs/ax/evaluate/evaluators/trace-and-session-evals/trace-level-evaluations/agent-trajectory-evaluations)).
- **DeepEval**: Task Completion (referenceless), Tool Correctness (deterministic, optional params/output matching) ([docs](https://deepeval.com/docs/metrics-tool-correctness)).
- **Ragas**: ToolCallAccuracy = argument accuracy x sequence aligned; ToolCallF1; AgentGoalAccuracy ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/)).
- **Galileo**: Tool Selection Quality, majority vote over multiple CoT evaluations, covers "no tool needed" ([docs](https://docs.galileo.ai/concepts/metrics/agentic/tool-selection-quality)).

## 7. Recommended judge design: candidate vs reference frontier trajectory

**Framing principle.** The reference trajectory is evidence of one valid path, not the answer key. Use it to (a) confirm solvability, (b) derive a minimal necessary-action set and expected end state, (c) calibrate "what good looks like", never as a strict template.

**Layered design (deterministic first, LLM last):**
1. **Outcome layer (code):** end-state check derived from the reference run. Primary score.
2. **Necessary-action layer (code):** causally required tool calls from the reference; `superset` / `trajectory_recall` semantics with relaxed per-tool arg matching. Report precision / step-efficiency separately as a diagnostic.
3. **Grounding layer (LLM, per-step, observation-anchored):** judge sees (prior observation, action, next observation) plus the task; binary checklist: action consistent with last observation? conclusion follows from tool output? claims a result it did not observe? Ternary label (+1 / 0 / -1) with error propagation.
4. **Rubric layer (LLM, pointwise, reference-informed):** 5-12 binary criteria from task spec plus reference; isolated judge call each, reason-then-verdict, "Unknown" option; veto criteria for safety/side effects.
5. **Pairwise layer (LLM, tie-break only):** both orders, reasoning text stripped, only actions plus observations plus final artifact; inconclusive if orderings disagree.

**Judge hygiene:** different model family than both candidate and reference; randomize rubric order; 3-5 samples with majority vote; report Cohen's kappa against a 100-200 item human calibration set; track judge false-positive rate on deliberately broken trajectories (expect a ~70% precision ceiling).

**Use the judge for:** ranking on outcome plus grounding; localizing first divergence; flagging fabricated observations, side effects, unsafe actions.

**Do NOT use the judge for:** deciding the reasoning text is sound as an end in itself; penalizing path deviation when the outcome is correct; grading verbosity; absolute Likert scores across runs; single-ordering pairwise verdicts; plan adherence when the reference plan differs. If a step's soundness matters, make it verifiable (run the code, re-query the tool, check the observation).
