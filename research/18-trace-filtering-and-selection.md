# How others filter, deduplicate, and select agent traces before using them as eval or training data

Source: research agent, 2026-08-26. Web search was over budget, so all sourcing is via direct fetches of arXiv full text, dataset cards, and vendor docs. Two sources (Nomic Atlas duplicate-detection docs and Lilac docs) could not be reached and are noted as gaps.

---

## 1. Trajectory datasets and their filter pipelines

### Toucan-1.5M (MCP tool-agentic data)
Source: https://arxiv.org/html/2510.01179

- Environment filter first: about 2,800 MCP servers crawled, 871 remained after requiring remote accessibility (30.6% retained), then servers needing third-party credentials were removed and functional testing left 495 servers (2,000+ tools).
- Query-level LLM scoring on six 1 to 5 dimensions: Tool Selection Difficulty, Tool Selection Uniqueness, Question Quality, Scenario Realism, Verifiable, Stability.
- Trajectory rule filters (Stage 5), quoted: "exclude trajectories that fail to start the agent or connect successfully with remote MCP servers"; trajectories that "do not contain tool calls"; those with "failures in tool responses"; those that "contain local file system paths"; and a check of "whether the trajectory uses the required tools specified by the task in the correct sequence".
- LLM judge (GPT-OSS-120B): Completeness ("fulfills the user's request end-to-end") and Conciseness ("minimum necessary steps and verbosity"), plus desired-tool-usage percentage and order correctness.
- SFT-subset thresholds: "question quality and scenario realism scores of 5, response completeness and conciseness scores of at least 4, and desired tool use percentage of 1.0". Result: 119.3K SFT instances (28.3K core + 40K irrelevance + 15.8K diversify + 35.2K multi-turn) out of 1,527,259 released trajectories, so the strict subset is about 7.8% of the release.
- No explicit dedup step described. Per-stage removal counts for trajectories are not published.

### Nemotron-Agentic-v1 (NVIDIA)
Source: https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1

- 335,122 samples (19,028 interactive-agent + 316,094 tool-calling), all LLM-simulated (user, agent, tool environment).
- Filter, quoted: "We employ a separate language model as a judge to score and filter the data, removing trajectories where any step appears inconsistent, incoherent, or that use the incorrect tools." The tool-calling subset adds "turn-level judgements". No removal fractions published.

### APIGen (single-turn function calling, xLAM-60K)
Source: https://arxiv.org/html/2406.18518

- Three hierarchical checks: (1) format: output "must strictly follow a JSON format with the 'query' and 'answer' fields"; (2) execution: "Unsuccessful executions are filtered out", covering "argument type errors, invalid parameters, runtime errors, timeout, syntax errors, missing arguments"; (3) semantic (LLM): call aligns with query and has proper arguments, chosen from available functions, "whether the number of function calls matches the user's intent", "whether the execution results contain errors or indicate unsuccessful function execution", results relevant to the query.
- Published drop counts per 40,000 generated:

| Generator | Passed | Fail format | Fail execution | Fail semantic | Pass rate |
|---|---|---|---|---|---|
| DeepSeek-Coder-33B | 13,769 | 4,311 | 15,496 | 6,424 | 34.42% |
| Mixtral-8x7B | 15,385 | 3,311 | 12,341 | 7,963 | 38.46% |
| Mixtral-8x22B | 26,384 | 1,680 | 5,073 | 6,863 | 65.96% |
| DeepSeek-V2-Chat | 33,659 | 817 | 3,359 | 2,165 | 84.15% |

- Human audit of the survivors: "Only 28 out of the 600 inspected samples have minor issues" (95.3% clean). Execution failure is the dominant filter for weaker generators; semantic failure dominates for strong ones.

### APIGen-MT (multi-turn, tau-bench style)
Source: https://arxiv.org/html/2504.03601

- Blueprint stage: structural validation of `<thought>/<instruction>/<actions>/<outputs>` plus "API names, argument names, and data types"; then "a committee of diverse LLM judges" with "a majority voting strategy" on Correctness, Completeness, Satisfaction, Creativity; then a semantic review requiring "an average score above a predefined threshold". Failed blueprints get up to 3 (retail) or 5 (airline) reflection iterations. Blueprint acceptance: 70% with feedback loop vs 28% without.
- Trajectory stage, quoted: "Only trajectories achieving the task goal are retained. Success is determined by comparing the final environment state to a_gt and the agent's final responses to o_gt." Simulated human uses Best-of-N (N=4) plus self-critique. Trajectory success rate 67%, so about one third of rollouts are dropped. Released: 5K trajectories averaging 6 user turns.

### xLAM
Source: https://arxiv.org/html/2409.03215

- Unified format, then quality verification targeting four error classes: undefined function calls (function not in the provided list), incorrect argument types, argument hallucination (LLM judge detects "mismatch between the arguments and the intended query or prior observations"), and low-quality reasoning. Plus 60,000 APIGen samples from 3,673 APIs. No fractions published.

### SWE-smith
Source: https://arxiv.org/html/2504.21798

- Task validity: "only keep patches that break one or more existing, passing tests". Yield by bug source: LM Modify 56.0%, LM Rewrite 35.0%, PR Mirror 33.8%, Procedural 40.2%, Combine 96.9%; overall 50.1% yield, 50,137 valid instances.
- Trajectories: 17,906 attempts on 8,686 instances produced 6,457 resolved trajectories (36% resolve rate); "We limit the number of times any SWE-smith task instance is represented in the training set to 3 trajectories", leaving 5,016 (about 22% of drops at this cap step).
- Difficulty by resolve rate: for LLM-rated difficulty 1/5/9, expert resolve rates were 58.6%, 41.0%, 17.0%. Training on difficulty buckets 2/4/6/8 gave 12.4/10.8/13.6/12.2% on SWE-bench Verified, so difficulty filtering did not clearly help.

### SWE-Gym and OpenHands trajectories
Sources: https://arxiv.org/html/2412.21139 , https://www.openhands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model

- Rejection sampling keeps only resolved runs: 491 successful trajectories (19 at temp 0, 87 more at temp 0.2 to 0.8, 299 more later); each averages 19 turns and about 19,000 tokens.
- "task success probability follows a long-tail distribution", so they apply "per-instance capping" and "a threshold of 2 achieves a good balance".
- Verifier training deliberately keeps failures: 443 + 875 successful trajectories balanced with "the same amount of unsuccessful trajectories from each subset (1,318 each)", 2,636 total.
- OpenHands LM 32B: fine-tuned "on examples that were resolved successfully"; the blog notes the model is "prone to generating repetitive steps", a hint that repetition was not filtered.

### AgentScaler (Alibaba, "Towards General Agentic Intelligence via Environment Scaling")
Source: https://arxiv.org/html/2509.13311

- Three-stage funnel: (1) Validity control "removes invalid interaction trajectories to ensure well-formed alternating user assistant exchanges" plus "an n-gram-based filtering procedure to eliminate severely repetitive reasoning segments"; (2) Environment state alignment "retains only those trajectories whose final database state matches the golden state"; (3) Function-calling exact match: "preserved only if the sequence of invoked tools and arguments exactly matches the overall intent".
- Notable non-filter: "we do not filter out trajectories in which tool calls return errors ... such trajectories may still accomplish the intended goal." No counts published.

### Agent-World
Source: https://arxiv.org/html/2604.18292

- Pass@5 consistency gate: "deploying a ReAct agent to solve it 5 separate times ... We retain the task only if the agent successfully reaches a consistent answer in at least two independent runs" (same for programmatic tasks: "at least 2 successful runs"). Tool validity: compiles, Acc > 0.5 on its test set, environment has at least one valid tool and case. 1,978 retained environments, 19,822 tools, 40K cold-start SFT trajectories. Whether SFT keeps all passing rollouts or a capped number per task is not stated.

### AutoForge
Source: https://arxiv.org/html/2512.22857

- Success is final-state comparison ("evaluate task completion based on the final environment state rather than the tool sequence"). Training uses "DAPO's dynamic sampling mechanism to exclude any samples where all trajectories are either fully correct or fully incorrect" (with 8 rollouts, that is the 0/8 and 8/8 buckets). 10 environments, 1,078 tasks; removal counts not disclosed.

### Envs-FORGE
Source: https://arxiv.org/html/2608.14312

- Seed pass rate p̂_i = mean verifier reward over rollouts. Frontier scoring targets τ = 0.5 with σ = 0.2 (Gaussian around 50% pass rate); too-easy seeds are made harder, too-hard seeds are reduced, frontier seeds are diversified. Gate: "An environment enters the training pool only if its oracle solution obtains reward 1 under the generated tests." Each method exports exactly 100 gold-verified bundles.

### SPADE
Source: https://arxiv.org/html/2608.19197

- Difficulty band reward: "pays environments whose Reasoning Agent win rate falls in a target band [0.4,0.6] and decays linearly outside it"; 16 plays with and 16 without hints per environment; environments validated for "syntactic correctness and executability", with "a deterministic reset gate that exercises every success criterion under several seeds, and an LLM check that every success criterion can be met by some tool."

### ToolACE
Source: https://arxiv.org/html/2409.00920

- Rule layer: API definition must comply with JSON Schema; executability (API name in tool list, all required params present, regex checks on parameter formats); dialog correctness; consistency ("Check if the API names in the function call and the tool response are consistent").
- Model layer: hallucination detection (parameter values "not mentioned in either the user query or the system prompt"), consistency validation, and tool-response verification against the API definition. No fractions published.

### ToolBench / ToolLLM
Sources: https://arxiv.org/html/2307.16789 , https://arxiv.org/html/2307.16789v2

- API filter: 10,853 tools / 53,190 APIs down to 3,451 tools / 16,464 APIs (31% of APIs kept); "APIs that consistently exhibit a long response time are omitted", and "APIs with low-quality responses, such as HTML source codes or other error messages" are dropped.
- Instruction filter: drop instructions "with the hallucinated relevant APIs", leaving ~200k pairs; then "only retain those passed solution paths" from DFSDT, giving 126,486 pairs (about 63% of instructions).
- Observation truncation as a data rule: "When the API response length exceeds 1024 tokens, we compress the response ... If the compressed response is still longer than 1024, we only retain the first 1024 tokens."

### Magpie-style dedup and quality filters
Sources: https://arxiv.org/html/2406.08464 , https://arxiv.org/html/2406.08464v2

- 4M generated, 300K selected. Dedup metric: "minimum neighbor distance in the embedding space" with all-mpnet-base-v2 and FAISS; the published filter is Min Neighbor Distance > 0 (i.e., drop exact embedding duplicates), not a MinHash threshold.
- Magpie-Pro-300K-Filtered config: Input Quality >= average, Min Neighbor Distance > 0, reward > -12, then select the longest outputs. Eight filter axes are exposed (length, category, quality, difficulty, neighbor distance, reward, reward difference).

### AgentInstruct (Orca-3) and Glaive
Sources: https://arxiv.org/html/2407.03502 , https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

- AgentInstruct: ~22M generated + 3.8M sourced = 25.8M pairs; it "can apply flows for verification and data filtering" but publishes no removal counts. Glaive function-calling v2: 112,960 samples, README empty, no filter documentation.

### AgentTuning (earliest published retention rate)
Source: https://arxiv.org/html/2310.12823

- "we filter trajectories for all tasks, except for Mind2Web, based on a final reward of r=1"; Mind2Web relaxed to r >= 2/3 "to ensure we obtain a sufficient number of trajectories". 35,341 trajectories reduced to 1,866 (5.3% kept). Held-in score 1.34 to 1.96 and held-out 0.47 to 0.65 with filtering.

### Kimi K2
Source: https://arxiv.org/html/2507.20534

- Simulated tools (3,000+ real MCP tools, 20,000+ synthetic), tasks with "explicit rubric that specifies success criteria, expected tool-use patterns, and evaluation checkpoints"; "An LLM-based judge evaluates each trajectory against the task rubrics. Only trajectories that meet the success criteria are retained". No fractions.

### Skywork-SWE, R2E-Gym, SWE-rebench trajectories
Sources: https://arxiv.org/html/2506.19290 , https://arxiv.org/html/2504.07164 , https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories

- Skywork-SWE: cap of "100 rollout turns per instance"; "A trajectory is considered valid if its final patch passes all tests"; >8,000 successes reduced to 8,209 after format-consistency filtering; no saturation seen with more data.
- R2E-Gym: 3,321 trajectories from 2,048 of 4,578 environments; hard caps "maximum of N=40 steps", "32K max tokens", "10-min" timeout; verifier data balanced positive/negative (5,700 total).
- SWE-rebench OpenHands trajectories: 67,074 trajectories, 32,161 resolved (48%), average 64.3 turns, max 100; `exit_status` recorded but no selection rule documented.

### 2026 work on trajectory quality, selection, and curation
- "A Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning of Code Agents" (https://arxiv.org/abs/2607.17205): two-axis Efficiency and Style scoring on the 67,074-trajectory SWE-trajectory set. Key result: at 500 to 1,000 trajectories, doubling data gives ~12.7% CE-loss reduction while the top-quality vs random gap is <1% (p > 0.10); at 2,000 the gap is 3.6% (p = 0.016). "error-retry rate" is the dominant quality sub-dimension.
- DeNovoSWE (https://arxiv.org/html/2606.10728): difficulty-aware thresholds where difficulty = mean pass rate over rollouts (3 rollouts, plus 3 more for imperfect instances). Keep-thresholds by difficulty band: [0,0.2): 0.90; [0.2,0.4): 0.85; [0.4,0.6): 0.80; [0.6,0.8): 0.70; [0.8,1.0]: 0.60, i.e., "easier instances are subject to stricter thresholds". ~11k trajectories kept from 4,818 instances; 0.488 to 0.500 vs best fixed threshold.
- P2T, "From Patches to Trajectories" (https://arxiv.org/html/2605.21996): "shortest-above-floor" selection ("among trajectories whose effectiveness clears a calibrated floor, take the shortest"); "trajectories whose patch fails the test suite T_i are discarded"; 1.8k of 2,438 SWE-Gym instances kept (working Docker plus passing reference patch).
- PROOF-Gen (https://arxiv.org/abs/2608.23911): "On tau2-bench, 57% of teacher trials fail, two-thirds of them near-misses"; instead of discarding, per-scenario reflection "recovers 93% of failed scenarios". Argues against pure generate-and-filter.
- Bittensor ShoppingBench distillation (https://arxiv.org/html/2606.10064): hard structural gates: "Every assistant tool-call must be matched by a corresponding tool-response in the next turn"; terminal function "must appear exactly once"; arguments "must parse as valid JSON"; max 14,336 tokens ("Longer traces silently truncate during training and are rejected"); "must not terminate on an assistant think turn"; keep only trajectories where the LM itself emits tool calls. Raw firehose 12,000 to 27,000 trajectories/day, only a small subset survives.
- BrowserForge (https://arxiv.org/html/2608.24848): 203,238 raw trajectories (~1.8M steps); rule filter "final action is the terminal Finish action ... malformed, are discarded"; VLM judge on task + last three screenshots; "only about 30% of the raw interaction steps survive, leaving roughly 600K verified steps".
- TAO-RL (https://arxiv.org/abs/2606.03762): discard trajectories "where all tool invocations fail to execute" and tasks "where all rollouts are either correct or incorrect".
- OpenVisTool (https://arxiv.org/abs/2608.08557): retain only if "its answer is correct" and "its tool observations causally contribute to that answer" (drops correct-by-luck no-tool answers).
- ClawTrack (https://arxiv.org/abs/2607.28037): process score on goal alignment, efficiency, information utilization, result verification with 12,541 rubric items; "process-based trajectory filtering yields consistent post-training improvements".
- Offline RLAIF / SFBC (https://arxiv.org/abs/2503.01062): "removes sub-trajectories preceding failures" rather than the whole trajectory.

---

## 2. Deduplication and diversity

- Magpie: embedding min-neighbor-distance > 0 with all-mpnet-base-v2 + FAISS (exact-duplicate removal in embedding space), then quality/reward/length filters (https://arxiv.org/html/2406.08464v2).
- Toucan: diversity is engineered upstream (five generator models, single-server / multi-server / featured-server sampling, max 3 tools per task, a "Tool Selection Uniqueness" score) rather than by post-hoc dedup; no dedup step is described (https://arxiv.org/html/2510.01179v1).
- Per-instance caps as diversity control: SWE-Gym cap 2 per task (https://arxiv.org/html/2412.21139); SWE-smith cap 3 per task (https://arxiv.org/html/2504.21798). Rationale in both: success is long-tailed, so uncapped rejection sampling over-represents easy tasks.
- Anthropic Clio (https://arxiv.org/html/2412.13678, full text grepped locally): facet extraction per conversation; embed the "request" facet with all-mpnet-base-v2 (768-d); k-means base clusters with k varied by dataset size ("we unfortunately cannot provide our precise values for k"); hierarchy built by grouping clusters into neighborhoods "so that the average number of clusters per neighborhood is 40" and asking Claude to propose merged parents; "Clusters are only retained if they exceed minimum size requirements for both unique accounts and conversations" (exact numbers not disclosed); privacy auditor on a 1 to 5 scale where 3 ("might narrow down identification to the order of a few thousand people") "and above being considered an acceptable level"; measured private-info rate falls from 10% in raw conversations to ~1.5% after summarization to undetectable in cluster summaries; auditor validated at 98% on 1,237 examples. Sampling note relevant to trace tools: "we take a random sample of Claude.ai outputs. Next, we deduplicate by keeping only the most recent output per conversation", which weights long conversations more; the alternative samples conversations directly. Public analysis used 1M conversations.
- Nomic Atlas and Lilac: both docs endpoints returned 404 / DNS failure during this session, so no citable thresholds. Treat as gap.

---

## 3. Production-trace curation by vendors and practitioners

- LangSmith (https://docs.langchain.com/langsmith/manage-datasets-in-application): "A technique to build datasets is to filter the most interesting traces, such as traces that were tagged with poor user feedback, and add them to a dataset." Reviewers "can optionally modify the inputs/outputs/reference outputs from a trace before it is added". Filter dimensions include feedback, metadata/tags, root-run and child-run properties, negative filters, full-text on the first 250 characters (https://docs.langchain.com/langsmith/filter-traces-in-application). Automation rules take a filter plus "a sampling rate of 50% sends half of the items that pass the filter to the action" with actions Add to Dataset / Add to Annotation Queue; thread rules close after idle time (default 10 min, minimum 2) (https://docs.langchain.com/langsmith/rules). Annotation queues: at most 100 runs per add action, configurable "Number of reviewers per run" (https://docs.langchain.com/langsmith/annotation-queues). Masking: `LANGSMITH_HIDE_INPUTS/OUTPUTS` and regex anonymizers for emails, phones, names, cards, SSNs (https://docs.langchain.com/langsmith/mask-inputs-outputs).
- Langfuse (https://langfuse.com/docs/evaluation/experiments/datasets): "A common workflow is to select production traces where the application did not perform as expected", then experts annotate correct outputs; add any span/generation to a dataset, or batch-add from a filtered Observations table with field mapping. Client-side trace sampling `LANGFUSE_SAMPLE_RATE`, default 1 (https://langfuse.com/docs/observability/features/sampling). Masking runs before export, so datasets built from traces only ever contain redacted values (https://langfuse.com/docs/observability/features/masking).
- Braintrust (https://www.braintrust.dev/docs/annotate/datasets/create): filter examples such as `scores.user_rating > 0.8`, `metadata.thumbs_up = false`, `comment IS NOT NULL and scores.correctness < 0.5`; "useful when you see a notably good or bad response in production and want to capture it as a test case"; span input maps to row input and span output "typically becomes the row's expected value"; bulk promotion via dataset pipelines. Review page filters like `scores.Preference > 0.75` and the option to "store a reference to the whole trace instead of copying span data" (https://www.braintrust.dev/docs/annotate/human-review/manage-review-work).
- Arize Phoenix (https://arize.com/docs/phoenix/datasets-and-experiments/how-to-datasets/creating-datasets): add a single span or "use the filters on the spans table and select multiple spans to add to a specific dataset"; no selection heuristics published.
- OpenAI (https://developers.openai.com/api/docs/guides/evaluation-best-practices): "Use a mix of production data ... hard-coded correct answers ... and historical data from logs"; "Ensure your test data includes typical cases, edge cases, and adversarial cases"; "Log as you develop so you can mine your logs for good eval cases"; warns against "eval datasets that don't faithfully reproduce production traffic patterns".
- Anthropic, Demystifying evals for AI agents (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): "20-50 simple tasks drawn from real failures is a great start"; source from "your bug tracker and support queue"; "A good task is one where two domain experts would independently reach the same pass/fail verdict"; test both should-trigger and should-not-trigger cases to avoid imbalance; "A 0% pass rate across many trials is most often a signal of a broken task, not an incapable agent"; pass@k vs pass^k depending on whether one success or consistency matters.
- Anthropic multi-agent research system (https://www.anthropic.com/engineering/multi-agent-research-system): "We started with a set of about 20 queries representing real usage patterns"; single LLM-judge call with 0.0 to 1.0 scores plus pass/fail was "most consistent and aligned with human judgements"; "Human evaluation catches what automation misses".
- Hamel Husain, evals FAQ (https://hamel.dev/blog/posts/evals-faq/): "you should aim to review at least 100 traces"; stop when "~20 traces don't turn up a new category"; "at least 100+ fresh traces each review cycle" every 2 to 4 weeks, and "10-20 traces weekly, focusing on outliers" in between; sampling ladder from random to clustering, extreme values, classifiers, and negative feedback, with "Keep some random traces in every batch"; binary labels over Likert; one "benevolent dictator" annotator. Field guide (https://hamel.dev/blog/posts/field-guide/): open coding of "dozens of conversations", then "Just three issues accounted for over 60% of all problems". Original evals post (https://hamel.dev/blog/posts/evals/): "keep reading logs until you feel like you aren't learning anything new".
- Eugene Yan (https://eugeneyan.com/writing/eval-process/): "we should have a 50:50 split of passes and fails that spans the distribution of inputs"; "annotate some data, prioritizing problematic outputs".

---

## 4. Handling specific anomalies as a data decision

- Retries and duplicates: no vendor or paper publishes a "which copy to keep" rule for production retries. Closest analogues: Clio keeps "only the most recent output per conversation" when sampling by completion (https://arxiv.org/html/2412.13678); SWE-Gym and SWE-smith cap repeated successes per task at 2 and 3 (https://arxiv.org/html/2412.21139, https://arxiv.org/html/2504.21798); the 2607.17205 study found "error-retry rate" is the single most predictive quality feature (https://arxiv.org/abs/2607.17205); PROOF-Gen shows two thirds of failures are near-misses worth recovering rather than dropping (https://arxiv.org/abs/2608.23911).
- Truncated observations: ToolBench caps tool responses at 1024 tokens after schema-based compression, and trains on the truncated form (https://arxiv.org/html/2307.16789v2); Bittensor rejects traces over 14,336 tokens because "Longer traces silently truncate during training" (https://arxiv.org/html/2606.10064); R2E-Gym caps 40 steps / 32K tokens / 10 min (https://arxiv.org/html/2504.07164); Skywork-SWE caps 100 turns (https://arxiv.org/html/2506.19290).
- Orphan tool calls and malformed structure: Bittensor's hard rule "Every assistant tool-call must be matched by a corresponding tool-response in the next turn" and "must not terminate on an assistant think turn" (https://arxiv.org/html/2606.10064); AgentScaler's validity control on "well-formed alternating user assistant exchanges" (https://arxiv.org/html/2509.13311); BrowserForge drops runs whose final action is not Finish (https://arxiv.org/html/2608.24848); ToolACE checks API name consistency between call and response (https://arxiv.org/html/2409.00920).
- Compaction summaries in context: Anthropic describes compaction as summarizing and "reinitiating a new context window with the summary", warns "overly aggressive compaction can result in the loss of subtle but critical context", and calls tool-result clearing the "safest lightest touch" form (https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). OpenHands' condenser preserves "the user's goals, the progress the agent has made, and what still has to be done" plus "critical files and failing tests", with only a small turn-count cost (54% vs 53% resolve) (https://www.openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents). No dataset paper reports filtering on the presence of compaction; the practical implication is that a compacted trace is not replayable from its own context alone.
- Sub-agent transcripts: sub-agents "return only a condensed, distilled summary of its work (often 1,000-2,000 tokens)" to the lead agent (https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents); Bittensor keeps only "axis-A" traces where the LM itself emits tool calls and drops traces where the LM is only "a classifier, scorer, or narrator" (https://arxiv.org/html/2606.10064).
- Chit-chat and no-tool conversations: Toucan drops trajectories that "do not contain tool calls" from the core set but then deliberately adds a 40K "Irrelevance" extension to teach abstention (https://arxiv.org/html/2510.01179); OpenVisTool drops correct answers whose tool observations did not "causally contribute" (https://arxiv.org/abs/2608.08557); TAO-RL drops trajectories "where all tool invocations fail to execute" (https://arxiv.org/abs/2606.03762).
- Failed episodes: dropped for SFT almost universally (AgentTuning 5.3% kept, SWE-Gym, Skywork, Kimi K2, APIGen-MT); kept when training verifiers (SWE-Gym 1,318 failures per subset, R2E-Gym balanced 5,700) and for RL (DAPO/AutoForge/TAO-RL only drop all-fail and all-pass groups); AgentScaler explicitly keeps tool-error trajectories if the final state matches (https://arxiv.org/html/2509.13311); SFBC removes only the sub-trajectory preceding a failure (https://arxiv.org/abs/2503.01062); Anthropic treats a 0% pass rate as a broken task signal (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- PII-masked traces: Langfuse and LangSmith mask before export, so datasets built from traces are masked by construction (https://langfuse.com/docs/observability/features/masking, https://docs.langchain.com/langsmith/mask-inputs-outputs); Toucan drops trajectories with "local file system paths" (https://arxiv.org/html/2510.01179); Clio's auditor drops clusters scoring below 3 of 5 on identifiability (https://arxiv.org/html/2412.13678).
- Near-duplicate tasks: Magpie neighbor distance > 0 (https://arxiv.org/html/2406.08464v2); per-instance caps (SWE-Gym, SWE-smith); Agent-World requires "consistent answer" in >= 2 of 5 runs to keep a task at all (https://arxiv.org/html/2604.18292).
- Extremely long trajectories: hard caps of 40 steps / 32K tokens (R2E-Gym), 100 turns (Skywork, SWE-rebench), 14,336 tokens (Bittensor); P2T prefers the shortest trajectory above an effectiveness floor (https://arxiv.org/html/2605.21996); Toucan's Conciseness score >= 4 (https://arxiv.org/html/2510.01179).
- Multi-intent conversations: APIGen's semantic check asks "whether the number of function calls matches the user's intent" (https://arxiv.org/html/2406.18518); LangSmith recommends thread-level rules "when reviewing full conversations rather than individual turns" (https://docs.langchain.com/langsmith/rules); Clio extracts facets per conversation, not per turn (https://arxiv.org/html/2412.13678).

---

## 5. Statistical guidance on how many traces per task or cluster

- Evan Miller, Adding Error Bars to Evals (https://arxiv.org/html/2411.00640): to detect a 3 percentage-point difference at 80% power, "at least n = ... ≈ 969 independent questions", so "new evals should contain at least 1,000 questions"; resampling each question K times helps until E[σ²_i]/K is small relative to Var(x), and "Going from K=1 to K=2, the total variance is reduced by 1/3" in his example; "Clustered standard errors can be over 3X larger than naive standard errors" when questions are grouped (e.g., many traces from one task cluster).
- tau-bench (https://arxiv.org/html/2406.12045): "at least 3 trials per task" for headline numbers and "8 i.i.d. trials" for pass^k; task debugging used ">40 gpt-4-turbo trials" per task to "check all tasks with zero or low success rates"; 4 tasks were fixed for "typo or ambiguity".
- Agent-World: 5 rollouts, keep if >= 2 succeed (https://arxiv.org/html/2604.18292). AutoForge: 8 rollouts, drop 0/8 and 8/8 (https://arxiv.org/html/2512.22857). SPADE: 16 plays per environment, target band [0.4, 0.6] (https://arxiv.org/html/2608.19197). Envs-FORGE: target pass rate 0.5 ± 0.2 (https://arxiv.org/html/2608.14312). DeNovoSWE: 3 + 3 rollouts (https://arxiv.org/html/2606.10728).
- Anthropic: 20 to 50 tasks to start (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents); ~20 queries for the research agent (https://www.anthropic.com/engineering/multi-agent-research-system).
- Hamel: 100 traces minimum for error analysis, stop after ~20 without a new category, 100+ per review cycle (https://hamel.dev/blog/posts/evals-faq/).
- Minimum cluster size: Clio enforces a minimum on both unique accounts and conversations per cluster but does not publish the numbers; its hierarchy targets an average of 40 clusters per neighborhood (https://arxiv.org/html/2412.13678). No other source in this survey publishes a minimum cluster size.

---

## (a) Concrete filter rules others use

| Rule | Threshold | Source | Fraction removed (if published) |
|---|---|---|---|
| Drop trajectory if agent failed to start or connect to tool server | any connection failure | Toucan https://arxiv.org/html/2510.01179 | not published |
| Drop trajectory with no tool calls (core set) | 0 tool calls | Toucan | not published |
| Drop trajectory with tool response failures | any failure | Toucan | not published |
| Drop trajectory containing local file system paths | regex match | Toucan | not published |
| Require required tools used in required order | desired tool use = 1.0 | Toucan | not published |
| LLM judge completeness and conciseness | >= 4 of 5; question quality and realism = 5 | Toucan | 119.3K of 1.53M kept (~7.8%) for SFT |
| LLM judge: any step inconsistent, incoherent, wrong tool | drop | Nemotron-Agentic https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 | not published |
| Format check (JSON with query/answer, valid args) | strict parse | APIGen https://arxiv.org/html/2406.18518 | 2.0% to 10.8% of 40K depending on generator |
| Execution check (type errors, invalid params, runtime, timeout, syntax, missing args) | any failure | APIGen | 8.4% to 38.7% |
| Semantic check (5 LLM criteria incl. number of calls matches intent) | LLM verdict | APIGen | 5.4% to 19.9%; overall pass 34% to 84% |
| Keep only trajectories reaching goal (final state and final response match ground truth) | exact match | APIGen-MT https://arxiv.org/html/2504.03601 | 33% of rollouts dropped (67% success) |
| Blueprint committee majority vote + avg score above threshold, up to 3 or 5 retries | majority | APIGen-MT | 30% of blueprints rejected with feedback loop, 72% without |
| Keep bug only if it breaks >= 1 passing test | F2P >= 1 | SWE-smith https://arxiv.org/html/2504.21798 | 49.9% of candidates dropped overall |
| Keep only resolved trajectories | patch passes tests | SWE-smith, SWE-Gym, Skywork-SWE, P2T | SWE-smith 64% dropped (36% resolve); SWE-rebench 52% unresolved |
| Cap successes per task instance | 3 (SWE-smith), 2 (SWE-Gym) | https://arxiv.org/html/2504.21798, https://arxiv.org/html/2412.21139 | SWE-smith 6,457 to 5,016 (22%) |
| Reward-based keep | r = 1 (r >= 2/3 for Mind2Web) | AgentTuning https://arxiv.org/html/2310.12823 | 94.7% dropped (35,341 to 1,866) |
| Validity control: alternating user/assistant structure, n-gram repetition filter | malformed or repetitive | AgentScaler https://arxiv.org/html/2509.13311 | not published |
| Final environment state must match golden state | exact | AgentScaler, AutoForge | not published |
| Tool sequence and arguments exactly match intent | exact | AgentScaler | not published |
| Task consistency gate | >= 2 successes of 5 runs | Agent-World https://arxiv.org/html/2604.18292 | not published |
| Drop tasks with all-pass or all-fail rollouts | 0/8 and 8/8 | AutoForge https://arxiv.org/html/2512.22857 ; TAO-RL https://arxiv.org/abs/2606.03762 | not published |
| Target pass-rate band | 0.5 ± 0.2 (Envs-FORGE); [0.4, 0.6] (SPADE) | https://arxiv.org/html/2608.14312 ; https://arxiv.org/html/2608.19197 | n/a (reweighting) |
| Oracle solution must get reward 1 under generated tests | exact | Envs-FORGE | not published |
| Difficulty-aware keep threshold on rollout score | 0.90 / 0.85 / 0.80 / 0.70 / 0.60 by pass-rate band | DeNovoSWE https://arxiv.org/html/2606.10728 | ~11k kept from 4,818 instances x 3 to 6 rollouts |
| API executability, required params, regex on formats, JSON Schema definition | rule pass | ToolACE https://arxiv.org/html/2409.00920 | not published |
| Parameter values must appear in query or system prompt (hallucination) | LLM | ToolACE, xLAM | not published |
| Drop APIs with long latency or HTML/error bodies | qualitative | ToolBench https://arxiv.org/html/2307.16789 | 69% of APIs dropped (53,190 to 16,464) |
| Drop instructions citing nonexistent APIs; keep only passing DFSDT paths | pass | ToolBench | ~37% of instructions dropped (200k to 126,486) |
| Truncate tool responses | 1024 tokens after compression | ToolBench https://arxiv.org/html/2307.16789v2 | n/a |
| Embedding near-duplicate removal | min neighbor distance > 0 (all-mpnet-base-v2) | Magpie https://arxiv.org/html/2406.08464v2 | 4M to 300K overall (all filters) |
| Every tool call must have a response next turn; terminal call exactly once; args valid JSON; no ending on think turn | hard reject | Bittensor https://arxiv.org/html/2606.10064 | not published |
| Max trace length | 14,336 tokens (Bittensor); 40 steps / 32K tokens / 10 min (R2E-Gym); 100 turns (Skywork) | https://arxiv.org/html/2606.10064 ; https://arxiv.org/html/2504.07164 ; https://arxiv.org/html/2506.19290 | not published |
| Final action must be Finish; malformed sequences dropped; VLM judge on last 3 screenshots | rule + judge | BrowserForge https://arxiv.org/html/2608.24848 | ~70% of steps dropped |
| Keep only if tool observations causally contribute to correct answer | LLM/causal test | OpenVisTool https://arxiv.org/abs/2608.08557 | not published |
| Shortest trajectory above effectiveness floor | calibrated floor | P2T https://arxiv.org/html/2605.21996 | 1.8k of 2,438 instances usable |
| Cluster retained only above minimum unique accounts and conversations; auditor score >= 3 of 5 | undisclosed minimums | Clio https://arxiv.org/html/2412.13678 | private info 10% to 1.5% to ~0% |
| Add trace to dataset when feedback is poor / score low | `scores.correctness < 0.5`, `thumbs_up = false`, `user_rating > 0.8` for positives | Braintrust https://www.braintrust.dev/docs/annotate/datasets/create ; LangSmith https://docs.langchain.com/langsmith/manage-datasets-in-application | n/a |
| Sample a percentage of filter matches into a queue | 0 to 100% (e.g., 50%) | LangSmith https://docs.langchain.com/langsmith/rules | n/a |
| Review at least N traces before deciding categories | 100; stop after ~20 with no new category | Hamel https://hamel.dev/blog/posts/evals-faq/ | n/a |
| Balance pass/fail in eval set | 50:50 | Eugene Yan https://eugeneyan.com/writing/eval-process/ | n/a |
| Eval size for 3pp resolution | ~1,000 questions; K=2 trials cuts variance by ~1/3 | Miller https://arxiv.org/html/2411.00640 | n/a |
| Trials per task for reliability metrics | >= 3, 8 for pass^k; >40 to debug zero-success tasks | tau-bench https://arxiv.org/html/2406.12045 | n/a |

---

## (b) Recommended minimal filter set for production traces going into a re-executable eval (ordered)

1. Structural integrity gate: drop traces with orphan tool calls (call without a response in the next turn), unparseable tool arguments, non-alternating role structure, or a trace that ends mid-turn on a think/plan message. Rationale: these traces cannot be re-executed deterministically and every pipeline that publishes hard rules (Bittensor, AgentScaler, BrowserForge, ToolACE) puts this first because it is cheap and unambiguous.
2. Infrastructure-failure gate: drop traces where the agent never started, the tool server failed to connect, or every tool invocation errored. Rationale: Toucan and TAO-RL drop these because the failure is in the harness, not the agent, so they cannot discriminate agent quality; keep single tool errors (see (c)).
3. Replayability gate: drop traces whose context depends on state you cannot reconstruct: compaction summaries with the original turns gone, sub-agent results with no sub-agent transcript, or tool outputs truncated below the length your replay will produce. Rationale: Anthropic and OpenHands document that compaction and sub-agent summaries lose information by design, and Bittensor rejects over-length traces precisely because they "silently truncate".
4. PII and secrets gate: drop or mask traces containing local file paths, credentials, or identifiers, and record whether masking happened upstream. Rationale: Toucan drops file paths outright; Langfuse and LangSmith mask before export, so any downstream dataset must assume masked values will not replay against real systems.
5. Exact and near-duplicate collapse: collapse identical inputs and retries to one representative (keep the last completed attempt, as Clio keeps the most recent output per conversation), then embedding near-duplicate removal at a strict threshold (Magpie uses neighbor distance > 0). Rationale: retries and repeated prompts distort pass rates and inflate cluster counts; the 2607.17205 study shows error-retry rate is the single strongest quality signal.
6. Per-task cap: cap traces per intent cluster at 2 to 3. Rationale: SWE-Gym (cap 2) and SWE-smith (cap 3) both found that success is long-tailed and uncapped selection over-represents easy tasks.
7. Ground-truth and consistency gate: only promote a trace to a reference case if it either passed a verifiable check (final state, test, or explicit user feedback) or two reviewers agree on the pass/fail verdict; re-run 3 to 5 times and drop tasks that never pass as broken. Rationale: Anthropic's "two domain experts would independently reach the same pass/fail verdict" and "0% pass rate ... signal of a broken task"; Agent-World's 2-of-5 gate; tau-bench's >40-trial audit of zero-success tasks.
8. Length and step cap: apply an explicit cap consistent with your replay budget (published caps range from 40 steps / 32K tokens to 100 turns). Rationale: caps keep replays affordable and Toucan's conciseness >= 4 and P2T's shortest-above-floor both prefer efficient traces.
9. Difficulty banding for the final set, not a hard filter: tag each cluster by observed pass rate and stratify so the eval is not dominated by 100% or 0% clusters. Rationale: AutoForge/DAPO, Envs-FORGE (0.5 ± 0.2), and SPADE ([0.4, 0.6]) all treat mid-band tasks as the informative ones; DeNovoSWE shows thresholds should loosen on hard tasks.
10. Human spot-check: read at least 100 candidate traces, keep some random ones in every batch, and stop adding categories when ~20 consecutive traces add nothing new. Rationale: Hamel's numbers, Anthropic's 20 to 50 starting tasks, and APIGen's 600-sample audit are the only published practices for validating a filter pipeline itself.

---

## (c) What NOT to filter and why

- Failed episodes with a clean structure: keep them, tagged. SWE-Gym and R2E-Gym need balanced failures for verifiers; AutoForge/TAO-RL only drop all-fail groups; Eugene Yan wants a 50:50 pass/fail split; Anthropic requires should-not-trigger cases to avoid one-sided evals.
- Single tool errors inside an otherwise successful run: AgentScaler explicitly keeps them because the goal was still reached; they are also exactly the recovery behaviors an eval should test.
- Near-miss failures: PROOF-Gen shows two thirds of failures are near-misses (one decisive wrong step) and are the most informative cases to recover or grade.
- No-tool or abstention conversations that are correct: Toucan added 40K irrelevance cases on purpose; dropping every no-tool trace removes the "should not call a tool" half of the distribution, but do drop correct-by-luck traces where tools did not contribute (OpenVisTool).
- Long traces below your hard cap: the 2607.17205 study found quality selection helps only at scale, and Skywork-SWE saw no saturation; do not over-prune on style or length alone at small dataset sizes.
- Hard tasks with low but nonzero pass rate: SWE-smith found difficulty does not predict downstream value; DeNovoSWE loosens thresholds for hard bands; only 0% across many trials indicates a broken task.
- Random samples: keep a random slice in every batch (Hamel) so the filters themselves do not blind you to unanticipated failure modes.
- Small clusters below a privacy threshold should be excluded from aggregate reporting (Clio), but not from the eval itself; the minimum cluster size is a reporting constraint, not a quality signal.

Gaps: no source publishes a rule for which retry copy to keep, none publishes a minimum cluster size number (Clio withholds it), and Nomic Atlas / Lilac dedup thresholds could not be fetched this session.