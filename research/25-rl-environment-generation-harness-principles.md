# R25. Harness design for environment-generation workflows: learn-harness-engineering vs RL environment-generation pipelines (2026-08-27)

Purpose: the founder frames our Harness as "a workflow for RL environment generation," with two hard rules: (1) the Runner must be frozen, (2) the Builder should keep improving until the harness reliably generates high-quality Environments. This report reads walkinglabs/learn-harness-engineering (a coding-agent harness course, not an RL environment-generation course) end to end, extracts every stated principle, and checks which of those principles actually transfer to an environment-generation harness versus which principles only the RL environment-generation literature (R12, R22, plus fresh 2026 sources) states. It ends with a concrete set of changes to the module list from R24.

## Coverage caveats

- Read in full with `cat`: all 14 lecture `index.md` files under `docs/en/lectures/` (this report quotes lectures 2, 3, 6, 8, 9, 10, 11, 12, 13, 14 as instructed; lectures 1, 4, 5, 7 were read but contain no principle not already superseded by the ten quoted here, so they are not separately quoted); all four harness-design writeups under `docs/en/harness-designs/` (claude-code, codex, pi, deepseek); `docs/en/projects/index.md`. Not read line-by-line: the `code/*.ts`, `code/*.py`, `code/*.md` template files under each lecture (these are illustrative snippets referenced by the prose already quoted, not additional principles) and the `resources/` tree (OpenAI advanced templates, reference docs) — skimmed for section titles only, nothing there is specific to environment generation.
- Local clone used: `/private/tmp/.../scratchpad/lhe`. File paths below are given relative to the repo root (`learn-harness-engineering/docs/en/...`) so they resolve on GitHub at `github.com/walkinglabs/learn-harness-engineering`.
- Section 2 reuses R12 (`12-env-synthesis-pipelines-tauforge-style.md`) sections 3-4 and R22 (`22-tau3-and-what-a-good-environment-looks-like.md`) section 5 verbatim where cited, and adds four 2026 sources not in either report, fetched today: EnvHarness/EnvRigger full HTML (arXiv 2608.19880), EvoEnv full HTML (arXiv 2605.14392), Prime Intellect's General Agent blog post, and ClawEnvKit (arXiv 2604.18543, PDF — extraction quality lower than the two HTML fetches, marked where used). WebSearch found nothing else materially new since R12's 2026-08-26 cutoff; the field moved one day, not one generation.
- R24 (`24-harness-design-principles.md`) is not re-derived here; section 5 assumes its module list and record schema as the baseline and states deltas against it.
- Quotes are verbatim including punctuation. My own connective prose never uses em or en dashes; quoted upstream text is reproduced exactly as written, dashes included.

---

## 1. Principles, checklist items, and design rules from learn-harness-engineering

### 1.1 Lecture 2 — What a harness actually is

`learn-harness-engineering/docs/en/lectures/lecture-02-what-a-harness-actually-is/index.md`

```
This lecture gives harness a precise, actionable definition — not an academic abstraction, but a framework you can put to work today. A harness consists of five subsystems: instructions, tools, environment, state, and feedback.
```
```
everything in the engineering infrastructure outside the model determines how much of the model's capability actually gets realized.
```
```
A good harness uses executable rules to constrain the agent, rather than enumerating instructions one by one. OpenAI says "enforce invariants, don't micromanage implementation"
```
```
To quantify each harness component's marginal contribution, remove them one at a time and see which removal causes the biggest performance drop.
```
```
Harness rots like code does. Audit regularly, and pay down harness debt just like you pay down technical debt.
```

Checklist item stated explicitly: the five subsystems (instructions, tools, environment, state, feedback) are each individually necessary; "Missing any one of the five subsystems means an incomplete harness."

### 1.2 Lecture 3 — Why the repository must become the system of record

`learn-harness-engineering/docs/en/lectures/lecture-03-why-the-repository-must-become-the-system-of-record/index.md`

```
information that doesn't exist in the repo, doesn't exist for the agent.
```
```
Knowledge Visibility Gap: The proportion of total project knowledge that's NOT in the repository. The bigger the gap, the higher the agent's failure rate.
```
```
Fresh Session Test: ... open a brand-new agent session, give it only the repository contents, and see if it can answer five basic questions.
```
```
Applying database transaction principles (Atomicity, Consistency, Isolation, Durability) to agent state management.
```
```
Atomicity: A "logical operation" ... is committed as a whole only once it's complete and verified. ... All or nothing.
Consistency: Define "consistent state" verification predicates ... After an operation, the system should be in a verifiably correct state.
Isolation: When multiple agents work concurrently, design state files to avoid race conditions.
Durability: Critical project knowledge lives in git-tracked files.
```

### 1.3 Lecture 6 — Why initialization needs its own phase

`learn-harness-engineering/docs/en/lectures/lecture-06-why-initialization-needs-its-own-phase/index.md`

```
Initialization and implementation have completely different optimization targets. The implementation phase aims to maximize the quantity and quality of verified features. The initialization phase aims to maximize the reliability and efficiency of all subsequent implementation.
```
```
Startup Readiness Checklist: The conditions under which a project can be unambiguously operated by a fresh agent session: can start, can test, can see progress, can pick up next steps. Four conditions, all required.
```
```
Starting from a template far outperforms starting from scratch.
```
```
Initialization completion criteria: Not "how much code was written," but whether the startup readiness checklist's four conditions are all met.
```

### 1.4 Lecture 8 — Why feature lists are harness primitives

`learn-harness-engineering/docs/en/lectures/lecture-08-why-feature-lists-are-harness-primitives/index.md`

```
Both Anthropic and OpenAI emphasize: artifacts must be externalized. Feature state must live in a machine-readable file in the repo, not in unstructured conversation text.
```
```
Triple structure: Each feature item contains three elements: (behavior description, verification command, current state).
```
```
Pass-state gating: The only way a feature moves from active to passing is by the verification command executing successfully. This transition is irreversible.
```
```
Feature lists as harness primitives play the same role as database-level constraints — the agent cannot bypass them.
```
```
The agent can't directly change a feature's state to passing. It can only submit a verification request. The harness executes the verification command and decides whether to allow the transition.
```
```
Calibrate granularity to "completable in one session." Too broad and it won't finish; too narrow and you can't manage it.
```

### 1.5 Lecture 9 — Why agents declare victory too early

`learn-harness-engineering/docs/en/lectures/lecture-09-why-agents-declare-victory-too-early/index.md`

```
modern neural networks are systematically overconfident — the confidence reported by models is significantly higher than their actual accuracy. AI coding agents are no different.
```
```
Verification-Validation Dual Gate: The first layer (verification) checks whether the code correctly implements the specified behavior; the second layer (validation) checks whether system-level behavior meets end-to-end requirements. Both must pass before the task is considered complete.
```
```
when an agent is asked to evaluate its own work, it systematically provides overly positive assessments ... The solution is to separate the "worker" from the "checker."
```
```
| Architecture | Runtime | Cost | Core Features Working? |
| Single agent (bare run) | 20 mins | $9 | No |
| Three agents (planner + generator + evaluator) | 6 hours | $200 | Yes |
```
```
Completion Priority Constraint: First verify functional correctness, then address performance, and finally handle style. No refactoring is allowed until core functionality has been verified.
```
```
error messages written for agents should include repair instructions ... "Test failed: POST /api/reset-password returned 500. Check that the email service config exists..."
```

### 1.6 Lecture 10 — Why end-to-end testing changes results

`learn-harness-engineering/docs/en/lectures/lecture-10-why-end-to-end-testing-changes-results/index.md`

```
The design philosophy of unit testing is isolation: mock dependencies, focus on the unit under test. This philosophy makes unit tests fast and precise, but it also creates systematic blind spots.
```
```
Only end-to-end testing can prove the absence of system-level defects.
```
```
when an agent knows its work will be validated by end-to-end tests, its coding behavior shifts.
```
```
for agent-generated codebases, architectural constraints must be established as early prerequisites on day one ... agents copy existing patterns in the repository, even when those patterns are inconsistent or suboptimal.
```
```
Enforce invariants; don't micromanage implementation.
```
```
Review Feedback Promotion: Converting recurring code review comments into automated tests. Every time a new category of repeated issue is found, add a rule, and the harness grows stronger automatically.
```

### 1.7 Lecture 11 — Why observability belongs inside the harness

`learn-harness-engineering/docs/en/lectures/lecture-11-why-observability-belongs-inside-the-harness/index.md`

```
Without observability, agents make decisions under uncertainty, evaluations become subjective judgments, and retries become blind wandering.
```
```
Runtime observability: System-level signals ... Answers "what did the system do." Process observability: Visibility into harness decision artifacts ... Answers "why should this change be accepted."
```
```
Sprint contract: A short-term agreement negotiated before coding begins, specifying task scope, verification standards, and exclusions.
```
```
Evaluator rubric: Transforms quality evaluation from subjective judgment into evidence-based structured scoring, enabling different evaluators to reach similar conclusions for the same output.
```
```
Agents don't know what they don't know. They won't proactively record signals they don't realize they need.
```
```
Early versions would identify reasonable issues, then talk themselves into dismissing those issues as not severe, ultimately approving the work. The fix: read the evaluator's logs, find the points where its judgment diverged from human judgment, and update the QA prompt.
```
```
Missing observability wastes 30-50% of session time on redundant diagnosis.
```

### 1.8 Lecture 12 — Why every session must leave a clean state

`learn-harness-engineering/docs/en/lectures/lecture-12-why-every-session-must-leave-a-clean-state/index.md`

```
long-term reliability depends on operational discipline, not just single-run success.
```
```
agents copy patterns already present in the repository, even when those patterns are inconsistent or suboptimal. Over time, this copying inevitably leads to drift.
```
```
Encode "golden rules" into the repository ... Establish periodic cleanup workflows ... Capture human taste once, enforce it continuously.
```
```
Clean state: The system must satisfy five conditions at session end — build passes, tests pass, progress recorded, no stale artifacts, startup path available.
```
```
Quality document: An active artifact that continuously records quality ratings for each module. Not a one-time assessment, but a tracker showing whether the codebase is getting stronger or weaker over time.
```
```
As models improve, periodically remove harness components that are no longer necessary. A constraint essential today may be unnecessary overhead in three months.
```
```
Every month, pick one harness component, temporarily disable it, and run benchmark tasks. If results don't degrade, remove it permanently.
```
```
Idempotent cleanup: Cleanup operations produce the same result regardless of how many times they run.
```

### 1.9 Lecture 13 — Loop engineering

`learn-harness-engineering/docs/en/lectures/lecture-13-loop-engineering/index.md`

```
/goal is essentially a loop. It has exactly three parts: a goal, a verification method, and a stopping condition.
```
```
the final step was taking "judging whether it's done" out of the hands of the agent doing the work, and handing it to an independent judge. It could be a different model, a script, or a test command — but the rule was the same: the person writing the code can't grade their own homework.
```
```
Generator/Evaluator Separation: The agent that writes the code and the agent that checks it must be separated. A model grading its own work is untrustworthy.
```
```
Worktree Isolation: Each parallel agent works in an independent git worktree, physically preventing file collisions.
```
```
External State: Memory that lives outside a single conversation ... Models forget everything between runs; memory must live on disk.
```
```
Someone in your crew must not believe you.
```
```
Four Silent Costs: verification debt, comprehension rot, cognitive surrender, token blowout.
```
```
stopping conditions must be machine-checkable, never "feels about right."
```
```
program.md is a Markdown document, not a Python script. ... one iron rule: never ask for human help, just keep going.
```
```
Fixed 5-minute wall-clock budget ... no matter what the agent changes, every experiment takes exactly the same time. This means all results are directly comparable under the same time budget.
```
```
Only commits that actually improved stay on the main branch. Everything that failed was rolled back. git log is a validated research log.
```

### 1.10 Lecture 14 — Graph engineering

`learn-harness-engineering/docs/en/lectures/lecture-14-graph-engineering/index.md`

```
Loops have a lot of room for forgiveness. Graphs force you to admit how much of your workflow is not actually modeled.
```
```
A loop is a deferred decision ... A graph is an up-front decision. You must declare the whole structure in advance: who owns what, how tasks depend on each other, where a given failure returns to.
```
```
1. Goodhart: The Numbers Went Up, the Business Got Worse ... the bot learned to close tickets.
```
```
2. Blindness Upward: It Never Asks "Is This the Right Goal?" ... A thermostat can't ask whether 68°F is the right temperature.
```
```
Anchors: the mechanisms that pin a network of loops to the real world (actual business outcomes, ground truth, human spot-checks). The easiest part of graph design to skip, and the one you can't afford to.
```
```
Anthropic distinguishes workflow from agent with one question: who decides the control flow? If your code fixes the steps, it's a workflow. If the model can change the steps at runtime, it's an agent. ... A graph is the container that holds both.
```
```
In a monolithic agent, "review" still runs in the same context, so the agent reviews itself; in a graph, verify must get a completely fresh context.
```
```
The Orchestration Tax ... starting an agent is cheap. Closing the loop on one is expensive. ... You are the GIL of your AI agents.
```
```
Five criteria — try at least three before you start: independently decomposable work units; branch or rollback paths; intermediate state worth saving; results verifiable; coordination benefits > coordination costs.
```

### 1.11 Harness design: Claude Code

`learn-harness-engineering/docs/en/harness-designs/claude-code/index.md`

```
At the core of Claude Code is a simple while loop ... But most of the code is not in that loop; it is in the systems surrounding the loop.
```
```
More specific instructions appear later in the context.
```
```
Compaction should be a staged funnel: lossless before lossy; do not start by summarizing everything.
```
```
Separation of responsibilities: CLAUDE.md manages "what," Skills manage "how," MCP manages "where to connect," and hooks manage "when to enforce."
```
```
Claude Code's permissions do not simply "ask about everything." They combine seven modes with an ML-based classifier.
```
```
PostToolUse hooks can force checks to run after tool execution and write the results back into the context; Stop hooks intervene when the agent declares completion.
```
```
Each subagent's conversation history is stored in a separate sidechain file and does not inflate the parent agent's context.
```

### 1.12 Harness design: Codex

`learn-harness-engineering/docs/en/harness-designs/codex/index.md`

```
the repository is the system of record, AGENTS.md is only a directory page, and engineering value lies in designing environments, expressing intent, and building feedback loops.
```
```
We therefore stopped treating AGENTS.md as an encyclopedia and began treating it as a directory page.
```
```
don't micromanage the implementation; focus on invariants.
```
```
Write: Persist context outside the window ... Select: Pull only the necessary tokens into the window ... Compress: Preserve what truly matters ... Isolate: Divide context across different boundaries.
```
```
every task runs in an independent git worktree, paired with a local observability stack (logs, metrics, and traces), so every change can be verified in an isolated environment.
```
```
Codex's spawn_agent / wait_agent are core tools: the model explicitly creates a subagent, gives it independent session history and a tool set, and waits for the result.
```
```
Approval policies and plan mode ... before high-risk operations execute, the system first produces a plan and requests approval, turning "task boundaries" and "human decision-making authority" into runtime controls.
```

### 1.13 Harness design: Pi

`learn-harness-engineering/docs/en/harness-designs/pi/index.md`

```
Pi's philosophy is "minimize the core + make extensions programmable." ... rules and extensions fully determine what the model can see and when it can see it, instead of those decisions being hard-coded into the core.
```
```
the compaction strategy itself is customizable: you can use extensions to implement topic-based compaction, code-aware summaries, or even use a different model for summarization.
```
```
sessions are stored as trees; /tree can return to any historical node and continue from there, with every branch stored in the same file.
```
```
Progressive disclosure ensures that skill details enter the context only when triggered, without blowing up the prompt cache.
```
```
Extensions ... intercepting dangerous commands (a permission gate), checkpointing code state when switching tasks, protecting paths ... modifying tool output before passing it to the model.
```
```
VISION.md (goals), PROGRESS.md (progress), LESSONS.md (lessons), and STANDARDS.md (standards) are all Markdown files persisted across sessions.
```

### 1.14 Harness design: DeepSeek Harness — detailed summary

`learn-harness-engineering/docs/en/harness-designs/deepseek/index.md`

DeepSeek Harness (`dsh`) is the one design in the course that treats the harness as a runtime independent of any specific model, and it is the closest analogue in the course to what we are building (an environment-generation workflow, not a single fixed coding agent). Its official definition: **Agent = Model + Environment + Tools + State**. The whole product is built as plugins on top of a minimal core called Cordis, which "owns no concrete agent capabilities":

```
Every part of the product is a plugin, including the model adapter, the tool registry, the session log, and the agent loop itself.
```
```
There is no privileged core to patch ... you extend dsh by mounting a plugin beside the others.
```

Three architectural pieces matter for us:

1. **Capability seams.** Every capability (filesystem, shell, sandbox, web, LLM, subagent) is split into a Service Definition (interface), a Service Provider (implementation, swappable — Local FS vs E2B FS vs Remote FS), and a Consumer (a model-facing tool). Quote:
```
a seam is a swappable capability with three roles: a Service Definition declaring the interface, a Service Provider implementing it, and a Consumer using it, commonly a model-facing tool.
```
This resolves "should an agent depend on concrete tools or on capability interfaces" in favor of interfaces: swapping a Provider leaves the tool surface exposed to the model unchanged while completely replacing the environment underneath.

2. **Event pipeline.** The turn loop (`turn/start → assemble → agent/pre-step → step/start → agent/request → llm/stream → assistant/message → tool/call → tools/pre-execute → tools/execute → tools/post-execute → tool/result → step/end`) is not a hardcoded sequence but a set of named event points plugins can subscribe to:
```
Want to run a security check before a tool executes? Listen to tools/pre-execute. Want to add memory? Inject it at agent/pre-step. Want to record behavior? Subscribe to session events.
```
This means permissions, memory, and observability attach as listeners rather than being hardcoded inside the loop.

3. **Append-only session log with an enforced invariant.** DeepSeek Harness's strongest engineering constraint:
```
Model-visible means logged. Anything that reaches a model request must be reconstructable from the log, and a runtime invariant asserts it.
```
This is stronger than "we log things": it is a runtime-checked guarantee that every token the model ever saw is recoverable from the append-only log, which is exactly the property a re-execution Runner needs to make a Verdict reproducible and auditable.

Course's own assessment table for DeepSeek Harness:

```
| Subsystem | DeepSeek Harness's Implementation | Assessment |
| Instructions | Plugin-based; rules/skills are injected as plugins | Extremely flexible, but lacks a built-in convention like CLAUDE.md |
| Tools | Service Definition → Provider → Consumer capability seam | Tool subsystem standardization taken to the extreme |
| Environment | Sandbox/FS/Shell Providers are all replaceable (including remote E2B) | The environment is fully pluggable |
| State | Append-only Session Event Log + Model-visible means logged | Observability is a first-principles constraint |
| Feedback | Permission / guard / policy / hook at tools/pre-execute | Feedback mechanisms are event-based |
```

And its own summary of the tradeoff:

```
Pi, Claude Code, and Codex all optimize the harness "inside a specific agent," while DeepSeek Harness defines the harness as an operating system independent of the model, with the agent itself merely a replaceable application running on that OS. The tradeoff is equally clear: greater flexibility means higher configuration cost.
```

For us, DeepSeek Harness is the strongest coding-agent precedent for the Builder-as-pipeline-of-swappable-stages design already sketched in R24 (`mine.py`, `compile_env.py`, `policy.py` as independently replaceable stages), and its "model-visible means logged" invariant is the strongest coding-agent precedent for a testable Runner determinism guarantee.

---

## 2. Principles specific to RL environment-generation workflows

This section re-derives principles from the pipelines already catalogued in R12 section 3-4 and R22 section 5, and adds four 2026 sources found today that neither report covers: EnvHarness/EnvRigger, EvoEnv, Prime Intellect's General Agent, and ClawEnvKit. For each, the axes requested: stages, code vs LLM, per-stage validation, failed-artifact handling, caching/versioning, quality measurement, and whether the generator itself improves over iterations.

### 2.1 Already covered in R12/R22 (not re-quoted, cited by name)

LOGIGEN, AutoForge, ScaleEnv, EnvScaler, AgentScaler, Envs-FORGE, EnvFactory, Agent-World, AgentSynth, Halluminate Westworld, Fleet (no design content found), Prime Intellect Environments Hub (packaging only), OpenEnv, verifiers, Simia, "TauForge" (not a public system) — full detail in `12-env-synthesis-pipelines-tauforge-style.md` sections 1-4 and `22-tau3-and-what-a-good-environment-looks-like.md` section 5. The seven cross-cutting patterns from R12 section 3 (ground truth by execution, policy as code, reward = state diff with column classes, verifier validation triad, difficulty calibration by pass rate, thin front-loaded human input, reported costs) are the baseline this section extends, not repeats.

### 2.2 EnvHarness / EnvRigger (Aug 2026), arXiv 2608.19880 — the frozen-verifier design point

This is the single most relevant 2026 source for the founder's "Runner must be frozen" framing, because it is the only pipeline in either report that treats the grader as explicitly immutable and only lets a wrapper layer around the environment change.

Stages, quoted:
```
Observe, Diagnose, Write, and Validate, where the last two steps form a write-and-validate loop
```
Code vs LLM: EnvRigger emits Python code (a class overriding per-step hooks: `Stage`, `Contract`, `Chain`), with the decision of what to write driven by LLM reasoning over rollout statistics; the wrapper hooks themselves execute as ordinary code at runtime.

Failed-artifact handling:
```
Components failing this evaluation undergo iterative revision until they succeed
```
```
did this mutation move SR toward the target band? If yes, the perturbation TYPE is right [otherwise] start over with a different perturbation type.
```

Quality measurement: an explicit ACCEPT/REFINE/REJECT decision referencing rollout statistics (success rate over K runs, failure distribution), with an unsolvability guard:
```
do not make the task unsolvable. A mutation that makes success impossible is not a difficulty increase
```

The frozen part, quoted twice for emphasis because it is the load-bearing claim:
```
every reshaped environment retains its original verifier
```
```
success is the benchmark's own verdict, so reshaping reward cannot move the eval metric
```

Generator improvement over iterations: EnvRigger itself is not trained; it is a prompted LLM agent whose outputs are validated and, on failure, retried with a different perturbation strategy chosen by an in-context rule ("if yes, type is right; if no, start over"), not by gradient updates. Reported gains: ALFWorld OOD +9.0, SWE-bench Verified +2.7 with 9.8% fewer steps, scaling from 47.67 to 54.79 success at 300 environments.

### 2.3 EvoEnv / "Learning to Build the Environment" (2605.14392) — the self-trained-generator design point

This is the opposite pole from EnvRigger: here the generator is not just prompted, it is RL-trained, using the same policy in two roles.

Stages, quoted:
```
candidate environments enter the pool only after staged validation, semantic self-review, solver-relative difficulty calibration, and novelty checks.
```
Five staged validation gates, quoted:
```
L1: Parseable Python; expected class and methods exist
L2: Instantiation and execution on multiple seeds/difficulty values
L3: Determinism under identical seeds; prompt, state, and reference answer are stable
L4: Non-triviality across seeds and difficulty values
L5: Scorer contract verification (reference answers score positively; perturbations fail)
```
Code vs LLM: "The same policy alternates between a generator role, which proposes Python environments... and a solver role" that attempts the generated tasks; both roles are the same trained model.

Failed-artifact handling: rejected L1-L4 candidates "receive penalty rewards but cannot become a reward source for solver training"; semantic review uses "an any-reject rule: if any review identifies a likely semantic bug, the candidate is rejected."

Caching/versioning: "caches of prompt and code embeddings for previously admitted environments" drive a novelty bonus; "pool rotation retires environments after fixed epochs, but original seed environments are protected by a floor" of 0.2 minimum representation — an explicit anti-catastrophic-forgetting mechanism for the seed distribution.

Quality measurement: pass rate over m=8 calibration instances must satisfy 0 < pass rate < 1 to avoid saturation or total failure, with a target a* = 0.3, chosen "below 0.5 because near-half accuracy candidates can become saturated quickly as the solver improves."

Generator improvement over iterations, the key finding for our question: the generator's own gradient comes from validation-layer success, solver-relative difficulty reward, and novelty bonus, applied via a "role-conditioned GRPO objective" — the Builder here is literally trained by reinforcement learning against the same signal that grades its output, and reports "once an environment is admitted, the reward source is a frozen executable path" as the mechanism that stops this from becoming a moving target (the executable path, not the pass-rate target, is what's frozen once admitted). Reported gain: Qwen3-4B-Thinking 72.4 to 74.8 (relative +3.3%).

### 2.4 Prime Intellect, General Agent (blog, 2026) — measured, tiered difficulty calibration

Five-step synthesis pipeline, quoted:
```
Design: Pick a novel domain, design a DB schema, and define the tool API.
Seed: Write the simplest useful task in the domain. Write a verification function and produce a passing gold solution.
Gate: Run the solver against the seed tier with 20 rollouts.
Evolve: For each subsequent tier (t1→t4), add evolution strategies, extend the DB, write the new task, validate it, and gate it.
Validate: Final check: the family must use ≥5 unique evolution strategies across its tiers.
```
Code vs LLM: "The synthesizer is an LLM agent running in a sandboxed environment with access to the general-agent CLI"; tools are "simple Python functions reading or manipulating the state of the database."

Failed-task handling: "If the solve rate is ≥0.80, the seed is accepted. If not, adjust and retry"; a failed verification check means "something about the task is fundamentally broken."

Quality measurement: per-tier target pass-rate bands (e.g., tier t3 targets 0.4-0.2), confirmed empirically: "Since we used GPT-5-Mini for difficulty calibration, its solve rates fall exactly within the target bands." This is calibration by measurement, not by assumed difficulty labels.

Generator improvement over iterations: not RL-trained; instead the corpus self-seeds forward — "hard tiers seed the next wave of extensions, letting the corpus grow progressively harder over time," with stated future work of "using the hardest tier as a seed and generating more tasks against a stronger gating model." The improvement mechanism is corpus-level ratcheting (each round's hardest output becomes next round's seed), not weight updates to a generator model.

### 2.5 ClawEnvKit (2604.18543) — three-phase pipeline with a hard retry cap

Three phases, quoted: Generation (Parser → Generator → Validator, "generates verified environment sets ℰ suitable for both agent evaluation and RL training"), Execution ("Sandbox Initialization, Harness Preparation, Agent Execution, Trajectory Collection"), Grading ("evaluates the audit log and agent output against C through five sequential steps").

Code vs LLM: Parser and Generator are LLM ("LLM-based multi-agent system"); Validator is deterministic code ("Format Check," "Coverage Check").

Failed-artifact handling, an explicit numeric cap not seen in R12's other pipelines: "If the generated environment fails any check, ClawEnvKit automatically retries generation up to three times before discarding the task."

Quality measurement: Validity (structural), Coherence (LLM judge), Clarity (1-5 scale). Note this is the one pipeline in this set that lets an LLM judge stand as part of the accepted-quality signal rather than only as an offline audit, which section 4 principle 11 below flags as a documented risk.

Generator improvement over iterations: none found — "the system lacks explicit self-improvement loops," per the extraction; quality is reported as static ("match or exceed human-curated ones") rather than trending.

### 2.6 What the 2026 additions change about R12's cross-cutting patterns

R12 section 3 pattern 5 said difficulty calibration by pass rate is "now standard." EnvRigger, EvoEnv, and General Agent all confirm this but split into two families that matter for the founder's Runner/Builder split:

- **Prompted, validated, retried, never gradient-trained**: EnvRigger, ClawEnvKit, General Agent, and everything in R12 except EvoEnv. The Builder is an LLM-in-a-loop with code gates; improvement across iterations is corpus-level (better seeds, better prompts, human-in-the-loop prompt fixes) not weight-level.
- **Gradient-trained against the same signal that grades the output**: only EvoEnv, and only because its domain (single-turn reasoning tasks with a Python-checkable scorer) has no external ground truth to protect and no real production system to stay faithful to. This is the one credible counterexample to "the generator must not be trained against a moving target," and it works specifically because EvoEnv freezes the *scoring mechanism* per accepted environment even while it trains the *policy that proposes environments*, which is a narrower freeze than EnvHarness/EnvRigger's "the verifier never changes at all."

---

## 3. Mapping learn-harness-engineering principles onto Runner (frozen) vs Builder (improving)

| # | Principle (source) | Runner (a) | Builder (b) | Neither (c) | Why |
|---|---|---|---|---|---|
| 1 | Five-subsystem harness model (Lec 2) | | x | | The Runner's five subsystems (its instructions, tools, environment, state, feedback) are themselves the output the Builder must produce and freeze; the Builder is the thing whose five subsystems keep changing. |
| 2 | Repo as system of record, ACID state (Lec 3) | | x | | The Builder's artifacts (ToolSig, EntitySchema, Constraint, Environment) are exactly the "knowledge that must be written down" the lecture describes; the Runner reads them but writes none of them. |
| 3 | Dedicated initialization phase (Lec 6) | x | x | | The Runner needs a clean, deterministic environment snapshot before each Run (its own "startup readiness checklist"); the Builder's `ingest`/`mine`/`compile_env` stages are themselves a dedicated initialization phase before any task can be attempted. |
| 4 | Feature lists as harness primitives, pass-state gating (Lec 8) | | x | | There is no analogue to "features" in the Runner; the Builder's task/constraint set is the environment-generation equivalent, and it must be gated the same way (state changes only on verification, not on the agent's say-so). |
| 5 | Verification-validation dual gate, don't self-grade (Lec 9) | x | x | | Both: the Runner's Verdict must never be the cheap model's own opinion (this is the whole point of "verdict on end state, not path or reasoning," ADR 0004); the Builder's `validate.py` gates must never be the generator LLM's own opinion either. |
| 6 | Only end-to-end testing proves absence of defects (Lec 10) | x | x | | The Runner's Verdict *is* an end-to-end test of a tool-use trajectory; the Builder must E2E-test each generated Environment by oracle replay before accepting it (R12 step 7-9). |
| 7 | Observability inside the harness, sprint contracts, rubrics (Lec 11) | x | x | | The Runner must log every event a cheap model saw (needed for Verdict reproducibility); the Builder must log why each artifact was accepted or rejected (needed to debug the generator, not the agent under test). |
| 8 | Clean state at session end, quality documents, periodic harness simplification (Lec 12) | | x | | This is a statement about *iterating a harness over calendar time*; the Runner is explicitly not supposed to iterate, so "simplify it monthly" does not apply to the Runner at all. It is exactly the discipline the founder wants applied to the Builder. |
| 9 | Loop engineering, maker/checker separation, external state (Lec 13) | x | x | | The generator/evaluator split is the same idea as "cheap model vs frozen Verdict," so it applies to the Runner's internal design once (not iteratively); it applies to the Builder repeatedly, since the Builder's compile-validate-repair cycle is itself a maker/checker loop that runs every time an artifact is generated. |
| 10 | Graph engineering, rollback edges, anchors (Lec 14) | | x | | The Runner is deliberately a single fixed path (ingest a Run, compute a Verdict), not a graph with branches; the Builder's pipeline (mine → compile → validate → repair-loop-back-to-compile) is exactly the rollback-edge graph the lecture describes, and needs an "anchor" (real trace replay) to stay pinned to production reality. |
| 11 | Claude Code: layered instructions by scope, hooks as deterministic checks separate from self-eval | x | | | Hooks-as-independent-check maps directly onto `verdict.py` never being a model call; layered CLAUDE.md-style instruction scoping has no clean analogue in either component. |
| 12 | Codex: AGENTS.md as directory page, enforce invariants not implementation, worktree isolation | x | x | | Worktree-style isolation maps onto the Runner (each Run executes against its own environment snapshot); "enforce invariants, don't micromanage" maps onto the Builder's `policy.py` (compile rules to predicates, leave everything else to the generator). |
| 13 | Codex: send only context deltas, subagents for isolation | x | | | This is a token-cost optimization for a chat-style agent loop; it applies, if at all, inside the cheap model's own loop (which we don't control), not to Builder or Runner design. |
| 14 | Pi: minimal core + programmable extension points, pluggable compaction, session tree | | x | | The Builder-as-pipeline-of-swappable-stages design (R24's `mine.py`, `compile_env.py`, `policy.py` as separate modules) is exactly Pi's "minimal core + extensions" idea; the Runner has no compaction problem since a Run's trajectory is bounded and stored, not compacted. |
| 15 | DeepSeek Harness: capability seams (Service Definition/Provider/Consumer) | | x | | The Builder's `route.py` priority order (code, then recording, then LLM stand-in) is already a capability seam; formalizing tool "Providers" this way is a Builder-design improvement, not a Runner one. |
| 16 | DeepSeek Harness: every loop step as an event point | x | | | This maps onto the Runner's `loop.py`/`route.py` emitting one Event per step, which R24 already specifies; it is a property the frozen Runner must have, not something the Builder needs. |
| 17 | DeepSeek Harness: "model-visible means logged" runtime invariant | x | | | This is precisely the guarantee a re-execution Runner needs for reproducible Verdicts: every token the cheap model saw must be reconstructable from the stored Run. |

### Principles that only the RL environment-generation literature states (not present, even implicitly, in the coding-agent harness course)

1. **Ground truth must come from replaying an executed path in the real (compiled) engine, never from an LLM generating the target state.** No coding-agent harness needs this because a coding agent's "ground truth" is a human-written test suite, not a synthesized target state. Source: R12 pattern 1 (LOGIGEN, AutoForge, ScaleEnv, AgentScaler, Envs-FORGE, CUA-Gym); the sole exception, Simia, is flagged as paying for the shortcut with an LLM-judged reward.
2. **The verifier/reward mechanism is explicitly frozen while everything else about the environment changes.** EnvHarness/EnvRigger states this as its central design point ("success is the benchmark's own verdict, so reshaping reward cannot move the eval metric"); no coding-agent harness in section 1 makes freezing the grader an explicit, load-bearing design decision the way EnvRigger does, because none of them are grading a benchmark whose score must stay comparable across models.
3. **Difficulty is calibrated to a numeric pass-rate band, empirically measured, before a task enters the pool.** Envs-FORGE's target 0.5, EvoEnv's target 0.3, General Agent's per-tier bands, RLAnything's [0.2, 0.8], SPADE's [0.4, 0.6]. Coding-agent harness lectures talk about "granularity" (Lecture 8: "completable in one session") but never a measured numeric target.
4. **The generator itself can be gradient-trained against the same signal used to score its output, with an explicit anti-collapse mechanism (novelty cache, seed floor).** EvoEnv's role-conditioned GRPO plus its 0.2 seed-representation floor. No coding-agent harness trains itself with RL; "harness simplification" (Lecture 12) is a human-driven ablation, not a learned one.
5. **Deliberately manufactured traps and boundary-adjacent states, tested for whether they actually discriminate.** LOGIGEN's boundary-adjacent seeding, ScaleEnv's distractors, R12 step 9c's "wrong-but-plausible path" test. Coding-agent harnesses have no concept of a task designed to be almost-violated on purpose.
6. **Rejected artifacts are discarded, not stored, as a default policy.** R12 section 3 cross-cutting pattern: "nothing is cached across runs except the accepted artifacts themselves, and rejects are discarded rather than stored." This is the opposite instinct from every coding-agent harness in section 1, all of which treat "append-only, nothing thrown away" as close to sacred (Tau, pi, Claude Code sessions, OpenHands event stream).
7. **Human audit is measured as an explicit agreement-rate target against a labeled sample, not just "read the transcripts."** R12 step 9f's ">90% agreement, else fix the generator prompt," tau3's "two reviewers... independently audited," "only 4 of 194 annotated trajectories contained task-critical user errors." Lecture 11's evaluator-rubric idea is a qualitative cousin of this but never states a numeric target.
8. **Reward is explicitly decomposed into column classes (exempt/hard/semantic) and separately validated components, with LLM judges excluded from the training signal by design.** ScaleEnv, tau2's composite reward, "Restrict LLM judges to qualitative criteria only" (2607.02577). No coding-agent harness needs a "column class" concept because none of them are diffing a database.

---

## 4. Principles for an environment-generation harness

1. **Ground truth by execution, never by generation.** Compile the environment first, then have an oracle solve each task inside it and snapshot the resulting state as the target; never ask a model to author the target state directly. *Sources*: R12 pattern 1 (LOGIGEN, AutoForge, ScaleEnv, AgentScaler, EnvFactory); ClawEnvKit's gold solution requirement. *Dissent*: Simia skips this and is markedly cheaper (5k seeds to 90k trajectories), at the cost of an LLM-judged, non-verifiable reward — acceptable only where no real system exists to replay against.

2. **The Runner (grader) is frozen relative to the object under comparison; only the environment supply is allowed to change.** *Sources*: EnvHarness/EnvRigger, quoted twice in section 2.2 ("success is the benchmark's own verdict... every reshaped environment retains its original verifier"); tau2's `evaluate_simulation` as a separate, stored-data pass (R24 section 2.2). *Dissent*: EvoEnv explicitly trains its generator against the same pass-rate signal that scores its output, and this works — but only because its domain (single-turn reasoning with a Python-checkable scorer) has no real production system to stay faithful to; the moment fidelity to a real customer's tools and policy matters, as it does for us, this dissent stops applying.

3. **Environment quality is measured against a numeric target, stated before generation starts, not assumed from design intent.** *Sources*: Envs-FORGE's MILP target 0.5, EvoEnv's target 0.3, General Agent's per-tier pass-rate bands confirmed empirically, RLAnything's [0.2, 0.8] band. *Dissent*: AgentSynth validates only by a difficulty gradient across levels (18% to 4%) rather than a per-task verifier; cheaper, but does not tell you whether any individual task is well-formed.

4. **Bound the repair loop and discard on failure rather than repairing indefinitely or silently keeping a broken artifact.** *Sources*: EnvFactory's revision budget, Envs-FORGE's "rejected records and intermediate artifacts do not enter RL training," ClawEnvKit's explicit "retries generation up to three times before discarding," EvoEnv's L1-L5 any-reject rule. *Dissent*: R12 section 4's own minimal-pipeline recommendation keeps a "residual" list for policy rules that fail to compile rather than discarding them outright, on the argument that partial salvage is worth a human's ten minutes when regeneration is expensive; total discard and partial-salvage-with-human-review are both defensible, and the choice should depend on how expensive a fresh generation attempt is.

5. **Compile policy to executable predicates wherever possible; leave only genuinely uncompilable rules to prose plus a human-reviewed residual list.** *Sources*: LOGIGEN's SQL triggers, APIGen-MT's policy-to-unit-test translation. *Dissent*: tau2 and ARE leave policy as prose for the model to read and grade violations with a mix of hard checks and an LLM judge on free text, which is far cheaper to build but reopens judge-hacking risk (RubricForge's 11-17% false-pass rate, AgenticAI-Supervisor's ~40% constraint misrepresentation under outcome-only reward).

6. **Validate the verifier itself, not just the environment, before it enters the pool: oracle passes, empty run scores below oracle, a plausible-but-wrong path fails, a second model's valid alternative path still passes, and the instruction is grepped for leaked constants.** *Sources*: R12 step 9 (a-e) in full; CUA-Gym's forced `r(s_init) < r(s_gold)`; Agent-World's Pass@5 tolerance; LOGIGEN's spoiler-free instruction rule. *Dissent*: none found as a documented disagreement — every credible pipeline in R12/R22 does some version of this; the only variation is how much of it is automated (Envs-FORGE, fully) versus human (SWE-smith, ~20 hours; tau3, "two reviewers... independently audited").

7. **Human input should be thin and front-loaded (seed docs, personas, theme review, an audited sample of outputs), not full-time authorship of individual artifacts.** *Sources*: R12 pattern 6; AutoEnv's "human theme review lifted yield 60% to 80%" as evidence a small human touch pays off disproportionately. *Dissent*: Mechanize's "Cheap RL tasks will waste compute" argues the opposite for frontier work — good tasks come from "full-time domain experts who spend months," and CoreCraft/APEX-Agents both use expert-authored worlds. This dissent matters less for us specifically because real customer traces substitute for the expert authorship Mechanize is arguing for; it matters more if we ever generate environments for a domain we have no traces for.

8. **The Builder may be improved by reading failure patterns and rewriting the generator's prompts, checklists, or seed corpus, but should not be closed-loop RL-trained against the Runner's own Verdicts.** *Sources*: Agent-World's diagnosis agent that "reads per-task failure traces... and writes guidelines for the next generation round"; AgentOmnia's failures-become-PRDs loop; General Agent's corpus-level ratcheting (hardest tier seeds the next wave). *Dissent*: EvoEnv is the direct counterexample (a generator trained by RL against a solver's pass rate, i.e. against the same metric it is judged by) and reports it works — but note even EvoEnv freezes the per-environment scoring function once admitted; the dissent is really "train the proposer, never the scorer," which is compatible with principle 2, not opposed to it.

9. **Traps and distractors are manufactured deliberately, and every trap is tested to confirm it actually discriminates rather than being decorative.** *Sources*: LOGIGEN's boundary-adjacent seeding, ScaleEnv's "10+ tools, only 3-7 needed," R12 step 9c. *Dissent*: this principle is stated everywhere but not always checked; the tool-calling validity audit (arXiv 2607.02577) found real disagreement rates on exactly the pipelines that claim to do this (MCP-Atlas 13.5%, tau2 Retail 9.8%), so "we manufacture traps" is not itself evidence the traps work.

10. **Version and content-hash the environment (and every artifact that produced it) the way ordinary software is versioned, so any Verdict can name exactly which Environment version it was computed against.** *Sources*: Prime Intellect hub's pyproject-versioned `load_environment`, OpenEnv's Docker manifest, Harbor's `task.toml` version, R24 principle 10. *Dissent*: none found as a stated disagreement, but the reported per-environment generation cost (Envs-FORGE 22.7k-28.8k tokens, ScaleEnv ~546k tokens per domain) argues against versioning at too fine a grain if each new version requires a full regeneration pass rather than an incremental patch.

11. **Decompose the reward/Verdict into independently checkable components (state diff by column class, communicated information, policy violations, side effects), and exclude LLM judgment from anything that feeds the training or comparison signal.** *Sources*: tau2's composite reward, ScaleEnv's exempt/hard/semantic column classes, "Restrict LLM judges to qualitative criteria only" and "Deploy deterministic state gates for factual verification before LLM judging" (arXiv 2607.02577). *Dissent*: MCP-Atlas and CoreCraft use claim-level or rubric-level LLM judging as their primary reward and report it works at their reported scale — but the same validity audit that flagged tau2's and BFCL's error rates also found MCP-Atlas's evaluator-human disagreement rate at 13.5%, so the dissenting approach carries a documented, non-trivial error rate that the decomposed-and-code-only approach does not.

12. **Where real production traces exist, ground the environment in them rather than in documentation or reverse-inferred themes, and measure the resulting fidelity gap rather than assuming traces are better.** *Sources*: MCP-Persona and ToolOmni's 93.8% vs 53.3% F1 with traces versus schema-plus-docs-only (R22 section 3.6, drawing on their own R22 sections 4.1-4.2). *Dissent*: the trade is real, not one-sided — EnvFactory and Agent-World, both doc-grounded, scale to hundreds or thousands of domains, which a trace-grounded pipeline structurally cannot do (traces are customer-specific); trace-grounding buys fidelity to one real system at the cost of never generalizing across many.

---

## 5. What this changes for our Builder and Runner

Deltas against R24's module list and data records (`records.py`, `ingest.py`, `mine.py`, `compile_env.py`, `policy.py`, `user_sim.py`, `loop.py`, `route.py`, `verdict.py`, `regrade.py`, `validate.py`, `pipeline.py`, `cli.py`):

- **Make "Runner frozen" a checkable invariant, not a convention.** Add a `RunnerVersion` record (content hash of `loop.py` + `route.py` + `verdict.py` + the routing-priority config) to `records.py`. Add a `runner_version` field to the existing `Verdict` record (R24 does not currently carry one). Any comparison across models is only valid within one `RunnerVersion`; changing `loop.py`, `route.py`, or `verdict.py` mid-comparison must be a hard error in `pipeline.py`, enforced the same way DeepSeek Harness's "model-visible means logged" is enforced: as a runtime assertion, not a code-review rule (`learn-harness-engineering/docs/en/harness-designs/deepseek/index.md`).

- **`validate.py`: add the R12 step-9 verifier-validation suite as first-class gates, not an afterthought.** Oracle-passes, null-run-scores-below-oracle, wrong-but-plausible-path-fails, second-model-cross-check, and instruction-leakage-grep should each be a named `GateResult`, run automatically before an `Environment` version is marked usable. Add a human-audit `GateResult` with a numeric target (>90% agreement on a 10% sample, per R12 step 9f and tau3's two-reviewer audit practice) rather than leaving audit as an unstructured "read some transcripts" step.

- **`policy.py`: keep a `residual` list for uncompiled rules, surfaced to a human, not silently dropped.** This is section 4 principle 4's dissent made concrete: total discard (Envs-FORGE, ClawEnvKit) is right for generated tool code where regeneration is cheap; for policy rules extracted from a real customer's policy doc, a short human-reviewed residual list (already in R24) is the better default, because losing a policy rule silently is worse than losing a generated tool.

- **`compile_env.py`: adopt EvoEnv's staged-gate structure (parseable, executes, deterministic, non-trivial, scorer contract) as explicit sequential `GateResult`s**, replacing the single "replay held-out calls" gate currently specified with five narrower gates in the same order, so a failure localizes to a specific property (syntax vs determinism vs scorer contract) rather than one opaque "replay failed."

- **`compile_env.py`: add deliberate trap/distractor generation plus a discrimination test, per section 4 principle 9.** For each generated distractor entity or boundary-adjacent state (LOGIGEN style), run the R12 step-9c wrong-but-plausible-path check specifically against it; a trap that a deliberately-told-to-skip-the-trap oracle still passes must be rewritten or dropped before the task enters the pool, not just flagged.

- **`mine.py`: record an evidence-strength field per `ToolSig`, distinguishing trace-observed from LLM-proposed schemas, and measure the fidelity gap when both exist.** This operationalizes section 4 principle 12 (MCP-Persona's 93.8% vs 53.3% F1 finding): if a tool's result schema is LLM-proposed because observed evidence was thin, that should be a visible quality flag on the `ToolSig` record, not an invisible detail.

- **`pipeline.py`: model the Builder explicitly as a graph with a rollback edge, not a linear script.** Per lecture 14's distinction, the current sequential description (ingest → mine → compile_env → policy → build → oracle-replay → loophole-probe) already has an implicit rollback edge (a failed gate sends the artifact back to LLM regeneration with a failure summary, as in Envs-FORGE and EnvFactory); make that edge explicit in `pipeline.py`'s control flow and log it as an edge traversal, so a human auditing the Builder's decisions can see "regenerated attempt 2/3 because gate `replay_fidelity` failed" rather than inferring it from raw stage logs. Attach one "anchor" per lecture 14 (a fixed sample of real held-out trace replays that must keep passing) so Builder iteration cannot drift away from the customer's actual production behavior while chasing an internal pass-rate target.

- **`pipeline.py` / whole Builder: adopt lecture 12's monthly ablation discipline as the concrete mechanism for "Builder should keep improving."** Once a month, disable one Builder gate or generation strategy and re-run the fixed anchor sample plus the human-audit sample; if quality (replay fidelity, human agreement rate) does not drop, retire that gate; if it drops, keep it or replace it with something cheaper that catches the same failures. This gives the founder's "keep improving the Builder" instruction a bounded, measured mechanism instead of open-ended tinkering, and keeps the improvement loop entirely on the Builder side — never touching `loop.py`, `route.py`, or `verdict.py`.

- **`route.py` / `records.py`: implement "model-visible means logged" as a runtime-checked invariant on the `Event` record**, not just a convention: any content that appears in a `model_call` event's prompt must be traceable to a prior `tool_result`, `user_turn`, or the initial environment state, checked by a small assertion in `pipeline.py`'s regrade path. This is the direct import of DeepSeek Harness's strongest single design decision (`learn-harness-engineering/docs/en/harness-designs/deepseek/index.md`) and it is what makes a `Verdict` defensible after the fact.

- **`cli.py`: add `freeze-runner` and `build --iterate` as distinct, differently-permissioned commands.** `freeze-runner` snapshots the `RunnerVersion` record described above and should require explicit confirmation, mirroring how a coding harness treats a production deploy versus a dev branch (Codex's plan-mode approval gate before high-risk operations is the closest coding-agent analogue). `build --iterate` runs the Builder pipeline again to produce a new `Environment` version gated by `validate.py`, and is the command that is expected to run often and change frequently, exactly the asymmetry the founder specified.

- **`regrade.py`: extend to support "environment regrade" in addition to "verdict regrade."** R24 already lets `regrade.py` recompute a `Verdict` from a stored `Run` against a new `verdict_version` without re-executing. Add the symmetric case: re-running `validate.py`'s oracle/null/leakage/human-audit gates against a new `Environment` version without touching any stored `Run`, so Builder iteration can be checked cheaply before spending cheap-model comparison budget on an environment that has not yet earned its way past the gates.

- **What stays exactly as R24 specified it, because nothing in this research argues against it**: the `route.py` priority order (code, then recording, then LLM stand-in, in that order, with the route recorded on the event); excluding LLM judges from the Verdict's training signal (section 4 principle 11, already R24's design); the flat-JSONL-per-Run shape with `parent_run_id` for branching instead of a session tree (R24's dissent 6, still correct — Pi's and Tau's tree structure buys branching a re-execution Runner does not need).
