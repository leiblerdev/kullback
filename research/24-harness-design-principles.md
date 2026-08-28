# R24. Principles of good harness design, from the people who built the small ones (2026-08-27)

Purpose: what builders of agent harnesses, evaluation harnesses and environment synthesis pipelines say makes a harness good, and how their smallest working versions are actually structured. Ends with a concrete design for our environment creator and re-execution harness.

## Coverage caveats

- Fetched directly: Mario Zechner's pi posts (mariozechner.at), pi session-format docs (pi.dev), Tau dev-notes under huggingface/tau (design/01, 02, 03, 04, 05 and ADR 0002), Anthropic "Building effective agents" and "Writing tools for agents", Claude Agent SDK overview and sessions docs, OpenAI Agents SDK docs, smolagents README and conceptual guide, Pydantic AI overview, testing and message-history docs, mini-swe-agent README, FAQ and `agents/default.py`, SWE-agent paper (arXiv 2405.15793 HTML), OpenHands paper (2407.16741 HTML), Harbor docs (tasks, task tutorial, regrade) and AGENTS.md, Inspect docs (overview, solvers, scorers, scoring workflow, eval logs, caching, scorer reference), tau2-bench `orchestrator.py`, `data_model/simulation.py`, `run.py`, `evaluator/evaluator.py` and the `tests/` tree, OpenEnv README, verifiers AGENTS.md, Prime Intellect "create environment" docs, ARE paper (2509.17158 HTML), lm-evaluation-harness paper (2405.14782 HTML), OpenAI evals `build-eval.md`, Halluminate Westworld post, Envs-FORGE (2608.14312), EnvFactory (2605.18703), Agent-World (2604.18292), AgentSynth (2506.14205), Hamel Husain "show me the prompt", Armin Ronacher "Agent design is still hard", Mitchell Hashimoto "My AI adoption journey", the Hacker News thread on Octomind's LangChain post, and arXiv 2606.25447 (harness design vs post-training).
- Not reachable: OpenAI's "Harness engineering" post (403; only the search snippet "Humans steer. Agents execute." is used), Octomind's original blog (DNS failure on both domains; quotes come from the Hacker News thread and search snippets, which I mark as such), verifiers `docs/environments.md` (404; the v1 `Taskset/Toolset/Judge` API is already in R17 section 3), OpenEnv RFC 001 (404; README and R17 section 2 used instead), OpenHands SDK architecture docs (page did not render), Fleet's site (marketing only, no design content).
- Already covered elsewhere and not repeated: full tau2 mock source and reward code (R17 sections 1.10 and 1.11), OpenEnv echo_env (R17 section 2), synthesis pipeline comparison LOGIGEN/AutoForge/ScaleEnv/EnvScaler (R12), tau3 bug taxonomy and "what a good environment looks like" (R22 sections 1.5 and 3), MCP-Persona and ToolOmni (R22 sections 4.1 and 4.2).
- Lines of code: reported only where the authors state a number. Otherwise "not stated". I did not count repositories myself.
- Quotes are verbatim, including their punctuation. Summaries produced by the fetch tool are marked "(fetch summary)" and are not treated as quotes.

---

## 1. Agent harnesses

### 1.1 pi (Mario Zechner), mariozechner.at and pi.dev

What makes it good, in the author's words:

```
"if I don't need it, it won't be built. And I don't need a lot of things."
"exactly controlling what goes into the model's context yields better outputs"
"I want to inspect every aspect of my interactions with the model"
"pi's system prompt and tool definitions together come in below 1000 tokens."
"As it turns out, these four tools are all you need for an effective coding agent."  (read, bash, edit, write)
"MCP servers are overkill for most use cases, and they come with significant context overhead"
"you have zero visibility into what that sub-agent does. It's a black box within a black box"
"to-do lists generally confuse models more than they help"
"pi runs in full YOLO mode and assumes you know what you're doing"
https://mariozechner.at/posts/2025-11-30-pi-coding-agent/
```

From the earlier "prompts are code" essay, the determinism and context-control stance:

```
"Prompts are code, .json/.md files are state"
"We want reproducible workflows. We want determinism, as much as possible within the limits of these inherently non-deterministic models."
"No tokens and turns wasted. The context contains only the information we need."
"Determinism: The same inputs always produce the same plan."
"The `portingState` field is crucial: it's how the LLM tracks what's been done across sessions."
https://mariozechner.at/posts/2025-06-02-prompts-are-code/
```

Persistence, from the session format doc:

```
"Sessions are stored as JSONL (JSON Lines) files. Each line is a JSON object with a `type` field."
"Session entries form a tree structure via `id`/`parentId` fields, enabling in-place branching without creating new files."
"`buildContextEntries()` walks from the current leaf to the root, producing the active entry list while honoring compaction."
custom: "Extension state persistence. Does NOT participate in LLM context"
custom_message: "Extension-injected messages that DO participate in LLM context"
https://pi.dev/docs/latest/session-format
```

- Core LOC: not stated in the posts fetched. The author's size claim is in tokens (system prompt plus tool definitions under 1,000).
- Abstractions: Tool (four built in), Session entry (typed JSONL line with id/parentId), Context (rebuilt from leaf to root), Extension (custom entries), Provider abstraction (own, no MCP).
- Tools: four hand-written tools; everything else is bash.
- State: append-only JSONL tree; compaction entries are checkpoints.
- Replay/resume: walk parent links from the active leaf; branching is "another child of the same parent" in the same file.
- Tests: not stated in fetched material.

### 1.2 Tau (huggingface/tau), dev-notes only

Only the design notes that add principles beyond what we already had (three layers, about 1,500 provider-neutral lines, tool dataclass, hooks, JSONL, UI consumes events):

```
"Owns the portable agent brain: messages, tools, events, the agent loop, harness, and session primitives."  (tau_agent)
"Every meaningful step should be observable through events so print mode, Rich rendering, and Textual can share the same core."
"The provider layer, agent loop, harness, tools, sessions, and UI should not exchange provider-specific objects."
"These models are intentionally small and provider-neutral. Provider adapters translate Anthropic, OpenAI-compatible, OpenAI Codex subscription, or other API payloads into Tau types before the agent loop or frontends see them."
https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/design/01-architecture.md
https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/design/02-agent-loop.md
https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/design/05-core-types-and-events.md
```

Agent loop responsibilities, verbatim list:

```
"Receive the current system prompt, transcript, tools, and model selection."
"Ask the provider to stream a response."
"Emit events as text and tool calls arrive."
"Collect the assistant message."
"Execute requested tools."
"Append tool results."
"Continue until the assistant produces no more tool calls."
https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/design/02-agent-loop.md
```

Sessions:

```
"an append-only session tree. Instead of mutating old state, Tau appends entries and reconstructs state by replaying them."
"Persistence is push-based: the coding session subscribes a listener to the harness, and every `message_end` notification appends that message to storage"
"the first user prompt is branchable while the assistant is still responding"
https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/design/04-sessions.md
```

Tool result shape (fetch summary of 03-tools.md): `AgentToolResult` has `ok`, `content` ("text that can be sent back to the model"), `data` ("optional structured metadata for UIs, logs, or future integrations"), `error`. The edit tool "validates every replacement before writing. If any edit fails validation, the file is left unchanged."

ADR 0002 is a small but telling decision: tool docs stay hand-written, "CI stays simple because the docs build remains markdown-only." (https://raw.githubusercontent.com/huggingface/tau/main/dev-notes/adr/0002-keep-tool-docs-hand-written.md)

Event vocabulary: `agent_start, agent_end, turn_start, turn_end, queue_update, retry, message_start, message_delta, thinking_delta, message_end, tool_execution_start, tool_execution_update, tool_execution_end, error`.

### 1.3 Anthropic, "Building effective agents" and "Writing tools for agents"

```
"the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."
"they often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug."
"If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error."
"think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good agent-computer interfaces (ACI)."
"Maintain simplicity in your agent's design. Prioritize transparency by explicitly showing the agent's planning steps. Carefully craft your agent-computer interface (ACI) through thorough tool documentation and testing."
https://www.anthropic.com/engineering/building-effective-agents
```

```
"More tools don't always lead to better outcomes. A common error we've observed is tools that merely wrap existing software functionality or API endpoints."
"Tool implementations should take care to return only high signal information back to agents. They should prioritize contextual relevance over flexibility."
"Agents also tend to grapple with natural language names, terms, or identifiers significantly more successfully than they do with cryptic identifiers."
"Even small refinements to tool descriptions can yield dramatic improvements."
"Start by generating lots of evaluation tasks, grounded in real world uses."
https://www.anthropic.com/engineering/writing-tools-for-agents
```

The second post matters for us in reverse: our tools are not designed, they are recovered from traces. "tools that merely wrap existing software functionality or API endpoints" is exactly what a customer's production tools usually are, and we must reproduce them faithfully rather than improve them (the cheaper model is being judged against the same interface the expensive one had).

### 1.4 Claude Agent SDK

```
"An agent is an application that completes a task by planning its own steps and calling tools that read files, run commands, or edit code. The Agent SDK gives you the same tools, agent loop, and context management that power Claude Code, programmable in Python and TypeScript."
"A session is the conversation history the SDK accumulates while your agent works. It contains your prompt, every tool call the agent made, every tool result, and every response. The SDK writes it to disk automatically so you can return to it later."
"Sessions persist the conversation, not the filesystem."
"Fork is different: it creates a new session that starts with a copy of the original's history. The original stays unchanged."
"Don't rely on session resume. Capture the results you need (analysis output, decisions, file diffs) as application state and pass them into a fresh session's prompt. This is often more robust than shipping transcript files around."
"Claude Code stores sessions under `~/.claude/projects/<encoded-cwd>/*.jsonl`."
https://code.claude.com/docs/en/agent-sdk/overview
https://code.claude.com/docs/en/agent-sdk/sessions
```

- Abstractions: query, Session (JSONL transcript), Hooks (PreToolUse/PostToolUse), Subagents, Permissions.
- LOC: not stated (closed core).
- Replay/resume: `resume=<id>`, `fork_session=True`, `continue`.

### 1.5 OpenAI Agents SDK

```
"Enough features to be worth using, but few enough primitives to make it quick to learn"
"Works great out of the box, but you can customize exactly what happens."
Agents: "LLMs equipped with instructions and tools"
Guardrails: "Enable validation of agent inputs and outputs"
Sessions: "A persistent memory layer for maintaining working context within an agent loop"
Function tools: "Turn any Python function into a tool with automatic schema generation and Pydantic-powered validation."
https://openai.github.io/openai-agents-python/
```

- Abstractions: Agent, Handoff, Guardrail, Session, Tracing (plus function_tool).
- LOC: not stated.
- State: Session backends (SQLite and others); tracing spans exported.

### 1.6 smolagents (Hugging Face)

```
"the logic for agents fits in ~1,000 lines of code"
"abstractions kept to their minimal shape above raw code."
"AI Agents are programs where LLM outputs control the workflow."
"For some low-level agentic use cases, like chains or routers, you can write all the code yourself. You'll be much better that way, since it will let you control and understand your system better."
"For the sake of simplicity and robustness, it's advised to regularize towards not using any agentic behaviour."
https://github.com/huggingface/smolagents (README)
https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents
```

The canonical loop, verbatim from the guide:

```python
memory = [user_defined_task]
while llm_should_continue(memory): # this loop is the multi-step part
    action = llm_get_next_action(memory) # this is the tool-calling part
    observations = execute_action(action)
    memory += [action, observations]
```

The guide also lists the minimum set of parts a multi-step agent needs: "An LLM that acts as the engine", "A list of tools", "A system prompt guiding the LLM on the agent logic", "A parser that extracts tool calls from the LLM output", "A memory", plus "error logging and retry mechanisms".

- Abstractions: Agent (CodeAgent/ToolCallingAgent), Tool, Model (callable), Memory (steps), Executor (sandboxed Python).
- LOC: about 1,000 (author claim).
- Tests: not stated in fetched material.

### 1.7 Pydantic AI

```
"Any model, one Python API."
"Typed end to end. Structured outputs, typed dependency injection, typed tools: your IDE, type checker, and coding agent all know what your agent returns"
"Measured, not vibes. OpenTelemetry-native instrumentation works with any OTel backend"
"One primitive, the capability, bundles tools, instructions, hooks, and model settings into reusable units"
https://pydantic.dev/docs/ai/overview/
```

Testing, which is the part most relevant to us:

```
"The simplest and fastest way to exercise most of your application code is using `TestModel`, this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent."
"If you find yourself typing out long assertions, use inline-snapshot"
ALLOW_MODEL_REQUESTS=False to "block any requests from being made to non-test models accidentally."
capture_run_messages to "inspect messages from the most recent run and assert the exchange between the agent and the model occurred as expected."
https://pydantic.dev/docs/ai/testing/
```

Message history:

```
"The intended way to do this is using a `TypeAdapter`"  (ModelMessagesTypeAdapter, to_jsonable_python / validate_json)
"If `message_history` is set and not empty, a new system prompt is not generated -- we assume the existing message history includes a system prompt."
https://pydantic.dev/docs/ai/message-history/
```

- Abstractions: Agent, Tool (decorated function with RunContext), Model (swappable; TestModel and FunctionModel for tests), ModelMessage (typed, JSON round-trippable), Capability.
- LOC: not stated.
- Replay: message list is data; re-run by passing `message_history`.

### 1.8 mini-swe-agent (Princeton / SWE-agent team)

```
"What if our agent was 100x simpler, and still worked nearly as well?"
"just 100 lines of python"  (core agent; ~100 more each for environment, model, run script)
"Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs."
"Has a completely linear history — every step of the agent just appends to the messages."
"Executes actions with `subprocess.run` — every action is completely independent."
https://mini-swe-agent.com/latest/
https://mini-swe-agent.com/latest/faq/
https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/main/README.md
```

Structure of `agents/default.py` (fetch summary): `AgentConfig` (templates, `step_limit`, `cost_limit`, `wall_time_limit_seconds`, `max_consecutive_format_errors`, `output_path`); `run()` loops `step()` until the last message has role `exit`; `step()` is `query()` then `execute_actions()`; control flow by exceptions (`FormatError`, `InterruptAgentFlow`, `LimitsExceeded`, `TimeExceeded`); environment call is one line: `outputs = [self.env.execute(action) for action in ...]`.

- Abstractions: Agent, Environment (`execute(action) -> output`), Model (`query(messages)`), Config, Messages (a list).
- Tools: none; bash string in, stdout string out.
- State: the message list is the trajectory; saved to `output_path`.
- Replay: the trajectory is the prompt, so any step can be re-issued by truncating the list.
- Tests: codecov integration and the SWE-bench Verified number (over 74%) as the acceptance test.

### 1.9 SWE-agent, "agent-computer interface" paper

```
"Actions should be simple and easy to understand for agents."
"Actions should be compact and efficient."
"Environment feedback should be informative but concise."
"Guardrails mitigate error propagation and hasten recovery."
https://arxiv.org/html/2405.15793
```

### 1.10 OpenHands

```
"the event stream, which is a chronological collection of past actions and observations, including the agent's own actions and user interactions"
agents "perceive the state of the environment (e.g., prior actions and observations) and produce an action for execution"
https://arxiv.org/html/2407.16741
```

- Abstractions: Event (Action | Observation), EventStream, Agent (`step(state) -> action`), Runtime (Docker sandbox with bash, IPython, browser), AgentSkills.
- LOC: not stated.
- State: the event stream is the state; "agent skills" are Python functions imported into IPython.

### 1.11 Harbor (Terminal-Bench harness)

```
"Simple, modular interfaces for environments, agents, and tasks"
tests/test.sh "must produce a reward file in `/logs/verifier/`."
"Harbor does not require any specific file to exist in that directory"  (environment/)
"Prefer explicit `if` checks that raise clear errors over `assert`"
"Prefer `Path.write_text()` / `Path.read_text()` over `with open(...)`"
https://harborframework.com/docs
https://www.harborframework.com/docs/tasks
https://raw.githubusercontent.com/harbor-framework/harbor/main/AGENTS.md
```

Regrade (the one feature we should copy directly):

```
re-score completed trials "with a new (or updated) verifier"
"No agent environment is started and the recorded agent is never re-instantiated, so regrading needs no agent credentials or API keys"
regrade is "a fork of the source trial"
https://www.harborframework.com/docs/run-jobs/regrade
```

- Abstractions: Task (`task.toml`, `instruction.md`, `environment/`, `solution/`, `tests/`), Agent (`BaseAgent`: `name`, `version`, `setup`, `run`), Environment (`BaseEnvironment`), Trial (`config.json` + `result.json` with checksums and per-phase timestamps), Job, Verifier (reward.txt or reward.json).
- Tests: pytest markers `unit`, `integration`, `runtime`; "Avoid testing CLI output formatting; test behavior instead" (fetch summary of AGENTS.md).

### 1.12 Practitioners: Ronacher, Hashimoto, OpenAI

Armin Ronacher, who wrote his own loop rather than using an SDK:

```
"The differences between models are significant enough that you will need to build your own agent abstraction."
"Explicit caching allows you to do certain things that are much harder otherwise. For instance, you can split off a conversation and have it run in two different directions simultaneously."
"Every time the agent runs a tool you have the opportunity to not just return data that the tool produces, but also to feed more information back into the loop."
"The todo write tool is a self-reinforcement tool. All it does is take from the agent a list of tasks that it thinks it should do and echo out what came in."
"Unlike prompts, you cannot just do the evals in some external system because there's too much you need to feed into it."
"A better tool caller will do the job in fewer tokens."
https://lucumr.pocoo.org/2025/11/21/agents-are-hard/
```

Mitchell Hashimoto, the origin of "harness engineering":

```
"Anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."
agents need "fast, high quality tools to automatically tell it when it is wrong."
https://mitchellh.com/writing/my-ai-adoption-journey
```

OpenAI's post (403 on fetch; search snippet only): "Humans steer. Agents execute." https://openai.com/index/harness-engineering/

arXiv 2606.25447 gives the one empirical result on harness design as a variable: "harness-aware post-training not only improves in-distribution performance but also enables agents to robustly adapt to OOD settings" and (fetch summary) minimal design effort in the harness leads to large degradation when the tool environment changes. Relevant to us because the cheaper model we re-execute with was post-trained on some harness, and ours differs.

---

## 2. Evaluation harnesses

### 2.1 Inspect AI (UK AISI)

```
"An Inspect evaluation is a Task that brings together three things"  (dataset, solver, scorer)
Solver: "produces an answer for each sample. This can be as simple as a single generate() call to the model, or as sophisticated as a full agent that uses tools over many turns."
Scorer: "evaluates the output—using text comparisons, model grading, or other custom schemes."
"Composable building blocks—datasets, agents, tools, and scorers—that make evaluations easy to write and reuse."
https://inspect.aisi.org.uk/
```

Minimal interfaces:

```python
async def solve(state: TaskState, generate: Generate) -> TaskState:
async def __call__(self, state: TaskState, target: Target) -> Score | None
# Score fields: value, answer, explanation, metadata
```

```
"A solver is a Python function that takes a TaskState and `generate` function, and then transforms and returns the TaskState."
generate is "a convenience function that takes a TaskState, calls the model with it, appends the assistant message, and sets the model output"
"setting the TaskState.completed field will result in forgoing remaining solvers"
https://inspect.aisi.org.uk/solvers.html
https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html
```

Logs, re-scoring, caching:

```
".eval" is a "Binary file format optimised for size and speed. Typically 1/8 the size of `.json` files."
"images and other large blocks of content are de-duplicated and stored as attachments"
round trip: "eval → log → export-config → eval"
`inspect log export-config` "reads a log file and writes a YAML (or JSON) file that captures the complete configuration used for that run."
"By default, model output in evaluations is automatically scored. However, you can defer scoring by using the `--no-score` option."
Append mode: "The new scores will be added alongside the existing scores in the log file, keeping both the old and new results."
Cache key: "Model name and base URL (e.g. `openai/gpt-5`), Model prompt (i.e. message history), Epoch number (for ensuring distinct generations per epoch), Generate configuration (e.g. `temperature`, `top_p`, etc.), Active `tools` and `tool_choice`."
"when you are iterating on a scorer you may want the model outputs served from a cache to both save time as well as for increased determinism."
https://inspect.aisi.org.uk/eval-logs.html
https://inspect.aisi.org.uk/scoring-workflow.html
https://inspect.aisi.org.uk/caching.html
```

- Abstractions: Task, Dataset/Sample, Solver, Scorer, TaskState, Sandbox, EvalLog.
- The key design fact: the log is a complete record (config, plan, every sample's messages and events, scores), so scoring is a separable pass over stored data.

### 2.2 tau2-bench orchestrator, runner, evaluator (sierra-research/tau2-bench)

Orchestrator (fetch summary of `src/tau2/orchestrator/orchestrator.py`): `BaseOrchestrator` is generic over Agent, User, Environment with a template `run()` = `initialize()` then `step()` loop then `_finalize()`; the half-duplex `Orchestrator` routes one message per step via `from_role`/`to_role`, executes tool calls via `_execute_tool_calls()`, appends to `trajectory: list[Message]` where messages are `AssistantMessage | UserMessage | ToolMessage | MultiToolMessage` each with `timestamp` and `turn_idx`; termination by `max_steps` (100), `max_errors` (10), wall-clock `timeout`, `###STOP###` from agent or user, and protocol violations; an optional `seed` is pushed to agent and user via `set_seed()`; `initialize()` loads `task.initial_state` (`initialization_data`, `initialization_actions`) and calls `environment.sync_tools()`.

Docstring:

```
"Provides the common infrastructure for managing simulations between Agent, User, and Environment. Subclasses implement specific communication patterns: Orchestrator: Half-duplex (turn-based) communication, trajectory of Messages; FullDuplexOrchestrator: Full-duplex (streaming) communication, trajectory of Ticks."
https://raw.githubusercontent.com/sierra-research/tau2-bench/main/src/tau2/orchestrator/orchestrator.py
```

A simulation is a data record (`src/tau2/data_model/simulation.py`, fetch summary): `SimulationRun(id, task_id, trial, timestamp, start_time, end_time, duration, termination_reason, agent_cost, user_cost, agent_usage, reward_info, messages, ticks, mode, review, user_only_review, hallucination_check, auth_classification)`; `RewardInfo(reward, db_check, action_checks, env_assertions, nl_assertions, reward_basis, reward_breakdown)`; `Results(timestamp, info, tasks, simulations, simulation_index)` stored as "Single JSON file with all data" or "metadata in results.json plus individual simulation files in simulations/ subdirectory". Re-evaluation needs only the task, the messages and a fresh environment; all three are preserved.

Runner (`run.py`, fetch summary): three layers, `run_simulation()` executes one orchestrator; `build_agent/build_user/build_environment/build_orchestrator` construct components from a `TextRunConfig`; `run_domain/run_tasks/run_single_task` batch with `max_concurrency`, `seed` (default 300), `max_retries`, `auto_resume`.

Evaluator:

```python
evaluate_simulation(simulation: SimulationRun, task: Task, evaluation_type: EvaluationType, solo_mode: bool, domain: str, mode: CommunicationMode = CommunicationMode.HALF_DUPLEX, env_kwargs: dict = None, strict_replay: bool = True) -> RewardInfo
# EvaluationType: ENV, COMMUNICATE, ACTION, ALL, NL_ASSERTIONS, ALL_WITH_NL_ASSERTIONS, ALL_IGNORE_BASIS, ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS
```

`strict_replay` controls "whether tool output mismatches trigger errors" when the recorded trajectory is replayed into a fresh environment; ENV evaluation rebuilds two environments (gold from `initial_state` plus golden `actions`, predicted from replaying the trajectory's mutating tool calls) and compares DB hashes (full code in R17 sections 1.10 and 1.11).

Tests (`tests/` tree): `test_agent.py, test_checkpoint.py, test_environment.py, test_evaluate_trajectories.py, test_llm_utils.py, test_orchestrator.py, test_results_format.py, test_review_model.py, test_run.py, test_tasks.py, test_user.py, test_utils.py`, plus `test_domains/, test_experiments/, test_gym/, test_runner/, test_streaming/, test_voice/`. Note the presence of `test_results_format.py` and `test_evaluate_trajectories.py`: the data record and the regrade path are tested as first-class units.

### 2.3 OpenEnv (meta-pytorch / huggingface)

```
"a standard for interacting with agentic execution environments via simple Gymnasium style APIs"
reset(): initialize episodes, return initial Observation; step(action): execute Actions, return Observation; state(): episode metadata
https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/README.md
```

(fetch summary) Typed `Action`, `Observation`, `State`, `StepResult` dataclasses; FastAPI server in Docker; WebSocket sessions; `openenv.yaml` manifest; scaffold from `openenv init` with `models.py`, `client.py`, `server/`, `Dockerfile`, `pyproject.toml`; RFCs 001 (baseline API), 002 (tool discoverability), 003 (MCP support), 004 (delayed rewards), 005 (agentic harness integration). Reward is left to the environment author (R22 item 14).

### 2.4 verifiers (Prime Intellect)

From AGENTS.md, the design rules:

```
"Code is the source of truth"
"Minimal config surface"
"Keep tasksets small"
"Prefer e2e tests over unit tests"
https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/AGENTS.md
```

Packaging (Prime docs): a module with `load_environment` that "returns a `vf.Environment` object" and "should encapsulate any necessary preprocessing, resource provisioning, exposing configurable args, etc."; `uv pip install -e .` then `uv run vf-eval my-environment`; `prime env push --auto-bump`; `pyproject.toml` is the version and dependency record. https://docs.primeintellect.ai/tutorials-environments/create

The v1 API (`Taskset`, `Toolset` with `@vf.tool`, `Judge`) is shown in full in R17 section 3; the tool docstring is the tool description, and `TOOL_PREFIX` namespaces tool names.

### 2.5 ARE (Meta, Agents Research Environments)

```
"Apps are collections of tools that interact with a data source."
"An environment is a Markov Decision Process with states, observations, actions, and transition rules."
"runs deterministically given a fixed starting state and seed."
"In ARE, an event is any agent action or app-state change. Each event is timestamped, logged."
"Apart from tool outputs, notifications are the only signals agents receive from the environment."
"We verify scenario successful completion by comparing agent actions with a ground truth, defined as the minimal sequence of write actions needed to solve a task."
https://arxiv.org/html/2509.17158
```

- Abstractions: App, Environment, Event (DAG scheduled), Notification (whitelist policy), Scenario (apps + scheduled events + verification logic).

### 2.6 lm-evaluation-harness, "Lessons from the trenches"

```
"Provide Evaluation Code: Whenever possible, new publications should be accompanied by release of the exact evaluation code used."
"Provide Prompts and Detail Methodology Thoroughly: Failing the release of code or to supplement the release of code, prompts should be clearly reported within a publication."
"Perform Statistical Analyses, and Report on Sources of Variance and Error: Instead of reporting isolated numbers, more effort should be made to contextualize the durability of performance estimates."
"Do Exploratory Analysis to Build Better Understanding of Results: To understand why a model is scoring so well or so poorly, it is important to do some sort of qualitative error analysis."
https://arxiv.org/html/2405.14782
```

(fetch summary) Operationalised as: bootstrap confidence intervals by default, YAML task configs so prompts change without code, `--log_samples` so every model output is kept.

### 2.7 OpenAI evals

```
"All templates expect an `"input"` key, which is the prompt, ideally specified in chat format (though strings are also supported)."
"The eval should be thematically consistent"
"The eval should be challenging. If GPT-4 or GPT-3.5-Turbo do well on all prompts, this is not as interesting."
"The eval should be directionally clear. The data should include good signal around what is the right behavior."
"The eval should be carefully crafted"
https://github.com/openai/evals/blob/main/docs/build-eval.md
```

Registry is YAML mapping `<eval_name>.<split>.<version>` to a class and a `samples_jsonl`. Samples are JSONL. The eval class is the only code.

---

## 3. Environment creators as software

### 3.1 EnvFactory (2605.18703)

(fetch summary, quotes marked) Three LLM agents: Search Agent "plans and sketches candidate environments with authentic external sources"; Code Agent "derives a stateful database schema" and "implements executable Python code"; Test Agent "creates unit test cases and validates the environment". Revision loop checks four things: "tool interfaces are consistent with metadata", "tools import and execute successfully", "execution results match expected behavior", "database states transition correctly after tool invocation". On failure "the Test Agent produces a structured error report that localizes the source" and "the Code Agent then updates the corresponding component and rebuilds the environment. This iterative validation-and-revision loop continues until all tests pass or a maximum revision budget is reached." Artifacts per environment: metadata, database schema, executable Python, tool interface; state uses "Pydantic schemas with standardized serialization interfaces for loading and dumping states". https://arxiv.org/html/2605.18703

Code vs LLM: schema and tool code are LLM-written; the four validation checks are code; the error report is LLM; the budget is code.

### 3.2 Envs-FORGE (2608.14312)

```
"Solve the six-action per-task MILP for seed s_i and select (a_i, d_i)"
"Check its schema, paths, prompt length, overlap, and container isolation"
"Build s~_i,a_i,d_i; run its oracle solution and generated tests."
"A task is accepted only when the verifier emits reward 1"
"Rejected records and intermediate artifacts do not enter RL training"
"Failed verification attempts are excluded from the training set; generation or repair can continue only within the synthesis budget."
https://arxiv.org/html/2608.14312
```

Stages: (1) frontier scoring, code; (2) action selection, code (SCIP MILP); (3) bundle synthesis, LLM (instruction, fixtures, oracle, tests, Docker); (4) static validation, code; (5) executable verification, code (oracle must score 1); repair prompt receives a "failure summary". Output is a Terminal-Bench task directory, so Harbor is the runner.

### 3.3 Agent-World (2604.18292)

```
"a tool is retained only if it satisfies all of the following: the function can be successfully compiled by the Python compiler"
"Acc(f̂; Ĉf̂) > 0.5 on its associated test set"
"we first synthesize a valid tool-call sequence and then generate the corresponding task description"
"We retain the task only if the agent successfully reaches a consistent answer in at least two independent runs"  (of 5)
"Execute(Vcode(y, y*))"
https://arxiv.org/html/2604.18292
```

Code vs LLM: tool code, verification scripts and task text are LLM; compile check, test-set accuracy gate, sandbox execution, Pass@5 consistency are code; a diagnosis agent reads "per-task failure traces (tool logs, intermediate observations)" and writes guidelines for the next generation round.

### 3.4 AgentSynth (2506.14205)

```
"AgentSynth constructs subtasks that are simple during generation but significantly more challenging when composed into long-horizon tasks"
"a low average cost of $0.60 per trajectory, orders of magnitude cheaper than human annotations"
https://arxiv.org/abs/2506.14205
```

Validation is by difficulty gradient (18% at level 1 to 4% at level 6), not by per-task verifier checks.

### 3.5 Halluminate Westworld

```
"Reproducibility: every agent sees the exact same state"
"Determinism: no CAPTCHAs, DOM drift, randomized search results, or data decay"
"Control: ability to scale difficulty, inject edge cases, log every step, and run RL"
"the environment focused on the actual skills we want to measure: search, filtering, reasoning, and multi-step execution."
"RLVR-style unit tests that deterministically evaluate the agent's final state"
https://www.halluminate.ai/blog/westworld
```

(fetch summary) Pipeline: production queries become task templates; templates are populated with synthetic data; dates are computed relative to run time so determinism survives calendar drift. Three verifier types: state-based (backend DB diff vs expected), component-level (structured selectors), real-time calculated (ground truth computed at run time). They simulate only the workflows they want to measure, not the whole app.

### 3.6 Fleet, Prime Intellect hub, MCP-Persona, ToolOmni

- Fleet: site contains no design content ("Fleet1 creates simulated worlds and real-world challenges"). Nothing to cite.
- Prime Intellect hub: an environment is a pip package with a `load_environment` entrypoint, versioned in `pyproject.toml`, tested with `vf-eval`, published with `prime env push`. The packaging principle is "an environment is a versioned Python package with one function", nothing more.
- MCP-Persona and ToolOmni: the only two pipelines that start from real traces; see R22 sections 4.1 and 4.2 (93.8% vs 53.3% F1 with traces plus schema vs docs only; failed calls teach error formats).

Cross-pipeline pattern (all of 3.1 to 3.5): LLM writes text and code; code decides acceptance; a bounded repair loop is the only failure handling; nothing is cached across runs except the accepted artifacts themselves, and rejects are discarded rather than stored.

---

## 4. General software principles, as invoked by harness authors

1. State as an append-only log (event sourcing). Tau: "Instead of mutating old state, Tau appends entries and reconstructs state by replaying them." Pi: "Session entries form a tree structure via `id`/`parentId` fields". OpenHands: "the event stream, which is a chronological collection of past actions and observations". ARE: "Each event is timestamped, logged." mini-swe-agent: "every step of the agent just appends to the messages." Claude Agent SDK: a session "contains your prompt, every tool call the agent made, every tool result, and every response."
2. Determinism and seeded randomness. ARE: "runs deterministically given a fixed starting state and seed." tau2: `seed` propagated via `set_seed()` (default 300 in batch runs). Westworld: "every agent sees the exact same state". Inspect: epoch number in the cache key "for ensuring distinct generations per epoch". Zechner: "We want determinism, as much as possible within the limits of these inherently non-deterministic models."
3. Idempotent, separable stages. Inspect: `--no-score` then `inspect score` with append or overwrite. Harbor regrade: "the recorded agent is never re-instantiated". tau2: `evaluate_simulation(simulation, task, ...)` takes stored data. Envs-FORGE: static validation before executable verification, each a gate.
4. Content-addressed caching of the model call. Inspect cache key = model, message history, epoch, generate config, tools, tool_choice; purpose "increased determinism" while iterating on a scorer. Harbor `result.json` carries checksums per trial. Ronacher: explicit cache points so a conversation can be split "in two different directions simultaneously".
5. The LLM call is the smallest replaceable unit. Inspect passes `generate` as a parameter to the solver. Pydantic AI: `TestModel`, `FunctionModel`, `ALLOW_MODEL_REQUESTS=False`. mini-swe-agent: `Model.query(messages)` behind a config. smolagents: model is a callable.
6. Schema-first tools with docstring descriptions. OpenAI Agents SDK: "automatic schema generation and Pydantic-powered validation". Tau: `AgentTool(name, description, input_schema, executor)`. tau2 and verifiers: schema from signature plus docstring. Anthropic: "Even small refinements to tool descriptions can yield dramatic improvements."
7. Separation of policy from mechanism. Tau's dependency direction `tau_coding -> tau_agent -> tau_ai` and "should not exchange provider-specific objects". ARE notifications governed by "a whitelist of events authorized to emit notifications". Harbor: "Separation of concerns between agents, tasks, jobs, trials, and sandboxes" (AGENTS.md, fetch summary). OpenAI evals: the eval class is code, the samples and registry are data.
8. Boring technology. mini-swe-agent: `subprocess.run` because shell sessions "don't obviously terminate" and "bad LM commands can kill sessions" (FAQ, fetch summary). Pi: no MCP, no background bash, use tmux. Tau ADR 0002: hand-written docs so "CI stays simple". Harbor: `Path.read_text()` over context managers, `if` checks over `assert`.
9. Golden files and snapshot tests. Pydantic AI: "use inline-snapshot". tau2: `test_results_format.py` and `test_evaluate_trajectories.py` pin the data record and the regrade path. Harbor: oracle solution must pass the task's own tests (`harbor run -a oracle`). Envs-FORGE: "A task is accepted only when the verifier emits reward 1".
10. Replayability as a first-class feature, not a debugging afterthought. Inspect: "eval → log → export-config → eval". Harbor regrade as "a fork of the source trial". tau2 `strict_replay`. Pi and Claude sessions fork by copying history.
11. Small hand-written tool surface beats a big generated one. Pi: four tools. mini-swe-agent: one (bash). Anthropic: "More tools don't always lead to better outcomes." Westworld: simulate only the workflows you measure.
12. Feedback quality is part of the harness. SWE-agent: "Environment feedback should be informative but concise." Hashimoto: "fast, high quality tools to automatically tell it when it is wrong." Ronacher: tool results are a place "to feed more information back into the loop."

---

## 5. Anti-patterns named by builders

- Abstractions that hide the prompt. Anthropic: frameworks "obscure the underlying prompts and responses, making them harder to debug." Hamel Husain: "the prompts sent by these tools to the LLM is a natural language description of what these tools are doing, and is the fastest way to understand how they work"; his fix is "a proxy that logs your outgoing API requests" (mitmproxy). https://hamel.dev/blog/posts/prompt/
- Abstractions on abstractions (LangChain, specifics). Octomind (search snippet, original page unreachable): LangChain "intentionally abstracts so many details that it often wasn't easy or possible to write the lower-level code they needed to", and became "a source of friction, not productivity". Hacker News thread on that post (https://news.ycombinator.com/item?id=40739982):

```
sc077y: "the second you need to something a little original you have to go through 5 layers of abstraction just to change a minute detail."
w4: "Most LLM applications require nothing more than string handling, API calls, loops, and maybe a vector DB if you're doing RAG. You don't need several layers of abstraction and a bucketload of dependencies to manage basic string interpolation, HTTP requests, and for/while loops."
causal: "anything that ultimately hides prompts behind code will create more friction than not."
tkellogg: "LangChain isn't usable beyond demos. It feels like even proper logging is pushing it beyond it's capabilities."
```

- Provider abstraction that flattens real differences. Ronacher: "The web search tool from Anthropic routinely destroys the message history with the Vercel SDK"; hence "you will need to build your own agent abstraction."
- Black-box subagents. Zechner: "you have zero visibility into what that sub-agent does. It's a black box within a black box".
- Context bloat from protocol layers. Zechner on MCP: "significant context overhead"; Ronacher and Zechner (Pragmatic Engineer summary): "MCP is a lossy middle layer."
- Non-replayable runs. Every evaluation harness in section 2 exists partly to remove this: Inspect logs, Harbor trial dirs, tau2 `SimulationRun`. lm-eval: "release of the exact evaluation code used" and prompts, because otherwise numbers cannot be reproduced.
- Implicit state. Claude Agent SDK warns "Sessions persist the conversation, not the filesystem", and advises capturing results "as application state" rather than "shipping transcript files around". OpenHands makes state explicit as the event stream. mini-swe-agent avoids stateful shells entirely.
- Tool results mutated or reformatted by the framework. Tau keeps `content` (model-facing) and `data` (UI-facing) as separate fields so rendering never changes what the model saw. Ronacher's caution about SDKs altering message history is the failure case.
- Magic that reads the model's mind. Zechner: "to-do lists generally confuse models more than they help"; no plan mode, "write it to a file". Ronacher on output tools: "Sometimes it just doesn't call the tool."
- Wrapping every endpoint as a tool. Anthropic: "tools that merely wrap existing software functionality or API endpoints." (For us this is a constraint, not a choice; see 1.3.)

---

## Core abstractions across harnesses

| Harness | Core LOC (author claim) | Core abstractions | Tool definition | Persistence | Replay / regrade |
|---|---|---|---|---|---|
| pi | not stated; prompt + tools < 1,000 tokens | Tool, Session entry, Context (leaf to root), Extension, Provider | 4 hand-written tools (read, bash, edit, write) | JSONL tree, id/parentId, append-only, compaction checkpoints | walk parent links; branch = new child, same file |
| Tau | ~1,500 provider-neutral (from prior reading) | AgentHarness, AgentTool, AgentToolResult, Message, Event, Session tree | dataclass: name, description, input_schema, async executor | JSONL append-only tree, push-based listener on `message_end` | replay entries; system prompt rebuilt from current config on resume |
| Claude Agent SDK | not stated | query, Session, Hook, Subagent, Permission | built-in plus MCP | JSONL under `~/.claude/projects/<cwd>/` | `resume`, `fork_session`, `continue` |
| OpenAI Agents SDK | not stated | Agent, Handoff, Guardrail, Session, Tracing | `function_tool` with auto schema, Pydantic validation | Session backends (SQLite etc.), trace spans | resume from session; traces viewable |
| smolagents | ~1,000 | Agent, Tool, Model, Memory (steps), Executor | class or decorator; code actions preferred | in-memory steps; optional Hub push | not a feature |
| Pydantic AI | not stated | Agent, Tool (RunContext), Model, ModelMessage, Capability | decorator, typed signature | `ModelMessagesTypeAdapter` JSON round trip | pass `message_history`; TestModel/FunctionModel for offline runs |
| mini-swe-agent | 100 agent + ~100 each env/model/script | Agent, Environment, Model, Config, messages list | none; bash string | messages list saved to `output_path` | truncate list and re-issue; trajectory browser |
| OpenHands | not stated | Event (Action/Observation), EventStream, Agent.step, Runtime, Skills | Python functions in IPython plus bash and browser | event stream | replay events |
| Harbor | not stated | Task, Agent, Environment, Trial, Job, Verifier | agent-owned; task exposes only tests | trial dir: config.json, result.json (checksums), artifacts/manifest.json | `harbor regrade` reruns verifier on stored artifacts |
| Inspect | not stated | Task, Dataset, Solver, Scorer, TaskState, Sandbox, EvalLog | `@tool` Python functions | `.eval` binary or JSON log with events and attachments | `inspect score` append/overwrite; `export-config` round trip; model cache |
| tau2-bench | evaluator ~450, environment.py 490, tasks.py 685 (R17) | Orchestrator, Agent, User, Environment, ToolKit, SimulationRun, RewardInfo | `@is_tool(ToolType)` methods, schema from signature + docstring | `Results` JSON or `results.json` + `simulations/` dir | `evaluate_simulation` over stored run; `strict_replay` |
| OpenEnv | echo_env in full in R17 | Action, Observation, State, Environment (reset/step/state), EnvClient | MCP tools on a FastMCP server | `State(episode_id, step_count)`; reward left to author | none built in (RFC 004 delayed reward) |
| verifiers | wiki_search ~200 (R17) | Taskset, Toolset, Judge, Task, `load_environment` | `@vf.tool` docstring-described, prefixed | transcript as `vf.State`; package versioned in pyproject | `vf-eval` reruns; judges swappable by config |
| ARE | not stated | App, Environment, Event (DAG), Notification, Scenario | app methods | timestamped event log | deterministic from start state + seed |

---

## Principles that survive across sources

1. The log is the state. Say it: Tau, pi, OpenHands, ARE, mini-swe-agent, Claude Agent SDK, Inspect, Harbor, tau2. Disagree: none. Nuance: Claude Agent SDK says the log is the conversation, not the filesystem, and prefers explicit application state for anything that must be reliable.
2. Grading is a separate pass over stored data. Say it: Inspect (`inspect score`), Harbor (regrade), tau2 (`evaluate_simulation`), verifiers (swappable judges). Disagree: none explicitly; agent SDKs (pi, Tau, OpenAI, Pydantic) do not have the concept because they are not evaluators.
3. Few, hand-written tools with good descriptions beat many generated ones. Say it: pi, mini-swe-agent, Anthropic, Westworld, Tau ADR 0002. Disagree: EnvFactory, Agent-World, ScaleEnv (R12) generate tools at scale and validate them by tests; our situation is closer to theirs because the tool set is dictated by the traces.
4. Determinism by construction: seed, fixed start state, cached model outputs. Say it: ARE, tau2, Westworld, Inspect, Zechner. Disagree: Ronacher notes context editing and caching interact badly; AgentSynth validates by difficulty curve rather than determinism.
5. Make the model call a parameter. Say it: Inspect (`generate`), Pydantic AI (TestModel), mini-swe-agent (Model class), smolagents. Disagree: Ronacher argues the abstraction must be your own because provider differences leak; pi and Tau agree and write their own provider layer, but still keep it as one replaceable object.
6. Every meaningful step emits an event, and UIs consume events. Say it: Tau, OpenHands, ARE, OpenAI Agents SDK tracing, Pydantic AI OTel. Disagree: mini-swe-agent has no event layer at all (the message list is the only record) and does fine at 100 lines. Resolution: for a re-execution harness whose output is a file, the JSONL record can be the event stream.
7. Validate generated artifacts by execution, with a bounded repair loop, and discard rejects. Say it: Envs-FORGE, EnvFactory, Agent-World, Harbor oracle, tau2 test suite. Disagree: none. (R12 section 3 reached the same conclusion for reward ground truth.)
8. Feedback to the model is part of the design. Say it: SWE-agent, Hashimoto, Ronacher, Anthropic "Writing tools". Constraint for us: feedback must match what the recorded production tool returned, not what would be nicer, or the cheaper model is graded on a different interface.
9. Understand the code under you, or write it. Say it: Anthropic, Octomind/HN, Ronacher, smolagents guide ("You'll be much better that way"). Disagree: OpenAI Agents SDK and Pydantic AI are frameworks and argue for a small primitive set instead; both keep the primitive count low (3 to 6), which is the reconciled position.
10. Version and package the environment as ordinary software. Say it: Prime hub (pyproject version, `load_environment`), OpenEnv (Docker, manifest), Harbor (task.toml version), lm-eval (task versions), OpenAI evals (`<name>.<split>.<version>`). Disagree: none.

---

## Design recommendations for our environment creator and re-execution harness

Constraints carried from the founder: simple, high quality, generalisable; from R12 section 4 and R22 section 5: replay fidelity on held-out real calls, reads excluded from the end-state hash, canonicalisation before hashing, no LLM-judged rewards in the verdict.

### Modules (one sentence each, one file each unless noted)

1. `records.py`: the Pydantic data records below, nothing else, so every other module speaks in these types and every record round-trips through JSON (Pydantic AI `TypeAdapter` pattern, tau2 `SimulationRun`).
2. `ingest.py`: normalises a customer's trace export into `Trace` records (turns, tool calls, tool results, errors) and assigns stable content hashes; pure code.
3. `mine.py`: derives `ToolSig` (name, arg schema, result schema, read/write class, observed error shapes) and `EntitySchema` (tables, columns, id formats) from the traces; code with one LLM call per unresolved tool to propose a result schema when observed results are too few (marked `source: llm` in the record).
4. `compile_env.py`: writes the Environment bundle in tau2 file shape (`data_model.py`, `tools.py`, `db.json`, `policy.md`, `tasks.json`) from `ToolSig` and `EntitySchema`; tools are generated as one function each over SQLite; LLM writes the function body, code writes the signature, docstring and schema (so descriptions match the trace exactly, Anthropic and Tau ADR 0002).
5. `policy.py`: turns policy sentences into `Constraint` records, each compiled to a before-write predicate in Python (LOGIGEN trigger style, R12); LLM proposes, code executes, one positive and one negative unit test generated per constraint.
6. `user_sim.py`: a rule-first simulated user built per Run from the trace (disclosure rules, refusals, walk-away branch, R22 item 9), with the LLM only filling utterances; rules are data in the `Run` record.
7. `loop.py`: the agent loop, mini-swe-agent shape: `while True: msg = model.query(messages); calls = parse(msg); results = [route(c) for c in calls]; messages += ...; if done: break`, target 100 to 150 lines, emitting one JSONL line per event.
8. `route.py`: sends each tool call to code (SQLite tool), recording (exact-match replay of a recorded result), or LLM stand-in, in that priority, and records which route was taken on the event.
9. `verdict.py`: computes the end-state verdict from the stored Run: write-set diff with column classes (exempt, hard, semantic), communicated-outcome check, policy violations fired, side-effect count; pure code; never calls a model.
10. `regrade.py`: re-runs `verdict.py` over stored Runs against a new Environment version without re-executing (Harbor regrade, Inspect `inspect score`, tau2 `evaluate_simulation`).
11. `validate.py`: the per-stage gates listed below, each a function returning a `GateResult`, run by `pipeline.py`.
12. `pipeline.py`: the stage runner: content-addressed, idempotent, skips stages whose input hash and code hash match a stored output.
13. `cli.py`: `ingest`, `build`, `run`, `verdict`, `regrade`, `report`; every command reads and writes records on disk, no hidden state.

Deliberately absent: an event bus, a plugin system, MCP, subagents, a UI framework (the dashboard reads JSONL), a provider abstraction beyond one `Model.query(messages, tools) -> Message` class with a `RecordedModel` and a `TestModel` implementation.

### Data records

- `Trace`: `trace_id`, `source`, `turns: list[Turn]`, `tool_calls: list[ToolCallRecord]` (name, args, result, error, latency), `hash`.
- `ToolSig`: `name`, `args_schema`, `result_schema`, `kind: read|write|generic`, `error_shapes`, `evidence: list[trace_id]`, `source: observed|llm`.
- `EntitySchema`: `tables`, `columns` with class `exempt|hard|semantic`, `id_patterns`.
- `Constraint`: `id`, `text`, `predicate_src`, `tests: {pos, neg}`, `compiled: bool`, `residual_reason` for the ones that could not be compiled.
- `Environment`: `env_id` (hash of the five files plus schema and constraints), `version`, `files`, `parent_env_id`.
- `Run`: `run_id`, `env_id`, `trace_id`, `model`, `seed`, `user_rules`, `events: list[Event]`, `end_state_hash`, `termination_reason`, `cost`, `route_counts`.
- `Event`: `idx`, `ts`, `type: model_call|tool_call|tool_result|user_turn|error|stop`, `payload`, `route: code|recording|llm`, `cache_key`.
- `Verdict`: `run_id`, `env_id`, `verdict_version`, `pass: bool`, `components` (state, communicated, policy, side_effects), `diff`, `class: pass|fail|transferred_without_acting|env_error`.
- `GateResult`: `stage`, `pass`, `metrics`, `failures: list[str]`.

One JSONL file per Run (events), one JSON per Environment version, one JSON per Verdict. Nothing else is persisted.

### Stage pipeline with validation per stage

| Stage | LLM or code | Gate (code) | On failure |
|---|---|---|---|
| ingest | code | every tool call has a parseable result or error; hashes stable across two runs | reject trace with reason |
| mine tools and schema | code, LLM only for missing result schemas | every `ToolSig` has at least 3 observed calls or is flagged `llm`; args in traces validate against the schema | flag, do not synthesise |
| compile tools | LLM body, code signature | replay held-out recorded calls (30 to 50) through the generated tool: hard columns must match, errors must match by shape; report success and error fidelity separately (R22 item 8) | one bounded repair round with the failing diff (EnvFactory), then mark the tool `recording-only` |
| compile policy | LLM predicate, code tests | pos and neg unit tests pass; the oracle path of every source trace is reachable (no gold-violates-policy, R22 item 1a) | drop the constraint into `residual` list, never into the verdict |
| build environment | code | tau2 file shape loads; `db.json` contains every ID the traces reference (R22 item 1b) | fail build |
| build user rules | code from trace, LLM utterances | every fact the agent asked for in the trace has a disclosure rule; a refusal branch exists where the trace shows one | flag Run as `user_rules_incomplete`, still runnable |
| oracle re-execution | code | replaying the original trace's calls reaches the recorded end state hash (strict replay) | environment rejected for that Run |
| loophole probe | LLM once | an oracle told to skip the policy step scores 0 | drop Run |
| cheap-model runs | LLM | k runs per Run with fixed seeds; each produces a complete JSONL | retry on infrastructure error only |
| verdict | code | golden-file test: verdict of the oracle Run is pass, verdict of an empty Run is fail | block regrade version |
| regrade | code | verdict version and env version recorded on each `Verdict` | none |

### What is LLM vs code

LLM: tool function bodies, policy predicate proposals, missing result schemas, user utterances, the loophole probe agent, the cheap models under test. Code: everything that decides (all gates, routing, hashing, diffing, verdicts, canonicalisation), everything that persists, and every schema and description shown to the model.

### What is cached

- Model calls: content-addressed on (model, messages, tools, config, seed, epoch), Inspect's key exactly; enables regrade and re-verdict without spend.
- Recorded tool results: keyed by (tool, canonical args, pre-state hash) so the `recording` route is an exact lookup.
- Stage outputs: keyed by (input record hash, code hash of the stage module); the pipeline skips unchanged stages.
- Nothing is cached by wall-clock time or run order.

### How it is tested

- Golden files: three anonymised traces plus their expected `ToolSig`, `Environment` files and `Verdict` records checked into tests; any change to outputs shows up as a diff (Pydantic AI inline-snapshot, tau2 `test_results_format.py`).
- Offline models: `TestModel` returns scripted tool calls; `RecordedModel` replays a stored Run's model outputs; `ALLOW_MODEL_REQUESTS=False` in CI.
- Verdict tests: oracle Run passes, empty Run fails, wrong-but-plausible Run fails, two different valid orders both pass (R12 step 9).
- Replay fidelity tests per generated tool on held-out calls, run in CI against the checked-in traces.
- One end-to-end test: ingest, build, oracle run, verdict, regrade on a 20-call trace; under 30 seconds with cached model calls (verifiers: "Prefer e2e tests over unit tests").

### Target size

- `loop.py` 100 to 150, `route.py` under 100, `verdict.py` 200 to 300, `records.py` 150 to 200, `pipeline.py` and `validate.py` together 250 to 350, `compile_env.py` plus `policy.py` 400 to 500, `mine.py` and `ingest.py` 300 to 400, `user_sim.py` 150 to 200, `cli.py` under 150. Total about 1,800 to 2,400 lines, comparable to Tau's core plus tau2's evaluator, with the generated Environment bundles outside that count.
- If the count drifts past 3,000, the first things to cut are the LLM stand-in route and the residual policy list, not the gates.

---

## Where sources disagree

1. Framework or no framework. Anthropic, Octomind, Ronacher, pi, mini-swe-agent: write the loop yourself. OpenAI Agents SDK, Pydantic AI, smolagents, Inspect: use a small primitive set. Our call: write the loop (it is 100 lines) but adopt Inspect's and Pydantic AI's testing idea (model as parameter) and tau2's data records verbatim.
2. Tools: hand-written few vs generated many. pi and mini-swe-agent say four or one; EnvFactory and Agent-World generate dozens and test them. Our tools are dictated by traces, so we generate but hold each to a replay-fidelity gate, and we keep descriptions verbatim from the trace rather than "improving" them.
3. Events vs plain message list. Tau, OpenHands and ARE have typed event vocabularies; mini-swe-agent has none. We take the middle: one JSONL event type per line with six `type` values, no bus.
4. Replay via recorded results vs LLM stand-in. tau2 replays recorded tool calls with `strict_replay`; Simia (R12) simulates tools with an LLM and pays with judged rewards; MCP-Persona (R22) shows LLM stand-ins fall to 53% F1 without traces. We prioritise code, then recording, then LLM stand-in, and record the route so verdicts can exclude Runs that leaned on the stand-in.
5. Determinism claims vs Ronacher's caution. ARE and Westworld promise full determinism; Ronacher reports that caching and context editing interact in ways that make cost and behaviour "hit and miss". We only claim determinism for code stages and cached model calls, and record seeds and cache keys so any non-determinism is attributable.
6. Sessions as trees vs flat lists. pi, Tau and Claude fork by copying or branching history; mini-swe-agent and tau2 keep a flat list per run. For re-execution, a Run is one flat JSONL; branching is a new Run with `parent_run_id`, which is enough and avoids the tree code.
7. Harness as a learned variable. arXiv 2606.25447 argues harness design should be co-trained with the model; every builder above treats the harness as fixed engineering. For us the harness must stay fixed within a comparison (same tools, same descriptions, same feedback for the expensive and the cheap model), so the paper is a warning about interpreting cross-model gaps, not a design input.
8. Where policy lives. LOGIGEN and our design compile policy to code; tau2 leaves it as `policy.md` for the model and grades outcomes; ARE puts policy into a notification whitelist. We do both: the model sees `policy.md`, and the compiled predicates only gate writes and count violations; rules that cannot compile are reported, never judged by an LLM in the verdict.
