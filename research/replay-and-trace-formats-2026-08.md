# Turning recorded agent traces into a replayable, gradable eval set

Technical sweep, August 2026. Sources are primary docs, specs, repos and papers fetched during the sweep; where a page could only be partially read this is flagged in "Limitations".

Convention used below: a "step" is one LLM call inside an agent episode. A step is replayable if, from the trace alone, you can rebuild the exact request (system prompt, full message history, tool schemas, sampling params) and you have the recorded assistant output (text and/or tool_calls) to compare against.

---

## 1. Trace formats in the wild

### 1.1 OpenTelemetry GenAI semantic conventions

Summary: The vendor-neutral target. Spans `chat {model}`, `execute_tool {tool}`, `invoke_agent {agent}` plus `create_agent`, `embeddings`. Content (messages, tool schemas, tool args, tool results) is opt-in and lives either on span attributes (JSON string) or on a log event `gen_ai.client.inference.operation.details`.

Exact details:
- Attributes: `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.definitions`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.top_k`, `gen_ai.request.max_tokens`, `gen_ai.request.seed`, `gen_ai.request.choice.count`, `gen_ai.agent.id`, `gen_ai.conversation.id`. All content-bearing ones are Opt-In. All `gen_ai.*` attributes remain `Status: Development` (only borrowed core attributes like `error.type` are Stable).
- Messages are structured as an array of `{role, parts: [{type: text|tool_call|tool_call_response|reasoning, ...}]}`; on spans the value "SHOULD be serialized to a JSON string". `gen_ai.tool.definitions` is flagged "Can be large".
- Content capture switch: `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` with values `no_content` (default), `span_only`, `event_only`, `span_and_event`. There is also a completion hook, `OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`, for writing full content to external storage (S3, DB) and leaving only a reference on the span, which the spec recommends for production.
- Older `gen_ai.prompt` / `gen_ai.completion` attributes (used by OpenLLMetry) are deprecated and removed as of semconv v1.38.
- The conventions have moved out of the main semconv repo into `open-telemetry/semantic-conventions-genai`; the main-site pages now redirect.

Replay verdict: fully replayable ONLY if the instrumentation populates all four of system_instructions, input.messages, tool.definitions, and request.* params. In the default configuration (`no_content`) you get model name, token counts and finish reasons and nothing replayable. Many instrumentations still ignore `gen_ai.tool.definitions` even when content capture is on.

Limitations: Development status means attribute names can still change; span attribute size limits in collectors and backends silently truncate large `gen_ai.input.messages`; `execute_tool` spans give you the tool result but the tool schema is only on the `chat` span if the instrumentor bothered.

URLs: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/ , https://github.com/open-telemetry/semantic-conventions-genai , https://opentelemetry.io/blog/2026/genai-observability/ , https://hidekazu-konishi.com/entry/opentelemetry_genai_semantic_conventions_guide.html , https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions , https://github.com/traceloop/openllmetry/issues/3515

### 1.2 OpenInference (Arize Phoenix)

Summary: Flattened, index-based attributes. Best of the observability formats for replay because tool schemas and invocation params are first-class.

Exact details:
- `openinference.span.kind` in {LLM, EMBEDDING, CHAIN, RETRIEVER, RERANKER, TOOL, AGENT, GUARDRAIL, EVALUATOR, PROMPT}.
- `llm.input_messages.<i>.message.role`, `.message.content`, `.message.tool_calls.<j>.tool_call.id`, `.tool_call.function.name`, `.tool_call.function.arguments`, `.message.tool_call_id` (for tool result messages).
- `llm.output_messages.<i>...` with the same shape.
- `llm.tools.<i>.tool.json_schema` ("Complete JSON schema of the tool").
- `llm.invocation_parameters` (JSON string of temperature, max_tokens, etc.), `llm.model_name`, `llm.provider`, `llm.system`.
- TOOL spans: `tool.name`, `tool.parameters`, `tool.json_schema`, plus `input.value` / `output.value` with `input.mime_type` / `output.mime_type`.

Replay verdict: yes, per LLM span, provided the instrumentor set `llm.tools.*` and `llm.invocation_parameters`. System prompt is just the first input message with role system.

Limitations: flattened attributes mean very long histories create thousands of attributes per span; some backends cap attribute count. Provider-specific content blocks (Anthropic thinking, OpenAI reasoning items) are not modeled, so replaying against a thinking model loses the signature-bearing blocks.

URLs: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md , https://arize.com/docs/phoenix/tracing/concepts-tracing/otel-openinference/semantic-conventions

### 1.3 Langfuse trace / observation model

Summary: trace -> nested observations of type span, generation, event, tool (plus agent, chain, retriever, embedding, guardrail, evaluator). Generation fields: `input`, `output`, `model`, `model_parameters`, `usage_details`, `cost_details`, `completion_start_time`, `prompt` (link to prompt version), `metadata`, `level`, `status_message`, `version`.

Exact details:
- No dedicated tool-definitions field. Whether tool schemas are captured depends entirely on the integration: the OpenAI/Anthropic/LangChain integrations put the provider request (messages, and usually `tools`) into `input`; hand-instrumented code often stores only the user prompt.
- OTel ingestion: Langfuse accepts `gen_ai.*` and OpenInference spans and maps them onto observations, so replayability equals the upstream instrumentation's replayability.
- `mask` client parameter and `mask_otel_spans` option apply a redaction function before export (relevant for section 5).

Replay verdict: yes IF `input` holds the raw provider request including `tools` and `model_parameters` holds the sampling params. Verify per integration; do not assume.

Limitations: The public docs do not enumerate size limits per field; large inputs are accepted but the UI truncates. Manual instrumentation commonly loses tool schemas.

URLs: https://langfuse.com/docs/observability/data-model , https://langfuse.com/docs/observability/features/observation-types , https://langfuse.com/docs/observability/sdk/instrumentation , https://langfuse.com/docs/observability/features/masking

### 1.4 LangSmith run tree

Summary: trace = root run + child runs; each run has `inputs`, `outputs`, `run_type` (llm, tool, chain, retriever...), `extra` (includes `invocation_params` when LangChain does the tracing), `metadata`, `dotted_order`, `trace_id`, `parent_run_id`, `child_run_ids`.

Exact details:
- LLM run inputs must be `{"messages": [...]}` where each message has `role` in {system, reasoning, user, assistant, tool} and `content` is an array of typed parts: text, image, file, audio, video, `tool_call` (with `name`, `args`, `id`), `server_tool_call`, `server_tool_result`; tool results are `{"type": "tool_result", "tool_call_id": ...}`.
- Model metadata via `ls_provider`, `ls_model_name`, `ls_temperature` etc. in `metadata`. Tool definitions: when traced through LangChain they land in `extra.invocation_params.tools`; when logged manually with `@traceable` they must be put in inputs or metadata by you.
- Export: `langsmith` CLI and SDK export to JSONL, one file per trace, hierarchy preserved; `read_run(load_child_runs=True)` hydrates the tree.
- Masking: `LANGSMITH_HIDE_INPUTS`, `LANGSMITH_HIDE_OUTPUTS`, `hide_inputs`/`hide_outputs` client functions, `create_anonymizer` (regex or callable). Anonymizer takes precedence over hide functions; `@traceable(process_inputs=...)` takes precedence over the client.

Replay verdict: usually yes for LangChain/LangGraph-traced apps (messages, tools, params all in the run); for manually-traced apps only if the author logged tools.

Limitations: `select` is recommended for large payloads, which implies size pressure; no explicit truncation figures in docs.

URLs: https://docs.langchain.com/langsmith/export-traces , https://docs.langchain.com/langsmith/log-llm-trace , https://docs.langchain.com/langsmith/mask-inputs-outputs , https://docs.smith.langchain.com/reference/python/run_trees/langsmith.run_trees.RunTree

### 1.5 OpenAI Chat Completions and Responses API logs

Summary: The request body itself (messages, tools, tool_choice, temperature, top_p, seed, max_tokens, response_format) is the perfect replay record; the problem is that OpenAI's server-side storage does not give it all back.

Exact details:
- Chat Completions with `store: true`: `GET /chat/completions/{id}` returns the ChatCompletion object (choices, usage, metadata, system_fingerprint). It does NOT return request messages, tools, or sampling params; a separate messages-list endpoint exists but tool definitions are not part of it.
- Responses API: `GET /responses/{id}/input_items` returns messages, `function_call`, `function_call_output`, `reasoning` (summary plus optional `encrypted_content`) and hosted-tool items; docs do not say `instructions` or `tools` are returned.
- Reasoning: raw reasoning tokens are never visible; `reasoning.summary` in {auto, concise, detailed} gives a summary. With `store=false` reasoning items carry `encrypted_content` (request it via `include: ["reasoning.encrypted_content"]`) which must be passed back verbatim between function calls; dropping them degrades quality.
- Evals API: data sources are JSONL files, stored completions or Responses; graders are `string_check`, `text_similarity`, `label_model`, `score_model`, `python`.

Replay verdict: log the outbound request body yourself (client-side middleware). Relying on `store=true` loses tool schemas and params. Encrypted reasoning items are model-bound: a candidate model cannot consume the recorded model's encrypted reasoning, so a replayed prefix for a different model must strip them (and accept the context loss).

Limitations: retention period and completeness of stored data not specified in the reference pages fetched.

URLs: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/retrieve , https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list , https://developers.openai.com/api/docs/guides/reasoning , https://developers.openai.com/cookbook/examples/responses_api/reasoning_items , https://developers.openai.com/api/docs/guides/evals

### 1.6 Anthropic Messages API

Summary: Assistant content blocks `text`, `tool_use {id, name, input}`, `thinking {thinking, signature}`, `redacted_thinking`; user content blocks `tool_result {tool_use_id, content, is_error}`. Any request that contains `tool_use` or `tool_result` blocks MUST include the matching `tools` array, so a stored request body is always complete for replay.

Exact details for replay with thinking:
- No `display` setting returns the raw chain of thought. `display: "summarized"` returns a summary (default on Opus 4.6 / Sonnet 4.6 and earlier); `display: "omitted"` returns thinking blocks with an empty `thinking` field (default on Opus 4.7, 4.8, Opus 5, Sonnet 5, Fable 5, Mythos 5). The `signature` field is an encrypted copy of the full reasoning either way, and billing is for full thinking tokens.
- Summarization "is processed by a different model from the one you target"; the thinking model never sees the summary.
- When returning tool results you must pass every `thinking` / `redacted_thinking` block back "complete and unmodified" with the `tool_use` block; within the latest assistant message the sequence cannot be rearranged, edited or partially dropped, otherwise 400. Outside tool use you may omit prior turns' thinking.
- Preservation by model: Opus 4.5 and 4.6+ keep prior-turn thinking blocks in context; Sonnet 4.5, Haiku 4.5 and earlier strip them automatically.
- Switching models mid-conversation: strip `thinking` and `redacted_thinking` from prior assistant turns; "Thinking blocks are tied to the model that produced them."
- Extended thinking `type: "enabled"` with `budget_tokens` is deprecated on 4.6 and rejected on 4.7+; adaptive thinking with `output_config.effort` replaces it. Changing thinking config between requests invalidates prompt cache.

Replay verdict: request body captured client-side is fully replayable for the same model. For a candidate model you must strip signature-bearing thinking blocks from the prefix, which means mid-tool-loop steps replay with less context than the original model had. `OTEL_LOG_RAW_API_BODIES` in Claude Code explicitly redacts extended-thinking content from logged bodies.

URLs: https://platform.claude.com/docs/en/build-with-claude/thinking , https://platform.claude.com/docs/en/build-with-claude/extended-thinking , https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview

### 1.7 Vercel AI SDK telemetry

Summary: `experimental_telemetry` emits OTel spans with custom `ai.*` attributes plus some `gen_ai.*`.

Exact details:
- Root spans `ai.generateText` / `ai.streamText`: `ai.prompt`, `ai.response.text`, `ai.response.toolCalls`, `ai.response.finishReason`, `ai.settings.maxOutputTokens`, `ai.model.id`, `ai.model.provider`, usage.
- Step spans `ai.generateText.doGenerate` / `ai.streamText.doStream`: `ai.prompt.messages` (full message array), `ai.prompt.tools` (array of stringified tool definitions incl. `inputSchema`), `ai.prompt.toolChoice`, `ai.response.*`, `ai.response.id`, `gen_ai.*`.
- Tool spans `ai.toolCall`: `ai.toolCall.name`, `ai.toolCall.id`, `ai.toolCall.args`, `ai.toolCall.result` (success only).
- `recordInputs` / `recordOutputs` default true.

Replay verdict: yes, at the doGenerate span level (messages + tools + settings). One of the few SDKs that records tool schemas by default.

Limitations: `ai.*` attributes are not the OTel GenAI convention (community thread on aligning); step spans only exist for multi-step calls so single-shot requests put everything on the root span; provider-specific reasoning blocks are flattened.

URLs: https://ai-sdk.dev/docs/ai-sdk-core/telemetry , https://community.vercel.com/t/opentelemetry-gen-ai-semantic-conventions-support/29859

### 1.8 Agent frameworks

OpenAI Agents SDK tracing: span data classes `AgentSpanData(name, handoffs, tools, output_type)`, `FunctionSpanData(name, input, output, mcp_data)`, `GenerationSpanData(input, output, model, model_config, usage)`, `ResponseSpanData(response, input, usage, _response_id)`, `HandoffSpanData`, `GuardrailSpanData`, `MCPListToolsSpanData(server, result)`. Key catch: `ResponseSpanData` exports only `response_id` and `usage`; the default OpenAI exporter relies on the server-stored response for content. `RunConfig.trace_include_sensitive_data=False` drops inputs/outputs. Replay: only if you use a third-party processor (Langfuse, Braintrust, OpenInference instrumentor) that serializes `input` and `response` itself, or record via the Responses API log. URL: https://openai.github.io/openai-agents-python/ref/tracing/span_data/ , https://openai.github.io/openai-agents-python/tracing/

LangGraph: traces through LangSmith (section 1.4) so tool definitions live in `invocation_params`. Separately, the checkpointer gives you time travel: `get_state_history`, rerun from a checkpoint config, `update_state` to fork ("pretend a node just emitted these values"). Replay re-executes nodes after the checkpoint, including LLM calls, so it is a live branch not a cached replay. URL: https://docs.langchain.com/oss/python/langgraph/use-time-travel

Pydantic AI: OTel-native. `InstrumentationSettings(include_content=True, event_mode="attributes"|"logs", include_model_request_parameters=..., version=5)`. Version 5 writes `gen_ai.input.messages` / `gen_ai.output.messages` on model request spans; `gen_ai.tool.definitions` (name, description, parameters) "is always emitted regardless of settings"; `include_model_request_parameters` serializes the full `ModelRequestParameters` including tool metadata and `return_schema`. Best-in-class for replay. URL: https://ai.pydantic.dev/api/models/instrumented/

Claude Agent SDK / Claude Code: the CLI emits OTel metrics, log events (`claude_code.user_prompt`, `claude_code.api_request`, `claude_code.tool_result`, ...) and beta trace spans `claude_code.interaction`, `claude_code.llm_request`, `claude_code.tool` (children `claude_code.tool.blocked_on_user`, `claude_code.tool.execution`), `claude_code.hook`. Telemetry is structural by default. Opt-ins: `OTEL_LOG_USER_PROMPTS=1`, `OTEL_LOG_TOOL_DETAILS=1`, `OTEL_LOG_TOOL_CONTENT=1` (60 KB truncation, `CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH`), and `OTEL_LOG_RAW_API_BODIES=1` or `file:<dir>` which logs the full Messages API request and response JSON as `claude_code.api_request_body` / `claude_code.api_response_body` (entire conversation history, thinking content redacted). The raw-bodies mode is the only path that is replay-complete. Community instrumentors (justinbarias/opentelemetry-instrumentation-claude-agent-sdk) use PreToolUse/PostToolUse hooks to emit GenAI-convention `execute_tool` spans. URL: https://code.claude.com/docs/en/agent-sdk/observability , https://github.com/justinbarias/opentelemetry-instrumentation-claude-agent-sdk

CrewAI: `opentelemetry-instrumentation-crewai` (OpenLLMetry family) produces kickoff -> task -> agent -> LLM call and tool call spans; prompts and completions go into span attributes by default using OpenLLMetry's `gen_ai.prompt.*` / `gen_ai.completion.*` and `llm.request.functions` (its own name for tool definitions). Replayable but in a deprecated attribute dialect. URL: https://pypi.org/project/opentelemetry-instrumentation-crewai/ , https://www.traceloop.com/docs/openllmetry/contributing/semantic-conventions

Mastra: built-in AI tracing with span types AGENT_RUN, MODEL_GENERATION, TOOL_CALL, WORKFLOW_STEP; exporters to Langfuse, LangSmith, Braintrust, OTel backends (Sentry maps MODEL_GENERATION -> `gen_ai.chat`, TOOL_CALL -> `gen_ai.execute_tool`). Span processors (`SpanOutputProcessor`) can redact or drop. Model parameters and tool execution details are captured; whether tool schemas are on the MODEL_GENERATION span depends on exporter. URL: https://mastra.ai/docs/observability/tracing/overview

### 1.9 Which formats lose what (table)

| Source | System prompt | Full prefix | Tool schemas | Sampling params | Recorded output incl. tool_calls | Reasoning |
|---|---|---|---|---|---|---|
| OTel GenAI, content on | yes (`gen_ai.system_instructions`) | yes | opt-in `gen_ai.tool.definitions`, often skipped | `gen_ai.request.*` opt-in | yes | `reasoning` part, provider-dependent |
| OTel GenAI, default | no | no | no | no | no | no |
| OpenInference | yes | yes | `llm.tools.*.tool.json_schema` | `llm.invocation_parameters` | yes | no |
| Langfuse generation | if in `input` | if in `input` | only if integration puts `tools` in `input` | `model_parameters` | yes | no |
| LangSmith LLM run | yes | yes | `extra.invocation_params.tools` (LangChain) else manual | `ls_*` metadata | yes | `reasoning` role supported |
| OpenAI stored completion | no | no (separate messages endpoint) | no | no | yes | summary only |
| OpenAI Responses input_items | no (`instructions` absent) | yes | no | no | yes | summary + encrypted |
| Anthropic request body (client-logged) | yes | yes | yes (required by API) | yes | yes | summary or omitted + signature |
| Vercel AI SDK doGenerate span | yes | `ai.prompt.messages` | `ai.prompt.tools` | `ai.settings.*` | yes | no |
| OpenAI Agents SDK default exporter | no | response_id only | AgentSpanData.tools (names only) | model_config | via server | no |
| Pydantic AI v5 | yes | yes | always | yes | yes | provider parts |
| Claude Code default OTel | no | no | no | model name only | no | no |
| Claude Code `OTEL_LOG_RAW_API_BODIES` | yes | yes | yes | yes | yes | redacted |

Takeaway: the only universally complete record is the outbound request body plus the raw response, captured at the HTTP client. Everything else is a lossy projection.

---

## 2. Replay strategies

### 2a. Step-level teacher-forced replay

Summary: Feed the recorded prefix (system, history, tools, params) to a candidate model, take one step, compare to the recorded step. This is the "offline sampling" setting in ToolPRMBench: "constrains the model to follow the golden trajectory prefix and only samples an alternative action at a specific step... isolates single-step mistakes." Each ToolPRMBench case holds "the interaction history, a correct action, a plausible but incorrect alternative, and relevant tool metadata", labels cleaned by a multi-LLM verification pipeline.

Exact mechanics that matter:
- Strip model-bound artifacts from the prefix before sending to a different model: Anthropic `thinking`/`redacted_thinking` blocks (tied to producing model; other models ignore but bill them), OpenAI `reasoning` items with `encrypted_content`. Keep `tool_use`/`tool_result` pairs intact and keep the `tools` array (Anthropic rejects otherwise).
- Provider format conversion is lossy in both directions (OpenAI `tool_calls[].function.arguments` string vs Anthropic `tool_use.input` object; system as message vs top-level `system`). Store a canonical form and provider-specific renderers.
- Set candidate `temperature` to the recorded value, or 0 for determinism; record `seed` where supported.

Limitations: teacher forcing measures agreement with the reference policy, not task success; a candidate that would have taken a different but valid route is penalized at every subsequent step because the prefix reflects the reference's route (exposure bias). ToolPRMBench notes offline sampling captures "localized deviations" while online sampling captures "realistic multi-step failures". Treat step accuracy as a screening metric, not the outcome metric.

URLs: https://arxiv.org/abs/2601.12294 , https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation

### 2b. Full-episode replay with mocked tools

Summary: Run the candidate from the task start; serve recorded observations when its tool call matches the recorded one; do something when it does not.

Evidence that static replay fails when the policy changes: "The Replay Gap" (arXiv 2608.08239) forked live SWE-bench trajectories at swap points and compared to static stitching. Model swaps produced +0.25 to +0.66 normalized edit distance over controls, 61 to 94 percent of post-swap actions were rewritten, 74 to 77 percent of early swaps diverged at the very first subsequent action (vs 6 to 35 percent of controls), "only 3 percent of replayed states remained valid", and replay "mispredicts every success-relevant outcome call" (patch similarity 0.00 to 0.11). Recommendation: branching rollouts (live continuation after the swap) instead of static replay.

Divergence handling options, cheapest first:
1. Exact match on (tool name, canonical JSON args) -> serve recorded result.
2. Normalized match (section 3) -> serve recorded result, flag as fuzzy.
3. Recorded-corpus lookup: if the same tool+args was observed anywhere in the trace corpus (other episodes), serve that. Works well for read-only tools with stable outputs.
4. Deterministic mock environment: reimplement tools over a seeded state (tau-bench: Python tools over JSON databases; AgentDojo: stateful Python functions over pydantic environment state; ToolSandbox: Python execution context as world state, milestones/minefields; BFCL multi-turn: backend class instances whose state is compared). Highest fidelity, highest build cost, only viable for a bounded tool set.
5. LLM-simulated tool responses: AWS ToolSimulator / Strands `ToolSimulator` (decorator captures name, docstring, type hints; `output_schema` Pydantic model validated; `initial_state_description`; `share_state_id` links tools on the same backend so writes affect later reads; two stages: parameter validation then LLM response generation). Research: Simia showed reasoning models generate plausible tool feedback; EnvScaler automated environment synthesis; ToolPO uses LLM-simulated APIs for RL. Fidelity warning from EnvSimBench (400 samples, 167 environments): near-perfect when state is unchanged, "catastrophic failures" on simultaneous multi-state updates (the "state change cliff"); failure modes are hallucination, logical inconsistency and silent state drift. Use LLM simulation for read-only or single-write tools, seed it with the recorded observations as few-shot examples, and validate output against the tool's response schema.
6. Simulated users: tau2-bench user simulator is an LLM with its own tools and a scenario; measured error rates from manual annotation were 16 percent (telecom, 6 percent critical), 40 percent (retail, 12 percent critical), 47 percent (airline, 13 percent critical). Tightly constraining the simulator with tools and observable state lowered its error rate. APIGen-MT: phase 1 committee-reviewed blueprints with ground-truth actions, phase 2 simulated human-agent interplay to produce trajectories, filtered by state and output checks.

Related replay work: Causal Agent Replay (arXiv 2606.08275) models the run as an SCM, resamples one step under the same policy and re-executes forward to measure outcome shift; reports LLM-judge step attribution at only about 14 percent step-level accuracy on Who&When, motivating intervention-based rather than judge-based attribution. AgentRR (arXiv 2505.17716) records interaction traces and distills them into multi-level "experiences" with a check function as a trust anchor during replay; it is about reuse, not eval, but the check-function idea (validate applicability before serving a recorded artifact) transfers. Deterministic-testing write-ups (langchain-replay pattern) record LLM decisions and replay them while letting tools run for real, which is the inverse of what an eval needs but useful for regression tests of the harness itself.

Limitations: any mocked-tool episode is only as faithful as the mock; once the candidate diverges, the recorded trajectory stops being a reference for the rest of the episode and you need an outcome-level grader (section 3).

URLs: https://arxiv.org/abs/2608.08239 , https://arxiv.org/abs/2606.08275 , https://arxiv.org/abs/2505.17716 , https://arxiv.org/abs/2605.07247 , https://aws.amazon.com/blogs/machine-learning/toolsimulator-scalable-tool-testing-for-ai-agents/ , https://strandsagents.com/docs/user-guide/evals-sdk/simulators/tool_simulation/ , https://arxiv.org/abs/2506.07982 , https://github.com/sierra-research/tau2-bench , https://arxiv.org/abs/2406.13352 , https://arxiv.org/abs/2408.04682 , https://arxiv.org/abs/2504.03601 , https://blog.sixty-north.com/deterministic-testing-for-langchain-agents.html

### 2c. Live replay against real tools

Summary: Only for idempotent read tools (search, fetch, DB read, file read on a snapshot). Sources of drift: web content, search rankings, time-dependent data. BFCL's "executable" category calls real APIs and tolerates it by choosing stable endpoints and checking structural properties of the response rather than exact values. Mitigation: snapshot the environment (container image, DB dump, git commit) as the Replay Gap paper does ("rebuilt environments"), and record `observed_at` on every recorded observation so you can bound staleness.

Limitations: side effects, rate limits, cost, and PII exposure to third parties. Never replay write tools live.

URLs: https://gorilla.cs.berkeley.edu/leaderboard.html , https://arxiv.org/abs/2608.08239

---

## 3. Comparing candidate vs recorded

### 3.1 Tool calls

Exact JSON match: canonicalize (sorted keys, no whitespace, unicode NFC, numbers normalized) then string-compare. Deterministic, zero cost, brittle to any harmless variation.

AST / typed match (BFCL `ast_checker.py`): `possible_answer` maps each parameter to a list of acceptable values; optional params are marked by including `""` in that list; `type_checker()` coerces via `PYTHON_TYPE_MAPPING` (int -> float allowed, tuple -> list), one level of nested container checking; `standardize_string()` removes spaces and the punctuation set `",./-_*^"`, lowercases, and swaps single quotes for double so "April 1, 2024" equals "April 1,2024"; `dict_checker()` requires key-set equality then standardized value match; `list_checker()` standardizes string elements; `parallel_function_checker_no_order()` matches possible answers to model outputs greedily with `matched_indices` for order independence. Error taxonomy: `simple_function_checker:wrong_func_name`, `type_error:nested`, `value_error:string`, `value_error:dict_key`, `parallel_function_checker_no_order:cannot_find_match`, and so on. The BFCL v4 LLM-judge mode reportedly has about 20 percent evaluator-human misalignment, so the deterministic checker is still the reference.

Library matchers:
- openevals `create_trajectory_match_evaluator(trajectory_match_mode=strict|unordered|subset|superset, tool_args_match_mode=exact|ignore|subset|superset, tool_args_match_overrides={tool_name: comparator_fn})`, plus `create_trajectory_llm_as_judge` with `TRAJECTORY_ACCURACY_PROMPT` and `TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE`.
- Ragas `ToolCallAccuracy(strict_order=True)` scores argument accuracy times (sequence aligned ? 1 : 0), exact-match args by default via `arg_comparison_metric`; `ToolCallF1` unordered precision/recall over (name, args) pairs; `AgentGoalAccuracy` with and without reference (binary, LLM).
- DeepEval `ToolCorrectnessMetric(evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT], should_consider_ordering, should_exact_match, threshold=0.5, available_tools=...)`: by default compares tool names only; score = correctly used tools / total tools called; optional LLM check of optimality if `available_tools` given.
- Braintrust autoevals `JSONDiff` (recursive structural diff, Levenshtein for strings, `NumericDiff` for numbers) for partial-credit JSON comparison.
- promptfoo trajectory assertions read `tool.name` / `tool.arguments` or `ai.toolCall.name` / `ai.toolCall.args` from OTel spans.

Normalized / semantic equivalence: dates (parse to ISO), case and whitespace, list order for set-valued args, numeric tolerance, alias tables (NYC / New York, Vegas / Las Vegas, CDG Airport vs Paris in the Arize example), URL canonicalization, search-query equivalence (embedding cosine above threshold or judged). Put per-tool comparators in the eval config, as openevals' `tool_args_match_overrides` does.

LLM-judged equivalence: Arize Phoenix ships a Tool Selection evaluator (right tool, or correctly no tool, given query and available tools) and a Tool Invocation evaluator (all required parameters present, values match what the user said). Their first pass matched ground truth only 36 percent before template refinement, which shows judge templates need iteration against a labeled slice. Guidance from tool-agent eval write-ups: run schema validation as a deterministic gate before the judge, score per layer (selection, arguments, groundedness of result use) rather than one aggregate.

### 3.2 Final answers

Reference-based with the frontier output as reference: two protocols.
- Pairwise "is B at least as good as A" (preference, with the recorded frontier answer as A). Pairwise is better calibrated and agrees better with humans, but has strong position bias; run both orders and aggregate (or require agreement). Self-preference bias exists when the judge is the same family as one candidate.
- Absolute rubric (direct assessment or per-criterion binary rubric). Rubric-based evaluation "implicitly resembles a multiple-choice setting" and shows position bias toward specific score options; direction is model-specific; the ordering of criteria also shifts scores; "a small number of random order permutations are sufficient to reduce the error" for most models (arXiv 2602.02219). Self-preference bias in rubric grading documented in arXiv 2604.06996.
- Practical rule: use pairwise-with-reference for "no regression vs frontier" gates, absolute rubric when you need a stable per-item score across many candidates, and always include a deterministic component (required facts, forbidden claims, format) that does not depend on the judge.

### 3.3 Outcome / state comparison (when the candidate diverges)

tau2-bench reward: apply initialization then the solution functions to get expected state, then check assertions: status assertions on final world state, DB state comparison, presence of required actions in the trajectory (with argument matching), and natural-language assertions on what was communicated; `pass^k` = fraction of tasks solved in all k of k independent runs (they report pass^1 and pass^4). AgentDojo: deterministic utility and security checks on environment state and outputs. BFCL multi-turn: compares backend instance state after each turn. For production traces you rarely have such checkers; the substitute is "state diff of the recorded episode" (set of writes) compared with the candidate's writes.

URLs: https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py , https://github.com/langchain-ai/openevals , https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/ , https://deepeval.com/docs/metrics-tool-correctness , https://github.com/braintrustdata/autoevals , https://www.promptfoo.dev/docs/tracing/ , https://arize.com/blog/how-to-evaluate-tool-calling-agents/ , https://arxiv.org/abs/2602.02219 , https://arxiv.org/pdf/2604.06996 , https://arxiv.org/html/2506.07982 , https://openreview.net/pdf?id=2GmDdhBdDk

---

## 4. Grading reasoning when it is hidden

What is observable:
- OpenAI o-series / GPT-5: raw reasoning tokens are never returned; you get a summary (`reasoning.summary`: auto | concise | detailed) and, with `store=false`, an opaque `encrypted_content` blob. Reasoning token counts are in usage.
- Anthropic: `thinking` blocks are a summary produced by a different model (`display: "summarized"`), or empty (`display: "omitted"`, the default on the newest models); `signature` is encrypted full reasoning; `usage.output_tokens_details.thinking_tokens` gives the count. Claude Code's raw-body logging redacts thinking.
- Open-weight reasoning models: full trace visible.

Should visible reasoning be graded?
- Anthropic, "Reasoning models don't always say what they think" (arXiv 2505.05410): CoTs reveal hint usage in "at least 1 percent of examples where they use the hint, but the reveal rate is often below 20 percent"; overall faithfulness about 25 percent for Claude 3.7 Sonnet and 39 percent for DeepSeek R1, lower (20 and 29 percent) on misalignment hints; outcome RL improves faithfulness then plateaus; reward hacking is almost never verbalized.
- "Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence Settings" (arXiv 2608.04735): monitors caught 60 to 94 percent of behavior changes under explicit influence but detection dropped 41 to 46 points under implicit influence, and a benign debiasing system prompt cut detection to 5 percent while the behavior shift remained.
- "Measuring Faithfulness Depends on How You Measure" (arXiv 2603.20172): faithfulness labels are sensitive to the classifier used, so a judge scoring "reasoning quality" has a measurement problem on top of the faithfulness problem.
- OpenAI / multi-lab position paper "Chain of Thought Monitorability: A New and Fragile Opportunity" (arXiv 2507.11473): monitorability is useful but fragile and degrades under optimization pressure.
- Summaries add a second layer: the summary is produced by another model, so grading it grades the summarizer.

Recommendation: do not score reasoning text as a primary metric. Grade actions (tool selection, arguments, ordering, stop decisions), observable intermediate artifacts (files written, messages sent, state diffs) and outcomes. Use reasoning text, when available, only for (a) diagnostic clustering of failures, (b) cheap red flags (mentions of guessing, ignoring the user constraint) reported separately and never aggregated into the pass rate, and (c) length / thinking-token budget as a cost metric. If you need a process signal, use a process reward model trained on action correctness (ToolPRMBench) rather than a judge reading prose.

URLs: https://arxiv.org/abs/2505.05410 , https://arxiv.org/html/2608.04735 , https://arxiv.org/pdf/2603.20172 , https://arxiv.org/pdf/2507.11473 , https://developers.openai.com/api/docs/guides/reasoning , https://platform.claude.com/docs/en/build-with-claude/thinking

---

## 5. Dataset construction hygiene

Redaction before storage:
- Do it in the client before export: Langfuse `mask` (Python and JS) and `mask_otel_spans`; LangSmith `hide_inputs` / `hide_outputs` functions, `create_anonymizer(patterns)` (regex or callable; anonymizer takes precedence over hide functions; `@traceable(process_inputs=...)` takes precedence over client-level), `LANGSMITH_HIDE_INPUTS=true` disables logging entirely (and skips the anonymizer). OTel: `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=no_content` default, or completion hook upload to a controlled store with references on spans. Claude Code: content opt-ins listed in 1.8 default off.
- Langfuse's PII guide lists four control points: client SDK, ingestion server, model gateway, after-the-fact detection. For evals you need consistent placeholders (same email -> same `<EMAIL_1>` within an episode) or tool-call argument matching breaks; regex plus a library like llm-guard or Presidio, reviewed on a sample.
- Redaction must be applied identically to prefix, recorded output, and recorded tool results, or the replayed model sees inconsistent tokens.

Dedup:
- Exact dedup on canonical (system prompt hash, tools hash, message history hash). Near-dup: MinHash over 5-char shingles, banded LSH (one published configuration: 112 hashes, 14 bands x 8, Jaccard threshold 0.8), Union-Find clustering, keep one representative per cluster; or embedding cosine with a two-stage within-group then cross-group pass. For agent traces dedup at two levels: whole-task level (same user goal) and prefix level (steps 0..k of the same episode share a prefix by construction; do not count them as independent samples).

Sampling and stratification:
- Cluster tasks (embedding of first user message plus tool set) and sample per cluster with a cap so head clusters do not dominate. Stratify by trajectory depth (step index and episode length), by tool used at the step, by outcome (success, failure, escalation) and by time window. Record `sampled_from_cluster`, `step_index`, `episode_length` on each item.
- Error analysis first: Husain and Shankar's workflow (sample real traces, open coding of failure notes by a domain expert, axial coding into 5 to 10 failure modes after 30 to 50 hand-coded traces, then count) tells you which strata matter before you fix the sampling plan.
- Clustered standard errors: steps from the same episode are correlated; "Adding Error Bars to Evals" (arXiv 2411.00640) recommends clustered SEs (cluster adjustments can inflate SE up to about 3x), paired differences when comparing two models on the same items, and k resamples per question for variance reduction; it suggests around 1000 questions for new evals. "Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints" (arXiv 2503.01747) recommends Bayesian or bootstrap / Wilson / Clopper-Pearson intervals below a few hundred items; a 100-item set at 80 percent pass has a 95 percent interval of roughly 72 to 88.

Holdout and contamination:
- Split by episode (never by step) and by user or tenant when possible; keep a time-based holdout (latest N weeks) as the drift check. If traces are also used for fine-tuning or few-shot prompt examples, hash every eval item's prefix and forbid those hashes from the training and prompt pools. Retire items whose behavior no longer exists, and record `added_at` per item to track age distribution (Langfuse golden-dataset guidance).
- Contamination via the provider: a stored completion or stored response may be used by the provider per its data policy; use `store=false` / ZDR for eval runs if that matters.

Size guidance: 30 to 50 traces for error analysis; a few hundred step items per stratum you want to make claims about; around 1000 items before CLT-style error bars are trustworthy; report clustered CIs by episode regardless.

URLs: https://langfuse.com/docs/observability/features/masking , https://langfuse.com/resources/engineering/pii-masking-llm-applications , https://docs.langchain.com/langsmith/mask-inputs-outputs , https://langfuse.com/resources/engineering/golden-dataset-evaluation , https://hamel.dev/blog/posts/evals-faq/ , https://arxiv.org/abs/2411.00640 , https://arxiv.org/abs/2503.01747 , https://zeroentropy.dev/concepts/deduplication/

---

## 6. Open-source building blocks

- Inspect AI (UK AISI): dataset -> Task -> Solver -> Scorer; agent loops with tools; sandboxes; transcript with ToolEvents. The `mockllm` provider takes `custom_outputs: Iterable[ModelOutput] | Generator | Callable[[list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig], ModelOutput]`, and `ModelOutput.for_tool_call(model, tool_name, tool_arguments, tool_call_id=...)`, so you can script a recorded policy against a harness or, inversely, script recorded tool results by implementing tools that look up the recording. Model-graded scorers neutralize structural delimiters. URL: https://inspect.aisi.org.uk/ , https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/model/_providers/mockllm.py
- OpenAI Evals: the hosted Evals API (JSONL, stored completions, Responses data sources; graders string_check, text_similarity, label_model, score_model, python) is the maintained path; the `openai/evals` GitHub registry still exists but is a legacy framework. URL: https://developers.openai.com/api/docs/guides/evals , https://github.com/openai/evals
- openevals (LangChain): trajectory match modes and LLM trajectory judge (3.1). URL: https://github.com/langchain-ai/openevals
- promptfoo: trajectory assertions over OTel spans, provider-agnostic runner. URL: https://www.promptfoo.dev/docs/tracing/
- DeepEval: `ToolCorrectnessMetric`, DAG metrics, conversational metrics. URL: https://deepeval.com/docs/metrics-tool-correctness
- Ragas: `ToolCallAccuracy`, `ToolCallF1`, `AgentGoalAccuracy`, `TopicAdherence`. URL: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/
- Braintrust autoevals: `JSONDiff`, `ExactMatch`, `Levenshtein`, `NumericDiff`, embedding similarity, Factuality and other LLM scorers. URL: https://github.com/braintrustdata/autoevals
- lm-evaluation-harness: strong for static academic tasks; no agentic tool-call loop of note as of this sweep; not the right base for trace replay. URL: https://github.com/EleutherAI/lm-evaluation-harness
- Langfuse SDK: `mask`, OTel ingestion, datasets and experiments; OpenLLMetry (Traceloop): instrumentors for OpenAI, Anthropic, LangChain, CrewAI, etc., emitting `gen_ai.prompt.*` / `llm.request.functions` (deprecated dialect, migrating). URLs: https://langfuse.com/docs , https://github.com/traceloop/openllmetry
- openai-agents tracing: custom `TracingProcessor` to serialize `GenerationSpanData` / `ResponseSpanData` fully instead of response_id. URL: https://openai.github.io/openai-agents-python/tracing/
- Pydantic AI instrumentation and pydantic-evals: the cleanest GenAI-convention emitter. URL: https://ai.pydantic.dev/api/models/instrumented/
- BFCL eval code: `bfcl_eval/eval_checker/ast_eval/ast_checker.py` is a reusable typed-argument checker with a possible-answers schema; multi-turn checker compares backend state. URL: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- tau2-bench: domains (airline, retail, telecom, banking_knowledge, mock), gym interface, `tau2 evaluate-trajs --fresh-tasks` to re-score stored trajectories, user simulator. URL: https://github.com/sierra-research/tau2-bench
- AgentDojo, ToolSandbox, APIGen-MT: reference implementations of stateful mock environments and simulated-user trajectory generation. URLs: https://arxiv.org/abs/2406.13352 , https://arxiv.org/abs/2408.04682 , https://arxiv.org/abs/2504.03601
- Judge libraries: openevals judges, DeepEval GEval, autoevals LLM scorers, Inspect `model_graded_qa`; for position bias run both orders (pairwise) and permute rubric options (rubric).
- Simulators: Strands `ToolSimulator` (AWS ToolSimulator) for LLM-simulated tools with shared state. URL: https://strandsagents.com/docs/user-guide/evals-sdk/simulators/tool_simulation/

---

## Synthesis

### Recommended minimal data model for a replayable step

Store one record per LLM call. Provider-neutral canonical form, with the raw request and response bodies retained verbatim (encrypted at rest, redacted copy for the eval set).

```
ReplayStep
  ids
    episode_id            # one agent run
    step_index            # 0-based LLM call index within the episode
    trace_id, span_id     # link back to the observability system
    parent_step_index     # for subagents / handoffs, else null
  provenance
    recorded_at           # ISO timestamp
    source_format         # otel_genai | openinference | langfuse | langsmith | raw_openai | raw_anthropic | vercel_ai | ...
    provider, model_id    # exact model string that produced the recorded output
    app_version, prompt_version, tools_version
    redaction_version     # which masking pass was applied
  request (canonical)
    system                # string or list of blocks
    messages[]            # {role, parts[{type: text|tool_call|tool_result|image|file|reasoning_ref, ...}]}
                          #   tool_call: {id, name, arguments(object)}
                          #   tool_result: {tool_call_id, content, is_error}
                          #   reasoning_ref: {provider, opaque:true}   # signature / encrypted_content are NOT copied into the eval set
    tools[]               # {name, description, input_schema(JSON Schema), strict?}
    tool_choice
    params                # {temperature, top_p, top_k, max_tokens, seed, stop, response_format, thinking/reasoning config}
    prefix_hash           # sha256 over canonical(system, messages, tools, params) for dedup and holdout
  reference_output
    assistant_parts[]     # text and tool_call parts exactly as produced
    finish_reason
    usage                 # input, output, reasoning/thinking token counts
    reasoning_summary     # optional, diagnostic only, never graded
  observations            # for full-episode replay
    tool_results[]        # {tool_call_id, name, arguments, result, is_error, latency_ms, observed_at, idempotent: bool}
  episode_context
    task_text             # first user message (redacted)
    episode_length        # total steps
    episode_outcome       # success | failure | escalated | unknown, and how it was labeled
    state_diff            # writes performed in the episode (tool name + args), for outcome comparison
  eval_metadata
    cluster_id, stratum   # task cluster and depth bucket
    split                 # train | dev | test | time_holdout
    added_at, retired_at
    grading_spec_ref      # which comparator ladder rung and which per-tool comparators apply
  raw
    request_body_ref, response_body_ref   # pointers to verbatim provider payloads in a restricted store
```

Rules: strip provider-bound reasoning artifacts (Anthropic `signature`/`redacted_thinking`, OpenAI `encrypted_content`) when replaying to a different model, but keep a `reasoning_ref` so same-model replay can re-attach them from `raw`. Keep `tools` on every step because the Anthropic API rejects `tool_use` history without them and because tool schemas change over time. Steps sharing an episode are one cluster for statistics.

### Recommended comparison ladder (cheapest and most deterministic first)

1. Schema gate: candidate output parses; every tool_call names a tool in `tools[]` and its arguments validate against `input_schema`. Deterministic, zero cost. Fail here is a hard fail.
2. Action-type match: did the candidate emit a tool call vs final answer vs clarification question, same as the reference. Deterministic.
3. Tool-name set match (unordered for parallel calls; ordered only when the reference has a dependency). Deterministic.
4. Canonical JSON argument match: sorted keys, normalized numbers and unicode, trimmed strings. Deterministic.
5. Typed / normalized argument match (BFCL-style): per-parameter acceptable-value lists, optional params, int-float coercion, string standardization (case, whitespace, punctuation), date parsing to ISO, set semantics for list args, alias tables per tool, numeric tolerance. Deterministic given a per-tool comparator config (openevals `tool_args_match_overrides` pattern). Emit partial credit per argument (JSONDiff-style) as a secondary number.
6. Corpus-equivalence for free-text arguments (search queries, messages to users): embedding similarity above a tuned threshold, or "same tool result when executed against the recorded corpus". Cheap, mostly deterministic, needs calibration on a labeled slice.
7. Outcome / state comparison when trajectories diverge: candidate's write-set vs reference `state_diff`; required actions present; final observable state assertions. Deterministic where checkers exist; otherwise skip to 8.
8. LLM-judged argument equivalence for the residual mismatches from rungs 5 and 6 only: tool-selection and tool-invocation judge templates seeded with the recorded call as reference, schema-validated input, adjudicate only the disagreement set. Report judge-human agreement on a labeled sample; expect to iterate templates (Arize started at 36 percent agreement).
9. Final-answer grading: deterministic checks first (required facts, forbidden claims, format, citations resolve), then pairwise "candidate at least as good as recorded frontier answer" with both orderings and a tie option; absolute rubric only when you need cross-candidate comparability, with permuted rubric option order. Never let the judge be the same model family as the candidate without a self-preference control.
10. Reasoning text: not in the ladder. Diagnostic only.

Reporting: pass rates per rung, per stratum (step depth, tool, task cluster), with clustered (by episode) bootstrap CIs; paired differences between candidate and baseline on identical items; and a divergence-depth histogram (at which step the candidate first left the recorded trajectory), which the Replay Gap results suggest is the most informative single number when swapping models.
