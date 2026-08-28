# Production agent traces in practice, and how vendors turn them into eval data

Source: research agent, 2026-08-26. WebSearch unavailable; direct fetches of vendor docs, source code, arXiv and blogs.

Method note: the session's WebSearch budget was already exhausted (200/200), so every claim below comes from WebFetch of official docs, source code, arXiv abstracts, or vendor engineering blogs. Where a page returned no example JSON, I say so rather than inventing one. Nothing was written to disk.

---

### 1. Real exported trace examples, per tool

#### 1.1 LangSmith (run record)
Verbatim from the run data format page (https://docs.langchain.com/langsmith/run-data-format), abbreviated:

```json
{"id":"497f6eca-...","name":"string","inputs":{},"run_type":"llm",
 "start_time":"2024-04-29T00:49:12.090000","end_time":"2024-04-29T00:49:12.459000",
 "extra":{},"error":"string","serialized":{},"outputs":{},
 "parent_run_id":"f8faf8c1-...","events":[{}],"tags":["foo"],
 "inputs_s3_urls":{},"outputs_s3_urls":{},
 "trace_id":"df570c03-...","dotted_order":"20240429T004912090000Z497f6eca-...",
 "status":"string","child_run_ids":[...],"parent_run_ids":[...],
 "feedback_stats":{"correctness":{"n":1,"avg":1.0}},"reference_example_id":"9fb06aaa-...",
 "total_tokens":0,"prompt_tokens":0,"completion_tokens":0,
 "total_cost":0.0,"prompt_cost":0.0,"completion_cost":0.0,
 "first_token_time":null,"session_id":"1ffd059c-...","in_dataset":true}
```
What is present: tool input/output live in the free-form `inputs`/`outputs` maps of a `run_type: "tool"` run; error is a single string; `status` is `error | pending | success` (pending is what a streaming or unfinished run looks like); user identity is not a first-class field (it goes in `extra`/metadata); session grouping is `session_id` (the project) plus `thread_id`/`session_id` metadata keys that must be propagated to every child run or thread filtering and token counting break (https://docs.langchain.com/langsmith/threads); cost and tokens are first-class; `inputs_s3_urls` shows that large payloads are offloaded rather than inlined; `dotted_order` exists precisely so children can be sorted after arriving out of order.

#### 1.2 Langfuse (observation)
The data-model page has no JSON example (https://langfuse.com/docs/observability/data-model), so the shape below is reconstructed from documented fields, not copied:

- Observation types: `event, span, generation, agent, tool, chain, retriever, evaluator, embedding, guardrail` (https://langfuse.com/docs/observability/features/observation-types). A tool observation is created with `startActiveObservation("weather-api-call", ..., { asType: "tool" })` and `update({ input: {location:"Paris",units:"metric"} })` then `update({ output: weather })`.
- Errors: `level` in `DEBUG | DEFAULT | WARNING | ERROR` plus free-text `statusMessage`; for OpenAI/LangChain integrations these are "automatically set based on the API response" (https://langfuse.com/docs/observability/features/log-levels).
- Trace-level `user_id`, `session_id`, `tags`, `metadata` "live on every observation within the trace"; traces are batched and flushed asynchronously, so short-lived processes lose data without `flush()` (https://langfuse.com/docs/observability/data-model).
- Ingestion: batches capped at 3.5 MB, deduplicated by event id, updates are upserts on the same body id, and input errors return HTTP 207 with a per-event error list instead of 4xx (https://raw.githubusercontent.com/langfuse/langfuse/main/web/public/generated/api/openapi.yml).
- Public demo project (docs Q&A bot with thumbs up/down feedback): https://cloud.langfuse.com/project/clkpwwm0m000gmm094odg11gi via https://langfuse.com/docs/demo. UI export is CSV or JSON, "all columns are always exported" (https://langfuse.com/docs/api-and-data-platform/features/export-from-ui).

#### 1.3 Braintrust (span)
No JSON example on the tracing pages, but the BTQL KB article enumerates the span fields: `id, span_id, root_span_id, span_parents, is_root, span_attributes.name, span_attributes.type, input, output, expected, metadata, metrics, scores, error, created` (https://braintrust.dev/docs/kb/understanding-traces-vs-spans-in-sql-btql-queries.md). Tool spans are selected with:
```
from: project_logs('<PROJECT_ID>') spans
| filter: span_attributes.type = "tool"
select: id, span_attributes.name, input, output
```
Span types are `llm, tool, task, function, score, classifier, eval` (https://www.braintrust.dev/docs/guides/tracing). Streaming is collapsed: "Braintrust automatically collects streamed chunks and logs the complete response as a single span" (https://www.braintrust.dev/docs/instrument/trace-llm-calls). User identity and session are metadata conventions, not schema.

#### 1.4 Arize Phoenix / OpenInference (span)
Verbatim (abbreviated) from https://arize.com/docs/phoenix/tracing/concepts-tracing/what-are-traces:
```json
{"name":"llm",
 "context":{"trace_id":"0x6c80880dbeb609e2ed41e06a6397a0dd","span_id":"0xd9bdedf0df0b7208"},
 "kind":"SpanKind.INTERNAL","parent_id":"0x7eb5df0046c77cd2",
 "start_time":"2024-05-08T21:46:11.480777Z","end_time":"2024-05-08T21:46:35.368042Z",
 "status":{"status_code":"OK"},
 "attributes":{"openinference.span.kind":"LLM",
   "llm.input_messages.0.message.role":"system",
   "llm.input_messages.1.message.content":"Hello",
   "llm.model_name":"gpt-4-turbo-preview",
   "llm.invocation_parameters":"{\"temperature\": 0.1, ...}",
   "output.value":"How are you?"},
 "events":[],"links":[],"resource":{"attributes":{},"schema_url":""}}
```
Note the flattened, index-numbered attribute keys (`retrieval.documents.2.document.score`), JSON-in-a-string for parameters and metadata, and OTel `status.status_code` for errors. Span kinds: `CHAIN, RETRIEVER, RERANKER, LLM, EMBEDDING, TOOL, AGENT`. Export is `client.spans.get_spans_dataframe()` or `SpanQuery().where("span_kind == 'RETRIEVER'").select(input="input.value")`, indexed by `span_id` (https://arize.com/docs/phoenix/tracing/how-to-tracing/importing-and-exporting-traces/extract-data-from-spans).

#### 1.5 W&B Weave (call)
Verbatim `CallSchema` from https://raw.githubusercontent.com/wandb/weave/master/weave/trace_server/trace_server_interface.py:
```
id, project_id, op_name, display_name, trace_id, parent_id, thread_id, turn_id,
started_at, ended_at, attributes: dict, inputs: dict, output: Any, exception: str | None,
summary: SummaryMap | None, wb_user_id, wb_run_id, deleted_at
```
`StartedCallSchemaForInsert` additionally carries `otel_dump`. Exceptions are a string. Session grouping is `thread_id` plus `turn_id`; turns are the top-level ops in a thread and nested calls are not counted as turns (https://docs.wandb.ai/weave/guides/tracking/threads). Cost/usage sits in `summary`.

#### 1.6 Helicone (tool call log)
Verbatim from https://docs.helicone.ai/integrations/tools (abbreviated):
```json
{"providerRequest":{"url":"custom-model-nopath",
   "json":{"_type":"tool","toolName":"weather_api","input":{"location":"San Francisco","units":"celsius"}},
   "meta":{"user_id":"user_123","session_id":"session_456","environment":"production"}},
 "providerResponse":{"json":{"_type":"tool","toolName":"weather_api","temperature":18.5,...},
   "status":200,"headers":{"content-type":"application/json"}},
 "timing":{"startTime":{"seconds":1625686222,"milliseconds":500},
           "endTime":{"seconds":1625686223,"milliseconds":750}}}
```
Session hierarchy is header-driven: `Helicone-Session-Id`, `Helicone-Session-Path` (e.g. `/abstract/outline`), `Helicone-Session-Name` (https://docs.helicone.ai/features/sessions). Cost is derived from the response `usage` block and model name; the cost page does not mention cached-token accounting (https://docs.helicone.ai/references/how-we-calculate-cost).

#### 1.7 Laminar
Span types `LLM`, `TOOL`, `DEFAULT`; spans carry input, output, attributes, session and user ids, cost, tokens, exceptions; the documented workflow is trace -> signals -> debugger -> "turn real failures into Datasets and run comprehensive Evaluations" (https://laminar.sh/docs/tracing/introduction). No JSON example on the intro page.

#### 1.8 Opik (Comet)
Trace and span fields: `id, trace_id, parent_span_id, name, type (llm | tool | general | guardrail), input, output, metadata, usage, total_estimated_cost, error_info, thread_id` (https://www.comet.com/docs/opik/tracing/log_traces). Agent traces show an "agent graph" and per-tool spans; multi-turn is `thread_id` (https://www.comet.com/docs/opik/tracing/log_agents).

#### 1.9 OpenAI Agents SDK (trace export payload)
Verbatim from source (https://raw.githubusercontent.com/openai/openai-agents-python/main/src/agents/tracing/spans.py and .../traces.py):
```python
{"object":"trace","id":trace_id,"workflow_name":name,"group_id":group_id,"metadata":metadata}
{"object":"trace.span","id":span_id,"trace_id":...,"parent_id":...,"started_at":...,"ended_at":...,
 "span_data":span_data.export(),"error":{"message":str,"data":dict|None}}
```
`span_data.export()` keys by type (https://openai.github.io/openai-agents-python/ref/tracing/span_data/): FunctionSpanData `type, name, input, output, mcp_data`; GenerationSpanData `type, input, output, model, model_config, usage`; ResponseSpanData `type, response_id, usage`; HandoffSpanData `from_agent, to_agent`; GuardrailSpanData `name, triggered`; MCPListToolsSpanData `server, result`; AgentSpanData `name, handoffs, tools, output_type`. `group_id` links traces of one conversation; `trace_include_sensitive_data` (default True) controls whether LLM and function inputs/outputs are captured at all (https://openai.github.io/openai-agents-python/tracing/). No cost field; usage only.

#### 1.10 Anthropic Claude Code transcript (JSONL) and Claude Agent SDK
Real fixture lines (https://raw.githubusercontent.com/daaain/claude-code-log/main/test/test_data/representative_messages.jsonl):
```json
{"type":"tool_use","id":"tool_002","name":"Bash","input":{"command":"python /tmp/decorator_example.py","description":"Run the decorator example to show output"}}
{"toolUseResult":"Hello, Alice!\n...","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_002","content":"Hello, Alice!\n...","is_error":false}]}}
"usage":{"input_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":35,"service_tier":"standard"}
```
Envelope fields per line (pydantic models at https://raw.githubusercontent.com/daaain/claude-code-log/main/claude_code_log/models.py): `parentUuid, isSidechain, userType, cwd, sessionId, version, uuid, timestamp, isMeta, agentId, gitBranch, teamName, spawnedAgentId`; user lines add `toolUseResult, sourceToolUseID`; assistant lines add `requestId` and `message.{id, model, content[], stop_reason, usage}`; `summary` lines have `summary, leafUuid` (compaction); `system` lines have `content, subtype, level`. A real system line: `"type":"system","content":"Claude Opus 4 limit reached, now using Sonnet 4"` (https://raw.githubusercontent.com/daaain/claude-code-log/main/test/test_data/system_model_change.jsonl). Sub-agents write `~/.claude/projects/{project}/{sessionId}/subagents/agent-{agentId}.jsonl`, get their own model (`model: haiku|sonnet|opus|inherit`), and their report returns to the parent as a plain `tool_result` of the Agent tool (https://code.claude.com/docs/en/sub-agents). Hooks see `session_id, transcript_path, cwd, permission_mode, tool_name, tool_input, tool_use_id` and PostToolUse adds `tool_response`; the docs warn the transcript "is written asynchronously and may lag the in-memory conversation" (https://code.claude.com/docs/en/hooks). The Agent SDK's `ResultMessage` carries `duration_ms, duration_api_ms, num_turns, total_cost_usd, usage, session_id, is_error, subtype` and subagent messages carry `parent_tool_use_id` and `parent_agent_id` (https://code.claude.com/docs/en/agent-sdk/python). Cost is only on the terminal ResultMessage; per-call cost must be recomputed from usage.

#### 1.11 MCP (JSON-RPC)
Verbatim from the spec (https://modelcontextprotocol.io/specification/2025-06-18/server/tools):
```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_weather","arguments":{"location":"New York"}}}
{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"Current weather in New York:..."}],"isError":false}}
{"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"Failed to fetch weather data: API rate limit exceeded"}],"isError":true}}
{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"Unknown tool: invalid_tool_name"}}
```
Two error channels (protocol `error` vs `isError:true` result), optional `structuredContent`, and the newer spec adds `resultType: "input_required"` with `requestState` for multi-round-trip tool calls and explicit state handles ("MCP has no protocol-level session") (https://modelcontextprotocol.io/specification/2026-07-28/server/tools). No user, cost, or latency fields at all; a recorder must add them.

#### 1.12 Vercel AI SDK
Spans `ai.generateText` > `ai.generateText.doGenerate` > `ai.toolCall` with attributes `ai.toolCall.name`, `ai.toolCall.args`, `ai.toolCall.result`, `ai.response.text`, `ai.usage.promptTokens/completionTokens`, `ai.telemetry.functionId`; `recordInputs`/`recordOutputs` flags can strip content entirely (https://ai-sdk.dev/docs/ai-sdk-core/telemetry).

Summary table (present = first-class field; meta = only via metadata convention; no = absent):

| Field | LangSmith | Langfuse | Braintrust | Phoenix | Weave | Helicone | OpenAI Agents | Claude Code | MCP |
|---|---|---|---|---|---|---|---|---|---|
| tool args / result | inputs/outputs | input/output | input/output | flattened attrs | inputs/output | providerRequest/Response | span_data.input/output | tool_use.input / tool_result.content | params / result.content |
| error | string + status | level + statusMessage | error | status_code | exception str | HTTP status | error{message,data} | is_error, toolUseResult | error vs isError |
| user id | meta | present | meta | attr | wb_user_id | meta.user_id | no | no (cwd, userType) | no |
| session | thread_id meta | session_id | meta | attr | thread_id, turn_id | header | group_id | sessionId, agentId | no |
| cost | present | present | metrics | no | summary | derived | no | usage only, cost on ResultMessage | no |
| latency | start/end, first_token_time | start/end | metrics | start/end | started/ended | timing | started/ended | timestamp only | no |
| env state | no | no | no | no | no | no | no | cwd, gitBranch | no |

---

### 2. Empirical characteristics practitioners report (2025 to 2026)

- Tool calls per session: Anthropic's analysis of 200,000 internal Claude Code transcripts found tool calls per session rose 116%, from 9.8 to 21.2 chained calls without interruption, while human turns per transcript fell from 6.2 to 4.1 (https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic).
- Automation vs augmentation: 79% of Claude Code conversations are automation vs 49% on Claude.ai; 35.8% of Claude Code interactions are "feedback loop" style where the human validates errors (https://www.anthropic.com/research/impact-software-development). On the 1P API, 77% of business uses are automation patterns and 97% of economic tasks are automation-dominant (https://www.anthropic.com/research/anthropic-economic-index-september-2025-report).
- Depth of agentic requests in the wild: Datadog reports 59% of agentic application requests made only a single service call and only 18% made three or more (https://www.datadoghq.com/state-of-ai-engineering/). Menlo Ventures finds only 16% of enterprise and 27% of startup deployments "qualify as true agents"; most are fixed-sequence or routing workflows around one model call (https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/).
- LLM error rates and causes: 5% of LLM call spans errored in February 2026 and 2% in March 2026; rate limits caused 60% of those errors in February and roughly a third in March (about 8.4M rate-limit errors) (https://www.datadoghq.com/state-of-ai-engineering/).
- Prompt caching artifacts: only 28% of LLM call spans show any cached-read input tokens, and 69% of all input tokens in customer traces are system prompt (same Datadog report). Claude transcripts carry `cache_creation_input_tokens` and `cache_read_input_tokens` per assistant message, so cost cannot be computed from `input_tokens` alone.
- Context growth: average prompt length on OpenRouter grew about fourfold from roughly 1.5K to over 6K tokens; reasoning models now exceed 50% of routed tokens; tool-calling token share rose consistently through 2025 (https://openrouter.ai/state-of-ai).
- Long-horizon benchmarks calibrated on real apps: Toolathlon tasks span 32 apps and 604 tools and take about 20 turns; Claude 4.5 Sonnet succeeds 38.6% with 20.2 tool-calling turns on average (https://arxiv.org/abs/2510.25726). CRMArena-Pro shows about 58% single-turn success dropping to about 35% multi-turn (https://arxiv.org/abs/2505.18878). LiveMCPBench (70 servers, 527 tools) attributes nearly half of failures to tool retrieval (https://arxiv.org/abs/2508.01780). MCPToolBench++ (4k+ servers, 40+ categories) reports that "the success rate of real-world MCP tool is not guaranteed and varies across different MCP servers" and that response formats are heterogeneous (https://arxiv.org/abs/2508.07575).
- Failure taxonomies from traces: MAST annotated 150 traces (1600+ in MAST-Data) into 14 failure modes across system design, inter-agent misalignment, and task verification (https://arxiv.org/abs/2503.13657); TRAIL provides 148 human-annotated long traces and finds the best model localizes issues in only 11% of cases (https://arxiv.org/abs/2505.08638); AgentErrorTaxonomy spans memory, reflection, planning, action, and system-level errors (https://arxiv.org/abs/2509.25370).
- Truncation of tool outputs is the norm, not the exception: Claude Code reads back at most 30,000 characters of bash output by default (max 150,000) (https://code.claude.com/docs/en/env-vars); Anthropic restricts Claude Code tool responses to 25,000 tokens by default and recommends pagination, range selection, filtering, and truncation for any tool (https://www.anthropic.com/engineering/writing-tools-for-agents). Anthropic's own transcript reading found "lots of redundant tool calls" and "lots of tool errors for invalid parameters" as the recurring patterns (same page). Real tool payloads can be 50,000 tokens for one meeting transcript or a 10,000-row spreadsheet (https://www.anthropic.com/engineering/code-execution-with-mcp).
- Sampling: Langfuse and LangSmith both default to 1.0 and sample at trace level client-side (`LANGFUSE_SAMPLE_RATE`, `LANGSMITH_TRACING_SAMPLING_RATE`); LangSmith recommends conditional tracing for zero-retention tenants (https://langfuse.com/docs/observability/features/sampling, https://docs.langchain.com/langsmith/sample-traces).
- PII: Langfuse ships regex masks for credit cards, emails, phones that rewrite `input`, `output`, `metadata` before export (https://langfuse.com/docs/observability/features/masking); Vercel AI SDK `recordInputs/recordOutputs` and OpenAI Agents `trace_include_sensitive_data` strip content entirely. CRMArena-Pro found agents "exhibit near-zero inherent confidentiality awareness" (https://arxiv.org/abs/2505.18878), so PII in tool results is expected.
- Missing tool_result: the Claude API rejects a message history with a `tool_use` that lacks an immediately following `tool_result` with a 400 ("tool_use ids were found without tool_result blocks immediately after") (https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls). `max_tokens` can cut off mid `tool_use`, producing an incomplete tool block that must be retried (https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons). Server tools (web search) run server-side with `pause_turn` and never produce a client `tool_result`. Claude Code hook transcripts lag the in-memory conversation (https://code.claude.com/docs/en/hooks).
- Duplicated retries: OpenAI's Python SDK retries connection errors, 408, 409, 429, and 5xx twice by default with backoff, without idempotency keys (https://raw.githubusercontent.com/openai/openai-python/main/README.md); Claude retries invalid tool calls "2-3 times with corrections" on its own (https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).
- Out-of-order and streaming records: LangSmith's `dotted_order` and `status: pending`, Langfuse's async batching with upsert-by-id and 207 partial success, and Braintrust's collapse of streamed chunks into one span all exist because spans arrive late and partial.
- Sub-agent nesting: Claude Code sidechains (`isSidechain: true`, separate `agent-{agentId}.jsonl`), OpenAI handoff spans (`from_agent`, `to_agent`), LangSmith `parent_run_ids`, Weave `turn_id`.
- Model drift within a session: real transcript line `"Claude Opus 4 limit reached, now using Sonnet 4"`; subagents resolve their model from `CLAUDE_CODE_SUBAGENT_MODEL` > invocation > frontmatter > parent (https://code.claude.com/docs/en/sub-agents). Datadog: over 70% of organizations use three or more models.

---

### 3. How vendors turn traces into datasets

| Vendor | Unit | Stored as input / expected | Tool calls in the item | Multi-step replay stance |
|---|---|---|---|---|
| Langfuse | Observation (Observations table > Actions > Add to dataset), with `source_trace_id` and `source_observation_id` kept; field mapping via JSON path (https://langfuse.com/docs/evaluation/experiments/datasets) | observation input -> `input`, observation output -> `expected_output` | Not addressed | Not addressed |
| LangSmith | Run (any intermediate run, "but not the root run" in annotation queues); Thread (up to 100, whole conversation) (https://docs.langchain.com/langsmith/annotation-queues, https://docs.langchain.com/langsmith/manage-datasets-in-application) | run inputs/outputs, reviewer can edit outputs or write Assertions as expected output; thread examples "include the conversation as input only. They do not include a reference output" | Trajectory matching via `agentevals` on OpenAI-style message lists with `tool_calls` and `tool` role; modes strict, unordered, subset, superset; `tool_args_match_mode` exact/ignore/subset (https://raw.githubusercontent.com/langchain-ai/agentevals/main/README.md) | Not documented on fetched pages |
| Braintrust | Span or full trace by reference; "The span's `input` maps to the dataset row's `input`, and the span's `output` typically becomes the row's `expected`"; full-trace rows store "a pointer to the logged trace" and break with "Referenced trace not found" after retention (https://braintrust.dev/docs/annotate/datasets/create.md); bulk via dataset pipelines with `scope: "span"` or `"trace"` and a BTQL filter, run manually with `--window` (https://braintrust.dev/docs/annotate/datasets/pipelines.md) | as above | Guidance: "Snapshot tool calls and responses from live environments" and "Stub external dependencies: Snapshot sufficient state from production or staging environments to simulate databases, APIs, and infrastructure" (https://braintrust.dev/docs/best-practices/agents.md) | Evaluate whole and each step; evaluate steps individually "to manage non-deterministic agent interactions" |
| Arize Phoenix | Span (trace detail > add to dataset, or bulk from filtered spans table) (https://arize.com/docs/phoenix/datasets-and-experiments/how-to-datasets/creating-datasets) | selected span attributes, editable before save | Tool spans are just spans | Not addressed |
| Weave | Call (`Dataset.from_calls([call1, call2])`, or UI field picker with column renaming); a separate "Add agent messages to a dataset" flow exists (https://docs.wandb.ai/weave/guides/core-types/datasets) | selected call fields | Agent turns and tool calls supported via that separate flow | Not addressed |
| Opik | Trace (select traces > Add to dataset); "The trace's input and output are always included"; optional nested `spans` array with inputs/outputs/metadata, tags, feedback scores, usage (https://www.comet.com/docs/opik/evaluation/manage_datasets) | trace input -> `input`, trace output -> `expected_output` | Kept as nested spans if opted in | Not addressed |
| Laminar | Trace/failure -> Dataset (https://laminar.sh/docs/tracing/introduction) | not detailed | not detailed | not detailed |
| OpenAI | Single completion: `{"type":"stored_completions","metadata":{...}}` with `{{item.input}}` and `{{sample.output_text}}` (https://developers.openai.com/cookbook/examples/evaluation/use-cases/completion-monitoring); JSONL items with human `correct_label` (https://developers.openai.com/api/docs/guides/evals) | messages -> input; output_text -> sample | Not addressed in evals; trace grading scores the whole trace "the end-to-end log of decisions, tool calls, and reasoning steps" from Agents SDK apps and Agent Builder (https://developers.openai.com/api/docs/guides/trace-grading) | Grade, do not replay |
| Anthropic | The fetched eval-tool page documents programmatic single-turn and multi-turn message-list test cases and does not describe tool-call replay or creation from logs (https://platform.claude.com/docs/en/test-and-evaluate/eval-tool) | input + expected_output | Not addressed | Not addressed |

Cross-vendor pattern: the unit is almost always one span or one LLM call, `expected` is just the production output copied over, tool results are frozen text, and nobody documents environment replay. Braintrust is the only vendor that explicitly says to snapshot external state.

---

### 4. What is systematically missing from production traces for a replica, and who captures it

Missing in every observability export above:
1. The tool definitions/schemas that were in the prompt at the time (only Claude Code transcripts and OpenAI `AgentSpanData.tools`/`MCPListToolsSpanData.result` keep any of this).
2. Backend state before and after each write (database rows, files, tickets); traces contain the tool's text reply, not the state it read from or mutated.
3. Tool semantics (read-only vs destructive, idempotent, open-world). MCP has hints for this, `readOnlyHint` default false, `destructiveHint` default true, `idempotentHint` default false, `openWorldHint` default true, but they are advisory and "not guaranteed to provide a faithful description" (https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2025-06-18/schema.ts).
4. Wall clock and randomness (current date drives errors like "Invalid departure date: must be in the future. Current date is 08/08/2025" in the MCP spec example).
5. Untruncated tool outputs (30k chars / 25k tokens limits above; S3 offload in LangSmith).
6. Identity and authorization scope (MCP tool lists "MAY vary by the authorization presented on the request", https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
7. Permission decisions and human interventions (Claude Code `permission_mode`, hook denials, MCP `input_required` elicitations).
8. Server-side tool executions (web search under `pause_turn`) that never produce client-visible results.

Tooling that captures some of it:
- agent-vcr (Python + TS, MIT): records MCP JSON-RPC into `.vcr` JSON with `metadata{version, recorded_at, transport, client_info, server_info, tags}`, the `initialize` handshake, and ordered interactions `{sequence, timestamp, direction, request, response, latency_ms}`; matching strategies `exact | method | method_and_params | subset | sequential`, excludes `jsonrpc`/`id` when matching, supports response overrides for error injection and `agent-vcr diff --fail-on-breaking` (https://github.com/jarvis2021/agent-vcr, https://pypi.org/pypi/agent-vcr/json).
- mcp-recorder ("VCR.py for MCP servers"): wire-level cassettes with timestamp, server info, protocol version, transport; replay and verify modes, motivated by "Tool schemas change, prompts drift, responses shift" (https://pypi.org/pypi/mcp-recorder/json).
- VCR.py for HTTP-level tool backends: YAML cassettes, record modes `once | new_episodes | none | all`, `match_on` method/uri/body/headers (https://vcrpy.readthedocs.io/en/latest/usage.html).
- Plato: sims are VM images of real apps ("a CRM, a Git server, a wiki, a Linux desktop", e.g. `espocrm`); a mutation is "a tracked change to the env (a database write, a file change) since the last `reset()`", `session.get_state()` returns the mutation set, and MUTATION-mode test cases declare the DB/file changes a successful run must produce (https://docs.plato.so/concepts).
- HUD: MCP/SSH/CDP/RFB environments, records every rollout as a trace, Harbor task conversion, `env.workspace()` sandboxes; no documented ingestion of customer traces (https://raw.githubusercontent.com/hud-evals/hud-python/main/README.md).
- OpenEnv: Gym-style `reset/step/state` with MCP tool actions (`CallToolAction`), no recorder or trace-to-env feature (https://github.com/meta-pytorch/OpenEnv).
- Veris AI: "a simulated copy of your systems, data, APIs, and users", synthetic data, "no production data" required; production failure logs are fed through an observability pipeline, an LLM analyzer builds a rubric and generates about 30 variant scenarios, tested on n=20 held-out sets; "Every system it touches is replaced by a mock that answers the way the real one would" (https://veris.ai, https://www.veris.ai/blog/never-waste-a-good-failure-how-veris-ai-turns-production-incidents-into-self-improving-agents, https://www.veris.ai/blog/the-loop-is-the-easy-part).
- Database snapshotting: Neon branching and Snapshots API give copy-on-write "restorable checkpoints" so agents can roll back or diff database state (https://neon.com/use-cases/ai-agents).
- Claude Code itself: `cwd`, `gitBranch`, hooks with `tool_input`/`tool_response`, and per-agent transcripts are the richest free source of environment context among the formats reviewed.

---

### 5. Enterprise agent tool mixes

No vendor publishes a production read/write split. What exists:
- Spend and use-case proxies: coding is 55% of departmental AI spend ($4.0B), then IT 10%, marketing 9%, customer success 9%; agent platforms are 10% of horizontal spend (https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/). LangChain's survey: customer service 26.5%, research and data analysis 24.4%, internal workflow automation 18% of agent use cases; 89% have observability and 62% detailed tracing (https://www.langchain.com/state-of-agent-engineering). Anthropic API traffic: software development dominates, office/administrative about 10% (https://www.anthropic.com/research/anthropic-economic-index-september-2025-report).
- Tool catalog shape: PulseMCP lists about 22,000 MCP servers with trending categories Developer Tools, Databases, Productivity (Notion, Google Workspace, Slack, Excel), Communication (Gmail, Outlook, WhatsApp), Cloud, Enterprise (SAP, Salesforce, Jira) (https://www.pulsemcp.com/servers). MCPToolBench++: 40+ categories including search, web crawlers, maps, financial data, file systems, browser (https://arxiv.org/abs/2508.07575). Anthropic's canonical enterprise examples are Google Drive, Salesforce, Slack, and "hundreds or thousands of tools across dozens of MCP servers" (https://www.anthropic.com/engineering/code-execution-with-mcp).
- Read vs write evidence from enterprise benchmarks: CRMArena-Pro gives agents only read-only SOQL/SOSL queries over 25 Salesforce objects plus a `Respond` action (https://arxiv.org/html/2505.18878); tau2-bench models telecom support where both agent and user hold state-mutating tools (https://arxiv.org/abs/2506.07982); Toolathlon mixes Google Calendar, Notion, WooCommerce, Kubernetes, BigQuery with real spreadsheets (https://arxiv.org/abs/2510.25726). Datadog's 59% single-call figure implies most production "agents" perform one read/lookup per request.

---

### 6. Synthesis

Minimal fields a customer trace must carry for a faithful replica:
1. Identity and grouping: `session/thread_id`, `trace_id`, `span_id`, `parent_id` (and agent id / `parent_tool_use_id` for sub-agents), plus a monotone ordering key (timestamp and a `dotted_order`/`sequence`).
2. Per LLM call: exact `model` id, full input messages including system prompt and the tool definitions array as sent, invocation parameters, output content blocks with `tool_use` ids, `stop_reason`, `usage` including cache creation/read tokens, `request_id`.
3. Per tool call: tool `name` and server/namespace, raw `arguments`, raw result content (untruncated, or hash plus pointer), error channel (`is_error`/`isError`/JSON-RPC error code), `tool_use_id` linkage, start/end time, and the MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) or an equivalent customer-provided read/write label.
4. Environment: a state reference before the trace (DB snapshot or branch id, file tree hash, ticket/CRM record ids), a mutation log during the trace (Plato-style), and cassettes for external APIs (agent-vcr / VCR.py) keyed by method and params.
5. Clock and seeds: wall-clock at trace start, any random seeds, timezone.
6. Actor context: `user_id`, authorization scope or tenant, `permission_mode`, human interventions and denials.
7. Outcome: final answer, feedback scores or annotations, `is_error`, `total_cost_usd`.
8. Versioning: application release, prompt version, tool schema version (agent-vcr `diff` exists because these drift).

Top 10 anomalies an ingestion pipeline must handle:
1. `tool_use` with no matching `tool_result` (interruption, `max_tokens` cut mid-block, transcript write lag, permission denial, server-side tools with `pause_turn`). Synthesize a terminal marker rather than dropping the turn.
2. Duplicate records from SDK retries (2 by default for 408/409/429/5xx) and model self-retries of invalid tool calls; dedupe on `request_id`/`message.id` and Langfuse-style event ids.
3. Out-of-order and late spans, and streaming spans that are `pending` at export time; sort by `dotted_order`/`sequence`, not arrival, and treat missing `end_time` as open.
4. Truncated tool outputs (30k chars, 25k tokens, S3 offload, "Referenced trace not found" after retention); flag truncation and refuse to use them as replay fixtures.
5. Three incompatible error encodings for the same event (JSON-RPC `error`, `isError:true` result, `is_error` on `tool_result`, `level: ERROR` + `statusMessage`, OTel `status_code`, string `error`); normalize to one taxonomy that separates protocol, execution, and rate-limit errors (rate limits were up to 60% of LLM errors).
6. Sampled or masked traces: `sample_rate` < 1 loses whole sessions, regex masking rewrites inputs/outputs so tool arguments no longer match cassettes; detect placeholder patterns and mark the span non-replayable.
7. Sub-agent transcripts in separate files or spans (`agent-{agentId}.jsonl`, `isSidechain`, handoff spans) whose results reappear in the parent as a plain `tool_result`; stitch by `parent_tool_use_id`.
8. Model and version drift inside one session (fallback to another model on limit, subagents on Haiku, prompt version changes); store model per call, never per session.
9. Prompt-cache token accounting (`cache_creation_input_tokens`, `cache_read_input_tokens`, only 28% of spans show cache reads) and system prompt dominance (69% of input tokens); cost must be recomputed per call and the cached prefix reconstructed for replay.
10. Non-deterministic and stateful tools: wall-clock dependent validation, search results, MCP `input_required` round trips, opaque state handles (`basket_id`), and `tools/list` that varies by authorization; record clock and handle values, and match cassettes on method plus normalized params with ids stripped (as agent-vcr does).

Sources used, in order of first appearance: https://docs.langchain.com/langsmith/run-data-format, https://docs.langchain.com/langsmith/threads, https://langfuse.com/docs/observability/data-model, https://langfuse.com/docs/observability/features/observation-types, https://langfuse.com/docs/observability/features/log-levels, https://raw.githubusercontent.com/langfuse/langfuse/main/web/public/generated/api/openapi.yml, https://langfuse.com/docs/demo, https://langfuse.com/docs/api-and-data-platform/features/export-from-ui, https://braintrust.dev/docs/kb/understanding-traces-vs-spans-in-sql-btql-queries.md, https://www.braintrust.dev/docs/guides/tracing, https://www.braintrust.dev/docs/instrument/trace-llm-calls, https://arize.com/docs/phoenix/tracing/concepts-tracing/what-are-traces, https://arize.com/docs/phoenix/tracing/how-to-tracing/importing-and-exporting-traces/extract-data-from-spans, https://raw.githubusercontent.com/wandb/weave/master/weave/trace_server/trace_server_interface.py, https://docs.wandb.ai/weave/guides/tracking/threads, https://docs.helicone.ai/integrations/tools, https://docs.helicone.ai/features/sessions, https://docs.helicone.ai/references/how-we-calculate-cost, https://laminar.sh/docs/tracing/introduction, https://www.comet.com/docs/opik/tracing/log_traces, https://www.comet.com/docs/opik/tracing/log_agents, https://raw.githubusercontent.com/openai/openai-agents-python/main/src/agents/tracing/spans.py, https://raw.githubusercontent.com/openai/openai-agents-python/main/src/agents/tracing/traces.py, https://openai.github.io/openai-agents-python/ref/tracing/span_data/, https://openai.github.io/openai-agents-python/tracing/, https://raw.githubusercontent.com/daaain/claude-code-log/main/test/test_data/representative_messages.jsonl, https://raw.githubusercontent.com/daaain/claude-code-log/main/claude_code_log/models.py, https://raw.githubusercontent.com/daaain/claude-code-log/main/test/test_data/system_model_change.jsonl, https://code.claude.com/docs/en/sub-agents, https://code.claude.com/docs/en/hooks, https://code.claude.com/docs/en/agent-sdk/python, https://modelcontextprotocol.io/specification/2025-06-18/server/tools, https://modelcontextprotocol.io/specification/2026-07-28/server/tools, https://ai-sdk.dev/docs/ai-sdk-core/telemetry, https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic, https://www.anthropic.com/research/impact-software-development, https://www.anthropic.com/research/anthropic-economic-index-september-2025-report, https://www.datadoghq.com/state-of-ai-engineering/, https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/, https://openrouter.ai/state-of-ai, https://arxiv.org/abs/2510.25726, https://arxiv.org/abs/2505.18878, https://arxiv.org/html/2505.18878, https://arxiv.org/abs/2508.01780, https://arxiv.org/abs/2508.07575, https://arxiv.org/abs/2503.13657, https://arxiv.org/abs/2505.08638, https://arxiv.org/abs/2509.25370, https://arxiv.org/abs/2506.07982, https://code.claude.com/docs/en/env-vars, https://www.anthropic.com/engineering/writing-tools-for-agents, https://www.anthropic.com/engineering/code-execution-with-mcp, https://langfuse.com/docs/observability/features/sampling, https://docs.langchain.com/langsmith/sample-traces, https://langfuse.com/docs/observability/features/masking, https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls, https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons, https://raw.githubusercontent.com/openai/openai-python/main/README.md, https://langfuse.com/docs/evaluation/experiments/datasets, https://docs.langchain.com/langsmith/annotation-queues, https://docs.langchain.com/langsmith/manage-datasets-in-application, https://raw.githubusercontent.com/langchain-ai/agentevals/main/README.md, https://braintrust.dev/docs/annotate/datasets/create.md, https://braintrust.dev/docs/annotate/datasets/pipelines.md, https://braintrust.dev/docs/best-practices/agents.md, https://arize.com/docs/phoenix/datasets-and-experiments/how-to-datasets/creating-datasets, https://docs.wandb.ai/weave/guides/core-types/datasets, https://www.comet.com/docs/opik/evaluation/manage_datasets, https://developers.openai.com/cookbook/examples/evaluation/use-cases/completion-monitoring, https://developers.openai.com/api/docs/guides/evals, https://developers.openai.com/api/docs/guides/trace-grading, https://platform.claude.com/docs/en/test-and-evaluate/eval-tool, https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2025-06-18/schema.ts, https://github.com/jarvis2021/agent-vcr, https://pypi.org/pypi/agent-vcr/json, https://pypi.org/pypi/mcp-recorder/json, https://vcrpy.readthedocs.io/en/latest/usage.html, https://docs.plato.so/concepts, https://raw.githubusercontent.com/hud-evals/hud-python/main/README.md, https://github.com/meta-pytorch/OpenEnv, https://veris.ai, https://www.veris.ai/blog/never-waste-a-good-failure-how-veris-ai-turns-production-incidents-into-self-improving-agents, https://www.veris.ai/blog/the-loop-is-the-easy-part, https://neon.com/use-cases/ai-agents, https://www.pulsemcp.com/servers, https://www.langchain.com/state-of-agent-engineering.

Pages that could not be fetched (403/404/429) and are therefore not cited: Weave tracing docs on weave-docs.wandb.ai, LangSmith agent-evaluation tutorial and limits pages, Braintrust API log schema, OpenAI State of Enterprise AI report, Datadog agent-monitoring blog, docs.hud.so (rate limited; GitHub README used instead), mcp-recorder GitHub (PyPI used instead).