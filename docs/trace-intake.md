# Customer trace intake

> Status (2026-09-10): open todo; questions 2 to 13 are unanswered.

## Formats read today

Every format enters through one adapter in `kullback/builder/sources/`; all four map to Traces.

| Format | Adapter | Reads | Rejects |
|--------|---------|-------|---------|
| tau2 export | `tau2_native` | the simulations list, one messages list per recording, `info.environment_info` | a recording the records refuse |
| terminus-2 rows | `terminus_2` | the `conversations` turn list, shell batches as JSON in assistant text | a recording the records refuse |
| OpenTelemetry GenAI spans | `otel_genai` (mapped, D318) | OTLP JSON or a flat span list, one recording per `traceId`; `gen_ai.input.messages`, `gen_ai.output.messages` (text, `tool_call`, `tool_call_response` parts), `gen_ai.system_instructions`, the `gen_ai.system.message`, `gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message` and `gen_ai.choice` events, `execute_tool` spans (`gen_ai.tool.call.id`, `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `error.type`, span status), `gen_ai.tool.definitions` or `gen_ai.request.tools` | a trace whose GenAI spans carry no message, event or tool call |
| Claude Code JSONL | `claude_code_jsonl` (mapped, D318) | one recording per `sessionId`; user, assistant and system lines with a `message` (text, `tool_use`, `tool_result` with `is_error`), ordered by `parentUuid`; other line types counted in the sidecar as `non_turn_lines` | a recording the records refuse |

Questions to answer when a customer's traces arrive (first: the vendor export expected in early September 2026, D56). Fill in the answers here; the ingestion todo is written around them.

| # | Question | Answer | Why it matters |
|---|----------|--------|----------------|
| 1 | Source: whose agent produced the traces (own agent, design partner, vendor export) and which domain (support, ticketing, sales, ops)? | vendor export; domain unknown | Domain decides which tool families and policy shapes to expect. |
| 2 | Format: raw model API request/response logs (OpenAI or Anthropic message JSON), a tracing export (LangSmith, Langfuse, OpenTelemetry GenAI spans), or the agent framework's own log? One example file answers this. | | Ingestion parser; how tool calls, results and errors are encoded (D45, R23 section 4). |
| 3 | Tool results: kept in full or truncated? | | Truncated results block S0 reconstruction for reads (D39, D40). |
| 4 | Tool definitions: is the `tools` list sent to the model present in the log? | | Day-one contracts for unseen tools (ADR-0006 rung 2). |
| 5 | Grouping: are the turns of one conversation linked by a session or thread id? | | Runs cannot be assembled without it (R18 filters). |
| 6 | Errors: do failed tool calls appear with their error text? How many per tool? | | The Environment copies the real error encoding; none seen means a guess (A27). |
| 7 | System prompt: present per Call? | | Policy lines compile to Hard constraints from it (D43 case 3). |
| 8 | Volume: conversations, period, distinct tools, share of multi-turn Runs. | | Task floor and CI (D36); Simulated user hold-out (D44). |
| 9 | Outcome signals: CSAT, escalation, reopen, refund reversal, anything the customer records after the Run? | | Reference confirmation (step 5) and audit calibration. |
| 10 | Labels: can a domain expert mark 20 to 50 conversations pass or fail? | | D50 proof 3; D48 check 1 second pair of eyes. |
| 11 | Policy or knowledge documents available beyond the system prompt? | | ADR-0006 rung 4. |
| 12 | Where the traces live: the gitignored `data/` folder of this repository, a folder outside the repository, or a bucket. Never committed. | | Customer data must not enter git history. |
| 13 | Retention and sharing rules the customer set (what may leave their boundary, for how long). | | ADR-0002 and rung 6 (snapshot inside their boundary). |
