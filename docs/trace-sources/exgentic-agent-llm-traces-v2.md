# Exgentic/agent-llm-traces-v2

## Link and owner

https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2, published by the Hugging Face org Exgentic.
Parent corpus: https://huggingface.co/datasets/Exgentic/traces-v2 (this dataset is a filtered derivative of it).

## What a row is

One row is one agent session: a full run of one harness plus one model against one benchmark task, recorded
as OpenTelemetry GenAI spans (one span per model call, with tool definitions, input messages, output messages
and usage attached to the span). The dataset card: "OpenTelemetry-shaped execution traces for 10,057 agent runs
across 6 benchmarks (AppWorld, SWE-bench, BrowseCompPlus, tau2-bench Airline/Retail/Telecom), filtered to the
agent under test's chat-only LLM calls." (dataset card, huggingface.co/datasets/Exgentic/agent-llm-traces-v2)

## Provenance

Model generated: five LLMs (DeepSeek-V3.2, Kimi-K2.5, claude-opus-4-5, gemini-3-pro-preview,
gpt-5.2-2025-12-11) run through five agent harnesses (claude_code, openai_solo, smolagents_code,
tool_calling, tool_calling_with_shortlisting) against six existing benchmarks' own task sets (AppWorld,
SWE-bench, BrowseCompPlus, tau2 airline, retail, telecom). None of the underlying tasks, environments or
instructions are new; Exgentic ran established benchmarks and captured the spans. Card: "Derived from the
full corpus Exgentic/traces-v2; re-running that single script on the full corpus reproduces this dataset
bit-for-bit." (huggingface.co/api/datasets/Exgentic/agent-llm-traces-v2, description field)

## Counts and size

10,056 rows (one parquet split, `train`), 231,705,307 bytes of parquet on disk, 20 columns
(datasets-server `/size` endpoint, checked 2026-09-24). Context.md's own scout pass, done while the Hub's
filter index was mid-rebuild, recorded 10,057 sessions and 241,674 chat spans; this pass counted 10,056 rows
directly from the 9 downloaded parquet shards (236 MB total on disk), so the row count is effectively the
same figure, one row apart, likely a single row added or removed between the two checks.

Sessions per benchmark (counted from all 9 shards, column `benchmark`):

| Benchmark | Sessions |
| --- | --- |
| swebench | 1,959 |
| browsecompplus | 1,948 |
| tau2_retail | 1,848 |
| tau2_telecom | 1,844 |
| appworld | 1,500 |
| tau2_airline | 957 |

Sessions per harness (column `harness`): tool_calling 2,643, openai_solo 2,608, smolagents_code 2,434,
claude_code 2,071, tool_calling_with_shortlisting 300. Only `tool_calling` and
`tool_calling_with_shortlisting` (2,943 sessions together) record one function call per assistant turn in
the shape our ingest expects; `claude_code` and `openai_solo` are coding-agent scaffolds that can emit
several tool calls or none per turn, and `smolagents_code` runs Python code blocks, not discrete tool calls.

Sessions per model (column `models`, a list so a row can count against more than one model though every
row observed here names exactly one): DeepSeek-V3.2 2,299, Kimi-K2.5 2,285, gpt-5.2-2025-12-11 2,124,
claude-opus-4-5 1,920, gemini-3-pro-preview 1,428.

Benchmark by harness (sessions):

| Benchmark | claude_code | openai_solo | smolagents_code | tool_calling | tool_calling_with_shortlisting |
| --- | --- | --- | --- | --- | --- |
| appworld | 200 | 400 | 200 | 400 | 300 |
| browsecompplus | 451 | 497 | 500 | 500 | 0 |
| swebench | 495 | 476 | 491 | 497 | 0 |
| tau2_airline | 210 | 249 | 249 | 249 | 0 |
| tau2_retail | 356 | 498 | 496 | 498 | 0 |
| tau2_telecom | 359 | 488 | 498 | 499 | 0 |

Benchmark by model (sessions):

| Benchmark | DeepSeek-V3.2 | Kimi-K2.5 | claude-opus-4-5 | gemini-3-pro-preview | gpt-5.2-2025-12-11 |
| --- | --- | --- | --- | --- | --- |
| appworld | 500 | 500 | 200 | 300 | 0 |
| browsecompplus | 400 | 399 | 351 | 398 | 400 |
| swebench | 399 | 390 | 379 | 393 | 398 |
| tau2_airline | 200 | 200 | 197 | 79 | 281 |
| tau2_retail | 400 | 397 | 397 | 129 | 525 |
| tau2_telecom | 400 | 399 | 396 | 129 | 520 |

Counted 2026-09-24 by downloading the 9 `data/train/*.parquet` shards with curl and reading only the
`benchmark`, `harness`, `benchmark_subset`, `models`, `run_id`, `session_id` columns (the `spans` column
holds the full OpenTelemetry payload and inflates a single row to roughly 15 MB decoded, so it was read
column-by-column, never in full, to keep the download and decode small).

## Runs per task

AppWorld: every row in this dataset is `benchmark_subset = test_normal`, the 168-task AppWorld test split.
The 1,500 appworld sessions split into exactly 15 distinct (harness, model) configurations, each with
precisely 100 sessions (for example `claude_code` x `DeepSeek-V3.2` is one `run_id` with 100 sessions). That
is a fixed-size 100-task sample of the 168-task split, repeated per configuration; whether it is the same
100 tasks in every configuration was not confirmed here (would need a task-level id, which is not a top-level
column and sits inside the OpenTelemetry span content, not pulled for this pass). If it is the same 100
tasks, that is up to 15 runs per task; if not, fewer. See docs/trace-sources/appworld.md for how this set
combines with AppWorld's other trace sources.

tau2 airline (`benchmark_subset = airline`): 24 distinct (harness, model) configurations, with session
counts per configuration of 1, 30, 49 or 50 (mostly 49 or 50), so up to 24 runs per task, assuming shared
task coverage across configurations (not confirmed at the task level, same caveat as AppWorld above).

tau2 retail (`benchmark_subset = retail`): 24 distinct (harness, model) configurations, with session counts
of 1, 26, 30, 98, 99 or 100 (mostly 98 to 100), so up to 24 runs per task, same caveat.

## Schema

Row-level columns (from the parquet schema, checked with `pyarrow.parquet.ParquetFile(...).schema_arrow`):
`schema_version`, `config_path`, `run_id`, `session_id`, `harness`, `benchmark`, `benchmark_subset`,
`models` (list<string>), `score`, `success`, `status`, `steps`, `action_count`, `agent_cost`,
`benchmark_cost`, `execution_time`, `total_tokens`, `max_tokens`, `spans`, `collected_at`. `spans` is a
list of OpenTelemetry GenAI spans, each: `span_id`, `trace_id`, `parent_span_id`, `name`, `kind`,
`start_time`, `end_time`, `status`, `attributes` (a struct including `gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.tool.definitions`, `gen_ai.system_instructions`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and others), and
`resource_attributes`.

## One excerpt

From a `tool_calling`-harness, `browsecompplus`-benchmark session (session_id
`10d642ac9c10_98c6dc29`, 4 steps, the smallest `tool_calling` session found in this pass, so the smallest
excerpt to trim), one span's `gen_ai.tool.definitions` shows the declared tool schema:

```json
{"type": "function", "name": "get_document",
 "description": "Retrieve the full document using its document id",
 "parameters": {"properties": {"docid": {"title": "Docid", "type": "string"}},
                "required": ["docid"], "title": "BrowseCompPlusGetDocumentsArgs", "type": "object"}}
```

That same span's `gen_ai.output.messages` records the model's call:

```json
[{"role": "assistant", "parts": [{"type": "tool_call",
   "id": "call_Azl3MTMtzlFGvhUS7TXhGoQQ", "name": "get_document",
   "arguments": {"docid": "68273"}}], "finish_reason": "tool_calls"}]
```

The following span's `gen_ai.input.messages` carries the tool's answer back as a role `tool` message, one
JSON-encoded search-result object per entry (docid, score, snippet), trimmed here for length. Each span is
one model turn, not one tool call; `gen_ai.output.messages` for `tool_calling` sessions holds exactly one
tool call per span in every case checked, matching our one-call-per-turn assumption for this harness.

## Tool-call shape against our Trace record

Read from kullback/runner/records.py: `ToolCall` has `name`, `args`, `result`, `requestor`, `error`,
`raw_ptr`, `result_ptr`; `Turn` has `idx`, `role`, `content`, `tool_call_ids`. A `tool_calling` or
`tool_calling_with_shortlisting` span maps cleanly: `gen_ai.output.messages[].parts[].name/arguments` gives
`ToolCall.name`/`args`, and the next span's matching `tool`-role message in `gen_ai.input.messages` gives
`ToolCall.result`. `claude_code`, `openai_solo` and `smolagents_code` spans do not hold this shape reliably:
`claude_code` and `openai_solo` are full coding-agent scaffolds that can bundle several tool calls, or code
execution with no discrete tool call, into one span; `smolagents_code` sessions execute Python code blocks
per turn, closer to AppWorld's own native format than to a tool-call trace. Only the 2,943
`tool_calling`/`tool_calling_with_shortlisting` sessions (29% of the dataset) are directly ingestable as
Trace/ToolCall records without rebuilding the call boundary from code or scaffold-specific logs.

## Environment and reference verdict

None. The dataset card states no environment or state is included; it is spans only. Grading (`score`,
`success`, `status`) is Exgentic's own recorded outcome from running each underlying benchmark's own
grader at collection time, not something a Kullback verifier could recompute from this dataset alone.
Reproducing any of it requires re-running the underlying benchmark (AppWorld, SWE-bench, BrowseCompPlus,
tau2) with its own harness, which each have their own environment and reference verdict (see
docs/trace-sources/appworld.md for AppWorld's).

## Licence

Not stated. No `license` field appears in the Hugging Face API metadata (`cardData.license` is absent) or
in the dataset card text for either `Exgentic/agent-llm-traces-v2` or its parent
`Exgentic/traces-v2` (checked via `https://huggingface.co/api/datasets/<id>`, both return no license key).
Unverified: whether a licence is stated anywhere off-Hub; none was found in this pass.

## How to download

`https://huggingface.co/api/datasets/Exgentic/agent-llm-traces-v2` lists 9 parquet shards at
`data/train/0000.parquet` through `data/train/0008.parquet`; each is fetchable directly with curl from
`https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2/resolve/main/data/train/<NNNN>.parquet`
(231 MB total, no auth token needed, public dataset). The `datasets-server` REST API
(`https://datasets-server.huggingface.co/{splits,size,rows}`) gives splits, size and single-row previews
without downloading the shards, though the `/statistics` and `/filter` endpoints for this dataset returned
server errors in this pass (`/statistics`: "I/O error: Permission denied"; `/filter`: "Parameter 'where'
contains errors or invalid symbols" on every clause tried, including the documented syntax), so per-column
aggregate counts had to be computed locally from the downloaded shards instead.

## Fit for Kullback

Gives several runs per task for AppWorld, tau2 airline and tau2 retail across five current models, which is
exactly the "more than one run per task" gap AppWorld and tau2 have on their own; the `tool_calling` harness
subset (2,943 sessions) is directly shaped like a Kullback Trace. Breaks: no environment or reference verdict
travels with the traces, so a Trace from here still depends on the underlying benchmark's own environment and
grader to become a Run with a Verdict; three of five harnesses (`claude_code`, `openai_solo`,
`smolagents_code`, 71% of rows) don't hold one-tool-call-per-turn shape and would need scaffold-specific
parsing to recover ToolCall boundaries; no stated licence. Several traces (sessions) per task, one trace per
session.

## Open questions

- No licence found anywhere in the Hub metadata for either this dataset or its parent; unverified whether
  one exists off-Hub.
- Whether the 100-task AppWorld sample (and the tau2 airline/retail samples) is the same task set repeated
  across all 15 (or 24) configurations was not confirmed; would need a task-level identifier pulled from
  span content, not a top-level column, which was not fetched in this pass to keep the download small.
- `datasets-server`'s `/statistics` and `/filter` endpoints errored for this dataset on every attempt in
  this pass; unclear if that is dataset-specific (parquet shard sizes triggering a server-side limit) or a
  general outage.
