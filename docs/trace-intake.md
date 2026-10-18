# Trace intake

Which trace formats Kullback reads, and what to ask a customer before their traces arrive. Every build starts
here, and the answers decide whether a build can work at all.

## Formats read today

Every format enters through one adapter in `kullback/builder/sources/`, and each adapter maps its format to
Traces. `kullback ingest` detects the format itself.

| Format | Adapter | One recording is |
|---|---|---|
| tau2 export | `tau2_native` | one entry of the simulations list |
| terminus-2 rows | `terminus_2` | one `conversations` turn list; shell batches are JSON in assistant text |
| OpenTelemetry GenAI spans | `otel_genai` (D318) | one `traceId`: `gen_ai.*` messages, events and `execute_tool` spans |
| Claude Code JSONL | `claude_code_jsonl` (D318) | one `sessionId`, turns ordered by `parentUuid` |

```bash
uv run kullback ingest path/to/traces.json --workdir work
```

A file is stored byte for byte under `raw/`, and the intake ruling goes to `intake_ruling.json`.

## Questions for a customer

Ask these before the traces arrive. One example file answers most of them.

| # | Question | Why it matters |
|---|---|---|
| 1 | Whose agent made the traces, and in which domain? | The domain decides which tools and policies to expect. |
| 2 | Which format: raw model API logs, a tracing export, or the framework's own log? | It picks the adapter. |
| 3 | Are tool results kept in full or truncated? | Truncated reads block rebuilding the Starting state. |
| 4 | Is the tool list sent to the model in the log? | It gives contracts for tools no trace calls. |
| 5 | Are the turns of one conversation linked by a session id? | Runs cannot be assembled without it. |
| 6 | Do failed tool calls appear with their error text? | Without one, the error encoding is a guess. |
| 7 | Is the system prompt present per call? | Policy lines compile to checks from it. |
| 8 | How many conversations, over what period, with how many tools? | It sets the Task floor. |
| 9 | Are outcome signals recorded (CSAT, escalation, reopen, refund reversal)? | They help confirm References. |
| 10 | Can a domain expert mark 20 to 50 conversations pass or fail? | A second pair of eyes on the Verifiers. |
| 11 | Are policy or knowledge documents available beyond the system prompt? | They are the next rung of evidence (ADR-0006). |
| 12 | Where do the traces live: the gitignored `data/`, a folder outside the repo, or a bucket? | Customer data never enters git. |
| 13 | What may leave the customer's boundary, and for how long? | Retention rules (ADR-0002). |
