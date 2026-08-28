# SDK Wrapper / Log-Drain

The capture layer. Per ADR-0001: log-drain + fail-open SDK wrapper, no proxy.

## What it captures (minimum for the first report)

- Prompt / input text
- Model + provider
- Token counts (input/output) + cost
- Latency
- (bonus) tool names, agent metadata, Run/tree context

## Fail-open rule (hard)

Logging is fire-and-forget. If logging fails, the customer's call still succeeds, unblocked and unchanged.

## First cut

- **Log-drain:** accept a customer's existing traces (Langfuse/Helicone export, JSONL, CSV).
- **SDK wrapper:** a thin client that wraps the OpenAI SDK, logs as a side effect, forwards to the real provider.

## Open questions (resolve when building)

- Which SDK first? (OpenAI? Anthropic? LangChain?)
- Schema for Trace/Run.
- Where traces are stored.
