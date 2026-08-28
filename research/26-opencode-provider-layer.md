# R26. OpenCode's provider adapter layer, and what `provider.py` takes from it

Date: 2026-08-27. Source: shallow clone of github.com/sst/opencode read by a subagent (Sonnet); file paths and line numbers from that clone. Requested in D69 ("see how open code does it").

## What OpenCode does

Files: `packages/opencode/src/provider/provider.ts` (2,067 lines), `transform.ts` (1,856), `auth.ts` (229), `error.ts` (195); plus `packages/core/src/models-dev.ts` (267), `session/retry.ts` (210), usage code in `session/session.ts`. About 5,000 to 6,000 lines for the concern, not counting the Vercel AI SDK packages it delegates to.

1. **Model identity.** `(providerID, modelID)` joined as `provider/model` (`provider.ts:2053-2059`). Each model carries `api: { id, url, npm }`: the wire-level id, base URL template, SDK package. Wire id and own id can differ.
2. **Registry is data, not code.** Providers and models come from models.dev JSON, fetched, cached on disk with a 5-minute TTL and a 60-minute background refresh, with a bundled snapshot fallback. Adding a provider is adding data; code changes only for providers with runtime quirks (`custom()` map, `provider.ts:174-999`).
3. **AI SDK does the wire work.** Streaming, tool plumbing, message format, per-provider HTTP. OpenCode wraps the model with a `transformParams` middleware (`llm.ts:325-338`) and does the request-shaping itself.
4. **transform.ts quirks** (concrete): strip unpaired UTF-16 surrogates; drop empty-content messages and empty reasoning parts for Anthropic and Bedrock (rejected upstream); tool-call ids restricted to `[a-zA-Z0-9_-]` for Claude, 9 alphanumerics for Mistral; DeepSeek needs a `reasoning` field on every assistant message; prompt-cache tags on the first 2 system and last 2 non-system messages (`applyCaching`, 359-408); unsupported image or file parts replaced by an explanatory text part; `providerOptions` key remapping per SDK package; Responses API `itemId` stripping when not stored; per-family temperature and top-p overrides; about 800 lines of reasoning-effort tables gated by model release dates; `maxOutputTokens = min(model limit, 32,000)`; OpenAI strict JSON-schema sanitizer for tool schemas (about 130 lines).
5. **Auth.** `~/.local/share/opencode/auth.json` (mode 0600) holding api key, oauth or well-known entries; `OPENCODE_AUTH_CONTENT` env override for CI; precedence plugin, config, env vars named by the provider record, stored auth, provider-specific loaders. Custom OpenAI-compatible endpoint is config only: `provider.<id>.options.baseURL` with `${VAR}` substitution, defaulting to the `@ai-sdk/openai-compatible` package (`provider.ts:1273, 1499, 1754-1773`).
6. **Cost.** Prices per 1M tokens from models.dev (`input, output, cache_read, cache_write`, optional context-size tiers). `Session.getUsage` (`session.ts:338-410`) runs on every step: normalizes counts, finds cache-write tokens under provider-specific metadata paths, subtracts cached tokens from input, picks the tier, computes the dot product with decimal arithmetic, reasoning tokens at the output rate. GitHub Copilot's billed amount is used directly when present.
7. **Retry.** `retry.ts`: never retry context overflow; 5xx always retryable; regex list over message text for rate-limit, overloaded, network and timeout phrases; honor `retry-after-ms` and `retry-after` headers (seconds or HTTP date); else `2000ms * 2^(attempt-1)` with 25% jitter capped at 30s; 5 retries. Separate header timeout (300s default) and SSE chunk timeout around `fetch` (`provider.ts:37-92, 1793-1824`).

## What `provider.py` takes (D69)

Copy:
- `provider/model` identity with a separate wire id and base URL per model.
- A static price table per model (input, output, cache read, cache write per 1M tokens); usage recorded on every call with cached tokens subtracted from input; cost as the dot product. Lives in `budget.py`, the table is data next to it.
- Retry classification: 5xx and network errors retryable regardless of SDK flag, `Retry-After` honored, exponential backoff with jitter, hard cap of 5, context overflow never retried.
- Three defensive normalizations that apply to our three targets: drop empty-content messages and empty reasoning parts before Anthropic; tool-call ids restricted to `[a-zA-Z0-9_-]`; strip unpaired surrogates from text.
- Cache-point placement (first system messages, last two non-system) when Anthropic prompt caching is on; the Runner's tool descriptions and system prompt are the same for every Run, so the cache pays.
- Base URL from config with `${VAR}` substitution, so a local vLLM or Ollama endpoint is a config line.

Do not copy: the live registry (models.dev fetch, TTL, locks), runtime package installation, the reasoning-effort tables (three explicit branches instead: Anthropic `thinking.budget_tokens`, OpenAI `reasoning_effort`, nothing for local), the OAuth plugin framework, the Effect-TS service graph, variants and catalog gating, the experimental native runtime.

Estimated size: 250 to 400 lines for `provider.py` including three adapters and retry; the price table and usage arithmetic sit in `budget.py`.

## Where this disagrees with the design so far

- Harness-design section 4 listed "a provider abstraction beyond `Model.query`" as deliberately absent. D69 keeps `Model.query` as the only interface but makes the adapters a named module; the absent-list entry was removed.
- OpenCode's usage accounting is per step inside the streaming loop. Ours records per call on the event (D65, `budget.py`), which is the same information at a coarser grain; if we ever stream, the per-step shape is the one to take.
- OpenCode clamps output at 32,000 tokens globally. We have no such clamp; the Runner should use the customer's recorded `max_tokens` when the trace has it and the model's limit otherwise, since a clamp changes Candidate behavior against the recorded agent.

## Coverage gaps

The subagent did not read `session/llm/native-runtime.ts`, `session/llm/ai-sdk.ts`, `session/llm/request.ts`, the models.dev schema, or the OpenAI JSON-schema sanitizer body line by line; their roles were inferred from names and call sites. The `@opencode-ai/llm` package was not inspected.
