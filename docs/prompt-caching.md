# Prompt caching

How the provider layer caches prompts and how to check that it works. It matters because, in the founder's
words, "you also need to explore how to cache prompts as well please or else the cost would be too much."

## Check a build

```bash
uv run kullback budget cache --workdir work
uv run kullback budget cache --workdir work --json
```

Read only. Per stage it prints the calls, the share of prompt tokens read from the cache, the writes (cold, on a
new prefix, after a gap past the stage's TTL), memo hits, and the ten largest uncached calls. Every priced call is
a `model_call` line in `events.jsonl` with its cache fields and a request fingerprint
(`kullback.ai.cache.fingerprint`: `prefix_hash`, `history_hash`, `parent_hash`, `n_messages`).

## What the code does

Every provider caches a prefix that is byte identical from one call to the next, in the order tools, system,
messages. OpenAI-shaped endpoints cache it on their own past 1,024 tokens. Anthropic, on its own API and on
Bedrock, caches only up to a block marked with `cache_control`.

### Cache points

`AnthropicModel.build_body` (`kullback/ai/provider.py`) marks the last tool (`cache_tools`), the last system block
(`cache_system`) and the last two messages (`cache_last_two`). Four marks is the API's limit, and `build_body`
refuses a fifth. No mark goes on the top level of the body: the legacy Bedrock stack refuses it.

### Choosing the TTL

`ModelConfig.cache_ttl` is `"5m"` (the default, sent as a bare `{"type": "ephemeral"}`) or `"1h"`.
`STAGE_CACHE_TTL` in `kullback/ai/cache.py` gives `builder_session` one-hour marks, because the Builder session
waits past five minutes on long prompts; every other stage writes five-minute marks. The OpenAI adapters send
no TTL.

### Usage and economics

- `Usage.input` (`kullback/ai/usage.py`) is the input neither read from nor written to the cache, on every
  adapter. `Usage.cache_write_1h` is the one-hour part of `cache_write`.
- `kullback/runner/budget.py` prices a one-hour write at the row's `cache_write_1h`, else twice the input price.
  A new price row needs its own `cache_write`.

### Stable prefixes

Tool lists render sorted by name, JSON in a prompt is dumped with sorted keys, and facts that differ per session
or per Run sit in the user message, not the system prompt.

## The rules, numbered

Code and tests cite these by number.

1. Stable part first, long enough to cache: what is the same for every call in a stage (schema, tool list,
   contract, policy text) goes in the system message; the case goes in the user message.
2. Retries append, never rewrite: a retry keeps the earlier messages and adds the failure as a new user turn.
3. The memo: `MemoModel` stores replies under `<workdir>/model_cache/`, keyed by the request. It wraps the stages
   `kullback/cli.py` wraps (`_wrapped_model`) and the readers step, never the judges, the agent sessions or a
   Candidate. A hit still counts as a call and adds to `memo_hits`.
4. `prompt_cache_key` on OpenAI-shaped endpoints: set per build and stage for the wrapped stages, and per session
   by the agent harness. The runner stage sets none; its prefix is stable without one.

## What not to do

Do not cache across customers or across builds with a shared key: the prefix carries customer data, so the memo
is per workdir. Do not lower temperature or pin seeds to raise memo hits on Candidate Runs, which would turn a
sample into a replay.
