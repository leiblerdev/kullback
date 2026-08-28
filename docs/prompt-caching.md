# Prompt caching (2026-08-28)

Asked for in my words: "you also need to explore how to cache prompts as well please or else the cost would be too much." This note is what the provider layer does today, where the tokens actually go, and the changes in the order they pay off.

## What is already there

`shared/provider.py` marks cache points on every Anthropic request (`cache_system` on the system prompt, `cache_last_two` on the last two messages) and reads `cache_read_input_tokens` and `cache_creation_input_tokens` back. The OpenAI adapter reads `prompt_tokens_details.cached_tokens` and subtracts it from `input`, so `Usage.input` is the uncached input everywhere. `shared/budget.py` prices `cache_read` and `cache_write` per model (Anthropic read 0.1x, write 1.25x; OpenAI read 0.1x to 0.25x, no write charge) and totals them per stage in `budget.json`. So the wire side and the accounting side are done.

## Why it mostly does nothing yet

A provider only caches a prefix that is long enough: 1,024 tokens on OpenAI, and on Anthropic a per-model minimum that `research/40-runner-and-provider.md` found ranges from 512 to 4,096 tokens by model generation, so the harness needs a per-model table rather than one number. Correction from the same report: OpenAI charges cache writes at 1.25x on GPT-5.6 class models, so the `PRICES` comment that OpenAI has no cache-write charge holds only for the two OpenAI models priced today and a new entry needs its own `cache_write` price. Below that a cache point is ignored. The system prompts in the Builder are 45 to 90 tokens each (`compile_env._SYSTEM` 78, `mine.KIND_SYSTEM` 61, `mine.SCHEMA_SYSTEM` 45, `memory._JUDGE_SYSTEM` 90, `policy._CONTRACT` in the same range). The variable evidence, the recorded calls, the rule and the case, is in the user message. So in the Builder the cached prefix is the short system prompt and it never reaches the minimum; every call pays full input price. The candidate loop in the Runner is the one place the layout is right: `new_run_state` puts the Environment's system prompt first, the tools ride along, and the conversation grows behind them, so each turn reuses the previous turn's prefix.

The two judges share one prompt per Run (same `_system(use)`, same rendered state), so the second judge can hit what the first wrote, if the two are the same model and the call comes within the 5 minute window. The judge's tool loop grows the same way the candidate loop does and caches correctly.

## Where the tokens go in a build

Per build, roughly in order of spend: `compile_env.write_tool_body` (one call per tool, the recorded calls as evidence, a retry per failed gate with the failure appended), `policy` (one or two calls per rule sentence, the rule and a case each), `verifier` derivation and `memory` relevance checks (one call per lesson per build), the judges (two calls per disputed Run, each with a tool loop), and `intent` and `cluster` naming (one call per Task). Candidate runs are the Runner's cost and scale with the number of Candidates times Runs times turns.

## Changes, in the order they pay off

1. Stable part first, long enough to cache. In `compile_env` the system message should carry the schema (tables and columns), the tool list and the emit rules, which is the same for every tool in a build; the tool and its calls follow in the user message. In `policy` the system message should carry the compile contract plus the policy text itself, which is the same for every rule; the sentence and the case follow. Both prefixes then clear 1,024 tokens on any real customer and every call in the stage after the first reads them at 0.1x.
2. Retries append, never rewrite. `body_messages` today folds the gate failure into the same user message, so a retry is a different prefix and a miss. The retry should keep the original messages and add the failure as a new user turn after the model's previous reply. Same for `policy`'s rewrite call.
3. A request memo on disk. Key: sha256 of model id, messages, tools and config. Value: the reply record, under `workdir/model_cache/<hash>.json`. Served hits cost nothing and are counted in `budget.json` as `memo_hits`. This is what makes `--iterate` cheap: when one new trace changes one tool's evidence, the other tools' requests are byte identical and never go to the network. It also makes a crashed stage resume for free. Off by default for live candidate runs (a candidate's answer must be a fresh sample), on for every Builder stage and for the judges.
4. `prompt_cache_key` on OpenAI, one per build and stage, so the provider routes calls with the same prefix to the same cache. One line in the OpenAI adapter, one field in `ModelConfig`.
5. Hit rate in the report: per stage, `cache_read / (input + cache_read)` and the dollars the reads saved against list price. It goes next to the spend so a layout that misses shows up as a number, not a feeling.
6. Anthropic 1 hour TTL only where a stage's calls are more than 5 minutes apart with the same prefix (`{"type": "ephemeral", "ttl": "1h"}`, write at 2x instead of 1.25x). Not expected to be needed; measure first.

What not to do: cache across customers or across builds with a shared key (the prefix carries customer data; the memo is per workdir on purpose), and do not lower temperature or pin seeds to raise memo hits on candidate runs, which would turn a sample into a replay.

## Measuring it

The first live build on tau2 retail with one model records `cache_read`, `cache_write`, `input` and `memo_hits` per stage in `budget.json`. Before changes 1 and 2 the Builder's read share should be close to zero, which is the number to beat. After them the per-stage read share should sit above 0.7 for `compile_tools` and `compile_policy`, and a second `--iterate` build with one added trace should show most calls as memo hits.
