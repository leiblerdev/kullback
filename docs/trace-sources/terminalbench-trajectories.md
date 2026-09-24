# terminalbench-trajectories

## Link and owner

https://huggingface.co/datasets/yoonholee/terminalbench-trajectories, published by Hub user `yoonholee`
(not the Terminal-Bench project itself). Source benchmark: Terminal-Bench 2.0,
https://www.tbench.ai/leaderboard/terminal-bench/2.0. The README states plainly: "Trajectories were scraped
from tbench.ai using the publicly available leaderboard data" (`Data source` section), so this is a
third-party re-export of the official leaderboard's public trajectory data, not a first-party release from
the Terminal-Bench team.

## What a row is

One row is one trial: one agent scaffold plus one underlying model attempting one Terminal-Bench task once.
"Each row is one trial: an agent attempting a task, with the complete step-by-step trace of messages, tool
calls, and observations." (dataset card summary)

## Provenance

Model generated. 109 distinct agent/model combinations (26 different agent scaffolds, e.g. `claude-code`,
`terminus-2`, `codex`, `openhands`, `mini-swe-agent`, `Factory Droid`, `gemini-cli`, plus many single-combo
scaffolds; 49 distinct underlying LLMs) were each run against the 89 Terminal-Bench 2.0 tasks, typically 5
trials per (task, agent) pair (README, `Dataset summary` table: "Trials per (task, agent): typically 5").
The tasks themselves are Terminal-Bench's own, real terminal/shell problems (installing software, writing
interpreters, cryptanalysis, video processing, and similar), not written for this dataset.

## Counts and size

52,104 total trajectories (rows), 89 tasks, 109 agent/model combinations, 26 scaffolds, 49 underlying
models, overall pass rate 39.6%, median 21 steps per trajectory (mean 47.1), 34,462 trials have non-null
`steps` content (the rest have a null `steps` column, meaning no step-level content was captured for that
trial even though its summary row exists). Download size 220,993,900 bytes (211 MB) across 2 parquet
shards; decoded size 939,348,659 bytes. Licence: apache-2.0. (dataset README `Dataset summary` table and
front matter; sizes cross-checked against `datasets-server/size`, 2026-09-24, which reports 52,104 rows,
220,993,900 bytes on disk, matching the README exactly.)

## Runs per task

Typically 5 trials per (task, agent) combination, stated directly in the README's summary table. With 109
agent/model combinations, a given task can appear in well over 100 trials total across all combinations,
though "trials per (task, agent)" is the more precise unit here since not every agent ran every task, and
not every trial has step content.

## Schema

Row columns: `task_name`, `agent`, `model`, `reward` (1/0), `duration_seconds`, `input_tokens`,
`output_tokens`, `cache_tokens`, `cost_cents` (the last four are null for some agents), `trial_name`,
`trial_id`, `started_at`, `ended_at`, `steps` (a JSON-encoded string, not a native nested column). Each
decoded step object: `src` (`"user"`, `"agent"` or `"system"`), `msg` (text), `tools` (a list of
`{fn, cmd}` objects, or `null`), `obs` (tool output, truncated to 5,000 characters, or `null`).

## One excerpt

Task `bn-fit-modify`, agent `terminus-2`, model `claude-opus-4-5-20251101@anthropic` (row offset 1001, one
of 22 steps), a step with a real tool call:

```json
{"src": "agent",
 "msg": "Analysis: I'm starting fresh in the /app directory. I need to analyze the dataset to recover a Bayesian Network DAG structure...",
 "tools": [{"fn": "bash_command", "cmd": "head -20 /app/bn_sample_10k.csv"},
           {"fn": "bash_command", "cmd": "wc -l /app/bn_sample_10k.csv"}],
 "obs": "$32"}
```

Note the `obs` value here is the literal string `"$32"`, not the actual command output; the same
placeholder-token pattern (`"$31"`, `"$32"`, ...) showed up in `msg` and `obs` fields on several rows
sampled in this pass, including ones with no truncation flagged by the `datasets-server` API. This looks
like an artifact carried over from however `tbench.ai`'s own leaderboard stores or templates trajectory
text, not a `datasets-server` display truncation (the API's `truncated_cells` field was empty on every row
checked). Unconfirmed whether it affects all rows or only some.

## Tool-call shape against our Trace record

Each step's `tools` list is close to `ToolCall.name`/`args` (`fn` maps to `name`, `cmd` maps to a single
positional argument rather than a `dict` of named `args`, so ingest would need to wrap it), and `obs` maps
to `ToolCall.result`, but one step can carry more than one tool call (see the excerpt above, two
`bash_command` calls in one step) while our `ToolCall` record is one call at a time with `result` singular,
so a step with multiple `tools` entries would need to become multiple `ToolCall` records sharing one
`Turn`. The `"$N"` placeholder issue above is a real risk to ingest: any `ToolCall.result` pulled naively
from this dataset could be a placeholder token instead of real output.

## Environment and reference verdict

Not included. Terminal-Bench 2.0 tasks run in Docker containers with per-task setup and a test script that
returns pass/fail; this dataset carries only the `reward` outcome column already computed by whichever
harness produced the leaderboard entry, not the container image, the task's test script, or any
replayable state. Reproducing verification requires Terminal-Bench's own repository and Docker setup,
neither of which travels with this dataset.

## Licence

Apache 2.0, stated plainly in the README's own `License` section ("Apache 2.0.") and in the Hub API's
`cardData.license` field (`apache-2.0`).

## How to download

Two parquet shards under `data/train-00000-of-00002.parquet` and `data/train-00001-of-00002.parquet`,
fetchable directly:
`https://huggingface.co/datasets/yoonholee/terminalbench-trajectories/resolve/main/data/train-00000-of-00002.parquet`
(and `-00001-of-00002`). Also readable row-by-row via
`https://datasets-server.huggingface.co/rows?dataset=yoonholee%2Fterminalbench-trajectories&config=default&split=train&offset=<N>&length=<M>`
without downloading the shards, which is how the excerpt above was pulled.

## Fit for Kullback

Gives a real multi-agent, multi-model comparison over one task set with several runs per (task, agent),
useful as a cross-scaffold sample of what different agent harnesses do on the same terminal tasks. Breaks:
a third-party scrape, not first-party, so provenance below "scraped from the public leaderboard" is opaque
(no per-trial link back to a raw session log); the `"$N"` placeholder artifact needs resolving or filtering
before any `ToolCall.result` from this dataset can be trusted; `steps` is null for about a third of rows
(17,642 of 52,104); no environment or task-verification harness travels with it, matching Terminal-Bench's
own Docker-per-task shape, not our seeded-state shape. Several trials (traces) per task.

## Open questions

- The `"$N"` placeholder pattern in `msg`/`obs` fields: whether it is a data artifact affecting a subset of
  rows, or something resolvable by fetching from `tbench.ai` directly, is unverified.
- No first-party confirmation from the Terminal-Bench team that this scrape is complete or kept in sync
  with the official leaderboard; it is a third-party publication.
- Why roughly a third of rows have a null `steps` column was not investigated.
