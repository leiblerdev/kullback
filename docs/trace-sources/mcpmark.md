# MCPMark

## Link and owner

Trajectory log: https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log. Benchmark: MCPMark, over Notion, GitHub, Postgres, filesystem and Playwright MCP servers (per docs/benchmark-landscape.md, 2026-08-29 survey, reused here).

## What a row is

One trial: a task attempted by one model against one MCP service, recorded as a folder of three files, `meta.json` (task id, model, timestamps, status), `messages.json` (turn-by-turn exchanges including model thoughts and tool calls), and `execution.log` (execution-time logs). Folder naming follows a `run-1`, `run-2` pattern per task/model (read via WebFetch of the HF card, 2026-09-24).

## Provenance

Real live backends: a real Notion workspace, real GitHub repos, a real Postgres instance, a real filesystem, a real browser via Playwright, reset per task by template re-import (per docs/benchmark-landscape.md). Model-run, not human-run, but against genuinely real systems rather than simulations of them.

## Counts and size

127 tasks (per docs/benchmark-landscape.md). Exact trajectory-file count could not be read this pass: the Hugging Face dataset viewer throws an ArrowTypeError on the `content` column ("changed from object to array"), which blocks the row-count and preview display.

## Schema

`meta.json`: task id, model, timestamps, status. `messages.json`: turn-by-turn exchanges with model thoughts and tool calls. `execution.log`: execution-time text log. The exact field names inside a `messages.json` tool-call entry (arguments key, result key) were not confirmed this pass because of the viewer error above.

## One excerpt

Not obtained this pass; the dataset viewer's parsing error blocked a preview. Downloading one `run-1` folder directly (not through the viewer) is the way to get a real excerpt.

## Tool-call shape

Structured JSON tool calls with arguments inside `messages.json`; whether the tool's result is a sibling field in the same JSON object or has to be read from `execution.log` instead is exactly what the viewer error is hiding, and is the first thing to check before building on this corpus.

## Environment and reference verdict

Partly reusable: the environment is real, resettable "by template re-import," which is a genuine advantage over a synthetic seed file, but it is not offline. Running it means holding a Notion workspace, a GitHub token, a Docker Postgres instance and Playwright, which is infrastructure Kullback's current retail/airline builds do not need.

## Licence (quoted)

MIT (read from the "License" field on the Hugging Face dataset card via WebFetch, 2026-09-24). This differs from docs/benchmark-landscape.md's earlier note of "Apache 2.0"; the fresh, direct card read is treated as authoritative here and the discrepancy should be checked again before relying on it.

## How to download

Via the Hugging Face Hub API, the "Copy to bucket" option on the dataset page, or `hf download Jakumetsu/mcpmark-trajectory-log --repo-type dataset`.

## Fit for Kullback

Good size (127 tasks), a real license we can act on (MIT), multiple runs per task-model pair for the held-out check, and real backend systems for strong provenance. The open risk is entirely E1: whether tool call arguments and results both live in `messages.json` in a shape our ingest can read, or whether results have to be reconstructed from `execution.log` text.

## Open questions

- Direct download and inspection of one `run-1`/`run-2` pair for the same task, to confirm the tool-result field shape (the viewer error blocks this from the HF UI).
- Total trajectory count.
- Whether the licence field genuinely reads MIT on the underlying repo files, not only the card metadata, given the discrepancy with the earlier survey's "Apache 2.0" note.
- Whether MCPMark's own repo (org not confirmed this pass) publishes the reset templates needed to actually replay a task, separate from the trajectory log dataset.
