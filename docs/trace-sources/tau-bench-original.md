# tau-bench (original)

## Link and owner

https://github.com/sierra-research/tau-bench, published by Sierra Research. This is the predecessor of tau2-bench, the corpus Kullback's retail and airline Environments already build from.

## What a row is

One historical trajectory: a full multi-turn conversation between a simulated user and an agent, with the agent's tool calls (arguments) and their results, over one Task. Trajectories are stored under `./historical_trajectories` in the repo (per the repo's own documentation, read 2026-09-24).

## Provenance

Human-written Tasks (a goal plus a persona) played out by an LLM-simulated user against an LLM agent. Not real customers; the same provenance shape as tau2-bench, which Kullback already treats as usable.

## Counts and size

Per docs/benchmark-landscape.md (2026-08-29 survey, reused here): retail 115 Tasks, airline 50 Tasks. Exact trajectory-file count not reconfirmed this pass.

## Schema

Not independently re-derived this pass; the repo's own documentation states trajectory records include tool calls with arguments, tool results, user turns and agent reasoning (read via WebFetch of the repo page, 2026-09-24). This is the same shape tau2-bench trajectories carry, which our ingest already parses.

## One excerpt

Not captured this pass (the repo page did not surface one on a single fetch). Given tau2-bench is a direct descendant of tau-bench with the same trajectory shape, our own `traces/` output from a tau2 build is a close proxy; a direct excerpt from `historical_trajectories/` should be pulled before building.

## Tool-call shape

Same family as tau2-bench: JSON tool calls against an in-memory seed database, tool results returned as structured data, not free text.

## Environment and reference verdict

Runnable: `pip install -e .` installs the package, and the leaderboard reports Pass^1 through Pass^4 scores, which implies the harness re-runs a Task several times per model. Several recorded trials per Task are therefore plausible, though not independently confirmed by row count this pass.

## Licence (quoted)

MIT. The repo's README badge and footer point to `./LICENSE` (read via WebFetch, 2026-09-24; the file itself was not fetched directly).

## How to download

```
git clone https://github.com/sierra-research/tau-bench.git
```
Trajectories are in `historical_trajectories/` once cloned. No account or token required.

## Fit for Kullback

Closest possible second reference to the corpus we already build from: same seed-db-in-memory shape, same MIT licence, same simulated-user provenance. docs/benchmark-landscape.md already flags it as "the weakest test of generalization" precisely because a Builder tuned on tau2 would be expected to pass it. Useful as a cheap sanity check, not a strong new stress test.

## Open questions

- Exact trajectory-file count and whether any Task has fewer than 2 trajectories (needed for the held-out check).
- Whether the trajectory JSON schema differs at all from tau2's, which would tell us whether the existing ingest parser needs changes.
- Whether the seed database ships inside the pip package or needs a separate download.
