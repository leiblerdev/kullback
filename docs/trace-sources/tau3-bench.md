# tau3-bench (tau2-bench repo, tag v1.0.1)

## Link and owner

https://github.com/sierra-research/tau2-bench, tag `v1.0.1` (released July 2026 per the repo, read 2026-09-24). Same owner and repo as tau2-bench, the corpus Kullback already builds retail and airline from. This is not a separate download; it is the next version of the tool we already run.

## What a row is

There is no published row yet. Running the harness locally would write simulations to `data/simulations/`, in the same shape tau2's own runs use.

## Provenance

Same lineage as tau2: human-written Tasks and policies, an LLM-simulated user, an LLM agent. Adds a banking domain (tau-Knowledge, "Knowledge-retrieval-based customer service domain with configurable RAG pipelines") and a Voice Full-Duplex domain ("end-to-end voice evaluation with realtime providers"), per the repo README (read 2026-09-24).

## Counts and size

75+ Task fixes over the tau2 domains (retail, airline, telecom), per docs/benchmark-landscape.md. Task counts for the new banking and voice domains were not stated on the page fetched this pass.

## Schema

Not applicable yet; no published trajectory to inspect. Expected to match tau2's simulation output shape, since it is the same codebase.

## One excerpt

None available; nothing is published.

## Tool-call shape

Same as tau2: JSON tool calls against an in-memory seed db (or, for tau-Knowledge, a RAG pipeline over a policy corpus), tool results as structured data.

## Environment and reference verdict

Installable and runnable: `uv` (not `pip install -e .`, that changed since tau2), requires Python `>=3.12, <3.14`. Needs configured LLM provider credentials (via LiteLLM) to run the simulated user and agent, and real-time audio provider credentials for the voice domain, so running it is not "offline" in the no-network sense, even though the seed state and tools are local. "Results produced with tau2-bench < 1.0.1 are not comparable with >= 1.0.1" per the repo's own upgrade note, which also means our existing tau2 export cannot simply be re-tagged as tau3.

## Licence (quoted)

MIT, per the repo's licence badge (read 2026-09-24; the LICENSE file text itself was not fetched).

## How to download

```
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench && git checkout v1.0.1
uv sync
```
Producing trajectories then means running the evaluation harness ourselves, with LLM API keys set, and reading the result from `data/simulations/`.

## Fit for Kullback

The most direct "next reference" available: same MIT licence, same lineage as the corpus already in production, and it is the exact benchmark docs/benchmark-landscape.md's open question 5 names ("which of the three offline traces-and-env references... comes next"). The work here is not integrating an unfamiliar environment, it is running a newer version of the tool we already operate and treating our own output the way Sierra's own raw export was treated for tau2.

## Open questions

- Exact Task counts for the banking and voice domains.
- Whether `grader/` and `reference_verdicts` (docs/benchmark-landscape.md section 4) need to be regenerated from tau3 rather than reused from tau2, given the 75+ task fixes changed some rewards.
- Whether running the voice domain is in scope at all, since it needs real-time audio provider credentials beyond a text LLM key.
- Cost and count of simulations we would need to run per Task to get 2+ trials for the held-out check.
