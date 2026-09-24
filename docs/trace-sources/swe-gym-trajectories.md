# SWE-Gym trajectories (OpenHands-Sampled-Trajectories)

## Link and owner

https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories, published under the SWE-Gym org. Distinct from the base task dataset `SWE-Gym/SWE-Gym` (which is task and solution patches, not multi-turn tool-call trajectories) and from two sibling trajectory sets on the same org, `SWE-Gym/OpenHands-SFT-Trajectories` and `SWE-Gym/OpenHands-Verifier-Trajectories`, neither deep-dived this pass.

## What a row is

One sampled OpenHands agent rollout against one SWE-Gym task instance: an `instance_id`, a `run_id`, a `resolved` boolean, a `messages` list (system prompt plus assistant/tool interactions), a `tools` list (tool definitions), and a `test_result` dictionary (evaluation metrics). (WebFetch of the HF card, 2026-09-24.)

## Provenance

Real GitHub issues and repositories (SWE-Gym follows SWE-bench's sourcing methodology), model-run resolutions by OpenHands agents.

## Counts and size

6,055 rows, 301 MB (WebFetch of the HF card, 2026-09-24). The base `SWE-Gym/SWE-Gym` task dataset has 2,438 instances (docs/benchmark-landscape.md).

## Schema

`instance_id` (string), `run_id` (string, 9 distinct values observed), `resolved` (boolean), `messages` (list, 7 to 101 items), `tools` (list, 3 items, tool definitions with descriptions), `test_result` (dictionary, evaluation metrics). (WebFetch of the HF card, 2026-09-24.)

## One excerpt

Not captured directly this pass. The card's own summary notes the data spans repositories such as `getmoto/moto`, `python/mypy` and `iterative/dvc`, with model outputs from `gpt-4o-2024-08-06` including failed resolution attempts (WebFetch of the HF card, 2026-09-24).

## Tool-call shape

Tool calls live inside `messages` (system prompt plus assistant/tool turns); the `tools` list documents the tool definitions available to the agent; results are captured via `test_result` at the trajectory level (pass/fail) rather than confirmed as a per-call result field, which is a real E1 gap relative to a corpus like SWE-bench Nebius that explicitly carries per-step observations.

## Environment and reference verdict

Partly reusable: SWE-Gym follows SWE-bench's own methodology, so per-instance Docker-style images are plausible, but this specific card does not confirm their availability directly, unlike the Nebius SWE-bench trajectory set, which does.

## Licence (quoted)

None stated on this card (checked directly via WebFetch and via `https://huggingface.co/api/datasets/SWE-Gym/OpenHands-Sampled-Trajectories`, `cardData.license` is `None`, 2026-09-24). The base `SWE-Gym/SWE-Gym` task dataset is MIT (docs/benchmark-landscape.md), but that licence is not confirmed to extend to this separate trajectory card. Per the eligibility test, "none stated" is a no for publishing until this is resolved.

## How to download

```
hf download SWE-Gym/OpenHands-Sampled-Trajectories --repo-type dataset
```

## Fit for Kullback

About 9 distinct `run_id` values per `instance_id` on average, which comfortably satisfies the held-out requirement, and a real-GitHub-issue domain close to SWE-bench's. The blocking issue is squarely licence: this specific trajectory card states no licence of its own, and it should not be treated as inheriting the base dataset's MIT licence without confirming that directly with the publisher.

## Open questions

- Whether the licence is confirmed by the SWE-Gym org (an issue or a direct question to the maintainers) to be MIT like the base task dataset, or something else.
- Whether SWE-Gym's per-instance Docker images are the same as SWE-bench's, or a separate build, and where they are published.
- Whether `test_result` is trajectory-level only, or whether a per-call result can be reconstructed from the raw `messages` content.
- How the sibling `OpenHands-SFT-Trajectories` and `OpenHands-Verifier-Trajectories` datasets compare on schema and licence, since one of them might turn out to be the better source.
