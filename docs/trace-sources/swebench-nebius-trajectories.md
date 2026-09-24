# SWE-bench trajectories (Nebius)

## Link and owner

https://huggingface.co/datasets/nebius/SWE-agent-trajectories, published by Nebius. Built on SWE-bench, whose task instances are real GitHub issues from real open-source Python repositories.

## What a row is

One trajectory: an agent's reasoning and actions paired with environment observations and test execution logs, for one attempt at one SWE-bench instance.

## Provenance

Real GitHub issues and real repositories (SWE-bench's own sourcing), with model-run resolutions layered on top. Not human-solved; the task itself is real, the attempt is not.

## Counts and size

80,036 trajectories.

## Schema

Not independently re-derived this pass; carried from the research pass done before the session restart. Reported fields: agent reasoning and actions, environment observations, test execution logs, paired per step.

## One excerpt

Not captured this pass; pulling one row via the Hugging Face datasets-server rows API is the next step before building.

## Tool-call shape

Structured actions against a real repository checkout (file edits, shell commands, test runs), with results including test execution output, not just the model's free text.

## Environment and reference verdict

Reusable: SWE-bench's own per-instance Docker images are separately, publicly available and are the standard way to rerun any instance (github.com/princeton-nlp/SWE-bench, the `swebench` PyPI package). This is a genuine advantage over most other candidates in this survey: the reference environment is not just theoretically reproducible, it is the field's standard tool for doing so.

## Licence (quoted)

CC-BY-4.0, with a clause that "users must respect the license of each specific repository" the instance is based on (since the underlying code itself carries its own repo licence), plus a Llama 3.1 licence clause that applies where trajectories used Llama-family model outputs. (Facts from the research pass done before the session restart; not independently re-fetched this pass, flagged as needing a direct re-check of the exact wording on the HF card before relying on the quote.)

## How to download

```
hf download nebius/SWE-agent-trajectories --repo-type dataset
```

## Fit for Kullback

Large (80,036 trajectories), multiple models attempt the same instance (several trajectories per SWE-bench instance, satisfying the held-out requirement), a real and standard reference environment already exists independent of this trajectory set, and the licence is a known open one (CC-BY-4.0) rather than "none stated." This is one of the strongest candidates in the whole survey precisely because the environment problem SWE-bench already solved (per-instance Docker images) removes the biggest risk other trajectory-only candidates carry.

## Open questions

- Exact wording of the CC-BY-4.0 clause and the Llama 3.1 clause, re-checked directly against the HF card rather than carried from a prior pass.
- Whether multiple trajectories against the same `instance_id` come from genuinely independent attempts (different models, different seeds) or share enough setup to not count as independent Runs for the held-out check.
- Row schema, confirmed directly rather than carried forward, including the exact field names for tool call arguments and results.

## Sibling dataset found this pass

`nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0, read via a cached README pulled before the session restart) is a related Nebius release worth comparing against directly: 1,823 repositories, 3,792 real-world issues resolved, 67,074 total trajectories, 32,161 successful, collected with Qwen3-Coder-480B-A35B-Instruct on OpenHands v0.54.0. Its own README quotes a comparison table against `SWE-bench/SWE-smith-trajectories` (49,897 trajectories, synthetic issues) and `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` (65,994 trajectories, synthetic issues), both weaker on provenance than the two real-world Nebius/SWE-Gym releases. A single, cleaner CC-BY-4.0 licence (versus this dataset's combined CC-BY-4.0-plus-per-repo-plus-Llama-3.1 clause) may make `SWE-rebench-openhands-trajectories` the simpler first pick between the two.
