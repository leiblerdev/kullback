# OSWorld trajectories

## Link and owner

https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs (primary candidate), published by the xlang-ai team behind OSWorld. Also https://huggingface.co/datasets/xlangai/osworld2.0-trajectory. Environment: https://github.com/xlang-ai/OSWorld.

## What a row is

One episode: screenshots and an action sequence, model reasoning traces, and a task-completion result, for one model's attempt at one OSWorld task on a real Ubuntu desktop.

## Provenance

Real desktop applications on a real (virtualized) Ubuntu environment; tasks are human-authored, attempts are model-run.

## Counts and size

1,000+ episodes, 15+ models, roughly 500 GB total.

## Schema

Not independently re-derived this pass; reported shape is screenshots plus action sequences plus model reasoning traces plus task-completion results, per the card.

## One excerpt

Not captured this pass.

## Tool-call shape

Actions (clicks, keystrokes, app-level operations) paired with observations (screenshots, and reportedly an accessibility tree in some OSWorld variants) and a final completion result; not a clean relational state, closer to GUI-agent trajectories than to tau2-style structured tool calls.

## Environment and reference verdict

Partly reusable, and this is the main open risk: the reference environment is a per-task VM or container snapshot, and whether these VM/container images are freely, publicly downloadable and runnable outside a paid cloud account (as opposed to only the trajectory recordings being public) was not confirmed this pass. `github.com/xlang-ai/OSWorld` is the place to check this directly.

## Licence (quoted)

MIT, quoted from the `xlangai/ubuntu_osworld_verified_trajs` card.

## How to download

```
hf download xlangai/ubuntu_osworld_verified_trajs --repo-type dataset
```
Given the ~500 GB size, a partial download (specific episodes or models) is likely the practical starting point rather than a full snapshot.

## Fit for Kullback

Large, clearly MIT-licensed, multiple models times step-limits attempting the same task (a plausible route to the held-out requirement, though "multiple models" is a different kind of repeat than "multiple independent recordings of the same policy"). The state is pixels plus an accessibility tree rather than clean relational records, which is a weaker E6 fit than tau2, AppWorld, or MCP-server-shaped benchmarks. Rebuilding a "tool body" here would mean rebuilding desktop-application behavior, not a database.

## Open questions

- Whether OSWorld's per-task VM or container images are freely downloadable and runnable without a paid cloud account; this is the single biggest open item before treating this as more than "eligible with work."
- Whether "multiple models attempt the same task" trajectories are similar enough in setup to count as independent Runs for the held-out check, or whether they differ enough (different step limits, different starting conditions) to not qualify.
- Exact row schema and whether the accessibility tree (structured) or only screenshots (pixels) are captured per step, since that changes the E6 answer significantly.
- How `xlangai/osworld2.0-trajectory` compares to `ubuntu_osworld_verified_trajs` on licence, size and schema; only the latter was checked this pass.
