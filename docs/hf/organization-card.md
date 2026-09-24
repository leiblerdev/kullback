# leibler

Leibler turns the traces a working agent already produced into an executable copy of the system it worked in, then grades other models on what they change in that copy. The graders are code, so they are cheap and have no opinions. [leibler.dev](https://leibler.dev)

[Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner behind it, under Apache-2.0. It rebuilds the world from traces, checks the rebuild by replaying them, derives a Verifier per Task from the recorded runs, and puts every artifact through a code gate no model may touch. The Environments below came out of it. Each one carries its own numbers and says what it cannot do yet.

Everything here is under active development. Environments are republished after each build round, and earlier rounds stay reachable by their tags. The next stage is to raise the trusted count on each Environment, then to generate Tasks synthetically over the rebuilt world.

## Environments

Smoke 7, 2026-09-23, Builder `openai/gpt-6-sol`, tag `build-20260923`. Fidelity over Tasks is the card's number:
Tasks whose Reference replays confirmed, over all Tasks. Call fidelity is over every recorded call.

| Environment | Fidelity over Tasks | Over Runs | Call fidelity | References confirmed | Verifiers | Trusted Tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [retail](https://huggingface.co/datasets/leibler/retail) | 98.5% (202 of 205) | 96.7% | 99.53% over 3220 calls | 202 | 20 | 14 | release |
| [airline](https://huggingface.co/datasets/leibler/airline) | 84.0% (100 of 119) | 84.5% | 95.64% over 1513 calls | 100 | 30 | 22 | preview |
| [telecom](https://huggingface.co/datasets/leibler/telecom) | telecom: not in this round | | | | | | preview |

Counts come from the workdir status when no round closed, as each card's "Counts from" row says.

Previous numbers (replay fidelity, trusted, round): retail 95.1%, 122 of 205, round 1; airline 72.3%, 38 of 119, round 3;
telecom 9.3%, 0 of 183, round 5.

A release replays at least 90% of its Tasks. A preview is below that bar. It is published anyway, with its numbers on its card, so the work stays visible while it improves.

## Fetch and run

```bash
uv pip install git+https://github.com/leiblerdev/kullback.git
uv run kullback fetch leibler/<environment> --out env
uv run kullback run --workdir env --task <task id> --model provider/model
uv run kullback verdict --workdir env
uv run kullback report --workdir env
```

## Links

- Harness: https://github.com/leiblerdev/kullback
- Site: https://leibler.dev
- Licence: Apache-2.0 for the harness and every package here. Each source corpus keeps its own licence, named on the Environment's card.

