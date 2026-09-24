# leibler

Leibler takes the traces a working agent already left behind and rebuilds the system it worked in as something you can run. Other models are then graded on what they change in that copy. The graders are plain code, so they are cheap to run and never have an opinion. [leibler.dev](https://leibler.dev)

[Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner that does this, under Apache-2.0. It rebuilds the world from the traces and checks the rebuild by replaying them. From the recorded runs it derives one Verifier per Task, and every artifact has to pass a code gate that no model can touch. Each Environment below came out of Kullback, and each card lists its numbers and what it cannot do yet.

All of this is still being built. We republish an Environment after each build, and older builds stay reachable by their tags. Next we want more trusted Tasks on every Environment, and after that, synthetic Tasks generated over the rebuilt world.

## Environments

Smoke 8, 2026-09-23, Builder `openai/gpt-6-sol`, tag `build-20260923`. Fidelity over Tasks counts the Tasks whose Reference replay was confirmed, out of all Tasks. Call fidelity counts every recorded call.

| Environment | Fidelity over Tasks | Over Runs | Call fidelity | References confirmed | Verifiers | Trusted Tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [retail](https://huggingface.co/datasets/leibler/retail) | 100.0% (205 of 205) | 100.0% (456 of 456) | 100.00% over 3220 calls | 205 | 205 | 133 of 205 | release |
| [airline](https://huggingface.co/datasets/leibler/airline) | 79.8% (95 of 119) | 78.5% (157 of 200) | 96.03% over 1513 calls | 95 | 95 | 53 of 119 | preview |
| [telecom](https://huggingface.co/datasets/leibler/telecom) | 9.3% (17 of 183) | 18.5% (27 of 146) | not on its card | 7 | 14 | 0 of 183 | preview |

Telecom was not rebuilt in smoke 8. Its row is still round 5, built 2026-09-09. No round closed in smoke 8, so the retail and airline counts come from the workdir status, as the "Counts from" row on each card says.

The build before this one, smoke 7 on the same day and the same Builder, ended with 14 trusted retail Tasks at 98.5% (202 of 205) and 22 trusted airline Tasks at 84.0% (100 of 119). Airline replays fewer Tasks than it did then, and its booking tool matches only 3 of its 29 recorded calls. It has more than twice as many trusted Tasks, though.

A release replays at least 90% of its Tasks, and a preview falls below that. We publish previews anyway, numbers included on the card, so the work stays in the open while it gets better.

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
- Licence: Apache-2.0 for the harness and every package here. Each source corpus keeps its own licence, which the Environment's card names.
