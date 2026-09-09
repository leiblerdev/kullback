# leibler

Leibler turns the traces a working agent already produced into an executable copy of the system it worked in, then grades other models on what they change in that copy. The graders are code, so they are cheap and have no opinions. [leibler.dev](https://leibler.dev)

[Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner behind it, under Apache-2.0. It rebuilds the world from traces, checks the rebuild by replaying them, derives a Verifier per Task from the recorded runs, and puts every artifact through a code gate no model may touch. The Environments below came out of it. Each one carries its own numbers and says what it cannot do yet.

Everything here is under active development. Environments are republished after each build round, and earlier rounds stay reachable by their tags. The next stage is to raise the trusted count on each Environment, then to generate Tasks synthetically over the rebuilt world.

## Environments

| Environment | Replay fidelity | Trusted Tasks | Round | Status |
| --- | --- | --- | --- | --- |
| [retail](https://huggingface.co/datasets/leibler/retail) | 95.1% | 122 of 205 | 1 | release |
| [airline](https://huggingface.co/datasets/leibler/airline) | 72.3% | 38 of 119 | 3 | preview |
| [telecom](https://huggingface.co/datasets/leibler/telecom) | 9.3% | 0 of 183 | 5 | preview |

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

