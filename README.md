<div align="center">
  <h1>Kullback</h1>
  <p><strong>Rebuild an executable environment from your agent's traces, then grade and train models on it.</strong></p>
  <p>
    <a href="https://huggingface.co/leibler"><strong>Environments on Hugging Face</strong></a> ·
    <a href="#quick-start"><strong>Quick start</strong></a> ·
    <a href="docs/architecture.md"><strong>Architecture</strong></a>
  </p>

  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache License, Version 2.0"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-%3E%3D3.11-blue" alt="Python 3.11 or newer"></a>
  <a href="https://huggingface.co/leibler"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-leibler-yellow" alt="Environments on Hugging Face"></a>

  <br><br>
  <img src="docs/assets/kullback-tui.webp" alt="The kullback terminal screen: the commands, and the builds running now" width="820">
</div>

Your traces already hold the tasks, the tool signatures, what the tools returned and the effect of every write. That is enough to rebuild a copy of your system that runs, and to grade any model on what it changed in that copy. The grader is code, so it is cheap and has no opinions.

Kullback is the open-source Builder and Runner behind [Leibler](https://leibler.dev), under Apache-2.0. It is under active development. The numbers below move every build round, and the interfaces can still change.

## News

- **2026-09-09.** The retail Environment is published on Hugging Face as a release at [leibler/retail](https://huggingface.co/datasets/leibler/retail), with airline and telecom beside it as previews. We are now extending the Tasks in it: first by raising the trusted count, then by generating new Tasks over the rebuilt world.

## Environments

| Environment | Replay fidelity | Trusted Tasks | Status |
| --- | --- | --- | --- |
| [leibler/retail](https://huggingface.co/datasets/leibler/retail) | 95.1% | 122 of 205 | release |
| [leibler/airline](https://huggingface.co/datasets/leibler/airline) | 72.3% | 38 of 119 | preview |
| [leibler/telecom](https://huggingface.co/datasets/leibler/telecom) | 9.3% | 0 of 183 | preview |

A release replays at least 90% of its Tasks. A preview is below that bar and is published anyway, with its numbers on its card. All three come from the public [tau2-bench](https://github.com/sierra-research/tau2-bench) corpora (MIT). A package holds the rebuilt world, the Task list and the Verifiers, and none of the recordings it was built from.

```bash
uv run kullback fetch leibler/retail --out env-retail
uv run kullback run --workdir env-retail --task <task id> --model provider/model
uv run kullback verdict --workdir env-retail
uv run kullback report --workdir env-retail
```

## Quick start

Build your own Environment from a trace export:

```bash
uv sync
uv run kullback ingest path/to/traces.json --workdir work
uv run kullback build --workdir work --model provider/model --workers 8
uv run kullback freeze-runner --workdir work --yes
uv run kullback run --workdir work --task <task id> --model provider/candidate
uv run kullback verdict --workdir work
uv run kullback report --workdir work
```

`kullback tui` watches a build as it runs. `kullback publish --workdir work --repo <org>/<name>` puts the result on Hugging Face. Live model calls need `HARNESS_ALLOW_MODEL_REQUESTS=1` and an API key in the environment or a `.env` file. Any `provider/model` id works. `--model provider/model` reaches any provider models.dev lists with an OpenAI-shaped API (Anthropic, OpenAI, OpenCode Go, DeepSeek, OpenRouter and others), with the key in the variable models.dev names for it, and `--base-url` reaches anything else.

## How it works

The Builder turns traces into a database, one function per tool, the policy as checks before every write, a simulated user, and a Verifier per Task. The Runner plays a candidate model through it and the Verdict is code over what changed. Every artifact passes a code gate, the Runner is frozen once a person confirms it, and a Run served by a stand-in is never counted. The map is in [docs/architecture.md](docs/architecture.md).

## Read more

- [docs/architecture.md](docs/architecture.md), the map
- [docs/harness-design.md](docs/harness-design.md), the spec
- [docs/decision-log.md](docs/decision-log.md), why
- [docs/todo.md](docs/todo.md), what comes next
- [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPING.md](DEVELOPING.md)

## Citation

```bibtex
@misc{kullback2026,
  title  = {Kullback: Executable Environments and Verifiers Rebuilt from Agent Traces},
  author = {Krrish Agarwalla},
  year   = {2026},
  note   = {Open-source Builder and Runner behind Leibler. https://github.com/leiblerdev/kullback}
}
```
