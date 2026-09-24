<div align="center">
  <h1>Kullback</h1>
  <p><strong>Rebuild an executable environment from your agent's traces, then grade and train models on it.</strong></p>
  <p><a href="https://huggingface.co/leibler"><strong>Environments on Hugging Face</strong></a> · <a href="#quick-start"><strong>Quick start</strong></a> · <a href="docs/architecture.md"><strong>Architecture</strong></a></p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache-2.0"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-%3E%3D3.11-blue" alt="Python 3.11+"></a>
  <a href="https://huggingface.co/leibler"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-leibler-yellow" alt="Environments on Hugging Face"></a>
  <br><br>
  <img src="docs/assets/kullback-tui.webp" alt="The kullback TUI" width="820">
</div>

Kullback rebuilds a running copy of your system from your agent's traces and grades any model, in code, on what it changes there. It is the open-source Builder and Runner behind [Leibler](https://leibler.dev), under Apache-2.0, and changes every build round.

## News

- **2026-09-24.** Smoke 9 on Opus 5.5 (Bedrock): retail replays all 223 Tasks with 193 trusted, airline all 130 with 82 trusted. Both published as `build-20260924`, airline now as a release.
- **2026-09-23.** Retail replays all 205 Tasks with 133 trusted; airline 95 of 119 with 53 trusted. Both published as `build-20260923`.
- **2026-09-09.** Retail published on Hugging Face as a release, airline as a preview.

## Environments

Latest build: smoke 9, 2026-09-24, built and examined with Opus 5.5 on Amazon Bedrock.

| | Retail | Airline |
|---|---|---|
| Tasks | 223 | 130 |
| Reference Traces confirmed by replay | 223 | 130 |
| Task replay fidelity | 1.00 | 1.00 |
| Run replay fidelity | 456 of 456 | 199 of 200 |
| Call fidelity | 100.00% of 3220 calls | 99.93% of 1513 calls |
| Verifiers derived | 222 | 128 |
| Trusted Tasks | 193 | 82 |
| Refused | 0 | 0 |
| Spend, USD | 71.01 | 74.67 |
| Of which the runner (Candidate Runs) | about 57 | about 59 |
| Of which the Examiner | about 9 | about 12 |
| Of which the Builder | about 2 | about 4 |

Building and examining are cheap; replaying and running Candidates is most of the cost.

Both packages on the Hub, [leibler/retail](https://huggingface.co/datasets/leibler/retail) and [leibler/airline](https://huggingface.co/datasets/leibler/airline), are this build. Both Environments come from the public [tau2-bench](https://github.com/sierra-research/tau2-bench) corpora (MIT); no package carries the recordings.

```bash
uv run kullback fetch leibler/retail --out env-retail
uv run kullback run --workdir env-retail --task <task id> --model provider/model
uv run kullback verdict --workdir env-retail
uv run kullback report --workdir env-retail
```

## Words

- Run and Trace: a Run is an agent's conversation with the tools; a Trace is a recorded one.
- Reference: the Trace a Task's user context comes from.
- Replay and agree: a replay re-drives a Trace's turns; a call agrees when its verdict is same, cosmetic or both refused.
- Confirmed: a replay where every call agrees, none was answered by the stand-in, and nothing is missing, reordered or crashed.
- Fidelity over Tasks: Tasks with a confirmed replay, over all Tasks.
- Fidelity over Runs: confirmed replays over all replays.
- Call fidelity: agreeing calls over all recorded calls.
- Verifier: a Task's End-state check, written only by the Examiner, never by the Builder.
- Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.
- Gates: oracle replay, suite, loosening, false rejection, trusted.
- Trusted: suite passed, probes fail, last version, no loosening, not over strict, not refused.
- Open: neither trusted nor refused.
- Refused: a Task the Builder showed nobody can finish.
- Release and preview: a release replays at least 90% of its Tasks; a preview is below that.

## Quick start

Build your own Environment from a trace export:

```bash
uv sync
uv run kullback ingest path/to/traces.json --workdir work
uv run kullback build --workdir work
uv run kullback freeze-runner --workdir work --yes
uv run kullback run --workdir work --task <task id> --model provider/candidate
uv run kullback verdict --workdir work
uv run kullback report --workdir work
```

The build defaults to Opus 5.5 on Amazon Bedrock's global profile, with `AWS_BEARER_TOKEN_BEDROCK` or the AWS key pair and `AWS_REGION` (default `us-east-2`). `--model` takes any id models.dev lists, OpenAI on Bedrock included (`bedrock/global.openai.gpt-6-sol`). Live calls need `HARNESS_ALLOW_MODEL_REQUESTS=1`; `kullback tui` watches a build and `kullback publish` puts it on Hugging Face.

## How it works

The Builder turns traces into a database, tools, policy checks, a simulated user and a Verifier per Task. The Runner plays a candidate through it and grades what changed in code.

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
