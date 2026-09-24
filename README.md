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

- **2026-09-23.** Smoke 8: two builds with `openai/gpt-6-sol` as the Builder on the wave 2 code (f913cd6), no ceiling, both finished. Retail confirms all 205 References, agrees on all 3220 recorded calls and has 133 trusted Tasks (smoke 7: 14); the Builder stopped itself with 72 Tasks open, saying their Verifiers were outside its accessible root. Airline confirms 95 of 119 References at 96.03% call fidelity with 53 trusted Tasks (smoke 7: 22) and was stopped by the stop rule. The trusted counts are under review, because 49 retail and 27 airline of them rest on a second-path seed read from a file that several Tasks wrote at once; they will be re-derived before anything is built on them. Both are on Hugging Face under the tag `build-20260923`: [leibler/retail](https://huggingface.co/datasets/leibler/retail) as a release and [leibler/airline](https://huggingface.co/datasets/leibler/airline) as a preview, since airline's 79.8% over Tasks is below the 90% bar. An investigation into the open Tasks and the spend is running.
- **2026-09-23.** Smoke 7: two builds with `openai/gpt-6-sol` as the Builder, no ceiling, killed on the founder's request at 13:44 UTC (commit 92d3f28). Replay is near complete: 202 of 205 retail References and 100 of 119 airline References are confirmed. The remaining walls are three body rules and the write gate replaying on the Starting state; trusted is limited by derivation volume (ten Tasks per examine call, killed early), not by Verifiers failing. The readings are in `.claude/reports/sol-behaviour-2026-09-23.md` and `.claude/reports/env-quality-vs-tau2-2026-09-23.md`.
- **2026-09-09.** The retail Environment is published on Hugging Face as a release at [leibler/retail](https://huggingface.co/datasets/leibler/retail), with airline and telecom beside it as previews. We are now extending the Tasks in it: first by raising the trusted count, then by generating new Tasks over the rebuilt world.

## Environments

Smoke 8, 2026-09-23, Builder `openai/gpt-6-sol`, published under the tag `build-20260923`. Fidelity over Tasks is the card's number: the Tasks with at least one confirmed replay, over all Tasks. Call fidelity is the share of every recorded call whose replay agrees with the recording.

| Environment | Fidelity over Tasks | Over Runs | Call fidelity | References confirmed | Verifiers | Trusted Tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [leibler/retail](https://huggingface.co/datasets/leibler/retail) | 100.0% (205 of 205) | 100.0% (456 of 456) | 100.00% over 3220 calls | 205 | 205 | 133 of 205 | release |
| [leibler/airline](https://huggingface.co/datasets/leibler/airline) | 79.8% (95 of 119) | 78.5% (157 of 200) | 96.03% over 1513 calls | 95 | 95 | 53 of 119 | preview |
| [leibler/telecom](https://huggingface.co/datasets/leibler/telecom) | 9.3% (17 of 183) | 18.5% (27 of 146) | not on its card | 7 | 14 | 0 of 183 | preview |

The trusted counts are under review, because 49 retail and 27 airline of them rest on a second-path seed read from a file that several Tasks wrote at once; they will be re-derived before anything is built on them. Telecom was not rebuilt in smoke 8; its row is still round 5 from 2026-09-09. No round closed in smoke 8, so the retail and airline cards take their counts from the workdir status.

Previous numbers (smoke 7, the same day and Builder, fidelity over Tasks, over Runs, call fidelity, Verifiers, trusted): retail 98.5% (202 of 205), 96.7%, 99.53%, 20 and 14 of 205; airline 84.0% (100 of 119), 84.5%, 95.64%, 30 and 22 of 119.

A release replays at least 90% of its Tasks. A preview is below that bar and is published anyway, with its numbers on its card. All three come from the public [tau2-bench](https://github.com/sierra-research/tau2-bench) corpora (MIT). A package holds the rebuilt world, the Task list and the Verifiers, and none of the recordings it was built from.

```bash
uv run kullback fetch leibler/retail --out env-retail
uv run kullback run --workdir env-retail --task <task id> --model provider/model
uv run kullback verdict --workdir env-retail
uv run kullback report --workdir env-retail
```

## Words

These are the harness's own definitions, each with the code it is computed in.

A Run is one conversation of an agent with the tools, and a Trace is a recorded production Run. A Task's Reference is the Trace its user context comes from, the first of its Runs with rules (`kullback/runner/world/environment.py:203`). Replaying a Trace drives the rebuilt tools with the Trace's own turns and compares every call with the recording. A call agrees when its verdict is same, cosmetic or both refused (`AGREES`, `kullback/runner/replay.py:56`).

A replay is confirmed when nothing in it is off: every write and read agrees, no write leaves a row the recording moved where it was, no recorded call goes unmade, no turn is out of order, nothing crashes and no call is answered by the stand-in model (`confirmed=not reasons`, `kullback/runner/replay.py:707`). A confirmed Reference on the cards is a Task with at least one confirmed replay of any of its Traces, the same count as fidelity over Tasks (`kullback/hub/package.py:263`).

Fidelity over Tasks is Tasks with at least one confirmed replay divided by the Tasks on the frozen list. Fidelity over Runs is confirmed replays divided by every replay of those Tasks (`replay_fidelity`, `kullback/hub/package.py:185`). Call fidelity is the calls whose verdict agrees divided by every recorded call replayed, summed over the tools; it reads the verdict on the call's own answer, so a write that answered right but left a row unmoved still counts as agreeing there while it fails the replay (`fidelity_index`, `kullback/builder/domain_tools.py:255`).

A Verifier is one Task's End-state check, a list of atoms (`kullback/runner/records.py:523`). An atom is required, allowed, forbidden, question, communicate or hard (`kullback/runner/records.py:23`); a hard atom is a Hard rule compiled from the Environment's policy. The Examiner derives a Verifier from the Task's confirmed Reference and re-runs and is its one writer: it has no write or edit, only `propose_verifier`, which runs the gates before a version lands (`kullback/examiner/derive.py:1`, `kullback/examiner/session.py:85`). The Builder cannot change a Verifier: it writes only under `tools/`, `intents/`, `refusals/` and `policy/` (`kullback/builder/session.py:44`). The gates on a Verifier file are the oracle replay, the suite, loosening, false rejection and trusted (`VERIFIER_GATES`, `kullback/gates/bindings.py:241`). Loosening lets a new version newly pass a Run only when that Run is in the legitimate pool: confirmed replays and finished re-rolls (`kullback/gates/loosening.py:46`).

A Task is trusted when all of these hold (`trusted_gate`, `kullback/gates/trust.py:151`). Its Verifier passed the nine suite checks: provenance spans, the oracle passes on the Reference, an empty Run fails, a plausible wrong Run fails, an unsolved state fails, a second path passes, the loophole probe fails, the leak check is clean and a mutation flips it (`validate_verifier`, `kullback/gates/verifier_suite.py:520`). No probe in its pool scores a pass. The file is the last accepted version and does not loosen past the frontier. The Task is not refused. And the Verifier is not over strict: over the legitimate Runs it was not derived from, it does not reject every one (`over_strict`, `kullback/gates/loosening.py:141`, threshold 1.0 at line 29). A pool of one Run that the Verifier rejects is enough to fail this.

An open Task is any other Task: it has no Verifier yet, or the first of those conditions that fails is its reason (`status_of`, `kullback/builder/domain_tools.py:563`). A refused Task is one the Builder refused in `refusals/<task>.json` as one nobody can finish, admitted only when no frontier Run of it finished (`refuse_gate`, `kullback/gates/trust.py:67`). A release replays at least 90% of its Tasks (`FIDELITY_BAR`, `kullback/hub/card.py:25`); a preview is below that bar and publishes only with `--preview`.

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

`kullback build` takes an optional `--model`; without it the Builder, the Examiner, the judges, the probe and the re-rolls all run on `openai/gpt-6-luna`. `kullback tui` watches a build as it runs. `kullback publish --workdir work --repo <org>/<name>` puts the result on Hugging Face. Live model calls need `HARNESS_ALLOW_MODEL_REQUESTS=1` and an API key in the environment or a `.env` file. Any `provider/model` id works. `--model provider/model` reaches any provider models.dev lists with an OpenAI-shaped API (Anthropic, OpenAI, OpenCode Go, DeepSeek, OpenRouter and others), with the key in the variable models.dev names for it, and `--base-url` reaches anything else.

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
