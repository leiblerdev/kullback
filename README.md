<div align="center">
  <h1>Kullback</h1>
  <p><strong>Rebuild an executable environment from your agent's traces, then grade models on it in code.</strong></p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache-2.0"></a>
  <a href="https://huggingface.co/leibler"><img src="https://img.shields.io/badge/Hugging%20Face-leibler-yellow" alt="Hugging Face"></a>
  <br><br>
  <img src="docs/assets/kullback-tui.webp" alt="The kullback TUI" width="820">
</div>

## What this is

Kullback turns the recorded conversations of a tool-using agent into a running copy of the system it worked in: a
database, tools, policy checks, a simulated user and a Verifier per Task. Any model can then be played through that
copy, and a Verifier decides in code, from the end state, whether the model did what the user asked. It is the
open-source harness behind [Leibler](https://leibler.dev).

## Announcements

- 2026-11-03: [leibler/travel-agency](https://huggingface.co/datasets/leibler/travel-agency) released on Hugging Face:
  65 Tasks for an online travel agency, each from a real public complaint, shipped as Harbor tasks.
- 2026-09-24: [leibler/retail](https://huggingface.co/datasets/leibler/retail) (223 Tasks) and
  [leibler/airline](https://huggingface.co/datasets/leibler/airline) (130 Tasks) released on Hugging Face.
- 2026-09-09: [leibler/retail](https://huggingface.co/datasets/leibler/retail) first released on Hugging Face, with
  airline as a preview.

## What we did

We rebuilt two Environments from the public [tau2-bench](https://github.com/sierra-research/tau2-bench) corpora and
released them on Hugging Face: [leibler/retail](https://huggingface.co/datasets/leibler/retail) with 223 Tasks and
[leibler/airline](https://huggingface.co/datasets/leibler/airline) with 130 Tasks. Every Task replays its recording in
the rebuilt world, and each Verifier is derived from the user's ask and the policy, never from the recorded answer.

The third, [leibler/travel-agency](https://huggingface.co/datasets/leibler/travel-agency), has 65 hard Tasks for an
online travel agency (TripDesk) across airlines, rail, hotels, rides, rental cars and cruises, each built from a real
public complaint and shipped as Harbor tasks. It differs from the first two: the customer sends one message and there
is no simulated user, and a run is graded on the end state, the policy checks on each write and checks on the reply.
All 263 deliberately wrong walks are caught and all 134 alternative right walks are accepted.

## How to use it

```bash
git clone https://github.com/leiblerdev/kullback.git && cd kullback && uv sync
uv run kullback fetch leibler/retail --out env-retail
uv run kullback run --workdir env-retail --task <task id> --model provider/model
uv run kullback verdict --workdir env-retail
uv run kullback report --workdir env-retail
```

`fetch` checks the content hash, and `ls env-retail/tasks` lists the Task ids. `run` plays one model on one Task; live
calls need `HARNESS_ALLOW_MODEL_REQUESTS=1` and the provider's key (see `.env.example`). `verdict` scores the stored
Runs and `report` writes `report.md`. To build an Environment from your own traces, start with `kullback ingest` and
`kullback build`; [docs/trace-intake.md](docs/trace-intake.md) covers the trace formats, and
[docs/README.md](docs/README.md) lists every doc.

The travel Tasks run under Harbor with Docker; Harbor's oracle agent plays each reference solution at no cost:

```bash
hf download leibler/travel-agency --repo-type dataset --local-dir travel-agency
harbor run -p travel-agency -a oracle -n 4
```

Grading a real agent needs a judge model for the reply checks; [the dataset card](https://huggingface.co/datasets/leibler/travel-agency)
lists the variables and the agent commands.

## Why it matters

Benchmarks written by hand are slow to make and rarely look like your own system. Your agent's traces already hold the
tools, the data and what users ask for. Kullback turns them into Tasks with checks in code, so you can compare models
on your own work, and train on it, without hand-writing a benchmark.

## Known issues

Some Tasks can be wrong. A Verifier can check the wrong thing, or a Task can be impossible to finish. Treat the result
of a single Task with care and the set as a whole as an eval. The decisions behind the harness are in
[docs/decision-log.md](docs/decision-log.md), and what comes next is in [docs/todo.md](docs/todo.md).

## Licence

Apache-2.0 for the harness and every package; each source corpus keeps its own licence (tau2-bench is MIT).
