# leibler

[Leibler](https://leibler.dev) rebuilds the system a working agent ran in from its traces, then grades other models in code on what they change there. [Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner behind it, under Apache-2.0.

## News

2026-10-17. Every Verifier in the retail and airline Environments is now derived from the user's ask in the recording and the policy. The recorded answer is a ruler for checking, never an input, a gated value must be one the tools can write, and a Task whose user withdrew the ask gets no Verifier. Blind model raters, one per Task under a written rubric and with no human review, found retail 10 of 194 Verifiers wrong against the recording and 3 of 76 trusted ones, airline 11 of 125 and 2 of 42. The held-out run of 2026-10-12 on the previous harness had 13 percent of its trusted Verifiers wrong against the recording under an earlier rubric; on 2026-09-25 about a third of trusted Tasks sat on a wrong Reference.

## Environments

| Environment | Tasks | Harbor Tasks | Tag | Status |
| --- | --- | --- | --- | --- |
| [leibler/airline](https://huggingface.co/datasets/leibler/airline) | 130 | 123 | build-20261017 | release |
| [leibler/retail](https://huggingface.co/datasets/leibler/retail) | 223 | 186 | build-20261017 | release |

## Terms

- Environment: a rebuilt system an agent works in, its data, tools and policy, packaged with Tasks.
- Task: one thing a user asked for, with what the agent is given to start.
- Trace: a recorded conversation between a working agent, its user and the tools.
- Verifier: a Task's check, in code, of the state the agent leaves behind.
- Release and preview: a release replays at least 90% of its Tasks as recorded; a preview is below that.
- Harbor format: the same Tasks under `harbor/`, laid out to run with the Harbor CLI.

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
- Licence: Apache-2.0 for the harness and every package; each corpus keeps its own.

