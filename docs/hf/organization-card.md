# leibler

[Leibler](https://leibler.dev) rebuilds the system a working agent ran in from its traces and grades other models in code on what they change there, with [Kullback](https://github.com/leiblerdev/kullback), its open-source harness.

## Environments

| Environment | Status | Tasks | Harbor Tasks |
| --- | --- | --- | --- |
| [leibler/airline](https://huggingface.co/datasets/leibler/airline) | release | 130 | 123 |
| [leibler/retail](https://huggingface.co/datasets/leibler/retail) | release | 223 | 186 |

## Use it

```bash
uv pip install git+https://github.com/leiblerdev/kullback.git
uv run kullback fetch leibler/<environment> --out env
uv run kullback run --workdir env --task <task id> --model provider/model
uv run kullback verdict --workdir env
uv run kullback report --workdir env
```

## Terms

- Environment: a simulated world with tools and rules that an agent acts in.
- Task: one goal the agent is asked to reach in an Environment.
- Run: one attempt by an agent at a Task, recorded step by step.
- Verifier: the check that decides from the end state whether a Run succeeded.

## Known issues

Some Tasks can be wrong. A Verifier can check the wrong thing or a Task can be unfinishable. Treat a single Task's result with care and the set as a whole as an eval.

## Links

- Harness: https://github.com/leiblerdev/kullback
- Site: https://leibler.dev
- Licence: Apache-2.0 for the harness and every package; each corpus keeps its own.

