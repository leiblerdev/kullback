# leibler

[Leibler](https://leibler.dev) rebuilds the system a working agent ran in from its traces, then grades other models in code on what they change there. [Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner behind it, under Apache-2.0.

## Environments

| Environment | Fidelity over Tasks | Call fidelity | Verifiers | Trusted | Status |
| --- | --- | --- | --- | --- | --- |
| [leibler/retail](https://huggingface.co/datasets/leibler/retail) | 100.0% (205 of 205) | 100.00% of 3220 | 205 | 133 of 205 | release |
| [leibler/airline](https://huggingface.co/datasets/leibler/airline) | 79.8% (95 of 119) | 96.03% of 1513 | 95 | 53 of 119 | preview |

The trusted counts are under review because some rest on a seed file several Tasks shared. Both come from the public [tau2-bench](https://github.com/sierra-research/tau2-bench) corpora (MIT); no package carries the recordings.

## Words

- Run and Trace: a Run is an agent's conversation with the tools; a Trace is a recorded one.
- Reference: the Trace a Task's user context comes from.
- Replay and agree: a replay re-drives a Trace's turns; a call agrees when its verdict is same, cosmetic or both refused.
- Confirmed: a replay where every call agrees and nothing is missing, reordered or crashed.
- Fidelity over Tasks: Tasks with a confirmed replay, over all Tasks.
- Fidelity over Runs: confirmed replays over all replays.
- Call fidelity: agreeing calls over all recorded calls.
- Verifier: a Task's End-state check, written only by the Examiner, never by the Builder.
- Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.
- Gates: oracle replay, suite, loosening, false rejection, trusted.
- Trusted: suite passed, probes fail, last version, no loosening, not over strict, not refused.
- Open: not yet trusted.
- Refused: a Task the Builder showed nobody can finish.
- Release and preview: a release replays at least 90% of its Tasks; a preview is below that.

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
