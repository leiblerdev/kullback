# leibler

Leibler takes the traces a working agent already left behind and rebuilds the system it worked in as something you can run. Other models are then graded on what they change in that copy. The graders are plain code, so they are cheap to run and never have an opinion. [leibler.dev](https://leibler.dev)

[Kullback](https://github.com/leiblerdev/kullback) is the open-source Builder and Runner that does this, under Apache-2.0. It rebuilds the world from the traces and checks the rebuild by replaying them. From the recorded runs it derives one Verifier per Task, and every artifact has to pass a code gate that no model can touch. Each Environment below came out of Kullback, and each card lists its numbers and what it cannot do yet.

All of this is still being built. We republish an Environment after each build, and older builds stay reachable by their tags. Next we want more trusted Tasks on every Environment, and after that, synthetic Tasks generated over the rebuilt world.

## Words

These are the harness's own definitions, each with the code it is computed in.

A Run is one conversation of an agent with the tools, and a Trace is a recorded production Run. A Task's Reference is the Trace its user context comes from, the first of its Runs with rules (`kullback/runner/world/environment.py:203`). Replaying a Trace drives the rebuilt tools with the Trace's own turns and compares every call with the recording. A call agrees when its verdict is same, cosmetic or both refused (`AGREES`, `kullback/runner/replay.py:56`).

A replay is confirmed when nothing in it is off: every write and read agrees, no write leaves a row the recording moved where it was, no recorded call goes unmade, no turn is out of order, nothing crashes and no call is answered by the stand-in model (`confirmed=not reasons`, `kullback/runner/replay.py:707`). A confirmed Reference on the cards is a Task with at least one confirmed replay of any of its Traces, the same count as fidelity over Tasks (`kullback/hub/package.py:263`).

Fidelity over Tasks is Tasks with at least one confirmed replay divided by the Tasks on the frozen list. Fidelity over Runs is confirmed replays divided by every replay of those Tasks (`replay_fidelity`, `kullback/hub/package.py:185`). Call fidelity is the calls whose verdict agrees divided by every recorded call replayed, summed over the tools; it reads the verdict on the call's own answer, so a write that answered right but left a row unmoved still counts as agreeing there while it fails the replay (`fidelity_index`, `kullback/builder/domain_tools.py:255`).

A Verifier is one Task's End-state check, a list of atoms (`kullback/runner/records.py:523`). An atom is required, allowed, forbidden, question, communicate or hard (`kullback/runner/records.py:23`); a hard atom is a Hard rule compiled from the Environment's policy. The Examiner derives a Verifier from the Task's confirmed Reference and re-runs and is its one writer: it has no write or edit, only `propose_verifier`, which runs the gates before a version lands (`kullback/examiner/derive.py:1`, `kullback/examiner/session.py:85`). The Builder cannot change a Verifier: it writes only under `tools/`, `intents/`, `refusals/` and `policy/` (`kullback/builder/session.py:44`). The gates on a Verifier file are the oracle replay, the suite, loosening, false rejection and trusted (`VERIFIER_GATES`, `kullback/gates/bindings.py:241`). Loosening lets a new version newly pass a Run only when that Run is in the legitimate pool: confirmed replays and finished re-rolls (`kullback/gates/loosening.py:46`).

A Task is trusted when all of these hold (`trusted_gate`, `kullback/gates/trust.py:151`). Its Verifier passed the nine suite checks: provenance spans, the oracle passes on the Reference, an empty Run fails, a plausible wrong Run fails, an unsolved state fails, a second path passes, the loophole probe fails, the leak check is clean and a mutation flips it (`validate_verifier`, `kullback/gates/verifier_suite.py:520`). No probe in its pool scores a pass. The file is the last accepted version and does not loosen past the frontier. The Task is not refused. And the Verifier is not over strict: over the legitimate Runs it was not derived from, it does not reject every one (`over_strict`, `kullback/gates/loosening.py:141`, threshold 1.0 at line 29). A pool of one Run that the Verifier rejects is enough to fail this.

An open Task is any other Task: it has no Verifier yet, or the first of those conditions that fails is its reason (`status_of`, `kullback/builder/domain_tools.py:563`). A refused Task is one the Builder refused in `refusals/<task>.json` as one nobody can finish, admitted only when no frontier Run of it finished (`refuse_gate`, `kullback/gates/trust.py:67`). A release replays at least 90% of its Tasks (`FIDELITY_BAR`, `kullback/hub/card.py:25`); a preview is below that bar and publishes only with `--preview`.

## Environments

Smoke 8, 2026-09-23, Builder `openai/gpt-6-sol`, tag `build-20260923`. Fidelity over Tasks counts the Tasks with at least one confirmed replay, out of all Tasks. The words are defined above. Call fidelity counts every recorded call.

| Environment | Fidelity over Tasks | Over Runs | Call fidelity | References confirmed | Verifiers | Trusted Tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [retail](https://huggingface.co/datasets/leibler/retail) | 100.0% (205 of 205) | 100.0% (456 of 456) | 100.00% over 3220 calls | 205 | 205 | 133 of 205 | release |
| [airline](https://huggingface.co/datasets/leibler/airline) | 79.8% (95 of 119) | 78.5% (157 of 200) | 96.03% over 1513 calls | 95 | 95 | 53 of 119 | preview |
| [telecom](https://huggingface.co/datasets/leibler/telecom) | 9.3% (17 of 183) | 18.5% (27 of 146) | not on its card | 7 | 14 | 0 of 183 | preview |

The trusted counts are under review, because 49 retail and 27 airline of them rest on a second-path seed read from a file that several Tasks wrote at once; they will be re-derived before anything is built on them. Telecom was not rebuilt in smoke 8. Its row is still round 5, built 2026-09-09. No round closed in smoke 8, so the retail and airline counts come from the workdir status, as the "Counts from" row on each card says.

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
