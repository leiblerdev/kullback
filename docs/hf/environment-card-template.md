# Environment card fields

Every card under the organisation is rendered by `kullback/hub/card.py` from the package's `manifest.json` and from nothing else, so a number on a page is a number the package carries. This is what each field means and which record it is read from. Nothing on a card is written by hand, and nothing on it quotes a Task, a tool result or a status row: every reason a card gives is one of the harness's fixed phrases.

| Field | Meaning | Read from |
| --- | --- | --- |
| name | The Environment's name, which is also its domain tag | `--name`, or the last segment of `--repo` |
| preview | Whether it is below the fidelity bar | `--preview` on the publish |
| round | The build round the numbers were measured in | the last record in `rounds.json` |
| tasks_total | Tasks on the frozen list | `tasks_frozen.json`, or the newest round snapshot |
| replay_fidelity.tasks_rate | Share of Tasks with at least one confirmed replay | `replays.json` |
| replay_fidelity.runs_rate | Share of replayed Runs that were confirmed | `replays.json` |
| reference_confirmed | Tasks whose recordings agreed on an End state | `task_status.json` |
| verifier_derived | Tasks with a Verifier on disk | `verifiers/` |
| trusted | Tasks whose Verifier passed the whole suite | the last round's trusted ruling |
| refused | Tasks the harness ruled nobody finished | the last round's refuse ruling |
| funnel | How many Tasks stopped at each rung | the per-Task index the export writes |
| buckets | Tasks and trusted Tasks per difficulty bucket | `difficulty.json` (D209), else computed |
| untrusted | Untrusted count and the commonest fixed reasons | the per-Task index |
| runner_version | Hash of the frozen Runner the numbers were measured under | `runner_version.json` |
| gates_version | Hash of the gates package | `runner_version.json` |
| kullback_version | Harness version | the installed distribution |
| git_sha | Commit the export ran from | `git rev-parse HEAD` |
| leak_scan | What the export checked against the source corpus, in counts | the export's own scan |
| content_hash | One hash over every file of the package | computed at export |
| files | sha256 per file, which is what a fetch verifies | computed at export |

## Front matter

`license` is the source corpus's licence, lowercased to the id a dataset host indexes, or `other` where the publisher named none. `tags` is always `kullback`, `environment`, `agent-evaluation`, `rl-environment` followed by the Environment's own name, which is the domain.

## Preview and release

A release requires replay fidelity over Tasks at or above 90%. Below that, `publish --preview` is the only form that is allowed, it sets `preview: true` in the manifest, and the card opens with a banner naming the bar and the Environment's own numbers.

