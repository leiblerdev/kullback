# Environment card fields

`kullback/hub/card.py` renders every card from the package's `manifest.json` alone, and this is what each field means and where it is read from.

| Field | Meaning | Read from |
| --- | --- | --- |
| name | The Environment's name, which is also its domain tag | `--name`, or the last segment of `--repo` |
| preview | Whether it is below the fidelity bar | `--preview` on the publish |
| round | The build round the numbers were measured in | the last record in `rounds.json` |
| tasks_total | Tasks on the frozen list | `tasks_frozen.json`, or the newest round snapshot |
| replay_fidelity.tasks_rate | Share of Tasks with at least one confirmed replay | `replays.json` |
| replay_fidelity.runs_rate | Share of replayed Runs that were confirmed | `replays.json` |
| replay_fidelity.calls_rate | Share of recorded calls that agree, over `calls_total` | `replays.json` |
| tag | The tag this publish carries | `round-<n>`, else `build-<YYYYMMDD>` of the newest session write |
| counts_source | Where the counts came from | the last round record, else the workdir status |
| reference_confirmed | Tasks with at least one confirmed replay | `replays.json` |
| verifier_derived | Tasks with a Verifier on disk | `verifiers/` |
| trusted | Tasks whose Verifier passed the whole suite | the last round's trusted ruling, else the Builder's status rule |
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

`license` is the corpus licence as the host's id, or `other` when none was named; `tags` is `kullback`, `environment`, `agent-evaluation`, `rl-environment` and the Environment's name.

## Preview and release

A release needs fidelity over Tasks of at least 90%; below that only `publish --preview` is allowed, and the card says so above its table.

