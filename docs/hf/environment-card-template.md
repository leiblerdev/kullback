# Environment card fields

`kullback/hub/card.py` renders every card from the package's `manifest.json` alone, in six parts: a summary with its status, use it, what is inside, known issues (a dated list that says some Tasks can be wrong), how it was built with a Built line, and the licence. These are the fields the card reads; every other number stays in `manifest.json` and `tasks_index.json`.

| Field | Meaning | Read from |
| --- | --- | --- |
| name | The Environment's name, which is also its domain tag | `--name`, or the last segment of `--repo` |
| source | The corpus, its licence and its URL | `--corpus`, `--corpus-license`, `--corpus-url` |
| tasks_total | Tasks on the frozen list | `tasks_frozen.json`, or the newest round snapshot |
| inside.tools | Tools the Environment serves | `tool_sigs.json` in the laid package |
| inside.example_ask | One Task's first ask, as its instruction states it | the first Harbor Task's `instruction.md`, else `tasks.jsonl` |
| replay_fidelity.tasks_rate | Share of Tasks with at least one confirmed replay | `replays.json` |
| preview | Whether it was published as a preview | `--preview` on the publish |
| tag | The tag this publish carries | `round-<n>`, else `build-<YYYYMMDD>` of the newest session write |
| git_sha | Commit the export ran from; the card also names the Harbor form's where it differs | `git rev-parse HEAD` |
| created_at | When the package was exported | the export's clock, UTC |
| content_hash | One hash over every file of the package, which a fetch checks | computed at export |

## Front matter

`license` is the corpus licence as the host's id, or `other` when none was named; `tags` is `kullback`, `environment`, `agent-evaluation`, `rl-environment` and the Environment's name.

## Harbor

Where the package carries a Harbor form, the card reads `harbor/manifest.json` for the exported count, the Tasks without an oracle and the example Task's path.

