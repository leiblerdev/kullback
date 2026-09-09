"""The cards: one per Environment, one for the organisation, and the template that documents both (D221).

A card is rendered from a manifest and from nothing else, so a number on a page is a number a
package actually carries and the two cannot drift. The front matter is what a dataset host indexes
on; the body is what a person reads before deciding whether to trust the Environment, which is why
the funnel and the untrusted count are on the page rather than in a file nobody opens.

Nothing here quotes a status row, a Task, a tool result or any other record: every reason a card
gives is one of the harness's own fixed phrases, chosen in package.py for that reason.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.hub.package import TASKS_ROWS_NAME

# What every Environment published from this harness is tagged with, before its own domain.
BASE_TAGS: tuple[str, ...] = ("kullback", "environment", "agent-evaluation", "rl-environment")
# The licence a card states when the caller named none for the source corpus. A host will not index
# an empty value, and claiming a permissive licence nobody checked would be worse than saying this.
UNKNOWN_LICENSE = "other"

# The bar an Environment has to clear to be published as a release rather than as a preview.
FIDELITY_BAR = 0.90


def front_matter(manifest: dict, tags: Iterable[str] = ()) -> str:
    """The YAML block a dataset host indexes: the licence of the source corpus and the tags."""
    source = manifest.get("source") or {}
    license_id = str(source.get("license") or UNKNOWN_LICENSE).strip().lower() or UNKNOWN_LICENSE
    names = list(BASE_TAGS) + [str(tag) for tag in tags if tag]
    seen, ordered = set(), []
    for tag in names:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    lines = ["---", f"license: {license_id}", "tags:"]
    lines += [f"  - {tag}" for tag in ordered]
    # The viewer renders one flat file; the rest of the package is the world and the graders.
    lines += ["configs:", "  - config_name: tasks", "    data_files:",
              "      - split: tasks", f"        path: {TASKS_ROWS_NAME}"]
    lines += ["---"]
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "not measured" if value is None else f"{float(value):.1%}"


def _numbers_table(manifest: dict) -> list[str]:
    fidelity = manifest.get("replay_fidelity") or {}
    rows = [
        ("Tasks", str(int(manifest.get("tasks_total") or 0))),
        ("Replay fidelity, Tasks", f"{_pct(fidelity.get('tasks_rate'))} "
                                   f"({fidelity.get('tasks', 0)} of {fidelity.get('tasks_total', 0)})"),
        ("Replay fidelity, Runs", f"{_pct(fidelity.get('runs_rate'))} "
                                  f"({fidelity.get('runs', 0)} of {fidelity.get('runs_total', 0)})"),
        ("Reference confirmed", str(manifest.get("reference_confirmed", 0))),
        ("Verifier derived", str(manifest.get("verifier_derived", 0))),
        ("Trusted Tasks", str(manifest.get("trusted", 0))),
        ("Refused Tasks", str(manifest.get("refused", 0))),
        ("Round", str(manifest.get("round") if manifest.get("round") is not None else "not recorded")),
        ("Content hash", _short(manifest.get("content_hash"))),
    ]
    rows += _domain_rows(manifest)
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _domain_rows(manifest: dict) -> list[tuple[str, str]]:
    """What the domain's own public material attests, and how much of it this package covers (D225).

    Two rows, and only where a domain was read: how many task archetypes came off the material, and
    how many of them no tool of this Environment realises. Someone deciding whether to use this
    package wants that second number, because it says what the Environment cannot be asked to do,
    and nothing else on the card says it. Neither row is a Task count and neither is added to one.
    """
    counts = manifest.get("domain") or {}
    if not isinstance(counts, dict) or not counts:
        return []
    read = int(counts.get("archetypes_extracted") or 0)
    mapped = int(counts.get("archetypes_mapped") or 0)
    return [("Domain archetypes read", f"{read} ({mapped} a tool of this Environment realises)"),
            ("Domain archetypes with no tool", str(int(counts.get("archetype_gaps") or 0)))]


def _short(value: Any) -> str:
    text = str(value or "")
    return (text[:16] if len(text) > 16 else text) or "not recorded"


def _bucket_table(manifest: dict) -> list[str]:
    rows = list(manifest.get("buckets") or ())
    if not rows:
        return []
    lines = ["", "### By difficulty", "",
             "A bucket is the number of writes a Task makes, the tools its Reference called and the "
             "paths to its End state, each banded. The same bucket means the same thing in every "
             "Environment.", "",
             "| Bucket | Tasks | Trusted |", "| --- | --- | --- |"]
    lines += [f"| {row.get('bucket', '')} | {row.get('tasks', 0)} | {row.get('trusted', 0)} |" for row in rows]
    return lines


def _limits(manifest: dict) -> list[str]:
    untrusted = manifest.get("untrusted") or {}
    lines = ["", "## What it cannot do yet", ""]
    lines.append(f"- {untrusted.get('count', 0)} of {manifest.get('tasks_total', 0)} Tasks are not trusted. "
                 "Their Verifier has not passed the suite, so the harness will not grade a candidate on them.")
    for entry in untrusted.get("reasons") or ():
        lines.append(f"- {entry.get('tasks', 0)} Tasks: {entry.get('reason', '')}.")
    assisted = list(manifest.get("assisted_tools") or ())
    if assisted:
        lines.append(f"- {len(assisted)} tools were served by a stand-in at some point in the build "
                     f"({', '.join(sorted(assisted))}). A Run that touches one is reported and never counted.")
    added = int(manifest.get("tasks_added_later") or 0)
    if added:
        lines.append(f"- {added} more Tasks appeared after the Task list was frozen. They are outside every "
                     "number on this page, because the denominator is fixed once and never moved.")
    lines.append("- The simulated user does not ship. Its facts come from the recordings, which stay private, "
                 "so a Task whose answer the user only gives mid conversation cannot be finished here, even "
                 "though its Verifier still grades it.")
    leaks = manifest.get("leak_scan") or {}
    lines.append(f"- The export checked {leaks.get('values_checked', 0)} strings across "
                 f"{leaks.get('files_scanned', 0)} graded files against the {leaks.get('corpus_strings', 0)} "
                 f"strings in the source corpus. It found {leaks.get('leaks', 0)} recorded strings (any would "
                 f"have stopped the export) and {leaks.get('value_echoes', 0)} short values that only a "
                 "Verifier's answer key accounts for.")
    return lines


def _banner(manifest: dict) -> list[str]:
    if not manifest.get("preview"):
        return []
    fidelity = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
    return ["", f"> **Preview.** This Environment is below the {FIDELITY_BAR:.0%} replay fidelity bar for a "
                f"release. It replays {_pct(fidelity)} of its Tasks and has {manifest.get('trusted', 0)} "
                f"trusted Tasks of {manifest.get('tasks_total', 0)}. It is published so the numbers are "
                "public while it improves. Read them before you use it.", ""]


FUNNEL_PROSE = (
    "Every Task climbs a funnel, and every rung is a code check. A Task clears replay fidelity when the "
    "rebuilt tools answer its recorded calls the way the real ones did. It keeps a Reference when the "
    "recordings agree on an End state, and gets a Verifier derived from that Reference. It counts as "
    "trusted once that Verifier rejects an empty Run, a plausible wrong Run and a mutated Run, passes no "
    "loophole probe, accepts a second route to the same End state, and turns away few enough held-out Runs "
    "that did reach the Reference. The table above is that funnel, rung by rung, so a Task that stops early "
    "stays visible instead of dropping out of the denominator."
)

DEVELOPMENT_NOTE = (
    "This Environment is under active development. Each build round republishes it with new numbers, "
    "and earlier rounds stay reachable by their tags. The next stage is to raise the trusted count, "
    "then to generate Tasks synthetically over the rebuilt world, on top of the recorded ones."
)


def card_markdown(manifest: dict, repo_id: str, *, github_url: str = "https://github.com/leiblerdev/kullback",
                  site_url: str = "https://leibler.dev") -> str:
    """The dataset card of one Environment, every number read off its manifest."""
    name = manifest.get("name") or repo_id.rsplit("/", 1)[-1]
    source = manifest.get("source") or {}
    corpus = source.get("corpus") or "not stated"
    license_name = source.get("license") or "not stated"
    lines = [front_matter(manifest, tags=[str(name)]), ""]
    lines += [f"# {name}", ""]
    lines += [f"An executable Environment for evaluating and training tool-using agents. [Kullback]({github_url}) "
              f"built it from recorded traces of a working agent, and [Leibler]({site_url}) publishes it.", ""]
    lines += ["The package holds the rebuilt world: a database, one function per tool that behaves the way the "
              "real tool was observed to behave, the compiled policy, and the Starting state each Task begins "
              "from. It also holds the Task list with the instruction a candidate gets, and a code-only "
              "Verifier per Task that grades the candidate on what it changed. It holds none of the "
              "recordings it was built from.", ""]
    lines += [f"`{TASKS_ROWS_NAME}` is the Task list as one row per line, which is what the viewer shows: "
              "the id, the instruction, how far the Task got up the funnel and its difficulty bucket. "
              "Everything else in the package is the world and the graders, and only the harness reads "
              "those.", ""]
    lines += [DEVELOPMENT_NOTE, ""]
    lines += _banner(manifest)
    lines += ["## Numbers", ""]
    lines += _numbers_table(manifest)
    lines += _bucket_table(manifest)
    lines += ["", "## How a Task becomes trusted", "", FUNNEL_PROSE, ""]
    lines += ["## Fetch and run", "", "```bash", f"uv run kullback fetch {repo_id} --out env-{name}",
              f"uv run kullback run --workdir env-{name} --task <task id> --model provider/model",
              f"uv run kullback verdict --workdir env-{name}",
              f"uv run kullback report --workdir env-{name}", "```", "",
              "`fetch` checks the package against the content hash above before laying it out, and refuses "
              "one that does not match. `--revision round-<n>` fetches an earlier round.", ""]
    lines += ["## Source", "",
              f"- Corpus: {corpus}", f"- Corpus licence: {license_name}"]
    if source.get("url"):
        lines.append(f"- Corpus source: {source['url']}")
    lines += [f"- Harness: kullback {manifest.get('kullback_version', 'unknown')}, "
              f"git {_short(manifest.get('git_sha'))}, runner {_short(manifest.get('runner_version'))}, "
              f"gates {_short(manifest.get('gates_version'))}",
              f"- Environment id: {_short(manifest.get('env_id'))}",
              f"- Built at: {manifest.get('created_at', 'not recorded')}"]
    lines += _limits(manifest)
    lines += ["", "## Licence", "",
              f"The harness and this package are Apache-2.0. The source corpus keeps its own licence "
              f"({license_name}).", ""]
    return "\n".join(lines) + "\n"


def organisation_card(rows: Iterable[dict], *, organisation: str = "leibler",
                      github_url: str = "https://github.com/leiblerdev/kullback",
                      site_url: str = "https://leibler.dev") -> str:
    """The organisation profile: what Leibler is, what Kullback is, and every Environment with its numbers.

    A row is `{name, repo_id, manifest}`, so this page and each Environment's own card cannot
    disagree about a number.
    """
    lines = [f"# {organisation}", ""]
    lines += ["Leibler turns the traces a working agent already produced into an executable copy of the "
              "system it worked in, then grades other models on what they change in that copy. The graders "
              f"are code, so they are cheap and have no opinions. [leibler.dev]({site_url})", ""]
    lines += [f"[Kullback]({github_url}) is the open-source Builder and Runner behind it, under Apache-2.0. "
              "It rebuilds the world from traces, checks the rebuild by replaying them, derives a Verifier "
              "per Task from the recorded runs, and puts every artifact through a code gate no model may "
              "touch. The Environments below came out of it. Each one carries its own numbers and says what "
              "it cannot do yet.", ""]
    lines += ["Everything here is under active development. Environments are republished after each build "
              "round, and earlier rounds stay reachable by their tags. The next stage is to raise the trusted "
              "count on each Environment, then to generate Tasks synthetically over the rebuilt world.", ""]
    lines += ["## Environments", "",
              "| Environment | Replay fidelity | Trusted Tasks | Round | Status |",
              "| --- | --- | --- | --- | --- |"]
    for row in rows:
        manifest = row.get("manifest") or {}
        fidelity = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
        repo_id = row.get("repo_id", "")
        status = "preview" if manifest.get("preview") else "release"
        lines.append(f"| [{row.get('name', repo_id)}](https://huggingface.co/datasets/{repo_id}) | "
                     f"{_pct(fidelity)} | {manifest.get('trusted', 0)} of {manifest.get('tasks_total', 0)} | "
                     f"{manifest.get('round', 'not recorded')} | {status} |")
    lines += ["", "A release replays at least 90% of its Tasks. A preview is below that bar. It is published "
                  "anyway, with its numbers on its card, so the work stays visible while it improves.", ""]
    lines += ["## Fetch and run", "", "```bash",
              "uv pip install git+" + github_url + ".git",
              f"uv run kullback fetch {organisation}/<environment> --out env",
              "uv run kullback run --workdir env --task <task id> --model provider/model",
              "uv run kullback verdict --workdir env",
              "uv run kullback report --workdir env", "```", ""]
    lines += ["## Links", "", f"- Harness: {github_url}", f"- Site: {site_url}",
              "- Licence: Apache-2.0 for the harness and every package here. Each source corpus keeps its "
              "own licence, named on the Environment's card.", ""]
    return "\n".join(lines) + "\n"


def environment_card_template(fields: Optional[Iterable[tuple[str, str]]] = None) -> str:
    """What every field of an Environment card means and where its value comes from."""
    rows = list(fields) if fields is not None else CARD_FIELDS
    lines = ["# Environment card fields", "",
             "Every card under the organisation is rendered by `kullback/hub/card.py` from the package's "
             "`manifest.json` and from nothing else, so a number on a page is a number the package carries. "
             "This is what each field means and which record it is read from. Nothing on a card is written "
             "by hand, and nothing on it quotes a Task, a tool result or a status row: every reason a card "
             "gives is one of the harness's fixed phrases.", "",
             "| Field | Meaning | Read from |", "| --- | --- | --- |"]
    lines += [f"| {name} | {meaning} | {source} |" for name, meaning, source in rows]
    lines += ["", "## Front matter", "",
              "`license` is the source corpus's licence, lowercased to the id a dataset host indexes, or "
              f"`{UNKNOWN_LICENSE}` where the publisher named none. `tags` is always "
              f"{', '.join('`' + tag + '`' for tag in BASE_TAGS)} followed by the Environment's own name, "
              "which is the domain.", "",
              "## Preview and release", "",
              f"A release requires replay fidelity over Tasks at or above {FIDELITY_BAR:.0%}. Below that, "
              "`publish --preview` is the only form that is allowed, it sets `preview: true` in the "
              "manifest, and the card opens with a banner naming the bar and the Environment's own "
              "numbers.", ""]
    return "\n".join(lines) + "\n"


CARD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("name", "The Environment's name, which is also its domain tag", "`--name`, or the last segment of `--repo`"),
    ("preview", "Whether it is below the fidelity bar", "`--preview` on the publish"),
    ("round", "The build round the numbers were measured in", "the last record in `rounds.json`"),
    ("tasks_total", "Tasks on the frozen list", "`tasks_frozen.json`, or the newest round snapshot"),
    ("replay_fidelity.tasks_rate", "Share of Tasks with at least one confirmed replay", "`replays.json`"),
    ("replay_fidelity.runs_rate", "Share of replayed Runs that were confirmed", "`replays.json`"),
    ("reference_confirmed", "Tasks whose recordings agreed on an End state", "`task_status.json`"),
    ("verifier_derived", "Tasks with a Verifier on disk", "`verifiers/`"),
    ("trusted", "Tasks whose Verifier passed the whole suite", "the last round's trusted ruling"),
    ("refused", "Tasks the harness ruled nobody finished", "the last round's refuse ruling"),
    ("funnel", "How many Tasks stopped at each rung", "the per-Task index the export writes"),
    ("buckets", "Tasks and trusted Tasks per difficulty bucket", "`difficulty.json` (D209), else computed"),
    ("untrusted", "Untrusted count and the commonest fixed reasons", "the per-Task index"),
    ("runner_version", "Hash of the frozen Runner the numbers were measured under", "`runner_version.json`"),
    ("gates_version", "Hash of the gates package", "`runner_version.json`"),
    ("kullback_version", "Harness version", "the installed distribution"),
    ("git_sha", "Commit the export ran from", "`git rev-parse HEAD`"),
    ("leak_scan", "What the export checked against the source corpus, in counts", "the export's own scan"),
    ("content_hash", "One hash over every file of the package", "computed at export"),
    ("files", "sha256 per file, which is what a fetch verifies", "computed at export"),
)
