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
    lines += ["---"]
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "not measured" if value is None else f"{float(value):.1%}"


def _numbers_table(manifest: dict) -> list[str]:
    fidelity = manifest.get("replay_fidelity") or {}
    total = int(manifest.get("tasks_total") or 0)
    rows = [
        ("Tasks", str(total)),
        ("Replay fidelity, Tasks", f"{_pct(fidelity.get('tasks_rate'))} "
                                   f"({fidelity.get('tasks', 0)} of {fidelity.get('tasks_total', 0)})"),
        ("Replay fidelity, Runs", f"{_pct(fidelity.get('runs_rate'))} "
                                  f"({fidelity.get('runs', 0)} of {fidelity.get('runs_total', 0)})"),
        ("Reference confirmed", str(manifest.get("reference_confirmed", 0))),
        ("Verifier derived", str(manifest.get("verifier_derived", 0))),
        ("Trusted Tasks", str(manifest.get("trusted", 0))),
        ("Refused Tasks", str(manifest.get("refused", 0))),
        ("Round", str(manifest.get("round") if manifest.get("round") is not None else "not recorded")),
        ("Runner version", _short(manifest.get("runner_version"))),
        ("Gates version", _short(manifest.get("gates_version"))),
        ("Environment id", _short(manifest.get("env_id"))),
        ("Content hash", _short(manifest.get("content_hash"))),
    ]
    lines = ["| Number | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _short(value: Any) -> str:
    text = str(value or "")
    return (text[:16] if len(text) > 16 else text) or "not recorded"


def _bucket_table(manifest: dict) -> list[str]:
    rows = list(manifest.get("buckets") or ())
    if not rows:
        return []
    lines = ["", "### Difficulty buckets", "",
             "A bucket names the Task's writes, the tools its Reference called and the paths to its "
             "End state, each banded, so the same bucket means the same thing on any corpus.", "",
             "| Bucket | Tasks | Trusted |", "| --- | --- | --- |"]
    lines += [f"| {row.get('bucket', '')} | {row.get('tasks', 0)} | {row.get('trusted', 0)} |" for row in rows]
    return lines


def _limits(manifest: dict) -> list[str]:
    untrusted = manifest.get("untrusted") or {}
    lines = ["", "## Known limits", ""]
    lines.append(f"- {untrusted.get('count', 0)} of {manifest.get('tasks_total', 0)} Tasks are not trusted: "
                 "their Verifier is not one the harness will grade a candidate on yet.")
    for entry in untrusted.get("reasons") or ():
        lines.append(f"- {entry.get('tasks', 0)} Tasks: {entry.get('reason', '')}.")
    assisted = list(manifest.get("assisted_tools") or ())
    if assisted:
        lines.append(f"- {len(assisted)} tools were served by a stand-in somewhere in the build "
                     f"({', '.join(sorted(assisted))}); a Run that touches one is reported, never counted.")
    added = int(manifest.get("tasks_added_later") or 0)
    if added:
        lines.append(f"- {added} further Tasks appeared after the Task list was frozen. They are outside "
                     "every number here, because the denominator a build is measured against is fixed "
                     "once and never moved.")
    lines.append("- The Simulated user is not in this package. Its facts are read off the recordings, "
                 "which do not ship, so a Task whose answer the user only gives mid conversation cannot "
                 "be reached by a candidate here even though its Verifier still grades it.")
    leaks = manifest.get("leak_scan") or {}
    lines.append(f"- The export checked {leaks.get('values_checked', 0)} strings over "
                 f"{leaks.get('files_scanned', 0)} graded files against the "
                 f"{leaks.get('corpus_strings', 0)} the source corpus holds. It found "
                 f"{leaks.get('leaks', 0)} recorded strings, which would have stopped the export, and "
                 f"{leaks.get('value_echoes', 0)} shorter values that only a Verifier's answer key "
                 "accounts for.")
    return lines


def _banner(manifest: dict) -> list[str]:
    if not manifest.get("preview"):
        return []
    fidelity = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
    return ["", f"> **Preview.** This Environment is below the {FIDELITY_BAR:.0%} replay fidelity bar this "
                f"harness publishes a release at. It replays {_pct(fidelity)} of its Tasks and holds "
                f"{manifest.get('trusted', 0)} trusted Tasks of {manifest.get('tasks_total', 0)}. Read the "
                "numbers below before using it for anything: it is here so the numbers are public while it "
                "is improved, not because it is finished.", ""]


FUNNEL_PROSE = (
    "Every Task starts as a cluster of recordings and climbs a funnel, and each rung is a code gate, "
    "never an opinion. It clears replay fidelity when the rebuilt tools answer its recorded calls the "
    "way the real ones did, keeps a Reference when the recordings agree on an End state, and gets a "
    "Verifier derived from that Reference. It is trusted only once that Verifier passes the full "
    "suite: it rejects an empty Run, a plausible wrong one and a mutated one, scores no pass on any "
    "loophole probe, accepts a second path to the same End state, and turns away few enough held-out "
    "Runs that reached the Reference. The counts on this page are that funnel, rung by rung, so a "
    "Task that stops early is visible instead of quietly leaving the denominator."
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
    lines += ["An executable Environment for evaluating and training tool-using agents, built by "
              f"[Kullback]({github_url}) from recorded traces of a working agent and published by "
              f"[Leibler]({site_url}).", ""]
    lines += ["It holds the rebuilt world (a database, one function per tool that behaves as the real tool "
              "was observed to behave, a compiled policy, and the Starting state each Task begins from), "
              "the Task list with the instruction a candidate is given, and a code-only Verifier per Task "
              "that grades the candidate on what it changed in that world. It holds none of the recordings "
              "it was built from.", ""]
    lines += _banner(manifest)
    lines += ["## Numbers", ""]
    lines += _numbers_table(manifest)
    lines += _bucket_table(manifest)
    lines += ["", "## The funnel", "", FUNNEL_PROSE, ""]
    lines += ["## Fetch and run", "", "```bash", f"uv run kullback fetch {repo_id} --out env-{name}",
              f"uv run kullback run --workdir env-{name} --task <task id> --model provider/model",
              f"uv run kullback verdict --workdir env-{name}",
              f"uv run kullback report --workdir env-{name}", "```", "",
              "`fetch` verifies the package against the content hash above before laying it out, and "
              "refuses a package that does not hash to it. Pass `--revision round-<n>` to fetch an "
              "earlier round instead of the newest one.", ""]
    lines += ["## Source", "",
              f"- Corpus: {corpus}", f"- Corpus licence: {license_name}"]
    if source.get("url"):
        lines.append(f"- Corpus source: {source['url']}")
    lines += [f"- Harness: kullback {manifest.get('kullback_version', 'unknown')}, "
              f"git {_short(manifest.get('git_sha'))}",
              f"- Built at: {manifest.get('created_at', 'not recorded')}",
              "- The card is rewritten every publish; older rounds stay reachable by their tags."]
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
    lines += [f"Leibler turns the traces a working agent already produced into an executable copy of the "
              f"system it worked in, and grades other models on what they change in that copy. The graders "
              f"are code, so they are cheap and have no opinions. [leibler.dev]({site_url})", ""]
    lines += [f"[Kullback]({github_url}) is the open-source Builder and Runner behind it, Apache-2.0. It "
              "rebuilds the world from traces, checks the rebuild by replaying those traces, derives a "
              "Verifier per Task from the recorded runs, and puts every artifact through a code gate no "
              "model may touch. The Environments below are what it produced; each carries its own numbers "
              "and says what it cannot do yet.", ""]
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
    lines += ["", "A release replays at least 90% of its Tasks. A preview is below that bar and is "
                  "published anyway, with its numbers on its card, so the work is visible while it "
                  "improves.", ""]
    lines += ["## Fetch and run", "", "```bash",
              "uv pip install git+" + github_url + ".git",
              f"uv run kullback fetch {organisation}/<environment> --out env",
              "uv run kullback run --workdir env --task <task id> --model provider/model",
              "uv run kullback verdict --workdir env",
              "uv run kullback report --workdir env", "```", ""]
    lines += ["## Links", "", f"- Harness: {github_url}", f"- Site: {site_url}",
              "- Licence: Apache-2.0 for the harness and every package here; each source corpus keeps "
              "its own licence, named on the Environment's card.", ""]
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
