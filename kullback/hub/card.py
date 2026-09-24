"""The cards: one per Environment, one for the organisation, and the template that documents both (D221).

A card is rendered from a manifest and nothing else, so a number on a page is a number the package
carries. Every reason a card gives is one of the harness's fixed phrases from package.py, never a
quoted record.
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


def _share(fidelity: dict, key: str) -> str:
    return f"{_pct(fidelity.get(key + '_rate'))} ({fidelity.get(key, 0)} of {fidelity.get(key + '_total', 0)})"


def _numbers_table(manifest: dict) -> list[str]:
    """Every number the manifest carries, in one table; a value nobody recorded gets no row."""
    fidelity = manifest.get("replay_fidelity") or {}
    rows = [
        ("Fidelity over Tasks", _share(fidelity, "tasks")),
        ("Fidelity over Runs", _share(fidelity, "runs")),
        ("Reference confirmed", str(manifest.get("reference_confirmed", 0))),
        ("Verifier derived", str(manifest.get("verifier_derived", 0))),
        ("Trusted", str(manifest.get("trusted", 0))),
        ("Refused", str(manifest.get("refused", 0))),
        _untrusted_row(manifest),
    ]
    rows += _coverage_rows(manifest) + _provenance_rows(manifest)
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _untrusted_row(manifest: dict) -> tuple[str, str]:
    untrusted = manifest.get("untrusted") or {}
    reasons = "".join(f"; {entry.get('tasks', 0)} {entry.get('reason', '')}" for entry in untrusted.get("reasons") or ())
    return "Not trusted", f"{untrusted.get('count', 0)} of {int(manifest.get('tasks_total') or 0)}{reasons}"


def _coverage_rows(manifest: dict) -> list[tuple[str, str]]:
    """Difficulty, the domain reading, stand-in tools and late Tasks, each only where there is one."""
    rows = []
    buckets = list(manifest.get("buckets") or ())
    if buckets:
        rows.append(("Tasks and trusted by difficulty", ", ".join(
            f"{row.get('bucket', '')} {row.get('tasks', 0)} and {row.get('trusted', 0)}" for row in buckets)))
    rows += _domain_rows(manifest)
    assisted = sorted(manifest.get("assisted_tools") or ())
    if assisted:
        rows.append(("Tools served by a stand-in", f"{len(assisted)} ({', '.join(assisted)})"))
    added = int(manifest.get("tasks_added_later") or 0)
    if added:
        rows.append(("Tasks added after the freeze, not counted", str(added)))
    return rows


def _provenance_rows(manifest: dict) -> list[tuple[str, str]]:
    """The leak scan, where the counts came from, the corpus and the build that made the package."""
    leaks = manifest.get("leak_scan") or {}
    rows = [("Leak scan", f"{leaks.get('leaks', 0)} recorded strings, {leaks.get('value_echoes', 0)} short "
                          f"values, {leaks.get('values_checked', 0)} checked in {leaks.get('files_scanned', 0)} "
                          f"files against {leaks.get('corpus_strings', 0)}")]
    if manifest.get("round") is not None:
        rows.append(("Round", str(manifest["round"])))
    rows.append(("Tag", f"{manifest.get('tag') or 'not recorded'}, "
                        f"{manifest.get('counts_source') or 'counts from the last round record'}"))
    source = manifest.get("source") or {}
    corpus = f"{source.get('corpus') or 'not stated'} ({source.get('license') or 'not stated'})"
    rows.append(("Corpus", f"{corpus}, {source['url']}" if source.get("url") else corpus))
    build = [f"kullback {manifest.get('kullback_version', 'unknown')}", f"git {_short(manifest.get('git_sha'))}"]
    build += [f"{label} {_short(manifest[key])}" for label, key in
              (("runner", "runner_version"), ("gates", "gates_version"), ("Environment", "env_id"))
              if manifest.get(key)]
    rows += [("Built", f"{manifest.get('created_at', 'not recorded')} by {', '.join(build)}"),
             ("Content hash", _short(manifest.get("content_hash")))]
    return rows


def _domain_rows(manifest: dict) -> list[tuple[str, str]]:
    """What the domain's own public material attests, and how much of it this package covers (D225).

    Only where a domain was read. The second row says what the Environment cannot be asked to do.
    Neither row is a Task count and neither is added to one.
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


def _status(manifest: dict) -> str:
    if manifest.get("preview"):
        return f"Preview: below the {FIDELITY_BAR:.0%} replay fidelity a release needs."
    return f"Release: replays at least {FIDELITY_BAR:.0%} of its Tasks."


def card_markdown(manifest: dict, repo_id: str, *, github_url: str = "https://github.com/leiblerdev/kullback",
                  site_url: str = "https://leibler.dev") -> str:
    """The dataset card of one Environment, every number read off its manifest."""
    name = manifest.get("name") or repo_id.rsplit("/", 1)[-1]
    lines = [front_matter(manifest, tags=[str(name)]), ""]
    lines += [f"# {name}", ""]
    lines += [f"An executable Environment for tool-using agents, rebuilt from traces by [Kullback]({github_url}) "
              f"and published by [Leibler]({site_url}): the world, the Tasks and a code Verifier per Task, no "
              f"recordings. `{TASKS_ROWS_NAME}` lists the Tasks.", ""]
    lines += [_status(manifest), ""]
    lines += _numbers_table(manifest)
    lines += ["", "Untrusted Tasks are not graded, and the simulated user does not ship.", ""]
    lines += ["```bash", f"uv run kullback fetch {repo_id} --out env-{name}",
              f"uv run kullback run --workdir env-{name} --task <task id> --model provider/model",
              f"uv run kullback verdict --workdir env-{name}",
              f"uv run kullback report --workdir env-{name}", "```", "",
              "`fetch` checks the content hash; `--revision round-<n>` fetches an earlier round.", ""]
    return "\n".join(lines) + "\n"


# The harness's own words, one line each, the same lines the README carries.
WORDS: tuple[str, ...] = (
    "Run and Trace: a Run is an agent's conversation with the tools; a Trace is a recorded one.",
    "Reference: the Trace a Task's user context comes from.",
    "Replay and agree: a replay re-drives a Trace's turns; a call agrees when its verdict is same, cosmetic or "
    "both refused.",
    "Confirmed: a replay where every call agrees and nothing is missing, reordered or crashed.",
    "Fidelity over Tasks: Tasks with a confirmed replay, over all Tasks.",
    "Fidelity over Runs: confirmed replays over all replays.",
    "Call fidelity: agreeing calls over all recorded calls.",
    "Verifier: a Task's End-state check, written only by the Examiner, never by the Builder.",
    "Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.",
    "Gates: oracle replay, suite, loosening, false rejection, trusted.",
    "Trusted: suite passed, probes fail, last version, no loosening, not over strict, not refused.",
    "Open: not yet trusted.",
    "Refused: a Task the Builder showed nobody can finish.",
    "Release and preview: a release replays at least 90% of its Tasks; a preview is below that.",
)


def _call_fidelity(manifest: dict) -> str:
    calls = manifest.get("call_fidelity") or {}
    if not isinstance(calls, dict) or calls.get("rate") is None:
        return "not measured"
    return f"{float(calls['rate']):.2%} of {calls.get('calls', 0)}"


def organisation_card(rows: Iterable[dict], *, organisation: str = "leibler",
                      github_url: str = "https://github.com/leiblerdev/kullback",
                      site_url: str = "https://leibler.dev") -> str:
    """The organisation profile: what Leibler and Kullback are, and one row per Environment passed.

    A row is `{name, repo_id, manifest}`, so this page and each Environment's own card cannot
    disagree about a number.
    """
    lines = [f"# {organisation}", ""]
    lines += [f"[Leibler]({site_url}) rebuilds the system a working agent ran in from its traces, then grades "
              f"other models in code on what they change there. [Kullback]({github_url}) is the open-source "
              "Builder and Runner behind it, under Apache-2.0.", ""]
    lines += ["## Environments", "",
              "| Environment | Fidelity over Tasks | Call fidelity | Verifiers | Trusted | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        manifest = row.get("manifest") or {}
        fidelity = manifest.get("replay_fidelity") or {}
        repo_id = row.get("repo_id", "")
        status = "preview" if manifest.get("preview") else "release"
        lines.append(f"| [{repo_id}](https://huggingface.co/datasets/{repo_id}) | {_share(fidelity, 'tasks')} | "
                     f"{_call_fidelity(manifest)} | {manifest.get('verifier_derived', 0)} | "
                     f"{manifest.get('trusted', 0)} of {manifest.get('tasks_total', 0)} | {status} |")
    lines += ["", "## Words", ""]
    lines += [f"- {word}" for word in WORDS]
    lines += ["", "## Fetch and run", "", "```bash",
              "uv pip install git+" + github_url + ".git",
              f"uv run kullback fetch {organisation}/<environment> --out env",
              "uv run kullback run --workdir env --task <task id> --model provider/model",
              "uv run kullback verdict --workdir env",
              "uv run kullback report --workdir env", "```", ""]
    lines += ["## Links", "", f"- Harness: {github_url}", f"- Site: {site_url}",
              "- Licence: Apache-2.0 for the harness and every package; each corpus keeps its own.", ""]
    return "\n".join(lines) + "\n"


def environment_card_template(fields: Optional[Iterable[tuple[str, str]]] = None) -> str:
    """What every field of an Environment card means and where its value comes from."""
    rows = list(fields) if fields is not None else CARD_FIELDS
    lines = ["# Environment card fields", "",
             "`kullback/hub/card.py` renders every card from the package's `manifest.json` alone, "
             "and this is what each field means and where it is read from.", "",
             "| Field | Meaning | Read from |", "| --- | --- | --- |"]
    lines += [f"| {name} | {meaning} | {source} |" for name, meaning, source in rows]
    lines += ["", "## Front matter", "",
              f"`license` is the corpus licence as the host's id, or `{UNKNOWN_LICENSE}` when none was named; "
              f"`tags` is {', '.join('`' + tag + '`' for tag in BASE_TAGS)} and the Environment's name.", "",
              "## Preview and release", "",
              f"A release needs fidelity over Tasks of at least {FIDELITY_BAR:.0%}; below that only "
              "`publish --preview` is allowed, and the card says so above its table.", ""]
    return "\n".join(lines) + "\n"


CARD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("name", "The Environment's name, which is also its domain tag", "`--name`, or the last segment of `--repo`"),
    ("preview", "Whether it is below the fidelity bar", "`--preview` on the publish"),
    ("round", "The build round the numbers were measured in", "the last record in `rounds.json`"),
    ("tasks_total", "Tasks on the frozen list", "`tasks_frozen.json`, or the newest round snapshot"),
    ("replay_fidelity.tasks_rate", "Share of Tasks with at least one confirmed replay", "`replays.json`"),
    ("replay_fidelity.runs_rate", "Share of replayed Runs that were confirmed", "`replays.json`"),
    ("tag", "The tag this publish carries", "`round-<n>`, else `build-<YYYYMMDD>` of the newest session write"),
    ("counts_source", "Where the counts came from", "the last round record, else the workdir status"),
    ("reference_confirmed", "Tasks with at least one confirmed replay", "`replays.json`"),
    ("verifier_derived", "Tasks with a Verifier on disk", "`verifiers/`"),
    ("trusted", "Tasks whose Verifier passed the whole suite",
     "the last round's trusted ruling, else the Builder's status rule"),
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
