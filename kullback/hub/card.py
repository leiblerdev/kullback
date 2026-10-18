"""The cards: one per Environment, one for the organisation, and the template that documents both (D221).

A card is rendered from a manifest and nothing else, so a number on a page is a number the package
carries. Every reason a card gives is one of the harness's fixed phrases from package.py, never a
quoted record.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.ai.provider import LIVE_ENV_VAR
from kullback.hub.harbor_world import JUDGE_VAR
from kullback.hub.package import HARBOR_DIR, TASKS_ROWS_NAME

# What every Environment published from this harness is tagged with, before its own domain.
BASE_TAGS: tuple[str, ...] = ("kullback", "environment", "agent-evaluation", "rl-environment")
# The licence a card states when the caller named none for the source corpus. A host will not index
# an empty value, and claiming a permissive licence nobody checked would be worse than saying this.
UNKNOWN_LICENSE = "other"

# The organisation every Environment is published under unless a caller names another.
ORGANISATION = "leibler"

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


def _cell(value: Any) -> str:
    """A table cell safe for publisher or corpus text: a pipe escaped, a line break folded to a space."""
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _share(fidelity: dict, key: str) -> str:
    return f"{_pct(fidelity.get(key + '_rate'))} ({fidelity.get(key, 0)} of {fidelity.get(key + '_total', 0)})"


def _numbers_table(manifest: dict) -> list[str]:
    """Every number the manifest carries, in one table; a value nobody recorded gets no row."""
    fidelity = manifest.get("replay_fidelity") or {}
    rows = [
        ("Fidelity over Tasks", _share(fidelity, "tasks")),
        ("Fidelity over Runs", _share(fidelity, "runs")),
        ("Call fidelity", _call_fidelity(fidelity)),
        ("Reference confirmed", str(manifest.get("reference_confirmed", 0))),
        ("Verifier derived", str(manifest.get("verifier_derived", 0))),
        ("Trusted", str(manifest.get("trusted", 0))),
        ("Refused", str(manifest.get("refused", 0))),
        _untrusted_row(manifest),
    ]
    rows += _coverage_rows(manifest) + _provenance_rows(manifest)
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {_cell(value)} |" for name, value in rows]
    return lines


def _untrusted_row(manifest: dict) -> tuple[str, str]:
    """The untrusted count and its reasons, which sum to it: an older manifest that lists only the commonest
    reasons gets the rest as one remainder."""
    untrusted = manifest.get("untrusted") or {}
    entries = list(untrusted.get("reasons") or ())
    count = int(untrusted.get("count") or 0)
    reasons = "".join(f"; {entry.get('tasks', 0)} {entry.get('reason', '')}" for entry in entries)
    rest = count - sum(int(entry.get("tasks") or 0) for entry in entries)
    if rest > 0:
        reasons += f"; {rest} for reasons the manifest does not list"
    return "Not trusted", f"{count} of {int(manifest.get('tasks_total') or 0)}{reasons}"


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
    build = [f"kullback {manifest.get('kullback_version', 'unknown')}", _built_at(manifest)]
    build += [f"{label} {_short(manifest[key])}" for label, key in
              (("code", "runner_version"), ("Environment", "env_id"))
              if manifest.get(key)]
    rows += [("Built", f"{manifest.get('created_at', 'not recorded')} by {', '.join(build)}"),
             ("Content hash", _short(manifest.get("content_hash")))]
    return rows


def _built_at(manifest: dict) -> str:
    """The commit the package was exported at and, where another commit laid its Harbor Tasks (a publish
    from a staged package), that one too: the `kullback_git_sha` every shipped `task.toml` carries."""
    package_sha = manifest.get("git_sha")
    laid = (manifest.get("harbor") or {}).get("git_sha")
    if laid and laid != package_sha:
        return f"package built at git {_short(package_sha)}, Harbor Tasks laid at git {_short(laid)}"
    return f"git {_short(package_sha)}"


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
                  site_url: str = "https://leibler.dev", announcement: Optional[str] = None) -> str:
    """The dataset card of one Environment, every number read off its manifest; `announcement`, the
    publisher's own words (`publish --announcement FILE`), goes under the status line as is (D399)."""
    name = manifest.get("name") or repo_id.rsplit("/", 1)[-1]
    lines = [front_matter(manifest, tags=[str(name)]), ""]
    lines += [f"# {name}", ""]
    lines += [f"An executable Environment for tool-using agents, rebuilt from traces by [Kullback]({github_url}) "
              f"and published by [Leibler]({site_url}): the world, the Tasks and a code Verifier per Task, no "
              f"recordings. `{TASKS_ROWS_NAME}` lists the Tasks.", ""]
    lines += [_status(manifest), ""]
    if announcement and announcement.strip():
        lines += ["## News", "", announcement.strip(), ""]
    lines += _numbers_table(manifest)
    lines += ["", "Untrusted Tasks are not graded, and the simulated user does not ship, so a fetched package "
              "cannot finish a Task that needs a fact the user gives mid conversation, though the Task "
              "still carries a Verifier.", ""]
    lines += ["```bash", f"uv run kullback fetch {repo_id} --out env-{name}",
              f"uv run kullback run --workdir env-{name} --task <task id> --model provider/model",
              f"uv run kullback verdict --workdir env-{name}",
              f"uv run kullback report --workdir env-{name}", "```", "",
              "`fetch` checks the content hash; `--revision round-<n>` fetches an earlier round.", ""]
    return "\n".join(lines) + "\n"


# What each Task set of a Harbor export holds, in the card's words (D399).
TASK_SETS = {"trusted": "the trusted Tasks", "verified": "every Task with a Verifier, trusted or not",
             "all": "every Task, the synthetic ones too"}
# A Task whose only leaking part is its oracle (D399): left out of the Harbor form, the dataset not refused.
SOLUTION_LEAKS = "solution leaks"
LABEL_ORDER = ("right", "partial", "wrong", "no_verifier", "unlabelled")


def _skipped(counts: dict) -> str:
    skipped = counts.get("skipped") or {}
    return ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items())) or "none"


def tier_label_table(harbor: dict) -> list[str]:
    """Tasks by tier (trusted, else the untrusted reason) and, where labels were given, by label (D399)."""
    table = (harbor.get("counts") or {}).get("by_tier_and_label") or {}
    if not table:
        return []
    labelled = harbor.get("labels") is not None
    seen = {label for row in table.values() for label in row}
    labels = [label for label in LABEL_ORDER if label in seen] + sorted(seen - set(LABEL_ORDER))
    columns = labels if labelled else []
    lines = ["| Tier | " + " | ".join(columns + ["Tasks"]) + " |", "| --- |" + " --- |" * (len(columns) + 1)]
    for tier in sorted(table, key=lambda cell: (cell != "trusted", cell)):
        row = table[tier]
        cells = [str(row.get(label, 0)) for label in columns] + [str(sum(row.values()))]
        lines.append(f"| {_cell(tier)} | " + " | ".join(cells) + " |")
    lines.append("")
    if labelled:
        rubric = (harbor.get("labels") or {}).get("rubric") or "not named"
        lines += [f"A label is one blind model rater's judgement of the Task's Verifier under the rubric "
                  f"{_cell(rubric)}, not a ground truth; `label` in a Task's `task.toml` carries it.", ""]
    return lines


def _solution_leak_lines(harbor: dict) -> list[str]:
    """The Tasks left out because their oracle repeats a recorded string, by masked id (D399)."""
    ids = [str(row.get("task")) for row in harbor.get("left_out") or () if row.get("reason") == SOLUTION_LEAKS]
    if not ids:
        return []
    return [f"Left out because their oracle would repeat a recorded string: {', '.join(ids)} (last four characters "
            "of the Task id).", ""]


def _no_oracle_lines(counts: dict) -> list[str]:
    """How many Tasks ship without solution/ because no Run passes on replay (D399)."""
    count = int(counts.get("no_oracle") or 0)
    if not count:
        return []
    return [f"{count} Tasks ship without `solution/`: no Run passes on replay through the exported world, so "
            "Harbor's oracle agent cannot run on them (`has_oracle = false` in their `task.toml`).", ""]


def harbor_section(harbor: dict) -> str:
    """The card's "Run it with Harbor" section, read off the Harbor dataset's manifest (D363)."""
    counts = harbor.get("counts") or {}
    chosen = str(harbor.get("tasks_chosen") or "trusted")
    rows = list(harbor.get("tasks") or ()) + list(harbor.get("synthetic") or ())
    example = rows[0]["path"] if rows else "<task>"
    lines = ["## Run it with Harbor", "",
             f"`{HARBOR_DIR}/` holds the same Tasks in the [Harbor](https://harborframework.com) task format: "
             f"{counts.get('exported', 0)} Tasks, {counts.get('needs_judge', 0)} of them with a judge item "
             f"among their gates. Those are scored by the model `{JUDGE_VAR}` names at test time, which also "
             f"needs `{LIVE_ENV_VAR}=1` and the model's keys; without a judge they score 0, never pass "
             "(`needs_judge` in a Task's `task.toml` and in `harbor/manifest.json`).", "",
             f"Task set: `{chosen}`, {TASK_SETS.get(chosen, chosen)}; {counts.get('exported', 0)} exported, "
             f"left out: {_skipped(counts)}. Each Task's `task.toml` carries its Spec tier as `spec_tier` "
             "and `spec_tier_reason`.", "",
             *tier_label_table(harbor),
             *_solution_leak_lines(harbor),
             *_no_oracle_lines(counts),
             "```bash", f"uv run harbor run -p {HARBOR_DIR}/{example} -a oracle",
             f"uv run harbor run -p {HARBOR_DIR}/{example} -a <agent> -m <provider/model>", "```", ""]
    return "\n".join(lines) + "\n"


# The harness's own words, one line each, the same lines the README carries.
WORDS: tuple[str, ...] = (
    "Run and Trace: a Run is an agent's conversation with the tools; a Trace is a recorded one.",
    "Reference: the Trace a Task's user context comes from.",
    "Replay and agree: a replay re-drives a Trace's turns; a call agrees when its verdict is same, cosmetic or "
    "both refused.",
    "Confirmed: a replay where every call agrees, none was answered by the stand-in, and nothing is "
    "missing, reordered or crashed.",
    "Fidelity over Tasks: Tasks with a confirmed replay, over all Tasks.",
    "Fidelity over Runs: confirmed replays over all replays.",
    "Call fidelity: agreeing calls over all recorded calls.",
    "Verifier: a Task's End-state check, written only by the Examiner, never by the Builder.",
    "Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.",
    "Reward: pass when every gate item holds (the sanity item, nothing else changed, is one); score 0 when a "
    "gate fails, else the weighted mean of every item, the gates holding at least half.",
    "Gates: oracle replay, suite, loosening, false rejection, trusted.",
    "Trusted: suite passed, probes fail, last version, no loosening, not over strict, not refused.",
    "Open: neither trusted nor refused.",
    "Refused: a Task the Builder showed nobody can finish.",
    "Release and preview: a release replays at least 90% of its Tasks; a preview is below that.",
)


def _call_fidelity(fidelity: dict) -> str:
    """Agreeing calls over every recorded call, read from the manifest's replay_fidelity."""
    if fidelity.get("calls_rate") is None:
        return "not measured"
    return f"{float(fidelity['calls_rate']):.2%} of {fidelity.get('calls_total', 0)}"


def organisation_card(rows: Iterable[dict], *, organisation: str = ORGANISATION,
                      github_url: str = "https://github.com/leiblerdev/kullback",
                      site_url: str = "https://leibler.dev", announcement: Optional[str] = None) -> str:
    """The organisation profile: what Leibler and Kullback are, the publisher's News where given, and one
    row per Environment passed.

    A row is `{name, repo_id, manifest}`, so this page and each Environment's own card cannot
    disagree about a number.
    """
    lines = [f"# {organisation}", ""]
    lines += [f"[Leibler]({site_url}) rebuilds the system a working agent ran in from its traces, then grades "
              f"other models in code on what they change there. [Kullback]({github_url}) is the open-source "
              "Builder and Runner behind it, under Apache-2.0.", ""]
    if announcement and announcement.strip():
        lines += ["## News", "", announcement.strip(), ""]
    lines += ["## Environments", "",
              "| Environment | Fidelity over Tasks | Call fidelity | Verifiers | Trusted | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        manifest = row.get("manifest") or {}
        fidelity = manifest.get("replay_fidelity") or {}
        repo_id = row.get("repo_id", "")
        status = "preview" if manifest.get("preview") else "release"
        lines.append(f"| [{_cell(repo_id)}](https://huggingface.co/datasets/{_cell(repo_id)}) | "
                     f"{_share(fidelity, 'tasks')} | {_call_fidelity(fidelity)} | {manifest.get('verifier_derived', 0)} | "
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
    ("replay_fidelity.calls_rate", "Share of recorded calls that agree, over `calls_total`", "`replays.json`"),
    ("tag", "The tag this publish carries", "`round-<n>`, else `build-<YYYYMMDD>` of the newest session write"),
    ("counts_source", "Where the counts came from", "the last round record, else the workdir status"),
    ("reference_confirmed", "Tasks with at least one confirmed replay", "`replays.json`"),
    ("verifier_derived", "Tasks with a Verifier on disk", "`verifiers/`"),
    ("trusted", "Tasks whose Verifier passed the whole suite",
     "the last round's trusted ruling, else the Builder's status rule"),
    ("refused", "Tasks the harness ruled nobody finished", "the last round's refuse ruling"),
    ("funnel", "How many Tasks stopped at each rung", "the per-Task index the export writes"),
    ("buckets", "Tasks and trusted Tasks per difficulty bucket", "`difficulty.json` (D209), else computed"),
    ("untrusted", "Untrusted count and every fixed reason, summing to it", "the per-Task index"),
    ("runner_version", "Hash of the Runner, gates and Spec code the numbers were measured under",
     "`runner_version.json`"),
    ("kullback_version", "Harness version", "the installed distribution"),
    ("git_sha", "Commit the export ran from; the card also names the Harbor form's where it differs",
     "`git rev-parse HEAD`"),
    ("leak_scan", "What the export checked against the source corpus, in counts", "the export's own scan"),
    ("content_hash", "One hash over every file of the package", "computed at export"),
    ("files", "sha256 per file, which is what a fetch verifies", "computed at export"),
)
