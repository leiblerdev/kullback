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


def _built_at(manifest: dict) -> str:
    """The commit the package was exported at and, where another commit laid its Harbor Tasks (a publish
    from a staged package), that one too: the `kullback_git_sha` every shipped `task.toml` carries."""
    package_sha = manifest.get("git_sha")
    laid = (manifest.get("harbor") or {}).get("git_sha")
    if laid and laid != package_sha:
        return f"package built at git {_short(package_sha)}, Harbor Tasks laid at git {_short(laid)}"
    return f"git {_short(package_sha)}"


def _short(value: Any) -> str:
    text = str(value or "")
    return (text[:16] if len(text) > 16 else text) or "not recorded"


def _one_line(value: Any) -> str:
    """Publisher or corpus text on one line: a line break folded to a space."""
    return " ".join(str(value).splitlines())


def _corpus(manifest: dict) -> str:
    source = manifest.get("source") or {}
    corpus = f"{_one_line(source.get('corpus') or 'not stated')} ({_one_line(source.get('license') or 'not stated')})"
    return f"{corpus}, {source['url']}" if source.get("url") else corpus


def _status(manifest: dict) -> str:
    """The replay fidelity over Tasks and whether that makes the package a release or a preview."""
    fidelity = _share(manifest.get("replay_fidelity") or {}, "tasks")
    if manifest.get("preview"):
        return f"{fidelity}: a preview, below the {FIDELITY_BAR:.0%} replay fidelity a release needs"
    return f"{fidelity}: a release, which needs at least {FIDELITY_BAR:.0%}"


# Said once on every card, because a Task's Verifier is a model's reading of the ask, not a ground truth.
DISCLAIMER = ("Some Tasks can be wrong. A Verifier can check the wrong thing or an untrusted Task can be "
              "unfinishable. Treat a single Task's result with care and the set as a whole as an eval.")


def card_markdown(manifest: dict, repo_id: str, *, github_url: str = "https://github.com/leiblerdev/kullback",
                  site_url: str = "https://leibler.dev", harbor: Optional[dict] = None) -> str:
    """The dataset card of one Environment: what it is, what was done, how to use it, why it matters and
    what built it, every number read off its manifest; `harbor`, the Harbor dataset's manifest where one
    was laid, adds how to run it with Harbor. Everything else stays in `manifest.json`."""
    name = manifest.get("name") or repo_id.rsplit("/", 1)[-1]
    lines = [front_matter(manifest, tags=[str(name)]), ""]
    lines += [f"# {name}", ""]
    lines += [f"An executable Environment for tool-using agents, rebuilt from traces by [Kullback]({github_url}) "
              f"and published by [Leibler]({site_url}): the world, the Tasks and a code Verifier per Task, no "
              f"recordings. It has {int(manifest.get('tasks_total') or 0)} Tasks. Source corpus: {_corpus(manifest)}.",
              ""]
    lines += ["## What we did", "",
              "Kullback rebuilt the world and its tools from the recorded conversations and replayed the "
              f"recordings through it; fidelity over Tasks is {_status(manifest)}. Each Verifier is derived "
              "from the user's ask in the recording and the policy, and checks the end state in code. A Task "
              "is trusted only when its Verifier clears every gate: oracle replay, suite, loosening, false "
              f"rejection and the trust ruling. {DISCLAIMER}", ""]
    lines += ["## How to use it", "", "```bash", f"uv run kullback fetch {repo_id} --out env-{name}",
              f"uv run kullback run --workdir env-{name} --task <task id> --model provider/model",
              f"uv run kullback verdict --workdir env-{name}",
              f"uv run kullback report --workdir env-{name}", "```", "",
              "`fetch` checks the content hash; `--revision <tag>` fetches an earlier publish. "
              "`tasks.jsonl` lists the Tasks. Untrusted Tasks are not graded, and the simulated user does not "
              "ship, so a fetched package cannot finish a Task that needs a fact the user gives mid conversation.",
              ""]
    if harbor is not None:
        lines += harbor_section(harbor)
    lines += ["## Why it matters", "",
              "A working agent's traces are enough to rebuild the system it ran in. Another model is then graded "
              "in code on what it changes there, not on how its answer reads.", ""]
    lines += [f"Built: {manifest.get('tag') or 'tag not recorded'}, {_built_at(manifest)}, "
              f"{manifest.get('created_at', 'time not recorded')}. Content hash: "
              f"{_short(manifest.get('content_hash'))}.", ""]
    return "\n".join(lines) + "\n"


# A Task whose only leaking part is its oracle (D399): left out of the Harbor form, the dataset not refused.
SOLUTION_LEAKS = "solution leaks"


def harbor_section(harbor: dict) -> list[str]:
    """How to run the Tasks with Harbor, read off the Harbor dataset's manifest (D363)."""
    counts = harbor.get("counts") or {}
    rows = list(harbor.get("tasks") or ()) + list(harbor.get("synthetic") or ())
    example = rows[0]["path"] if rows else "<task>"
    lines = [f"`{HARBOR_DIR}/` holds {counts.get('exported', 0)} Tasks in the [Harbor](https://harborframework.com) "
             f"task format; a judge item is scored at test time by the model `{JUDGE_VAR}` names, with "
             f"`{LIVE_ENV_VAR}=1` and the model's keys, and scores 0 without one.", ""]
    no_oracle = int(counts.get("no_oracle") or 0)
    if no_oracle:
        lines += [f"{no_oracle} of them ship without `solution/`, so Harbor's oracle agent has nothing to run on "
                  "them.", ""]
    lines += ["```bash", f"uv run harbor run -p {HARBOR_DIR}/{example} -a oracle",
              f"uv run harbor run -p {HARBOR_DIR}/{example} -a <agent> -m <provider/model>", "```", ""]
    return lines


# The few terms a reader of the organisation card needs, in plain words; the README keeps the harness's own.
TERMS: tuple[str, ...] = (
    "Environment: a rebuilt system an agent works in, its data, tools and policy, packaged with Tasks.",
    "Task: one thing a user asked for, with what the agent is given to start.",
    "Trace: a recorded conversation between a working agent, its user and the tools.",
    "Verifier: a Task's check, in code, of the state the agent leaves behind.",
    f"Release and preview: a release replays at least {FIDELITY_BAR:.0%} of its Tasks as recorded; a preview is "
    "below that.",
    f"Harbor format: the same Tasks under `{HARBOR_DIR}/`, laid out to run with the Harbor CLI.",
)


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
              "| Environment | Tasks | Harbor Tasks | Tag | Status |",
              "| --- | --- | --- | --- | --- |"]
    for row in rows:
        manifest = row.get("manifest") or {}
        repo_id = row.get("repo_id", "")
        harbor = manifest.get("harbor")
        laid = ((harbor.get("counts") or {}).get("exported", 0)) if isinstance(harbor, dict) else "none"
        status = "preview" if manifest.get("preview") else "release"
        lines.append(f"| [{_cell(repo_id)}](https://huggingface.co/datasets/{_cell(repo_id)}) | "
                     f"{manifest.get('tasks_total', 0)} | {laid} | {_cell(manifest.get('tag') or 'not recorded')} | "
                     f"{status} |")
    lines += ["", "## Terms", ""]
    lines += [f"- {term}" for term in TERMS]
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
    """What every field an Environment card reads means and where its value comes from."""
    rows = list(fields) if fields is not None else CARD_FIELDS
    lines = ["# Environment card fields", "",
             "`kullback/hub/card.py` renders every card from the package's `manifest.json` alone, in five parts: "
             "what this is, what we did (with a disclaimer that some Tasks can be wrong), how to use it, why it "
             "matters, and a Built line. These are the fields the card reads; every other number, the trusted "
             "count among them, stays in `manifest.json` and `tasks_index.json`.", "",
             "| Field | Meaning | Read from |", "| --- | --- | --- |"]
    lines += [f"| {name} | {meaning} | {source} |" for name, meaning, source in rows]
    lines += ["", "## Front matter", "",
              f"`license` is the corpus licence as the host's id, or `{UNKNOWN_LICENSE}` when none was named; "
              f"`tags` is {', '.join('`' + tag + '`' for tag in BASE_TAGS)} and the Environment's name.", "",
              "## Preview and release", "",
              f"A release needs fidelity over Tasks of at least {FIDELITY_BAR:.0%}; below that only "
              "`publish --preview` is allowed, and the card says so where it gives the fidelity.", "",
              "## Harbor", "",
              f"Where the package carries a Harbor form, the card reads `{HARBOR_DIR}/manifest.json` for the "
              "exported count, the Tasks without an oracle and an example Task path.", ""]
    return "\n".join(lines) + "\n"


CARD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("name", "The Environment's name, which is also its domain tag", "`--name`, or the last segment of `--repo`"),
    ("source", "The corpus, its licence and its URL", "`--corpus`, `--corpus-license`, `--corpus-url`"),
    ("tasks_total", "Tasks on the frozen list", "`tasks_frozen.json`, or the newest round snapshot"),
    ("replay_fidelity.tasks_rate", "Share of Tasks with at least one confirmed replay", "`replays.json`"),
    ("preview", "Whether it is below the fidelity bar", "`--preview` on the publish"),
    ("tag", "The tag this publish carries", "`round-<n>`, else `build-<YYYYMMDD>` of the newest session write"),
    ("git_sha", "Commit the export ran from; the card also names the Harbor form's where it differs",
     "`git rev-parse HEAD`"),
    ("created_at", "When the package was exported", "the export's clock, UTC"),
    ("content_hash", "One hash over every file of the package, which a fetch checks", "computed at export"),
)
