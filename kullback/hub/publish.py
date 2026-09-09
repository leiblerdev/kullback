"""Publishing an Environment and fetching one back (D221).

`publish` exports the workdir into a staging directory, writes the card beside the package and
uploads the whole directory as one commit, then tags that commit `round-<n>` so an older publish
stays reachable after the card's numbers have moved. Publishing again is the same call: a new commit
on the same repository, a rewritten card, the old tags untouched.

The bar between a release and a preview is one number and one rule. Replay fidelity over Tasks at or
above the bar publishes as a release; below it, only `--preview` is allowed, the manifest says
`preview: true` and the card opens with a banner naming the bar and the Environment's own numbers.
The refusal is on the publish rather than on the export, because a package with honest numbers is
always worth writing; what needs a rule is calling one finished.

`fetch` downloads a revision, verifies every file against the manifest's hashes and the manifest
against its own content hash, and refuses a package that does not verify. What lands is a directory
`kullback run` takes as a workdir: the world, the Task list, the Verifiers and the manifest, and no
builder state at all.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from kullback.hub import package as package_mod
from kullback.hub.card import FIDELITY_BAR, card_markdown
from kullback.hub.client import HostedRepo, HubClient, hub_client, url_for
from kullback.hub.package import CARD_NAME
from kullback.runner.records import read_json

RELEASE_FIDELITY = FIDELITY_BAR

# What `kullback run` opens in a workdir before it can run a candidate. A fetch reports any of these
# the package did not carry, so a directory that will not run says so on arrival rather than at the
# first run.
RUN_INPUTS: tuple[str, ...] = ("environment.json", "schema.json", "tool_sigs.json", "bodies.json",
                               "env/db.json", "env/policy.md")


class PublishError(RuntimeError):
    """A publish that must not happen: below the bar without --preview, or a package that will not verify."""


def _fidelity(manifest: dict) -> Optional[float]:
    rate = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
    return None if rate is None else float(rate)


def release_ruling(manifest: dict, preview: bool, bar: float = RELEASE_FIDELITY) -> str:
    """`release` or `preview`, or the refusal in words.

    A fidelity nobody measured is not a fidelity above the bar: an Environment whose replays were
    never scored publishes as a preview or not at all.
    """
    if preview:
        return "preview"
    rate = _fidelity(manifest)
    if rate is None:
        raise PublishError("replay fidelity was not measured for this Environment: publish it with --preview")
    if rate < bar:
        raise PublishError(f"replay fidelity is {rate:.1%} over Tasks, below the {bar:.0%} release bar: "
                           "publish it with --preview, or build another round")
    return "release"


def stage(workdir: Any, out: Any, repo_id: str, *, name: Optional[str] = None, preview: bool = False,
          corpus: Optional[str] = None, corpus_license: Optional[str] = None,
          corpus_url: Optional[str] = None, bar: float = RELEASE_FIDELITY) -> dict:
    """Export into `out`, decide release or preview, and write the card beside the package.

    Answers the manifest with the ruling on it. The card is written after the manifest and is not
    part of the content hash: it is prose about the package, rewritten on every publish, and a
    package whose card someone edited has not been tampered with.
    """
    resolved = name or repo_id.rsplit("/", 1)[-1]
    manifest = package_mod.export(workdir, out, name=resolved, corpus=corpus, corpus_license=corpus_license,
                                  corpus_url=corpus_url, preview=preview)
    manifest["status"] = release_ruling(manifest, preview, bar)
    Path(out, CARD_NAME).write_text(card_markdown(manifest, repo_id), encoding="utf-8")
    return manifest


def publish(workdir: Any, repo_id: str, *, client: Optional[HubClient] = None, name: Optional[str] = None,
            preview: bool = False, corpus: Optional[str] = None, corpus_license: Optional[str] = None,
            corpus_url: Optional[str] = None, keep: Optional[Any] = None,
            bar: float = RELEASE_FIDELITY) -> tuple[HostedRepo, dict]:
    """Export, card, upload, tag. Answers where it landed and the manifest that landed there."""
    host = hub_client(client)
    root = Path(keep) if keep is not None else Path(tempfile.mkdtemp(prefix="kullback-publish-"))
    out = root / repo_id.rsplit("/", 1)[-1]
    if out.exists():
        shutil.rmtree(out)
    try:
        manifest = stage(workdir, out, repo_id, name=name, preview=preview, corpus=corpus,
                         corpus_license=corpus_license, corpus_url=corpus_url, bar=bar)
        url = host.create_repo(repo_id)
        tag = f"round-{manifest.get('round')}" if manifest.get("round") is not None else "round-unknown"
        commit = host.upload_folder(repo_id, out, _commit_message(manifest, tag))
        host.tag(repo_id, tag, revision=commit or None)
    finally:
        if keep is None:
            shutil.rmtree(root, ignore_errors=True)
    return HostedRepo(repo_id=repo_id, url=url or url_for(repo_id), commit=commit or None, tag=tag), manifest


def _commit_message(manifest: dict, tag: str) -> str:
    fidelity = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
    rate = "not measured" if fidelity is None else f"{float(fidelity):.1%}"
    return (f"{tag}: {manifest.get('status', 'preview')}, replay fidelity {rate} over "
            f"{manifest.get('tasks_total', 0)} Tasks, {manifest.get('trusted', 0)} trusted; "
            f"env {str(manifest.get('env_id') or '')[:12]}, "
            f"content {str(manifest.get('content_hash') or '')[:12]}, "
            f"runner {str(manifest.get('runner_version') or '')[:12]}, "
            f"gates {str(manifest.get('gates_version') or '')[:12]}")


def fetch(repo_id: str, out: Any, *, client: Optional[HubClient] = None,
          revision: Optional[str] = None) -> dict:
    """Download a revision, verify it, and answer the manifest of what landed.

    A package that does not verify is taken away again rather than left on disk half trusted: the
    only thing that makes a fetched Environment worth running is that it is the one that was
    published.
    """
    host = hub_client(client)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    host.download(repo_id, out, revision=revision)
    problems = package_mod.verify_package(out)
    if problems:
        shutil.rmtree(out, ignore_errors=True)
        raise PublishError(f"{repo_id} does not verify against its manifest: " + "; ".join(problems[:5]))
    manifest = dict(read_json(out / package_mod.MANIFEST_NAME, {}) or {})
    manifest["missing_run_inputs"] = missing_run_inputs(out)
    return manifest


def missing_run_inputs(directory: Any) -> list[str]:
    """The records `kullback run` opens that this directory does not hold; empty is a runnable workdir."""
    root = Path(directory)
    missing = [name for name in RUN_INPUTS if not (root / name).is_file()]
    if not any((root / "tasks").glob("*.json")):
        missing.append("tasks/")
    if not (root / "overlays").is_dir():
        missing.append("overlays/")
    return missing


def publish_organisation_card(organisation: str, markdown: str, *,
                              client: Optional[HubClient] = None) -> tuple[bool, str]:
    """Try to set the organisation's profile page; answer whether it took and what the host said."""
    from kullback.hub.client import HubError

    try:
        return True, hub_client(client).set_organisation_card(organisation, markdown)
    except HubError as error:
        return False, str(error)
