"""Whether a workdir is ready to publish: fidelity, frozen Runner, leak scan, token, repo name.

`publish --check` prints these rows and uploads nothing, and a plain `publish` prints them
first and then proceeds. The fidelity and leak rows come from staging the exact package a
publish would upload, so the checklist cannot disagree with the upload. No row ever carries
a secret: the token row names where the token came from, never its value.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kullback.hub import package as package_mod
from kullback.hub.publish import RELEASE_FIDELITY


@dataclass(frozen=True)
class Row:
    """One checklist row: what was checked, whether it holds, and the detail behind it."""
    name: str
    done: bool
    detail: str


FIDELITY_ROW = "fidelity over Tasks"
RUNNER_ROW = "runner frozen"
LEAK_ROW = "leak scan"
TOKEN_ROW = "HF token"
REPO_ROW = "repo name"

READY_RELEASE = "ready to release"
PREVIEW_ONLY = "preview only"
NOT_READY = "not ready"

_REPO_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")


def _runner_row(workdir: Path) -> Row:
    """Frozen is a version file on disk; its detail is the version id the Verdicts would carry."""
    path = workdir / "runner_version.json"
    if not path.is_file():
        return Row(RUNNER_ROW, False, "not frozen: run `kullback freeze-runner`")
    try:
        version = (json.loads(path.read_text(encoding="utf-8")) or {}).get("runner_version")
    except (OSError, ValueError):
        return Row(RUNNER_ROW, False, "runner_version.json holds no readable version")
    if not version:
        return Row(RUNNER_ROW, False, "runner_version.json holds no version")
    return Row(RUNNER_ROW, True, f"frozen {version}")


def _repo_row(repo: str) -> Row:
    """A repo is an organisation and a name; anything else cannot be uploaded to."""
    if repo and _REPO_PATTERN.fullmatch(repo):
        return Row(REPO_ROW, True, repo)
    return Row(REPO_ROW, False, f"repo must read as organisation/name, got {repo!r}")


def _cached_login() -> bool:
    """Whether the Hub's own cache holds a login, without reading its value."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    try:
        return bool(get_token())
    except Exception:  # noqa: BLE001 - a broken cache reads as no login, not a failed check
        return False


def _token_row() -> Row:
    """The token row names where the token came from, never the value itself."""
    if os.environ.get("HF_TOKEN"):
        return Row(TOKEN_ROW, True, "HF_TOKEN is set")
    if _cached_login():
        return Row(TOKEN_ROW, True, "the cached Hugging Face login is set")
    return Row(TOKEN_ROW, False, "missing: set HF_TOKEN or log in")


def _stage_manifest(workdir: Path, repo: str) -> dict:
    """The manifest a publish would upload, staged in a temporary directory and taken away again."""
    name = repo.rsplit("/", 1)[-1] or "checklist"
    tmp = Path(tempfile.mkdtemp(prefix="kullback-checklist-"))
    try:
        return package_mod.export(workdir, tmp / name, name=name, scan=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fidelity_row(manifest: Optional[dict], note: str) -> Row:
    """Fidelity over Tasks against the release bar, or why no release can claim it."""
    if manifest is None:
        return Row(FIDELITY_ROW, False, f"not measured: {note}")
    rate = (manifest.get("replay_fidelity") or {}).get("tasks_rate")
    rate = None if rate is None else float(rate)
    if rate is None:
        return Row(FIDELITY_ROW, False, f"not measured (bar {RELEASE_FIDELITY:.2f}): preview only")
    ruling = "release" if rate >= RELEASE_FIDELITY else "preview only"
    return Row(FIDELITY_ROW, rate >= RELEASE_FIDELITY,
               f"{rate:.2f} over Tasks (bar {RELEASE_FIDELITY:.2f}): {ruling}")


def _leak_row(manifest: Optional[dict]) -> Row:
    """The leak scan publish itself runs, in counts; values are customer data and are never said."""
    if manifest is None:
        return Row(LEAK_ROW, False, "not checked: the package could not be staged")
    scan = manifest.get("leak_scan") or {}
    leaks = int(scan.get("leaks", 0))
    echoes = int(scan.get("value_echoes", 0))
    files = int(scan.get("files_scanned", 0))
    return Row(LEAK_ROW, leaks == 0, f"{leaks} leaks and {echoes} value echoes over {files} files")


def publish_checklist(workdir: Any, repo: str) -> list[Row]:
    """Every publish gate as rows: fidelity over Tasks, frozen Runner, leak scan, token, repo name.

    The fidelity and leak rows stage the package a publish would upload, so a leak or an
    unexportable workdir fails the checklist the same way it would fail the upload.
    """
    root = Path(workdir)
    try:
        manifest: Optional[dict] = _stage_manifest(root, repo)
        note = ""
    except package_mod.ExportError as error:
        manifest, note = None, str(error)
    return [_fidelity_row(manifest, note), _runner_row(root), _leak_row(manifest), _token_row(),
            _repo_row(repo)]


def readiness(rows: list[Row]) -> str:
    """The one line under the rows: release when every row holds, preview when only the bar or the
    freeze is missing, not ready when anything a preview publish needs is missing."""
    by_name = {row.name: row for row in rows}
    if rows and all(row.done for row in rows):
        return READY_RELEASE
    token = by_name.get(TOKEN_ROW)
    repo = by_name.get(REPO_ROW)
    leak = by_name.get(LEAK_ROW)
    if token is not None and token.done and repo is not None and repo.done and (leak is None
                                                                                 or leak.done):
        return PREVIEW_ONLY
    return NOT_READY
