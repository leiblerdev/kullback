"""The whole surface a dataset host is spoken to through, and the one adapter that speaks it (D221).

Five calls: make a repository, upload a folder into it, tag the commit that made, download a
revision of it, and set the profile page of an organisation. Nothing else in the harness imports a
hosting library, so publishing is testable against an in-memory stand-in that implements the same
five, and swapping the host is one class.

The token is never a parameter here and never passed on a command line: the adapter takes whatever
the hosting library has already cached for the machine, and no code path in the harness reads its
value, prints it or writes it into a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

# The kind of repository an Environment is: a dataset, because that is what a world plus its graders
# is to anybody fetching it.
REPO_TYPE = "dataset"


class HubError(RuntimeError):
    """A host refused, or the hosting library is not installed."""


@dataclass(frozen=True)
class HostedRepo:
    """Where a publish landed: the repository, the commit and the tag naming it."""

    repo_id: str
    url: str
    commit: Optional[str] = None
    tag: Optional[str] = None


class HubClient(Protocol):
    """What publishing needs of a host, and nothing more."""

    def create_repo(self, repo_id: str) -> str:
        """Make the repository if it is not there; answer its URL."""

    def upload_folder(self, repo_id: str, folder: Path, message: str) -> str:
        """Upload every file under `folder` as one commit; answer the commit id."""

    def tag(self, repo_id: str, tag: str, revision: Optional[str] = None) -> None:
        """Name a commit, replacing a tag of the same name."""

    def download(self, repo_id: str, out: Path, revision: Optional[str] = None) -> Path:
        """Lay a revision of the repository out under `out`; answer the directory."""

    def set_organisation_card(self, organisation: str, markdown: str) -> str:
        """Set an organisation's profile page; raise HubError where the host has no such call."""


class HuggingFaceHub:
    """The one adapter, over huggingface_hub. Constructed only when a command actually publishes."""

    def __init__(self, api: Any = None):
        self._api = api if api is not None else _api()

    def create_repo(self, repo_id: str) -> str:
        self._api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=False, exist_ok=True)
        return url_for(repo_id)

    def upload_folder(self, repo_id: str, folder: Path, message: str) -> str:
        info = self._api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=str(folder),
                                       commit_message=message)
        return str(getattr(info, "oid", None) or getattr(info, "commit_id", None) or "")

    def tag(self, repo_id: str, tag: str, revision: Optional[str] = None) -> None:
        try:
            self._api.delete_tag(repo_id=repo_id, tag=tag, repo_type=REPO_TYPE)
        except Exception:  # noqa: BLE001 - a tag that is not there is the state we wanted
            pass
        self._api.create_tag(repo_id=repo_id, tag=tag, repo_type=REPO_TYPE, revision=revision)

    def download(self, repo_id: str, out: Path, revision: Optional[str] = None) -> Path:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, repo_type=REPO_TYPE, revision=revision,
                                 local_dir=str(out))
        return Path(path)

    def set_organisation_card(self, organisation: str, markdown: str) -> str:
        """An organisation profile is not a repository, and the Hub API exposes no write for it.

        Saying so is the honest answer: the card is written to disk either way and a person pastes it
        into the organisation's settings. This raises rather than pretending a publish happened.
        """
        raise HubError(
            f"the Hub has no API for an organisation profile: paste the card into "
            f"https://huggingface.co/organizations/{organisation}/settings/profile by hand")


def url_for(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - the dependency is declared in pyproject
        raise HubError("huggingface_hub is not installed: `uv sync` in the project") from error
    return HfApi()


def hub_client(client: Optional[HubClient] = None) -> HubClient:
    """The client a command uses: the one it was handed, else the adapter over the real host."""
    return client if client is not None else HuggingFaceHub()
