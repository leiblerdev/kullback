"""A task registry: the starting container and verifier behind a recorded task id.

A registry answers one question: for the task id a recording names, what did the
task definition hold. The definition is the five-piece layout every publisher of
this shape repeats: an instruction, a task file with timeouts and an optional
pinned image, a container file, and optional test, solution and seed files. The
same layout arrives two ways: packed as one encoded archive per row of a table
with `path` and `task_binary` columns, or unpacked as plain directories with one
subdirectory per task id. This module reads both and names nothing about where
any row came from.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import io
import json
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

# Refusals for archives that would write outside their own directory or smuggle
# a link in. The caps keep one hostile row from exhausting memory.
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_MEMBERS = 1024


@dataclass(frozen=True)
class TaskDefinition:
    """One task definition, however it arrived: packed row or plain directory."""

    task_id: str = ""
    instruction: str | None = None
    dockerfile: str | None = None
    docker_image: str | None = None
    task_toml: dict = field(default_factory=dict)
    tests: dict[str, bytes] = field(default_factory=dict)
    solution: dict[str, bytes] = field(default_factory=dict)
    seeds: dict[str, bytes] = field(default_factory=dict)
    source_ref: str = ""


def unpack_task_binary(blob: str | bytes, task_id: str = "",
                       source_ref: str = "") -> TaskDefinition:
    """One encoded archive to its definition: base64 of a gzip tar of the task directory.

    The archive is never written to disk, only read, and every member name is
    checked before any byte is kept: absolute paths, `..` segments and links
    (symbolic or hard) are refused with a ValueError naming the reason, as is
    an archive over the size or member caps. A blob that is not base64, not
    gzip, or not a tar is refused the same way.
    """
    try:
        raw = base64.b64decode(blob, validate=True) if isinstance(blob, str) else bytes(blob)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"task archive is not base64: {error}") from error
    try:
        expanded = gzip.decompress(raw)
    except (OSError, EOFError) as error:
        raise ValueError(f"task archive is not gzip: {error}") from error
    if len(expanded) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"task archive expands past {MAX_ARCHIVE_BYTES} bytes")
    try:
        members = _safe_members(expanded)
    except tarfile.TarError as error:
        raise ValueError(f"task archive is not a tar: {error}") from error
    return _definition_from_members(members, task_id=task_id, source_ref=source_ref)


def _safe_members(expanded: bytes) -> list[tuple[str, bytes]]:
    """Every file member of the archive as (name, bytes), after the safety checks."""
    out: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r") as tar:
        names = tar.getnames()
        if len(names) > MAX_MEMBERS:
            raise ValueError(f"task archive holds {len(names)} members, over the cap {MAX_MEMBERS}")
        for info in tar:
            _refuse_unsafe_member(info)
            if not info.isfile():
                continue
            if info.size > MAX_MEMBER_BYTES:
                raise ValueError(f"task archive member {info.name!r} is over {MAX_MEMBER_BYTES} bytes")
            handle = tar.extractfile(info)
            out.append((info.name, handle.read() if handle is not None else b""))
    return out


def _refuse_unsafe_member(info: tarfile.TarInfo) -> None:
    """Refuse one member that would escape its directory or smuggle a link."""
    name = info.name
    if info.issym() or info.islnk():
        raise ValueError(f"task archive member {name!r} is a link, links are refused")
    if Path(name).is_absolute():
        raise ValueError(f"task archive member {name!r} is an absolute path, refused")
    if ".." in Path(name).parts:
        raise ValueError(f"task archive member {name!r} escapes its directory, refused")


def _definition_from_members(members: list[tuple[str, bytes]], task_id: str,
                             source_ref: str) -> TaskDefinition:
    """The five-piece layout out of flat (name, bytes) pairs, whatever root wraps them."""
    by_name = {name: body for name, body in members}
    instruction = _text_of(by_name, "instruction.md")
    dockerfile = _text_of(by_name, "environment/Dockerfile")
    task_toml = _parse_toml(_text_of(by_name, "task.toml") or "")
    environment = _nested(task_toml, "environment")
    return TaskDefinition(
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        docker_image=environment.get("docker_image") if isinstance(environment, dict) else None,
        task_toml=task_toml,
        tests={name: body for name, body in members if _under(name, "tests/")},
        solution={name: body for name, body in members if _under(name, "solution/")},
        seeds={name: body for name, body in members if _under(name, "environment/seeds/")},
        source_ref=source_ref,
    )


def _text_of(by_name: dict[str, bytes], *names: str) -> str | None:
    """The text of the first matching member, by exact name or by trailing path."""
    for key, body in by_name.items():
        if any(key == name or key.endswith("/" + name) for name in names):
            return body.decode("utf-8", errors="replace")
    return None


def _under(name: str, folder: str) -> bool:
    """True when the member sits under the folder, with or without a wrapping root."""
    return ("/" + folder) in ("/" + name) or name.startswith(folder)


def _parse_toml(text: str) -> dict:
    """The task file as a dict, empty when absent or unparseable rather than fatal."""
    if not text.strip():
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _nested(mapping: Any, key: str) -> Any:
    """One level of a parsed task file, or nothing when the shape is not a mapping."""
    return mapping.get(key) if isinstance(mapping, dict) else None


def read_task_dir(path: str | Path) -> TaskDefinition:
    """One plain task directory to its definition; the directory name is the task id."""
    root = Path(path)
    members = [(str(relative).replace("\\", "/"), (root / relative).read_bytes())
               for relative in _walk_files(root)]
    return _definition_from_members(members, task_id=root.name, source_ref=str(root))


def _walk_files(root: Path) -> list[Path]:
    """Every file under the directory, relative, sorted so reads are stable."""
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


class Registry(Protocol):
    """The lookup a rescue reads: a task id to its definition, or nothing."""

    def lookup(self, task_id: str) -> TaskDefinition | None:
        """The definition for the task id, or None when the registry has no such id."""
        ...  # pragma: no cover


class DirectoryRegistry:
    """One subdirectory per task id under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def lookup(self, task_id: str) -> TaskDefinition | None:
        """The plain directory named by the task id, or None when it is missing."""
        target = self.root / task_id
        if not target.is_dir():
            return None
        return read_task_dir(target)


class TableRegistry:
    """Rows of mappings with `path` (the task id) and `task_binary` (the archive)."""

    def __init__(self, rows: Any) -> None:
        self._by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows or []:
            if isinstance(row, Mapping) and isinstance(row.get("path"), str):
                self._by_id.setdefault(row["path"], row)

    def lookup(self, task_id: str) -> TaskDefinition | None:
        """The decoded row whose path equals the task id, or None when none does."""
        row = self._by_id.get(task_id)
        if row is None or row.get("task_binary") is None:
            return None
        return unpack_task_binary(row["task_binary"], task_id=task_id,
                                  source_ref=f"row:{task_id}")


def open_registry(path: str | Path) -> Registry:
    """The registry behind a local path: a directory of task directories, or a rows file.

    A fetch from a remote table is out of scope, so the path must be local; a
    path that is neither a directory nor a file is refused with a ValueError.
    """
    target = Path(path)
    if target.is_dir():
        return DirectoryRegistry(target)
    if target.is_file():
        return TableRegistry(rows_from_file(target))
    raise ValueError(f"no registry at {target}: not a directory or a rows file")


def rows_from_file(path: str | Path) -> list[dict]:
    """The rows file as a list of mappings: JSON always, parquet only with pyarrow.

    Parquet needs pyarrow, which is not a declared dependency of this project, so
    asking for a parquet file raises a ValueError saying so instead of importing
    something the install cannot provide. A fetch from a remote table is out of
    scope: the path must be local.
    """
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise ValueError("a parquet registry needs pyarrow, which this project "
                             "does not depend on; pass a JSON rows file instead") from None
        from pyarrow import parquet as parquet_mod

        return [dict(row) for row in parquet_mod.read_table(target).to_pylist()]
    body = json.loads(target.read_text(encoding="utf-8"))
    rows = body if isinstance(body, list) else body.get("rows", [])
    return [row for row in rows if isinstance(row, dict)]


def verifier_strength(defn: TaskDefinition) -> Literal["content", "existence", "none"]:
    """How strong the fetched verifier is, as a label, never a Verdict.

    "none" when the definition carries no tests at all. "content" when a test
    compares output or state against an expected value: a test file reads an
    expected file or embeds an expected string beside a comparison (an equality
    or ordering operator, or a compare or diff tool). "existence" covers
    everything else with tests, including a check that only asserts a path
    exists. The rule reads the text of the tests only.
    """
    if not defn.tests:
        return "none"
    texts = [body.decode("utf-8", errors="replace") for body in defn.tests.values()]
    joined = "\n".join(texts)
    if ("expected" in joined or "==" in joined or "!=" in joined
            or "cmp" in joined or "diff" in joined):
        return "content"
    return "existence"


def instruction_matches(defn: TaskDefinition, recorded_instruction: str | None) -> bool:
    """True when the definition's instruction is the recording's own, exactly.

    Both sides are whitespace-normalized first (every run of whitespace becomes
    one space, ends stripped), so a re-wrapped copy still matches while a
    different problem never does. Either side missing is not a match.
    """
    if defn.instruction is None or recorded_instruction is None:
        return False
    return " ".join(defn.instruction.split()) == " ".join(recorded_instruction.split())


def definition_dict(defn: TaskDefinition) -> dict:
    """The definition as JSON-safe data: file bytes travel base64-encoded."""
    return {
        "task_id": defn.task_id,
        "instruction": defn.instruction,
        "dockerfile": defn.dockerfile,
        "docker_image": defn.docker_image,
        "task_toml": defn.task_toml,
        "tests": {name: base64.b64encode(body).decode("ascii")
                  for name, body in defn.tests.items()},
        "solution": {name: base64.b64encode(body).decode("ascii")
                     for name, body in defn.solution.items()},
        "seeds": {name: base64.b64encode(body).decode("ascii")
                  for name, body in defn.seeds.items()},
        "source_ref": defn.source_ref,
    }
