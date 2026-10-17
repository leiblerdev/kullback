"""The scenario record and its store (D372).

One record per real-world situation, in general words. A record says where it came from (the
recordings whose opening asks it groups, a search page and the byte range of the span it rests on,
or a line a customer typed), the policy lines it touches, each checked by code to be a substring of
the policy, and how many opening asks it groups. The Examiner's verdicts land on it later.

The store is one JSON file per record under <workdir>/scenarios/, an index beside them and a marker
file saying what the folder is. Nothing here touches tasks.json or anything the trusted count reads:
a scenario is not a Task. The loader refuses an undeclared key the way the Task record does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import Field

from kullback.runner.records import Record, read_json, write_json

DIR = "scenarios"
INDEX = "index.json"
MARKER = "STORE"
ASKS = "asks.txt"
HUMAN = "human.json"
GENERATED = "generated.json"
FORMAT = 1
MARKER_TEXT = ("A scenario store (D372): one JSON per scenario plus index.json. Scenarios are not Tasks; "
               "nothing here is read by the trusted count.\n")

SourceKind = Literal["recordings", "search", "ask"]
Status = Literal["born", "fixing", "passed", "unsure", "infeasible", "struck"]
SOURCE_KINDS: tuple[str, ...] = ("recordings", "search", "ask")
STATUSES: tuple[str, ...] = ("born", "fixing", "passed", "unsure", "infeasible", "struck")
# The statuses a review has settled; the rest are still open to a review session.
SETTLED: tuple[str, ...] = ("passed", "unsure", "infeasible", "struck")
# How many characters of a Task id any print shows.
MASK_CHARS = 4


class Source(Record):
    """Where one scenario came from. Recordings name Task ids, search a URL and a byte range into the
    cached page text (the D225 cache and range shape), ask the line as typed."""

    kind: SourceKind
    task_ids: list[str] = Field(default_factory=list)
    url: str = ""
    key: str = ""
    start: Optional[int] = None
    end: Optional[int] = None
    line: str = ""


class Scenario(Record):
    """One real-world situation: the statement, its source, the policy lines it touches, how often the
    recordings open with it, the Examiner's verdicts and its status; the plan's situation and variant it
    was written for, the rows of the world it is bound to (table to key) and what the world lacks for it
    (D381); the user's own opening words, first person, and what the world looks like after a correct
    agent (both empty on older records)."""

    id: str
    statement: str
    source: Source
    also_from: list[Source] = Field(default_factory=list)
    policy_lines: list[str] = Field(default_factory=list)
    frequency: int = 0
    review: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    status: Status = "born"
    fixes: int = 0
    situation: str = ""
    variant: str = ""
    binding: dict[str, str] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    ask: str = ""
    end: str = ""


def scenario_id(statement: str, kind: str) -> str:
    """A record's name: a content hash of its statement and source kind, nothing else."""
    return "sc_" + hashlib.sha256(f"{kind}\n{statement}".encode("utf-8")).hexdigest()[:12]


def mask(task_id: Any) -> str:
    """A Task id as any print shows it: its last four characters."""
    return "..." + str(task_id)[-MASK_CHARS:]


def policy_lines_in(lines: Iterable[Any], policy: str) -> tuple[list[str], int]:
    """The quoted lines that are substrings of the policy, and how many were not.

    A policy line the model wrote but the policy does not hold is a paraphrase, and a paraphrase cannot
    be audited, so it is dropped from the record and counted; the statement stays.
    """
    kept: list[str] = []
    dropped = 0
    for line in lines or ():
        text = str(line or "").strip()
        if text and text in str(policy or "") and text not in kept:
            kept.append(text)
        elif text:
            dropped += 1
    return kept, dropped


def born(statement: str, source: Source, policy_lines: Iterable[str] = (), frequency: int = 0) -> Scenario:
    """A new record, status born, its id from the statement and the source kind."""
    text = " ".join(str(statement or "").split())
    return Scenario(id=scenario_id(text, source.kind), statement=text, source=source,
                    policy_lines=list(policy_lines), frequency=int(frequency))


def store_dir(workdir: Any) -> Path:
    return Path(workdir) / DIR


def record_path(workdir: Any, scenario_id_: str) -> Path:
    name = str(scenario_id_)
    if not name.startswith("sc_") or "/" in name or "." in name:
        raise ValueError(f"unknown scenario {name!r}")
    return store_dir(workdir) / f"{name}.json"


def write_scenario(workdir: Any, record: Scenario) -> Path:
    """One record on disk, and the index and the marker written again beside it."""
    path = write_json(record_path(workdir, record.id), record.model_dump(mode="json"))
    _write_index(workdir)
    return path


def write_all(workdir: Any, records: Iterable[Scenario]) -> None:
    for record in records:
        write_json(record_path(workdir, record.id), record.model_dump(mode="json"))
    _write_index(workdir)


def _write_index(workdir: Any) -> None:
    folder = store_dir(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / MARKER
    if not marker.is_file():
        marker.write_text(MARKER_TEXT, encoding="utf-8")
    rows = [{"id": record.id, "kind": record.source.kind, "status": record.status}
            for record in load_scenarios(workdir)]
    write_json(folder / INDEX, {"format": FORMAT, "scenarios": rows})


def load_scenario(workdir: Any, scenario_id_: str) -> Scenario:
    """One record; a ValueError for an unknown id or a file carrying an undeclared key."""
    path = record_path(workdir, scenario_id_)
    if not path.is_file():
        raise ValueError(f"unknown scenario {scenario_id_!r}")
    return Scenario.model_validate(read_json(path))


def load_scenarios(workdir: Any) -> list[Scenario]:
    """Every record of the store, in id order. A file with an undeclared key raises, never drops."""
    folder = store_dir(workdir)
    return [Scenario.model_validate(read_json(path))
            for path in sorted(folder.glob("sc_*.json"))] if folder.is_dir() else []


__all__ = ["ASKS", "DIR", "HUMAN", "INDEX", "MARKER", "GENERATED", "SETTLED", "SOURCE_KINDS", "STATUSES", "Scenario",
           "Source", "born", "load_scenario", "load_scenarios", "mask", "policy_lines_in", "record_path",
           "scenario_id", "store_dir", "write_all", "write_scenario"]
