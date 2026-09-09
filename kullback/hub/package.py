"""The Environment package: the rebuilt world and the graders, with a manifest and a leak scan (D221).

A build's workdir holds two different things side by side. One is the world the Builder rebuilt and
the graders the Examiner derived, which is the Environment anybody can run a candidate against. The
other is how it got there: the customer's export under raw/, the Runs and Traces those produced, the
two agents' transcripts, and every cache keyed on a model call. Only the first is an Environment,
and only the first may leave the machine it was built on.

This module names the first explicitly. `EXPORTED` is an allow list, not a deny list: an artifact a
later build starts writing is out of the package until someone puts it in, which is the only
ordering that stays safe as the workdir grows. On top of that an exported Task record drops the ids
of the recordings it was clustered from, because those name Traces the package does not carry.

The manifest carries the numbers a reader needs before trusting anything: how far replay fidelity
got over Tasks and over Runs, how many References were confirmed, Verifiers derived and Tasks
trusted, the difficulty buckets, the versions of the runner and the gates the numbers were measured
under, the harness version and git sha, and a content hash over every file so a fetch can tell a
package apart from a tampered copy of one.

The leak scan is the check that the two halves stayed apart. It reads every string the customer's
export holds, and fails the export when a file outside the world files repeats one of them verbatim
while nothing in the world files says it: a value like that can only have come from a recording. The
baseline is the package's own world files, and the manifest also carries the stricter count over
`env/` alone, so a reader can see both readings rather than the one that happened to pass.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import difficulty
from kullback.runner.records import content_hash, read_json, write_json

PACKAGE_FORMAT = 1
MANIFEST_NAME = "manifest.json"
TASKS_INDEX_NAME = "tasks_index.json"
# The card publish.py writes beside the package. Named here because the hashes have to leave it out.
CARD_NAME = "README.md"

# The rebuilt world: the Environment package the Builder compiled, the per-Task Starting states, and
# the records the Runner needs to load a toolkit over them. These are also the leak scan's baseline,
# because a value they hold is part of the world the package ships and not a quotation of a
# recording.
WORLD_DIRS: tuple[str, ...] = ("env", "overlays")
WORLD_FILES: tuple[str, ...] = ("environment.json", "schema.json", "tool_sigs.json", "bodies.json",
                                "canon-rules.json")
# The graders and the Task list. Scanned against the baseline rather than trusted.
GRADER_DIRS: tuple[str, ...] = ("tasks", "verifiers")

EXPORTED_DIRS: tuple[str, ...] = WORLD_DIRS + GRADER_DIRS
EXPORTED_FILES: tuple[str, ...] = WORLD_FILES

# Named so the guard below can say what it refused, and so a reader of this module can see what is
# being kept out without reading a build. Nothing here is ever a source path of an exported file.
NEVER_EXPORTED: tuple[str, ...] = ("raw", "runs", "cache", "model_cache", "traces", "builder",
                                   "examiner", "grader", "web_cache", "memory", "probes", "repairs",
                                   "intents", "pipeline", "user_rules", "readers")

# The Task fields that name recordings. A package holds no Trace, so it names none: a fetched Task
# that still listed them would point at files nobody has.
RECORDING_FIELDS: tuple[str, ...] = ("run_ids", "anchor_run_ids")

# A string shorter than this collides between any two corpora by accident, so it is no evidence that
# anything leaked. Ids, names, addresses and free text are all longer.
LEAK_MIN_LENGTH = 6
# Where a string stops being a field value and starts being prose. A row's value (an id, a name, an
# address, a date) is shorter than a sentence on any corpus; a recorded turn, a system prompt or a
# quoted policy is longer. The two are treated differently because a grader has to carry values and
# must never carry a transcript.
CONTENT_MIN_LENGTH = 80
LEAK_BASELINE = ("the package's world files, the harness's own source, and, for a Verifier, the Task "
                 "instruction the candidate is handed")

# The ladder a Task climbs, in order. `funnel_stage` answers the last rung it reached.
FUNNEL: tuple[str, ...] = ("clustered", "replay_confirmed", "reference_confirmed", "verifier_derived",
                           "verifier_passed", "trusted")
REFUSED_STAGE = "refused"

# What stopped a Task, in the harness's own fixed words. A status row's own reason quotes the
# customer's world back (see difficulty.py), so it never reaches a manifest or a card.
STOPPED_AT = {
    "clustered": "no replay of the Task was confirmed",
    "replay_confirmed": "the Reference is not confirmed",
    "reference_confirmed": "no Verifier was derived",
    "verifier_derived": "the D79 suite did not pass",
    "verifier_passed": "the Verifier passed the suite and is not the trusted one",
    REFUSED_STAGE: "the Task was refused: no frontier Run of it finished",
}


class ExportError(RuntimeError):
    """An export that must not produce a package: a workdir with no Environment, or a leak."""


# --- reading a workdir --------------------------------------------------------------


def last_round(workdir: Path) -> dict:
    """The counts of the last round on disk, or an empty body where a build wrote none."""
    rounds = read_json(workdir / "rounds.json", []) or []
    if not isinstance(rounds, list) or not rounds:
        return {}
    last = rounds[-1]
    counts = dict(last.get("counts") or {}) if isinstance(last, dict) else {}
    counts.setdefault("round", last.get("round") if isinstance(last, dict) else None)
    return counts


def frozen_task_ids(workdir: Path) -> list[str]:
    """The Task list this build's numbers are measured against, newest snapshot first.

    A round snapshot (`rounds/<n>/tasks.json`) is the list as that round left it and is preferred
    where one exists; without one the frozen list stands, and without that the Tasks on disk do. The
    three are read in that order so a workdir written by any of them exports the same way.
    """
    snapshots = sorted((workdir / "rounds").glob("*/tasks.json")) if (workdir / "rounds").is_dir() else []
    if snapshots:
        latest = max(snapshots, key=lambda path: _round_number(path.parent.name))
        body = read_json(latest, None)
        ids = _task_ids_of(body)
        if ids:
            return ids
    ids = _task_ids_of(read_json(workdir / "tasks_frozen.json", None))
    if ids:
        return ids
    return sorted(path.stem for path in (workdir / "tasks").glob("*.json"))


def _round_number(name: str) -> int:
    return int(name) if name.isdigit() else -1


def _task_ids_of(body: Any) -> list[str]:
    if isinstance(body, dict):
        ids = body.get("task_ids")
        if ids:
            return [str(task_id) for task_id in ids]
        return [str(task.get("id")) for task in body.get("tasks") or () if isinstance(task, dict)]
    if isinstance(body, list):
        return [str(task.get("id")) if isinstance(task, dict) else str(task) for task in body]
    return []


def replay_fidelity(replays: dict, task_ids: Iterable[str]) -> dict:
    """Replay fidelity over Tasks and over Runs, both over the frozen list.

    A Task clears fidelity when any replay of it was confirmed, which is the count round_end reports;
    the Run number is every replay of every Task on the list, which is the finer of the two and the
    one a reader should look at when a Task holds many recordings.
    """
    ids = list(task_ids)
    tasks_confirmed, runs_total, runs_confirmed = 0, 0, 0
    for task_id in ids:
        per_task = (replays or {}).get(task_id) or {}
        confirmed = [run_id for run_id, row in per_task.items() if _confirmed(row)]
        runs_total += len(per_task)
        runs_confirmed += len(confirmed)
        tasks_confirmed += 1 if confirmed else 0
    return {"tasks": tasks_confirmed, "tasks_total": len(ids), "tasks_rate": _share(tasks_confirmed, len(ids)),
            "runs": runs_confirmed, "runs_total": runs_total, "runs_rate": _share(runs_confirmed, runs_total)}


def _confirmed(row: Any) -> bool:
    return bool(row.get("confirmed", False)) if isinstance(row, dict) else False


def _share(part: int, whole: int) -> Optional[float]:
    return round(part / whole, 4) if whole else None


def funnel_stage(task_id: str, *, status_row: Any, replay_confirmed: bool, has_verifier: bool,
                 trusted_ids: Iterable[str], refused: Iterable[str]) -> str:
    """The last rung of the funnel a Task reached, or that it was refused.

    Refusal is terminal and is answered first: a refused Task is not a Task that stopped somewhere,
    it is one the harness ruled nobody finished (D128).
    """
    if task_id in set(refused or ()):
        return REFUSED_STAGE
    row = status_row if isinstance(status_row, dict) else {}
    if task_id in set(trusted_ids or ()):
        return "trusted"
    if row.get("verifier_passed"):
        return "verifier_passed"
    if has_verifier:
        return "verifier_derived"
    if row.get("reference_confirmed"):
        return "reference_confirmed"
    return "replay_confirmed" if replay_confirmed else "clustered"


def stopped_because(stage: str, status_row: Any) -> Optional[str]:
    """Why a Task went no further, in fixed words; None for a Task that reached the top.

    A Task stopped at the D79 suite names the checks that failed, because those names are the
    harness's own and let a reader group the Tasks that stop there (D198). Nothing else off the
    status row is quoted.
    """
    if stage == "trusted":
        return None
    reason = STOPPED_AT.get(stage, "")
    if stage == "verifier_derived":
        row = status_row if isinstance(status_row, dict) else {}
        failed = sorted(name for name, ok in (row.get("checks") or {}).items() if ok is False)
        if failed:
            return f"{reason}: {', '.join(failed)}"
    return reason


def task_index(workdir: Path, task_ids: Iterable[str]) -> list[dict]:
    """One row per Task: the trust ruling, the funnel stage it reached, its difficulty record and bucket."""
    status = read_json(workdir / "task_status.json", {}) or {}
    replays = read_json(workdir / "replays.json", {}) or {}
    counts = last_round(workdir)
    trusted_ids = set(counts.get("trusted_ids") or ())
    refused = set((counts.get("refused") or {}))
    records = _difficulty_records(workdir)
    rows = []
    for task_id in task_ids:
        row = status.get(task_id) or {}
        confirmed = any(_confirmed(entry) for entry in (replays.get(task_id) or {}).values())
        has_verifier = (workdir / "verifiers" / f"{task_id}.json").is_file()
        stage = funnel_stage(task_id, status_row=row, replay_confirmed=confirmed, has_verifier=has_verifier,
                             trusted_ids=trusted_ids, refused=refused)
        record = records.get(task_id) or {}
        rows.append({
            "task_id": task_id,
            "trusted": stage == "trusted",
            "refused": stage == REFUSED_STAGE,
            "stage": stage,
            "stopped_because": stopped_because(stage, row),
            "replay_confirmed": confirmed,
            "reference_confirmed": bool(row.get("reference_confirmed")),
            "verifier": has_verifier,
            "recordings": int(row.get("recordings") or 0),
            "difficulty": {key: value for key, value in record.items() if key != "task_id"} or None,
            "bucket": record.get("bucket"),
        })
    return rows


def _difficulty_records(workdir: Path) -> dict[str, dict]:
    """Every Task's difficulty record (D209), off difficulty.json where a round wrote one, else computed.

    Computing reads the workdir and writes nothing, so an export never touches the build it exports.
    """
    body = read_json(workdir / difficulty.FILE_NAME, None)
    if not isinstance(body, dict) or not body.get("tasks"):
        try:
            body = difficulty.compute(workdir)
        except (OSError, ValueError, TypeError, KeyError):
            body = {"tasks": []}
    return {str(record.get("task_id")): record for record in body.get("tasks") or ()
            if isinstance(record, dict)}


def untrusted_reasons(rows: Iterable[dict], top: int = 3) -> dict:
    """How many Tasks are not trusted and the commonest reasons, in the harness's fixed words."""
    counted: dict[str, int] = {}
    total = 0
    for row in rows:
        if row.get("trusted"):
            continue
        total += 1
        reason = row.get("stopped_because") or "not trusted"
        counted[reason] = counted.get(reason, 0) + 1
    ranked = sorted(counted.items(), key=lambda pair: (-pair[1], pair[0]))[:top]
    return {"count": total, "reasons": [{"reason": reason, "tasks": count} for reason, count in ranked]}


# --- the package on disk ------------------------------------------------------------


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _exported_task(body: Any, intent: Optional[str] = None) -> dict:
    """One Task record as the package carries it: its instruction, and no id of a recording.

    The instruction is the one thing a candidate is given, and on a finished build it sits in
    `intents/<task>.json` rather than on the Task record. That file also holds the spans the Intent
    was grounded in, which are verbatim turns of the recording, so the file itself never ships and
    only its text is copied onto the Task where `Task.intent` already belongs.
    """
    record = dict(body) if isinstance(body, dict) else {}
    for field in RECORDING_FIELDS:
        if field in record:
            record[field] = []
    if intent and not record.get("intent"):
        record["intent"] = intent
    return record


def _intent_text(workdir: Path, task_id: str) -> Optional[str]:
    """The Intent the build mined for one Task, text only."""
    body = read_json(workdir / "intents" / f"{task_id}.json", None)
    text = body.get("text") if isinstance(body, dict) else None
    return str(text) if text else None


def _lay_out(workdir: Path, out: Path, task_ids: list[str]) -> list[str]:
    """Copy the world and the graders into `out`, and answer the relative paths written."""
    written: list[str] = []
    for name in WORLD_DIRS:
        folder = workdir / name
        for path in sorted(folder.rglob("*")) if folder.is_dir() else ():
            if path.is_file():
                relative = path.relative_to(workdir).as_posix()
                _copy(path, out / relative)
                written.append(relative)
    for name in WORLD_FILES:
        path = workdir / name
        if path.is_file():
            _copy(path, out / name)
            written.append(name)
    for task_id in task_ids:
        source = workdir / "tasks" / f"{task_id}.json"
        if source.is_file():
            write_json(out / "tasks" / f"{task_id}.json",
                       _exported_task(read_json(source, {}), _intent_text(workdir, task_id)))
            written.append(f"tasks/{task_id}.json")
        verifier = workdir / "verifiers" / f"{task_id}.json"
        if verifier.is_file():
            _copy(verifier, out / "verifiers" / f"{task_id}.json")
            written.append(f"verifiers/{task_id}.json")
    _refuse_forbidden(written)
    return written


def _refuse_forbidden(written: Iterable[str]) -> None:
    """A guard on the allow list itself: nothing under a directory that holds recordings ever ships."""
    for relative in written:
        head = relative.split("/", 1)[0]
        if head in NEVER_EXPORTED:
            raise ExportError(f"{relative} sits under {head}, which holds recordings and never ships")


# --- the leak scan ------------------------------------------------------------------


def _strings(body: Any, out: set[str], min_length: int) -> None:
    if isinstance(body, str):
        if len(body) >= min_length:
            out.add(body)
    elif isinstance(body, dict):
        for key, value in body.items():
            if isinstance(key, str) and len(key) >= min_length:
                out.add(key)
            _strings(value, out, min_length)
    elif isinstance(body, list):
        for value in body:
            _strings(value, out, min_length)


def _file_strings(path: Path, min_length: int) -> set[str]:
    out: set[str] = set()
    if path.suffix == ".json":
        try:
            _strings(json.loads(path.read_text(encoding="utf-8")), out, min_length)
        except (OSError, ValueError):
            return out
    return out


def corpus_strings(raw_dir: Path, min_length: int = LEAK_MIN_LENGTH) -> set[str]:
    """Every string the customer's export holds, as values and as keys."""
    out: set[str] = set()
    for path in sorted(raw_dir.rglob("*.json")) if raw_dir.is_dir() else ():
        out |= _file_strings(path, min_length)
    return out


def _baseline(package: Path, names: Iterable[str], min_length: int) -> tuple[set[str], str]:
    """What the baseline says: its string leaves, and everything it says as one blob.

    Both readings are needed. The leaf set is exact and O(1), which is what makes the scan cheap over
    a large world. The blob catches a value stated inside a longer string: a tool body, a policy
    sentence, or a Task instruction that says an address in the middle of a sentence rather than as a
    field of its own. Only the strings the leaf set misses ever reach the blob.
    """
    values: set[str] = set()
    text: list[str] = []
    for name in names:
        root = package / name
        paths = sorted(root.rglob("*")) if root.is_dir() else ([root] if root.is_file() else [])
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix == ".json":
                values |= _file_strings(path, min_length)
            try:
                text.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return values, "\n".join(text)


def harness_text() -> str:
    """The harness's own source, as one blob.

    Half the strings a Verifier carries are the harness's own vocabulary, not the corpus's: an atom
    kind, a column class, a tool kind, the name of a check. Those words also occur in a corpus that
    was produced by an agent harness, so without this they read as values the recordings contributed
    and the scan drowns in its own nouns. A string the harness's source says is accounted for by the
    harness.
    """
    root = Path(__file__).resolve().parents[1]
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                     for path in sorted(root.rglob("*.py")))


def leak_scan(package: Path, raw_dir: Path, *, min_length: int = LEAK_MIN_LENGTH,
              content_min_length: int = CONTENT_MIN_LENGTH, corpus: Optional[set[str]] = None) -> dict:
    """Every string a scanned file repeats from the customer's export that nothing else accounts for.

    Three things legitimately account for a string in a package. The world files account for the
    world's own rows, whoever first wrote them down. The harness's own source accounts for its
    vocabulary. And the Task's instruction accounts for what the candidate is handed, which is why a
    Verifier is scanned against the Task list as well and a Task only against the first two.

    What is left splits in two, and both are reported. A string long enough to be prose rather than a
    field value is a recording that got into the package: a turn, a system prompt, a quoted policy.
    That fails the export. A shorter one is a value, and a grader's answer key holds values the
    Starting state does not, by construction: a Task where the user gives a new address mid
    conversation has that address in its Verifier and nowhere else, and it cannot be graded without
    it. Calling that a leak would forbid the package from carrying a working grader, so it is counted
    as an echo, reported on the card, and does not stop the export. The count over `env/` alone is
    measured beside it, because that is the strictest reading and a reader should see both.

    Counts only. A leaking string is customer data, so the scan names the file and never the string.
    """
    corpus = corpus_strings(raw_dir, min_length) if corpus is None else corpus
    world_values, world_text = _baseline(package, WORLD_DIRS + WORLD_FILES, min_length)
    env_values, env_text = _baseline(package, ("env",), min_length)
    task_values, task_text = _baseline(package, ("tasks",), min_length)
    harness = harness_text()

    def accounted(value: str, values: set[str], text: str) -> bool:
        return value in values or value in text or value in harness

    scanned: list[tuple[Path, set[str], str]] = []
    for path in sorted((package / "tasks").rglob("*.json")) if (package / "tasks").is_dir() else ():
        scanned.append((path, world_values, world_text))
    later = world_values | task_values
    later_text = world_text + "\n" + task_text
    for path in sorted((package / "verifiers").rglob("*.json")) if (package / "verifiers").is_dir() else ():
        scanned.append((path, later, later_text))
    if (package / TASKS_INDEX_NAME).is_file():
        scanned.append((package / TASKS_INDEX_NAME, later, later_text))

    checked, leaking_files, echo_files, strict = 0, [], [], 0
    for path, values, text in scanned:
        found = _file_strings(path, min_length)
        checked += len(found)
        unaccounted = {value for value in found if value in corpus and not accounted(value, values, text)}
        leaks = {value for value in unaccounted if len(value) >= content_min_length}
        echoes = unaccounted - leaks
        strict += sum(1 for value in found
                      if value in corpus and not (value in env_values or value in env_text
                                                  or value in harness))
        relative = path.relative_to(package).as_posix()
        if leaks:
            leaking_files.append({"file": relative, "values": len(leaks)})
        if echoes:
            echo_files.append({"file": relative, "values": len(echoes)})
    return {"baseline": LEAK_BASELINE, "min_length": min_length, "content_min_length": content_min_length,
            "corpus_strings": len(corpus), "files_scanned": len(scanned), "values_checked": checked,
            "leaks": sum(entry["values"] for entry in leaking_files),
            "leaking_files": leaking_files[:20],
            "value_echoes": sum(entry["values"] for entry in echo_files),
            "value_echo_files": len(echo_files),
            "strict_env_only_unaccounted": strict}


# --- the manifest -------------------------------------------------------------------


def file_hashes(package: Path) -> dict[str, str]:
    """sha256 per file of the packaged Environment.

    Two files are left out. The manifest carries this map, so it cannot carry its own hash. The card
    is prose about the package rather than part of it: it is rewritten on every publish and a person
    who edits it has not tampered with a Verifier, so a package with an edited card still verifies.
    """
    skip = {MANIFEST_NAME, CARD_NAME}
    out = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative in skip or _hidden(path.relative_to(package)):
            continue
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _hidden(relative: Path) -> bool:
    """A path any segment of which starts with a dot.

    A package carries no hidden file, and a host lays its own down beside the ones it was given: a
    download leaves `.gitattributes` and a `.cache/` tree in the directory. Hashing those would make
    every fetched package fail to verify against the manifest that was uploaded, so they are not part
    of the Environment and are not hashed on either side.
    """
    return any(part.startswith(".") for part in relative.parts)


def package_hash(package: Path) -> str:
    """One hash over every file of the package, so a fetch can tell it apart from a tampered copy."""
    return content_hash(file_hashes(package))


def verify_package(package: Path) -> list[str]:
    """Everything wrong with a package on disk, as words; an empty list is a package that verifies."""
    manifest = read_json(package / MANIFEST_NAME, None)
    if not isinstance(manifest, dict):
        return [f"no {MANIFEST_NAME} in {package}"]
    problems = []
    recorded = dict(manifest.get("files") or {})
    found = file_hashes(package)
    for name in sorted(set(recorded) | set(found)):
        if name not in found:
            problems.append(f"{name} is named in the manifest and missing from the package")
        elif name not in recorded:
            problems.append(f"{name} is in the package and named in no manifest")
        elif recorded[name] != found[name]:
            problems.append(f"{name} does not hash to what the manifest recorded")
    computed = content_hash(found)
    if manifest.get("content_hash") != computed:
        problems.append("the package does not hash to the manifest's content_hash")
    return problems


def kullback_version() -> str:
    try:
        from importlib.metadata import version

        return version("kullback")
    except Exception:  # noqa: BLE001 - a package not installed is a version nobody can report
        return "unknown"


def git_sha(root: Optional[Path] = None) -> str:
    """The commit the harness is running from, or `unknown` outside a checkout."""
    root = root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S603, S607
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def export(workdir: Any, out: Any, *, name: Optional[str] = None, corpus: Optional[str] = None,
           corpus_license: Optional[str] = None, corpus_url: Optional[str] = None,
           preview: bool = False, scan: bool = True) -> dict:
    """Write a self-contained Environment package from a workdir and answer its manifest.

    The workdir is only read. The package holds the rebuilt world, the Task list, the Verifier of
    every Task that has one, and a manifest with the numbers, the versions and a content hash; it
    holds nothing of the recordings the world was rebuilt from. A leak found by the scan raises
    rather than being reported, because a package with a leak in it must not exist on disk.
    """
    workdir, out = Path(workdir), Path(out)
    if not (workdir / "env").is_dir():
        raise ExportError(f"{workdir} holds no env/ package: there is no Environment to export")
    out.mkdir(parents=True, exist_ok=True)
    task_ids = frozen_task_ids(workdir)
    _lay_out(workdir, out, task_ids)
    rows = task_index(workdir, task_ids)
    write_json(out / TASKS_INDEX_NAME, {"format": PACKAGE_FORMAT, "tasks": rows})
    counts = last_round(workdir)
    environment = read_json(workdir / "environment.json", {}) or {}
    versions = read_json(workdir / "runner_version.json", {}) or {}
    fidelity = replay_fidelity(read_json(workdir / "replays.json", {}) or {}, task_ids)
    manifest = {
        "format": PACKAGE_FORMAT,
        "name": name,
        "preview": bool(preview),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"corpus": corpus, "license": corpus_license, "url": corpus_url},
        "round": counts.get("round"),
        "tasks_total": len(task_ids),
        # Tasks the build holds a ruling for that are not on the frozen list: they appeared after the
        # freeze, so D96 keeps them out of the denominator and the round line counts them in. Naming
        # the number is what stops the two readings from looking like a mistake.
        "tasks_added_later": len(set(read_json(workdir / "task_status.json", {}) or {}) - set(task_ids)),
        "replay_fidelity": fidelity,
        "reference_confirmed": sum(1 for row in rows if row["reference_confirmed"]),
        "verifier_derived": sum(1 for row in rows if row["verifier"]),
        "trusted": sum(1 for row in rows if row["trusted"]),
        "refused": sum(1 for row in rows if row["refused"]),
        "funnel": {stage: sum(1 for row in rows if row["stage"] == stage)
                   for stage in FUNNEL + (REFUSED_STAGE,)},
        "buckets": _buckets(rows),
        "untrusted": untrusted_reasons(rows),
        "env_id": environment.get("env_id"),
        "policy_version": environment.get("policy_version"),
        "schema_version": environment.get("schema_version"),
        "tools_version": environment.get("tools_version"),
        "assisted_tools": list(environment.get("assisted_tools") or ()),
        "runner_version": versions.get("runner_version"),
        "gates_version": versions.get("gates_version"),
        "kullback_version": kullback_version(),
        "git_sha": git_sha(),
        "leak_scan": leak_scan(out, workdir / "raw") if scan else {"baseline": LEAK_BASELINE, "leaks": 0,
                                                                   "skipped": True},
    }
    if manifest["leak_scan"].get("leaks"):
        # The files are on disk by now, because the scan reads the package rather than the workdir.
        # A package with a leak in it must not exist, so what was laid out is taken away again and
        # the caller is told in counts; the values themselves are customer data and are never said.
        shutil.rmtree(out, ignore_errors=True)
        raise ExportError(
            f"the leak scan found {manifest['leak_scan']['leaks']} values from the customer's export in "
            f"{len(manifest['leak_scan']['leaking_files'])} file(s) that the world files do not hold; "
            "the package was taken away again")
    manifest["files"] = file_hashes(out)
    manifest["content_hash"] = content_hash(manifest["files"])
    write_json(out / MANIFEST_NAME, manifest)
    return manifest


def _buckets(rows: Iterable[dict]) -> list[dict]:
    """Tasks and trusted Tasks per difficulty bucket, off the index the export just wrote."""
    records = [dict(row["difficulty"] or {}, task_id=row["task_id"], bucket=row.get("bucket"))
               for row in rows if row.get("bucket")]
    trusted = [row["task_id"] for row in rows if row.get("trusted")]
    return difficulty.bucket_rows(records, trusted)
