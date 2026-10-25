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

import ast
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import difficulty, domain, round_snapshot
from kullback.gates.verifier_suite import PROSE_MIN_LENGTH, elide_verifier_prose
from kullback.runner.records import VERDICT_VERSION, Verifier, as_dict, content_hash, read_json, write_json
from kullback.runner.verdict import REWARD_SHAPE, item_counts
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.loading import EnvironmentError as WorldError
from kullback.spec.schema import Spec, spec_path

# 2: the manifest carries what the domain reading attests and what this Environment cannot execute
# of it, so a card's numbers say how much of the domain the package covers (D225).
PACKAGE_FORMAT = 2
MANIFEST_NAME = "manifest.json"
TASKS_INDEX_NAME = "tasks_index.json"
# The same Task list as one flat row per line, which is the shape a dataset host's viewer renders.
# It repeats what tasks/ and the index already carry, so it adds nothing a scan has not seen.
TASKS_ROWS_NAME = "tasks.jsonl"
# The card publish.py writes beside the package. Named here because the hashes have to leave it out.
CARD_NAME = "README.md"
# Where a publish lays the Harbor form of the Environment beside the package (D363). It carries its own
# manifest and hashes, so the package's hashes leave it out the way they leave out the card.
HARBOR_DIR = "harbor"
# The call verdicts that count as agreement, the same three the Runner and the README name.
AGREEING_VERDICTS = ("same", "cosmetic", "both_refused")
UNRECORDED_VERDICT = "unrecorded"  # a call the replay made and the recording never did

# The rebuilt world: the Environment package the Builder compiled, the per-Task Starting states, and
# the records the Runner needs to load a toolkit over them. These are also the leak scan's baseline,
# because a value they hold is part of the world the package ships and not a quotation of a
# recording.
WORLD_DIRS: tuple[str, ...] = ("env", "overlays")
WORLD_FILES: tuple[str, ...] = ("environment.json", "schema.json", "tool_sigs.json", "bodies.json",
                                "canon-rules.json")

# The policy a Run is given, under the world's env/ (D404): the recorded agent prompt where the build has no file.
POLICY_FILE = "env/policy.md"
# The D97 versions a Verdict carries, each the content hash of one shipped file (D407).
ENVIRONMENT_FILE = "environment.json"
VERSIONED_FILES: dict[str, str] = {"schema_version": "schema.json", "tools_version": "tool_sigs.json",
                                   "policy_version": POLICY_FILE}

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
CONTENT_MIN_LENGTH = PROSE_MIN_LENGTH  # the one line, shared with the derivation's prose check (D292)
LEAK_BASELINE = ("the package's world files, the harness's own source, and, for a Verifier, the Task "
                 "instruction the candidate is handed")

# The ladder a Task climbs, in order. `funnel_stage` answers the last rung it reached.
FUNNEL: tuple[str, ...] = ("clustered", "replay_confirmed", "reference_confirmed", "verifier_derived",
                           "verifier_passed", "trusted")
REFUSED_STAGE = "refused"


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


# The one line a card and manifest carry when no round closed, so a reader knows where the counts are from.
NO_ROUND_SOURCE = "counts from the workdir status, no round closed"


def trusted_ids(workdir: Path, counts: Optional[dict] = None, *, workers: Optional[int] = 1) -> set[str]:
    """The trusted Task ids: the last round's ruling where one closed, else the Builder's status rule.

    The status rule is `status_of` in the Builder's domain tools, the same reading its status tool
    answers, so a build killed before any round closed publishes what that tool reported rather than
    nothing. The hub is the top layer and may import the Builder, so there is one rule, not a copy.
    That rule scores every Task, the slowest step of an export, so `workers` runs it on processes (D401).
    """
    counts = last_round(workdir) if counts is None else counts
    if counts:
        return set(counts.get("trusted_ids") or ())
    from kullback.builder.domain_tools import status_of
    return {row["task_id"] for row in status_of(workdir, workers=workers)["tasks"] if row.get("state") == "trusted"}


def build_tag(workdir: Path, counts: Optional[dict] = None) -> str:
    """`round-<n>` for the last round on disk, else `build-<YYYYMMDD>` of the newest session or bus write."""
    counts = last_round(workdir) if counts is None else counts
    if counts.get("round") is not None:
        return f"round-{counts['round']}"
    sessions = workdir / "sessions"
    marks = list(sessions.glob("*.jsonl")) if sessions.is_dir() else []
    marks += [path for path in (workdir / "bus.jsonl",) if path.is_file()]
    stamp = max((path.stat().st_mtime for path in marks), default=None)
    moment = datetime.fromtimestamp(stamp, timezone.utc) if stamp is not None else datetime.now(timezone.utc)
    return f"build-{moment:%Y%m%d}"


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
    """Replay fidelity over Tasks, over Runs and over calls, all over the frozen list.

    A Task clears fidelity when any replay of it was confirmed, which is the count round_end reports;
    the Run number is every replay of every Task on the list, which is the finer of the two and the
    one a reader should look at when a Task holds many recordings. The call number is every recorded
    call of those replays, agreeing when its verdict is one of AGREEING_VERDICTS (`_recorded_calls`).
    """
    ids = list(task_ids)
    tasks_confirmed, runs_total, runs_confirmed, calls_total, calls_agreeing = 0, 0, 0, 0, 0
    for task_id in ids:
        per_task = (replays or {}).get(task_id) or {}
        confirmed = [run_id for run_id, row in per_task.items() if _confirmed(row)]
        runs_total += len(per_task)
        runs_confirmed += len(confirmed)
        tasks_confirmed += 1 if confirmed else 0
        for row in per_task.values():
            recorded, agreeing = _recorded_calls(row)
            calls_total += recorded
            calls_agreeing += agreeing
    return {"tasks": tasks_confirmed, "tasks_total": len(ids), "tasks_rate": _share(tasks_confirmed, len(ids)),
            "runs": runs_confirmed, "runs_total": runs_total, "runs_rate": _share(runs_confirmed, runs_total),
            "calls": calls_agreeing, "calls_total": calls_total, "calls_rate": _share(calls_agreeing, calls_total)}


def _recorded_calls(row: Any) -> tuple[int, int]:
    """How many calls one replayed Trace recorded, and how many of them the replay agreed on.

    The whole is the recording's, never the replay's: a check the replay made for a call the
    recording never made (verdict `unrecorded`) is not counted, and a recorded call the replay never
    made (`counts.unmade`, the calls left on the recording when the replay ended) counts as one
    that did not agree. A record written before `unmade` was kept is counted from its checks alone.
    replays.json holds ReplayReport rows, whose checks sit under `calls` (`counts.calls` is only
    their number); a Runner ReplayResult keeps the same checks under `checks`, so either is read.
    """
    if not isinstance(row, dict):
        return 0, 0
    listed = row.get("calls") if isinstance(row.get("calls"), list) else row.get("checks")
    checks = [check for check in listed or ()
              if isinstance(check, dict) and check.get("verdict") != UNRECORDED_VERDICT]
    counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}
    unmade = int(counts.get("unmade") or 0)
    return len(checks) + unmade, sum(1 for check in checks if check.get("verdict") in AGREEING_VERDICTS)


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


def task_index(workdir: Path, task_ids: Iterable[str], *, workers: Optional[int] = 1) -> list[dict]:
    """One row per Task: the trust ruling, the funnel stage it reached, its difficulty record and bucket."""
    status = read_json(workdir / "task_status.json", {}) or {}
    replays = read_json(workdir / "replays.json", {}) or {}
    counts = last_round(workdir)
    trusted = trusted_ids(workdir, counts, workers=workers)
    refused = set((counts.get("refused") or {}))
    records = _difficulty_records(workdir)
    trust = _trust_rows(workdir)
    rows = []
    for task_id in task_ids:
        row = status.get(task_id) or {}
        confirmed = any(_confirmed(entry) for entry in (replays.get(task_id) or {}).values())
        verifier_file = workdir / "verifiers" / f"{task_id}.json"
        has_verifier = verifier_file.is_file()
        stage = funnel_stage(task_id, status_row=row, replay_confirmed=confirmed, has_verifier=has_verifier,
                             trusted_ids=trusted, refused=refused)
        record = records.get(task_id) or {}
        rows.append({
            "task_id": task_id,
            "trusted": stage == "trusted",
            "refused": stage == REFUSED_STAGE,
            "stage": stage,
            # D333: why a Task is not trusted, and the two flags that never gate (None: unscored).
            **{key: (trust.get(task_id) or {}).get(key) for key in TRUST_FIELDS},
            "replay_confirmed": confirmed,
            # A Task with any confirmed replay, the count replay fidelity reports, so the two agree.
            "reference_confirmed": confirmed,
            "verifier": has_verifier,
            # The Verifier's gate and scored items (D329), so a reader sees which Tasks score past pass.
            "items": item_counts(Verifier.model_validate(read_json(verifier_file))) if has_verifier else None,
            "recordings": int(row.get("recordings") or 0),
            "difficulty": {key: value for key, value in record.items() if key != "task_id"} or None,
            "bucket": record.get("bucket"),
        })
    return rows


TRUST_FIELDS = ("trust_reason", "reference_passes", "solvable")


def _trust_rows(workdir: Path) -> dict[str, dict]:
    """Each Task's trust reason and flags off the last closed round's table; none where no round closed."""
    snapshot = round_snapshot.read_snapshot(workdir) or {}
    return {str(row.get("task_id")): row for row in snapshot.get("rows") or () if isinstance(row, dict)}


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


def _domain_counts(workdir: Path) -> dict:
    """What the domain reading attests and what this Environment cannot execute of it (D225).

    Three counts and nothing else: how many archetypes were read off the domain's own public
    material, how many of them a tool of this Environment realises, and how many fell in the gap.
    They are read off the reading the build already wrote, never recounted here, and a workdir
    where nothing has read a domain carries none of them rather than carrying zeros.
    """
    try:
        counts = domain.counts_of(workdir)
    except (OSError, ValueError, TypeError):
        return {}
    named = ("archetypes_extracted", "archetypes_mapped", "archetype_gaps")
    return {key: int(counts[key]) for key in named if counts.get(key)}


# --- the package on disk ------------------------------------------------------------


def inside(package: Path, harbor: Optional[dict] = None) -> dict:
    """What a card says is inside a laid package: how many tools it serves and one Task's first ask.

    The ask is the first listed line of the first Harbor Task's instruction, the Task the card's Harbor
    commands name, else the first instruction in the viewer rows; both passed the package's leak scan.
    """
    tools = read_json(package / "tool_sigs.json", None)
    rows = list((harbor or {}).get("tasks") or ()) + list((harbor or {}).get("synthetic") or ())
    texts = []
    if rows:
        instruction = package / HARBOR_DIR / str(rows[0]["path"]) / "instruction.md"
        texts.append(instruction.read_text(encoding="utf-8").split("\n## ", 1)[0] if instruction.is_file() else "")
    viewer = package / TASKS_ROWS_NAME
    if viewer.is_file():
        texts += [json.loads(line).get("instruction") or "" for line in viewer.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    return {"tools": len(tools) if isinstance(tools, (list, dict)) else None,
            "example_ask": next((ask for ask in map(_first_ask, texts) if ask), None)}


def _first_ask(text: str) -> Optional[str]:
    """The first listed line of an instruction, else its first line."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    listed = [line[2:].strip() for line in lines if line.startswith("- ")]
    return (listed or lines or [None])[0]


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _exported_task(body: Any, intent: Optional[str]) -> dict:
    """One Task record as the package carries it: its ask as `intent`, and no id of a recording.

    The ask is the one thing a candidate is given (task_ask). The Spec and the mined Intent it comes
    from also hold spans of the recording, so neither ships; only the ask is put where `Task.intent`
    already belongs, an empty string where the Task has none, so the manifest can count it (D403).
    """
    record = dict(body) if isinstance(body, dict) else {}
    for field in RECORDING_FIELDS:
        if field in record:
            record[field] = []
    record["intent"] = intent or ""
    return record


def _intent_text(workdir: Path, task_id: str) -> Optional[str]:
    """The Intent an older build mined for one Task, text only; a Spec build writes none."""
    body = read_json(workdir / "intents" / f"{task_id}.json", None)
    text = body.get("text") if isinstance(body, dict) else None
    return str(text) if text else None


# How the user's one message opens when it lists what they volunteered.
FACTS_LEAD = "A user writes in. What they tell you, one line each:"
# The lead of what the user said later in the recording: on request, or after a refusal or a proposal (D422).
LATER_LEAD = "What they tell you later, when you ask or when what they first asked for cannot be done:"
# The stance an Intent fact has when it is the user's own ask (spec/schema.py IntentFact).
ASKED = "volunteered"


def _spec(workdir: Path, task_id: str) -> Optional[Spec]:
    try:
        return Spec.model_validate_json(spec_path(workdir, task_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class AskedFacts(list):
    """The fact lines an instruction renders, the volunteered ones first; `later` counts those at the end the
    user said later (stance fallback or accepted), which render under LATER_LEAD."""

    def __init__(self, first: Iterable[str] = (), later: Iterable[str] = ()) -> None:
        later = list(later)
        super().__init__([*first, *later])
        self.later = len(later)


def rendered(fact: Any) -> bool:
    """Whether the instruction carries this Intent fact: any stance, but no echo of the agent, nothing ruled
    out, no fact of a scenario's end, which no user said (D414), and no empty text (D422)."""
    from kullback.spec.items import is_end_fact

    return not fact.echo and not fact.unfounded and not is_end_fact(fact) and bool(fact.text.strip())


def asked_facts(spec: Optional[Spec]) -> AskedFacts:
    """The Intent facts the user said, as the user would say them (`rendered`): the volunteered ones, then what
    they said on request or after a refusal or proposal (D422). A value given when asked is the user's value."""
    if spec is None:
        return AskedFacts()
    said = [fact for fact in spec.intent.facts if rendered(fact)]
    return AskedFacts([fact.text.strip() for fact in said if fact.stance == ASKED],
                      [fact.text.strip() for fact in said if fact.stance != ASKED])


def opening(facts: list[str], intent: Optional[str]) -> Optional[str]:
    """The user's one message: the volunteered facts, one line each, then the later ones under LATER_LEAD, else
    the Task's Intent text."""
    if not facts:
        return intent.strip() if intent and intent.strip() else None
    later = getattr(facts, "later", 0)
    first, rest = list(facts)[:len(facts) - later], list(facts)[len(facts) - later:]
    lines = [FACTS_LEAD, *[f"- {fact}" for fact in first]]
    if rest:
        lines += ["", LATER_LEAD, *[f"- {fact}" for fact in rest]]
    return "\n".join(lines)


def task_ask(workdir: Path, task_id: str, record: Any) -> tuple[list[str], Optional[str]]:
    """A Task's ask, the one both the package and its Harbor form ship (D403): the facts the user
    volunteered in its Spec Intent and the message they make, else the Task's own or mined Intent text."""
    body = record if isinstance(record, dict) else {}
    facts = asked_facts(_spec(workdir, task_id))
    return facts, opening(facts, body.get("intent") or _intent_text(workdir, task_id))


def without_ask_count(package: Path, task_ids: Iterable[str]) -> int:
    """How many of the Tasks a laid package carries no ask for."""
    return sum(1 for task_id in task_ids if not (read_json(package / "tasks" / f"{task_id}.json", {}) or {}).get("intent"))


def environment_policy(workdir: Path, task_ids: Iterable[str]) -> Optional[str]:
    """The policy text a fetched package's Runs are given (D404), or None where there is no one text.

    The workdir's env/policy.md where it has one, else the recorded agent's system prompt, read the way a
    Run reads it and the Harbor form lays it per Task, when every listed Task runs under the same one. A
    package carries no Traces, so Tasks that ran under different prompts cannot share one file: none is
    written and the fetch names env/policy.md as missing.
    """
    shipped = workdir / POLICY_FILE
    if shipped.is_file():
        return shipped.read_text(encoding="utf-8")
    try:
        env = BuiltEnvironment(workdir)
        prompts = {env.system_prompt(env.task(task_id)) for task_id in task_ids}
    except (OSError, ValueError, WorldError):
        return None
    return next(iter(prompts)) if len(prompts) == 1 and None not in prompts else None


def environment_record(directory: Path) -> Optional[dict]:
    """environment.json for a workdir or a package, written where it lacks the D97 versions (D407).

    Each version is the sha256 of the file it names, the same hash the manifest keeps for that file, and
    env_id hashes the three, so a build workdir and the package laid from it carry the same values. The
    policy is the text a package ships (environment_policy). A record that already names all four is kept.
    Where a versioned file is missing or no single policy exists it writes nothing and answers the record
    on disk as it stands, or None where there is none.
    """
    directory = Path(directory)
    path = directory / ENVIRONMENT_FILE
    record = read_json(path, {}) if path.is_file() else {}
    record = record if isinstance(record, dict) else {}
    if all(record.get(field) for field in ("env_id", *VERSIONED_FILES)):
        return record
    policy = environment_policy(directory, frozen_task_ids(directory))
    if policy is None:
        return record or None
    versions = {}
    for field, name in VERSIONED_FILES.items():
        source = directory / name
        if name == POLICY_FILE:
            versions[field] = hashlib.sha256(policy.encode("utf-8")).hexdigest()
        elif source.is_file():
            versions[field] = hashlib.sha256(source.read_bytes()).hexdigest()
        else:
            return record or None
    record = {**record, "env_id": content_hash(versions), **versions}
    write_json(path, record)
    return record


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
    policy = environment_policy(workdir, task_ids)
    if policy and not (out / POLICY_FILE).is_file():
        (out / POLICY_FILE).parent.mkdir(parents=True, exist_ok=True)
        (out / POLICY_FILE).write_text(policy, encoding="utf-8")
        written.append(POLICY_FILE)
    if environment_record(out) and ENVIRONMENT_FILE not in written:
        written.append(ENVIRONMENT_FILE)
    for task_id in task_ids:
        source = workdir / "tasks" / f"{task_id}.json"
        if source.is_file():
            record = read_json(source, {})
            write_json(out / "tasks" / f"{task_id}.json", _exported_task(record, task_ask(workdir, task_id, record)[1]))
            written.append(f"tasks/{task_id}.json")
        verifier = workdir / "verifiers" / f"{task_id}.json"
        if verifier.is_file():
            _copy_verifier(verifier, out / "verifiers" / f"{task_id}.json")
            written.append(f"verifiers/{task_id}.json")
    _refuse_forbidden(written)
    return written


def _copy_verifier(source: Path, target: Path) -> int:
    """Copy one Verifier, with the recorded agent's own prose taken out of it (D292), and count the atoms.

    A Verifier derived before D292 can pin free text the recorded agent wrote into a tool argument.
    The atom is reduced here exactly as the derivation now reduces it, so a finished build packages
    without a rebuild, and the workdir is left as it is: its Verifier versions are named by content
    hash in the exam history, the probe pools and the rulings, and a rewrite there would orphan them.
    The reduced atom was an allowed one, which no check scores, so the package grades as the workdir.
    A Verifier with nothing to reduce is copied byte for byte.
    """
    body = read_json(source, None)
    if not isinstance(body, dict):
        _copy(source, target)
        return 0
    verifier, changed = elide_verifier_prose(Verifier.model_validate(body))
    if not changed:
        _copy(source, target)
        return 0
    write_json(target, as_dict(verifier))
    return changed


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
    try:
        if path.suffix == ".json":
            _strings(json.loads(path.read_text(encoding="utf-8")), out, min_length)
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _strings(json.loads(line), out, min_length)
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


def _literals(text: str) -> list[str]:
    """The string constants of a string that parses as Python source (a predicate), else none.

    A predicate quotes a value through repr, which escapes quotes and line breaks, so the value is
    read back as the constant it is rather than searched for in the escaped text.
    """
    if "(" not in text:
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _prose_index(corpus: set[str], content_min_length: int) -> dict[str, list[str]]:
    """The corpus's prose strings keyed by their opening window, to find one quoted inside another string."""
    index: dict[str, list[str]] = {}
    for value in corpus:
        if len(value) >= content_min_length:
            index.setdefault(value[:content_min_length], []).append(value)
    return index


def embedded_prose(found: Iterable[str], index: dict[str, list[str]], width: int) -> set[str]:
    """Corpus prose a string quotes inside itself: a description or a predicate that repeats a turn (D292).

    The whole-string reading only sees a leaf that is itself a corpus string; a sentence that says
    "the field is <a recorded paragraph>" is not one, and neither is a predicate that passes it as an
    argument. Each window of the string is looked up by the prose's opening, so the cost is the length
    of what is scanned and not the size of the corpus.
    """
    out: set[str] = set()
    for value in found:
        for text in [value, *_literals(value)]:
            for start in range(len(text) - width + 1):
                for prose in index.get(text[start:start + width], ()):
                    if prose != value and text.startswith(prose, start):
                        out.add(prose)
    return out


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
    index = _prose_index(corpus, content_min_length)

    def accounted(value: str, values: set[str], text: str) -> bool:
        return value in values or value in text or value in harness

    scanned: list[tuple[Path, set[str], str]] = []
    for path in sorted((package / "tasks").rglob("*.json")) if (package / "tasks").is_dir() else ():
        scanned.append((path, world_values, world_text))
    later = world_values | task_values
    later_text = world_text + "\n" + task_text
    for path in sorted((package / "verifiers").rglob("*.json")) if (package / "verifiers").is_dir() else ():
        scanned.append((path, later, later_text))
    for name in (TASKS_INDEX_NAME, TASKS_ROWS_NAME):
        if (package / name).is_file():
            scanned.append((package / name, later, later_text))

    checked, leaking_files, echo_files, strict = 0, [], [], 0
    for path, values, text in scanned:
        found = _file_strings(path, min_length)
        checked += len(found)
        found |= embedded_prose(found, index, content_min_length)
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
    The Harbor form a publish lays beside the package is left out too: it hashes itself (D363).
    """
    skip = {MANIFEST_NAME, CARD_NAME}
    out = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative in skip or _hidden(path.relative_to(package)) or relative.startswith(f"{HARBOR_DIR}/"):
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


def _row_counts(rows: list[dict]) -> dict:
    """The manifest's Task counts: confirmed, derived, trusted, refused, and each funnel stage."""
    return {
        "reference_confirmed": sum(1 for row in rows if row["reference_confirmed"]),
        "verifier_derived": sum(1 for row in rows if row["verifier"]),
        "trusted": sum(1 for row in rows if row["trusted"]),
        "refused": sum(1 for row in rows if row["refused"]),
        "trust_reasons": {reason: sum(1 for row in rows if row.get("trust_reason") == reason)
                          for reason in sorted({row["trust_reason"] for row in rows if row.get("trust_reason")})},
        **{flag: {"true": sum(1 for row in rows if row.get(flag) is True),
                  "false": sum(1 for row in rows if row.get(flag) is False)} for flag in TRUST_FIELDS[1:]},
        "funnel": {stage: sum(1 for row in rows if row["stage"] == stage)
                   for stage in FUNNEL + (REFUSED_STAGE,)},
    }


def export(workdir: Any, out: Any, *, name: Optional[str] = None, corpus: Optional[str] = None,
           corpus_license: Optional[str] = None, corpus_url: Optional[str] = None,
           preview: bool = False, scan: bool = True, workers: Optional[int] = None,
           index: Optional[list[dict]] = None) -> dict:
    """Write a self-contained Environment package from a workdir and answer its manifest.

    The workdir is only read. The package holds the rebuilt world, the Task list, the Verifier of
    every Task that has one, and a manifest with the numbers, the versions and a content hash; it
    holds nothing of the recordings the world was rebuilt from. A leak found by the scan raises
    rather than being reported, because a package with a leak in it must not exist on disk.
    `workers` is how many processes rule the Tasks (None: the default of kullback/processes.py; the package is the same
    whatever the count, D401), and `index` is the Task index where the caller already holds it.
    """
    workdir, out = Path(workdir), Path(out)
    if not (workdir / "env").is_dir():
        raise ExportError(f"{workdir} holds no env/ package: there is no Environment to export")
    out.mkdir(parents=True, exist_ok=True)
    task_ids = frozen_task_ids(workdir)
    _lay_out(workdir, out, task_ids)
    without_ask = without_ask_count(out, task_ids)
    rows = task_index(workdir, task_ids, workers=workers) if index is None else index
    write_json(out / TASKS_INDEX_NAME, {"format": PACKAGE_FORMAT, "tasks": rows})
    write_task_rows(out, rows)
    counts = last_round(workdir)
    environment = read_json(out / ENVIRONMENT_FILE, {}) or {}
    versions = read_json(workdir / "runner_version.json", {}) or {}
    fidelity = replay_fidelity(read_json(workdir / "replays.json", {}) or {}, task_ids)
    manifest = {
        "format": PACKAGE_FORMAT,
        "name": name,
        "preview": bool(preview),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"corpus": corpus, "license": corpus_license, "url": corpus_url},
        "round": counts.get("round"),
        "tag": build_tag(workdir, counts),
        "counts_source": "the last round record" if counts else NO_ROUND_SOURCE,
        "tasks_total": len(task_ids),
        # D403: Tasks the package ships with an empty ask, because their Spec Intent volunteered nothing.
        "without_ask": without_ask,
        # Tasks the build holds a ruling for that are not on the frozen list: they appeared after the
        # freeze, so D96 keeps them out of the denominator and the round line counts them in. Naming
        # the number is what stops the two readings from looking like a mistake.
        "tasks_added_later": len(set(read_json(workdir / "task_status.json", {}) or {}) - set(task_ids)),
        "replay_fidelity": fidelity,
        **_row_counts(rows),
        "buckets": _buckets(rows),
        # D329: what a Run's reward is, pass and score, under the Verdict version the graders score at.
        "reward": dict(REWARD_SHAPE, verdict_version=VERDICT_VERSION),
        # D225: what the domain's own public material was read to attest, and how much of it this
        # Environment holds no tool for. Both are counts and nothing of any page travels with them.
        "domain": _domain_counts(workdir),
        "env_id": environment.get("env_id"),
        "policy_version": environment.get("policy_version"),
        "schema_version": environment.get("schema_version"),
        "tools_version": environment.get("tools_version"),
        "assisted_tools": list(environment.get("assisted_tools") or ()),
        "runner_version": versions.get("runner_version"),
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


def task_rows(package: Path, index_rows: Iterable[dict]) -> list[dict]:
    """The flat row per Task a dataset viewer shows: id, instruction, funnel stage, bucket, trust."""
    rows = []
    for row in index_rows:
        task = read_json(package / "tasks" / f"{row['task_id']}.json", {}) or {}
        rows.append({
            "task_id": row["task_id"],
            "instruction": str(task.get("intent") or ""),
            "stage": row["stage"],
            "trusted": bool(row["trusted"]),
            "refused": bool(row["refused"]),
            "bucket": row.get("bucket"),
            "replay_confirmed": bool(row["replay_confirmed"]),
            "recordings": int(row["recordings"]),
        })
    return rows


def write_task_rows(package: Path, index_rows: Iterable[dict]) -> Path:
    target = package / TASKS_ROWS_NAME
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in task_rows(package, index_rows)]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def _buckets(rows: Iterable[dict]) -> list[dict]:
    """Tasks and trusted Tasks per difficulty bucket, off the index the export just wrote."""
    records = [dict(row["difficulty"] or {}, task_id=row["task_id"], bucket=row.get("bucket"))
               for row in rows if row.get("bucket")]
    trusted = [row["task_id"] for row in rows if row.get("trusted")]
    return difficulty.bucket_rows(records, trusted)
