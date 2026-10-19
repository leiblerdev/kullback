"""An Environment as a Harbor dataset: one Harbor task directory per Kullback Task (D363).

Harbor (harborframework.com, task schema 1.3) is the task format Terminal-Bench 2 runs on: a task is
a directory holding instruction.md, task.toml, environment/Dockerfile, tests/test.sh, which writes
the reward to /logs/verifier/reward.txt, and solution/solve.sh, the oracle. This module writes one
such directory per Task and a manifest over them.

What each part holds:

- environment/ is the image: a pinned Python base, the kullback wheel, and the one Task's world
  (the rendered toolkit, the Starting state, the Task's overlay, the policy). The image never holds
  a Verifier, a Reference, a recording or a recorded call: the toolkit's own recorded-call files stay
  out, because they are the answers.
- instruction.md is what the user asks, rendered from the Intent's volunteered facts (never from a
  transcript), then the policy the agent works under, then how to reach the tools.
- tests/ holds the Verifier and test.sh, which plays the agent's call log and scores it by code
  (kullback/hub/harbor_world.py). Harbor uploads tests/ at verify time, after the agent is gone.
- solution/solve.sh replays a passing Run's writes through the `tools` CLI, asking the user first
  and ending on a reply that says what the Verifier needs said, judge answers included. It never
  quotes a recorded turn, and a prose-length argument is put as a neutral line. The export plays it and keeps it only if it passes.

The leak scan is the package's (D221), run over what a Harbor task adds: the instruction's lines,
the oracle's calls and reply, the Verifier and a rater's words in its metadata. A leak removes the
dataset and refuses the export, naming each leaking Task (masked) and part and how many strings.

`--tasks verified` (D399) exports every recorded Task with a Verifier on disk, trusted or not, each
marked with its Spec tier and tier reason (tiers.json) and, given label files, a rater's label.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import processes
from kullback.hub import harbor_world, listed_ids, repair
from kullback.hub.card import ORGANISATION, SOLUTION_LEAKS
from kullback.hub.harbor_world import DB_NAME, JUDGE_ENV, REPLY_TOOL, TASK_NAME, UNSCORED_EXIT, WORLD_DIR, play
from kullback.hub.listed_ids import READS_DROPPED_ID
from kullback.hub.package import (
    CONTENT_MIN_LENGTH,
    FACTS_LEAD,
    LATER_LEAD,
    LEAK_MIN_LENGTH,
    MANIFEST_NAME,
    WORLD_FILES,
    ExportError,
    _baseline,
    _file_strings,
    _prose_index,
    _spec,
    _strings,
    build_tag,
    content_hash,
    corpus_strings,
    embedded_prose,
    file_hashes,
    frozen_task_ids,
    git_sha,
    harness_text,
    kullback_version,
    opening,
    rendered,
    task_ask,
    task_index,
)
from kullback.runner.records import Run, Verifier, as_dict, read_json, write_json
from kullback.runner.verdict import REWARD_SHAPE
from kullback.runner.world.environment import BuiltEnvironment
from kullback.spec.items import unreadable_path
from kullback.spec.report import REPORT_FILE as TIERS_FILE
from kullback.spec.schema import spec_verifier_path
from kullback.spec.split import reference_classes
from kullback.spec.trust import NO_VERIFIER as NO_VERIFIER_REASON
from kullback.spec.trust import SYNTHETIC as SYNTHETIC_TIER
from kullback.spec.trust import runs_on_disk
from kullback.synthesise import DIR as SYNTHETIC_DIR

HARBOR_FORMAT = 1
SCHEMA_VERSION = "1.3"
# python:3.12.7-slim-bookworm, by digest, so a rebuild months later runs the same interpreter.
BASE_IMAGE = ("python:3.12.7-slim-bookworm@sha256:"
              "60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d")
VERIFIER_TIMEOUT_SEC = 120.0
AGENT_TIMEOUT_SEC = 900.0
BUILD_TIMEOUT_SEC = 600.0
# The world the image carries: the toolkit, the Starting state and the policy, never env/'s recorded
# calls or its source Task list.
ENV_FILES: tuple[str, ...] = ("data_model.py", "tools.py", "db.json")
POLICY_NAME = "policy.md"
VERIFIER_NAME = "verifier.json"
ORACLE_NAME = "oracle.json"
# A Task none of whose Runs, replayed through the exported world, scores 1.0: it ships no oracle.
NO_ORACLE = "no Run passes on replay"
# A Task whose Verifier holds a cell the scorer cannot read on its start row: it can never hold (D423).
UNREADABLE_CELL = "unreadable_cell"
ORACLE_PART = "oracle"
MASK = 4
PROSE_LINE = "The user asked for this change; the details are in the conversation."
TASKS_CHOICES = ("trusted", "verified", "all")
# A Task with no row in the tier report, and one with no row in the label files given.
NOT_RECORDED = "not_recorded"
UNLABELLED = "unlabelled"
# What a label row carries into a task's metadata, under these names.
LABEL_FIELDS = (("label", "label"), ("matches_gold", "label_matches_gold"), ("gold_diff", "label_gold_diff"))
TOOLS_LINE = ("The tools are served two ways from the same world: the MCP server named `tools` in this task, "
              "and the `tools` command (`tools list`, then `tools call <name> --json '{...}'`). Everything "
              f"you would say to the user goes through `{REPLY_TOOL}`; the user answers each message, and "
              "your last message is your final reply.")
ASK_FIRST = "I am about to make the changes you asked for. Shall I go ahead?"
DONE_LINE = "Done."


# --- what one Task is made of -------------------------------------------------------


def verifier_of(workdir: Path, task_id: str, synthetic: bool = False) -> Optional[Path]:
    """The Verifier the trust ruling scored: the Spec's where one was written, else the build's."""
    if synthetic:
        candidates = [workdir / SYNTHETIC_DIR / "verifiers" / f"{task_id}.json"]
    else:
        candidates = [spec_verifier_path(workdir, task_id), workdir / "verifiers" / f"{task_id}.json"]
    return next((path for path in candidates if path.is_file()), None)


# Why a dropped state item's cells are free in the shipped end states (D422).
UNRENDERABLE_WHY = "dropped: rests only on facts the instruction does not render"
STATE_ROW_KINDS = ("row_is", "row_keeps")


def _cited(atom: Any) -> set[str]:
    return {str(fact_id) for fact_id in ((atom.target or {}).get("from") or {}).get("fact_ids") or ()}


def unrenderable(verifier: Verifier, spec: Any) -> list[str]:
    """The ids of the items that cite facts and every one a fact the instruction does not render (D422): a
    right Run cannot be told them, so they are dropped, not weighed. A scenario's end facts are what the Verifier
    checks and no user says (D414), so an item on them stays; so does an item citing no fact, and one citing a
    policy line, which the instruction carries."""
    if spec is None:
        return []
    from kullback.spec.items import is_end_fact

    hidden = {fact.id for fact in spec.intent.facts if not rendered(fact) and not is_end_fact(fact)}
    return [atom.id for atom in verifier.atoms
            if _cited(atom) and _cited(atom) <= hidden and not ((atom.target or {}).get("from") or {}).get("policy_line")]


def _cells(targets: Iterable[dict]) -> set[tuple]:
    """(table, row, field) of every cell the state items among `targets` expect."""
    return {(t.get("table"), t.get("row"), field) for t in targets if t.get("kind") in STATE_ROW_KINDS
            for field in t.get("expect") or {}}


def _freed(state: Any, freed: set[tuple], gone: set[str], tables: list[str]) -> Any:
    """One end state without the freed cells and the dropped new rows, which it now allows."""
    allowed = [{"table": table, "row_id": row, "field": field, "why": UNRENDERABLE_WHY}
               for table, row, field in sorted(freed, key=str)] + [{"table": table} for table in tables]
    return state.model_copy(update={
        "cells": [cell for cell in state.cells if (cell.table, cell.row_id, cell.field) not in freed],
        "new_rows": [row for row in state.new_rows if row.get("item") not in gone],
        "allowed": list(state.allowed) + allowed})


def without_items(verifier: Verifier, dropped: Iterable[str]) -> Verifier:
    """The Verifier without the named items. A dropped state item's cells leave every end state and the cells
    become free, so a Run may leave the row as it was or move it; a dropped new row frees its table."""
    gone = set(dropped)
    kept = [atom for atom in verifier.atoms if atom.id not in gone]
    out = [atom.target or {} for atom in verifier.atoms if atom.id in gone]
    freed = _cells(out) - _cells(atom.target or {} for atom in kept)
    tables = sorted({str(t.get("table")) for t in out if t.get("kind") == "row_new"})
    return verifier.model_copy(update={"atoms": kept,
                                       "expected": [_freed(state, freed, gone, tables) for state in verifier.expected]})


def _unprosed(value: Any) -> Any:
    """`value` with every string long enough to be prose (the leak scan's bar) put as PROSE_LINE: a
    recorded Run's free text, a summary or a note, is the recorded agent's words and never ships."""
    if isinstance(value, dict):
        return {key: _unprosed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unprosed(item) for item in value]
    return PROSE_LINE if isinstance(value, str) and len(value) >= CONTENT_MIN_LENGTH else value


def _calls(run: Run, tools: set[str]) -> list[dict]:
    return [{"name": str(event.payload.get("name")), "arguments": _unprosed(dict(event.payload.get("args") or {}))}
            for event in run.events
            if event.type == "tool_call" and str(event.payload.get("name")) in tools]


def candidate_calls(workdir: Path, task_id: str, tools: set[str], runs: Iterable[Run]) -> list[list[dict]]:
    """The tool calls of each Run of the Task (`runs`, as the trust ruling reads them off runs/), reads and
    writes in order, since a Verifier may ask that a read come before a write; its kept References first,
    then the others by run id; a list another Run already gave is left out. The oracle is the first that
    passes."""
    row = (read_json(workdir / "references.json", {}) or {}).get(task_id) or {}
    order = [str((reference or {}).get("run_id") or "") for reference in row.get("references") or ()]
    rank = {run_id: i for i, run_id in reversed(list(enumerate(order)))}
    out: list[list[dict]] = []
    for run in sorted(runs, key=lambda run: (rank.get(run.run_id, len(order)), run.run_id)):
        calls = _calls(run, tools)
        if calls not in out:
            out.append(calls)
    return out


def reference_calls(workdir: Path, task_id: str, runs: Iterable[Run]) -> list[dict]:
    """Every tool call of the Task's kept References, arguments as recorded (D429 reads them for dropped ids)."""
    row = (read_json(workdir / "references.json", {}) or {}).get(task_id) or {}
    kept = {str((reference or {}).get("run_id") or "") for reference in row.get("references") or ()}
    return [{"name": str(event.payload.get("name")), "arguments": dict(event.payload.get("args") or {})}
            for run in runs if run.run_id in kept for event in run.events if event.type == "tool_call"]


def oracle_holds(folder: Path) -> bool:
    """Whether the task's own solve steps, played on a copy of its image's /app and scored by its tests,
    hold every item, a judge item left unscored aside: reward 1.0 where no judge gate stands, and what
    solve.sh then test.sh give in the container."""
    with tempfile.TemporaryDirectory() as scratch:
        app = Path(scratch) / "app"
        shutil.copytree(folder / "environment", app)
        for step in solve_steps(*_solution(folder)):
            harbor_world.call(app, step["name"], step["arguments"])
        body = harbor_world.score(app, folder / "tests" / VERIFIER_NAME, Path(scratch) / "logs", use_judge=False)
    rows = body.get("items") or []
    return bool(rows) and all(row["held"] is True or (row["kind"] == "judge" and row["held"] is None) for row in rows)


def said_values(verifier: Verifier) -> list[str]:
    """Every value a `said` item of the Verifier needs the agent to have said, then the Spec's own answer
    each judge item is held to (its anchor), in item order."""
    out: list[str] = []
    for atom in verifier.atoms:
        target = atom.target or {}
        if target.get("kind") == "said":
            out += [str(value) for value in target.get("values") or () if str(value) not in out]
    answers = [((atom.target or {}).get("anchor") or {}).get("answer") for atom in verifier.atoms if atom.judge]
    return out + [str(answer) for answer in answers if answer and str(answer) not in out]


def needs_judge(verifier: Verifier) -> bool:
    """Whether a gate of the Verifier is a judge item: without a judge such a Task can only score 0."""
    return any(atom.gate and atom.judge for atom in verifier.atoms)


def confirm_gate(verifier: Verifier) -> bool:
    """Whether a gate asks the user's yes before a write, which the trial's consent user answers."""
    return any(atom.gate and (atom.target or {}).get("first") == "confirm_turn" for atom in verifier.atoms)


def _synthetic_tasks(workdir: Path) -> dict[str, dict]:
    folder = workdir / SYNTHETIC_DIR / "tasks"
    out = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        body = read_json(path, None)
        if isinstance(body, dict) and isinstance(body.get("task"), dict):
            out[path.stem] = body
    return out


# --- the tier and the label of each Task (D399) ------------------------------------


def _owner(key: str, exact: set[str], ids: list[str]) -> Optional[str]:
    """The Task a row's key names: the id itself, else the one id ending with it (a report masks an id to
    its last characters); None where no Task or more than one does."""
    if key in exact:
        return key
    ends = [task_id for task_id in ids if key and task_id.endswith(key)]
    return ends[0] if len(ends) == 1 else None


def keyed_rows(pairs: Iterable[tuple[str, dict]], task_ids: Iterable[str]) -> tuple[dict[str, dict], int]:
    """Each (key, row) given to the Task its key names, a later row winning a clash; answers the rows by
    Task id and how many keys named no single Task."""
    ids = list(task_ids)
    exact = set(ids)
    out: dict[str, dict] = {}
    unmatched = 0
    for key, row in pairs:
        owner = _owner(str(key), exact, ids)
        if owner is None:
            unmatched += 1
            continue
        out[owner] = row
    return out, unmatched


def tier_pairs(path: Path) -> list[tuple[str, dict]]:
    """A tier report's rows: tiers.json's `rows`, each with its task_id, or a `tasks` map from id to row."""
    body = read_json(path, {}) or {}
    if not isinstance(body, dict):
        raise ExportError(f"{path} is not a tier report")
    if isinstance(body.get("tasks"), dict):
        return [(str(key), row) for key, row in body["tasks"].items() if isinstance(row, dict)]
    return [(str(row.get("task_id") or ""), row) for row in body.get("rows") or () if isinstance(row, dict)]


def label_pairs(path: Path) -> list[tuple[str, dict]]:
    """A label file's rows: one JSON list, each row keyed by its `task`."""
    body = read_json(path, None)
    if not isinstance(body, list):
        raise ExportError(f"{path} is not a JSON list of label rows")
    return [(str(row.get("task") or row.get("task_id") or ""), row) for row in body if isinstance(row, dict)]


def tier_cell(spec_tier: str, reason: Optional[str]) -> str:
    """One word for a Task's tier as the run reports it: trusted, else the untrusted reason, else the tier."""
    return reason if spec_tier != "trusted" and reason else spec_tier


# --- writing one Harbor task --------------------------------------------------------


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _verifier_lines(judged: bool) -> str:
    """A Task with judge items lets the test reach a judge: the model and keys come from the host at
    test time (empty when unset, so an unset judge leaves the trial unscored, D405), and only the verifier phase has a
    network; the agent's never does."""
    if not judged:
        return ""
    env = "\n".join(f"{name} = {_toml('${' + name + ':-}')}" for name in JUDGE_ENV)
    return f'network_mode = "public"\n\n[verifier.env]\n{env}\n'


def task_toml(*, name: str, description: str, keywords: list[str], metadata: dict, org: str,
              judged: bool = False) -> str:
    meta = "\n".join(f"{key} = {_toml(value)}" for key, value in metadata.items() if value is not None)
    return f"""schema_version = {_toml(SCHEMA_VERSION)}

[task]
name = {_toml(name)}
authors = [{{ name = {_toml(org)} }}]
description = {_toml(description)}
keywords = {_toml(keywords)}

[metadata]
{meta}

[verifier]
timeout_sec = {_toml(VERIFIER_TIMEOUT_SEC)}
{_verifier_lines(judged)}
[agent]
timeout_sec = {_toml(AGENT_TIMEOUT_SEC)}

[environment]
build_timeout_sec = {_toml(BUILD_TIMEOUT_SEC)}
network_mode = "no-network"
workdir = "/app"

[[environment.mcp_servers]]
name = "tools"
transport = "stdio"
command = "python"
args = ["-m", "kullback.hub.harbor_world", "mcp"]
"""


def dockerfile(wheel_name: str) -> str:
    return f"""FROM {BASE_IMAGE}
WORKDIR /app
COPY {wheel_name} /tmp/{wheel_name}
RUN pip install --no-cache-dir /tmp/{wheel_name} && rm /tmp/{wheel_name}
RUN printf '#!/bin/sh\\nexec python -m kullback.hub.harbor_world tools "$@"\\n' > /usr/local/bin/tools \\
    && chmod +x /usr/local/bin/tools
COPY {WORLD_DIR}/ /app/{WORLD_DIR}/
COPY {DB_NAME} /app/{DB_NAME}
"""


TEST_SH = f"""#!/bin/bash
# Plays the agent's call log from the Starting state and scores the Run by code (kullback D363).
mkdir -p /logs/verifier
cd /app
python -m kullback.hub.harbor_world verify /tests/{VERIFIER_NAME} --logs /logs/verifier
status=$?
# {UNSCORED_EXIT}: a judge item went unanswered, so the trial is unscored and its reward.json says why (D405).
if [ "$status" -ne 0 ] && [ "$status" -ne {UNSCORED_EXIT} ]; then
  echo 0 > /logs/verifier/reward.txt
  echo '{{"reward": 0}}' > /logs/verifier/reward.json
fi
exit $status
"""


def solve_steps(calls: list[dict], reply: str) -> list[dict]:
    """The oracle's tool calls: the user asked first where it writes, the writes, then the final reply."""
    steps = [{"name": REPLY_TOOL, "arguments": {"text": ASK_FIRST}}] if calls else []
    return steps + calls + [{"name": REPLY_TOOL, "arguments": {"text": reply}}]


def solve_sh(calls: list[dict], reply: str) -> str:
    lines = ["#!/bin/bash", "# A passing Run's writes through the tools CLI, asked first, then the final reply."]
    for step in solve_steps(calls, reply):
        lines.append(f"tools call {shlex.quote(step['name'])} --json "
                     f"{shlex.quote(json.dumps(step['arguments'], ensure_ascii=False))}")
    return "\n".join(lines) + "\n"


def _solution(folder: Path) -> tuple[list[dict], str]:
    body = read_json(folder / "solution" / ORACLE_NAME, {}) or {}
    return list(body.get("calls") or ()), str(body.get("reply") or "")


def _lay_world(workdir: Path, target: Path, task_id: str, task_record: dict, instruction: str,
               policy: Optional[str]) -> None:
    """The one Task's world under `target`: what HarborEnvironment loads, and nothing else."""
    for name in WORLD_FILES:
        if (workdir / name).is_file():
            shutil.copyfile(workdir / name, target / name)
    (target / "env").mkdir(parents=True, exist_ok=True)
    for name in ENV_FILES:
        source = workdir / name if name == DB_NAME and (workdir / name).is_file() else workdir / "env" / name
        if source.is_file():
            shutil.copyfile(source, target / "env" / name)
    if policy:
        (target / "env" / POLICY_NAME).write_text(policy, encoding="utf-8")
    overlay = workdir / "overlays" / f"{task_id}.json"
    if overlay.is_file():
        (target / "overlays").mkdir(exist_ok=True)
        shutil.copyfile(overlay, target / "overlays" / f"{task_id}.json")
    record = {key: value for key, value in task_record.items() if key not in ("run_ids", "anchor_run_ids")}
    write_json(target / "tasks" / f"{task_id}.json", dict(record, id=task_id, run_ids=[], intent=instruction))
    write_json(target / TASK_NAME, {"task_id": task_id, "instruction": instruction})


def prune_world(world: Path, task_id: str, plan: dict) -> None:
    """Drop the planned list members from a laid world: the shared rows and the rows the Task's overlay pins (D429)."""
    if not plan:
        return
    db = read_json(world / "env" / DB_NAME, {}) or {}
    for table, gone in plan.items():
        listed_ids.prune_rows(db.get(table), gone)
    write_json(world / "env" / DB_NAME, db)
    overlay = world / "overlays" / f"{task_id}.json"
    if overlay.is_file():
        payload = read_json(overlay, {}) or {}
        listed_ids.prune_overlay(payload, plan)
        write_json(overlay, payload)


def unreadable_cells(verifier: Verifier, start: dict) -> list[str]:
    """Each cell of the Verifier's end states whose path the scorer cannot read on its start row (D423)."""
    found = []
    for end_state in verifier.expected:
        for cell in end_state.cells:
            row = (start.get(cell.table) or {}).get(cell.row_id)
            why = unreadable_path(row, cell.field) if cell.field else ""
            if why:
                found.append(f"{cell.table}.{cell.row_id}: {why}")
    return found


def starting_state(app: Path) -> dict:
    """The world the agent starts in, as `db.json` shows it: the Task's world played with no call."""
    with tempfile.TemporaryDirectory() as scratch:
        played = play(app, [], Path(scratch))
        return played.episode._router.world()  # noqa: SLF001 - the Episode exposes no world reader


def write_task(directory: Path, *, workdir: Path, task_id: str, task_record: dict, instruction: str,
               policy: Optional[str], verifier_path: Path, calls: Optional[list[dict]], wheel: Path,
               toml: str, prune: Optional[dict] = None) -> None:
    """One Harbor task directory, laid out and checked to load before anything else reads it; with `calls`
    None it has no solution/ (no Run passes on replay, D399). `prune` names the listed ids its world drops (D429)."""
    environment = directory / "environment"
    world = environment / WORLD_DIR
    world.mkdir(parents=True, exist_ok=True)
    _lay_world(workdir, world, task_id, task_record, instruction, policy)
    prune_world(world, task_id, prune or {})
    write_json(environment / DB_NAME, starting_state(environment))
    shutil.copyfile(wheel, environment / wheel.name)
    (environment / "Dockerfile").write_text(dockerfile(wheel.name), encoding="utf-8")
    text = "\n\n".join(part for part in (instruction, f"## Policy\n\n{policy}" if policy else "",
                                         f"## Tools\n\n{TOOLS_LINE}") if part)
    (directory / "instruction.md").write_text(text + "\n", encoding="utf-8")
    (directory / "task.toml").write_text(toml, encoding="utf-8")
    (directory / "tests").mkdir(exist_ok=True)
    shutil.copyfile(verifier_path, directory / "tests" / VERIFIER_NAME)
    (directory / "tests" / "test.sh").write_text(TEST_SH, encoding="utf-8")
    (directory / "tests" / "test.sh").chmod(0o755)
    if calls is None:
        return
    verifier = Verifier.model_validate(read_json(verifier_path, {}))
    reply = " ".join([DONE_LINE, *said_values(verifier)])
    (directory / "solution").mkdir(exist_ok=True)
    write_json(directory / "solution" / ORACLE_NAME, {"calls": calls, "reply": reply})
    (directory / "solution" / "solve.sh").write_text(solve_sh(calls, reply), encoding="utf-8")
    (directory / "solution" / "solve.sh").chmod(0o755)


# --- the wheel the image installs ---------------------------------------------------


def build_wheel(out: Path) -> Path:
    """Build the running kullback as a wheel into `out`; the image installs exactly this code."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise ExportError(f"{root} holds no pyproject.toml: kullback runs from no checkout to build a wheel from")
    done = subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out), str(root)],  # noqa: S603, S607
                          capture_output=True, text=True, timeout=600)
    wheels = sorted(out.glob("kullback-*.whl"))
    if done.returncode != 0 or not wheels:
        raise ExportError(f"the kullback wheel did not build: {done.stderr.strip()[-400:]}")
    return wheels[-1]


# --- the leak scan ------------------------------------------------------------------


def _text_strings(texts: Iterable[str], min_length: int) -> set[str]:
    out: set[str] = set()
    for text in texts:
        _strings(text, out, min_length)
    return out


def _parts(row: dict, min_length: int) -> list[tuple[str, set[str], str]]:
    """What one Harbor task adds, by part: (part, its strings, the text it may repeat besides the world)."""
    oracle: set[str] = set()
    _strings(row["calls"], oracle, min_length)
    _strings(row["reply"], oracle, min_length)
    parts = [("instruction", _text_strings(row["facts"], min_length), ""),
             ("Verifier", _file_strings(row["verifier"], min_length), row["instruction"]),
             ("oracle", oracle, row["instruction"]),
             ("label", _text_strings([row.get("label") or ""], min_length), row["instruction"])]
    # A Task with no solution/ has no oracle part to scan (D399).
    return [part for part in parts if part[0] != ORACLE_PART or row.get("has_oracle", True)]


def leak_scan(dataset: Path, raw_dir: Path, scanned: list[dict], *, min_length: int = LEAK_MIN_LENGTH,
              content_min_length: int = CONTENT_MIN_LENGTH, corpus: Optional[set[str]] = None) -> dict:
    """The package's scan (D221) over what each Harbor task adds, counts only.

    Per task: the instruction's lines against the world and the harness, then the Verifier, the
    oracle's calls and its reply and a rater's words in the metadata (D399) against the world, the
    harness and the instruction. Each leaking part is named by its Task, masked, and its part. The policy is
    declared rather than scanned: it is the recorded agent's own system prompt, the text every Run
    the build made was given, and it ships in instruction.md as the policy the agent works under.
    """
    corpus = corpus_strings(raw_dir, min_length) if corpus is None else corpus
    index = _prose_index(corpus, content_min_length)
    harness = harness_text()
    leaking, echoes, checked, files = [], 0, 0, 0
    for row in scanned:
        world_values, world_text = _baseline(row["world"], WORLD_FILES + ("env", "overlays"), min_length)
        policy = row.get("policy") or ""
        parts = _parts(row, min_length)
        files += len(parts)
        for part, found, extra in parts:
            checked += len(found)
            found = found | embedded_prose(found, index, content_min_length)
            unaccounted = {value for value in found if value in corpus and not (
                value in world_values or value in world_text or value in harness or value in policy
                or value in extra)}
            leaks = {value for value in unaccounted if len(value) >= content_min_length}
            echoes += len(unaccounted - leaks)
            if leaks:
                leaking.append({"task": str(row["task"])[-MASK:], "part": part, "values": len(leaks)})
    return {"corpus_strings": len(corpus), "values_checked": checked, "files_scanned": files,
            "leaks": sum(row["values"] for row in leaking), "leaking_files": leaking,
            "value_echoes": echoes, "policy": "declared, not scanned"}


# --- the dataset --------------------------------------------------------------------


def _description(text: str) -> str:
    return next((line.strip("- ").strip() for line in text.splitlines()
                 if line.strip() and line.strip() not in (FACTS_LEAD, LATER_LEAD)), "")


def _policy(env: BuiltEnvironment, task: Any, fallback: Optional[str]) -> Optional[str]:
    try:
        return env.system_prompt(task) or fallback
    except Exception:  # noqa: BLE001 - a Task whose prompt cannot be read takes the Environment's
        return fallback


@dataclass
class _Build:
    """What every Task of one export shares."""
    workdir: Path
    out: Path
    name: str
    org: str
    env: BuiltEnvironment
    rows: dict
    classes: dict
    runs: dict
    policy: Optional[str]
    tag: Optional[str]
    sha: Optional[str]
    wheel: Path
    tasks: str = "trusted"
    tiers: Optional[dict] = None
    # None where no label file was given: then no Task carries a label field.
    labels: Optional[dict] = None
    # Where a Verifier with items dropped as unrenderable is written before it is laid (D422).
    scratch: Optional[Path] = None

    @property
    def tool_names(self) -> set[str]:
        return {sig.name for sig in self.env.sigs}


def _fallback_policy(env: BuiltEnvironment) -> Optional[str]:
    """The first Task's recorded policy, for a Task whose own cannot be read."""
    return next((prompt for prompt in (_policy(env, task, None) for task in env.tasks().values()) if prompt), None)


def _choose(workdir: Path, rows: dict, tasks: str, only: Optional[Iterable[str]]) -> tuple[list[str], dict]:
    """The recorded Task ids and the synthetic store Tasks to export, narrowed to `only` where given.
    `verified` takes every recorded Task; the ones with no Verifier are left out with that reason."""
    chosen = [task_id for task_id, row in rows.items() if tasks in ("all", "verified") or row["trusted"]]
    synthetic = _synthetic_tasks(workdir) if tasks == "all" else {}
    if only is not None:
        keep = set(only)
        chosen = [task_id for task_id in chosen if task_id in keep]
        synthetic = {task_id: body for task_id, body in synthetic.items() if task_id in keep}
    return chosen, synthetic


def _inputs(build: _Build, task_id: str, body: Optional[dict]) -> dict:
    """The record, facts, instruction, oracle candidates, policy and index row of one Task; `body` is a
    synthetic store Task's file, None for a recorded Task."""
    if body is not None:
        record = dict(body["task"])
        walk = [{"name": str(step.get("tool")), "arguments": dict(step.get("args") or {})}
                for step in body.get("walk") or () if str(step.get("tool")) in build.tool_names]
        return {"record": record, "facts": [], "instruction": opening([], record.get("intent")),
                "candidates": [walk], "reference": walk, "policy": build.policy, "index": {"bucket": body.get("bucket")}}
    record = read_json(build.workdir / "tasks" / f"{task_id}.json", {}) or {}
    facts, instruction = task_ask(build.workdir, task_id, record)
    known = task_id in build.env.tasks()
    return {"record": record, "facts": facts, "instruction": instruction,
            "candidates": candidate_calls(build.workdir, task_id, build.tool_names, build.runs.get(task_id, ())),
            "reference": reference_calls(build.workdir, task_id, build.runs.get(task_id, ())),
            "policy": _policy(build.env, build.env.task(task_id), build.policy) if known else build.policy,
            "index": build.rows[task_id]}


def _marks(build: _Build, task_id: str) -> dict:
    """The Task's Spec tier and reason off the tier report and, where label files were given, its label."""
    tier = (build.tiers or {}).get(task_id) or {}
    marks = {"spec_tier": str(tier.get("tier") or NOT_RECORDED), "spec_tier_reason": tier.get("reason")}
    if build.labels is not None:
        row = build.labels.get(task_id) or {"label": UNLABELLED}
        marks.update({name: row.get(key) for key, name in LABEL_FIELDS})
    return marks


def _metadata(build: _Build, task_id: str, synthetic: bool, index: dict, verifier: Verifier) -> dict:
    return {
        "kullback_task_id": task_id,
        "origin": "synthetic" if synthetic else "recorded",
        "tier": "trusted" if index.get("trusted") else "untrusted",
        "trust_reason": index.get("trust_reason"),
        "reference_class": None if synthetic else build.classes.get(task_id),
        "difficulty": index.get("bucket"),
        "needs_judge": needs_judge(verifier),
        "judge_items": any(atom.judge for atom in verifier.atoms),
        "confirm_gate": confirm_gate(verifier),
        **({} if synthetic else _marks(build, task_id)),
        "kullback_build": build.tag,
        "kullback_git_sha": build.sha,
    }


def _write(job: dict, metadata: dict, calls: Optional[list[dict]]) -> None:
    given, folder = job["given"], job["folder"]
    category = sorted({str(given["record"].get("category_id") or "")} - {""})
    keywords = category + sorted({call["name"] for call in calls or ()} & job["write_tools"])
    toml = task_toml(name=f"{job['org']}/{folder.name}", description=_description(given["instruction"]),
                     keywords=keywords, metadata=metadata, org=job["org"], judged=metadata["judge_items"])
    write_task(folder, workdir=job["workdir"], task_id=job["task_id"], task_record=given["record"],
               instruction=given["instruction"], policy=given["policy"], verifier_path=job["verifier_path"],
               calls=calls, wheel=job["wheel"], toml=toml, prune=job.get("prune"))


def _lay_with_oracle(job: dict) -> Optional[list[dict]]:
    """Write the task with each candidate oracle in turn and keep the first that holds; None if none does.
    Under `verified` and `all` a Task none holds for is then written without solution/, marked has_oracle
    false (D399); it answers [] for that Task. `job` is the whole input of one Task (`_job`), so a process
    pool can lay Tasks side by side (D401)."""
    metadata, folder = job["metadata"], job["folder"]
    for tried in job["given"]["candidates"]:
        _write(job, dict(metadata, has_oracle=True), tried)
        if oracle_holds(folder):
            metadata["has_oracle"] = True
            return tried
        shutil.rmtree(folder)
    if job["tasks"] == "trusted":
        return None
    metadata.update(has_oracle=False, no_oracle_reason=NO_ORACLE)
    _write(job, metadata, None)
    return []


def _verified_skip(build: _Build, task_id: str, body: Optional[dict]) -> Optional[str]:
    """Under `verified`, a recorded Task the tier report rules no_verifier is "no Verifier" whatever file is on
    disk (a draft the Spec set aside, D399), and one of the synthetic tier is "synthetic"; else None."""
    if build.tasks != "verified" or body is not None:
        return None
    row = (build.tiers or {}).get(task_id) or {}
    if row.get("reason") == NO_VERIFIER_REASON:
        return "no Verifier"
    return "synthetic" if row.get("tier") == SYNTHETIC_TIER else None


def _job(build: _Build, task_id: str, body: Optional[dict]) -> tuple[Optional[str], dict]:
    """(why the Task is left out or None, the whole input of laying it): what needs the build's shared
    Environment and Runs is read here, once, so laying the Task needs nothing else."""
    given = _inputs(build, task_id, body)
    verifier_path = verifier_of(build.workdir, task_id, body is not None)
    reason = ("no Verifier" if verifier_path is None else _verified_skip(build, task_id, body)
              or ("no Run" if not given["candidates"] else "no instruction" if not given["instruction"] else None))
    if reason:
        return reason, {}
    verifier = Verifier.model_validate(read_json(verifier_path, {}))
    dropped = [] if body is not None else unrenderable(verifier, _spec(build.workdir, task_id))
    verifier = without_items(verifier, dropped) if dropped else verifier
    effects, effects_from = repair.tool_effects(build.workdir)
    verifier, allowed = repair.allow_effects(verifier, effects, given["reference"])  # D432
    scratch = Path(build.scratch or build.out) / "verifiers"
    if dropped or allowed:
        verifier_path = write_json(scratch / f"{task_id}.json", as_dict(verifier))
    folder = build.out / (SYNTHETIC_DIR if body is not None else "") / f"{build.name}-{task_id}"
    return None, {"task_id": task_id, "given": given, "verifier_path": verifier_path, "folder": folder,
                  "metadata": dict(_metadata(build, task_id, body is not None, given["index"], verifier),
                                   items_unrenderable=dropped, effects_allowed=allowed, effects_from=effects_from),
                  "org": build.org, "out": build.out, "workdir": build.workdir, "wheel": build.wheel,
                  "tasks": build.tasks, "write_tools": set(build.env.write_tools()), "scratch": scratch}


def _start_of(job: dict) -> dict:
    """The Task's Starting state, as its image's db.json will hold it, off its world laid in a scratch folder."""
    given = job["given"]
    with tempfile.TemporaryDirectory() as scratch:
        app = Path(scratch)
        (app / WORLD_DIR).mkdir()
        _lay_world(job["workdir"], app / WORLD_DIR, job["task_id"], given["record"], given["instruction"],
                   given["policy"])
        return starting_state(app)


def listed_plan(start: dict) -> tuple[dict, dict[str, int], list[str]]:
    """What the Task's shipped world drops off its Starting state (D429): the plan by table and column, the
    count dropped per column, and the ambiguous columns, from which nothing is dropped."""
    found = listed_ids.columns(start)
    plan = listed_ids.drops(found)
    dropped = {column.name: len(column.missing) for column in found if column.keys is not None}
    return plan, dropped, [column.name for column in found if column.keys is None]


def _export_one(job: dict) -> tuple[Optional[str], dict, dict]:
    """(why the Task is left out or None, its manifest row, its leak scan row), off the Task's whole input. A Task
    whose Reference reads a listed id the shipped world drops is left out as READS_DROPPED_ID (D429), and a
    Verifier with a cell no Run can make hold as UNREADABLE_CELL, both before any oracle is tried."""
    start = _start_of(job)
    verifier = repair.resolve_job(job, Verifier.model_validate(read_json(job["verifier_path"], {})), start)  # D431
    plan, dropped, ambiguous = listed_plan(start)
    gone = {member for columns in plan.values() for members in columns.values() for member in members}
    if gone and listed_ids.reads(job["given"].get("reference") or (), gone):
        return READS_DROPPED_ID, {}, {}
    for table, columns in plan.items():
        listed_ids.prune_rows(start.get(table), columns)
    if unreadable_cells(verifier, start):
        return UNREADABLE_CELL, {}, {}
    calls = _lay_with_oracle(dict(job, prune=plan))
    if calls is None:
        return NO_ORACLE, {}, {}
    given, metadata, folder = job["given"], job["metadata"], job["folder"]
    path = folder.relative_to(job["out"]).as_posix()
    listed = {"harbor_name": f"{job['org']}/{folder.name}", "path": path, **metadata,
              "ids_dropped": dropped, "ambiguous_columns": ambiguous}
    solve = folder / "solution" / "solve.sh"
    scanned = {"name": path, "task": job["task_id"], "world": folder / "environment" / WORLD_DIR,
               "facts": given["facts"] or [given["instruction"]], "label": str(metadata.get("label_gold_diff") or ""),
               "instruction": given["instruction"], "policy": given["policy"], "verifier": folder / "tests" / VERIFIER_NAME,
               "has_oracle": metadata["has_oracle"], "calls": calls,
               "reply": solve.read_text(encoding="utf-8") if solve.is_file() else ""}
    return None, listed, scanned


def leak_lines(scan: dict) -> str:
    """Which Task (masked) and which part leaked and how many strings, never the strings."""
    return "; ".join(f"Task {row['task']} {row['part']} {row['values']}" for row in scan.get("leaking_files") or ())


def by_tier_and_label(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Exported recorded Tasks counted by tier cell, then by label where the rows carry one."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = tier_cell(str(row.get("spec_tier") or NOT_RECORDED), row.get("spec_tier_reason"))
        label = str(row.get("label") or "")
        out.setdefault(cell, {})
        out[cell][label] = out[cell].get(label, 0) + 1
    return out


def solution_only(scan: dict, scanned: list[dict]) -> list[dict]:
    """The scanned Tasks whose every leaking part is the oracle, the solution/ directory. A masked id that
    names more than one scanned Task is never taken, so such a leak still refuses the export."""
    parts: dict[str, set[str]] = {}
    for row in scan.get("leaking_files") or ():
        parts.setdefault(row["task"], set()).add(row["part"])
    owners: dict[str, list[dict]] = {}
    for row in scanned:
        owners.setdefault(str(row["task"])[-MASK:], []).append(row)
    return [owners[masked][0] for masked, found in parts.items()
            if found == {ORACLE_PART} and len(owners.get(masked) or ()) == 1]


def _drop_solution_leaks(out: Path, scan: dict, scanned: list[dict], listed: list[dict],
                         skipped: dict) -> tuple[dict, list[str]]:
    """Take each Task whose only leaking part is its oracle out of the dataset (D399); answers the scan
    over what stays, which still refuses the export on any other part, and the ids taken out."""
    dropped = solution_only(scan, scanned)
    names = {row["name"] for row in dropped}
    for name in names:
        shutil.rmtree(out / name, ignore_errors=True)
    listed[:] = [row for row in listed if row["path"] not in names]
    if dropped:
        skipped[SOLUTION_LEAKS] = len(dropped)
    masked = {str(row["task"])[-MASK:] for row in dropped}
    kept = [row for row in scan.get("leaking_files") or () if row["task"] not in masked]
    taken = [row for row in scan.get("leaking_files") or () if row["task"] in masked]
    scan = dict(scan, leaks=sum(row["values"] for row in kept), leaking_files=kept, solution_leaks=taken)
    return scan, [str(row["task"]) for row in dropped]


def _counts(listed: list[dict], recorded: list[dict], skipped: dict) -> dict:
    def count(test: Any) -> int:
        return sum(1 for row in listed if test(row))

    return {"exported": len(listed), "recorded": len(recorded),
            "synthetic": count(lambda row: row["origin"] == "synthetic"),
            "needs_judge": count(lambda row: row["needs_judge"]),
            "confirm_gate": count(lambda row: row["confirm_gate"]),
            "skipped": skipped,
            "no_oracle": count(lambda row: row.get("has_oracle") is False),
            # Items shipped nowhere because they rest only on facts the instruction does not render (D422).
            "items_unrenderable": sum(len(row.get("items_unrenderable") or ()) for row in listed),
            "tasks_with_items_unrenderable": count(lambda row: bool(row.get("items_unrenderable"))),
            # Listed ids the shipped world drops, the columns they leave, and the columns read as ambiguous (D429).
            "ids_dropped": sum(sum((row.get("ids_dropped") or {}).values()) for row in listed),
            "tasks_with_ids_dropped": count(lambda row: bool(row.get("ids_dropped"))),
            "columns_pruned": sorted({name for row in listed for name in row.get("ids_dropped") or {}}),
            "tasks_with_ambiguous_column": count(lambda row: bool(row.get("ambiguous_columns"))),
            "ambiguous_columns": sorted({name for row in listed for name in row.get("ambiguous_columns") or ()}),
            **repair.counts(listed),
            "by_tier_and_label": by_tier_and_label(recorded)}


def _manifest(build: _Build, tasks: str, listed: list[dict], skipped: dict, left: list[tuple[str, str]],
              solution_leaks: Iterable[str] = ()) -> dict:
    recorded = [row for row in listed if row["origin"] == "recorded"]
    return {
        "format": HARBOR_FORMAT,
        "harbor_schema_version": SCHEMA_VERSION,
        "name": build.name,
        "org": build.org,
        "tasks_chosen": tasks,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": build.tag,
        "kullback_version": kullback_version(),
        "git_sha": build.sha,
        "reward": dict(REWARD_SHAPE, harbor="reward.txt and reward.json are 1 when every gate holds, else 0 (D433); "
                                            "items.json keeps the gated then weighted shaped_reward; an unanswered judge "
                                            f"item leaves the trial unscored (reward.json says why, exit {UNSCORED_EXIT})"),
        "counts": _counts(listed, recorded, skipped),
        # Exported with no solution/: no Run passes on replay (D399). Masked, as left_out.
        "no_oracle": [{"task": row["kullback_task_id"][-MASK:], "reason": NO_ORACLE}
                      for row in listed if row.get("has_oracle") is False],
        "items_unrenderable": [{"task": row["kullback_task_id"][-MASK:], "items": row["items_unrenderable"]}
                               for row in listed if row.get("items_unrenderable")],
        "tasks": recorded,
        "synthetic": [row for row in listed if row["origin"] == "synthetic"],
        # Masked to the last four characters, as every report masks a Task id.
        "ids_dropped": [{"task": row["kullback_task_id"][-MASK:], "columns": row["ids_dropped"]}
                        for row in listed if row.get("ids_dropped")],
        "ambiguous_columns": [{"task": row["kullback_task_id"][-MASK:], "columns": row["ambiguous_columns"]}
                              for row in listed if row.get("ambiguous_columns")],
        "left_out": [{"task": task_id[-MASK:], "reason": reason} for task_id, reason in left]
        + [{"task": task_id[-MASK:], "reason": SOLUTION_LEAKS} for task_id in solution_leaks],
    }


def _read_marks(workdir: Path, task_ids: list[str], tiers: Optional[Iterable[Any]],
                labels: Optional[Iterable[Any]], rubric: Optional[str]) -> tuple[dict, Optional[dict], dict]:
    """The tier rows (the workdir's tiers.json unless files are named), the label rows (None without
    label files), and what the manifest says of both. Later files win a clash."""
    tier_files = [Path(path) for path in tiers] if tiers else [workdir / TIERS_FILE]
    tier_rows, tiers_unmatched = keyed_rows(
        [pair for path in tier_files if path.is_file() for pair in tier_pairs(path)], task_ids)
    said = {"tiers": {"files": [path.name for path in tier_files if path.is_file()], "rows": len(tier_rows),
                      "unmatched": tiers_unmatched}}
    if not labels:
        return tier_rows, None, said
    label_files = [Path(path) for path in labels]
    label_rows, labels_unmatched = keyed_rows([pair for path in label_files for pair in label_pairs(path)], task_ids)
    said["labels"] = {"files": [path.name for path in label_files], "rows": len(label_rows),
                      "unmatched": labels_unmatched, "rubric": rubric or "not named"}
    return tier_rows, label_rows, said


def _lay_all(build: _Build, work: list[tuple[str, Optional[dict]]],
             workers: Optional[int]) -> tuple[list[dict], list[dict], dict[str, int], list[tuple[str, str]]]:
    """Lay every Task of `work` on `workers` processes: (manifest rows, scan rows, skipped by reason, the
    Tasks no Run holds for or set aside as reading a dropped id, each with its reason), in `work`'s order
    whatever the count (D401)."""
    jobs = [_job(build, task_id, body) for task_id, body in work]
    laid = iter(processes.each([job for reason, job in jobs if not reason], _export_one, workers))
    listed: list[dict] = []
    scanned: list[dict] = []
    skipped: dict[str, int] = {}
    left: list[tuple[str, str]] = []
    for (task_id, _), (reason, _) in zip(work, jobs, strict=True):
        reason, row, scan_row = (reason, {}, {}) if reason else next(laid)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            left += [(task_id, reason)] if reason in (NO_ORACLE, READS_DROPPED_ID) else []
            continue
        listed.append(row)
        scanned.append(scan_row)
    return listed, scanned, skipped, left


def export_harbor(workdir: Any, out: Any, *, tasks: str = "trusted", org: str = ORGANISATION,
                  name: Optional[str] = None, scan: bool = True, wheel: Optional[Path] = None,
                  only: Optional[Iterable[str]] = None, tiers: Optional[Iterable[Any]] = None,
                  labels: Optional[Iterable[Any]] = None, labels_rubric: Optional[str] = None,
                  workers: Optional[int] = None, index: Optional[list[dict]] = None) -> dict:
    """Write a Harbor dataset of the workdir's Tasks to `out` and answer its manifest.

    `tasks` is `trusted` (the default), `verified` (every recorded Task with a Verifier on disk, none
    of the synthetic tier) or `all`; synthetic store Tasks are never trusted, so they
    come only with `all`, under `synthetic/`. `only` narrows the list to the named Task ids. Every
    recorded Task carries its Spec tier off `tiers` (the workdir's tiers.json by default) and, with
    `labels`, a rater's label, `unlabelled` where no row names it (D399). A Task whose only leaking part
    is its oracle is left out as SOLUTION_LEAKS and listed masked; a leak in any other part refuses. A Task
    with no Verifier, no Run or no instruction, or a Verifier cell the scorer cannot read (UNREADABLE_CELL,
    D423), is left out and counted by reason. The shipped world drops the listed ids that name no row where the
    table a column keys is read without doubt, counted per Task; a Task whose Reference reads one is left out as
    READS_DROPPED_ID (D429). The oracle is the
    first of the Task's Runs whose calls, replayed through the exported world, hold every item a judge
    is not needed for; a Task with none is left out too (NO_ORACLE). The Tasks are laid on `workers`
    processes (None: the default of kullback/processes.py) and the dataset is the same whatever the count (D401);
    `index` is the package's Task index where the caller already holds it, so it is not ruled twice.
    """
    workdir, out = Path(workdir), Path(out)
    if tasks not in TASKS_CHOICES:
        raise ExportError(f"--tasks takes {' or '.join(TASKS_CHOICES)}, not {tasks}")
    if not (workdir / "env").is_dir():
        raise ExportError(f"{workdir} holds no env/ package: there is no Environment to export")
    env = BuiltEnvironment(workdir)
    if index is None:
        index = task_index(workdir, frozen_task_ids(workdir), workers=workers)
    rows = {row["task_id"]: row for row in index}
    chosen, synthetic = _choose(workdir, rows, tasks, only)
    tier_rows, label_rows, said = _read_marks(workdir, list(rows), tiers, labels, labels_rubric)
    policy = _fallback_policy(env)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        build = _Build(workdir=workdir, out=out, name=name or workdir.resolve().name, org=org, env=env, rows=rows,
                       classes=reference_classes(workdir) or {}, runs=runs_on_disk(workdir), policy=policy,
                       tag=build_tag(workdir), sha=git_sha(), wheel=wheel or build_wheel(Path(scratch)),
                       tasks=tasks, tiers=tier_rows, labels=label_rows, scratch=Path(scratch))
        work = [(task_id, None) for task_id in chosen] + list(synthetic.items())
        listed, scanned, skipped, left = _lay_all(build, work, workers)
    found = leak_scan(out, workdir / "raw", scanned) if scan else {"leaks": 0, "skipped": True}
    found, solution_leaks = _drop_solution_leaks(out, found, scanned, listed, skipped)
    manifest = dict(_manifest(build, tasks, listed, skipped, left, solution_leaks), **said)
    manifest["leak_scan"] = found
    if manifest["leak_scan"].get("leaks"):
        shutil.rmtree(out, ignore_errors=True)
        raise ExportError(f"the leak scan found {manifest['leak_scan']['leaks']} recorded strings in "
                          f"{len(manifest['leak_scan']['leaking_files'])} Harbor task part(s) "
                          f"({leak_lines(manifest['leak_scan'])}); the dataset was taken away again")
    manifest["files"] = file_hashes(out)
    manifest["content_hash"] = content_hash(manifest["files"])
    write_json(out / MANIFEST_NAME, manifest)
    return manifest
