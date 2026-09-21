"""The one place a Simulated user is built, whoever asks for it (G5).

Three callers need the same driver: the round scorer (`kullback/rounds.py`), the CLI score
(`kullback/cli.py`) and real Runs (`kullback/builder/build.py`). Only the last passed the full
inputs (`goal_writes`, `answer_strip`, `trace`), so the scorer examined a different user from the
one real Runs meet. Every caller now builds through `build_user`, which fills whatever the caller
leaves out from the workdir on disk, so all three meet the same inputs the build passes today.

`build_user` keeps no static edge back to the package that drives the Runs: the strip closure is
loaded lazily through `importlib` (the CLI already loads modules that way), so the user package
still imports only what D214 allows. Everything else comes from the workdir files the build
writes (`replays.json`, `traces`, `user_rules`, `vocabulary.json`, `tool_sigs.json`,
`user_lessons.json`, `schema.json`, `canon-rules.json`).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

from kullback.runner import canon
from kullback.runner.records import EntitySchema, read_json
from kullback.user import context as context_mod
from kullback.user import fidelity as fidelity_mod
from kullback.user import lesson as lesson_mod
from kullback.user import rules as rules_mod
from kullback.user.agent import AgentUser

# What the built user is for. The two purposes build the same user; they differ only in what a
# Task with no reference recording on disk means. An examination has nothing to examine, so it
# gets None (the scorer skips None). A Run was promised its inputs by the build, so it is an
# error there.
PURPOSE_RUN = "run"
PURPOSE_SCORE = "score"
PURPOSES = (PURPOSE_RUN, PURPOSE_SCORE)

SCHEMA_FILE = "schema.json"
CANON_FILE = "canon-rules.json"

# A caller that names no value leaves the workdir to answer. Explicit None still means None: a
# goal that implies no write (`goal_writes=None`) is not a goal with an empty write set.
_MISSING = object()


def build_user(workdir: Any, task_id: str, model: Any, purpose: str, *,
               ctx: Any = _MISSING, fallback: Any = _MISSING,
               record_values: Any = _MISSING, vocab: Any = _MISSING,
               write_tools: Any = _MISSING, goal_writes: Any = _MISSING,
               answer_strip: Any = _MISSING, trace: Any = _MISSING) -> Optional[AgentUser]:
    """The Simulated user of one Task: the agent user over the rules floor, fully specified.

    The caller passes what it has live and the workdir supplies the rest, so the round scorer and
    the CLI meet the same inputs real Runs do (`goal_writes`, `answer_strip`, `trace` included).
    A caller that passes everything (a real Run) never touches the disk.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose is one of {list(PURPOSES)}, not {purpose!r}")
    given = (("ctx", ctx), ("fallback", fallback), ("record_values", record_values),
             ("vocab", vocab), ("write_tools", write_tools), ("goal_writes", goal_writes),
             ("answer_strip", answer_strip), ("trace", trace))
    if all(value is not _MISSING for _, value in given):
        disk = {}
    else:
        disk = _disk_state(workdir, task_id, purpose)
        if disk is None:
            return None
    for key, value in given:
        if value is not _MISSING:
            disk[key] = value
    return AgentUser(
        disk["ctx"], disk["fallback"], model, vocab=disk["vocab"],
        write_tools=disk["write_tools"], goal_writes=disk["goal_writes"],
        answer_strip=disk["answer_strip"], record_values=disk["record_values"],
        trace=disk["trace"],
    )


def _disk_state(workdir: Any, task_id: str, purpose: str) -> Optional[dict]:
    """Everything the workdir knows about one Task's user, or None where the score has nothing.

    No live world exists here, so the fallback takes no starting_state_reader.
    """
    index = fidelity_mod.trace_index(workdir)
    trace_id = fidelity_mod.references(workdir).get(task_id)
    trace = index.get(trace_id) if trace_id else None
    user_rules = fidelity_mod.rules_for(workdir, trace_id) if trace_id else None
    if trace is None or user_rules is None:
        if purpose == PURPOSE_SCORE:
            return None
        raise ValueError(f"no reference recording for task {task_id!r} under {workdir}")
    writes = fidelity_mod.write_tools_of(workdir)
    vocab = fidelity_mod.vocabulary_of(workdir)
    record = context_mod.mine_record_values(trace)
    lessons = lesson_mod.lines_for(lesson_mod.load_lessons(workdir), task_id)
    ctx = context_mod.curate(task_id, user_rules, trace, vocab=vocab, write_tools=writes,
                             record_fields=sorted(record), lessons=lessons)
    goal_writes = rules_mod.goal_write_set(trace, writes)
    answer_strip = _strip_for(workdir, task_id, index)
    return {
        "ctx": ctx,
        "fallback": rules_mod.SimulatedUser(user_rules, vocab=vocab, write_tools=writes,
                                            goal_writes=goal_writes, answer_strip=answer_strip),
        "record_values": record,
        "vocab": vocab,
        "write_tools": writes,
        "goal_writes": goal_writes,
        "answer_strip": answer_strip,
        "trace": trace,
    }


def _strip_for(workdir: Any, task_id: str, index: dict) -> Any:
    """The D196 strip over this Task's own evidence, or None where the build had none either.

    A caller that names no recordings gets no strip (the build does the same), and a workdir with
    no evidence files reads as no strip rather than failing the score. The closure is the build's
    own, loaded lazily so this module keeps no static import the D214 test forbids.
    """
    members = _members_of(workdir, task_id, index)
    if not members:
        return None
    try:
        intent = importlib.import_module("kullback.builder.intent")
    except ImportError:
        return None
    return intent.value_strip(members, schema=_schema_of(workdir), rules=_canon_of(workdir))


def _members_of(workdir: Any, task_id: str, index: dict) -> list:
    """This Task's own recordings, in a fixed order, off the replay rows on disk.

    The build reads them off the Task's run ids; on disk the replay rows are the map from a Task
    to its recordings. Only rows whose trace file is still on disk count.
    """
    body = read_json(Path(workdir) / fidelity_mod.REPLAYS, {}) or {}
    rows = body.get(task_id) or {}
    ids: set[str] = set()
    if isinstance(rows, dict):
        for key, row in rows.items():
            ids.add(str(key))
            if isinstance(row, dict) and row.get("trace_id"):
                ids.add(str(row["trace_id"]))
    return [index[tid] for tid in sorted(ids) if tid in index]


def _schema_of(workdir: Any) -> Optional[EntitySchema]:
    """The mined schema off disk, or None where the build has not written one yet."""
    path = Path(workdir) / SCHEMA_FILE
    if not path.is_file():
        return None
    try:
        return EntitySchema.model_validate(read_json(path, {}))
    except ValueError:
        return None


def _canon_of(workdir: Any) -> Any:
    """The canonicalizer rules off disk, or None where the build has not written them yet."""
    path = Path(workdir) / CANON_FILE
    if not path.is_file():
        return None
    try:
        return canon.load_rules(path)
    except (OSError, ValueError):
        return None

