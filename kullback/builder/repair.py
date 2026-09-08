"""Phase 6 repair verbs: the ways a round fixes what the gates refused (D130, D135, D136, D138).

Verbs: `repair_recompile(name, hint)`, `repair_grow(table, count)`, `repair_intent(task_id, hint)`,
`repair_rewrite_skill(name, content)`, `repair_refuse_task(task_id, reason)`,
`repair_escalate(task_id, queue)`. The `repair_`
prefix keeps them clear of the Builder tools (`grow` already exists there, and the registry
rejects duplicate names). Each verb records its request as JSONL under `workdir/repairs/` and
returns a short `RepairResult`, which is the whole of what the two deciding verbs (refuse a Task,
escalate it) do, and their descriptions say so: a decision moves no gate. The acting verbs are
registered from `builder/tools.py` instead, where the build plan is: they record the same request
and then run the stage that repairs the artifact, so the gates rule on the new body, the grown table
or the rewritten Intent in the same tool result. `repair_rewrite_skill`
is written here and registered nowhere: a model that rewrites its own prompt unchecked is the GEPA
caution in `docs/todo.md`, and the verb waits for the gate that accepts an edit (D125, D132).
Nothing here calls a model or edits `kullback/gates` or `kullback/runner` (D122): the ratchet and
the lesson are code over artifacts the pipeline already wrote.

This module also owns what one target of one verb is made of, which is the answer to two questions
a gate cannot answer. `target_hash` and `change_of` hash that one artifact either side of the call,
so a request records whether it moved anything of its own: a round-wide fingerprint says `intents`
changed and cannot say which of seven repaired Tasks it was. `target_ruling` reads the same artifact
back as the line the result opens with (`repair_intent task_x: grounded: "..."`), because a gate
rules over every target at once and its first failure names whichever target sorts first, not the
one that was just repaired.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback import gates
from kullback.agent.tools import AgentTool, ToolResult
from kullback.builder import memory as memory_mod
from kullback.runner.records import content_hash

Sink = Callable[[Any], Awaitable[None]]


# --- result ---------------------------------------------------------------


class RepairResult(BaseModel):
    """What one repair verb recorded: the verb, its target, and where the request went."""

    model_config = ConfigDict(extra="forbid")

    verb: str
    target: str
    status: str = "recorded"
    detail: str = ""
    path: str = ""


# --- args ------------------------------------------------------------------


class RecompileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The mined tool whose body to compile again.")
    hint: str = Field(default="", description="What went wrong with the last body and what to try "
                                              "instead; kept as a lesson for this tool (D87).")


class GrowRepairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="The table of the Starting state to grow.")
    count: int = Field(ge=1, description="How many rows the table should hold after growing (D107).")


class IntentRepairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task whose Intent to write again.")
    hint: str = Field(default="", description="What the new Intent should say or avoid, in one line; "
                                              "it goes into the prompt for this Task and nowhere else.")


class RewriteSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The skill to rewrite (workdir skills/<name>/SKILL.md).")
    content: str = Field(description="The new SKILL.md content.")


class RefuseTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task to refuse (no Verifier, no training signal).")
    reason: str = Field(description="Why the Task is refused (e.g. no frontier Run finishes it).")


class EscalateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The Task to escalate to a person.")
    queue: str = Field(default="review", description="The queue the Task is escalated to.")


class RecordFindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(description="The tool whose recorded calls show the behaviour.")
    finding: str = Field(description="The behaviour, as the recorded calls show it and the tool's description "
                                     "does not say: the leaf where they part and what the recording does there.")
    evidence: list[str] = Field(default_factory=list,
                                description="The recorded call ids that agree with each other on it.")


# --- the one target a repair verb acts on -------------------------------------

# The artifact each acting verb rewrites, under the name a round's counts give it (`rounds.py`).
# The deciding verbs are absent on purpose: refusing a Task and escalating one write a row for the
# report and change no artifact, which is what their own descriptions say.
REPAIR_ARTIFACT: dict[str, str] = {
    "repair_recompile": "bodies",
    "repair_grow": "starting_state",
    "repair_intent": "intents",
}
NO_ATTEMPT = "the compiler recorded no attempt"
NO_FAILURE = "the gates recorded no failure"
NO_INTENT = "no Intent is recorded for this Task"

# The stage's ruling on the body each tool already had, one row per tool: the outcome (kept, beaten
# or could not run), both scores, how many recompiles in a row have now scored no higher, and how
# many of the evidence calls the Reference replay put back (D184, D191). It is written by the
# compile_tools stage in `builder/build.py` and read back here, because the sentence a repair opens
# with has to say the same thing the stage decided: "cleared the gates" is about the body that was
# released, and the released body is often the one that was already there.
KEPT_BODIES_FILE = "kept_bodies.json"
# How many recompiles in a row may score no higher before the tool is called stalled. The third
# strike shape of D181 rule 2, for the Builder's own verb: two rounds that bought nothing on one
# tool are the evidence that a third will buy nothing either.
STALLED_AFTER = 2


def _json_at(workdir: Any, relative: Any, default: Any) -> Any:
    """One record file of a workdir, or the default when it is missing or half written."""
    try:
        return json.loads((Path(workdir) / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def intent_record(workdir: Any, task_id: str) -> dict:
    """The Intent the intent stage last wrote for one Task, or {} where it wrote none."""
    body = _json_at(workdir, Path("intents") / f"{task_id}.json", {})
    return body if isinstance(body, dict) else {}


def tool_body(workdir: Any, name: str) -> Optional[str]:
    """One tool's compiled body out of `bodies.json`, or None where the build wrote none."""
    bodies = _json_at(workdir, "bodies.json", {})
    bodies = bodies.get("bodies", bodies) if isinstance(bodies, dict) else {}
    body = bodies.get(name) if isinstance(bodies, dict) else None
    return body if isinstance(body, str) else None


def tool_build(workdir: Any, name: str) -> dict:
    """What `tool_builds.json` recorded for one tool: whether it ended assisted, and its attempts."""
    builds = _json_at(workdir, "tool_builds.json", {})
    row = builds.get(name) if isinstance(builds, dict) else None
    return row if isinstance(row, dict) else {}


def table_rows(workdir: Any, table: str) -> Optional[dict]:
    """One table of the Starting state out of `db.json`, or None where the world holds no such table."""
    db = _json_at(workdir, "db.json", {})
    rows = db.get(table) if isinstance(db, dict) else None
    return rows if isinstance(rows, dict) else None


def target_artifact(workdir: Any, verb: str, target: str) -> Any:
    """What one call of a repair verb is about: this Task's Intent, this tool's body, this table's rows.

    A deciding verb owns no artifact, so it is None here and `changed` can only ever be False for it.
    """
    if verb == "repair_intent":
        return intent_record(workdir, target)
    if verb == "repair_recompile":
        return tool_body(workdir, target)
    if verb == "repair_grow":
        return table_rows(workdir, target)
    return None


def target_hash(workdir: Any, verb: str, target: str) -> Optional[str]:
    """A short hex over the one artifact this call is about; None for a verb that owns none."""
    if verb not in REPAIR_ARTIFACT:
        return None
    return content_hash(target_artifact(workdir, verb, target))[:12]


def change_of(workdir: Any, verb: str, target: str, before: Optional[str]) -> dict:
    """What one repair did to its own target: the hash either side of the call and whether it moved.

    `before` is `target_hash` taken before the stage ran. A round-wide fingerprint cannot answer
    this: a round that rewrote six Intents and left a seventh refused changed `intents` once, and
    every one of the seven requests would read as having changed it.
    """
    after = target_hash(workdir, verb, target)
    return {"changed": before != after, "hash_before": before, "hash_after": after}


# --- the target's own ruling, off the artifact the stage just wrote -----------


SHAPES_SHOWN = 3
_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+\b")


def failure_shape(text: str) -> str:
    """One failure sentence with its values generalized, so two calls that failed the same way group."""
    return _LITERAL.sub("*", " ".join(str(text).split()))


def failure_shapes(failures: Iterable[str]) -> list[tuple[str, int]]:
    """The distinct kinds of failure among a gate's sentences: the first of each kind, and how many share it.

    A gate that ruled over sixty recorded calls leaves sixty sentences, and one of them was all the
    mechanic ever saw, so the hint it wrote answered one example and the next recompile met the
    fifty-nine it had not been told about. Grouped by shape, the same sixty sentences say how many
    kinds of failure there are, which is what a hint has to answer.
    """
    seen: dict[str, list[Any]] = {}
    for failure in failures:
        row = seen.setdefault(failure_shape(failure), [str(failure), 0])
        row[1] += 1
    return [(text, count) for text, count in seen.values()]


def _failure_detail(node: Any) -> str:
    """What went wrong in one compile attempt: the first failure, or every shape when calls failed in more than one way."""
    if not isinstance(node, dict):
        return ""
    for failure in node.get("failures") or []:
        return str(failure)
    for ruling in node.get("gates") or []:
        if not (isinstance(ruling, dict) and not ruling.get("pass")):
            continue
        stage = ruling.get("stage")
        failures = [str(f) for f in (ruling.get("failures") or [])]
        if not failures:
            return f"{stage} failed"
        if len(failures) == 1:
            return f"{stage}: {failures[0]}"
        shapes = failure_shapes(failures)
        shown = "; ".join(f"{text} ({count} call{'' if count == 1 else 's'})"
                          for text, count in shapes[:SHAPES_SHOWN])
        more = len(shapes) - SHAPES_SHOWN
        plural = "" if len(shapes) == 1 else "s"
        return (f"{stage}: {len(failures)} calls failed in {len(shapes)} shape{plural}: {shown}"
                + (f"; {more} more shapes" if more > 0 else ""))
    return ""


def intent_ruling(workdir: Any, task_id: str) -> str:
    """Whether this Task's Intent grounded, off the file the intent stage just wrote."""
    record = intent_record(workdir, task_id)
    if record.get("grounded"):
        return f'repair_intent {task_id}: grounded: "{record.get("text") or ""}"'
    return f"repair_intent {task_id}: still refused: {record.get('reason') or NO_INTENT}"


def kept_body_ruling(workdir: Any, name: str) -> dict:
    """What the compile_tools stage decided about the body this tool already had (D184, D191)."""
    rows = _json_at(workdir, KEPT_BODIES_FILE, {})
    row = rows.get(name) if isinstance(rows, dict) else None
    return row if isinstance(row, dict) else {}


def stalled_note(row: dict) -> str:
    """How many recompiles in a row have bought this tool nothing, once that is worth saying (D191).

    Said at the second one and every one after, because the point of saying it is that the next
    round should be spent elsewhere: a Builder that reads "still assisted" and nothing else asks the
    same question again, which is what four live rounds of one build did on the same four tools.
    """
    unbeaten = int(row.get("unbeaten") or 0) if isinstance(row, dict) else 0
    if unbeaten < STALLED_AFTER:
        return ""
    return (f"; stalled: {unbeaten} recompiles in a row scored no higher than the body it has, so "
            f"the next one will not either unless something other than the hint changes")


def score_note(workdir: Any, name: str) -> str:
    """The score pair this recompile was judged on, and which body the stage released (D191).

    The stage scores every attempt against the body the tool already has, on the key the compiler
    ranks its own attempts by (gates passed, then recorded calls matched), and releases whichever
    won. Until this said so, the ruling named only the released body's gates, so a recompile whose
    attempt lost by ninety calls and a recompile that won read the same to the model.
    """
    row = kept_body_ruling(workdir, name)
    outcome = str(row.get("outcome") or "")
    if not outcome:
        return ""
    attempt, kept = row.get("attempt_score"), row.get("kept_score")
    pair = f"the attempt scored {attempt} against the kept body's {kept} (gates passed, calls matched)"
    if outcome == "beaten":
        released = f"{pair} and was released"
    elif outcome == "could_not_run":
        released = (f"{pair}, and the body already there answered no call at all under this world, "
                    f"so the attempt was released")
    else:
        released = f"{pair}, so the body already there stands and this recompile changed nothing"
    replayed = int(row.get("from_replay") or 0)
    evidence = int(row.get("evidence_calls") or 0)
    from_replay = (f"; {replayed} of the {evidence} evidence calls are from_replay, put back because "
                   f"the Reference replay failed on them") if replayed else ""
    return f" ({released}{from_replay}{stalled_note(row)})"


def recompile_ruling(workdir: Any, name: str) -> str:
    """Whether this tool's new body cleared the gates, off `tool_builds.json` and `kept_bodies.json`.

    A tool that ended assisted is a body no attempt got through (D49), so the line carries what went
    wrong in its last attempt: that is what the next hint has to answer. Where the recorded calls
    failed in more than one way, every shape is named with how many calls fell into it, since a hint
    written against the first sentence alone repairs one call and leaves the rest as they were.

    The assisted reading is the released body's, and the released body is the one the stage kept
    whenever the attempt did not beat it, so the line goes on to say what the attempt scored against
    it and whose body came out (D191). "cleared the gates" on its own was read as a repair that
    worked, over rounds in which the repair had changed nothing at all.
    """
    build = tool_build(workdir, name)
    if not build:
        return f"repair_recompile {name}: {NO_ATTEMPT}"
    score = score_note(workdir, name)
    if not build.get("assisted"):
        return f"repair_recompile {name}: cleared the gates{score}"
    failures = [text for text in (_failure_detail(node) for node in build.get("nodes") or []) if text]
    # A body that never read its arguments is a different repair from a body with a defect in it, so
    # the line says which one this is before it says what the gates saw.
    stood = "still assisted and hardcoded" if build.get("hardcoded") else "still assisted"
    return f"repair_recompile {name}: {stood}{score}: {failures[-1] if failures else NO_FAILURE}"


def grow_ruling(workdir: Any, table: str, count: int) -> str:
    """How many rows the grown table holds now, against the count the call asked for (D107)."""
    rows = table_rows(workdir, table)
    if rows is None:
        return f"repair_grow {table}: the Starting state holds no table of that name"
    if len(rows) >= count:
        return f"repair_grow {table}: {len(rows)} rows, the {count} asked for"
    return f"repair_grow {table}: {len(rows)} rows, short of the {count} asked for"


def target_ruling(workdir: Any, verb: str, args: Any) -> str:
    """The line a repair result opens with: how the one target it was called on came out.

    Read off the artifact the stage just wrote, never off a gate's failure list, because a gate
    rules over every target at once and its first failure is about whichever target sorts first.
    """
    if verb == "repair_intent":
        return intent_ruling(workdir, args.task_id)
    if verb == "repair_recompile":
        return recompile_ruling(workdir, args.name)
    if verb == "repair_grow":
        return grow_ruling(workdir, args.table, args.count)
    return ""


# --- request log ------------------------------------------------------------


def _repairs_dir(workdir: Any) -> Path:
    path = Path(workdir) / "repairs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_request(workdir: Any, verb: str, target: str, body: dict, round_no: int = 1) -> Path:
    """One repair request appended to `repairs/<verb>.jsonl`; the file the round's report reads.

    `round` is the round the request was made in (D126). The timestamp says when, and rounds.json
    carries no clock of its own, so the round number is what places a repair against the rulings
    that round left in `gates_by_round.json`: that is how a report says whether a repair turned a
    red gate green. A build with no round driver is one pass, which is round 1.

    An acting verb puts `change_of` in `body`, so the row also carries `changed` with the hash of
    the one target's artifact either side of the call (`hash_before`, `hash_after`). That is what
    says whether this repair moved anything; a round-wide fingerprint cannot, because six Intents
    rewritten in one round and one left alone share it.
    """
    path = _repairs_dir(workdir) / f"{verb}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"verb": verb, "target": target, "at": time.time(),
                             "round": int(round_no), **body}, sort_keys=True) + "\n")
    return path


def _render(result: RepairResult) -> str:
    return f"{result.verb} {result.target}: {result.status}" + (f" ({result.detail})" if result.detail else "")


# --- the same decision, taken again --------------------------------------------

# How many times one Task may be refused for one reason before the verb stops taking it. D181's
# three strikes stopped the Examiner repairing one Verifier against one check for ever; the Builder's
# `repair_refuse_task` had no such stop, and one live build made 17 of them, had 0 admitted, and in
# round 3 refused four Tasks again with the reason word for word from round 1 and the same answer.
# Refusing is not a repair: nothing about the Task moves between two identical requests, so the
# second is the last one that can tell anybody anything.
REFUSE_STOP = 2
# What buys something a third refusal cannot. Whether a Task is refused is the refuse gate's, over
# the Examiner's own `refuse` and only when no frontier Run finished (D128); on the Builder's side
# what is left is repairing what the Task is actually blocked on, or escalating it to a person.
REFUSE_ALTERNATIVES = "repair_recompile or repair_intent on what blocks it, or repair_escalate"


def refusals_recorded(workdir: Any, verb: str = "repair_refuse_task") -> list[dict]:
    """Every request this workdir has recorded for `verb`, oldest first; a half-written line is skipped."""
    path = Path(workdir) / "repairs" / f"{verb}.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _reason_of(row: dict) -> str:
    return str((row.get("arguments") or {}).get("reason") or "").strip()


def refuse_lock(workdir: Any, task_id: str, reason: str, limit: int = REFUSE_STOP) -> Optional[str]:
    """Why this refusal is refused, or None when it is still open.

    The count is per Task and per reason, so a Task refused once for one reason and once for another
    is still open: two reasons are two things to say. The same reason `limit` times is the request
    that has already been answered twice, and the message says so with what buys something instead.
    """
    reason = (reason or "").strip()
    seen = [row for row in refusals_recorded(workdir)
            if str(row.get("target") or "") == task_id and _reason_of(row) == reason and not row.get("blocked")]
    if len(seen) < limit:
        return None
    rounds = ", ".join(str(row.get("round") or "?") for row in seen)
    return (f"task {task_id} has already been refused {len(seen)} times for this reason (round {rounds}) "
            f"and nothing about it has moved; a third refusal is refused. What buys something "
            f"instead: {REFUSE_ALTERNATIVES}.")


def refused_twice(workdir: Any, limit: int = REFUSE_STOP) -> list[str]:
    """The Tasks refused `limit` times for one reason, for the status line: "refused twice: <task>"."""
    counts: dict[tuple[str, str], int] = {}
    for row in refusals_recorded(workdir):
        if row.get("blocked"):
            continue
        counts[(str(row.get("target") or ""), _reason_of(row))] = \
            counts.get((str(row.get("target") or ""), _reason_of(row)), 0) + 1
    return sorted({task for (task, _), seen in counts.items() if task and seen >= limit})


def refuse_repeats(workdir: Any, round_no: Optional[int] = None) -> int:
    """How many refusals repeated a Task and reason already recorded, blocked ones counted.

    A blocked third attempt is recorded with `blocked` so the count is readable off the same file
    the round report reads and does not need the session; a repeat is any row whose Task and reason
    an earlier row already carried. `round_no` narrows it to one round.
    """
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for row in refusals_recorded(workdir):
        key = (str(row.get("target") or ""), _reason_of(row))
        if key in seen and (round_no is None or int(row.get("round") or 0) == int(round_no)):
            repeats += 1
        seen.add(key)
    return repeats


def _executor(workdir: Any, verb: str, target_of: Any, round_of: Optional[Callable[[], int]] = None,
              detail: Optional[str] = None, guard: Optional[Callable[[Any, Any], Optional[str]]] = None) -> Any:
    async def execute(args: Any) -> RepairResult:
        target = target_of(args)
        locked = guard(workdir, args) if guard is not None else None
        if locked is not None:
            # The blocked attempt is recorded too, marked, so the round report and `refuse_repeats`
            # read the whole of what the session tried off the one file, and the strike count itself
            # skips the marked rows: a block is not a third answer.
            record_request(workdir, verb, target,
                           {"arguments": args.model_dump(mode="json"), "blocked": True,
                            "changed": False, "hash_before": None, "hash_after": None},
                           round_no=int(round_of()) if round_of is not None else 1)
            raise PermissionError(f"{verb} refused: {locked}")
        extra: dict[str, Any] = {}
        before = target_hash(workdir, verb, target)
        if verb == "repair_rewrite_skill":
            from kullback.builder import skills as skills_mod
            written = skills_mod.write_skill(workdir, args.name, args.content)
            extra = {"skill_hash": written["hash"]}
        # Every request carries the same change fields, so a report reads one shape whether the verb
        # decided something (no artifact, never changed) or acted (the hash either side of the call).
        path = record_request(workdir, verb, target,
                              {"arguments": args.model_dump(mode="json"), **extra,
                               **change_of(workdir, verb, target, before)},
                              round_no=int(round_of()) if round_of is not None else 1)
        return RepairResult(verb=verb, target=target, path=str(path),
                            detail=detail or f"request in {path.name}")
    return execute


def repair_tools(workdir: Any, sink: Optional[Sink] = None,
                 round_of: Optional[Callable[[], int]] = None) -> list[AgentTool]:
    """The repair verbs over one workdir as request records; `sink` is accepted for symmetry.

    A session registers the three deciding verbs from here (`repair_refuse_task`, `repair_escalate`,
    `repair_record_finding`)
    and takes the two acting ones from `builder/tools.py`, which run the repairing stage as well as
    recording the request. `repair_rewrite_skill` is registered nowhere yet (the GEPA caution).

    `round_of` is asked, at the moment a verb is called, which round the driver is in; the verbs are
    registered once and the round moves under them, so the round cannot be bound here. Without it
    every request is round 1, which is what a build with no round driver is.
    """
    return [
        AgentTool("repair_recompile", "Compile one tool's body again from its recorded calls.",
                  RecompileArgs, RepairResult,
                  _executor(workdir, "repair_recompile", lambda a: a.name, round_of), render=_render),
        AgentTool("repair_grow", "Grow one table of the Starting state with synthetic rows (D107).",
                  GrowRepairArgs, RepairResult,
                  _executor(workdir, "repair_grow", lambda a: a.table, round_of), render=_render),
        AgentTool("repair_rewrite_skill", "Rewrite one Builder skill; the edit is a memory-tree node with its content hash.",
                  RewriteSkillArgs, RepairResult,
                  _executor(workdir, "repair_rewrite_skill", lambda a: a.name, round_of), render=_render),
        AgentTool("repair_refuse_task", "Record that a Task is not worth deriving anything from. This "
                  "repairs nothing and moves no gate: it writes one row for the round report, and "
                  "whether the Task is refused is the Examiner's under the refuse gate. An Intent the "
                  "evidence does not support is repaired with repair_intent, not refused here. "
                  "The same Task refused twice for the same reason is refused a third time here, "
                  "with what buys something instead.",
                  RefuseTaskArgs, RepairResult,
                  _executor(workdir, "repair_refuse_task", lambda a: a.task_id, round_of,
                            detail="one row for the round report; no gate moves and no artifact changes",
                            guard=lambda wd, a: refuse_lock(wd, a.task_id, a.reason)),
                  render=_render),
        AgentTool("repair_escalate", "Escalate a Task to a person on a named queue.",
                  EscalateArgs, RepairResult,
                  _executor(workdir, "repair_escalate", lambda a: a.task_id, round_of), render=_render),
        AgentTool("repair_record_finding", "Record a behaviour of the recorded system that its tool's "
                  "description does not say and the body reproduces (D155): the leaf where the recorded "
                  "calls agree with each other and part from the description, with the call ids as "
                  "evidence. The recording is the standard a Candidate is graded against, so the body "
                  "reproduces it; the row is what the customer reads. Moves no gate, changes no artifact.",
                  RecordFindingArgs, RepairResult,
                  _executor(workdir, "repair_record_finding", lambda a: a.tool, round_of,
                            detail="one row for the customer's report; no gate moves and no artifact changes"),
                  render=_render),
    ]


# --- ratchet -----------------------------------------------------------------

def load_prior_bodies(workdir: Any) -> dict:
    """The last `bodies.json` this workdir wrote, or {} when the build never got that far."""
    path = Path(workdir) / "bodies.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def ratchet_bodies(prior: dict, new: dict, gate_passed: dict[str, bool],
                   refused: Iterable[str] = ()) -> dict:
    """Never replace a passing artifact with a failing one: keep the prior body where the new gate failed.

    `gate_passed` names, per tool, whether the new body cleared its gates; a tool absent from the
    map keeps its new body. Tools only in `prior` stay; tools only in `new` are kept.

    `refused` names the tools whose prior body is itself refused today, and those are not kept: a
    body cleared the gates of the build it was written in, and a build that adds a gate (D162's
    `compile_tools.memorised_values` is the first) can hold a prior body no gate accepts any more.
    Ratcheting onto it would keep a body every later round has to repair and never can.
    """
    out = dict(new)
    refused = set(refused)
    prior_bodies = prior.get("bodies", prior) if isinstance(prior, dict) else {}
    new_bodies = new.get("bodies", new) if isinstance(new, dict) else {}
    if not isinstance(prior_bodies, dict) or not isinstance(new_bodies, dict):
        return new
    merged = dict(new_bodies)
    for name, body in prior_bodies.items():
        if name in refused:
            continue  # the prior body fails a gate of this build; there is nothing to ratchet onto
        if name not in merged:
            merged[name] = body  # the new build omits it; the last passing body stays
        elif name in gate_passed and gate_passed[name] is False:
            merged[name] = body
    if isinstance(new, dict) and "bodies" in new:
        out = dict(new)
        out["bodies"] = merged
        return out
    return merged


def apply_ratchet(workdir: Any, new_bodies: dict, gate_passed: dict[str, bool],
                  refused: Iterable[str] = ()) -> dict:
    """`ratchet_bodies` against this workdir's last `bodies.json`."""
    return ratchet_bodies(load_prior_bodies(workdir), new_bodies, gate_passed, refused)


def ratchet_hook(workdir: Any, refused: Iterable[str] = ()) -> Any:
    """A `tool_result` handler: a `compile_tool` failure restores the prior passing body in details.

    `refused` is `ratchet_bodies`' own: a tool whose prior body fails a gate of this build is not
    restored, because there is nothing there to ratchet onto (D162).
    """
    refused = set(refused)

    def on_result(call: Any, result: ToolResult) -> Optional[ToolResult]:
        name = getattr(call, "name", None)
        if name != "compile_tool" or result.is_error or not result.details:
            return None
        rulings = result.details.get("stage_gates") or []
        failed = [r.get("stage") for r in rulings if isinstance(r, dict) and not r.get("passed")]
        if not failed:
            return None
        args = getattr(call, "arguments", {}) or {}
        tool = args.get("name", "")
        if tool in refused:
            return None
        prior = load_prior_bodies(workdir)
        bodies = prior.get("bodies", prior) if isinstance(prior, dict) else {}
        if not isinstance(bodies, dict) or tool not in bodies:
            return None
        details = dict(result.details)
        details["ratchet_restored"] = tool
        details["ratchet_restored_body"] = bodies[tool]
        content = f"{result.content}\nratchet: kept prior passing body for {tool}"
        return ToolResult(content=content, details=details, is_error=False)

    on_result.hook_name = "ratchet"  # type: ignore[attr-defined]
    return on_result


# --- lesson --------------------------------------------------------------------

def _rulings_of(workdir: Any, tool: str) -> list[dict]:
    """This tool's recorded rulings off `tool_builds.json`, the latest attempt's first.

    The compile_tools stage writes every attempt's rulings there, so a lesson is read off what code
    wrote and never off a model. An empty list when the file is missing, unreadable or silent about
    this tool.
    """
    path = Path(workdir) / "tool_builds.json"
    if not path.is_file():
        return []
    try:
        builds = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    row = builds.get(tool) if isinstance(builds, dict) else None
    nodes = row.get("nodes") if isinstance(row, dict) else None
    return [ruling for node in reversed(nodes if isinstance(nodes, list) else [])
            for ruling in (node.get("gates") if isinstance(node, dict) else None) or []
            if isinstance(ruling, dict)]


def gate_exception_line(workdir: Any, tool: str) -> str:
    """The exception this tool's latest failing gate reports, as `ExceptionClass: message`.

    Read off `tool_builds.json`, which the compile_tools stage writes with every attempt's rulings;
    the latest attempt is asked first, so the line is the one the tool crashed on last. "" when the
    file is missing, unreadable, or names no exception for this tool.
    """
    # Local: repair.py is imported by the session tools, and compile_env pulls the sandbox and the
    # gates behind it. Nothing here needs them until a lesson is actually written.
    from kullback.builder.compile_env import exception_in

    for ruling in _rulings_of(workdir, tool):
        if ruling.get("pass"):
            continue
        for failure in ruling.get("failures") or []:
            found = exception_in(str(failure))
            if found:
                return found
    return ""


def memorised_values_lesson(workdir: Any, tool: str) -> str:
    """The one sentence a body refused for memorising the recordings leaves behind (D162).

    Read off `tool_builds.json` the same way `gate_exception_line` is: the latest attempt first, so
    a tool that has since been written properly leaves nothing. The gate's own failures name the
    literals, which belong to one attempt; this is the lesson, which belongs to the tool, and it
    says the one thing every such failure is repaired by.
    """
    for ruling in _rulings_of(workdir, tool):
        if ruling.get("stage") == gates.MEMORISED_STAGE and not ruling.get("pass"):
            return gates.MEMORISED_LESSON
    return ""


def sensitivity_lesson(workdir: Any, tool: str) -> str:
    """The one sentence a body refused for answering two worlds alike leaves behind (D195).

    Read off `tool_builds.json` the way the memorised-values lesson is, and it names the columns the
    ruling recorded, because "read the world" is what the writer of a memorising body already
    believes it did. The columns are the ruling's own metric, so nothing here parses a failure
    sentence. Only the latest ruling of this stage is asked, whichever way it went: a tool whose
    last attempt read the columns has learned the lesson, and repeating it to the next writer is
    telling it to repair what it has already repaired.
    """
    for ruling in _rulings_of(workdir, tool):
        if ruling.get("stage") == gates.SENSITIVITY_STAGE:
            return ("" if ruling.get("pass") else
                    gates.sensitivity_lesson((ruling.get("metrics") or {}).get("columns") or []))
    return ""


def record_tool_lesson(workdir: Any, tool: str, failures: list[str]) -> Path:
    """Write one gate-failure sequence to the Builder memory for this workdir.

    The mechanic's hint says what it thinks went wrong; the gate says what the body actually raised,
    and the two are not the same sentence. Both go into the lesson, so the compiler's next prompt
    (`memory.lesson_for`, read by `compile_tool`) carries the error line it must not raise again and
    not only the advice. The line is added once and only when the hint does not already quote it.
    The memorised-values lesson (D162) is added the same way: a body refused for holding an id out
    of a recorded call is repaired by one sentence, and the hint rarely says it.
    """
    failures = list(failures)
    error = gate_exception_line(workdir, tool)
    if error and not any(error in failure for failure in failures):
        failures.append(f"the last attempt raised {error}; do not raise it again")
    memorised = memorised_values_lesson(workdir, tool)
    if memorised and not any(memorised in failure for failure in failures):
        failures.append(memorised)
    # D195 the same way: a body that answered two Tasks alike is repaired by naming the columns it
    # has to read off the world, and the hint almost never names a column.
    sensitivity = sensitivity_lesson(workdir, tool)
    if sensitivity and not any(sensitivity in failure for failure in failures):
        failures.append(sensitivity)
    return memory_mod.record_lesson(workdir, tool, failures)


def lesson_for_tool(workdir: Any, tool: str) -> str:
    """The failure sequences to inject into the next attempt's prompt for this tool."""
    return memory_mod.lesson_for(workdir, tool)
