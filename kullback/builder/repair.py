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
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

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


def _first_failure(node: Any) -> str:
    """The first thing that went wrong in one compile attempt, in the gate's own words.

    An attempt that never reached the sandbox (a refusal, an empty reply) carries `failures`; one
    that ran carries the gates it was put through, whose rulings use the record alias `pass`.
    """
    if not isinstance(node, dict):
        return ""
    for failure in node.get("failures") or []:
        return str(failure)
    for ruling in node.get("gates") or []:
        if isinstance(ruling, dict) and not ruling.get("pass"):
            failures = ruling.get("failures") or []
            return f"{ruling.get('stage')}: {failures[0]}" if failures else f"{ruling.get('stage')} failed"
    return ""


def intent_ruling(workdir: Any, task_id: str) -> str:
    """Whether this Task's Intent grounded, off the file the intent stage just wrote."""
    record = intent_record(workdir, task_id)
    if record.get("grounded"):
        return f'repair_intent {task_id}: grounded: "{record.get("text") or ""}"'
    return f"repair_intent {task_id}: still refused: {record.get('reason') or NO_INTENT}"


def recompile_ruling(workdir: Any, name: str) -> str:
    """Whether this tool's new body cleared the gates, off `tool_builds.json`.

    A tool that ended assisted is a body no attempt got through (D49), so the line carries the first
    failure of its last attempt: that is what the next hint has to answer.
    """
    build = tool_build(workdir, name)
    if not build:
        return f"repair_recompile {name}: {NO_ATTEMPT}"
    if not build.get("assisted"):
        return f"repair_recompile {name}: cleared the gates"
    failures = [text for text in (_first_failure(node) for node in build.get("nodes") or []) if text]
    return f"repair_recompile {name}: still assisted: {failures[-1] if failures else NO_FAILURE}"


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


def _executor(workdir: Any, verb: str, target_of: Any, round_of: Optional[Callable[[], int]] = None,
              detail: Optional[str] = None) -> Any:
    async def execute(args: Any) -> RepairResult:
        target = target_of(args)
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
    """The five repair verbs over one workdir as request records; `sink` is accepted for symmetry.

    A session registers the two deciding verbs from here (`repair_refuse_task`, `repair_escalate`)
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
                  "evidence does not support is repaired with repair_intent, not refused here.",
                  RefuseTaskArgs, RepairResult,
                  _executor(workdir, "repair_refuse_task", lambda a: a.task_id, round_of,
                            detail="one row for the round report; no gate moves and no artifact changes"),
                  render=_render),
        AgentTool("repair_escalate", "Escalate a Task to a person on a named queue.",
                  EscalateArgs, RepairResult,
                  _executor(workdir, "repair_escalate", lambda a: a.task_id, round_of), render=_render),
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


def ratchet_bodies(prior: dict, new: dict, gate_passed: dict[str, bool]) -> dict:
    """Never replace a passing artifact with a failing one: keep the prior body where the new gate failed.

    `gate_passed` names, per tool, whether the new body cleared its gates; a tool absent from the
    map keeps its new body. Tools only in `prior` stay; tools only in `new` are kept.
    """
    out = dict(new)
    prior_bodies = prior.get("bodies", prior) if isinstance(prior, dict) else {}
    new_bodies = new.get("bodies", new) if isinstance(new, dict) else {}
    if not isinstance(prior_bodies, dict) or not isinstance(new_bodies, dict):
        return new
    merged = dict(new_bodies)
    for name, body in prior_bodies.items():
        if name not in merged:
            merged[name] = body  # the new build omits it; the last passing body stays
        elif name in gate_passed and gate_passed[name] is False:
            merged[name] = body
    if isinstance(new, dict) and "bodies" in new:
        out = dict(new)
        out["bodies"] = merged
        return out
    return merged


def apply_ratchet(workdir: Any, new_bodies: dict, gate_passed: dict[str, bool]) -> dict:
    """`ratchet_bodies` against this workdir's last `bodies.json`."""
    return ratchet_bodies(load_prior_bodies(workdir), new_bodies, gate_passed)


def ratchet_hook(workdir: Any) -> Any:
    """A `tool_result` handler: a `compile_tool` failure restores the prior passing body in details."""

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

def gate_exception_line(workdir: Any, tool: str) -> str:
    """The exception this tool's latest failing gate reports, as `ExceptionClass: message`.

    Read off `tool_builds.json`, which the compile_tools stage writes with every attempt's rulings;
    the latest attempt is asked first, so the line is the one the tool crashed on last. "" when the
    file is missing, unreadable, or names no exception for this tool.
    """
    # Local: repair.py is imported by the session tools, and compile_env pulls the sandbox and the
    # gates behind it. Nothing here needs them until a lesson is actually written.
    from kullback.builder.compile_env import exception_in

    path = Path(workdir) / "tool_builds.json"
    if not path.is_file():
        return ""
    try:
        builds = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    row = builds.get(tool) if isinstance(builds, dict) else None
    nodes = row.get("nodes") if isinstance(row, dict) else None
    for node in reversed(nodes if isinstance(nodes, list) else []):
        rulings = node.get("gates") if isinstance(node, dict) else None
        for ruling in rulings if isinstance(rulings, list) else []:
            if not isinstance(ruling, dict) or ruling.get("pass"):
                continue
            for failure in ruling.get("failures") or []:
                found = exception_in(str(failure))
                if found:
                    return found
    return ""


def record_tool_lesson(workdir: Any, tool: str, failures: list[str]) -> Path:
    """Write one gate-failure sequence to the Builder memory for this workdir.

    The mechanic's hint says what it thinks went wrong; the gate says what the body actually raised,
    and the two are not the same sentence. Both go into the lesson, so the compiler's next prompt
    (`memory.lesson_for`, read by `compile_tool`) carries the error line it must not raise again and
    not only the advice. The line is added once and only when the hint does not already quote it.
    """
    failures = list(failures)
    error = gate_exception_line(workdir, tool)
    if error and not any(error in failure for failure in failures):
        failures.append(f"the last attempt raised {error}; do not raise it again")
    return memory_mod.record_lesson(workdir, tool, failures)


def lesson_for_tool(workdir: Any, tool: str) -> str:
    """The failure sequences to inject into the next attempt's prompt for this tool."""
    return memory_mod.lesson_for(workdir, tool)
