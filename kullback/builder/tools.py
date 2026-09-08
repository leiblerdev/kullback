"""The Builder's stages and repair verbs as tools of the agent core: pydantic arguments in, a short
result out (D120, D123, D135, D138).

Each acting tool runs one target of the build plan's graph through `build.execute`, so the
scheduler, not the model, decides what runs first: a tool whose inputs are stale gets them resolved
upstream, a tool whose inputs are current is served from the cache. The graph is kept for that and
for nothing else; which target to ask for is the model's call. `build(target)` is the general form,
where the target is `environment` (the whole build), a stage name or an artifact name; the others
are the verbs a Builder session reaches for by name. `recluster()` re-runs the clustering under its
fixed configuration, `grow(table, count)` grows one table of the Starting state with synthetic rows
(D107), `compile_tool(name)` recompiles one tool body, `replay(task)` replays one Task's Traces,
`reroll(task)` re-rolls one Task.

`status()` is where a round starts: it reads the red lights off what code wrote (gates.json, the
fidelity records in replays.json, the assisted tools in tool_builds.json) and never off a model,
and names for each the stage, the tool or Task, the failure in the gate's own words and the repair
verb that owns it. It shows all of them. A build with hundreds of red lights used to be rendered as
the first twenty-five and "and 355 more, all in details", and a model that never asked for the
details acted on the four tools it could see; so the rendering groups instead of cutting: a headline,
then one block per failing gate holding every tool it names with a count and one failure, and every
Task id it names on a wrapped line grouped by the kind of failure. `status(gate=...)` and
`status(target=...)` zoom on one of those and list its red lights in full, uncut and unclamped. The
repair verbs are `repair_recompile(name, hint)`, `repair_grow(table, count)`
and `repair_intent(task_id, hint)`, which record the request and then run the stage that repairs the
artifact, and `repair_refuse_task` and `repair_escalate` from `builder/repair.py`, which record a
decision and change no artifact. A ruling is mapped to a deciding verb only where nothing can be
repaired: an Intent is a line a model wrote, so the intent gate leads to `repair_intent` (D136), and
`derive_verifier` leads to neither, because the Examiner owns it (D123). `repair_rewrite_skill` is
deliberately not registered (the GEPA caution in docs/todo.md).

A result is what the model reads plus what it does not. The rendered text is a few lines: for a
repair verb the target's own ruling first, then the status, the stages that ran, and the ruling
names with pass or fail. The target's ruling is there because a gate rules over every target at
once: a live build repaired seven Intents, six of them grounded, and every result carried the
intent gate's first failure, which was about a Task nobody had touched, so the model read all seven
as refused and stopped. The gate-wide line stays underneath and now counts, so a gate failing over
many targets says how many and names its first as an example rather than as a verdict. Everything
else (the Task ids, the environment id, each stage's report) is in the result model and reaches the
transcript's `details`, which never enters the context.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback import round_delta
from kullback.agent.tools import AgentTool, NoArgs, counted_ruling_line
from kullback.builder import build as build_module
from kullback.builder import repair as repair_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.gates import Ruling, ruling_of
from kullback.gates.fidelity import unconfirmed_reason

Sink = Callable[[Any], Awaitable[None]]

# The tools that run the graph, so a driver can tell a build from a reading or a decision.
BUILD_TOOLS: tuple[str, ...] = ("build", "recluster", "grow", "compile_tool", "replay", "reroll",
                                "repair_recompile", "repair_grow", "repair_intent")


class StageReport(BaseModel):
    """How one stage ended, from the pipeline's own report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    cached: bool = False
    attempts: int = 0
    rulings: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    produced: list[str] = Field(default_factory=list)


class BuildResult(BaseModel):
    """What one Builder tool produced: a summary, the stage gates' rulings pass or fail, and the payload apart.

    `stage_gates` is the ruling of each stage's own gate, the one that can roll a stage back, and
    `passed` rules over exactly those. Every ruling the run recorded through the ledger, which is
    the longer list, is in `payload["rulings"]`.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str
    target: str
    status: str
    passed: bool
    # How the one Task, tool or table a repair verb was called on came out, read off the artifact
    # the stage just wrote (`repair.target_ruling`). Empty for a tool that has no single target.
    target_ruling: str = ""
    stage_gates: list[Ruling] = Field(default_factory=list)
    stages: list[StageReport] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    stopped: Optional[dict[str, Any]] = None
    payload: dict[str, Any] = Field(default_factory=dict)


def render(result: BuildResult) -> str:
    """The lines the model reads: the target's own ruling, the status, the stages, the rulings.

    A repair verb's result opens with what happened to the one target it was called on, before
    anything a gate says. The gate-wide line still follows, and a gate failing over many targets
    now says how many and names its first as an example (`counted_ruling_line`), so neither line
    can be read as a verdict on the target the other one is about. The payload stays in details.
    """
    lines = [result.target_ruling] if result.target_ruling else []
    lines.append(result.summary)
    # status is already "cached" for a stage the cache served, so the flag would only say it twice.
    ran = [f"{s.name} ({s.status})" for s in result.stages if s.status != "pending"]
    if ran:
        lines.append("stages: " + ", ".join(ran))
    if result.stage_gates:
        lines.append(counted_ruling_line("rulings", result.stage_gates))
    if result.failed_stage:
        lines.append(f"failed stage: {result.failed_stage}")
    if result.stopped:
        lines.append(f"stopped: {result.stopped.get('reason') or 'spend ceiling reached'}")
    return "\n".join(lines)


class BuildArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(default=TARGET_ALL,
                        description="`environment` for the whole build, or the name of one stage or one artifact.")


class GrowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="The table of the Starting state to grow.")
    count: int = Field(ge=1, description="How many rows the table should hold after growing (D107).")


class CompileToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The mined tool whose body to compile again.")


class ReplayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The Task whose Traces to replay through the built tools.")


class RerollArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The Task to re-roll with the frontier model (D112).")


# --- the red lights: what is failing, and which verb owns it ------------------

# A ruling no Builder verb answers because another agent owns the artifact; the red light says so
# rather than naming a verb that changes nothing.
EXAMINER_OWNS = "the Examiner owns it"

# Which verb answers which ruling. The ruling names are the registry's (kullback/gates), so the map
# is over the harness's own vocabulary and holds for any customer's traces. Every entry is a verb
# that changes the artifact the ruling is about: a verb that only records a decision belongs here
# only where nothing can be repaired.
REPAIR_VERB_FOR: dict[str, str] = {
    "parses": "repair_recompile",
    "executes_on_s0": "repair_recompile",
    "deterministic": "repair_recompile",
    "non_trivial": "repair_recompile",
    "replay_fidelity": "repair_recompile",
    "refuses_unknown": "repair_recompile",
    "confined": "repair_recompile",
    "compile_tools": "repair_recompile",
    "compile_tools.parses": "repair_recompile",
    "compile_tools.executes": "repair_recompile",
    "compile_tools.deterministic": "repair_recompile",
    "compile_tools.non_trivial": "repair_recompile",
    "compile_tools.replay_fidelity": "repair_recompile",
    # A body holding an id or a value it copied out of a recorded call is written again, with the
    # lookup over the world's tables the hint asks for (D162).
    "compile_tools.memorised_values": "repair_recompile",
    # A Task whose Traces do not replay to their End state is a tool that answers differently.
    "replay_reference": "repair_recompile",
    # A row the Traces name that the built world does not hold is a table to grow (D107).
    "build_environment": "repair_grow",
    # An Intent is a line a model wrote, so it is repaired, not refused: the phrases the Runs do not
    # evidence go back to the model with the hint (D136). Refusing was the answer here for one live
    # build, and repair_refuse_task moved no gate 41 times.
    "intent": "repair_intent",
    # Re-rolls that all died on a provider error say nothing about the Task; the Builder re-rolls it
    # again rather than refusing a Task on the strength of an error.
    "rerolls": "reroll",
    # The Verifier is derived by the Examiner (D123): no Builder verb touches it.
    "derive_verifier": EXAMINER_OWNS,
}
DEFAULT_REPAIR_VERB = "repair_escalate"
NO_BODY = " has no body"
# A tool that ended assisted is the one red light no ruling names, so it is written here and read
# back here; the headline counts the assisted tools off this wording and off nothing else.
ASSISTED = " is assisted: no generated body cleared the gates (D49)"
# Read back the same way, off `tool_builds.json`: an assisted body that answered every recorded call
# the same way whatever its arguments were, which a hint written against one failing call cannot reach.
HARDCODED = " and hardcoded: it answers every recorded call alike, whatever it is given"


def verb_for(stage: str) -> str:
    """The verb that owns a ruling; anything the Builder cannot act on goes to a person or its owner."""
    return REPAIR_VERB_FOR.get(stage, DEFAULT_REPAIR_VERB)


class RedLight(BaseModel):
    """One failing gate as the model reads it: where, on what, in the gate's own words, and the verb."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    kind: str = "build"
    target: str = ""
    failure: str = ""
    verb: str = DEFAULT_REPAIR_VERB


class StatusResult(BaseModel):
    """Every red light the workdir holds, with the rulings that are passing beside them.

    `red_lights` is the whole set for a plain `status()` and the matching set for a zoom; `passing`
    and `failing` are the whole picture either way, so a zoom never hides which gates are red.
    `zoom` is the filter in the words the model passed it, empty when there was none, and it is
    what tells the rendering to list every red light in full instead of grouping them.

    `unbuilt` is the workdir that holds no ruling at all: nothing has been built, so no gate has
    ruled and the green wording would be a lie (D166).
    """

    model_config = ConfigDict(extra="forbid")

    summary: str
    red_lights: list[RedLight] = Field(default_factory=list)
    passing: list[str] = Field(default_factory=list)
    failing: list[str] = Field(default_factory=list)
    zoom: str = ""
    unbuilt: bool = False


class StatusArgs(BaseModel):
    """How to zoom `status`: no argument is the whole grouped picture, either one narrows it."""

    model_config = ConfigDict(extra="forbid")

    gate: str = Field(default="", description="Only the red lights of this gate, each in full "
                                              "(a stage or ruling name off the grouped status).")
    target: str = Field(default="", description="Only the red lights about this tool or Task, each "
                                                "in full (a name or Task id off the grouped status).")


def _read_json(path: Path, default: Any) -> Any:
    """One record file, or the default when it is missing or half written."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _named_by(failure: str) -> tuple[str, str]:
    """The tool or Task a failure names, and which of the two, from the gate's own wording.

    The gates write `task <id>: <reason>` for a Task, `<tool>: <reason>` for a tool and
    `<tool> has no body` for a tool with nothing compiled; anything else names neither. A sandbox
    ruling names the call rather than the tool (`get_order({"id": 1}): ...`), so the arguments are
    cut off: the verb takes a tool name, and the model would have to cut them itself otherwise.
    """
    text = failure.strip()
    if text.startswith("task "):
        return text[len("task "):].split(":", 1)[0].strip(), "task"
    if text.endswith(NO_BODY):
        return text[: -len(NO_BODY)].strip(), "tool"
    head, _, rest = text.partition(":")
    head = head.strip()
    if rest and head and " " not in head:
        return head.split("(", 1)[0].strip() or head, "tool"
    return "", "build"


def red_lights(workdir: Any) -> list[RedLight]:
    """Every failing gate this workdir holds, off the records code wrote and never off a model.

    Four files, because one is not enough: `gates.json` is every ruling the stages recorded, but
    the compile_tools stage overwrites it with its own per-tool rulings, so the fidelity records in
    `replays.json` are read for the Tasks whose replay ruling is no longer in the file, and
    `tool_builds.json` for the tools that ended assisted, which is a body the Builder could not
    write (D49) and the one red light no ruling names. `tool_fidelity.json` says what an assisted
    tool costs in Tasks (D171), which the assisted light alone does not.
    """
    workdir = Path(workdir)
    out: list[RedLight] = []
    for row in _read_json(workdir / "gates.json", []) or []:
        if not isinstance(row, dict) or row.get("pass"):
            continue
        stage = str(row.get("stage") or "")
        failures = [str(f) for f in (row.get("failures") or [])] or [f"{stage} did not pass"]
        for failure in failures:
            target, kind = _named_by(failure)
            out.append(RedLight(stage=stage, kind=kind, target=target, failure=failure,
                                verb=verb_for(stage)))
    seen = {(light.stage, light.target) for light in out}
    for task_id, per_task in sorted((_read_json(workdir / "replays.json", {}) or {}).items()):
        rows = per_task if isinstance(per_task, dict) else {}
        if any(r.get("confirmed") for r in rows.values()) or ("replay_reference", task_id) in seen:
            continue
        out.append(RedLight(stage="replay_reference", kind="task", target=task_id,
                            failure=f"task {task_id}: {unconfirmed_reason(rows)}",
                            verb=verb_for("replay_reference")))
    fidelity = _read_json(workdir / "tool_fidelity.json", {}) or {}
    # D191: how many recompiles in a row this tool has bought nothing with, so a Builder reading a
    # red light knows the difference between one it has not tried and one it has tried four times.
    kept = _read_json(workdir / repair_module.KEPT_BODIES_FILE, {}) or {}
    for name, row in sorted((_read_json(workdir / "tool_builds.json", {}) or {}).items()):
        if isinstance(row, dict) and row.get("assisted"):
            out.append(RedLight(stage="compile_tools", kind="tool", target=name,
                                failure=f"{name}{ASSISTED}"
                                        f"{HARDCODED if row.get('hardcoded') else ''}"
                                        f"{_blocked_note(fidelity, name)}"
                                        f"{_declined_note(row)}"
                                        f"{repair_module.stalled_note(kept.get(name) or {} if isinstance(kept, dict) else {})}",
                                verb="repair_recompile"))
    return out


def _declined_note(row: dict) -> str:
    """The last recompile of this tool scored no higher than the body kept, so it was declined
    (D174); saying so is what stops the next request from asking the same question."""
    declined = row.get("recompile_declined")
    if not isinstance(declined, dict):
        return ""
    attempt, kept = declined.get("attempt_score") or [], declined.get("kept_score") or []
    return (f"; the last recompile scored {attempt} against the kept body's {kept} "
            f"(gates passed, calls matched) and was declined: change the hint, not the request")


def _blocked_note(fidelity: Any, name: str) -> str:
    """What an assisted tool actually costs, off tool_fidelity.json (D171).

    Assisted is a corpus ruling: one recorded call the body answers differently is enough. Left at
    that, the model reads the tool as the blocker of every Task that calls it, and on one live
    build 51 of the 64 Tasks that read as blocked have no own call any assisted body answers
    differently. This says how many Tasks call the tool and how many have an own call it answers
    differently, which is what a recompile of it buys.
    """
    rows = [row[name] for row in ((fidelity or {}).get("tasks") or {}).values()
            if isinstance(row, dict) and name in row]
    if not rows:
        return ""
    per_tool = ((fidelity or {}).get("tools") or {}).get(name) or {}
    blocked = sum(1 for row in rows if row.get("differing"))
    return (f" ({per_tool.get('replayed', 0)} of {per_tool.get('calls', 0)} recorded calls replay; "
            f"{_count(len(rows), 'Task')} call it, {blocked} blocked by their own differing calls)")


def _kinded(light: RedLight) -> tuple[str, str]:
    """One failure split into the kind two targets can share and the detail that tells them apart.

    The gates write `task <id>: <kind>: <detail>`, so cutting the id off the front and taking the
    head before the next colon leaves the kind ("noun phrases with no span") and the detail that
    varies (the ungrounded phrases themselves). A reason with no colon left is its own kind. This
    reads the gates' own wording and holds for any customer's traces: nothing here knows a domain.
    """
    text = " ".join(light.failure.split())
    prefix = f"task {light.target}:"
    if light.target and text.startswith(prefix):
        text = text[len(prefix):].strip()
    kind, _, detail = text.partition(":")
    kind = kind.strip()
    return (kind or text), detail.strip()


def _no_verdict(lights: list[RedLight]) -> tuple[list[str], str]:
    """The Tasks a red light leaves without a Verdict, and the kind of failure most of them share.

    Every gate whose failures name Tasks (intent, replay_reference, derive_verifier) leaves those
    Tasks uncounted rather than failing the build (design section 6), so the Tasks a status is about
    are exactly the targets of its Task-shaped red lights.
    """
    tasks = sorted({light.target for light in lights if light.kind == "task" and light.target})
    kinds = [_kinded(light)[0] for light in lights if light.kind == "task"]
    top = max(sorted(set(kinds)), key=kinds.count) if kinds else ""
    return tasks, top


def _assisted(lights: list[RedLight]) -> list[str]:
    """The tools that ended assisted, off the wording `red_lights` wrote for them (D49)."""
    return sorted({light.target for light in lights if light.target and ASSISTED in light.failure})


def _hardcoded(lights: list[RedLight]) -> list[str]:
    """The assisted tools whose kept body never read its arguments, off the same wording."""
    return sorted({light.target for light in lights if light.target and HARDCODED in light.failure})


def _count(n: int, noun: str, plural: str = "s") -> str:
    return f"{n} {noun}{'' if n == 1 else plural}"


def _headline(lights: list[RedLight], passing: list[str], failing: list[str], delta: str = "") -> str:
    """The one line the model reads first: how much is red, how many Tasks it costs, what is assisted.

    `delta` is what the round before moved (`round_delta.delta_line`). Without it the picture is of
    the artifacts as they stand, and one live build lost a third of its trusted Tasks over three
    rounds with every round reading the same as the last.
    """
    tasks, top = _no_verdict(lights)
    hardcoded = _hardcoded(lights)
    # Listed apart, because they are not the same repair: an assisted body has a defect a hint can
    # name, and a hardcoded one has no argument in it at all.
    assisted = [name for name in _assisted(lights) if name not in hardcoded]
    gates = len(set(passing) | set(failing))
    parts = [_count(len(lights), "red light"), f"{len(failing)} of {_count(gates, 'gate')} red"]
    parts.append(f"{_count(len(tasks), 'Task')} with no Verdict" + (f", most of them {top}" if top else "")
                 if tasks else "no Task left without a Verdict")
    parts.append(f"{_count(len(assisted), 'tool')} assisted: " + ", ".join(assisted)
                 if assisted else "no assisted tool")
    if hardcoded:
        parts.append(f"{_count(len(hardcoded), 'tool')} hardcoded, answering alike whatever they are "
                     f"given: " + ", ".join(hardcoded))
    if delta:
        parts.append(delta)
    return "status: " + "; ".join(parts)


# How the grouped picture is shaped. It is grouped rather than cut: every gate, every tool and every
# Task id is in it, and only a single failure text is ever shortened, which the zoom then shows whole.
STATUS_WIDTH = 100  # where a wrapped line of Task ids or examples folds
FAILURE_CHARS = 240  # how much of one representative failure the grouped view carries
KIND_EXAMPLES = 5  # how many Tasks of one kind show what their failure actually says
ZOOM_HINT = ('zoom: status(gate="<gate>") or status(target="<tool or Task>") lists every red light '
             'there in full.')


def _clamped(text: str) -> str:
    """One failure on one line; a very long one says how much of it the zoom would add."""
    text = " ".join(text.split())
    if len(text) <= FAILURE_CHARS:
        return text
    return f"{text[:FAILURE_CHARS].rstrip()} [+{len(text) - FAILURE_CHARS} characters, zoom for all of it]"


def _wrapped(prefix: str, items: list[str], indent: str, sep: str = ", ") -> list[str]:
    """One long enumeration folded onto as many lines as it takes; nothing is dropped."""
    lines = textwrap.wrap(prefix + sep.join(items), width=STATUS_WIDTH, initial_indent=indent,
                          subsequent_indent=indent + "  ", break_long_words=False,
                          break_on_hyphens=False)
    return lines or [indent + prefix.rstrip()]


def _task_lines(lights: list[RedLight]) -> list[str]:
    """The Task-shaped red lights of one gate: every id listed, grouped by the kind of failure."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for light in lights:
        kind, detail = _kinded(light)
        groups.setdefault(kind, []).append((light.target, detail))
    lines: list[str] = []
    for kind in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        rows = sorted(groups[kind])
        lines += _wrapped(f"{kind}: {_count(len(rows), 'Task')}: ", [t for t, _ in rows], "  ")
        shown = [f"{t}: {d}" for t, d in rows if d][:KIND_EXAMPLES]
        if shown:
            seen = f" ({len(shown)} of {len(rows)})" if len(rows) > len(shown) else ""
            lines += _wrapped(f"what fails{seen}: ", shown, "    ", sep="; ")
    return lines


def _gate_block(stage: str, lights: list[RedLight]) -> list[str]:
    """One failing gate: what it is about, then a line per tool and every Task id it names."""
    tasks = [light for light in lights if light.kind == "task"]
    rest = [light for light in lights if light.kind != "task"]
    kinds = {light.kind for light in lights if light.target}
    noun = {"task": "Task", "tool": "tool"}.get(next(iter(kinds)), "target") if len(kinds) == 1 else "target"
    targets = len({light.target for light in lights if light.target})
    verbs = list(dict.fromkeys(light.verb for light in lights))
    over = f" over {_count(targets, noun)}" if targets else ""
    head = f"{stage}: {_count(len(lights), 'red light')}{over} -> {', '.join(verbs)}"
    lines = [head, *_task_lines(tasks)]
    by_target: dict[str, list[RedLight]] = {}
    for light in rest:
        if light.target:
            by_target.setdefault(light.target, []).append(light)
    for target in sorted(by_target):
        rows = by_target[target]
        # The shortest failure is the representative: it is the one a long dump of differing keys
        # cannot crowd out, and `min` keeps the first of equals, so the choice never moves.
        owner = f" -> {rows[0].verb}" if len(verbs) > 1 else ""
        lines.append(f"  {target} ({len(rows)}): {_clamped(min((r.failure for r in rows), key=len))}{owner}")
    for light in rest:
        if not light.target:
            owner = f" -> {light.verb}" if len(verbs) > 1 else ""
            lines.append(f"  {_clamped(light.failure)}{owner}")
    return lines


UNBUILT_SUMMARY = ("status: nothing has been built in this workdir; no gate has ruled yet, so build "
                   "the target before reading the gates")


def status_of(workdir: Any, gate: str = "", target: str = "") -> StatusResult:
    """The red lights and the rulings that are passing, as the status tool returns them.

    `gate` and `target` narrow which red lights come back; the rulings passing and the gates failing
    are the whole picture either way, so a zoom answers about one gate without hiding the rest.

    A workdir holding no ruling at all says so instead of reading as green (D166): "0 red lights;
    0 of 0 gates red" and "the gates are green" is what a model read on a fresh workdir, after which
    it answered without building anything and the round closed.
    """
    workdir = Path(workdir)
    lights = red_lights(workdir)
    rows = [r for r in (_read_json(workdir / "gates.json", []) or []) if isinstance(r, dict)]
    passing = sorted({str(r.get("stage") or "") for r in rows if r.get("pass")})
    failing = list(dict.fromkeys(light.stage for light in lights))
    unbuilt = not rows and not (_read_json(workdir / "replays.json", {}) or {}) \
        and not (_read_json(workdir / "tool_builds.json", {}) or {})
    asked = [part for part in (f"gate={gate}" if gate else "", f"target={target}" if target else "") if part]
    shown = [light for light in lights
             if (not gate or light.stage == gate) and (not target or light.target == target)]
    summary = _headline(lights, passing, failing, round_delta.delta_line(workdir))
    if unbuilt:
        summary = UNBUILT_SUMMARY
    elif asked:
        summary = (f"status({', '.join(asked)}): {_count(len(shown), 'red light')} "
                   f"of {len(lights)} in all, each in full")
    return StatusResult(summary=summary, red_lights=shown, passing=passing, failing=failing,
                        zoom=", ".join(asked), unbuilt=unbuilt)


def _in_full(result: StatusResult) -> list[str]:
    """A zoom: one line per red light, nothing grouped, nothing shortened, nothing left out."""
    if not result.red_lights:
        where = ", ".join(result.failing) or "none"
        return [f"no red light matches; the gates that are red: {where}"]
    lines = []
    for light in result.red_lights:
        where = f" {light.target}" if light.target else ""
        lines.append(f"- {light.stage}{where}: {light.failure} -> {light.verb}")
    return lines


def render_status(result: StatusResult) -> str:
    """The headline, then every gate grouped; a zoom prints its red lights one by one instead.

    Nothing is cut from the grouped picture: every failing gate is a block, every tool it names is a
    line and every Task id it names is on a wrapped line under the kind of failure it shares. The
    one thing that is shortened is a single very long failure text, which says how much is left and
    is shown whole by the zoom the last line names. The order is the order the records were read in
    and then alphabetical, so two runs over one workdir render the same text.

    A workdir with no ruling in it renders one line, and not the green one (D166).
    """
    if result.unbuilt:
        return result.summary
    if result.zoom:
        return "\n".join([result.summary, *_in_full(result)])
    if not result.red_lights:
        return "\n".join([result.summary, "nothing is failing; the gates are green"])
    by_gate: dict[str, list[RedLight]] = {}
    for light in result.red_lights:
        by_gate.setdefault(light.stage, []).append(light)
    lines = [result.summary]
    for stage, lights in by_gate.items():
        lines += _gate_block(stage, lights)
    lines.append(ZOOM_HINT)
    return "\n".join(lines)


def result_of(plan: BuildPlan, target: str, result: Any, verb: str) -> BuildResult:
    """A pipeline result as the tool's result model, gate rulings read from the run's own reports."""
    stages = [StageReport(name=r.name, status=r.status, cached=r.cached, attempts=r.attempts,
                          rulings=list(r.rulings), elapsed_ms=r.elapsed_ms, produced=list(r.produced))
              for r in result.reports.values()]
    stage_gates = [ruling_of(g) for g in result.gates]
    ran = [s for s in stages if s.status != "pending"]
    cached = sum(1 for s in ran if s.cached)
    # A run whose every stage came from the cache rewrote no artifact, so no ruling can have moved.
    # The first line the model reads says so, rather than leaving two counts to be compared.
    served = (f"nothing changed: all {len(ran)} stages from cache" if ran and cached == len(ran)
              else f"{len(ran)} stages, {cached} from cache")
    summary = f"{verb} {target}: {result.status}; {served}, {len(result.rulings)} rulings recorded"
    environment = result.artifacts.get("environment")
    return BuildResult(
        summary=summary, target=target, status=result.status,
        passed=result.status == "complete" and all(r.passed for r in stage_gates),
        stage_gates=stage_gates, stages=stages,
        produced=[name for s in stages for name in s.produced],
        failed_stage=result.failed_stage, stopped=result.stopped,
        payload={"workdir": str(plan.workdir), "env_id": getattr(environment, "env_id", None),
                 "tasks": [t.id for t in result.artifacts.get("tasks", [])],
                 "rulings": list(result.rulings)})


def _bridged(previous: Optional[Callable[[Any], Any]], sink: Optional[Sink],
             loop: asyncio.AbstractEventLoop) -> Any:
    """The plan's stage events, sent on to the harness's stream from the thread the build runs on.

    The build runs on a worker thread so the event loop stays free; each typed event is handed to
    the sink on the loop and waited for, so the stream carries them in the order the stages emitted
    them rather than in the order the loop got round to them. A subscriber that raises still cannot
    stop a build: the exception travels back to `Pipeline._emit_typed`, which drops it, because a
    screen has no business failing a build that was going fine. A cancelled harness is the one case
    that guard cannot catch, since CancelledError is a BaseException, so the event is dropped here:
    a screen that has gone away is not a reason to fail the build that was writing to it.
    """

    def emit(event: Any) -> None:
        if previous is not None:
            previous(event)
        if sink is not None:
            try:
                asyncio.run_coroutine_threadsafe(sink(event), loop).result()
            except asyncio.CancelledError:
                return

    return emit


def _executor(plan: BuildPlan, sink: Optional[Sink], verb: str, target_of: Callable[[Any], str],
              narrowing_of: Callable[[Any], dict]) -> Callable[[Any], Awaitable[BuildResult]]:
    async def execute(args: Any) -> BuildResult:
        target, narrowing = target_of(args), narrowing_of(args)
        loop = asyncio.get_running_loop()
        previous = plan.emit
        plan.emit = _bridged(previous, sink, loop)
        try:
            result = await asyncio.to_thread(build_module.execute, plan, target, **narrowing)
        finally:
            plan.emit = previous
        return result_of(plan, target, result, verb)

    return execute


def _status_executor(plan: BuildPlan) -> Callable[[Any], Awaitable[StatusResult]]:
    """`status(gate, target)`: the workdir's records read on a worker thread, so the loop stays free."""

    async def execute(args: Any) -> StatusResult:
        return await asyncio.to_thread(status_of, plan.workdir, args.gate.strip(), args.target.strip())

    return execute


def _repair_executor(plan: BuildPlan, sink: Optional[Sink], verb: str, target_of: Callable[[Any], str],
                     narrowing_of: Callable[[Any], dict], stage: str,
                     before: Optional[Callable[[Any], dict]] = None) -> Callable[[Any], Awaitable[BuildResult]]:
    """A repair verb that acts: the stage that repairs the artifact runs, and the request is recorded.

    `before` runs first and returns what else the request carries (the hint kept as a lesson), so a
    repair leaves a record of what was asked for as well as the artifact it produced. The one
    target's artifact is hashed either side of the stage, so the request says whether this call
    moved anything of its own (D142); the record is written in a `finally`, so a stage that raises
    still leaves the row the round's report reads. The result then opens with that target's own
    ruling, read back off the artifact the stage just wrote.
    """
    run_stage = _executor(plan, sink, verb, lambda _a: stage, narrowing_of)

    async def execute(args: Any) -> BuildResult:
        extra = before(args) if before is not None else {}
        target = target_of(args)
        hash_before = repair_module.target_hash(plan.workdir, verb, target)
        try:
            result = await run_stage(args)
        finally:
            repair_module.record_request(
                plan.workdir, verb, target,
                {"arguments": args.model_dump(mode="json"), **extra,
                 **repair_module.change_of(plan.workdir, verb, target, hash_before)},
                round_no=plan.round)
        result.target_ruling = repair_module.target_ruling(plan.workdir, verb, args)
        return result

    return execute


def _keep_hint(plan: BuildPlan) -> Callable[[Any], dict]:
    """The hint a recompile carries, written to the Builder's memory as a lesson for that tool (D87)."""

    def keep(args: Any) -> dict:
        if not args.hint.strip():
            return {}
        repair_module.record_tool_lesson(plan.workdir, args.name, [args.hint.strip()])
        return {"lesson": args.hint.strip()}

    return keep


def repair_verb_tools(plan: BuildPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The six repair verbs a Builder session may call (D135, D136, D138, D155).

    Three act: they record the request and run the stage that repairs the artifact, so the gates
    rule on what came out in the same tool result. Three decide: `repair_refuse_task`,
    `repair_escalate` and `repair_record_finding` come straight from `builder/repair.py`, record
    their request and change no artifact. `repair_rewrite_skill` is not here: a model rewriting its own prompt stays proposed,
    gated and versioned, and no gate accepts an edit yet (the GEPA caution in docs/todo.md).
    """
    recording = {tool.name: tool for tool in
                 repair_module.repair_tools(plan.workdir, sink, round_of=lambda: plan.round)}
    return [
        AgentTool("repair_recompile",
                  "Repair one tool body: compile it again from its recorded calls, with a hint saying "
                  "what was wrong with the last one. The hint is kept as a lesson for that tool. The "
                  "result opens with that tool's own ruling: `cleared the gates`, or `still assisted` "
                  "with the failure the next hint has to answer.",
                  repair_module.RecompileArgs, BuildResult,
                  _repair_executor(plan, sink, "repair_recompile", lambda a: a.name,
                                   lambda a: {"tools": [a.name]}, "compile_tools",
                                   before=_keep_hint(plan)), render=render),
        AgentTool("repair_grow",
                  "Repair the Starting state: grow one table to a row count with synthetic rows (D107). "
                  "The result opens with how many rows that table holds now.",
                  repair_module.GrowRepairArgs, BuildResult,
                  _repair_executor(plan, sink, "repair_grow", lambda a: a.table,
                                   lambda a: {"grow": {**dict(plan.grow or {}), a.table: a.count}},
                                   "starting_state"), render=render),
        AgentTool("repair_intent",
                  "Repair one Task's Intent: write it again with a hint saying what the Task's Runs "
                  "evidence. An Intent with a noun phrase no Run says in those words leaves the Task "
                  "with no Verdict; the hint is how you say what the words should be. The result "
                  "opens with that Task's own ruling: `grounded` with the line it settled on, or "
                  "`still refused` with the reason.",
                  repair_module.IntentRepairArgs, BuildResult,
                  _repair_executor(plan, sink, "repair_intent", lambda a: a.task_id,
                                   lambda a: {"intent_tasks": [a.task_id],
                                              "intent_hints": {a.task_id: a.hint}},
                                   "intent"), render=render),
        recording["repair_refuse_task"],
        recording["repair_escalate"],
        recording["repair_record_finding"],
    ]


def builder_tools(plan: BuildPlan, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The stage tools and `status` over one plan; `sink` is where the stage events go (`harness.emit`)."""
    return [
        AgentTool("status", "The red lights, all of them: a headline, then one block per failing gate "
                  "holding every tool it names with a count and one failure, and every Task id it names "
                  "grouped by the kind of failure, with the repair verb that answers it. Nothing is cut "
                  "from the list. Pass `gate` or `target` to zoom: that one gate's or that one tool's or "
                  "Task's red lights, each in full. Read off the records, never off a model.",
                  StatusArgs, StatusResult, _status_executor(plan), render=render_status),
        AgentTool("build", "Build a target of the Environment: `environment` for everything, or one stage "
                  "or artifact by name; whatever it reads that is stale is rebuilt first.",
                  BuildArgs, BuildResult,
                  _executor(plan, sink, "build", lambda a: a.target, lambda a: {}), render=render),
        AgentTool("recluster", "Cluster the Runs into Tasks again under the fixed configuration.",
                  NoArgs, BuildResult,
                  _executor(plan, sink, "recluster", lambda a: "cluster", lambda a: {}), render=render),
        AgentTool("grow", "Grow one table of the Starting state to a row count with synthetic rows (D107).",
                  GrowArgs, BuildResult,
                  _executor(plan, sink, "grow", lambda a: "starting_state",
                            lambda a: {"grow": {**dict(plan.grow or {}), a.table: a.count}}), render=render),
        AgentTool("compile_tool", "Compile one tool's body again from its recorded calls, through the sandbox gates.",
                  CompileToolArgs, BuildResult,
                  _executor(plan, sink, "compile_tool", lambda a: "compile_tools",
                            lambda a: {"tools": [a.name]}), render=render),
        AgentTool("replay", "Replay one Task's Traces through the built tools (the Reference Runs, D108).",
                  ReplayArgs, BuildResult,
                  _executor(plan, sink, "replay", lambda a: "replay_reference",
                            lambda a: {"replay_tasks": [a.task]}), render=render),
        AgentTool("reroll", "Re-roll one Task with the frontier model inside the built Environment (D112).",
                  RerollArgs, BuildResult,
                  _executor(plan, sink, "reroll", lambda a: "rerolls",
                            lambda a: {"reroll_tasks": [a.task]}), render=render),
    ]
