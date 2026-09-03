"""Phase 6 repair verbs: the five ways a round fixes what the gates refused (D130).

Verbs: `recompile(name)`, `grow(table, count)`, `rewrite_skill(name, content)`,
`refuse_task(task_id, reason)`, `escalate(task_id, queue)`. Each verb records its
request as JSONL under `workdir/repairs/` and returns a short `RepairResult`; the
next round's driver picks the requests up. Nothing here calls a model or edits
`kullback/gates` or `kullback/runner` (D122): the ratchet and the lesson are code
over artifacts the pipeline already wrote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, ToolResult
from kullback.builder import memory as memory_mod

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


class GrowRepairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(description="The table of the Starting state to grow.")
    count: int = Field(ge=1, description="How many rows the table should hold after growing (D107).")


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


# --- request log ------------------------------------------------------------


def _repairs_dir(workdir: Any) -> Path:
    path = Path(workdir) / "repairs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(workdir: Any, verb: str, target: str, body: dict) -> Path:
    path = _repairs_dir(workdir) / f"{verb}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"verb": verb, "target": target, "at": time.time(), **body},
                            sort_keys=True) + "\n")
    return path


def _render(result: RepairResult) -> str:
    return f"{result.verb} {result.target}: {result.status}" + (f" ({result.detail})" if result.detail else "")


def _executor(workdir: Any, verb: str, target_of: Any) -> Any:
    async def execute(args: Any) -> RepairResult:
        target = target_of(args)
        extra: dict[str, Any] = {}
        if verb == "rewrite_skill":
            from kullback.builder import skills as skills_mod
            written = skills_mod.write_skill(workdir, args.name, args.content)
            extra = {"skill_hash": written["hash"]}
        path = _record(workdir, verb, target, {"arguments": args.model_dump(mode="json"), **extra})
        return RepairResult(verb=verb, target=target, path=str(path),
                            detail=f"request in {path.name}")
    return execute


def repair_tools(workdir: Any, sink: Optional[Sink] = None) -> list[AgentTool]:
    """The five repair verbs over one workdir; `sink` is accepted for symmetry with builder_tools."""
    return [
        AgentTool("recompile", "Compile one tool's body again from its recorded calls.",
                  RecompileArgs, RepairResult,
                  _executor(workdir, "recompile", lambda a: a.name), render=_render),
        AgentTool("grow", "Grow one table of the Starting state with synthetic rows (D107).",
                  GrowRepairArgs, RepairResult,
                  _executor(workdir, "grow", lambda a: a.table), render=_render),
        AgentTool("rewrite_skill", "Rewrite one Builder skill; the edit is a memory-tree node with its content hash.",
                  RewriteSkillArgs, RepairResult,
                  _executor(workdir, "rewrite_skill", lambda a: a.name), render=_render),
        AgentTool("refuse_task", "Refuse a Task: no Verifier, no training signal from it.",
                  RefuseTaskArgs, RepairResult,
                  _executor(workdir, "refuse_task", lambda a: a.task_id), render=_render),
        AgentTool("escalate", "Escalate a Task to a person on a named queue.",
                  EscalateArgs, RepairResult,
                  _executor(workdir, "escalate", lambda a: a.task_id), render=_render),
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
        if name in gate_passed and gate_passed[name] is False and name in merged:
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
        content = f"{result.content}\nratchet: kept prior passing body for {tool}"
        return ToolResult(content=content, details=details, is_error=False)

    on_result.hook_name = "ratchet"  # type: ignore[attr-defined]
    return on_result


# --- lesson --------------------------------------------------------------------

def record_tool_lesson(workdir: Any, tool: str, failures: list[str]) -> Path:
    """Write one gate-failure sequence to the Builder memory for this workdir."""
    return memory_mod.record_lesson(workdir, tool, list(failures))


def lesson_for_tool(workdir: Any, tool: str) -> str:
    """The failure sequences to inject into the next attempt's prompt for this tool."""
    return memory_mod.lesson_for(workdir, tool)
