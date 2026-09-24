"""The autonomous Builder as an extension on the core.

`builder_extension(root)` is the `setup(api)` the harness loads: the base tools over the
Builder's root `env/`, the domain tools, the prompt sections, and the hooks in order. `build`
drives one session: expose and explode, then the model until it answers with no tool call.

The hooks, in order: `refuse_paths` over the gates and the runner (kept, D122), the fence
that keeps writes and edits under `tools/`, `intents/`, `refusals/` or `policy/` (the read
surface is read-only for the model), `gate_writes` over the Builder's root, and the hook
that guards an unacted ruling and regenerates the rendered modules after every write or
edit under `tools/`.

Carried: D86 and D135 (the spend ceiling stops the session), D122 (no agent writes gates
or runner), D124 (an unacted ruling stays in front of the model: the write result carrying
a failed ruling is protected until the next write of the same path, as
`kullback/builder/extension.py:gate_rulings_hook` does), D155 (the recording is the
standard), D171 (per-call rows ride the rulings), D224 (synthetic rows never in a trusted
Task), learnings section 7.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.agent.base_tools import DEFAULT_ALLOWLIST, TOOL_NAMES, register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig
from kullback.agent.events import ToolExecutionEnd
from kullback.agent.extensions import ExtensionAPI, load_extensions, refuse_paths
from kullback.agent.harness import AgentHarness
from kullback.agent.session.store import SessionStore
from kullback.ai.cache import ttl_for
from kullback.ai.provider import ModelConfig
from kullback.builder import env_files
from kullback.builder import prompt as prompt_mod
from kullback.builder.domain_tools import domain_tools, opening_for, status_of
from kullback.builder.world_tools import ingest_files
from kullback.gates import names_protected_path
from kullback.gates.hook import PATH_KEYS, gate_writes
from kullback.runner import budget

#: The directories under env/ the model may write; everything else there is read-only for it.
WRITABLE_PREFIXES = ("tools/", "intents/", "refusals/", "policy/")
WRITABLE_NAMES = ("tools", "intents", "refusals", "policy")

#: The base tools the Builder gets: every one but web_search, withdrawn while search is unconfigured.
BUILDER_BASE_TOOLS = tuple(name for name in TOOL_NAMES if name != "web_search")

#: The tool_call hook (D122): the rule and the walk are the gates' own, the refusal is the core's.
no_agent_writes_gates_or_runner = refuse_paths(
    names_protected_path, "a path under kullback/gates or kullback/runner, which no agent may write (D122)",
    "no_agent_writes_gates_or_runner")

OPENING = ("Your root is the workdir's env/ directory and every path is relative to it: "
           "tools/, calls/, tasks/, never env/ itself. Derive the world when the traces are stored, edit tool "
           "bodies where the rulings point, replay and run the Tasks, examine when ready. "
           "Answer with no tool call when every Task is trusted or refused with a reason, "
           "when the spend ceiling is reached, or when you state why you cannot go further.")


def opening(workdir: Any) -> str:
    """The opening message for this workdir: its stage and next step first, then OPENING."""
    return opening_for(workdir, OPENING)


@dataclass
class BuilderRoot:
    """What the Builder extension is bound to: the workdir, its env root, examine_fn, ceiling,
    and the models examine hands the judges, the probe and the re-rolls."""

    workdir: Path
    env_root: Optional[Path] = None
    examine_fn: Optional[Callable[[Any, Any], Any]] = None
    ceiling_usd: Optional[float] = None
    judge_model: Any = None
    probe_model: Any = None
    reroll_model: Any = None

    @property
    def env(self) -> Path:
        """The Builder's root: everything it reads and writes."""
        return Path(self.env_root) if self.env_root is not None else Path(self.workdir) / "env"


@dataclass
class _GuardPlan:
    """The two fields `ceiling_guard` reads: the workdir ledger and the ceiling."""

    workdir: Path
    ceiling: Any = None

CEILING_STAGE = "builder_session"

#: How many follow-ups the harness sends when the model stops with Tasks open (F20).
CONTINUATIONS = 3


def spent_in(workdir: Any) -> float:
    """What this workdir has paid for so far, off the one ledger budget.py writes."""
    return float(budget.load_totals(workdir)["total"].get("usd") or 0.0)


def ceiling_guard(harness: AgentHarness, plan: _GuardPlan) -> tuple[Callable[[], None], dict]:
    """Cancel the run the moment the workdir's spend reaches the build's ceiling (D86, D135).

    The ceiling is the hard stop of a model-driven session: a stage that crosses it raises inside
    its own tool and the pipeline records the stop, but nothing stopped the model from calling the
    next tool, and with no turn cap that is an unbounded run. The guard reads the ledger after every
    tool call and cancels, so the tool that reached the ceiling is the last one. The dict it returns
    is empty until then and afterwards is the ceiling's own stop report, which is what the caller
    hands to a report (D86: stop where you are and say what it cost).
    """
    stop: dict = {}
    ceiling = plan.ceiling
    if ceiling is None:
        return (lambda: None), stop

    def watch(event: Any) -> None:
        if stop or not isinstance(event, ToolExecutionEnd):
            return
        spent = spent_in(plan.workdir)
        if spent < ceiling.usd:
            return
        ceiling.spent = max(ceiling.spent, spent)
        stop.update(ceiling.report(CEILING_STAGE, event.tool_name))
        harness.cancel()

    return harness.subscribe(watch), stop


def _written_path(call: Any) -> Optional[str]:
    arguments = getattr(call, "arguments", None) or {}
    if not isinstance(arguments, dict):
        return None
    for key in PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def write_fence(call: Any) -> None:
    """Refuse a write or edit outside the four writable directories (the read surface is read-only)."""
    if getattr(call, "name", None) not in ("write", "edit"):
        return None
    path = _written_path(call)
    if path is None:
        return None
    text = path.replace("\\", "/").lstrip("/")
    if text in WRITABLE_NAMES or any(text.startswith(prefix) for prefix in WRITABLE_PREFIXES):
        return None
    raise PermissionError(
        f"the path {path!r} is outside tools/, intents/, refusals/ and policy/, "
        "which are the only directories this agent may write")


write_fence.hook_name = "builder_write_fence"  # type: ignore[attr-defined]


def refusal_waits_for_note(workdir: Any) -> Callable[[Any], None]:
    """Refuse a write or edit of refusals/<task>.json while the Task's note has no Examiner ruling.

    An open note blocks that one Task's refusal and nothing else: every other write goes through.
    """
    from kullback.examiner.domain_tools import open_note

    def refuse(call: Any) -> None:
        if getattr(call, "name", None) not in ("write", "edit"):
            return None
        path = _written_path(call)
        text = (path or "").replace("\\", "/").lstrip("/")
        if not text.startswith("refusals/"):
            return None
        task_id = Path(text).stem
        if open_note(workdir, task_id) is None:
            return None
        raise PermissionError(
            f"task {task_id} has an open note the Examiner has not ruled on; examine the Task so it "
            "rules, then refuse it if the ruling leaves it unverifiable")

    refuse.hook_name = "builder_refusal_waits_for_note"  # type: ignore[attr-defined]
    return refuse


def _rel_in_env(env: Path, written_path: str) -> Optional[str]:
    try:
        return str(Path(env, written_path).resolve().relative_to(env.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return None


def regenerate_and_guard(api: ExtensionAPI, root: BuilderRoot) -> Callable[[Any, Any], Any]:
    """Guard an unacted ruling and regenerate the rendered modules (D124).

    A write result carrying a failed ruling is protected until the next write of the same
    path, which releases it; after every write or edit under `tools/` the bodies and the
    rendered modules are regenerated from the files the model wrote.
    """
    workdir = Path(root.workdir)
    env = root.env
    guards: dict[str, str] = {}

    def hook(call: Any, result: Any) -> Any:
        if getattr(call, "name", None) not in ("write", "edit"):
            return None
        if result is None or getattr(result, "is_error", False):
            return None
        path = _written_path(call)
        rel = _rel_in_env(env, path) if path is not None else None
        if rel is None:
            return None
        entry = api.entry_id_for(call.id)
        if entry is not None:
            earlier = guards.pop(rel, None)
            if earlier is not None and earlier != entry:
                api.unprotect([earlier])
            rulings = (getattr(result, "details", None) or {}).get("rulings") or []
            failed = sorted({str(ruling.get("name")) for ruling in rulings
                             if isinstance(ruling, dict) and not ruling.get("accepted")})
            if failed:
                guards[rel] = entry
                api.protect([entry], f"unacted ruling: {', '.join(failed)} on {rel}")
        if rel == "tools" or rel.startswith("tools/"):
            env_files.regenerate(workdir, env)
        return None

    hook.hook_name = "builder_regenerate_and_guard"  # type: ignore[attr-defined]
    return hook


def builder_extension(root: BuilderRoot) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools, domain tools, prompt sections, the four hooks."""

    def setup(api: ExtensionAPI) -> None:
        env = root.env
        env.mkdir(parents=True, exist_ok=True)
        register_base_tools(api, env, allowlist=DEFAULT_ALLOWLIST, only=BUILDER_BASE_TOOLS)
        model = getattr(getattr(api, "harness", None), "model", None)
        for tool in domain_tools(workdir=root.workdir, model=model, examine_fn=root.examine_fn,
                                 judge_model=root.judge_model, probe_model=root.probe_model,
                                 reroll_model=root.reroll_model, env=env):
            api.register_tool(tool)
        for name, text in prompt_mod.sections():
            api.add_prompt_section(f"builder_{name}", text)
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_call(write_fence)
        api.tool_call(refusal_waits_for_note(root.workdir))
        api.tool_result(gate_writes(root=env, workdir=root.workdir,
                                    execute=env_files.executor(root.workdir, env)))
        api.tool_result(regenerate_and_guard(api, root))

    return setup

def _collect(aiter: Any) -> list:
    async def go() -> list:
        return [event async for event in aiter]

    return asyncio.run(go())


def build(workdir: Any, model: Any, *, files: Optional[list] = None,
          examine_fn: Optional[Callable[[Any, Any], Any]] = None,
          ceiling_usd: Optional[float] = None, session_path: Any = None,
          subscribers: Iterable[Callable[[Any], Any]] = (),
          max_turns: Optional[int] = None, judge_model: Any = None, probe_model: Any = None,
          reroll_model: Any = None,
          on_harness: Optional[Callable[[AgentHarness], Any]] = None) -> dict:
    """Run one Builder session over the workdir and say how it stopped.

    `files` are customer files to ingest first, as paths under the workdir. `session_path`
    defaults to workdir/sessions/builder.jsonl. The dict carries the status tool's rows,
    the trusted, refused and open counts, the spend, the turns, how it stopped (one of:
    no tool call, ceiling, cancelled, max turns, error) and the last assistant line, or the
    provider's error message when the last turn ended in an error. A stop with no tool call
    while Tasks are open and the spend is under the ceiling draws a follow-up (`continuation`),
    at most CONTINUATIONS of them; "continued" counts those sent. `judge_model`, `probe_model` and `reroll_model` reach examine;
    each is `model` when not named. `on_harness` is handed the live harness once it exists and
    before the first model turn, so a caller on another thread can steer, follow up, compact or
    cancel it while this call runs.
    """
    root = Path(workdir)
    env = root / "env"
    env.mkdir(parents=True, exist_ok=True)
    if files:
        ingest_files([(root / name if not Path(name).is_absolute() else Path(name))
                      for name in files], root)
    env_files.expose(root)
    env_files.explode(root)
    if (root / env_files.SIGS_FILE).is_file() and (root / env_files.SCHEMA_FILE).is_file():
        env_files.regenerate(root)
    session = SessionStore.load(Path(session_path) if session_path is not None
                                else root / "sessions" / "builder.jsonl")
    # The session waits on its own long tools (examine, run) between turns, past the five-minute
    # TTL, so its cache points are written to live an hour (cache.STAGE_CACHE_TTL says why).
    harness = AgentHarness(
        model=model, session=session, config=ModelConfig(cache_ttl=ttl_for(CEILING_STAGE)),
        context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
        bus=Bus(root / "bus.jsonl", agent="builder"), max_turns=max_turns)
    load_extensions(harness, [builder_extension(
        BuilderRoot(workdir=root, env_root=env, examine_fn=examine_fn, ceiling_usd=ceiling_usd,
                    judge_model=judge_model, probe_model=probe_model, reroll_model=reroll_model))])
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    ceiling = budget.Ceiling.from_totals(root, ceiling_usd) if ceiling_usd is not None else None
    harness.subscribe(budget.subscriber(root, CEILING_STAGE, getattr(model, "name", None), ceiling))
    guard_stop: dict = {}
    if ceiling is not None:
        _, guard_stop = ceiling_guard(harness, _GuardPlan(workdir=root, ceiling=ceiling))
    if on_harness is not None:
        on_harness(harness)
    turns: list = []
    continued = 0
    message = opening(root)
    while True:
        events = _collect(harness.prompt(message))
        run_turns = [event for event in events if event.type == "turn_end"]
        turns += run_turns
        last_message = run_turns[-1].message if run_turns else None
        stopped = _stop_of(last_message, run_turns, guard_stop, max_turns)
        status = status_of(root)
        # A follow-up answered with no tool call is the model stating why it cannot go further.
        answered_with_work = continued == 0 or len(run_turns) > 1
        follow = (continuation(status, spent_in(root), ceiling_usd)
                  if stopped == "no tool call" and answered_with_work else None)
        if follow is None or continued >= CONTINUATIONS:
            break
        continued += 1
        message = follow
    last_line = str(getattr(last_message, "content", None) or "")
    if stopped == "error":
        last_line = str(getattr(last_message, "error_message", None) or last_line)
    states = [row["state"] for row in status["tasks"]]
    return {"status": status, "trusted": states.count("trusted"), "refused": states.count("refused"),
            "open": states.count("open"), "spend": spent_in(root), "turns": len(turns),
            "stopped": stopped, "last_line": last_line, "continued": continued}


def _stop_of(last_message: Any, turns: list, guard_stop: dict, max_turns: Optional[int]) -> str:
    """How one harness run stopped: ceiling, error, max turns, cancelled or no tool call."""
    if guard_stop:
        return "ceiling"
    if getattr(last_message, "stop_reason", None) == "error":
        return "error"
    if max_turns is not None and len(turns) >= max_turns:
        return "max turns"
    if last_message is not None and list(getattr(last_message, "tool_calls", None) or []):
        return "cancelled"
    return "no tool call"


def continuation(status: dict, spent: float, ceiling_usd: Optional[float]) -> Optional[str]:
    """The follow-up the harness sends when the model stops early, or None when it may stop (F20).

    The stop rule lives in the prompt; this holds the model to it. While a Task is open and the
    spend is under the ceiling (or there is none), the answer is the counts and the rule, built
    from the status alone.
    """
    states = [row["state"] for row in status.get("tasks") or []]
    open_count = states.count("open")
    if not open_count or (ceiling_usd is not None and spent >= ceiling_usd):
        return None
    budget_text = (f"{spent:.4f} USD spent of the {ceiling_usd:.4f} USD ceiling" if ceiling_usd is not None
                   else f"{spent:.4f} USD spent, no ceiling")
    return (f"{open_count} Tasks are open, {states.count('trusted')} trusted, "
            f"{states.count('refused')} refused; {budget_text}. The stop rule is not met. "
            "Replay every Task, examine the confirmed ones, run them, "
            "or state in one sentence why you cannot go further.")

__all__ = ["BUILDER_BASE_TOOLS", "BuilderRoot", "CEILING_STAGE", "CONTINUATIONS", "OPENING", "WRITABLE_NAMES", "WRITABLE_PREFIXES", "build",
           "builder_extension", "ceiling_guard", "continuation", "no_agent_writes_gates_or_runner", "opening", "refusal_waits_for_note", "regenerate_and_guard",
           "spent_in", "write_fence"]
