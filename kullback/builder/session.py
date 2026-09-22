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

from kullback.agent.base_tools import DEFAULT_ALLOWLIST, register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig
from kullback.agent.extensions import ExtensionAPI, load_extensions, refuse_paths
from kullback.agent.harness import AgentHarness
from kullback.agent.session.store import SessionStore
from kullback.builder import env_files
from kullback.builder import prompt as prompt_mod
from kullback.builder.domain_tools import domain_tools
from kullback.builder.world_tools import ingest_files
from kullback.gates import names_protected_path
from kullback.gates.hook import PATH_KEYS, gate_writes
from kullback.runner import budget

#: The directories under env/ the model may write; everything else there is read-only for it.
WRITABLE_PREFIXES = ("tools/", "intents/", "refusals/", "policy/")
WRITABLE_NAMES = ("tools", "intents", "refusals", "policy")

#: The tool_call hook (D122): the rule and the walk are the gates' own, the refusal is the core's.
no_agent_writes_gates_or_runner = refuse_paths(
    names_protected_path, "a path under kullback/gates or kullback/runner, which no agent may write (D122)",
    "no_agent_writes_gates_or_runner")

OPENING = ("Begin in your root env/. Derive the world when the traces are stored, edit tool "
           "bodies where the rulings point, replay and run the Tasks, examine when ready. "
           "Answer with no tool call when every Task is trusted or refused with a reason, "
           "when the spend ceiling is reached, or when you state why you cannot go further.")


@dataclass
class BuilderRoot:
    """What the Builder extension is bound to: the workdir, its env root, examine_fn, ceiling."""

    workdir: Path
    env_root: Optional[Path] = None
    examine_fn: Optional[Callable[[Any, Any], Any]] = None
    ceiling: Optional[float] = None

    @property
    def env(self) -> Path:
        """The Builder's root: everything it reads and writes."""
        return Path(self.env_root) if self.env_root is not None else Path(self.workdir) / "env"


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
        register_base_tools(api, env, allowlist=DEFAULT_ALLOWLIST)
        model = getattr(getattr(api, "harness", None), "model", None)
        for tool in domain_tools(workdir=root.workdir, model=model, examine_fn=root.examine_fn):
            api.register_tool(tool)
        for name, text in prompt_mod.sections():
            api.add_prompt_section(f"builder_{name}", text)
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_call(write_fence)
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
          ceiling: Optional[float] = None, max_turns: Optional[int] = None,
          subscribers: Iterable[Callable[[Any], Any]] = ()) -> dict:
    """Run one Builder session over the workdir and say how it stopped.

    `files` are customer files to ingest first, as paths under the workdir. The session ends
    when the model answers with no tool call, when the spend ceiling is reached (D86, D135),
    or when the harness stops another way; the dict names which of the three stopped it.
    """
    root = Path(workdir)
    env = root / "env"
    env.mkdir(parents=True, exist_ok=True)
    if files:
        ingest_files([(root / name if not Path(name).is_absolute() else Path(name))
                      for name in files], root)
    env_files.expose(root)
    env_files.explode(root)
    session = SessionStore.load(root / "sessions" / "builder.jsonl")
    harness = AgentHarness(
        model=model, session=session,
        context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
        bus=Bus(root / "bus.jsonl", agent="builder"), max_turns=max_turns)
    load_extensions(harness, [builder_extension(
        BuilderRoot(workdir=root, env_root=env, examine_fn=examine_fn, ceiling=ceiling))])
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    state = {"ceiling_reached": False}
    if ceiling is not None:
        def watch(event: Any) -> None:
            try:
                spent = float(budget.load_totals(root)["total"].get("usd") or 0.0)
            except (OSError, ValueError):
                return
            if spent >= ceiling and not state["ceiling_reached"]:
                state["ceiling_reached"] = True
                harness.cancel()

        harness.subscribe(watch)
    events = _collect(harness.prompt(OPENING))
    if state["ceiling_reached"]:
        stopped = f"the spend ceiling of {ceiling} USD is reached"
    else:
        stopped = "the model answered with no tool call"
    return {"stopped": stopped, "events": len(events),
            "ceiling_reached": state["ceiling_reached"]}


__all__ = ["BuilderRoot", "OPENING", "WRITABLE_NAMES", "WRITABLE_PREFIXES", "build", "builder_extension",
           "no_agent_writes_gates_or_runner", "regenerate_and_guard", "write_fence"]
