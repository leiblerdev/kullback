"""The training agent as an extension on the core, run over a workdir.

`trainer_extension(ctx)` is a `setup(api)` the harness loads, the way the
other agents load theirs. `train` builds the context off the workdir files,
plays one session on the opening message, and answers what ran and what was
filed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.agent.base_tools import register_base_tools
from kullback.agent.bus import Bus
from kullback.agent.context import ContextConfig, prompt_block
from kullback.agent.extensions import ExtensionAPI, load_extensions
from kullback.agent.harness import AgentHarness
from kullback.agent.skills import skills_from_dir
from kullback.runner import budget
from kullback.runner.world.environment import BuiltEnvironment
from kullback.trainer import act as act_mod
from kullback.trainer import prompt as prompt_mod
from kullback.trainer import remote as remote_mod
from kullback.trainer.tools import TrainerContext, trainer_tools

TRAINER_BASE_ONLY = ("read", "write", "edit", "grep", "find", "ls")
TRAIN_SKILL_NAME = "train"
TRAIN_OPENING = "Train. Start with check_env and gpu_status."
# The harness's own words when a session stops on its cap (agent/loop.py).
CAP_STOP = "stopped after max_turns="


def _folder_skills(base: Path, found: dict[str, str]) -> dict[str, str]:
    """Add each folder holding SKILL.md, by folder name, where the name is still free."""
    if base.is_dir():
        for path in sorted(base.iterdir()):
            if path.is_dir() and path.name not in found:
                skill = path / "SKILL.md"
                if skill.is_file():
                    found[path.name] = skill.read_text(encoding="utf-8")
    return found


def _extra_skills(workdir: Any) -> dict[str, str]:
    """training/skills plus bundled skills, by name; the workdir wins on a clash."""
    base = Path(workdir) / "training" / "skills"
    found = _folder_skills(base, skills_from_dir(base))
    return _folder_skills(Path(__file__).parent / "skills", found)


def trainer_extension(ctx: TrainerContext) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: base tools over training/, domain tools, prompt, skills."""

    def setup(api: ExtensionAPI) -> None:
        register_base_tools(api, ctx.workdir / "training", only=TRAINER_BASE_ONLY)
        for tool in trainer_tools(ctx):
            api.register_tool(tool)
        for name, text in prompt_mod.sections():
            api.add_prompt_section(f"trainer_{name}", prompt_block(name, text))
        api.add_prompt_section("skills", api.context.skills_section())
        api.catalog_skill(TRAIN_SKILL_NAME, prompt_mod.TRAIN_SKILL, loaded=True)
        for name, text in sorted(_extra_skills(ctx.workdir).items()):
            api.catalog_skill(name, text, loaded=True)

    return setup


def _read_findings(workdir: Any) -> list[dict]:
    """The filed notes, oldest first; nothing filed reads as empty."""
    path = Path(workdir) / "training" / "findings.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            out.append(body)
    return out


def train(workdir: Any, model: Any, *, policy_model: Any = None,
          policy_model_id: Optional[str] = None, train_run_id: str,
          max_turns: int = 200, subscribers: Iterable[Callable] = (),
          opening: Optional[str] = None) -> dict:
    """Run one training-agent session over the workdir and answer turns, stop and findings."""
    root = Path(workdir)
    try:
        cfg = remote_mod.load_config(root)
    except FileNotFoundError:
        cfg = None
    env = BuiltEnvironment(root) if (root / "environment.json").is_file() else None
    policy = act_mod.policy_from_model(policy_model) if policy_model is not None else None
    ctx = TrainerContext(
        workdir=root,
        env=env,
        cfg=cfg,
        bus=Bus(root / "bus.jsonl", agent="trainer"),
        policy=policy,
        policy_model_id=(policy_model_id or getattr(policy_model, "name", None) or ""),
        train_run_id=train_run_id,
    )
    harness = AgentHarness(model=model, max_turns=max_turns,
                           context=ContextConfig(window=budget.window_for(getattr(model, "name", None))),
                           bus=ctx.bus)
    for subscriber in subscribers:
        harness.subscribe(subscriber)
    harness.subscribe(budget.subscriber(root, "trainer", getattr(model, "name", None)))
    load_extensions(harness, [trainer_extension(ctx)])
    turns, capped = _run_session(harness, opening or TRAIN_OPENING)
    return {"turns": turns, "stopped": "cap" if capped else "done",
            "findings": _read_findings(root)}


def _run_session(harness: AgentHarness, opening_message: str) -> tuple[int, bool]:
    """Play the session on its opening message; answer the turn count and the cap."""
    turns = 0
    capped = False

    async def go() -> None:
        nonlocal turns, capped
        async for event in harness.prompt(opening_message):
            if getattr(event, "type", None) == "turn_end":
                turns += 1
                if str(getattr(event.message, "error_message", None) or "").startswith(CAP_STOP):
                    capped = True

    asyncio.run(go())
    return turns, capped
