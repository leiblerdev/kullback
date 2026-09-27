"""The training agent's tools: thin calls over the trainer backends.

Each tool takes validated arguments in and answers a plain dict out. A tool
whose backend needs a remote config, a built Environment, a policy or the bus
and has none answers with an error saying what is missing and how to set it,
instead of raising.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback import curriculum
from kullback.agent.tools import AgentTool, NoArgs
from kullback.trainer import act as act_mod
from kullback.trainer import observe as observe_mod
from kullback.trainer import remote as remote_mod


@dataclass
class TrainerContext:
    """What the tools close over: the workdir, the backends, and this run's ids."""

    workdir: Path
    env: Any = None
    cfg: Any = None
    bus: Any = None
    policy: Any = None
    policy_model_id: str = ""
    train_run_id: str = ""


class DictResult(BaseModel):
    """Any plain dict a backend answers."""

    model_config = ConfigDict(extra="allow")


def _missing_cfg() -> dict:
    return {"error": "no remote config: create training/remote.json with host, root, python, timeout_s"}


def _missing_env(ctx: TrainerContext) -> dict:
    return {"error": f"no built Environment under {ctx.workdir}: build one there first"}


def _missing_policy() -> dict:
    return {"error": "no policy under training: pass --policy-model with --policy-base-url"}

def _resolve_run_id(ctx: TrainerContext, given: Optional[str]):
    """The asked run id or the session run, or an error dict where it is bad."""
    run_id = given or ctx.train_run_id
    if remote_mod.RUN_ID_RE.fullmatch(run_id) is None:
        return {"error": f"bad run_id {run_id!r}"}
    return run_id


def _sync_error(ctx: TrainerContext, run_id: str) -> Optional[str]:
    """Sync one run and answer the failure text, or None where the sync held."""
    synced = remote_mod.sync(ctx.cfg, run_id, ctx.workdir)
    if isinstance(synced, dict):
        error = synced.get("error")
        return str(error) if error else None
    return None


RUN_FILES = ("solve_rates.jsonl", "metrics.jsonl", "state.json", "run.log")
RUN_FILE_LINES = 200


class TrainingStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to read; the session run by default.")
    tail: int = Field(default=40, description="How many trailing log lines to read.")


class MetricsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to read; the session run by default.")
    window: int = Field(default=3, description="How many recent checkpoints to fold.")


class ReadRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to read; the session run by default.")
    name: str = Field(description="One of solve_rates.jsonl, metrics.jsonl, state.json, run.log.")


class SmokeTestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=2, description="How many sample Tasks to play.")
    k: int = Field(default=2, description="Plays per sampled Task.")
    key: str = Field(default="smoke", description="Sample draw key.")
    checkpoint: str = Field(default="smoke", description="Checkpoint name the records carry.")


class PushArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subdir: str = Field(default="scripts", description="Training subdir to copy to the GPU machine.")


class StartRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to start; the session run by default.")
    script: str = Field(description="Script path under the GPU root, e.g. scripts/train.py.")
    args: str = Field(default="", description="Extra arguments for the script.")


class StopRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to stop; the session run by default.")


class SyncArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(default=None, description="Run to sync; the session run by default.")


class AskTasksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: curriculum.Level = Field(description="Level to ask for: easy, medium or hard.")
    count: int = Field(ge=1, le=60, description="How many Tasks to ask for.")
    kind: Optional[str] = Field(default=None, description="Kind to ask for, if any.")
    step: int = Field(description="Training step the ask is filed at.")
    checkpoint: str = Field(description="Checkpoint the ask is filed at.")
    evidence: dict = Field(default_factory=dict, description="Numbers supporting the ask.")


class FindingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Kind of note to file.")
    text: str = Field(description="What was found, in one or two sentences.")


class ShellArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="Shell command to run on the GPU machine.")


def _observe_tools(ctx: TrainerContext) -> list[AgentTool]:
    """The read only tools of the training agent, bound to one context."""
    tools: list[AgentTool] = []

    async def training_status(args: TrainingStatusArgs) -> dict:
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        sync_error = _sync_error(ctx, run_id) if ctx.cfg is not None else None
        status = observe_mod.training_status(ctx.workdir, run_id, args.tail)
        if sync_error is not None:
            status = {**status, "sync_error": sync_error}
        return status

    tools.append(AgentTool("training_status", "State, log tail and error lines of one run, synced first.",
                           TrainingStatusArgs, DictResult, training_status))

    async def metrics(args: MetricsArgs) -> dict:
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        sync_error = _sync_error(ctx, run_id) if ctx.cfg is not None else None
        bands = observe_mod.metrics(ctx.workdir, run_id, args.window)
        if sync_error is not None:
            bands = {**bands, "sync_error": sync_error}
        return bands

    tools.append(AgentTool("metrics", "Bands, decision and pass estimates over one run, synced first.",
                           MetricsArgs, DictResult, metrics))

    async def task_pool(args: NoArgs) -> dict:
        if ctx.env is None:
            return _missing_env(ctx)
        return observe_mod.task_pool(ctx.workdir)

    tools.append(AgentTool("task_pool", "Totals of the built Environment and its unseen Task ids.",
                           NoArgs, DictResult, task_pool))

    async def synth_status(args: NoArgs) -> dict:
        return observe_mod.synth_status(ctx.workdir / "bus.jsonl", ctx.train_run_id)

    tools.append(AgentTool("synth_status", "Asked, answered and open Task requests of this run.",
                           NoArgs, DictResult, synth_status))

    async def gpu_status(args: NoArgs) -> dict:
        if ctx.cfg is None:
            return _missing_cfg()
        return remote_mod.gpu_status(ctx.cfg, workdir=ctx.workdir)

    tools.append(AgentTool("gpu_status", "GPUs of the GPU machine, empty with an error where none answer.",
                           NoArgs, DictResult, gpu_status))

    async def read_run(args: ReadRunArgs) -> dict:
        if args.name not in RUN_FILES:
            return {"error": f"unknown run file {args.name!r}: one of {', '.join(RUN_FILES)}"}
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        path = ctx.workdir / "training" / run_id / args.name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return {"error": f"cannot read {path}: {exc}"}
        return {"run_id": run_id, "name": args.name, "lines": lines[-RUN_FILE_LINES:],
                "total_lines": len(lines)}

    tools.append(AgentTool("read_run", "Last 200 lines of one run file: rates, metrics, state or log.",
                           ReadRunArgs, DictResult, read_run))

    async def runs(args: NoArgs) -> dict:
        return {"runs": observe_mod.runs(ctx.workdir)}

    tools.append(AgentTool("runs", "Run ids under training/, sorted.", NoArgs, DictResult, runs))

    return tools


def _test_tools(ctx: TrainerContext) -> list[AgentTool]:
    """The check tools of the training agent, bound to one context."""
    tools: list[AgentTool] = []

    async def check_env(args: NoArgs) -> dict:
        if ctx.env is None:
            return _missing_env(ctx)
        return act_mod.check_env(ctx.env)

    tools.append(AgentTool("check_env", "Load counts plus a do-nothing play that must not pass.",
                           NoArgs, DictResult, check_env))

    async def smoke_test(args: SmokeTestArgs) -> dict:
        if ctx.env is None:
            return _missing_env(ctx)
        if ctx.policy is None:
            return _missing_policy()
        return act_mod.smoke_test(ctx.env, ctx.policy, n=args.n, k=args.k,
                                  outdir=ctx.workdir / "training" / "smoke", key=args.key,
                                  model_id=ctx.policy_model_id, checkpoint=args.checkpoint,
                                  train_run_id=ctx.train_run_id)

    tools.append(AgentTool("smoke_test", "Play a small sample under the policy and summarise it.",
                           SmokeTestArgs, DictResult, smoke_test))

    return tools


def _act_tools(ctx: TrainerContext) -> list[AgentTool]:
    """The write tools of the training agent, bound to one context."""
    tools: list[AgentTool] = []

    async def push(args: PushArgs) -> dict:
        if ctx.cfg is None:
            return _missing_cfg()
        return remote_mod.push(ctx.cfg, ctx.workdir, args.subdir)

    tools.append(AgentTool("push", "Copy training scripts to the GPU machine.", PushArgs, DictResult, push))

    async def start_run(args: StartRunArgs) -> dict:
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        if ctx.cfg is None:
            return _missing_cfg()
        pushed = remote_mod.push(ctx.cfg, ctx.workdir)
        if pushed.get("error"):
            return {"started": False, "reason": f"push failed: {pushed['error']}", "push": pushed}
        started = remote_mod.start_run(ctx.cfg, run_id, args.script, args=args.args,
                                       workdir=ctx.workdir)
        return {"run_id": run_id, "script": args.script, "push": pushed, **started}

    tools.append(AgentTool("start_run", "Push scripts, then launch one script on the GPU machine.",
                           StartRunArgs, DictResult, start_run))

    async def stop_run(args: StopRunArgs) -> dict:
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        if ctx.cfg is None:
            return _missing_cfg()
        return remote_mod.stop_run(ctx.cfg, run_id, workdir=ctx.workdir)

    tools.append(AgentTool("stop_run", "Stop one running run with TERM then KILL.", StopRunArgs,
                           DictResult, stop_run))

    async def sync(args: SyncArgs) -> dict:
        run_id = _resolve_run_id(ctx, args.run_id)
        if isinstance(run_id, dict):
            return run_id
        if ctx.cfg is None:
            return _missing_cfg()
        return remote_mod.sync(ctx.cfg, run_id, ctx.workdir)

    tools.append(AgentTool("sync", "Copy one run back from the GPU machine.", SyncArgs, DictResult, sync))

    async def ask_tasks(args: AskTasksArgs) -> dict:
        if ctx.bus is None:
            return {"error": "no bus: the session opens one per run"}
        return act_mod.ask_tasks(ctx.bus, train_run_id=ctx.train_run_id,
                                 model_id=ctx.policy_model_id, checkpoint=args.checkpoint,
                                 step=args.step, level=args.level, count=args.count,
                                 kind=args.kind, evidence=args.evidence)

    tools.append(AgentTool("ask_tasks", "Ask for new Tasks of one level, once per open request.",
                           AskTasksArgs, DictResult, ask_tasks))

    async def finding(args: FindingArgs) -> dict:
        folder = ctx.workdir / "training"
        folder.mkdir(parents=True, exist_ok=True)
        line = {"time": time.time(), "run_id": ctx.train_run_id,
                "kind": args.kind, "text": args.text}
        with open(folder / "findings.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return {"written": True, "kind": args.kind, "run_id": ctx.train_run_id}

    tools.append(AgentTool("finding", "File one note to training/findings.jsonl.", FindingArgs,
                           DictResult, finding))

    async def shell(args: ShellArgs) -> dict:
        if ctx.cfg is None:
            return _missing_cfg()
        return remote_mod.shell(ctx.cfg, args.command, workdir=ctx.workdir)

    tools.append(AgentTool("shell", "Run one shell command on the GPU machine.", ShellArgs,
                           DictResult, shell))

    return tools


def trainer_tools(ctx: TrainerContext) -> list[AgentTool]:
    """Every tool of the training agent, bound to one context."""
    return _observe_tools(ctx) + _test_tools(ctx) + _act_tools(ctx)
