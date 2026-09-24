"""Anyone's handle on a Run in a built Environment: reset, step, a reward by code (G1).

An Episode drives one Task through the same world half the Builder's Candidate Runs run in: the
rendered toolkit off disk under confinement, the Task's overlay, the rule-driven Simulated user,
and the stop rules, advanced one assistant message at a time through `advance` from the step-split
patch. It never asks a policy itself and never imports the Builder. Every Run is recorded from day
one under a folder of its own: the seed in the footer, the route on every tool result, the Run
file beside nothing the build wrote.

The same seed, Task and messages give the same Run: nothing on this path reads the machine's clock, a random
source or a model, and the seed only labels the record. What the record already treats as its own
(run id, file path) is the only thing that differs between two such Runs.
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kullback.runner import loop, route
from kullback.runner.records import as_dict
from kullback.runner.verdict import load_run, verdict
from kullback.runner.world import clock, loading
from kullback.runner.world.environment import BuiltEnvironment, episode_message

RUNS_DIR = "episodes"


class EpisodeError(RuntimeError):
    """The Episode cannot do that: no reset yet, the Run is over, or the Task needs a model user."""


@dataclass
class Reset:
    """What a policy needs to act: the prompt, the specs, the opening user message, the Run's names."""

    run_id: str
    seed: int
    system_prompt: Optional[str]
    tools: list[dict] = field(default_factory=list)
    opening: Optional[str] = None


@dataclass
class StepOut:
    """What one assistant message brought back: each tool answer, the user turn, whether it ended."""

    results: list[dict] = field(default_factory=list)
    user_message: Optional[str] = None
    done: bool = False
    stop_reason: Optional[str] = None


@dataclass
class Reward:
    """The stored Run scored by code alone: 1 for pass, 0 for fail, no number where code cannot say."""

    score: Optional[int]
    class_: str
    reason: str


def _needs_world_step() -> None:
    if not (hasattr(loop, "ask") and hasattr(loop, "advance")):
        raise EpisodeError("the step-split patch is not applied: advance is what steps the world")


def _plain_run(run: Any) -> dict:
    """One Run as plain JSON-ready data."""
    return json.loads(json.dumps(as_dict(run), ensure_ascii=False, default=str))


class Episode:
    """One recorded Run over one Task, driven from outside (G1)."""

    def __init__(self, env: BuiltEnvironment, *, outdir: Any = None, max_turns: int = 30,
                 user_factory: Any = None):
        _needs_world_step()
        self.env = env
        self.outdir = Path(outdir) if outdir is not None else env.root / RUNS_DIR
        self.max_turns = max_turns
        # Who answers as the Task's user: a callable(env, task, router, rules, reference) the
        # caller provides. The world owns no user extension, so without one the Run has no
        # Simulated user and ends where a package Run ends. The runner tool passes its own
        # user object straight into the Run state instead of building one here.
        self._user_factory = user_factory
        self.task_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self._state: Any = None
        self._router: Any = None
        self._tools: list[dict] = []

    def reset(self, task_id: str, seed: int = 0) -> Reset:
        """Start a recorded Run of the Task under the seed: the prompt, the specs, the opening turn."""
        if not task_id or task_id in (".", "..") or "/" in task_id or "\\" in task_id:
            raise EpisodeError("task_id must be a single safe path component")
        if self._state is not None and not self._state.stopped:
            raise EpisodeError("finish the current Run before resetting")
        task = self.env.task(task_id)
        if self.env.agent_driven(task_id):
            raise EpisodeError(f"Task {task_id} is driven by the agent user, which needs a model "
                               "the interface does not have")
        overlay, overlay_rows = self.env.overlay(task_id)
        source = self.env.toolkit_source()
        db = copy.deepcopy(self.env.db)
        toolkit = loading.load_toolkit(source, db, overlay=overlay, overlay_values=overlay_rows)
        clock.start_context(toolkit, seed, self.env.clock(task_id))
        self._router = route.Router(
            env_tools_module=toolkit, starting_state=copy.deepcopy(self.env.db),
            overlay=overlay, overlay_rows=overlay_rows, tool_sigs=self.env.sigs,
            canon_rules=self.env.canon_rules,
            synthetic_rows=getattr(self.env.schema, "synthetic_rows", None) or ())
        rules = self.env.rules(task)
        reference = self.env.reference(task)
        # The caller builds the Simulated user from the Task's own evidence (its reference,
        # its recordings, the write set the reference implies); without a factory, or where
        # the directory carries no rules, the Run has no Simulated user.
        simulated = self._user_factory(self.env, task, self._router, rules, reference) \
            if rules is not None and self._user_factory is not None else None
        run_id = f"{task_id}-{seed}"
        system_prompt = self.env.system_prompt(task)
        self._tools = self.env.tool_specs()
        task_outdir = self.outdir / task_id
        task_outdir.mkdir(parents=True, exist_ok=True)
        recording_dir = Path(tempfile.mkdtemp(prefix="run-", dir=task_outdir))
        self._state = loop.new_run_state(
            run_id, workdir=recording_dir, env_id=self.env.env_id, task_id=task_id,
            model="episode", seed=seed, user=simulated,
            user_rules=rules, max_turns=self.max_turns, system_prompt=system_prompt,
            first_user=task.intent if simulated is None else None)
        self._state.tools = copy.deepcopy(self._tools)
        opening = loop.open_with_user(self._state)
        if simulated is None:
            opening = task.intent
        self.task_id, self.run_id = task_id, run_id
        return Reset(run_id=run_id, seed=self._state.run.seed, system_prompt=system_prompt,
                     tools=copy.deepcopy(self._tools), opening=opening)

    def step(self, message: dict) -> StepOut:
        """Advance the world with one assistant message and return what came back.

        The message has the shape `advance` takes: content plus tool calls with id, name and
        arguments. The turn is counted and the message joins the transcript the way `ask` would
        have put it there, so a replayed Run reads exactly as the Builder's Run read.
        """
        if self._state is None:
            raise EpisodeError("reset before step: no Run has started")
        if self._state.stopped:
            raise EpisodeError(f"Run {self.run_id} is over ({self._state.run.termination_reason}): "
                               "it takes no further steps")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        raw_calls = message.get("tool_calls", ()) if isinstance(message, dict) \
            else getattr(message, "tool_calls", ()) or ()
        normalized = copy.deepcopy(episode_message(content, raw_calls))
        self._state.turn += 1
        self._state.messages.append({"role": "assistant", "content": normalized["content"],
                                     "tool_calls": normalized["tool_calls"]})
        # The driven message is recorded the way `ask` records the policy's answer: the Verdict
        # reads assistant text off `model_call` events, so a Run without them mis-scores every
        # atom that checks what the Candidate said. No cost rides it: the policy spends outside
        # the Run, under the solve-rate ceiling, the way the ledger holds a build's spend.
        loop.emit(self._state, "model_call", {"reply": {"content": normalized["content"],
                                                          "tool_calls": normalized["tool_calls"]}})
        seen = len(self._state.run.events)
        try:
            loop.advance(self._state, {"content": normalized["content"],
                                       "tool_calls": normalized["tool_calls"]}, self._router)
        except Exception as exc:
            loop._crashed(self._state, exc, self._router)
            raise EpisodeError(f"Run {self.run_id} ended on an environment failure: "
                               f"{type(exc).__name__}: {exc}") from exc
        fresh = self._state.run.events[seen:]
        results = [{"id": e.payload.get("id"), "name": e.payload.get("name"),
                    "result": e.payload.get("result"), "error": e.payload.get("error"),
                    "route": e.route}
                   for e in fresh if e.type == "tool_result"]
        user_message = next((e.payload.get("text") for e in reversed(fresh)
                             if e.type == "user_turn"), None)
        done = bool(self._state.stopped)
        if done and not self._state.finished:
            loop.finish(self._state, self._router)
        return StepOut(results=copy.deepcopy(results), user_message=user_message, done=done,
                       stop_reason=self._state.run.termination_reason if done else None)

    def abort(self, reason: str = "cancelled") -> None:
        if reason not in ("cancelled", "budget_exceeded", "policy_error"):
            raise EpisodeError("unknown interruption reason")
        if self._state is None:
            raise EpisodeError("reset before abort: no Run has started")
        if self._state.stopped:
            return
        loop.emit(self._state, "error", {"class": "env_error", "message": f"episode interrupted: {reason}"})
        loop._stop(self._state, reason)
        loop.finish(self._state, self._router)

    def reward(self) -> Reward:
        """Score the stored Run with the Verdict by code alone: 1, 0, or no number with the reason.

        Judge atoms arrive with no judge results, so a Verifier holding a must-hold judge atom
        comes out not verdicted: no number at all, with the class and the reason. An environment
        failure is the same: the Run says what broke, the reward says nothing. Never invented.

        The Run is scored from its own file, the way the build scores stored Runs: a tool result
        payload aliases the toolkit object it came from, and a later write mutates that object in
        place, so the in-memory record rewrites its own history while the JSONL stays frozen at
        emit time. Scoring memory would mis-score exactly the Runs that wrote.
        """
        if self._state is None or not self._state.stopped:
            raise EpisodeError("reward after the Run is over: it is not over yet")
        verifier = self.env.verifier(self.task_id or "")
        if verifier is None:
            return Reward(score=None, class_="no_verifier",
                          reason=f"Task {self.task_id} holds no stored Verifier")
        write_tools = self.env.write_tools()
        flagged = {sig.name for sig in self.env.sigs if getattr(sig, "unclassified", False)}
        run = load_run(self._state.path)
        result = verdict(run, verifier, self.env.canon_rules, None,
                         schema=self.env.schema, write_tools=write_tools or None,
                         flagged_tools=flagged)
        if result.class_ == "pass":
            return Reward(score=1, class_=result.class_, reason="")
        if result.class_ in ("not_verdicted", "env_error"):
            return Reward(score=None, class_=result.class_,
                          reason=result.failing_atom or "; ".join(result.notes[-2:]))
        return Reward(score=0, class_=result.class_, reason=result.failing_atom or "")

    def run_file(self) -> Optional[Path]:
        """Where this Run is recorded, under the Episode's own folder.

        Named off `runpy.run_path`, which the runner's import boundary refuses by name: a
        Run file accessor that kept the old name tripped that gate on every call site once
        the world moved under runner/.
        """
        return self._state.path if self._state is not None else None

    def run_path(self) -> Optional[Path]:
        """The old name, kept while scripts/episode_acceptance.py still calls it."""
        return self.run_file()

    def transcript(self) -> list[dict]:
        """The transcript the policy is asked against: system prompt, turns so far, in order."""
        if self._state is None:
            raise EpisodeError("no Run has started")
        return copy.deepcopy(self._state.messages)

    def run_record(self) -> dict:
        """The stored Run as plain data; the events come from the Run's own file (see `reward`)."""
        if self._state is None:
            raise EpisodeError("no Run has started")
        path = self._state.path
        if path is not None:
            try:
                stored = load_run(path)
            except FileNotFoundError:
                stored = None
            if stored is not None:
                run = self._state.run.model_copy(update={"events": stored.events})
                return _plain_run(run)
        if self._state.run.events:
            raise EpisodeError(f"{path} holds no record of Run {self._state.run.run_id}: "
                               "the Run's record is missing")
        return _plain_run(self._state.run)
