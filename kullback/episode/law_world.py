"""One Task's built Environment as a world the D258 laws are checked against.

An EnvironmentLawWorld adapts one Task of a BuiltEnvironment to the LawWorld
protocol in kullback.laws: reset rebuilds the Task's Starting state through the
compiled toolkit the way an Episode reset does, tools names the toolkit's tools
with argument shapes drawn only from that Task's own recorded calls, call routes
through the same Router a Run uses with no model stand-in and no Simulated
user, and snapshot returns the canonical state the laws compare. It never
imports the Builder: it stands on the environment's own readers and the Runner.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

from kullback.episode import loading
from kullback.episode.environment import BuiltEnvironment
from kullback.laws import ArgSpec, Outcome, ToolInfo
from kullback.runner import route
from kullback.runner.records import Task, Trace, plain

LAWS_VERSION = 1
"""The version of the law set a consistency number was checked under.

The laws live in kullback.laws, which this package does not own: this number
marks the output, so a later law set is told apart from this one. Bump it when
LAW_NAMES changes.
"""


def _scalar_type(values: list) -> Optional[str]:
    """The laws' type name of the first scalar recorded, else None where none is one."""
    for value in values:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
    return None


def _recorded_calls(members: list[Trace]) -> tuple:
    """Per tool, the recorded argument values and presence counts, and the calls per tool.

    Only what the Task's own traces show: the harness draws its sequences from
    these names alone, never from the mined signatures.
    """
    values: dict = {}
    present: dict = {}
    totals: dict = {}
    for trace in members:
        for call in trace.tool_calls or []:
            totals[call.name] = totals.get(call.name, 0) + 1
            seen = values.setdefault(call.name, {})
            counts = present.setdefault(call.name, {})
            for name, value in (call.args or {}).items():
                seen.setdefault(name, []).append(value)
                counts[name] = counts.get(name, 0) + 1
    return values, present, totals


def _error_text(error: Any) -> str:
    """One stable string for a routed error: its class, with its payload where it has one."""
    if error.payload is None:
        return str(error.class_)
    return f"{error.class_}: {error.payload}"


class EnvironmentLawWorld:
    """One Task's world behind the LawWorld protocol: reset, tools, call, snapshot."""

    def __init__(self, env: BuiltEnvironment, task_id: str, *, seed: int = 0):
        self._env = env
        self._task_id = task_id
        self._seed = seed
        self._router: Any = None

    def reset(self) -> None:
        """Rebuild the Task's Starting state through the compiled toolkit.

        The same readers an Episode reset uses: the Task's overlay, the toolkit
        source off disk, the shared db with the overlay merged at load, and the
        same Router arguments. No Simulated user is built and no Run is
        recorded: the laws need the world, not the Run around it.
        """
        self._task()
        overlay, overlay_rows = self._env.overlay(self._task_id)
        source = self._env.toolkit_source()
        toolkit = loading.load_toolkit(source, copy.deepcopy(self._env.db),
                                       overlay=overlay, overlay_values=overlay_rows)
        context = getattr(toolkit, "ctx", None)
        if context is not None:
            context.reseed(self._seed)
        self._router = route.Router(
            env_tools_module=toolkit, starting_state=copy.deepcopy(self._env.db),
            overlay=overlay, overlay_rows=overlay_rows, tool_sigs=self._env.sigs,
            canon_rules=self._env.canon_rules,
            synthetic_rows=getattr(self._env.schema, "synthetic_rows", None) or ())
        route.refuse_stand_in(self._router)

    def tools(self) -> list:
        """The toolkit's tools with argument shapes from the Task's own recorded calls.

        Each argument name, its type and whether it is optional comes only from
        the arguments of the recorded calls of this Task's traces, which is what
        the harness draws its sequences from. A tool the Task never recorded
        keeps its name and kind with no arguments. The shape is laws.ToolInfo.
        """
        values, present, totals = _recorded_calls(self._env.members(self._task()))
        infos = []
        for sig in self._env.sigs:
            specs = []
            for name in sorted(values.get(sig.name, {})):
                arg_type = _scalar_type(values[sig.name][name])
                if arg_type is None:
                    continue
                specs.append(ArgSpec(name, arg_type, present[sig.name][name] < totals[sig.name]))
            infos.append(ToolInfo(sig.name, sig.kind, tuple(specs)))
        return infos

    def call(self, name: str, args: Mapping[str, Any]) -> Outcome:
        """One call through the Run's Router, answered by code or refused, never invented.

        No model stand-in rides the Router (reset refused one) and no Simulated
        user answers: an unlisted tool is refused the way a Run refuses it.
        """
        result = self._ready_router().route(name, dict(args), requestor="assistant")
        if result.error is None:
            return Outcome(True, plain(result.result))
        return Outcome(False, plain(result.result), _error_text(result.error))

    def snapshot(self) -> Any:
        """The world's canonical state, the value the laws compare for equality."""
        return self._ready_router().world()

    def _task(self) -> Task:
        """The Task, or the refusal naming it where the directory holds none."""
        return self._env.task(self._task_id)

    def _ready_router(self) -> Any:
        """The Router reset built, or the refusal when reset has not run yet."""
        if self._router is None:
            raise RuntimeError(f"reset before use: no world is built for Task {self._task_id}")
        return self._router
