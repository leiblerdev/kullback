"""One Task's built Environment as a world the D258 laws are checked against.

An EnvironmentLawWorld adapts one Task of a BuiltEnvironment to the LawWorld
protocol in kullback.laws: reset rebuilds the Task's Starting state through the
compiled toolkit the way an Episode reset does, tools names the fit tools with
argument shapes from the mined signatures refined by the Task's recorded calls,
call routes through the same Router a Run uses with no model stand-in and no
Simulated user, snapshot returns the canonical state the laws compare, and
unfit names the tools no sequence may check. It never imports the Builder: it
stands on the environment's own readers and the Runner.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

from kullback.episode import loading
from kullback.episode.environment import BuiltEnvironment
from kullback.laws import ArgSpec, Outcome, ToolInfo, _type_bucket
from kullback.runner import route
from kullback.runner.records import Task, ToolSig, Trace, plain

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


def _recorded_calls(members: list[Trace]) -> dict:
    """Per tool, per argument, the recorded values: the type fallback where the signature names none."""
    calls: dict = {}
    for trace in members:
        for call in trace.tool_calls or []:
            seen = calls.setdefault(call.name, {})
            for name, value in (call.args or {}).items():
                seen.setdefault(name, []).append(value)
    return calls


def _sig_names(sig: ToolSig) -> list:
    """Every argument name the signature offers, fields and schema properties together, sorted."""
    names = {field.name for field in sig.args_fields or []}
    props = (sig.args_schema or {}).get("properties") or {}
    if isinstance(props, dict):
        names.update(props)
    return sorted(names)


def _sig_types(sig: ToolSig) -> dict:
    """Candidate type names per argument, the fields first and the schema properties after."""
    out: dict = {}
    for field in sig.args_fields or []:
        out.setdefault(field.name, []).extend(field.types or [])
    props = (sig.args_schema or {}).get("properties") or {}
    if not isinstance(props, dict):
        return out
    for name, prop in props.items():
        declared = prop.get("type") if isinstance(prop, dict) else None
        if isinstance(declared, str):
            out.setdefault(name, []).append(declared)
        elif isinstance(declared, list):
            out.setdefault(name, []).extend(t for t in declared if isinstance(t, str))
    return out


def _sig_optional(sig: ToolSig) -> dict:
    """Whether each argument is optional: the fields say so, else the schema's required list."""
    out = {field.name: bool(field.optional) for field in sig.args_fields or []}
    required = (sig.args_schema or {}).get("required") or []
    wanted = set(required) if isinstance(required, list) else set()
    props = (sig.args_schema or {}).get("properties") or {}
    if not isinstance(props, dict):
        return out
    for name in props:
        out.setdefault(name, name not in wanted)
    return out


def _drawable_type(candidates: list, values: list) -> Optional[str]:
    """The first type the laws can draw: the signature's names first, else the recorded scalars.

    A named type the laws cannot draw is treated as unnamed: it falls through
    to the recorded values rather than crashing the checker.
    """
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        try:
            _type_bucket(candidate)
        except ValueError:
            continue
        return candidate
    return _scalar_type(values)


def _schema_columns(schema: Any) -> set:
    """Every column name the schema's tables hold, with the id pattern columns beside them."""
    names = set()
    for column in getattr(schema, "columns", None) or ():
        if getattr(column, "name", None):
            names.add(column.name)
    for key in getattr(schema, "id_patterns", None) or ():
        tail = str(key).split(".")[-1]
        if tail:
            names.add(tail)
    return names


def _id_shaped(name: str) -> bool:
    """Whether an argument name reads as an identifier: id itself, a name ending in _id, or one ending in _number."""
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.endswith("_number")


def _is_id_param(name: str, arg_type: Optional[str], columns: set) -> bool:
    """Whether an argument draws existing ids: a scalar id-shaped name the schema's tables hold.

    The signature gives the type and the name; the schema's tables confirm the name is a
    column the world stores, so a label that only reads like an id never counts.
    """
    if arg_type is None:
        return False
    try:
        bucket = _type_bucket(arg_type)
    except ValueError:
        return False
    if bucket not in ("int", "str"):
        return False
    return _id_shaped(name) and name in columns


def _side_of(sig: ToolSig) -> Optional[str]:
    """The requestor the Environment lists a tool for, or None where it lists neither side.

    The mined callers decide: assistant where listed, else the first listed side. A tool
    with no mined list keeps the historical caller of assistant.
    """
    callers = getattr(sig, "callers", None)
    if callers is None:
        return "assistant"
    if not callers:
        return None
    if "assistant" in callers:
        return "assistant"
    return callers[0]


def _tool_specs(sig: ToolSig, recorded: dict, columns: set) -> tuple:
    """One tool's drawable ArgSpecs, or None with the reason the tool is unfit.

    The mined signature is the source of argument names, requiredness and type;
    the recorded calls only refine the type where the signature names none the
    laws can draw. A required argument with no drawable type makes the tool
    unfit; an optional one is left out of the specs. Id-shaped arguments the
    schema's tables hold are marked so the drawer tries existing ids.
    """
    types = _sig_types(sig)
    optional = _sig_optional(sig)
    specs = []
    missing = []
    for name in _sig_names(sig):
        arg_type = _drawable_type(types.get(name, []), recorded.get(name, []))
        if arg_type is None:
            if optional.get(name, True):
                continue
            missing.append(name)
        else:
            specs.append(ArgSpec(name, arg_type, optional.get(name, True),
                                 _is_id_param(name, arg_type, columns)))
    if missing:
        reasons = "; ".join(f"required argument {name} has no type" for name in missing)
        return None, reasons
    return specs, None


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
        """The toolkit's fit tools with drawable argument shapes. The shape is laws.ToolInfo.

        The mined signature is the source of argument names, requiredness and
        type; the Task's recorded calls only refine the type where the
        signature names none the laws can draw. Each tool carries the requestor
        the Environment lists it for, so the laws call it the way a Run would.
        Tools with an untyped required argument, and tools listed for neither
        side, are left out of the sequences; unfit() names them and why.
        """
        recorded = _recorded_calls(self._env.members(self._task()))
        columns = _schema_columns(self._env.schema)
        infos = []
        for sig in self._env.sigs:
            side = _side_of(sig)
            if side is None:
                continue
            specs, _ = _tool_specs(sig, recorded.get(sig.name, {}), columns)
            if specs is not None:
                infos.append(ToolInfo(sig.name, sig.kind, tuple(specs), (), side))
        return infos

    def unfit(self) -> dict:
        """The tools tools() leaves out, each with its reason.

        A tool is unfit when a required argument has no drawable type (neither
        the signature names one the laws can draw nor the recorded calls show a
        scalar), or when the Environment lists it for neither side. Such a tool
        is never reported as checked.
        """
        recorded = _recorded_calls(self._env.members(self._task()))
        columns = _schema_columns(self._env.schema)
        gaps = {}
        for sig in self._env.sigs:
            if _side_of(sig) is None:
                gaps[sig.name] = "tool is listed for neither side"
                continue
            _, reason = _tool_specs(sig, recorded.get(sig.name, {}), columns)
            if reason is not None:
                gaps[sig.name] = reason
        return gaps

    def call(self, name: str, args: Mapping[str, Any]) -> Outcome:
        """One call through the Run's Router, answered by code or refused, never invented.

        No model stand-in rides the Router (reset refused one) and no Simulated
        user answers: each tool routes under the requestor the Environment lists
        it for, and an unlisted tool is refused the way a Run refuses it.
        """
        result = self._ready_router().route(name, dict(args), requestor=self._side_for(name))
        if result.error is None:
            return Outcome(True, plain(result.result))
        return Outcome(False, plain(result.result), _error_text(result.error))

    def _side_for(self, name: str) -> str:
        """The requestor one call routes under: the tool's listed side, else the historical caller."""
        for sig in self._env.sigs:
            if sig.name == name:
                return _side_of(sig) or "assistant"
        return "assistant"

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
