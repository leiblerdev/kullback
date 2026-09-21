"""What a Run needs from the Builder, owned below it (G1).

Everything here moved down out of `kullback.builder` byte for byte: the Builder imports it back
from this module, so there is one code path. The episode driver (`environment`, `episode`) stands
on these and on `ask` and `advance` from the step-split patch, and never imports the Builder.
"""



from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import sampling
from kullback.gates.confinement import PROVIDED_HELPERS, TOOLS_CLASS, source_confinement
from kullback.runner import arith
from kullback.runner.records import (
    OverlayRow,
    OverlayStep,
    Task,
    TaskOverlay,
    UserRules,
)
from kullback.user import vocabulary


class EnvironmentError(RuntimeError):
    """The directory holds no built Environment, or it lacks a file a Run cannot do without."""


_MISSING = object()

_SHAPES = {"object": dict, "array": list}


def _shaped_json(path: Path, expected: str, default: Any, *, what: str) -> Any:
    """The JSON `path` holds, the default where no file sits, else a refusal.

    A present file must parse and hold the expected shape: a torn file or a
    value of another shape raises EnvironmentError naming the path and the shape.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"{path} is unreadable or torn") from exc
    if not isinstance(body, _SHAPES[expected]):
        raise EnvironmentError(f"{path} holds {what} that is not an {expected}")
    return body


def _record(path: Path, model: Any) -> Any:
    """The record `path` holds, validated as `model`, else a refusal naming both."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentError(f"{path} is missing: expected a {model.__name__}") from exc
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"{path} is unreadable or torn") from exc
    try:
        return model.model_validate(body)
    except ValueError as exc:
        raise EnvironmentError(f"{path} does not hold a valid {model.__name__}") from exc


# --- failures and names shared by both loaders (moved from kullback.builder.sandbox) ---

class SandboxError(RuntimeError):
    """The generated module did not load, or the subprocess crashed or ran out of time."""


DB_CLASS = "DomainDB"


# What both loaders put in the generated module's namespace beside `__name__`, and nothing else:
# code-owned functions a tool body may call by name without importing anything. Only
# `evaluate_arithmetic` so far, for the reason runner/arith.py gives: a recorded tool that evaluates
# an expression string cannot be written without a parser, `ast`, `eval`, `exec` and `compile` are
# refused, and the parser the model writes instead is wrong in a new way on every build.
#
# Keyed off the confinement gate's own `PROVIDED_HELPERS` rather than written out again, so the two
# cannot drift: a name the gate counts as bound with no function behind it fails at import here, and
# a function the gate has never heard of is never bound into a body's namespace, where it would have
# been refused as a name nothing binds.
_HELPER_FUNCTIONS = {"evaluate_arithmetic": arith.evaluate_arithmetic}
HELPERS = {name: _HELPER_FUNCTIONS[name] for name in sorted(PROVIDED_HELPERS)}


# --- overlays (moved from kullback.builder.compile_env) ---

OVERLAY_DIR = "overlays"

class OverlayConflict(ValueError):
    """Two Tasks pin the same row in different versions: a Gate failure for the benchmark export (D74)."""


def _with_run_scope(overlay: TaskOverlay, scope: Optional[dict]) -> TaskOverlay:
    """The Task overlay with one Run's own rows and its own step sequence laid over it (D213)."""
    if not scope:
        return overlay
    replaced = {(str(row["table"]), str(row["id"])): row for row in scope.get("rows") or ()}
    rows = [OverlayRow.model_validate(replaced[(row.table, row.id)])
            if (row.table, row.id) in replaced else row for row in overlay.rows]
    return overlay.model_copy(update={
        "rows": rows, "steps": [OverlayStep.model_validate(step) for step in scope.get("steps") or ()]})


def load_overlay(workdir: Path | str, task_id: str,
                 run_id: Optional[str] = None) -> tuple[TaskOverlay, dict]:
    """The overlay one Run of the Task replays against, and its row values (D74, D213).

    Without a Run, or with one this Task has no layer for, which is every generated Candidate Run,
    the Reference Run's layer is served: one Run's world whole rather than a mix of two. route.py
    reads this before the shared db.json.
    """
    path = Path(workdir) / OVERLAY_DIR / f"{task_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentError(f"{path} holds no overlay for Task {task_id}") from exc
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"{path} is unreadable or torn") from exc
    if not isinstance(payload, dict):
        raise EnvironmentError(f"{path} holds an overlay that is not an object")
    try:
        overlay = TaskOverlay.model_validate(payload["overlay"])
        values = payload["values"]
    except (KeyError, ValueError) as exc:
        raise EnvironmentError(f"{path} holds no valid overlay for Task {task_id}") from exc
    runs = payload.get("runs") or {}
    if not isinstance(values, dict) or not isinstance(runs, dict):
        raise EnvironmentError(f"{path} holds no valid overlay for Task {task_id}")
    scope = runs.get(run_id) if run_id else None
    if scope is None:
        scope = runs.get(str(payload.get("reference_run_id") or ""))
    return _with_run_scope(overlay, scope), values


def merge_overlays(db: dict, overlays: Iterable[TaskOverlay], values: dict,
                   conflicts: Optional[list[str]] = None) -> dict:
    """Merge every Task overlay into the one db the benchmark export's harness loads (D74).

    Two Tasks pinning one row in different versions is a disagreement about the world, which the
    per-Task overlays in the Runner never have to settle: each Task reads its own. The single db of
    the benchmark export has to pick one, so with `conflicts` given it keeps the version a Task saw before
    any write (over one seen after a write) and, between two of the same standing, the first Task's,
    and appends one line per conflict for the export gate. Without `conflicts` a disagreement raises,
    which is the contract the single-overlay callers rely on.
    """
    merged = copy.deepcopy(db)
    pinned: dict[tuple[str, str], tuple[str, str, bool]] = {}
    for overlay in overlays:
        for row in overlay.rows:
            seen = pinned.get((row.table, row.id))
            if seen and seen[0] != row.version_hash:
                if conflicts is None:
                    raise OverlayConflict(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row "
                                          f"{row.id} in different versions")
                conflicts.append(f"tasks {seen[1]} and {overlay.task_id} pin {row.table} row {row.id} in "
                                 f"different versions; the benchmark export keeps "
                                 f"{overlay.task_id if seen[2] and not row.after_write else seen[1]}'s")
                if not (seen[2] and not row.after_write):
                    continue  # the pinned version stands: it was seen before a write, or both were
            pinned[(row.table, row.id)] = (row.version_hash, overlay.task_id, row.after_write)
            if row.version_hash in values:
                merged.setdefault(row.table, {})[row.id] = values[row.version_hash]
    return merged


# --- the toolkit, executed under confinement (moved from kullback.builder.compile_env) ---

def load_toolkit(source: str, db: dict, class_name: str = TOOLS_CLASS, db_class: str = DB_CLASS,
                 overlay: Optional[TaskOverlay] = None, overlay_values: Optional[dict] = None):
    """The generated module loaded in this process, with the Task's Starting state inside it.

    Given a Task's overlay, the world handed to the toolkit is the shared db with that overlay
    merged, so the March Task's code-routed call sees the March row and not the June one (D74). A
    generated body reads `self.db` and cannot do the overlay lookup itself, so a toolkit built on the
    shared db alone leaves the overlay dead for every code route; `route.py`'s StateView stays the
    lookup for the recording and stand-in routes.

    dont_inherit keeps a caller's `from __future__ import annotations` out of the generated module:
    a postponed annotation has no module globals for pydantic to resolve against. This is the loader
    the Runner's router is given; the gates use the subprocess Sandbox instead.

    `evaluate_arithmetic` is put in the namespace, not imported by the body: `ast`, `eval`, `exec`
    and `compile` stay refused, and a recorded tool that evaluates an expression string is served by
    the one code-owned evaluator instead of a parser the model writes again on every build (see
    runner/arith.py). `PROVIDED_HELPERS` is what the confinement gate reads it under, so the two
    cannot drift: a name bound here that the gate does not know is a body refused for a NameError it
    would never have raised.
    """
    if overlay is not None:
        db = merge_overlays(db, [overlay], overlay_values or {})
    refused = source_confinement(source, class_name)
    if refused:
        raise SandboxError("the generated module is not confined and would run in this process: "
                           + "; ".join(refused))
    namespace: dict = {"__name__": "generated_tools", **HELPERS}
    exec(compile(source, "<generated>", "exec", dont_inherit=True), namespace)  # noqa: S102
    return namespace[class_name](namespace[db_class].model_validate(db))


# --- the Candidate Run's inputs (moved from kullback.builder.build) ---

# D212: the kind a Candidate-shaped Run's seed is drawn under, keyed on its Run id
RUN_SEED_KIND = "run_seed"


def run_seed(run_id: str, salt: str) -> int:
    """The seed of one Run from its id and the build salt (D212), drawn the way the Builder draws it.

    A Run discarded and made again under its own id draws the seed of the Run it replaces, so the
    re-made Run is the one it replaced rather than the next number off a counter.
    """
    return sampling.sample_seed(RUN_SEED_KIND, run_id, salt)


def _system_prompt_for(task: Task, traces: dict, policy_text: Optional[str] = None) -> Optional[str]:
    """The instructions a Candidate runs under: the recorded agent's own system prompt, else the policy text."""
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        if trace is not None and trace.system_prompt:
            return trace.system_prompt
    return policy_text or None


_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
               "list": "array", "NoneType": "null"}


def _tool_definitions(sigs: list, vocab: Optional[vocabulary.Vocabulary] = None) -> list[dict]:
    """The mined signatures in the shape provider.py sends to a model: name, description, parameters.

    mine.py records Python type names (`["str"]`); a model endpoint wants JSON Schema names, and
    OpenAI refuses a tool whose parameter type it does not know. An argument the Vocabulary knows
    gets the values the corpus showed for it as its description: build 8's re-roll model dropped
    the mark from 132 order ids over a schema that showed it nothing, and the traces declare no
    tool descriptions of their own.
    """
    out = []
    for sig in sigs:
        schema = sig.args_schema if isinstance(sig.args_schema, dict) and "properties" in sig.args_schema else {
            "type": "object", "properties": {name: {"type": "string"} for name in (sig.args_schema or {})}}
        parameters = _json_schema(schema)
        for arg, prop in (parameters.get("properties") or {}).items():
            spec = vocab.get(arg) if vocab is not None else None
            if spec is not None and spec.examples and isinstance(prop, dict) and not prop.get("description"):
                prop["description"] = "for example " + ", ".join(str(v) for v in spec.examples[:3])
        out.append({"name": sig.name, "description": sig.description or f"{sig.kind} tool {sig.name}",
                    "parameters": parameters})
    return out


def _json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {key: _json_schema(value) for key, value in node.items() if key != "type"}
        if "type" in node:
            types = node["type"] if isinstance(node["type"], list) else [node["type"]]
            names = sorted({_JSON_TYPES.get(str(t), str(t)) for t in types})
            out["type"] = names[0] if len(names) == 1 else names
        return out
    if isinstance(node, list):
        return [_json_schema(item) for item in node]
    return node


def _vocab_from(workdir: Path) -> vocabulary.Vocabulary:
    """The build's Vocabulary off disk, for a Run made outside the pipeline (run_batch, the probe)."""
    stored = _shaped_json(Path(workdir) / "vocabulary.json", "object", None, what="a vocabulary")
    return vocabulary.Vocabulary.model_validate(stored) if stored else vocabulary.GENERIC


def _user_rules(workdir: Path, task: Task) -> Optional[UserRules]:
    for run_id in task.run_ids:
        path = Path(workdir) / "user_rules" / f"{run_id}.json"
        if not path.is_file():
            continue
        return _record(path, UserRules)
    return None
