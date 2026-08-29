"""The stage graph: topological, content-addressed, one rollback edge on a failed gate, the held-out anchor
(D81) and the spend ceiling (D86)."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel

from harness.shared.records import ALL_RECORDS, GateResult, Task, as_dict, content_hash

ANCHOR_SHARE = 0.20
ANCHOR_MIN_RUNS = 3
ANCHOR_SEED = 20260827
ANCHOR_NAME = "anchor.json"
MAX_ATTEMPTS = 3
CACHE_FORMAT = 2  # part of every cache key, so entries written by an older encoder are never read back
RECORD_TYPES = {cls.__name__: cls for cls in ALL_RECORDS}
STATUS_STYLE = {
    "pending": "fill:#f5f5f5,stroke:#9e9e9e", "ran": "fill:#e8f5e9,stroke:#43a047",
    "cached": "fill:#e3f2fd,stroke:#1e88e5", "rolled_back": "fill:#fff8e1,stroke:#f9a825",
    "failed": "fill:#ffebee,stroke:#e53935", "stopped": "fill:#ede7f6,stroke:#5e35b1",
    "crashed": "fill:#fbe9e7,stroke:#d84315",
}


class PipelineError(RuntimeError):
    """Anything this module refuses to do."""


class GraphError(PipelineError):
    """The stage graph does not hold together: duplicate names, two producers, a missing input."""


class CycleError(GraphError):
    """The stage graph has a cycle, so there is no order to run it in."""


class AnchorLeak(PipelineError):
    """A Builder stage reached for the held-out anchor (D81); the anchor is for later stages only."""


class BudgetStop(PipelineError):
    """The spend ceiling was reached (D86): stop where you are, report as is, ask before continuing."""

    def __init__(self, report: dict):
        self.report = dict(report)
        self.item = self.report.get("item", "")
        super().__init__(f"spend ceiling reached in stage {self.report.get('stage')} on {self.item}")


# --- the anchor (D81) -------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    """A fixed share of every Task's Runs, held out once and never used as a Builder seed."""
    held_out: dict[str, list[str]]
    unguarded: list[str]
    share: float = ANCHOR_SHARE
    min_runs: int = ANCHOR_MIN_RUNS
    seed: int = ANCHOR_SEED

    def anchor_runs(self, task_id: str) -> list[str]:
        return list(self.held_out.get(task_id, []))

    def is_held_out(self, run_id: str) -> bool:
        return any(run_id in runs for runs in self.held_out.values())

    def seed_runs(self, task_id: str, run_ids: Iterable[str]) -> list[str]:
        """What the Builder may build from: every Run of the Task except its anchor."""
        held = set(self.held_out.get(task_id, []))
        return [r for r in run_ids if r not in held]

    def mark(self, tasks: Sequence[Any]) -> list[Any]:
        """Copies of the Task records with anchor_run_ids and the unguarded flag filled in.

        A Task this anchor never saw is unguarded: nothing was held out of it, so calling it guarded
        would hide a Task the Builder built from all of its Runs (D81).
        """
        out = []
        for task in tasks:
            task_id = _task(task)[0]
            record = Task.model_validate(task) if isinstance(task, dict) else task
            out.append(record.model_copy(update={
                "anchor_run_ids": self.anchor_runs(task_id),
                "unguarded": task_id in self.unguarded or task_id not in self.held_out}))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Anchor":
        return cls(**data)


def _task(task: Any) -> tuple[str, list[str]]:
    """A Task record or the same thing as a dict, as (id, run_ids)."""
    if isinstance(task, dict):
        return task["id"], list(task["run_ids"])
    return task.id, list(task.run_ids)


def anchor_path(workdir: str | Path) -> Path:
    return Path(workdir) / ANCHOR_NAME


def load_anchor(workdir: str | Path) -> Optional[Anchor]:
    """The stored anchor, or None when this build has not chosen one yet."""
    path = anchor_path(workdir)
    return Anchor.from_dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def choose_anchor(tasks: Sequence[Any], workdir: str | Path, share: float = ANCHOR_SHARE,
                  min_runs: int = ANCHOR_MIN_RUNS, seed: int = ANCHOR_SEED) -> Anchor:
    """Pick the held-out Runs once with a fixed seed and store them.

    Every Task the stored anchor already knows keeps exactly the Runs it was given. A Task that
    appeared afterwards (an iterate build, a split) is drawn now under the stored settings and
    appended, because a Task with nothing held out is a Task the Builder can fit to (D81).
    """
    stored = load_anchor(workdir)
    held_out: dict[str, list[str]] = dict(stored.held_out) if stored is not None else {}
    unguarded: list[str] = list(stored.unguarded) if stored is not None else []
    if stored is not None:
        share, min_runs, seed = stored.share, stored.min_runs, stored.seed
    added = False
    for task in tasks:
        task_id, run_ids = _task(task)
        if task_id in held_out:
            continue
        added = True
        run_ids = sorted(run_ids)
        if len(run_ids) < min_runs:
            held_out[task_id] = []
            unguarded.append(task_id)
            continue
        count = max(1, int(len(run_ids) * share))
        held_out[task_id] = sorted(random.Random(f"{seed}:{task_id}").sample(run_ids, count))
    anchor = Anchor(held_out=held_out, unguarded=sorted(set(unguarded)), share=share, min_runs=min_runs, seed=seed)
    if stored is None or added:
        path = anchor_path(workdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(anchor.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return anchor


# --- stages -----------------------------------------------------------------

@dataclass
class Stage:
    """One node: a function over named artifacts, with the gate that accepts or rejects what it made.

    A stage must declare everything it reads: artifacts in `inputs`, and files or directories under
    the workdir in `input_paths`. What is not declared is not in the cache key, and a stage that
    reads an undeclared file is served its first output forever (design section 8).
    """
    name: str
    fn: Callable[["StageContext", dict], dict]
    inputs: Sequence[str] = ()
    outputs: Sequence[str] = ()
    gate: Optional[Callable[["StageContext", dict], GateResult]] = None
    builder: bool = False
    code_version: Optional[str] = None
    max_attempts: int = MAX_ATTEMPTS
    input_paths: Sequence[str] = ()


class StageContext:
    """What a stage is given beside its inputs: the attempt, the last gate failure, money, the anchor."""

    def __init__(self, stage: Stage, workdir: Path, anchor: Optional[Anchor],
                 charge: Callable[[float, str], None], attempt: int = 1, failure: Optional[str] = None):
        self.stage, self.name, self.workdir = stage, stage.name, workdir
        self.attempt, self.failure = attempt, failure
        self._anchor, self._charge = anchor, charge

    @property
    def anchor(self) -> Optional[Anchor]:
        if self.stage.builder:
            raise AnchorLeak(f"stage {self.name} is a Builder stage; the anchor is held out from it (D81)")
        return self._anchor

    def seed_runs(self, task_id: str, run_ids: Iterable[str]) -> list[str]:
        """The Runs a Builder stage may use: the Task's Runs minus its anchor."""
        if self._anchor is None:
            raise PipelineError(f"stage {self.name} asked for the seed Runs of Task {task_id} and this "
                                "build has no anchor, so nothing is held out; choose the anchor before "
                                "the first Builder stage (D81)")
        return self._anchor.seed_runs(task_id, run_ids)

    def charge(self, usd: float, item: str = "") -> None:
        """Record one item's spend; raises when the ceiling is reached, which stops the build (D86)."""
        self._charge(usd, item)


@dataclass
class PipelineResult:
    """What the run produced and how far it got."""
    status: str
    artifacts: dict = field(default_factory=dict)
    statuses: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)
    gates: list = field(default_factory=list)
    log: list = field(default_factory=list)
    stopped: Optional[dict] = None
    failed_stage: Optional[str] = None


def code_hash(stage: Stage) -> str:
    """The code side of a cache key: which function this is and the bytes of its module.

    The module's bytes alone are not the function: two stages of the same name in one module, or two
    partials of one function with different bound parameters, would share a key and the second would
    be served the first's output. A function whose source cannot be found is refused rather than
    hashed as a constant.
    """
    if stage.code_version:
        return stage.code_version
    return content_hash(_fn_identity(stage.fn, stage.name))


def _fn_identity(fn: Any, stage_name: str) -> dict:
    if isinstance(fn, functools.partial):
        return {"partial": _fn_identity(fn.func, stage_name),
                "args": [content_hash(a) for a in fn.args],
                "keywords": {k: content_hash(v) for k, v in sorted(fn.keywords.items())}}
    source = _source_bytes(fn)
    if source is None:
        raise PipelineError(
            f"stage {stage_name}: the source of its function cannot be found, so its code hash would "
            "be the same constant for every such stage and a changed function would be served a stale "
            "cache entry; give the Stage an explicit code_version")
    return {"qualname": getattr(fn, "__qualname__", ""), "module": getattr(fn, "__module__", ""),
            "code": hashlib.sha256(source).hexdigest()}


def _source_bytes(fn: Any) -> Optional[bytes]:
    """The stage function's own source, not the whole module: one edit in a 500-line build.py must
    not cache-bust every other stage that file defines. The whole file is a fallback only, for a
    function whose own source `inspect.getsource` cannot recover on its own."""
    try:
        return inspect.getsource(fn).encode("utf-8")
    except (OSError, TypeError):
        pass
    try:
        path = inspect.getsourcefile(fn)
        return Path(path).read_bytes() if path and Path(path).is_file() else None
    except (OSError, TypeError):
        return None


def _encode(value: Any, where: str = "artifact") -> dict:
    """A stage output as JSON that decodes back to the same thing, or a refusal to cache it.

    Everything the encoder cannot round-trip is refused at write time: a Path, a set or a tuple would
    come back as a string or a list on the next build, and the second run would see a different world
    than the first (design section 8).
    """
    if isinstance(value, BaseModel):
        name = type(value).__name__
        if RECORD_TYPES.get(name) is not type(value):
            raise PipelineError(f"{where} is a {name}, which is not one of the records in records.py; "
                                "the cache carries records, dicts, lists and JSON scalars")
        return {"kind": "record", "class": name, "value": as_dict(value)}
    if isinstance(value, list):
        return {"kind": "list", "value": [_encode(v, f"{where}[{i}]") for i, v in enumerate(value)]}
    if isinstance(value, dict):
        wrong = sorted(str(k) for k in value if not isinstance(k, str))
        if wrong:
            raise PipelineError(f"{where} has non-string keys {wrong}; JSON turns them into strings, so "
                                "the cache hit would not be what the stage returned")
        return {"kind": "dict", "value": {k: _encode(v, f"{where}.{k}") for k, v in value.items()}}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "json", "value": value}
    raise PipelineError(f"{where} is a {type(value).__name__}, which does not survive JSON; return "
                        "records, dicts, lists and JSON scalars from a stage")


def _decode(blob: dict) -> Any:
    kind = blob["kind"]
    if kind == "record":
        return RECORD_TYPES[blob["class"]].model_validate(blob["value"])
    if kind == "list":
        return [_decode(v) for v in blob["value"]]
    if kind == "dict":
        return {k: _decode(v) for k, v in blob["value"].items()}
    return blob["value"]


def _budget_types() -> tuple:
    """BudgetStop plus budget.py's own ceiling error, imported late so this module stands alone."""
    try:
        from harness.shared.budget import BudgetExceeded
        return (BudgetStop, BudgetExceeded)
    except Exception:
        return (BudgetStop,)


def _node(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _label(text: Any) -> str:
    """Mermaid labels are double quoted, so a quote inside one has to be the entity."""
    return str(text).replace('"', "#quot;")


class Pipeline:
    """Runs the stages in topological order, caches what it can, and stops on a gate or the ceiling."""

    def __init__(self, stages: Sequence[Stage], workdir: str | Path,
                 anchor: Optional[Anchor] = None, ceiling: Any = None,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.stages, self.workdir, self.ceiling = list(stages), Path(workdir), ceiling
        self.on_event = on_event
        self.anchor = anchor if anchor is not None else load_anchor(self.workdir)
        self.producers = self._producers()
        self.order = self._toposort()
        self.statuses = {s.name: "pending" for s in self.stages}
        self.attempts = {s.name: 0 for s in self.stages}
        self.rollbacks: dict[str, list[str]] = {}
        self.result: Optional[PipelineResult] = None

    def _emit(self, kind: str, **fields: Any) -> None:
        """Tell a watcher where the build is, without letting the watcher stop the build.

        state.json is only written once the pipeline is done, so nothing outside this object can
        see a build while it is running; a live view needs to be told. The callback is a screen,
        not a stage: a screen that raises has no business failing a build that was going fine, so
        whatever it throws is dropped here.
        """
        if self.on_event is None:
            return
        try:
            self.on_event({"kind": kind, **fields})
        except Exception:
            pass

    def _producers(self) -> dict[str, str]:
        producers: dict[str, str] = {}
        seen: set[str] = set()
        for stage in self.stages:
            if stage.name in seen:
                raise GraphError(f"two stages are named {stage.name}")
            seen.add(stage.name)
            for name in stage.outputs:
                if name in producers:
                    raise GraphError(f"artifact {name} is produced by {producers[name]} and {stage.name}")
                producers[name] = stage.name
        return producers

    def _toposort(self) -> list[Stage]:
        by_name = {s.name: s for s in self.stages}
        state: dict[str, int] = {}
        order: list[Stage] = []

        def visit(name: str, trail: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise CycleError("cycle in the stage graph: " + " -> ".join(trail + [name]))
            state[name] = 1
            for artifact in by_name[name].inputs:
                if self.producers.get(artifact) is not None:
                    visit(self.producers[artifact], trail + [name])
            state[name] = 2
            order.append(by_name[name])

        for stage in self.stages:
            visit(stage.name, [])
        return order

    # --- caching, design section 8: (input record hash, code hash of the stage module) ---

    def _cache_path(self, stage: Stage, inputs: dict) -> Path:
        key = content_hash({
            "format": CACHE_FORMAT,
            "stage": stage.name,
            "code": code_hash(stage),
            "inputs": {k: content_hash(v) for k, v in sorted(inputs.items())},
            "paths": {name: self._path_hash(name) for name in sorted(stage.input_paths)},
            # The anchor is part of the world a stage sees: seed_runs and ctx.anchor both move with
            # it, so an output computed under one anchor is not an answer under another (D81).
            "anchor": content_hash(self.anchor.to_dict()) if self.anchor is not None else None,
        })
        return self.workdir / "cache" / f"{stage.name}.{key[:16]}.json"

    def _path_hash(self, name: str) -> str:
        """The bytes behind a declared path, so a changed input file moves the key."""
        path = self.workdir / name
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_dir():
            return content_hash({str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(path.rglob("*")) if p.is_file()})
        return "missing"

    def _read_cache(self, path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        return {name: _decode(blob) for name, blob in json.loads(path.read_text(encoding="utf-8")).items()}

    def _write_cache(self, path: Path, stage: Stage, outputs: dict) -> None:
        blob = {name: _encode(value, f"stage {stage.name}: artifact {name}") for name, value in outputs.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob), encoding="utf-8")

    # --- the spend ceiling (D86) ---

    def _charger(self, stage_name: str, items_left: int) -> Callable[[float, str], None]:
        def charge(usd: float, item: str = "") -> None:
            if self.ceiling is not None:
                self.ceiling.add(usd, stage_name, item, items_left)
        return charge

    def _stop_report(self, stage: str, item: str, items_left: int, exc: Optional[BaseException]) -> dict:
        """What the report needs when a build stops: where it was, what it spent, what is left."""
        if isinstance(exc, BudgetStop):
            return dict(exc.report)
        report = getattr(self.ceiling, "report", None)
        out = dict(report(stage, item, items_left)) if callable(report) else {
            "stage": stage, "spent": getattr(exc, "spent", None), "items_left": items_left,
            "ceiling_usd": getattr(exc, "ceiling_usd", None),
            "estimate_to_finish": getattr(exc, "estimate_to_finish", None)}
        out["item"] = getattr(exc, "item", None) or item
        if exc is not None:
            out["reason"] = str(exc)
        return out

    # --- running ---

    def run(self, artifacts: Optional[dict] = None) -> PipelineResult:
        store = dict(artifacts or {})
        for stage in self.stages:
            for name in stage.inputs:
                if name not in store and name not in self.producers:
                    raise GraphError(f"stage {stage.name} needs artifact {name}, which nothing produces")
        result = PipelineResult(status="complete", artifacts=store)
        current: Optional[str] = None
        try:
            for position, stage in enumerate(self.order):
                current = stage.name
                items_left = len(self.order) - position
                if self.ceiling is not None and getattr(self.ceiling, "remaining", 1.0) <= 0:
                    result.status = "stopped"
                    result.stopped = self._stop_report(stage.name, "before stage", items_left, None)
                    self.statuses[stage.name] = "stopped"
                    self._emit("stage", stage=stage.name, state="stopped", attempt=0)
                    break
                outcome = self._run_stage(stage, store, items_left, result)
                if outcome != "ok":
                    self._emit("stage", stage=stage.name, state=outcome, attempt=self.attempts[stage.name])
                    result.status = outcome
                    break
        except Exception as exc:
            # The report has to be able to say where the build died, so the state is written before
            # the exception leaves (D86, section 4 item 18).
            result.status = "crashed"
            result.failed_stage = current
            if current is not None:
                self.statuses[current] = "crashed"
            result.log.append(f"{current}: {type(exc).__name__}: {exc}")
            self._emit("stage", stage=current or "", state="crashed", attempt=0)
            self._finish(result)
            raise
        self._finish(result)
        return result

    def _finish(self, result: PipelineResult) -> None:
        result.statuses, result.attempts = dict(self.statuses), dict(self.attempts)
        self.result = result
        self._write_state(result)
        self._emit("pipeline", state=result.status, failed_stage=result.failed_stage)

    def _run_stage(self, stage: Stage, store: dict, items_left: int, result: PipelineResult) -> str:
        """One node, up to max_attempts times; a failed gate is the rollback edge back into this stage."""
        failure: Optional[str] = None
        if stage.builder and self.anchor is None:
            raise PipelineError(f"stage {stage.name} is a Builder stage and this build has no anchor, so "
                                "it would be built from every Run with nothing held out; choose the anchor "
                                "before the first Builder stage (D81)")
        for attempt in range(1, stage.max_attempts + 1):
            self.attempts[stage.name] = attempt
            self._emit("stage", stage=stage.name, state="start", attempt=attempt)
            inputs = {name: store[name] for name in stage.inputs}
            cache_path = self._cache_path(stage, inputs)
            cached = self._read_cache(cache_path) if attempt == 1 else None
            ctx = StageContext(stage, self.workdir, self.anchor, self._charger(stage.name, items_left),
                               attempt=attempt, failure=failure)
            if cached is not None:
                outputs, status = cached, "cached"
            else:
                try:
                    outputs = stage.fn(ctx, inputs) or {}
                except _budget_types() as exc:
                    self.statuses[stage.name] = "stopped"
                    result.stopped = self._stop_report(stage.name, "", items_left, exc)
                    return "stopped"
                status = "ran" if attempt == 1 else "rolled_back"
            if stage.gate is not None:
                gate_result = stage.gate(ctx, outputs)
                result.gates.append(gate_result)
                self._emit("gate", stage=stage.name, passed=bool(gate_result.passed),
                           failures=list(gate_result.failures), attempt=attempt)
                if not gate_result.passed:
                    failure = "; ".join(gate_result.failures) or "gate failed"
                    self._log_rollback(stage, attempt, gate_result, result)
                    if attempt == stage.max_attempts:
                        self.statuses[stage.name] = "failed"
                        result.failed_stage = stage.name
                        return "failed"
                    continue
            if cached is None:
                self._write_cache(cache_path, stage, outputs)
            store.update(outputs)
            self.statuses[stage.name] = status
            self._emit("stage", stage=stage.name, state=status, attempt=attempt)
            return "ok"
        self.statuses[stage.name] = "failed"
        result.failed_stage = stage.name
        return "failed"

    def _log_rollback(self, stage: Stage, attempt: int, gate: GateResult, result: PipelineResult) -> None:
        """One edge traversal per retry, in the words of design section 4 item 16."""
        name = gate.stage or getattr(stage.gate, "__name__", "gate")
        if attempt < stage.max_attempts:
            label = f"attempt {attempt + 1} of {stage.max_attempts}, gate {name} failed"
            self.rollbacks.setdefault(stage.name, []).append(label)
            result.log.append(f"{stage.name}: {label}")
        else:
            result.log.append(f"{stage.name}: gate {name} failed {stage.max_attempts} times, stage failed")

    def _write_state(self, result: PipelineResult) -> None:
        path = self.workdir / "pipeline" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": result.status, "statuses": result.statuses, "attempts": result.attempts,
            "log": result.log, "gates": [as_dict(g) for g in result.gates], "stopped": result.stopped,
            "failed_stage": result.failed_stage, "mermaid": self.to_mermaid()}, indent=2, default=str),
            encoding="utf-8")

    def to_mermaid(self) -> str:
        """The graph with each stage's status, for the report to embed."""
        lines = ["flowchart TD"]
        lines += [f'    {_node(s.name)}["{_label(s.name)}<br/>{_label(self.statuses[s.name])}"]'
                  for s in self.stages]
        for stage in self.stages:
            for name in stage.inputs:
                if self.producers.get(name):
                    lines.append(f"    {_node(self.producers[name])} --> {_node(stage.name)}")
        for name, labels in self.rollbacks.items():
            lines.append(f'    {_node(name)} -. "{_label(labels[-1])}" .-> {_node(name)}')
        used: dict[str, list[str]] = {}
        for stage in self.stages:
            used.setdefault(self.statuses[stage.name], []).append(_node(stage.name))
        for status, nodes in used.items():
            lines.append(f"    classDef {status} {STATUS_STYLE.get(status, 'stroke:#888')};")
            lines.append(f"    class {','.join(nodes)} {status};")
        return "\n".join(lines)
