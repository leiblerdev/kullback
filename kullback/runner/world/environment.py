"""A built Environment on disk, workdir or fetched package, read without the Builder (G1).

A workdir holds everything a Run needs: the rendered toolkit under `env/`, the shared Starting
state, one overlay per Task, the Task list, the Verifiers, the canon rules, the Vocabulary and the
Simulated user's rules. A fetched package holds the same shape minus what never leaves the build
machine: no Traces, no Runs, no user rules and no Vocabulary (`kullback.hub.package` names the
allow list). This module reads either shape with the same code and says what is missing, so a Run
from a published package is an honest Run: it has no Simulated user, and its system prompt falls
back to the shipped policy text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback import sampling
from kullback.runner import canon
from kullback.runner.records import EntitySchema, Task, ToolSig, Trace, UserRules, Verifier
from kullback.runner.world import loading
from kullback.runner.world.loading import EnvironmentError, _record, _shaped_json

# Every name the old kullback.episode.environment exported, privates included: the old module
# path is a star-import shim, and only names listed here travel through it.
__all__ = [
    "BuiltEnvironment",
    "EnvironmentError",
    "_field",
    "_json",
    "_text",
    "episode_message",
]

# Where the rendered toolkit lives: the data model plus the tools, as two files whose contents
# join to the single module `loading.load_toolkit` executes. `module_source` renders the model,
# then the toolkit without its file header, so the tools file is the header plus that same tail.
_TOOLKIT_CLASS = "class _ToolKitBase"


class BuiltEnvironment:
    """The files one Run needs, read once and shared by every Episode over them."""

    def __init__(self, root: Any):
        self.root = Path(root)
        self.environment: dict = _shaped_json(self.root / "environment.json", "object", {},
                                                what="environment metadata")
        self.schema = EntitySchema.model_validate(_shaped_json(self.root / "schema.json", "object", {},
                                                               what="an entity schema"))
        self.sigs = [ToolSig.model_validate(s) for s in _shaped_json(self.root / "tool_sigs.json",
                                                                      "array", [], what="tool signatures")]
        self.bodies: dict = _shaped_json(self.root / "bodies.json", "object", {},
                                         what="tool bodies")
        if "bodies" in self.bodies and isinstance(self.bodies["bodies"], dict):
            self.bodies = self.bodies["bodies"]
        self.db: dict = _shaped_json(self.root / "db.json", "object", None, what="a database")
        if self.db is None:
            self.db = _shaped_json(self.root / "env" / "db.json", "object", {}, what="a database")
        self.policy_text: Optional[str] = _text(self.root / "env" / "policy.md")
        self.canon_rules = canon.load_rules(self.root / "canon-rules.json")
        self.vocab = loading._vocab_from(self.root)
        self.env_id: Optional[str] = self.environment.get("env_id")
        self._tasks: Optional[dict[str, Task]] = None
        self._traces: Optional[dict[str, Trace]] = None
        anchor = _shaped_json(self.root / "anchor.json", "object", {}, what="an anchor record")
        held_out = anchor.get("held_out", {})
        if not isinstance(held_out, dict):
            raise EnvironmentError(f"{self.root / 'anchor.json'} holds held_out that is not an object")
        self._held_out = {run_id for runs in held_out.values() for run_id in runs}
        self._replays = _shaped_json(self.root / "replays.json", "object", {},
                                     what="replay records")
        loading._check_replays(self.root / "replays.json", self._replays)

    def _seed_task(self, task: Task) -> Task:
        return task.model_copy(update={"run_ids": [rid for rid in task.run_ids if rid not in self._held_out]})

    def _rules_path(self, run_id: str) -> Path:
        if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
            raise EnvironmentError("recording id must be a single safe path component")
        return self.root / "user_rules" / f"{run_id}.json"

    @property
    def real_tools(self) -> dict[str, dict]:
        """The tools environment.json declares run for real (D262), each declaration naming its tool.

        Empty when the file declares none; a declaration that is not an object is a refusal, not
        a tool silently imitated.
        """
        declared = self.environment.get("real_tools") or {}
        if not isinstance(declared, dict):
            raise EnvironmentError(f"{self.root / 'environment.json'} holds real_tools that is not an object")
        tools = {}
        for name, declaration in declared.items():
            if not isinstance(declaration, dict):
                raise EnvironmentError(f"{self.root / 'environment.json'} declares real tool {name} "
                                       "with a declaration that is not an object")
            tools[name] = {"name": name, **declaration}
        return tools

    def _reference_id(self, task: Task) -> Optional[str]:
        return loading._reference_run_id(task, self._replays.get(task.id), self._held_out,
                                         self.root / "user_rules")

    @property
    def salt(self) -> str:
        """The build salt when the directory stored one, else the sampling default.

        Reading never writes: `sampling.build_salt` would store a salt file, which a read-only
        copy of a workdir must not grow. A package without a salt file draws under the default,
        which only labels the Run, never its behaviour.
        """
        return sampling.read_salt(self.root) or sampling.DEFAULT_SALT

    def task_ids(self) -> list[str]:
        """Every Task id with a Task file, in filename order."""
        return sorted(self.tasks().keys())

    def tasks(self) -> dict[str, Task]:
        """Every Task by id, read once."""
        if self._tasks is None:
            folder = self.root / "tasks"
            if not folder.is_dir():
                raise EnvironmentError(f"{self.root} holds no tasks/ directory")
            self._tasks = {}
            for path in sorted(folder.glob("*.json")):
                if path.name == "tasks.json":
                    continue
                task = _record(path, Task)
                self._tasks[task.id] = task
        return self._tasks

    def task(self, task_id: str) -> Task:
        """One Task, or the refusal naming it."""
        try:
            return self.tasks()[task_id]
        except KeyError:
            raise EnvironmentError(f"{self.root} holds no Task {task_id}") from None

    def traces(self) -> dict[str, Trace]:
        """Every Trace by trace id, or nothing where the directory carries none (a package)."""
        if self._traces is None:
            folder = self.root / "traces"
            self._traces = {}
            for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
                trace = _record(path, Trace)
                self._traces[trace.trace_id] = trace
        return self._traces

    def system_prompt(self, task: Task) -> Optional[str]:
        """The recorded agent's own system prompt, else the shipped policy text (D240).

        A package carries no Traces, so every Task there falls back to the policy text; a workdir
        whose Task names no recorded prompt does the same.
        """
        return loading._system_prompt_for(self._seed_task(task), self.traces(), self.policy_text)

    def tool_specs(self) -> list[dict]:
        """The mined signatures in the shape a model endpoint takes, one per tool."""
        return loading._tool_definitions(self.sigs, self.vocab)

    def toolkit_source(self) -> str:
        """The rendered toolkit as one module, joined from the two files on disk.

        The interface executes what the build wrote, never a re-render: rendering stays in the
        Builder. The join is byte-identical to `module_source` where the files are fresh, which
        the acceptance test checks per workdir before any Run is replayed.
        """
        model_path = self.root / "env" / "data_model.py"
        tools_path = self.root / "env" / "tools.py"
        if not model_path.is_file() or not tools_path.is_file():
            raise EnvironmentError(f"{self.root} holds no rendered toolkit under env/")
        tools_text = tools_path.read_text(encoding="utf-8")
        found = tools_text.find(_TOOLKIT_CLASS)
        if found < 0:
            raise EnvironmentError(f"{tools_path} carries no toolkit base; refusing to guess")
        head = tools_text.rfind("\ntry:", 0, found)
        start = head + 1 if head >= 0 else found
        return model_path.read_text(encoding="utf-8") + "\n\n" + tools_text[start:]

    def overlay(self, task_id: str, trace_id: Optional[str] = None) -> tuple[Any, dict]:
        """The Task's overlay and its row values (D74), on one Run's own layer when named (D213).

        Without a Trace, or with one the Task has no layer for, the Reference Run's layer is served.
        """
        return loading.load_overlay(self.root, task_id, trace_id)

    def rules(self, task: Task) -> Any:
        """The Simulated user's rules for the Task, or None where the directory has none.

        A fetched package never carries user rules, so a Run there has no Simulated user: it ends
        where the Builder's own rule-less Run ends, when the assistant answers without calling.
        """
        run_id = self._reference_id(task)
        if run_id is None:
            return None
        return _record(self._rules_path(run_id), UserRules)

    def members(self, task: Task) -> list[Trace]:
        """The Task's own recordings, which are the evidence its answers are stripped against."""
        traces = self.traces()
        return [traces[run_id] for run_id in self._seed_task(task).run_ids if run_id in traces]

    def reference(self, task: Task) -> Optional[Trace]:
        """The recording the Task's user context comes from: the first of its Runs with rules."""
        run_id = self._reference_id(task)
        return self.traces().get(run_id) if run_id is not None else None

    def verifier(self, task_id: str) -> Optional[Verifier]:
        """The Task's stored Verifier, or None where none was derived."""
        path = self.root / "verifiers" / f"{task_id}.json"
        if not path.is_file():
            return None
        return _record(path, Verifier)

    def write_tools(self) -> set[str]:
        """The tools that change the world, by mined kind, for the Simulated user."""
        return {sig.name for sig in self.sigs if getattr(sig, "kind", None) == "write"}

    def agent_driven(self, task_id: str) -> bool:
        """Whether the build's own ruling gave this Task to the agent user (D214 rule 3).

        The interface drives the rule user only: it has no model to lend the agent user, so a
        Task the agent earned is refused at reset rather than driven by the wrong user.
        """
        # Read straight off the scores file: the world may not import the user extension,
        # and this is the whole of what the ruling needs (kullback.user.fidelity.load_scores
        # reads the same file and keeps only bodies in its own format).
        try:
            body = json.loads((self.root / "user_fidelity.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(body, dict) or body.get("format") != 1:
            return False
        rows = body.get("tasks") or ()
        row = next((r for r in rows if isinstance(r, dict) and r.get("task_id") == task_id), None)
        return bool(row and row.get("drives"))

    def missing_run_inputs(self) -> list[str]:
        """The run-time inputs a published package lacks, in the reviewer's own words.

        `hub.publish.missing_run_inputs` answers for the package files; this answers for what a
        Run reads: the Vocabulary (tool-spec descriptions only), the user rules (the Simulated
        user) and the Traces (the recorded system prompt). Each has a fallback beside it.
        """
        missing = []
        if not (self.root / "vocabulary.json").is_file():
            missing.append("vocabulary.json: tool specs carry no corpus examples")
        if not (self.root / "user_rules").is_dir():
            missing.append("user_rules/: the Run has no Simulated user")
        if not (self.root / "traces").is_dir():
            missing.append("traces/: the system prompt falls back to env/policy.md")
        return missing


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"{path} is unreadable or torn") from exc


def _text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise EnvironmentError(f"{path} is unreadable or torn") from exc


def episode_message(content: Optional[str], calls: Optional[Iterable[Any]]) -> dict:
    """One assistant message in the shape `advance` takes, from plain content and calls.

    The calls stay plain dicts: `advance` reads them off the message and routes each through
    its own shim, and the transcript keeps the shape `ask` appends, so a replayed Run carries
    the same lines the Builder's Run carried.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": _field(c, "id"), "name": _field(c, "name"),
                           "arguments": _field(c, "arguments")}
                          for c in (calls or [])],
    }


def _field(call: Any, name: str) -> Any:
    if isinstance(call, dict):
        return call.get(name)
    return getattr(call, name, None)
