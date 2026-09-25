"""Readers a workdir carries for formats the harness does not map yet.

A workdir/sources/<name>.py file defines a module-level ADAPTER behind the
intake seam plus the FOR_FILE sha256 of the file it was written for. Loading
registers the adapter so ingest and dry runs read the format, but only after
the reader passes its checks against that file. A reader that fails to import,
names no FOR_FILE, has no file to check against, or fails a check is skipped
with its reason, never raised, so one bad reader cannot break ingest of known
formats. No reader is ever registered without a passing check.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Optional

from kullback.builder.sources import (
    MapContext,
    by_name,
    detect_format,
    register,
    registered,
    unregister,
)

# Loaded reader modules by resolved path, so a second load in one process
# reuses the module instead of executing the file again.
_MODULES: dict[str, tuple[float, ModuleType]] = {}


def load_workdir_readers(workdir: str | Path) -> list[str]:
    """Register every workdir reader that passes, returning the names registered."""
    names, _ = _load(Path(workdir))
    return names


def load_with_reasons(workdir: str | Path, candidate: Optional[Path] = None
                      ) -> tuple[list[str], dict[str, str]]:
    """Register every workdir reader that passes, with the skip reason per failure.

    load_workdir_readers is the plain entry point; ingest and dry runs pass the
    file being ingested as candidate so a reader is checked against it on the
    very first ingest, before any raw file is stored.
    """
    return _load(Path(workdir), candidate)


@contextlib.contextmanager
def active(workdir: str | Path, candidate: Optional[Path] = None
           ) -> Iterator[tuple[list[str], dict[str, str]]]:
    """Register the passing workdir readers for one ingest, then restore the registry.

    A long-lived process must not leak one workdir's readers into the next
    file's vote, so on exit every name this call registered is unregistered
    again and any adapter a reader displaced is put back.
    """
    previous = {adapter.name: adapter for adapter in registered()}
    names, skipped = _load(Path(workdir), candidate)
    try:
        yield (names, skipped)
    finally:
        for name in names:
            unregister(name)
        for name, adapter in previous.items():
            if by_name(name) is not adapter:
                register(adapter)


def adapter_from_path(path: str | Path) -> Any:
    """One reader file's ADAPTER, for the check command that points at a file."""
    module = _load_module(Path(path))
    adapter = getattr(module, "ADAPTER", None)
    if adapter is None:
        raise ValueError(f"{path} defines no module-level ADAPTER, so there is no reader to check")
    return adapter


def check_reader(adapter: Any, path: str | Path, fixtures: Optional[list] = None) -> list[str]:
    """The problems that keep a reader out of the registry; empty means it passes.

    The reader's own detect must vote positive on the file, every recording
    must map to a Trace that validates, tool calls must pair with results by
    id, two runs must give byte-identical Trace JSON, and no fixture file that
    already detects may change hands to the new reader.
    """
    from kullback.builder.ingest import INGEST_VERSION, _decode
    from kullback.runner.records import Trace, as_dict

    problems: list[str] = []
    target = Path(path)
    for need in ("detect", "recordings", "to_trace", "environment"):
        if not callable(getattr(adapter, need, None)):
            problems.append(f"the reader has no {need}, so its mapping cannot be checked")
    if problems:
        return problems
    payload = target.read_bytes()
    raw_hash = hashlib.sha256(payload).hexdigest()
    document, jsonl = _decode(payload)
    try:
        confidence, _ = adapter.detect(document, jsonl)
    except Exception as exc:
        return [f"the reader's detect fails on {target.name}: {type(exc).__name__}: {exc}"]
    if not _positive(confidence):
        problems.append(f"the reader's own detect gives no positive vote on {target.name}, "
                        "so ingest would never pick it")
    try:
        recordings = list(adapter.recordings(document))
    except Exception as exc:
        return problems + [f"the reader yields no recordings: {type(exc).__name__}: {exc}"]
    try:
        environment = adapter.environment(document)
    except Exception as exc:
        problems.append(f"the reader's environment fails: {type(exc).__name__}: {exc}")
        environment = {}
    if not recordings:
        problems.append("the reader yields no recordings from the file, so there is nothing to map")
    for index, recording in enumerate(recordings):
        problems.extend(_check_recording(adapter, recording, index, environment, raw_hash,
                                         INGEST_VERSION, Trace, as_dict))
    problems.extend(_fixture_problems(adapter, fixtures))
    return problems


def check_reader_isolated(reader_path: str | Path, file_path: str | Path,
                          timeout: float = 60) -> list[str]:
    """The problems check_reader finds, computed in a child process with no network.

    New readers run before anyone trusts them, and a drafted one is model
    written, so the check itself runs under python -I with a cleared
    environment and the network import block installed first, the way the
    tool sandbox runs a generated body. A timeout or a crash reads back as
    one problem saying so.
    """
    import subprocess
    import sys
    import tempfile

    reader = Path(reader_path)
    target = Path(file_path)
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="kullback-reader-check-") as tmp:
        runner = Path(tmp) / "run_check.py"
        out_path = Path(tmp) / "out.json"
        runner.write_text(_ISOLATED_RUNNER, encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(runner), str(reader), str(target), str(out_path),
                 str(repo)],
                input="", env={}, cwd=str(repo), capture_output=True, text=True,
                timeout=timeout)
        except subprocess.TimeoutExpired:
            return [f"the isolated check timed out after {timeout} seconds, so the reader is refused"]
        if done.returncode != 0 or not out_path.is_file():
            detail = (done.stderr or done.stdout or "").strip()[-400:]
            return [f"the isolated check crashed, so the reader is refused: {detail}"]
        try:
            problems = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [f"the isolated check crashed, so the reader is refused: {exc}"]
        return [str(item) for item in problems] if isinstance(problems, list) else [
            f"the isolated check crashed, so the reader is refused: {problems!r}"]

# The network import block the child installs around the reader import. Socket
# joins the sandbox blocklist because a reader never needs the network at
# all, so even importing it is a refusal.
_ISOLATED_BLOCKED = ("urllib", "urllib3", "requests", "httpx", "ftplib", "smtplib",
                      "telnetlib", "subprocess", "multiprocessing", "webbrowser", "socket")

_ISOLATED_RUNNER = (
    "import importlib.abc\n"
    "import importlib.util\n"
    "import json\n"
    "import sys\n"
    "reader_path, file_arg, out_arg, repo_root = sys.argv[1:5]\n"
    "sys.path.insert(0, repo_root)\n"
    "from kullback.builder.sources.workdir_readers import check_reader\n"
    "from kullback.builder.ingest import INGEST_VERSION, _decode\n"
    "from kullback.runner.records import Trace, as_dict\n"
    "import kullback.ai.provider\n"
    "import socket\n"
    "BLOCKED = " + repr(_ISOLATED_BLOCKED) + "\n"
    "kept = dict(sys.modules)\n"
    "class _NoNetwork(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] in BLOCKED:\n"
    "            raise ImportError('blocked in the reader check: ' + name)\n"
    "        return None\n"
    "sys.meta_path.insert(0, _NoNetwork())\n"
    "for _name in list(sys.modules):\n"
    "    if _name.split('.')[0] in BLOCKED:\n"
    "        del sys.modules[_name]\n"
    "spec = importlib.util.spec_from_file_location('kullback_isolated_reader', reader_path)\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "try:\n"
    "    spec.loader.exec_module(module)\n"
    "except Exception as exc:\n"
    "    json.dump(['the reader does not import: ' + type(exc).__name__ + ': ' + str(exc)], open(out_arg, 'w'))\n"
    "    raise SystemExit(0)\n"
    "finally:\n"
    "    for _name, _mod in kept.items():\n"
    "        sys.modules.setdefault(_name, _mod)\n"
    "adapter = getattr(module, 'ADAPTER', None)\n"
    "if adapter is None:\n"
    "    json.dump(['the reader defines no module-level ADAPTER, so there is no reader to check'], open(out_arg, 'w'))\n"
    "    raise SystemExit(0)\n"
    "def _cut(*args, **kwargs):\n"
    "    raise OSError('the network is blocked in the reader check')\n"
    "socket.socket.connect = socket.socket.connect_ex = socket.socket.bind = _cut\n"
    "socket.create_connection = _cut\n"
    "try:\n"
    "    found = check_reader(adapter, file_arg)\n"
    "except Exception as exc:\n"
    "    found = ['the check itself fails: ' + type(exc).__name__ + ': ' + str(exc)]\n"
    "json.dump(list(found), open(out_arg, 'w'))\n"
)


def _check_recording(adapter: Any, recording: Any, index: int, environment: Any, raw_hash: str,
                     ingest_version: str, trace_model: Any, as_dict: Any) -> list[str]:
    """Map one recording twice: it must validate, repeat byte-identical, and pair by id."""
    problems: list[str] = []
    first = _map_once(adapter, recording, index, environment, raw_hash, ingest_version)
    if isinstance(first, str):
        return [f"recording {index} does not map: {first}"]
    if not isinstance(first, trace_model):
        return [f"recording {index} maps to {type(first).__name__}, not a Trace"]
    try:
        trace_model.model_validate(as_dict(first))
    except Exception as exc:
        return [f"recording {index} maps to a Trace that does not validate: {exc}"]
    second = _map_once(adapter, recording, index, environment, raw_hash, ingest_version)
    if isinstance(second, str):
        return [f"recording {index} maps once then fails: {second}"]
    if json.dumps(as_dict(first), sort_keys=True) != json.dumps(as_dict(second), sort_keys=True):
        problems.append(f"recording {index} maps differently on a second run, "
                        "so the reader is not deterministic")
    seen: set[str] = set()
    for call in first.tool_calls or []:
        call_id = call.id
        if not call_id:
            continue
        if call_id in seen:
            problems.append(f"tool call id {call_id} is issued twice in recording {index}, "
                            "so results cannot pair with calls by id")
        seen.add(call_id)
        paired = bool(call.resolved or call.has_result or call.result is not None
                      or call.error is not None)
        if not paired:
            problems.append(f"tool call {call_id} has no paired tool result in recording {index}")
    return problems


def _map_once(adapter: Any, recording: Any, index: int, environment: Any, raw_hash: str,
              ingest_version: str) -> Any:
    """One mapping attempt, returning the Trace or the failure in words."""
    ctx = MapContext(raw_hash=raw_hash, index=index,
                     environment=environment if isinstance(environment, dict) else {},
                     ingest_version=ingest_version)
    try:
        return adapter.to_trace(recording, ctx)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _positive(confidence: Any) -> bool:
    """A real positive vote, not zero, negative, or a non-number."""
    return (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and confidence > 0)


def _fixture_problems(adapter: Any, fixtures: Optional[list]) -> list[str]:
    """A problem per fixture file that already detects and would change hands."""
    from kullback.builder.ingest import _decode

    files = [Path(item) for item in fixtures] if fixtures is not None else _default_fixtures()
    baselines: list[tuple[Path, Any, bool, str]] = []
    for cand in files:
        try:
            document, jsonl = _decode(cand.read_bytes())
        except (OSError, ValueError):
            continue
        if document is None:
            continue
        try:
            winner = detect_format(document, jsonl).winner
        except Exception:
            continue
        if winner != "unknown":
            baselines.append((cand, document, jsonl, winner))
    if not baselines:
        return []
    name = getattr(adapter, "name", None)
    previous = by_name(name) if name else None
    register(adapter)
    try:
        problems = []
        for cand, document, jsonl, before in baselines:
            try:
                after = detect_format(document, jsonl).winner
            except Exception as exc:
                problems.append(f"the reader breaks detection of {cand.name}: "
                                f"{type(exc).__name__}: {exc}")
                continue
            if after != before:
                problems.append(f"the reader steals {cand.name} from {before}: it now detects "
                                f"as {after}, so a known format would change hands")
        return problems
    finally:
        if previous is None:
            if name:
                unregister(name)
        else:
            register(previous)


def _default_fixtures() -> list[Path]:
    """The repo's own trace fixtures, for the check command that names no list."""
    folder = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
    if not folder.is_dir():
        return []
    return sorted(item for item in folder.glob("*.json") if item.is_file())


def _load(workdir: Path, candidate: Optional[Path] = None) -> tuple[list[str], dict[str, str]]:
    """Import, check, and register every reader file, collecting skip reasons.

    A reader is checked against the candidate when its FOR_FILE is the
    candidate's sha256, else against the stored raw file when the workdir holds
    it; with neither, or with no FOR_FILE at all, the reader is skipped with
    its reason. Nothing registers without a passing check.
    """
    names: list[str] = []
    skipped: dict[str, str] = {}
    folder = workdir / "sources"
    if not folder.is_dir():
        return (names, skipped)
    digest = _candidate_hash(candidate)
    for cand in sorted(folder.glob("*.py")):
        try:
            module = _load_module(cand)
        except Exception as exc:
            skipped[cand.name] = f"the reader does not import: {type(exc).__name__}: {exc}"
            continue
        adapter = getattr(module, "ADAPTER", None)
        if adapter is None:
            continue
        label = str(getattr(adapter, "name", cand.name))
        for_file = getattr(module, "FOR_FILE", None)
        if not isinstance(for_file, str):
            skipped[label] = "the reader names no FOR_FILE, so there is no file to check it against"
            continue
        target: Optional[Path] = None
        if digest is not None and for_file == digest and candidate is not None:
            target = Path(candidate)
        else:
            stored = workdir / "raw" / (for_file + ".json")
            if stored.is_file():
                target = stored
        if target is None:
            skipped[label] = "the file it was written for is not here to check it against"
            continue
        try:
            found = check_reader(adapter, target)
        except Exception as exc:
            found = [f"the check itself fails: {type(exc).__name__}: {exc}"]
        if found:
            skipped[label] = "; ".join(found)
            continue
        register(adapter)
        names.append(label)
    return (names, skipped)


def _candidate_hash(candidate: Optional[Path]) -> Optional[str]:
    """The sha256 of the file being ingested, or None when there is none to check."""
    if candidate is None:
        return None
    try:
        return hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
    except OSError:
        return None


def _load_module(path: Path) -> ModuleType:
    """Execute one reader file under a synthetic name, reusing it while unchanged."""
    resolved = str(path.resolve())
    stamp = path.stat().st_mtime_ns
    cached = _MODULES.get(resolved)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    name = "kullback_workdir_reader_" + hashlib.sha1(resolved.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"the reader at {path} is not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[resolved] = (stamp, module)
    return module
