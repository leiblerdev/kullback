"""Readers a workdir carries for formats the harness does not map yet.

A workdir/sources/<name>.py file defines a module-level ADAPTER behind the
intake seam plus the FOR_FILE sha256 of the raw file it was written for.
Loading registers the adapter so ingest and dry runs read the format, but only
after the reader passes its checks against that raw file when the workdir
stores it. A reader that fails to import or fails a check is skipped with its
reason, never raised, so one bad reader cannot break ingest of known formats.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from kullback.builder.sources import MapContext, by_name, detect_format, register, unregister

# Loaded reader modules by resolved path, so a second load in one process
# reuses the module instead of executing the file again.
_MODULES: dict[str, tuple[float, ModuleType]] = {}


def load_workdir_readers(workdir: str | Path) -> list[str]:
    """Register every workdir reader that passes, returning the names registered."""
    names, _ = _load(Path(workdir))
    return names


def load_with_reasons(workdir: str | Path) -> tuple[list[str], dict[str, str]]:
    """Register every workdir reader that passes, with the skip reason per failure.

    load_workdir_readers is the plain entry point; ingest and dry runs use this
    one so the result can say why a reader was skipped.
    """
    return _load(Path(workdir))


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


def _load(workdir: Path) -> tuple[list[str], dict[str, str]]:
    """Import, check, and register every reader file, collecting skip reasons."""
    names: list[str] = []
    skipped: dict[str, str] = {}
    folder = workdir / "sources"
    if not folder.is_dir():
        return (names, skipped)
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
        stored = workdir / "raw" / (for_file + ".json") if isinstance(for_file, str) else None
        if stored is not None and stored.is_file():
            try:
                found = check_reader(adapter, stored)
            except Exception as exc:
                found = [f"the check itself fails: {type(exc).__name__}: {exc}"]
            if found:
                skipped[label] = "; ".join(found)
                continue
        register(adapter)
        names.append(label)
    return (names, skipped)


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
