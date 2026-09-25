"""A model drafted reader for a format the harness maps nothing of yet.

A person whose file draws unknown from detect_format gets a reader in
minutes. The model sees the structure-only shape summary plus the two
example adapters, never file values unless the person passes show_values.
Each draft runs the isolated check, and the reader lands in
workdir/sources only after it passes.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from kullback.builder.sources.shape import shape_summary
from kullback.builder.sources.workdir_readers import check_reader_isolated

# The model answers with one fenced block; the first match wins.
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def draft_reader(file_path: str | Path, workdir: str | Path, model: Any, name: str,
                 show_values: bool = False, repairs: int = 2) -> dict:
    """Draft a reader with the model, repair it against the isolated check, save it on a pass.

    The prompt carries shape_summary, the two example adapters and the
    adapter contract in words. A failing draft is sent back with its
    problems listed, at most repairs more calls. The saved file is only
    written on a pass. Answers name, path (or None), passed, problems and
    calls made.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"refusing reader name {name!r}: use letters, numbers, dash or underscore")
    target = Path(file_path)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    summary = shape_summary(target)
    prompt = _prompt(summary, show_values=show_values, file_path=target)
    problems: list[str] = []
    attempts: list[list[str]] = []
    calls = 0
    for _ in range(1 + max(0, repairs)):
        body = _ask(model, prompt if not problems else _repair_prompt(prompt, problems))
        calls += 1
        source = _extract(body)
        if source is None:
            problems = ["the reply holds no fenced python block, so there is no reader to check"]
            attempts.append(list(problems))
            continue
        source = _with_for_file(source, digest)
        candidate = _write_temp(source)
        try:
            problems = check_reader_isolated(candidate, target)
        finally:
            candidate.unlink(missing_ok=True)
        attempts.append(list(problems))
        if not problems:
            saved = _save(workdir, name, source)
            return {"name": name, "path": str(saved), "passed": True,
                    "problems": [], "calls": calls, "attempts": attempts}
    return {"name": name, "path": None, "passed": False, "problems": problems,
            "calls": calls, "attempts": attempts}


def _prompt(summary: dict, show_values: bool = False, file_path: Optional[Path] = None) -> str:
    """The draft prompt: shape lines, example adapters and the contract, no file values."""
    lines = [f"file: {summary.get('file')} jsonl={summary.get('jsonl')} "
             f"records={summary.get('records')}"]
    for path, info in sorted(summary.get("paths", {}).items()):
        line = f"{path}: {','.join(info.get('types', []))} x{info.get('count', 0)}"
        if "min_len" in info:
            line += f" len {info['min_len']}..{info['max_len']}"
        for hint in info.get("hints", []):
            line += f" {hint}"
        if "values" in info:
            line += f" ={'|'.join(info['values'])}"
        lines.append(line)
    examples = _examples()
    contract = _contract()
    body = (contract + "\n\nStructure of the unknown file (paths, types, counts only):\n"
            + "\n".join(lines) + "\n\nClosest examples:\n" + examples
            + "\nReply with one fenced python block holding the whole reader module.")
    if show_values and file_path is not None:
        sample = _sample_values(file_path)
        if sample:
            body += ("\n\nFirst recordings verbatim, sent because the person passed "
                     "show_values:\n" + sample)
    return body


def _repair_prompt(prompt: str, problems: list[str]) -> str:
    """The prompt again with the isolated check problems listed for repair."""
    return (prompt + "\n\nThe last draft failed its checks. Fix every problem below "
            "and reply with one fenced python block holding the whole reader module:\n"
            + "\n".join(f"- {problem}" for problem in problems))


def _contract() -> str:
    """The reader contract in words, the same seam check_reader enforces."""
    return (
        "Write a reader module with a module-level FOR_FILE string holding the sha256 "
        "of the file it was written for, and a module-level ADAPTER object with "
        "name (a short format name), maps True, display, and methods detect, "
        "recordings, to_trace, environment and sidecar. Detect takes (document, "
        "jsonl) and returns (confidence, reasons), voting above zero only on files "
        "ctx.ingest_version and returns a Trace whose turns carry role, content and "
        "a raw pointer into the file, and whose tool calls pair with their results "
        "by id, or by position where neither side carries ids and the counts match. "
        "A call with no mapped result carries no result. Every role word must map "
        "explicitly: a word equal to user, assistant, system or tool maps to itself, "
        "any other word comes from an explicit map the reader carries, and to_trace "
        "raises ValueError naming a word with no mapping. Environment returns a dict "
        "and sidecar returns a dict. Every derived "
        "field cites the raw file through RawPtr, two runs map byte-identical, and "
        "no already mapped fixture file may change hands."
    )


def _examples() -> str:
    """The two example adapters as source, the closest mapped shapes to follow."""
    folder = Path(__file__).resolve().parent
    out = []
    for leaf in ("claude_code_jsonl.py", "otel_genai.py"):
        try:
            out.append(f"--- {leaf} ---\n" + (folder / leaf).read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(out)


def _sample_values(file_path: Path) -> str:
    """The first 3 recordings verbatim, only for the show_values path."""
    from kullback.builder.ingest import _decode

    try:
        payload = file_path.read_bytes()
    except OSError:
        return ""
    try:
        document, _ = _decode(payload)
    except Exception:
        return ""
    if isinstance(document, list):
        sample = document[:3]
    elif isinstance(document, dict):
        sample = document
    else:
        return ""
    try:
        return json.dumps(sample, indent=2, sort_keys=True, default=str)[:8000]
    except (TypeError, ValueError):
        return ""


def _ask(model: Any, prompt: str) -> str:
    """One completion carrying the prompt, returning its text."""
    messages = [{"role": "system", "content": "Write a reader module for the harness intake seam."},
                {"role": "user", "content": prompt}]
    reply = model.query(messages)
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content or "")


def _extract(body: str) -> Optional[str]:
    """The first fenced python block, or None where the reply holds none."""
    match = _FENCE.search(body or "")
    if not match:
        return None
    text = match.group(1).strip()
    return text or None


def _with_for_file(source: str, digest: str) -> str:
    """The source with FOR_FILE set to this file hash, so the check can find its file."""
    updated, count = re.subn(r'^FOR_FILE\s*=\s*["\'][^"\']*["\']',
                             f'FOR_FILE = "{digest}"', source, count=1, flags=re.MULTILINE)
    if count:
        return updated
    return f'FOR_FILE = "{digest}"\n' + source


def _write_temp(source: str) -> Path:
    """One temp file holding a candidate reader, for the isolated check to import."""
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="kullback-draft-",
                                         delete=False, encoding="utf-8")
    try:
        handle.write(source)
    finally:
        handle.close()
    return Path(handle.name)


def _save(workdir: str | Path, name: str, source: str) -> Path:
    """Store a passing reader under workdir/sources, making the folder as needed."""
    folder = Path(workdir) / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.py"
    target.write_text(source, encoding="utf-8")
    return target
