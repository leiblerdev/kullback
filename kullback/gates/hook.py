"""The `tool_result` hook that runs the path-bound gates over an agent's writes.

`gate_writes(root, workdir)` returns a hook for the core: when the tool was a write or
an edit and its path matches a binding, it runs `rulings_for` and attaches every ruling
to the result. Reads and everything else pass through untouched, and the hook itself
refuses nothing: a refusal rides the result as a ruling the agent reads.

The hook takes callables and imports no agent package: `attach` defaults to the local
`attach_ruling(result, ruling)`, which appends one line per ruling to the content and
one record per ruling (`name`, `accepted`, `line`, `rows`) to `details["rulings"]`.
Pass the core's attach when it exists; it takes the same two arguments and shape.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from kullback.gates.bindings import binding_for, rulings_for

#: The base tools this hook rules on: a write or an edit that matches a binding.
WRITE_TOOLS = ("write", "edit")

#: Where the written path hides in the tool arguments, first hit wins.
PATH_KEYS = ("path", "file", "target", "filename")


def ruling_line(ruling: Any) -> str:
    """One ruling as the model reads it: name, pass or fail, first reason, row count, and its note
    on a line of its own."""
    verdict = "pass" if ruling.passed else "fail"
    line = f"{ruling.stage} {verdict}"
    if ruling.failures and not ruling.passed:
        line += f" ({ruling.failures[0]})"
    if getattr(ruling, "rows", None):
        line += f" [{len(ruling.rows)} rows]"
    if getattr(ruling, "note", ""):
        line += f"\n{ruling.note}"
    return line


def attach_ruling(result: Any, ruling: Any) -> Any:
    """Attach one ruling to a tool result in place: a line on the content, a record
    (`name`, `accepted`, `line`, `rows`) on `details["rulings"]`."""
    line = ruling_line(ruling)
    content = getattr(result, "content", "") or ""
    result.content = f"{content}\n{line}" if content else line
    details = dict(getattr(result, "details", None) or {})
    rulings = list(details.get("rulings") or [])
    rulings.append({"name": ruling.stage, "accepted": bool(ruling.passed), "line": line,
                    "rows": [dict(row) for row in (ruling.rows or [])]})
    result.details = details | {"rulings": rulings}
    return result


def _written_path(call: Any) -> Optional[str]:
    arguments = getattr(call, "arguments", None) or {}
    if not isinstance(arguments, dict):
        return None
    for key in PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def gate_writes(root: Any, workdir: Any, *, attach: Any = None, execute: Any = None,
                evidence: Optional[dict] = None) -> Callable[[Any, Any], Any]:
    """A `tool_result` hook ruling on writes and edits under the bound paths.

    `attach(result, ruling)` defaults to the local `attach_ruling`; `execute` and
    `evidence` pass through to `rulings_for`. The hook returns the (possibly ruled)
    result, or None when the call drew no ruling.
    """
    attach_fn = attach if attach is not None else attach_ruling

    def hook(call: Any, result: Any) -> Any:
        if getattr(call, "name", None) not in WRITE_TOOLS:
            return None
        if result is None or getattr(result, "is_error", False):
            return None
        path = _written_path(call)
        if path is None:
            return None
        rulings = rulings_for(root, path, workdir, execute=execute, evidence=evidence)
        if not rulings:
            return None
        for ruling in rulings:
            attach_fn(result, ruling)
        return result

    hook.hook_name = "gate_writes"  # type: ignore[attr-defined]
    return hook


__all__ = ["PATH_KEYS", "WRITE_TOOLS", "attach_ruling", "gate_writes", "ruling_line", "binding_for",
           "rulings_for"]
