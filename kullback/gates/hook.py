"""The `tool_result` hook that runs the path-bound gates over an agent's writes.

`gate_writes(root, workdir)` returns a hook for the core: when the tool was a write or
an edit and its path matches a binding, it runs `rulings_for` and attaches every ruling
to the result. Reads and everything else pass through untouched, and the hook itself
refuses nothing: a refusal rides the result as a ruling the agent reads.

The hook takes callables and imports no agent package: `attach` defaults to the local
`attach_ruling(result, ruling)`, which appends one line per ruling to the content and
one record per ruling (`name`, `accepted`, `line`, `rows`) to `details["rulings"]`.
Pass the core's attach when it exists; it takes the same two arguments and shape.

A failing ruling's line lists every failing row, grouped by the column that differed or the
error class raised, each row naming its Trace, call, arguments and both answers (F43, F53).
`ruled` is the same rulings on a file as it is on disk; the Builder's `rulings` tool calls it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from kullback.gates.bindings import ROWS_FOR, binding_for, load_calls, load_rules, load_schema, rulings_for
from kullback.runner.records import ToolCall

#: The base tools this hook rules on: a write or an edit that matches a binding.
WRITE_TOOLS = ("write", "edit")

#: Where the written path hides in the tool arguments, first hit wins.
PATH_KEYS = ("path", "file", "target", "filename")


#: How many rows a failing ruling lists, and the most characters its text runs to, before "and N more rows".
ROWS_SHOWN = 60
TEXT_CAP = 6000
#: How long one argument set or one value runs in a row line before it is cut.
FIELD_CAP = 160

#: The stages whose rows are one per recorded call, rebuilt whole (past the bindings cap) and given
#: the call's Trace, arguments and the body line the sandbox stood on.
CALL_STAGES = ("replay_fidelity", "executes_on_s0")


def compact(value: Any, cap: int = FIELD_CAP) -> str:
    """A value as compact JSON, cut at `cap` characters."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                               default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= cap else text[:cap - 3] + "..."


def _group_of(row: dict) -> str:
    """What a row is grouped under: its error class where it raised, else the column that differed."""
    exception = row.get("exception")
    if exception:
        return f"raise {str(exception).split(':', 1)[0].strip() or 'an error'}"
    if row.get("column"):
        return f"differ at {row['column']}"
    return "fail"


def _row_text(row: dict) -> str:
    """One failing row: the Trace, the call, the arguments, then ours against recorded or the error."""
    if "call" not in row:
        return compact({key: value for key, value in row.items()}, FIELD_CAP * 2)
    arguments = row.get("arguments", row.get("args"))
    head = " ".join(part for part in (f"trace {row['trace']}" if row.get("trace") else "",
                                      f"call {row.get('call')}",
                                      f"args {compact(arguments)}" if arguments is not None else "") if part)
    if row.get("exception"):
        text = f"{head}: {compact(row['exception'])}"
        return text + (f" at `{compact(row['line'])}`" if row.get("line") else "")
    return f"{head}: ours {compact(row.get('ours'))} recorded {compact(row.get('recorded'))}"


def _row_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """The rows grouped by the differing column or error class, the largest group first."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_of(row), []).append(row)
    return sorted(groups.items(), key=lambda item: -len(item[1]))


def rows_text(rows: Any, shown_failure: str = "") -> list[str]:
    """Every failing row, grouped by the differing column or error class, each group headed by its
    count, capped at ROWS_SHOWN rows and TEXT_CAP characters with the rest counted."""
    rest = sum(int(row.get("rest") or 0) for row in rows if "rest" in row)
    listed = [row for row in rows if "rest" not in row and row.get("failure") != shown_failure]
    lines: list[str] = []
    size, shown = 0, 0
    for group, members in _row_groups(listed):
        header = f"{len(members)} rows {group}:"
        for index, row in enumerate(members):
            text = f"  {_row_text(row)}"
            add = ([header] if index == 0 else []) + [text]
            if shown >= ROWS_SHOWN or size + sum(len(line) + 1 for line in add) > TEXT_CAP:
                break
            lines += add
            size += sum(len(line) + 1 for line in add)
            shown += 1
    more = len(listed) - shown + rest
    if more > 0:
        lines.append(f"and {more} more rows")
    return lines


def ruling_line(ruling: Any) -> str:
    """One ruling as the model reads it: name, pass or fail, first reason, row count, every failing
    row grouped under it (F43, F53), and its note on a line of its own."""
    verdict = "pass" if ruling.passed else "fail"
    line = f"{ruling.stage} {verdict}"
    first = ruling.failures[0] if ruling.failures and not ruling.passed else ""
    if first:
        line += f" ({first})"
    rows = list(getattr(ruling, "rows", None) or [])
    if rows:
        count = len([row for row in rows if "rest" not in row]) + sum(
            int(row.get("rest") or 0) for row in rows if "rest" in row)
        line += f" [{count} rows]"
    if rows and not ruling.passed:
        line = "\n".join([line, *rows_text(rows, first)])
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


def _capturing(execute: Any, seen: dict) -> Any:
    """The executor, keeping its first answer: the recorded calls run once, the rows read them.

    Keywords the caller hands the executor (a trace prefix) pass through, and the wrapper takes
    `**kwargs` so a signature check on it sees that it accepts them."""
    if execute is None:
        return None

    def run(source: Any, calls: Any, db: Any, **kwargs: Any) -> Any:
        results = list(execute(source, calls, db, **kwargs))
        seen.setdefault("results", results)
        return results

    if callable(getattr(execute, "render", None)):
        run.render = execute.render  # type: ignore[attr-defined]
    return run


def _call_rows(ruling: Any, calls: list, results: list, evidence: dict) -> list[dict]:
    """A call ruling's rows rebuilt whole, each with its Trace, its arguments and the body line."""
    try:
        rows = list(ROWS_FOR[ruling.stage](evidence, ruling))
    except Exception:  # noqa: BLE001, a row rebuild never fails a ruling
        return list(ruling.rows or [])
    by_key = {(call.id or index): index for index, call in enumerate(calls)}
    out = []
    for row in rows:
        index = by_key.get(row.get("call"))
        if index is not None:
            call = calls[index]
            answered = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            row = {**row, "arguments": call.args, "trace": call.trace_id}
            if answered.get("line"):
                row["line"] = answered["line"]
        out.append(row)
    return out or list(ruling.rows or [])


def ruled(root: Any, path: str, workdir: Any, *, execute: Any = None,
          evidence: Optional[dict] = None) -> list[Any]:
    """`rulings_for` on the file as it is on disk, with every failing call row named in full.

    The bindings cap the rows a ruling carries; for the per-call stages the rows are rebuilt off
    the same execution, uncapped, and each names its Trace, its arguments and, where the body
    raised, the source line the sandbox stood on. Writes nothing.
    """
    seen: dict = {}
    rulings = rulings_for(root, path, workdir, execute=_capturing(execute, seen), evidence=evidence)
    failing = [ruling for ruling in rulings if not ruling.passed and ruling.stage in CALL_STAGES]
    if not failing or "results" not in seen:
        return rulings
    given = evidence or {}
    calls = [call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
             for call in (given.get("calls") if given.get("calls") is not None
                          else load_calls(Path(workdir), path) or [])]
    extra = {"calls": calls, "_results": seen["results"],
             "schema": given.get("schema") or load_schema(Path(workdir), path),
             "rules": given.get("rules") or load_rules(Path(workdir), path), "readers": given.get("readers")}
    for ruling in failing:
        ruling.rows = _call_rows(ruling, list(calls), seen["results"], extra)
    return rulings


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
        rulings = ruled(root, path, workdir, execute=execute, evidence=evidence)
        if not rulings:
            return None
        for ruling in rulings:
            attach_fn(result, ruling)
        return result

    hook.hook_name = "gate_writes"  # type: ignore[attr-defined]
    return hook


__all__ = ["PATH_KEYS", "WRITE_TOOLS", "attach_ruling", "binding_for", "compact", "gate_writes", "ruled",
           "ruling_line", "rulings_for", "rows_text"]
