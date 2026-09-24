"""The world clock: the one time a tool body can see, read off the recording (D283).

A body reads the call's time as `self.ctx.now()`. At replay the recording's own time for the Run
is the only answer that can match what the recording wrote, and a live Run of the same Task is
held to the same world, so both are served the time the Task's recording shows. The time comes
from the recording alone: the first write that created a row (an id-keyed value no argument named
and no earlier call showed) and carries exactly one moment that nothing before it showed. A
recording with no such write has no clock, and the context falls back to its seeded draw.

The real clock is never a source. `wall_clock_reads` is the static rule the Builder's gate and
the Runner's loader both apply: a body that imports a clock module or calls a clock reading
other than the context's is refused, with the line that does it.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable, Optional

from kullback.runner.records import Trace

# A moment: a calendar date with a clock time. A bare date is a day a row is about (a departure,
# a birthday), not the time a call happened, so it never sets the clock.
MOMENT = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:?\d{2})?$")

# The modules that read the machine's clock, and the calls that read it off any receiver.
CLOCK_MODULES = frozenset({"datetime", "time", "calendar", "zoneinfo"})
CLOCK_CALLS = frozenset({"now", "today", "utcnow", "time_ns", "monotonic", "monotonic_ns",
                         "perf_counter", "perf_counter_ns", "process_time", "localtime", "gmtime",
                         "ctime", "asctime"})
WORLD_CLOCK = "self.ctx.now()"
# The comment the compile step closes a line of its own with, where it writes code into a tool
# method ahead of the body the model wrote; body line numbers start after such lines.
HARNESS_LINE = "# written by the harness"


def _leaves(node: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(node, dict):
        for name in sorted(node, key=str):
            yield from _leaves(node[name], str(name))
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _leaves(item, key)
    else:
        yield key, node


def _is_id_key(key: str) -> bool:
    name = key.lower()
    return name == "id" or name.endswith("_id")


def _strings(node: Any) -> set[str]:
    return {value for _, value in _leaves(node) if isinstance(value, str)}


def trace_clock(trace: Trace, write_tools: Iterable[str]) -> Optional[str]:
    """The recording's time for this Run: the moment its first row-creating write stamped.

    A write created a row when its result holds an id-keyed value that neither its arguments nor
    any earlier call showed. Its moment is the one moment-shaped value of that result its
    arguments did not name, preferring the one nothing earlier showed: in a world with one clock
    an earlier read shows the same time on a stored row, so a time seen before still counts when
    it is the only one. Two candidates leave the call ambiguous and the next creation is asked.
    """
    writes = set(write_tools)
    seen: set[str] = set()
    for call in trace.tool_calls:
        args, result = call.args or {}, call.result
        own = _strings(args)
        if call.name in writes and call.error is None and isinstance(result, (dict, list)):
            leaves = list(_leaves(result))
            created = any(_is_id_key(key) and isinstance(value, str) and value
                          and value not in own and value not in seen for key, value in leaves)
            moments = {value for _, value in leaves if isinstance(value, str)
                       and MOMENT.match(value.strip()) and value not in own}
            fresh = moments - seen
            if created and len(fresh if fresh else moments) == 1:
                return next(iter(fresh if fresh else moments))
        seen |= own | _strings(result)
    return None


def task_clock(traces: Iterable[Optional[Trace]], write_tools: Iterable[str]) -> Optional[str]:
    """The Task's time: the first of its recordings, in the order given, that shows one."""
    writes = set(write_tools)
    for trace in traces:
        if trace is not None:
            clock = trace_clock(trace, writes)
            if clock is not None:
                return clock
    return None


def start_context(toolkit: Any, seed: int, clock: Optional[str]) -> None:
    """Reseed the toolkit's context for a new Run and set its world clock.

    A toolkit rendered before the clock existed has no `set_clock`; it keeps its seeded draw.
    """
    context = getattr(toolkit, "ctx", None)
    if context is None:
        return
    context.reseed(seed)
    set_clock = getattr(context, "set_clock", None)
    if set_clock is not None:
        set_clock(clock)


# --- the static rule: no body reads the machine's clock ---

def _is_context(node: ast.AST) -> bool:
    """`self.ctx`, the one receiver whose now() is the world clock."""
    return (isinstance(node, ast.Attribute) and node.attr == "ctx"
            and isinstance(node.value, ast.Name) and node.value.id == "self")


def _clock_reading(node: ast.AST) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
        return any(name.split(".")[0] in CLOCK_MODULES for name in names)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if _is_context(node.func.value):
            return False
        return node.func.attr in CLOCK_CALLS
    return False


def _body_start(member: ast.AST, lines: list[str]) -> int:
    """The line the model's body starts on: past the rendered docstring and the harness lines."""
    statements = list(member.body)
    if len(statements) > 1 and isinstance(statements[0], ast.Expr) \
            and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    while len(statements) > 1 and lines[statements[0].lineno - 1].rstrip().endswith(HARNESS_LINE):
        statements = statements[1:]
    return statements[0].lineno


def wall_clock_reads(source: str, class_name: str) -> list[str]:
    """Every line of a tool method that reads the machine's clock, named with its text.

    Lines are counted from the method's first body line, the way the body file shows them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the parses gate rules on that
    lines = source.splitlines()
    out: list[str] = []
    for klass in ast.walk(tree):
        if not (isinstance(klass, ast.ClassDef) and klass.name == class_name):
            continue
        for member in klass.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or not member.body:
                continue
            first = _body_start(member, lines)
            found = sorted({node.lineno for node in ast.walk(member) if _clock_reading(node)})
            out += [f"{member.name} body line {number - first + 1} reads the machine's clock "
                    f"(`{lines[number - 1].strip()}`); read the world clock as {WORLD_CLOCK}"
                    for number in found]
    return out


__all__ = ["CLOCK_CALLS", "CLOCK_MODULES", "HARNESS_LINE", "MOMENT", "WORLD_CLOCK", "start_context",
           "task_clock", "trace_clock", "wall_clock_reads"]
