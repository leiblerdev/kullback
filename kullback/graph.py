"""The tool-call dependency graph the recordings show, and the walks taken over it (D224).

Every Task the harness holds was derived from a recorded trace, so a thin corpus has as many Tasks
as it has recordings and no more. Two things want more than that: the held-out pool that measures
over-strictness is one Run wide on the Tasks that stood alone (D133, D205), and the difficulty knob
(D209) reports buckets it has nothing to fill. Neither wants a new recording; both want more paths
through the world the recordings already paid for.

This module is the first half of that: the dependency graph, mined, and a walk over it. A node is a
tool. An edge from A to B carries one argument of B and one result path of A whose values matched
inside a Run, which is the same value match row identity is read with (D188), and a second kind of
edge joins a read to a write that touched a row the read touched. An edge's weight is how many Runs
showed it. No edge is invented: a pair of tools the recordings never chained has no edge, and a walk
can never take one.

A walk starts at a tool some Run started with, follows edges with probability by weight, and stops
when it holds the writes and the distinct tools the requested difficulty band asks for. What a walk
is worth is decided elsewhere, by running it: this module proposes, and `synthesise.py` binds the
arguments to rows of the rebuilt world, runs the walk through the Builder's own tool bodies and
throws away the ones a body refuses. Nothing here knows a table, a column or a tool name; an
argument name and a result path are the whole of the vocabulary, so the same graph is mined on any
corpus.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from kullback.examiner.variants import MAX_ID_CHARS, row_ids
from kullback.runner.records import plain
from kullback.sampling import sample_key, sample_unit

# The artifact this module writes and the shape's version, bumped when an edge's fields or the way
# a weight is counted change, so a file written under an older meaning reads as an older file.
FILE_NAME = "graph.json"
FORMAT = 1

# An edge from a result value to the argument that later carried it, and an edge from a read to a
# write on a row the read touched. The two are counted apart because they say different things: the
# first is a value the agent carried forward, the second is an order the world imposed.
VALUE_EDGE = "value"
ROW_EDGE = "row"
KINDS = (VALUE_EDGE, ROW_EDGE)

# How many observed values of one argument the graph keeps, so a walk has something to bind an
# argument no earlier call feeds. They are evidence and not invention: a value no Run ever passed
# to this argument is never bound to it.
MAX_ARG_VALUES = 64
# Where a walk gives up. A band asking for more tools than the graph can reach is answered with a
# shorter walk and a count, never with a loop around one edge.
MAX_STEPS = 12


def is_collection(value: Any) -> bool:
    """A dict keyed by each value's own id, which is a collection of rows and not a set of fields.

    The same home rule `mine.nested_rows` and `synth.py` read: a key that appears as a value inside
    the row it keys is that row's id. A path through such a dict has to collapse the way a list
    does, or every edge mined through it would carry one customer's row id inside the column name
    and could never bind to another row.
    """
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(item, dict) and any(str(field) == str(key) for field in item.values())
               for key, item in value.items())


def _leaves(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Every scalar inside a value as (path, text), a list or a collection collapsed to one path.

    The path is what an edge carries as its column, so two Runs that read the same field of the same
    answer show the same edge however deeply the field sits, however long the list was and whichever
    row of a keyed collection it sat under.
    """
    value = plain(value)
    if is_collection(value):
        return [pair for item in value.values() for pair in _leaves(item, f"{prefix}{{}}")]
    if isinstance(value, dict):
        return [pair for key, item in value.items()
                for pair in _leaves(item, f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(value, (list, tuple)):
        return [pair for item in value for pair in _leaves(item, f"{prefix}[]")]
    if value is None or isinstance(value, bool):
        return []
    text = str(value)
    return [(prefix, text)] if text and len(text) <= MAX_ID_CHARS else []


def _call_name(call: Any) -> str:
    return str((call or {}).get("name") or "")


def _answered(call: dict) -> bool:
    """A call the world answered: a call that errored produced nothing to carry forward (D67)."""
    return not call.get("error")


def result_leaves(call: dict) -> dict[str, str]:
    """The values one call's answer carries, by value, each at the first path it sits under."""
    out: dict[str, str] = {}
    if not _answered(call):
        return out
    for path, text in _leaves(call.get("result")):
        out.setdefault(text, path)
    return out


def values_at(result: Any, column: str) -> list[str]:
    """Every value one answer carries at a column, which a collapsed path can name more than one of."""
    return [text for path, text in _leaves(result) if path == str(column)]


def argument_leaves(call: dict) -> list[tuple[str, str]]:
    """The values one call's arguments carry, as (argument path, value)."""
    return _leaves(call.get("args"))


def _producer(calls: list[dict], upto: int, text: str) -> Optional[tuple[int, str]]:
    """The nearest earlier call whose answer carried this value, and the path it sat under.

    Nearest and not every earlier call: a value four calls carried is one dependency taken four
    times, and crediting all four would weight the graph by how chatty a Run was rather than by
    what it chained.
    """
    for index in range(upto - 1, -1, -1):
        path = result_leaves(calls[index]).get(text)
        if path is not None:
            return index, path
    return None


def _call_rows(call: dict) -> frozenset[str]:
    """The rows one call touched: the ids in its arguments and in the answer it was given."""
    return row_ids(call.get("args")) | (row_ids(call.get("result")) if _answered(call) else frozenset())


def _run_edges(calls: list[dict], write_tools: set[str]) -> set[tuple[str, str, str, str, str]]:
    """The edges one Run shows, each at most once however often the Run took it."""
    edges: set[tuple[str, str, str, str, str]] = set()
    named = [call for call in calls if _call_name(call)]
    for index, call in enumerate(named):
        for arg_path, text in argument_leaves(call):
            found = _producer(named, index, text)
            if found is None:
                continue
            source, column = found
            edges.add((_call_name(named[source]), _call_name(call), arg_path, column, VALUE_EDGE))
        if _call_name(call) not in write_tools:
            continue
        touched = _call_rows(call)
        for source in range(index):
            earlier = named[source]
            if _call_name(earlier) in write_tools or not (touched & _call_rows(earlier)):
                continue
            edges.add((_call_name(earlier), _call_name(call), "", "", ROW_EDGE))
    return edges


def mine(runs: Iterable[Iterable[dict]], write_tools: Iterable[str] = ()) -> dict:
    """The dependency graph over the recordings: nodes, edges with their weights, and the starts.

    `runs` is one list of calls per recorded Run, each call `{name, args, result, error}`, which is
    what `variants.call_results` reads off a Run and what a Trace's own calls already are. A weight
    is a count of Runs and never of calls, so a Run that took one edge ten times weighs it once.
    """
    writes = {str(name) for name in write_tools or ()}
    nodes: Counter = Counter()
    starts: Counter = Counter()
    weights: Counter = Counter()
    arg_values: dict[str, dict[str, list[str]]] = {}
    arg_seen: Counter = Counter()
    total = 0
    for calls in runs or ():
        named = [dict(call) for call in calls or () if _call_name(call)]
        if not named:
            continue
        total += 1
        starts[_call_name(named[0])] += 1
        for name in {_call_name(call) for call in named}:
            nodes[name] += 1
        for call in named:
            name = _call_name(call)
            for arg_path, text in argument_leaves(call):
                arg_seen[(name, arg_path)] += 1
                seen = arg_values.setdefault(name, {}).setdefault(arg_path, [])
                if text not in seen and len(seen) < MAX_ARG_VALUES:
                    seen.append(text)
        for edge in _run_edges(named, writes):
            weights[edge] += 1
    edges = [{"from": a, "to": b, "arg": arg, "column": column, "kind": kind, "weight": int(weight)}
             for (a, b, arg, column, kind), weight in sorted(weights.items())]
    return {
        "format": FORMAT,
        "runs": total,
        "nodes": [{"name": name, "runs": int(count), "write": name in writes,
                   "starts": int(starts.get(name, 0))}
                  for name, count in sorted(nodes.items())],
        "edges": edges,
        "args": {tool: {arg: {"values": values, "seen": int(arg_seen[(tool, arg)])}
                        for arg, values in sorted(by_arg.items())}
                 for tool, by_arg in sorted(arg_values.items())},
    }


def summary(graph: dict) -> dict:
    """The graph's size as a report reads it: no value of the customer's world in it."""
    edges = graph.get("edges") or []
    weights = sorted(int(edge.get("weight") or 0) for edge in edges)
    by_kind = Counter(str(edge.get("kind")) for edge in edges)
    return {"runs": int(graph.get("runs") or 0),
            "nodes": len(graph.get("nodes") or []),
            "edges": len(edges),
            "value_edges": int(by_kind.get(VALUE_EDGE, 0)),
            "row_edges": int(by_kind.get(ROW_EDGE, 0)),
            "starts": sum(1 for node in graph.get("nodes") or [] if node.get("starts")),
            "weight_min": weights[0] if weights else 0,
            "weight_median": weights[len(weights) // 2] if weights else 0,
            "weight_max": weights[-1] if weights else 0}


# --- walking ------------------------------------------------------------------------

def write_names(graph: dict) -> set[str]:
    return {str(node.get("name")) for node in graph.get("nodes") or [] if node.get("write")}


def start_names(graph: dict) -> list[tuple[str, int]]:
    """The tools a recorded Run started with, with how many Runs started there."""
    return sorted(((str(node.get("name")), int(node.get("starts") or 0))
                   for node in graph.get("nodes") or [] if node.get("starts")),
                  key=lambda pair: (-pair[1], pair[0]))


def _by_weight(options: list[tuple[Any, int]], key: str, ident: str) -> Optional[Any]:
    """One option drawn with probability by weight, from a keyed draw and nothing else (D212).

    The draw is a function of the id alone, so the same request under the same seed proposes the
    same walk however many other walks were proposed before it.
    """
    total = sum(max(1, weight) for _option, weight in options)
    if not options or total <= 0:
        return None
    target = sample_unit(key, ident, "") * total
    running = 0.0
    for option, weight in options:
        running += max(1, weight)
        if target < running:
            return option
    return options[-1][0]


def out_edges(graph: dict, tools: Iterable[str]) -> list[dict]:
    """Every edge leaving a tool the walk has already taken."""
    seen = {str(name) for name in tools}
    return [edge for edge in graph.get("edges") or [] if str(edge.get("from")) in seen]


def bands(bucket: str) -> tuple[int, int, int]:
    """The three counts a bucket name asks for: writes, tools touched, paths.

    `w1t2p1` reads as one write, two tools, one path; a band written `3+` asks for its cap, which is
    the least a Task in that band has. An unreadable name asks for nothing, and the caller refuses it.
    """
    text = str(bucket or "")
    out = []
    for letter in ("w", "t", "p"):
        at = text.find(letter)
        if at < 0:
            return (-1, -1, -1)
        digits = ""
        for char in text[at + 1:]:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            return (-1, -1, -1)
        out.append(int(digits))
    return (out[0], out[1], out[2])


def walk(graph: dict, bucket: str, ident: str, *, max_steps: int = MAX_STEPS,
         must_visit: Iterable[str] = ()) -> Optional[list[dict]]:
    """One walk over the graph for a bucket, as the steps a run of it would take.

    Each step names its tool and the bindings its arguments take from earlier steps; a step with no
    binding starts its arguments from the world. The walk prefers an edge into a write while it owes
    writes and an edge into a tool it has not touched while it owes tools, and stops as soon as it
    owes neither. A band the graph cannot reach yields the walk it did reach, which is a shorter
    walk and a count, and never a loop around one edge.

    `must_visit` is what a shaped walk asks for (D225): the tools some archetype's expected effects
    were mapped onto. They are preferred over every other edge while any is still owed, they hold
    the walk open past the band's counts, and a walk that ends without all of them is refused
    outright rather than returned short. The edges are still the recordings' own: a tool the graph
    cannot reach from a start is a tool no shaping can conjure a path to.
    """
    want_writes, want_tools, _paths = bands(bucket)
    if want_writes < 0:
        return None
    writes = write_names(graph)
    owed_tools = [str(name) for name in must_visit or ()]
    starts = start_names(graph)
    first = _by_weight([(name, count) for name, count in starts], "synth-start", ident)
    if first is None:
        return None
    steps = [{"tool": first, "bindings": []}]
    for depth in range(max_steps):
        touched = {step["tool"] for step in steps}
        made = sum(1 for step in steps if step["tool"] in writes)
        owed = [name for name in owed_tools if name not in touched]
        if made >= want_writes and len(touched) >= want_tools and not owed:
            break
        # A band that owes no more writes may not step into one: a walk that overshoots its write
        # count is a walk of another band, and the band is the whole of the request.
        options = [edge for edge in out_edges(graph, touched)
                   if made < want_writes or str(edge.get("to")) in owed
                   or str(edge.get("to")) not in writes]
        if not options:
            break
        wanted = [edge for edge in options
                  if (made < want_writes and str(edge.get("to")) in writes)
                  or (len(touched) < want_tools and str(edge.get("to")) not in touched)]
        owing = [edge for edge in options if str(edge.get("to")) in owed]
        pool = owing or wanted or options
        chosen = _by_weight([(edge, int(edge.get("weight") or 1)) for edge in pool],
                            "synth-edge", f"{ident}:{depth}")
        if chosen is None:
            break
        tool = str(chosen.get("to"))
        bindings = [{"arg": str(edge.get("arg")), "column": str(edge.get("column")),
                     "from": max(index for index, step in enumerate(steps) if step["tool"] == str(edge.get("from")))}
                    for edge in options
                    if str(edge.get("to")) == tool and edge.get("kind") == VALUE_EDGE and edge.get("arg")]
        steps.append({"tool": tool, "bindings": _one_per_arg(bindings, f"{ident}:{depth}")})
    if any(name not in {step["tool"] for step in steps} for name in owed_tools):
        return None
    return steps


def _one_per_arg(bindings: list[dict], ident: str) -> list[dict]:
    """One source per argument: an argument two earlier steps could feed takes the keyed draw."""
    by_arg: dict[str, list[dict]] = {}
    for binding in bindings:
        by_arg.setdefault(binding["arg"], []).append(binding)
    out = []
    for arg, rows in sorted(by_arg.items()):
        rows.sort(key=lambda row: (row["from"], row["column"]))
        out.append(rows[sample_key("synth-binding", f"{ident}:{arg}", "") % len(rows)])
    return out


def reached(steps: Iterable[dict], write_tools: Iterable[str]) -> tuple[int, int]:
    """What a walk actually holds: writes made and distinct tools touched."""
    steps = list(steps or ())
    writes = {str(name) for name in write_tools}
    return (sum(1 for step in steps if str(step.get("tool")) in writes),
            len({str(step.get("tool")) for step in steps}))


__all__ = ["FILE_NAME", "FORMAT", "KINDS", "MAX_ARG_VALUES", "MAX_STEPS", "ROW_EDGE", "VALUE_EDGE",
           "argument_leaves", "bands", "is_collection", "mine", "out_edges", "reached", "result_leaves",
           "start_names", "summary", "values_at", "walk", "write_names"]
