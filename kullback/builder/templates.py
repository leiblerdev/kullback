"""The reader a homed prose result never got, aligned out of the tool's own recorded results.

`compile_env.home_partial_result` places a keyless result on the row the call's own arguments
named, and then asks a reader what columns that result asserts (D176, D180). A result whose reader
answers nothing pins nothing: the sighting is counted as `unread_partial_results` and the seed
world's one placeholder value answers every read of that row in every Run. A tool the readers
stage proposed no reader for, because its requestor is the assistant and D176 proposes only for
another requestor's toolkit, leaves every Task that reads it unpinned, and until now nothing in the
build noticed.

This module closes that gap without a model wherever the recording is enough to close it, in two
steps.

Step one is code alone, and it is D183's render idea run backwards. The results of one tool are
grouped by how many tokens they hold, and inside a group the positions every result agrees on are
the template and the positions that vary are its slots. Each slot's values are then matched, result
by result, against the values the corpus holds for the row that result was homed on: exactly, and
then through `canon.canonicalize`, which is the equality the whole harness compares by, so a date
written two ways or an amount with a trailing zero still matches. A slot whose values match one
column of the homed row on every result of the template is bound to that column; a slot that
matches none, or more than one, is left unbound and counted. A template every slot of which is
bound compiles to a reader, `def read(result)`, and a render, `def render(row)`, registered exactly
as a proposed pair is and held to the same round trip: `render` of the row the corpus held when a
result was answered has to be that result again, character for character. Nothing else is asked of
it, and a pair that cannot write its own results back is dropped rather than trusted.

Step two is one model call for a tool step one could not close: a template with an unbound slot, or
results too few to align at all. The call carries the templates, the slot values masked to their
shapes, and the columns of the homed row as evidence, and its reply goes through the same round
trip. It is capped at one call per tool per stage run, so a corpus of many prose tools costs a
bounded number of calls once and never again, since the stage is cached on the results it saw.

What is still unread after both steps is counted per tool and named by its masked shape, so the
Examiner can file it and a person can read it, rather than the build silently pinning nothing.

Nothing here names a domain, a tool, a column or a value: the template is read off the corpus and
the code is the shape of a function over a string.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.builder import readers
from kullback.builder.mine import _reply_json, is_assistant_call
from kullback.builder.sandbox import parse_result
from kullback.runner.canon import canonicalize
from kullback.runner.records import EntitySchema, RawPtr, ToolCall, Trace

MIN_RESULTS = 3  # distinct results of one token width, below which a template is not aligned
MIN_LITERAL_SHARE = 0.25  # of the mean result length, so a template is a sentence and not a comb
MIN_LITERAL_CHARS = 4  # below which the literal text names nothing, whatever share of the result it is
MAX_SLOTS = 16
MAX_TEMPLATES = 24  # per tool, commonest first
MAX_RESULTS = 400  # distinct results of one tool the alignment walks
MAX_FORCED = 20  # tools a stage run may ask the model about, so the spend of step two is bounded
MAX_COLUMNS_SHOWN = 40
MAX_UNREAD_SHOWN = 6
MAX_REVEALED = 3
MAX_SAMPLES = 8

_PTR = RawPtr(file_hash="templates")  # every ToolCall cites a raw location (D66); none of these are recorded
_SPLIT = re.compile(r"(\s+)")


# --- alignment --------------------------------------------------------------


def tokenise(text: str) -> list[str]:
    """One result as words and the whitespace between them, so joining the pieces is the text again.

    Splitting on whitespace and keeping it is what lets a template be compared position by position
    and rendered back exactly: a line break the recording made is a token of the template like any
    word, not something the render has to guess at.
    """
    return _SPLIT.split(text)


def _common_prefix(values: list[str]) -> str:
    first = values[0]
    for index, letter in enumerate(first):
        if any(len(other) <= index or other[index] != letter for other in values):
            return first[:index]
    return first


def _affixes(values: list[str]) -> tuple[str, str]:
    """The head and tail every one of these values shares, leaving each middle non-empty.

    A varying token is rarely a column value on its own: a status word carries the sentence's full
    stop, an identifier carries the word that labels it. The shared head and tail belong to the
    template and what is left is the value, which is what a column can be matched against.
    """
    head = _common_prefix(values)
    while head and any(len(value) <= len(head) for value in values):
        head = head[:-1]
    rest = [value[len(head):] for value in values]
    tail = _common_prefix([value[::-1] for value in rest])[::-1]
    while tail and any(len(value) <= len(tail) for value in rest):
        tail = tail[1:]
    return head, tail


@dataclass
class Template:
    """One aligned template of one tool: literal text with a slot between each pair of segments.

    `frame` is the literal text the results agree on, one piece more than there are slots, and
    `raw` holds the whole varying token each result carried at each slot. A varying token is not
    always a column value on its own: an identifier may be written with the word that labels it, a
    status word with the sentence's full stop. `heads` and `tails` are what every value at a slot
    shares, and `trimmed` says, per slot, whether those affixes were handed to the template so the
    slot holds the middle alone. Which of the two a slot is read as is not decided here: it is
    decided by which of them matches a column of the row (`bind`), because the whole point of a
    slot is to be a column's value.

    `segments` and `values` are the template as the generated code sees it, so the result is
    `segments[0] + value0 + segments[1] + ...` and a render that fills the slots writes the
    recorded result back exactly.
    """
    frame: list[str]
    heads: list[str] = field(default_factory=list)
    tails: list[str] = field(default_factory=list)
    raw: list[list[str]] = field(default_factory=list)
    trimmed: list[bool] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    columns: list[Optional[str]] = field(default_factory=list)

    @property
    def slots(self) -> int:
        return len(self.raw)

    @property
    def values(self) -> list[list[str]]:
        return [self.slot_values(index) for index in range(self.slots)]

    def slot_values(self, index: int, trimmed: Optional[bool] = None) -> list[str]:
        """What this slot holds in each result, as a whole token or as its middle."""
        if trimmed is None:
            trimmed = self.trimmed[index] if index < len(self.trimmed) else False
        if not trimmed:
            return list(self.raw[index])
        head, tail = self.heads[index], self.tails[index]
        return [value[len(head):len(value) - len(tail)] if tail else value[len(head):]
                for value in self.raw[index]]

    @property
    def segments(self) -> list[str]:
        out = []
        current = self.frame[0]
        for index in range(self.slots):
            trimmed = self.trimmed[index] if index < len(self.trimmed) else False
            out.append(current + (self.heads[index] if trimmed else ""))
            current = (self.tails[index] if trimmed else "") + self.frame[index + 1]
        out.append(current)
        return out

    @property
    def bare(self) -> bool:
        """Whether the result is the value and nothing else: one slot and no literal text at all.

        A tool may answer a bare word, an amount or a date with no sentence around it. There is
        nothing to align there, but there is still a column to find: whatever the row holds that
        every one of those results equals. Such a template matches any string, so it is always
        tried after the ones that have a sentence, and it is only kept when its one slot binds and
        the round trip holds over every recorded result of the tool.
        """
        return not any(self.frame)

    @property
    def usable(self) -> bool:
        """Whether the segments can split a result: nothing between two slots cannot."""
        return all(segment for segment in self.segments[1:-1])

    @property
    def bound(self) -> bool:
        return bool(self.columns) and all(name for name in self.columns)

    def to_dict(self) -> dict:
        return {"segments": list(self.segments), "results": len(self.results),
                "columns": [name or "" for name in self.columns],
                "shape": readers.mask(self.results[0]) if self.results else ""}


def align(results: list[str]) -> Optional[Template]:
    """The template these results share, or None when they share too little to be one.

    Refused when the results do not tokenise to one width, when nothing varies or too much does,
    or when the literal text is a small enough share of the result that the template would match
    almost anything.
    """
    parts = [tokenise(text) for text in results]
    width = len(parts[0])
    if len(results) < 2 or any(len(row) != width for row in parts):
        return None
    varying = {index for index in range(width) if len({row[index] for row in parts}) > 1}
    if not varying or len(varying) > MAX_SLOTS:
        return None
    frame: list[str] = []
    heads, tails, raw = [], [], []
    current = ""
    for index in range(width):
        if index not in varying:
            current += parts[0][index]
            continue
        here = [row[index] for row in parts]
        head, tail = _affixes(here)
        frame.append(current)
        heads.append(head)
        tails.append(tail)
        raw.append(here)
        current = ""
    frame.append(current)
    literal = sum(len(piece) for piece in frame)
    mean = sum(len(text) for text in results) / len(results)
    # One slot and no literal text at all is the result that is the value itself, which is a real
    # shape and not a template that matches anything by accident; two or more slots with nothing
    # between them is the latter, and is refused.
    if not (len(raw) == 1 and not literal) and (
            literal < MIN_LITERAL_CHARS or literal < MIN_LITERAL_SHARE * mean):
        return None
    return Template(frame=frame, heads=heads, tails=tails, raw=raw,
                    trimmed=[False] * len(raw), results=list(results),
                    columns=[None] * len(raw))


def templates_for(results: Iterable[str], min_results: int = MIN_RESULTS
                  ) -> tuple[list[Template], list[str]]:
    """The templates these results align into, and the results no template could be aligned from.

    Results are grouped by token width, which is the coarsest grouping that can be aligned position
    by position; a group under `min_results` distinct results is not aligned, because two results
    agreeing on a word says nothing about whether the word is a value or the sentence.
    """
    by_width: dict[int, list[str]] = {}
    for text in sorted(set(results))[:MAX_RESULTS]:
        by_width.setdefault(len(tokenise(text)), []).append(text)
    made: list[Template] = []
    left: list[str] = []
    for _width, group in sorted(by_width.items()):
        template = align(group) if len(group) >= min_results else None
        if template is None:
            left += group
        else:
            made.append(template)
    made.sort(key=lambda t: (t.bare, -len(t.results), t.segments))
    return made[:MAX_TEMPLATES], left + [text for t in made[MAX_TEMPLATES:] for text in t.results]


# --- what the corpus says the homed row holds -------------------------------


@dataclass
class Homing:
    """One recorded prose result, the row the call named, and the run it was answered in."""
    tool: str
    text: str
    table: str
    row_id: str
    trace: str = ""

    @property
    def row(self) -> tuple[str, str]:
        return (self.table, self.row_id)


def homed_prose(traces: Iterable[Trace], schema: EntitySchema,
                read_result: Optional[Callable[[str, Any], Any]] = None) -> list[Homing]:
    """Every keyless prose result the corpus homes on a row and no reader of this build reads.

    The walk is `_observations`' own: an assistant call whose result states no row of its own is
    still about a row when the call named one. A result some reader already answers columns for is
    not here, so a tool whose proposed reader covers half its shapes is derived for the other half
    and left alone on the first.
    """
    from kullback.builder.compile_env import home_partial_result, walk_result_rows

    out: list[Homing] = []
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None or not is_assistant_call(call):
                continue
            result = parse_result(call.result)
            if not isinstance(result, str) or not result.strip():
                continue
            if walk_result_rows(schema, result, call.args)[0]:
                continue
            read = read_result(call.name, result) if read_result is not None else None
            if isinstance(read, dict) and read:
                continue
            homed = home_partial_result(schema, call.name, result, call.args, None)
            if homed is not None:
                out.append(Homing(tool=call.name, text=result, table=homed.table,
                                  row_id=homed.row_id, trace=trace.trace_id))
    return out


def corpus_rows(traces: Iterable[Trace], schema: EntitySchema) -> dict[tuple[str, str], dict[str, list]]:
    """Per row the corpus states anywhere, the values each of its columns was seen holding.

    This is the evidence a slot is matched against: what the homed row holds in the world the
    recording describes, taken from every result of every tool that stated that row, whichever tool
    that was. A column seen in two values keeps both, because the reading a slot matches may be
    either of them.
    """
    from kullback.builder.compile_env import extract_rows

    out: dict[tuple[str, str], dict[str, list]] = {}
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None or not is_assistant_call(call):
                continue
            for table, row_id, row in extract_rows(schema, parse_result(call.result), call.args):
                held = out.setdefault((table, row_id), {})
                for name, value in (row or {}).items():
                    seen = held.setdefault(str(name), [])
                    if value is not None and value not in seen:
                        seen.append(value)
    return out


def _matches(value: str, seen: Iterable[Any]) -> bool:
    """Whether a slot's text is one of the values a column was seen holding.

    Exactly first, so a bound slot renders back character for character; then through
    `canon.canonicalize`, which is how the harness compares two values everywhere else, so a date,
    an amount, a boolean or an enum written another way still counts as the same reading.
    """
    seen = list(seen)
    if any(str(item) == value for item in seen):
        return True
    wanted = canonicalize(value)
    return any(canonicalize(item) == wanted for item in seen)


def _candidates(template: Template, index: int, trimmed: bool,
                rows_of: dict[str, list[tuple[str, str]]],
                values: dict[tuple[str, str], dict[str, list]]) -> set:
    """The columns whose value equals this slot's on every result of the template.

    `rows_of` maps a result's text to the rows the corpus homed it on, which is more than one where
    two calls answered the same sentence about two rows: a column has to match on all of them, so a
    coincidence on one call cannot bind a slot.
    """
    found: Optional[set] = None
    for text, value in zip(template.results, template.slot_values(index, trimmed), strict=False):
        for row in rows_of.get(text) or []:
            held = values.get(row) or {}
            here = {name for name, seen in held.items() if _matches(value, seen)}
            found = here if found is None else (found & here)
            if not found:
                return set()
    return found or set()


def bind(template: Template, rows_of: dict[str, list[tuple[str, str]]],
         values: dict[tuple[str, str], dict[str, list]]) -> Template:
    """Bind each slot to the one column of the homed row its values match on every result.

    The whole varying token is tried first and its middle second, so a value written with the word
    that labels it or with the sentence's punctuation still finds its column, and a value that is
    the whole token is not needlessly cut down. A slot matching no column either way, or more than
    one, stays unbound and the tool goes to step two.
    """
    columns: list[Optional[str]] = []
    trimmed: list[bool] = []
    for index in range(template.slots):
        whole = _candidates(template, index, False, rows_of, values)
        cut = _candidates(template, index, True, rows_of, values) if len(whole) != 1 else set()
        chosen, use_middle = (whole, False) if len(whole) == 1 else (cut, True)
        columns.append(sorted(chosen)[0] if len(chosen) == 1 else None)
        trimmed.append(use_middle and len(chosen) == 1)
    template.columns = columns
    template.trimmed = trimmed
    return template


def _duplicates_agree(template: Template) -> bool:
    """Whether two slots bound to one column carry one value, so the reader may write it once."""
    by_column: dict[str, list[list[str]]] = {}
    for name, slot in zip(template.columns, template.values, strict=False):
        by_column.setdefault(str(name), []).append(slot)
    return all(all(slot == group[0] for slot in group) for group in by_column.values())


# --- the code a bound template compiles to ----------------------------------


def _payload(templates: Iterable[Template]) -> str:
    rows = [[list(t.segments), [str(name) for name in t.columns]] for t in templates]
    return json.dumps(rows, sort_keys=False)


def reader_source(templates: Iterable[Template]) -> str:
    """`def read(result)`: the first template that matches the result, as the columns it binds."""
    return (
        "def read(result):\n"
        "    text = result if isinstance(result, str) else str(result)\n"
        f"    templates = {_payload(templates)}\n"
        "    for segments, columns in templates:\n"
        "        head, tail = segments[0], segments[-1]\n"
        "        if not text.startswith(head) or (tail and not text.endswith(tail)):\n"
        "            continue\n"
        "        body = text[len(head):len(text) - len(tail)] if tail else text[len(head):]\n"
        "        values = {}\n"
        "        cursor = 0\n"
        "        ok = True\n"
        "        for index in range(len(columns)):\n"
        "            last = index == len(columns) - 1\n"
        "            stop = len(body) if last else body.find(segments[index + 1], cursor)\n"
        "            if stop <= cursor:\n"
        "                ok = False\n"
        "                break\n"
        "            values[columns[index]] = body[cursor:stop]\n"
        "            cursor = stop if last else stop + len(segments[index + 1])\n"
        "        if ok and cursor == len(body):\n"
        "            return values\n"
        "    return None\n"
    )


def render_source(templates: Iterable[Template]) -> str:
    """`def render(row)`: the result this tool answers when the row stands as it is given."""
    ordered = sorted(templates, key=lambda t: (t.bare, -len(set(t.columns)), t.segments))
    return (
        "def render(row):\n"
        f"    templates = {_payload(ordered)}\n"
        "    for segments, columns in templates:\n"
        "        if any(row.get(name) is None for name in columns):\n"
        "            continue\n"
        "        out = segments[0]\n"
        "        for index in range(len(columns)):\n"
        "            out = out + str(row[columns[index]]) + segments[index + 1]\n"
        "        return out\n"
        "    return ''\n"
    )


# --- the derived pair and its round trip ------------------------------------


@dataclass
class Derived:
    """One tool's derived reader and render, and how the round trip ruled on them."""
    tool: str
    table: str
    reader: str
    render: str
    columns: list[str] = field(default_factory=list)
    # Columns of the homed table the corpus states nowhere else, which this read is what reveals:
    # the column name to the values the reader answered for it, kept so the schema can class it.
    revealed: dict[str, list[str]] = field(default_factory=dict)
    templates: list[dict] = field(default_factory=list)
    round_trips: tuple[int, int] = (0, 0)
    forced: bool = False
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tool": self.tool, "table": self.table, "reader": self.reader,
                "render": self.render, "columns": list(self.columns),
                "revealed": {name: list(seen) for name, seen in self.revealed.items()},
                "templates": list(self.templates), "forced": self.forced,
                "round_trips": list(self.round_trips), "failures": list(self.failures)}

    @classmethod
    def from_dict(cls, body: dict) -> "Derived":
        return cls(tool=str(body.get("tool") or ""), table=str(body.get("table") or ""),
                   reader=str(body.get("reader") or ""), render=str(body.get("render") or ""),
                   columns=[str(name) for name in body.get("columns") or []],
                   revealed={str(name): [str(v) for v in seen or []]
                             for name, seen in (body.get("revealed") or {}).items()},
                   templates=[dict(row) for row in body.get("templates") or []],
                   forced=bool(body.get("forced")),
                   round_trips=tuple(int(n) for n in (body.get("round_trips") or [0, 0]))[:2] or (0, 0),
                   failures=[str(line) for line in body.get("failures") or []])


def derived_from(artifact: Any) -> list[Derived]:
    """The derived pairs the `readers` artifact holds, in tool order; none where there are none."""
    body = artifact if isinstance(artifact, dict) else {}
    return [Derived.from_dict(row) for row in body.get("derived") or [] if isinstance(row, dict)]


def _pair_proposal(pairs: Iterable[tuple[str, str, str]]) -> readers.Proposal:
    """These (tool, reader, render) triples as the object `readers._module` compiles and runs.

    A derived pair is not a proposal of a table: its columns are the homed row's own, already in the
    schema, so the object carries no table and no columns and exists only to be compiled.
    """
    pairs = list(pairs)
    return readers.Proposal(
        requestor="", table="", columns=[],
        readers=[readers.ToolReader(tool=tool, source=reader) for tool, reader, _render in pairs],
        renders=[readers.ToolRender(tool=tool, source=render) for tool, _reader, render in pairs])


def round_trip(tool: str, reader: str, render: str, homings: list[Homing],
               values: dict[tuple[str, str], dict[str, list]], workdir: Path | str
               ) -> tuple[int, int, list[str], dict[str, list[str]]]:
    """Read every recorded result, render the row it came from, and count what came back the same.

    The row a render is asked about is the row the corpus held, not the reader's own answer, so a
    slot bound through canon to a column the recording writes another way reads fine and renders
    wrong, and this is where that is caught. The exception is the point of the whole stage: where a
    reader answers a column the corpus row does not hold, that column is what this read reveals
    about the row (D176), and the render is asked about the row as the read leaves it. A reveal
    cannot be checked against a column that is not there, so it is held to two other things: one row
    answers one value for it within a run, or it is not a column of the row but something that
    varies under it; and it takes more than one value over the corpus, or it asserts nothing. Both
    functions run in the same subprocess sandbox a tool body runs in, never here.
    """
    proposal = _pair_proposal([(tool, reader, render)])
    source, refused = readers._module(proposal)
    if refused:
        return 0, len(homings), refused, {}
    for check in (readers.gate_parses(source), readers.gate_confined(source)):
        if not check.passed:
            return 0, len(homings), [f"the derived module {check.stage}: {line}"
                                     for line in check.failures], {}
    reader_name = readers._method_names(proposal)[tool]
    render_name = readers._render_names(proposal)[tool]
    jobs: dict[tuple[str, str], tuple[str, dict]] = {}
    for homing in homings:
        row = {name: seen[0] for name, seen in (values.get(homing.row) or {}).items() if seen}
        jobs.setdefault((homing.text, readers.canon_text(row)), (homing.text, row))
    ordered = [jobs[key] for key in sorted(jobs)]
    sandbox = Path(workdir) / "readers"
    reads, refusal = readers._run(
        source, [ToolCall(name=reader_name, args={"result": text}, raw_ptr=_PTR)
                 for text, _row in ordered], sandbox)
    if refusal or len(reads) != len(ordered):
        return 0, len(ordered), [refusal or "the derived module answered fewer results than it was "
                                            "asked"], {}
    was_read: list[dict] = []
    for answer in reads:
        value = answer.get("value") if answer.get("ok") else None
        was_read.append({str(name): held for name, held in value.items()}
                        if isinstance(value, dict) else {})
    revealed: dict[str, list[str]] = {}
    for (_text, row), value in zip(ordered, was_read, strict=False):
        for name, held in value.items():
            if name in row or held is None:
                continue
            seen = revealed.setdefault(name, [])
            if str(held) not in seen and len(seen) < MAX_SAMPLES:
                seen.append(str(held))
    failures: dict[str, str] = {}
    if len(revealed) > MAX_REVEALED:
        return 0, len(ordered), [f"{tool}: the reader answers {len(revealed)} columns the homed row "
                                 f"does not hold, which is more than a read reveals about a row"], {}
    for name, seen in revealed.items():
        if len(seen) < 2:
            failures[f"reveal:{name}"] = (f"{tool} column {name!r}: the one value it takes over the "
                                          f"whole corpus says nothing about the row it is laid on")
    by_text = {text: value for (text, _row), value in zip(ordered, was_read, strict=False)}
    held_once: dict[tuple, str] = {}
    for homing in homings:
        for name in revealed:
            held = (by_text.get(homing.text) or {}).get(name)
            if held is None:
                continue
            first = held_once.setdefault((homing.trace, homing.row, name), str(held))
            if first != str(held):
                failures[f"varies:{name}"] = (
                    f"{tool} column {name!r}: one row was answered two values in one run, so what "
                    f"the sentence states varies under the row rather than being a column of it")
    if failures:
        return 0, len(ordered), sorted(failures.values()), {}
    writes, refusal = readers._run(
        source, [ToolCall(name=render_name, args={"row": {**row, **{name: held for name, held
                                                                   in value.items()
                                                                   if name not in row}}},
                          raw_ptr=_PTR)
                 for (_text, row), value in zip(ordered, was_read, strict=False)], sandbox)
    if refusal or len(writes) != len(ordered):
        return 0, len(ordered), [refusal or "the derived module answered fewer results than it was "
                                            "asked"], {}
    matched = 0
    for (text, _row), value, written in zip(ordered, was_read, writes, strict=False):
        shape = readers.mask(text)
        if not value:
            failures.setdefault(shape, f"{tool} shape {shape!r}: the derived reader read nothing out "
                                       f"of it")
            continue
        if not written.get("ok") or written.get("value") != text:
            failures.setdefault(shape, f"{tool} shape {shape!r}: the round trip does not hold from the "
                                       f"row the corpus held when the result was answered")
            continue
        matched += 1
    return matched, len(ordered), sorted(failures.values()), revealed


def derive(traces: Iterable[Trace], schema: EntitySchema, workdir: Path | str,
           read_result: Optional[Callable[[str, Any], Any]] = None,
           min_results: int = MIN_RESULTS) -> tuple[list[Derived], dict]:
    """Step one: a reader and a render for every tool whose homed prose results the corpus can align.

    Returns the pairs the round trip kept and a report of what it could not close: per tool the
    templates whose slots did not all bind, the results too few to align, and the results still
    unread, which is what step two is asked about and what the finding names.
    """
    traces = list(traces)
    homings = homed_prose(traces, schema, read_result)
    values = corpus_rows(traces, schema)
    by_tool: dict[str, list[Homing]] = {}
    for homing in homings:
        by_tool.setdefault(homing.tool, []).append(homing)
    made: list[Derived] = []
    report: dict[str, dict] = {}
    constant: dict[str, int] = {}
    for tool in sorted(by_tool):
        seen = by_tool[tool]
        rows_of: dict[str, list[tuple[str, str]]] = {}
        for homing in seen:
            rows_of.setdefault(homing.text, []).append(homing.row)
        # One string over every recorded call is an acknowledgement, and D183 already ruled on it:
        # it asserts nothing, so there is no column to find and nothing here is a gap. Counted apart
        # from what is unread, because reporting it as unread would send the Builder after a
        # sentence that says nothing about the row.
        if len(rows_of) < 2:
            constant[tool] = len(seen)
            continue
        templates, left = templates_for((h.text for h in seen), min_results=min_results)
        bound, unbound = [], []
        for template in templates:
            bind(template, rows_of, values)
            if template.bound and template.usable and _duplicates_agree(template):
                bound.append(template)
            else:
                unbound.append(template)
        table = sorted({h.table for h in seen})[0]
        entry = {"table": table, "results": len(seen),
                 "unbound": [t.to_dict() for t in unbound],
                 "slots_unbound": sum(1 for t in unbound for name in t.columns if not name),
                 "too_few": sorted(set(left)),
                 "columns": sorted({name for row in values.values() for name in row}
                                   & {name for row_id in {h.row for h in seen}
                                      for name in (values.get(row_id) or {})})}
        if not bound:
            entry["unread"] = sorted({h.text for h in seen})
            report[tool] = entry
            continue
        reader, render = reader_source(bound), render_source(bound)
        covered = [h for h in seen if any(_reads(t, h.text) for t in bound)]
        matched, asked, failures, revealed = round_trip(tool, reader, render, covered, values, workdir)
        if failures or not matched:
            entry["unread"] = sorted({h.text for h in seen})
            entry["failures"] = failures
            report[tool] = entry
            continue
        made.append(Derived(tool=tool, table=table, reader=reader, render=render,
                            columns=sorted({str(name) for t in bound for name in t.columns}),
                            revealed=revealed,
                            templates=[t.to_dict() for t in bound], round_trips=(matched, asked)))
        still = sorted({h.text for h in seen if not any(_reads(t, h.text) for t in bound)})
        if still or unbound:
            entry["unread"] = still
            report[tool] = entry
    return made, {"tools": report, "constant": constant}


def _reads(template: Template, text: str) -> bool:
    """Whether this template matches a result, decided here the way the generated reader decides it."""
    head, tail = template.segments[0], template.segments[-1]
    if not text.startswith(head) or (tail and not text.endswith(tail)):
        return False
    body = text[len(head):len(text) - len(tail)] if tail else text[len(head):]
    segments = template.segments
    cursor = 0
    for index in range(len(template.columns)):
        last = index == len(template.columns) - 1
        stop = len(body) if last else body.find(segments[index + 1], cursor)
        if stop <= cursor:
            return False
        cursor = stop if last else stop + len(segments[index + 1])
    return cursor == len(body)


# --- step two: one forced proposal per tool ---------------------------------

_FORCED_SYSTEM = (
    "A tool of a recording answers with a prose string rather than an object, so nothing reads a "
    "row out of it and the rebuilt world pins nothing when a Run reads it. The row the result is "
    "about is known: the call's own arguments name it. What is not known is which columns of that "
    "row the sentence states.\n\n"
    "You are given, as one JSON object: the templates the recorded results align into, each as the "
    "literal segments of the sentence with a slot between them and the shapes the slot values took "
    "with the digits masked as N; the results that aligned into no template, verbatim; and the "
    "columns the row holds in the recording, with the values each was seen holding.\n\n"
    "Propose two functions. A reader, `def read(result):` returning a dict of the column values "
    "that result asserts, or None for a result that asserts none. A render, `def render(row):` "
    "returning the result string the tool answers when the row stands as it is given, the inverse "
    "of the reader: for every recorded result, render of the row the recording held has to be that "
    "result again, character for character, and that round trip is what the proposal is held to.\n\n"
    "A key the reader returns is either one of the columns you were given, or a new name for "
    "something the sentence states about the row that no column you were given holds. The second "
    "is a column this read reveals about the row: name it as the customer's own system would, from "
    "what the sentence says rather than from the tool's name, and the render then takes it off the "
    "row like any other column. Reveal a column only for something that is a standing fact about "
    "that row, the same every time the recording asks about it. A value the sentence states that "
    "follows from columns the row already holds is not a reveal and not a column: compute it in the "
    "render and do not ask for a column for it.\n"
)

_FORCED_SHAPE = (
    "Answer with one JSON object and nothing else:\n"
    '{"reader": "def read(result):\\n    ...", "render": "def render(row):\\n    ..."}\n'
)


def forced_system_prompt() -> str:
    """The stable prefix of every forced call: the same bytes for every tool of every build."""
    return "\n".join([_FORCED_SYSTEM, _FORCED_SHAPE, readers._code_rules()])


def forced_payload(tool: str, entry: dict) -> dict:
    """What one forced proposal is asked from: the templates, the unread results and the columns."""
    templates = [{"segments": row.get("segments"), "slots": [readers.mask(str(name or "?"))
                                                             for name in row.get("columns") or []],
                  "shape": readers.mask(str(row.get("shape") or ""))}
                 for row in entry.get("unbound") or []]
    unread = [readers.mask(text) for text in (entry.get("unread") or [])[:MAX_UNREAD_SHOWN]]
    return {"tool": tool, "templates": templates, "unaligned_results": unread,
            "columns": sorted(entry.get("columns") or [])[:MAX_COLUMNS_SHOWN]}


def force(model: Any, tool: str, entry: dict, homings: list[Homing],
          values: dict[tuple[str, str], dict[str, list]], workdir: Path | str) -> Optional[Derived]:
    """One model call for one tool, gated by the same round trip a derived pair is held to.

    One call and no retry: the loop that repairs a proposal belongs to the readers stage, where a
    whole requestor's world is at stake; here a tool either gets a reader the recording ratifies or
    is counted as unread and filed as a finding, which is the honest answer and costs one call.
    """
    messages = [{"role": "system", "content": forced_system_prompt()},
                {"role": "user", "content": json.dumps(forced_payload(tool, entry),
                                                       default=str, sort_keys=True)}]
    body = _reply_json(model.query(messages))
    reader, render = str(body.get("reader") or ""), str(body.get("render") or "")
    if not reader.strip() or not render.strip():
        return None
    matched, asked, failures, revealed = round_trip(tool, reader, render, homings, values, workdir)
    if failures or not matched:
        return None
    return Derived(tool=tool, table=str(entry.get("table") or ""), reader=reader, render=render,
                   columns=[], revealed=revealed, templates=[], round_trips=(matched, asked),
                   forced=True)


def close_gaps(model: Any, traces: Iterable[Trace], schema: EntitySchema, workdir: Path | str,
               read_result: Optional[Callable[[str, Any], Any]] = None,
               min_results: int = MIN_RESULTS, max_forced: int = MAX_FORCED
               ) -> tuple[list[Derived], dict]:
    """Both steps: derive what the corpus can settle, then ask about one tool at a time for the rest.

    The counts come back as `totals`, which the stage writes into the readers artifact and the round
    reads without a report: `readers_derived`, `readers_forced`, `columns_revealed`,
    `slots_unbound` and `results_unread_by_tool`.
    """
    traces = list(traces)
    made, found = derive(traces, schema, workdir, read_result, min_results=min_results)
    report, constant = found["tools"], found["constant"]
    values = corpus_rows(traces, schema)
    by_tool: dict[str, list[Homing]] = {}
    for homing in homed_prose(traces, schema, read_result):
        by_tool.setdefault(homing.tool, []).append(homing)
    asked = 0
    if model is not None:
        for tool in sorted(report, key=lambda name: (-len(report[name].get("unread") or []), name)):
            if asked >= max_forced or not (report[tool].get("unread") or report[tool].get("unbound")):
                continue
            asked += 1
            unread = set(report[tool].get("unread") or [])
            homings = [h for h in by_tool.get(tool) or [] if not unread or h.text in unread]
            got = force(model, tool, report[tool], homings, values, workdir)
            if got is not None:
                made.append(got)
                report[tool]["unread"] = []
    made.sort(key=lambda d: d.tool)
    for entry in report.values():
        entry["unread_shapes"] = sorted({readers.mask(text) for text in entry.get("unread") or []})
    unread = {tool: len(entry.get("unread") or []) for tool, entry in sorted(report.items())
              if entry.get("unread")}
    totals = {"readers_derived": sum(1 for d in made if not d.forced),
              "readers_forced": sum(1 for d in made if d.forced),
              "forced_calls": asked,
              "columns_revealed": sum(len(d.revealed) for d in made),
              "slots_unbound": sum(int(entry.get("slots_unbound") or 0) for entry in report.values()),
              "results_unread_by_tool": unread,
              "results_unread": sum(unread.values()),
              "results_constant_by_tool": dict(sorted(constant.items()))}
    return made, {"tools": report, "totals": totals}


def apply_revealed(schema: EntitySchema, pairs: Iterable[Derived]) -> EntitySchema:
    """Add every column the derived readers reveal to the homed table, marked by the tool.

    Without this the column is homed on the row and named by nothing: the schema is what the tool
    bodies are written from and what the export shows, so a column a read reveals has to stand in
    it like one a proposal revealed (`readers.apply_to_schema`, D176), classed from the values the
    reader answered and marked `revealed_by` the tool rather than passed off as the customer's own.
    """
    pairs = [pair for pair in pairs if pair.table and pair.revealed]
    if not pairs:
        return schema
    return readers.apply_to_schema(
        schema,
        [readers.Proposal(requestor=pair.tool, table=pair.table, columns=sorted(pair.revealed))
         for pair in pairs],
        {pair.tool: {name: list(seen) for name, seen in pair.revealed.items()} for pair in pairs})


def unread_shapes(report: Any, tool: str, limit: int = MAX_UNREAD_SHOWN) -> list[str]:
    """The masked shapes of the results still unread for one tool, for the finding that names them."""
    entry = ((report or {}).get("tools") or {}).get(tool) or {}
    return sorted({readers.mask(text) for text in entry.get("unread") or []})[:limit]


def derived_reader(artifact: Any, traces: Iterable[Trace], workdir: Path | str
                   ) -> dict[tuple[str, str], dict]:
    """What each derived reader reads, as a table `result_reader` serves alongside the proposed ones.

    Every derived reader of a build runs once here over the distinct results its tool answered, in
    the same subprocess sandbox, and the answers are served from the table; nothing runs a reader in
    this process.
    """
    pairs = derived_from(artifact)
    if not pairs:
        return {}
    traces = list(traces)
    proposal = _pair_proposal([(d.tool, d.reader, d.render) for d in pairs])
    source, refused = readers._module(proposal)
    if refused or not readers.gate_parses(source).passed or not readers.gate_confined(source).passed:
        return {}
    names = readers._method_names(proposal)
    jobs: list[tuple[str, str]] = []
    seen: set = set()
    for trace in traces:
        for call in trace.tool_calls:
            if call.error is not None or call.name not in names:
                continue
            key = (call.name, str(parse_result(call.result)))
            if key not in seen:
                seen.add(key)
                jobs.append(key)
    calls = [ToolCall(name=names[tool], args={"result": text}, raw_ptr=_PTR) for tool, text in jobs]
    results, refusal = readers._run(source, calls, Path(workdir) / "readers")
    if refusal:
        return {}
    out: dict[tuple[str, str], dict] = {}
    for key, result in zip(jobs, results, strict=False):
        value = result.get("value") if result.get("ok") else None
        if isinstance(value, dict) and value:
            out[key] = value
    return out
