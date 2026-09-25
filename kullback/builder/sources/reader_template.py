"""A reader module rendered from hand named field paths, with no model involved.

A person names which dotted paths hold the recordings, the role and the
content, plus optionally tool calls and results, in the path syntax
shape.py prints. This module turns that mapping into source text for a
workdir reader (ADAPTER plus FOR_FILE) that maps the file the way the two
example adapters map theirs, with every derived field citing its raw file.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# Mapping keys render_reader understands. Recordings holds one recording
# each, or "." where each line of a JSONL file is one recording. Role and
# content are required. Roles is an optional map of role word to role. The
# rest are optional tool and time paths.
KNOWN_KEYS = (
    "recordings",
    "role",
    "content",
    "roles",
    "tool_calls",
    "tool_name",
    "tool_arguments",
    "tool_call_id",
    "tool_results",
    "tool_result_id",
    "tool_result_content",
    "timestamp",
)

_REQUIRED = ("recordings", "role", "content")

# The four roles a Turn carries. A recorded word equal to one maps to itself.
_ROLES = ("user", "assistant", "system", "tool")

# A reader name is also a file stem under workdir/sources, so it stays narrow.
_NAME = re.compile(r"[A-Za-z0-9_-]+")


def render_reader(name: str, for_file: str, mapping: Mapping[str, Any]) -> str:
    """The source of a reader module for this mapping, defining ADAPTER and FOR_FILE.

    The mapping carries dotted key paths in the syntax shape.py prints, such
    as "items" for the recordings list and "items[].role" for the role
    inside each one, or "." for a JSONL file where each line is one
    recording. Roles optionally maps each other role word to one of the four
    roles; a word with no mapping is refused at map time, never guessed. The
    reader votes positive only where those paths resolve, so ingest never
    picks it for a file of another shape.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"refusing reader name {name!r}: use letters, numbers, dash or underscore")
    missing = [key for key in _REQUIRED if not mapping.get(key)]
    if missing:
        raise ValueError(f"the mapping needs {', '.join(missing)}, so there is no reader to render")
    unknown = sorted(key for key in mapping if key not in KNOWN_KEYS)
    if unknown:
        raise ValueError(f"unknown mapping keys {', '.join(unknown)}, so there is no reader to render")
    for key, value in mapping.items():
        if key == "roles":
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"mapping {key} is not a dotted path, so there is no reader to render")
    roles = _checked_roles(mapping.get("roles"))
    class_name = _class_name(name)
    body = _TEMPLATE.format(
        name=name,
        class_name=class_name,
        for_file=repr(for_file),
        name_repr=repr(name),
        recordings=repr(mapping["recordings"]),
        role=repr(mapping["role"]),
        content=repr(mapping["content"]),
        roles=repr(roles),
        tool_calls=_as_none(mapping.get("tool_calls")),
        tool_name=_as_none(mapping.get("tool_name")),
        tool_arguments=_as_none(mapping.get("tool_arguments")),
        tool_call_id=_as_none(mapping.get("tool_call_id")),
        tool_results=_as_none(mapping.get("tool_results")),
        tool_result_id=_as_none(mapping.get("tool_result_id")),
        tool_result_content=_as_none(mapping.get("tool_result_content")),
        timestamp=_as_none(mapping.get("timestamp")),
    )
    return body


def _checked_roles(roles: Any) -> dict[str, str]:
    """The roles map as a plain dict, refused unless every role is one of the four."""
    if roles is None:
        return {}
    if not isinstance(roles, dict) or not all(
            isinstance(word, str) and isinstance(role, str) for word, role in roles.items()):
        raise ValueError("mapping roles is word to role, so there is no reader to render")
    bad = sorted({role for role in roles.values() if role not in _ROLES})
    if bad:
        raise ValueError(f"roles map to one of {', '.join(_ROLES)}, not {', '.join(bad)}")
    return dict(roles)


def _as_none(value: Any) -> str:
    """One mapping value as Python source: its repr, or None where unset."""
    if value is None:
        return "None"
    if not isinstance(value, str) or not value:
        raise ValueError(f"mapping value {value!r} is not a dotted path, so there is no reader to render")
    return repr(value)


def _class_name(name: str) -> str:
    parts = [chunk for chunk in "".join(char if char.isalnum() else " " for char in name).split() if chunk]
    if not parts:
        return "MappedAdapter"
    text = "".join(chunk[:1].upper() + chunk[1:] for chunk in parts)
    if text[:1].isdigit():
        text = "Reader" + text
    return text + ("" if text.endswith("Adapter") else "Adapter")


_TEMPLATE = '''"""{name} reader, hand mapped from the file structure with no model involved.

The raw file stays the only truth. Every turn and tool call cites where it
was read from through its raw pointer.
"""

from __future__ import annotations

from typing import Any, Iterator

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn

FOR_FILE = {for_file}
NAME = {name_repr}

_RECORDINGS = {recordings}
_ROLE = {role}
_CONTENT = {content}
_ROLE_MAP = {roles}
_TOOL_CALLS = {tool_calls}
_TOOL_NAME = {tool_name}
_TOOL_ARGUMENTS = {tool_arguments}
_TOOL_CALL_ID = {tool_call_id}
_TOOL_RESULTS = {tool_results}
_TOOL_RESULT_ID = {tool_result_id}
_TOOL_RESULT_CONTENT = {tool_result_content}
_TIMESTAMP = {timestamp}


class {class_name}:
    """One recording per entry of the mapped list, one turn per mapped message."""

    name = NAME
    maps = True
    display = NAME

    def detect(self, document: Any, jsonl: bool = False) -> tuple[float, list[str]]:
        found = _probe(document)
        if found:
            return (0.8, [f"carries the mapped {{self.name}} paths: " + ", ".join(found)])
        return (0.0, ["none of the mapped paths resolve on this file"])

    def recordings(self, document: Any) -> Iterator[Any]:
        if _RECORDINGS == ".":
            if isinstance(document, list):
                return iter(document)
            return iter([document])
        holder = _get_one(document, _RECORDINGS)
        if isinstance(holder, list):
            return iter(holder)
        return iter([])

    def environment(self, document: Any) -> dict:
        return {{}}

    def sidecar(self, recording: Any, document: Any) -> dict:
        return {{}}

    def to_trace(self, recording: Any, ctx: Any) -> Trace:
        index = ctx.index
        trace_id = _trace_id(recording, ctx)
        roles = _get_all(recording, _rel(_ROLE))
        texts = _get_all(recording, _rel(_CONTENT))
        count = max(len(roles), len(texts), 1)
        turns: list[Turn] = []
        for position in range(count):
            role = _role(_pick(roles, position, "assistant"))
            content = _text(_pick(texts, position, None))
            here = RawPtr(file_hash=ctx.raw_hash, sim_index=index, msg_index=position)
            turns.append(Turn(idx=position, role=role, content=content, raw_ptr=here))
        calls = _calls(recording, trace_id, ctx)
        ptr = RawPtr(file_hash=ctx.raw_hash, sim_index=index)
        return Trace(
            trace_id=trace_id,
            raw_hash=ctx.raw_hash,
            ingest_version=ctx.ingest_version,
            source=self.name,
            turns=turns,
            tool_calls=calls,
            tools_declared=None,
            system_prompt=None,
            tools_declared_ptr=None,
            system_prompt_ptr=None,
            info_ptr=None,
            raw_ptr=ptr,
        )


def _probe(document: Any) -> list[str]:
    """The mapped paths that resolve with the expected shape, for the detect vote."""
    if _RECORDINGS == ".":
        if not isinstance(document, list) or not document:
            return []
        sample: Any = document[0]
        rel_role, rel_content = _rel(_ROLE), _rel(_CONTENT)
    else:
        holder = _get_one(document, _RECORDINGS)
        if not isinstance(holder, list) or not holder:
            return []
        sample = holder[0]
        rel_role, rel_content = _rel(_ROLE), _rel(_CONTENT)
    found = []
    roles = _get_all(sample, rel_role)
    texts = _get_all(sample, rel_content)
    if roles and all(isinstance(value, str) for value in roles):
        found.append(_ROLE)
    if texts and all(value is None or isinstance(value, (str, int, float, bool, list, dict)) for value in texts):
        found.append(_CONTENT)
    if _ROLE not in found or _CONTENT not in found:
        return []
    return found


def _rel(path: str) -> str:
    """One absolute mapped path as a path inside a single recording."""
    if _RECORDINGS == ".":
        if path == "." or path == "[]":
            return "."
        if path.startswith("[]."):
            return path[3:]
        return path
    if path == _RECORDINGS or path == _RECORDINGS + "[]":
        return "."
    prefix = _RECORDINGS + "[]."
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _split(path: str) -> list[str]:
    """Dotted path steps, keeping a trailing [] on list steps."""
    return path.split(".") if path else []


def _get_one(obj: Any, path: str) -> Any:
    """The single value at a dotted path, or None where nothing resolves."""
    values = _get_all(obj, path)
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _get_all(obj: Any, path: str) -> list:
    """Every value at a dotted path, expanding each [] over its list."""
    if path == "." or not path:
        return [obj]
    current: list[Any] = [obj]
    for step in _split(path):
        listed = step.endswith("[]")
        key = step[:-2] if listed else step
        nxt: list[Any] = []
        for item in current:
            if listed and isinstance(item, list):
                nxt.extend(item)
            elif isinstance(item, dict) and key in item:
                value = item[key]
                if listed and isinstance(value, list):
                    nxt.extend(value)
                else:
                    nxt.append(value)
            elif isinstance(item, list) and not listed:
                for entry in item:
                    if isinstance(entry, dict) and key in entry:
                        nxt.append(entry[key])
        current = nxt
        if not current:
            return []
    return current


def _last(path: str | None) -> str:
    """The final field of a mapped path, with any [] removed."""
    if not path:
        return ""
    return path.split(".")[-1].removesuffix("[]")


def _pick(values: list, position: int, default: Any) -> Any:
    """The value for this turn: the one at its index, or the last one seen."""
    if not values:
        return default
    return values[position] if position < len(values) else values[-1]


def _role(value: Any) -> str:
    """A recorded role word as one of the four roles a Turn carries.

    A word equal to one of the four maps to itself, any other word comes
    from the roles map, and a word with no mapping raises, so the check
    refuses the reader instead of guessing.
    """
    if value in ("user", "assistant", "system", "tool"):
        return value
    if value in _ROLE_MAP:
        return _ROLE_MAP[value]
    raise ValueError(f"unmapped role word {{value!r}}, add it to the roles map")


def _text(value: Any) -> str | None:
    """A recorded content value as text, dumping structures the way adapters do."""
    if value is None or isinstance(value, str):
        return value
    import json as _json

    return _json.dumps(value, ensure_ascii=False, default=str)


def _trace_id(recording: Any, ctx: Any) -> str:
    """The recording id where one is stored, else the hash and index fallback."""
    if isinstance(recording, dict):
        for key in ("id", "trace_id", "run_id"):
            value = recording.get(key)
            if isinstance(value, str) and value:
                return value
    return f"{{ctx.raw_hash[:12]}}-{{ctx.index}}"


def _items(values: list) -> list:
    """One resolved list as its entries: a mapped list reads as its members, not one value."""
    if len(values) == 1 and isinstance(values[0], list):
        return values[0]
    return values


def _calls(recording: Any, trace_id: str, ctx: Any) -> list:
    """The tool calls of one recording, each paired with its result."""
    from kullback.runner.records import RawPtr, ToolCall

    here = RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index, msg_index=0)
    if _TOOL_CALLS:
        objs = _items(_get_all(recording, _rel(_TOOL_CALLS)))
    elif _TOOL_NAME:
        objs = _items(_get_all(recording, _rel(_TOOL_NAME)))
        objs = [{{_last(_TOOL_NAME): value}} if not isinstance(value, dict) else value for value in objs]
    else:
        return []
    results = _result_map(recording)
    ordered = list(results.values())
    calls: list[ToolCall] = []
    for position, obj in enumerate(objs):
        if isinstance(obj, dict):
            name = _field(obj, _TOOL_NAME)
            args = _field(obj, _TOOL_ARGUMENTS)
            call_id = _field(obj, _TOOL_CALL_ID)
        else:
            name, args, call_id = obj, {{}}, None
        call_name = str(name) if name is not None else "tool"
        call_args = dict(args) if isinstance(args, dict) else {{}}
        cid = str(call_id) if call_id is not None else None
        if cid is not None:
            answer = results.get(cid)
        elif len(objs) == len(ordered):
            answer = ordered[position]
        else:
            answer = None
        if answer is not None:
            content, at = answer
            calls.append(ToolCall(id=cid or f"call-{{position}}", name=call_name, args=call_args,
                                  result=content, requestor="assistant", has_result=True,
                                  resolved=True, raw_ptr=here,
                                  result_ptr=RawPtr(file_hash=ctx.raw_hash, sim_index=ctx.index,
                                                    msg_index=at),
                                  trace_id=trace_id))
        else:
            calls.append(ToolCall(id=cid or f"call-{{position}}", name=call_name, args=call_args,
                                  requestor="assistant", has_result=False,
                                  raw_ptr=here, trace_id=trace_id))
    return calls


def _field(obj: dict, path: str | None) -> Any:
    """One call field off its mapped path, reading the final field name."""
    if not path:
        return None
    key = _last(path)
    if key in obj:
        return obj[key]
    found = _get_all(obj, key)
    return found[0] if found else None


def _result_map(recording: Any) -> dict:
    """Tool result id to (content, position), for pairing calls with answers."""
    if _TOOL_RESULTS:
        objs = _items(_get_all(recording, _rel(_TOOL_RESULTS)))
    elif _TOOL_RESULT_CONTENT:
        objs = _items(_get_all(recording, _rel(_TOOL_RESULT_CONTENT)))
        return {{f"result-{{i}}": (value, i) for i, value in enumerate(objs)}}
    else:
        return {{}}
    out: dict[str, tuple[Any, int]] = {{}}
    for position, obj in enumerate(objs):
        if isinstance(obj, dict):
            rid = _field(obj, _TOOL_RESULT_ID)
            content = _field(obj, _TOOL_RESULT_CONTENT)
            if content is None and _TOOL_RESULT_CONTENT is None:
                content = obj
            key = str(rid) if rid is not None else f"result-{{position}}"
            out[key] = (content, position)
        else:
            out[f"result-{{position}}"] = (obj, position)
    return out


ADAPTER = {class_name}()
'''
