"""The world a synthetic generator session reads, and the statement rule its tools hold to (D372).

The world is the policy, the tools, the Starting state and the one-line description of the business.
The rule is checked by code on every statement any source writes: a tool name of the Environment (or
its words with the underscores read as spaces) is the harness speaking, and an id shape ties the
statement to one row where a scenario says the kind of thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import read_json
from kullback.spec import intent_tools

# A row value is identifying when it is at least this long and carries a digit, an underscore, an
# at sign, or two words with a capital: an id, a code, an address, a name. A plain word is not.
MIN_VALUE_CHARS = 4
# How many identifying values one Starting state yields to the check, at most.
VALUE_CAP = 50000

TOOL_NAME = "tool_name"
ID_SHAPE = "id_shape"
ROW_VALUE = "row_value"
EMPTY = "empty_statement"

# A letter-and-digit token of four characters or more, a run of five digits, or a number after a hash.
_ID = re.compile(r"#\d+|\b(?=\w*\d)(?=\w*[A-Za-z])\w{4,}\b|\b\d{5,}\b")


@dataclass
class World:
    """What the miner and the review read of the Environment: the policy, the tools, the Starting state
    and the one-line description of the business."""

    workdir: Path
    policy: str = ""
    tools: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    description: str = ""
    values: list[str] = field(default_factory=list)


def _policy(workdir: Path) -> str:
    path = workdir / "env" / "policy.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    for trace in intent_tools.trace_index(workdir).values():
        if trace.system_prompt:
            return str(trace.system_prompt)
    return ""


def tool_rows(workdir: Path) -> list[dict]:
    """Each mined tool: name, read or write, argument names and its description where it has one."""
    sigs = read_json(workdir / "tool_sigs.json", None) or []
    sigs = sigs.get("sigs", []) if isinstance(sigs, dict) else sigs
    out = []
    for sig in sigs if isinstance(sigs, list) else []:
        if isinstance(sig, dict) and sig.get("name"):
            out.append({"name": str(sig["name"]), "kind": sig.get("kind", "read"),
                        "args": sorted(str(f.get("name")) for f in sig.get("args_fields") or []
                                       if isinstance(f, dict)),
                        "description": sig.get("description")})
    return out


def _state(workdir: Path) -> dict:
    for path in (workdir / "db.json", workdir / "env" / "db.json"):
        body = read_json(path, None)
        if isinstance(body, dict):
            return body
    return {}


def identifying(value: Any) -> bool:
    text = str(value)
    if len(text) < MIN_VALUE_CHARS:
        return False
    return (any(ch.isdigit() for ch in text) or "_" in text or "@" in text
            or (" " in text.strip() and any(ch.isupper() for ch in text)))


def _leaves(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value)
    elif isinstance(node, str):
        yield node


def row_values(state: dict, cap: int = VALUE_CAP) -> list[str]:
    """The identifying row keys and string values of the Starting state, lowercased, at most `cap`."""
    found: set[str] = set()
    for rows in state.values():
        if not isinstance(rows, dict):
            continue
        for key, row in rows.items():
            for text in (str(key), *_leaves(row)):
                if identifying(text):
                    found.add(text.strip().lower())
            if len(found) >= cap:
                return sorted(found)
    return sorted(found)


def world_of(workdir: Any, description: str = "") -> World:
    root = Path(workdir)
    state = _state(root)
    meta = read_json(root / "environment.json", {}) or {}
    line = description or (str(meta.get("description") or "") if isinstance(meta, dict) else "")
    return World(workdir=root, policy=_policy(root), tools=tool_rows(root), state=state,
                 description=line, values=row_values(state))


def _named(lowered: str, needle: str) -> bool:
    return needle in lowered and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lowered) is not None


def statement_refusal(text: str, world: World, *, tools_only: bool = False) -> str:
    """Why a statement breaks the rule, as a drop reason, or nothing where it keeps it.

    A tool name (or its words with the underscores read as spaces) is the harness speaking; an id shape
    or an identifying row value ties the statement to one row, where a scenario says the kind of thing.
    """
    lowered = str(text or "").lower()
    for tool in world.tools:
        name = str(tool.get("name") or "").lower()
        if name and (_named(lowered, name) or ("_" in name and _named(lowered, name.replace("_", " ")))):
            return TOOL_NAME
    if tools_only:
        return ""
    if _ID.search(str(text or "")):
        return ID_SHAPE
    if any(_named(lowered, value) for value in world.values):
        return ROW_VALUE
    return ""


def byte_range(text: str, span: str) -> Optional[tuple[int, int]]:
    """The UTF-8 byte range of the span in the page text, or nothing where the page does not hold it."""
    at = str(text).find(str(span)) if span else -1
    if at < 0 or not str(span).strip():
        return None
    start = len(str(text)[:at].encode("utf-8"))
    return start, start + len(str(span).encode("utf-8"))


__all__ = ["EMPTY", "ID_SHAPE", "ROW_VALUE", "TOOL_NAME", "World", "byte_range", "row_values", "statement_refusal",
           "tool_rows", "world_of"]
