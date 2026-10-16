"""The world a synthetic generator session reads, and the statement rule its tools hold to (D372).

The world is the policy, the tools, the Starting state and the one-line description of the business.
The rule is checked by code on every statement any source writes: a tool name of the Environment (or
its words with the underscores read as spaces) is the harness speaking, and an id shape (a code, an order
number, an email, a phone number) ties the statement to one row where a scenario says the kind of thing.
No value of the Starting state is read for the rule; the review still reads rows through its lookups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import read_json
from kullback.spec import intent_tools

TOOL_NAME = "tool_name"
ID_SHAPE = "id_shape"
EMPTY = "empty_statement"

# A letter-and-digit token of four characters or more, a run of five digits, or a number after a hash.
_ID = re.compile(r"#\d+|\b(?=\w*\d)(?=\w*[A-Za-z])\w{4,}\b|\b\d{5,}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# A phone shape: digits with spaces, dots, dashes or brackets between them, seven digits at least.
_PHONE = re.compile(r"\+?\(?\d[\d\s().-]{5,}\d")
PHONE_DIGITS = 7


@dataclass
class World:
    """What the miner and the review read of the Environment: the policy, the tools, the Starting state
    and the one-line description of the business."""

    workdir: Path
    policy: str = ""
    tools: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    description: str = ""


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


def world_of(workdir: Any, description: str = "") -> World:
    root = Path(workdir)
    state = _state(root)
    meta = read_json(root / "environment.json", {}) or {}
    line = description or (str(meta.get("description") or "") if isinstance(meta, dict) else "")
    return World(workdir=root, policy=_policy(root), tools=tool_rows(root), state=state,
                 description=line)


def _named(lowered: str, needle: str) -> bool:
    return needle in lowered and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lowered) is not None


def _id_shaped(text: str) -> bool:
    if _ID.search(text) or _EMAIL.search(text):
        return True
    return any(sum(ch.isdigit() for ch in found.group(0)) >= PHONE_DIGITS for found in _PHONE.finditer(text))


def statement_refusal(text: str, world: World, *, tools_only: bool = False) -> str:
    """Why a statement breaks the rule, as a drop reason, or nothing where it keeps it.

    Two checks only. A tool name (or its words with the underscores read as spaces) is the harness
    speaking; an id shape (a code, an order number, an email, a phone number) ties the statement to one
    row, where a scenario says the kind of thing. Nothing reads the Starting state for this rule.
    """
    lowered = str(text or "").lower()
    for tool in world.tools:
        name = str(tool.get("name") or "").lower()
        if name and (_named(lowered, name) or ("_" in name and _named(lowered, name.replace("_", " ")))):
            return TOOL_NAME
    if not tools_only and _id_shaped(str(text or "")):
        return ID_SHAPE
    return ""


def byte_range(text: str, span: str) -> Optional[tuple[int, int]]:
    """The UTF-8 byte range of the span in the page text, or nothing where the page does not hold it."""
    at = str(text).find(str(span)) if span else -1
    if at < 0 or not str(span).strip():
        return None
    start = len(str(text)[:at].encode("utf-8"))
    return start, start + len(str(span).encode("utf-8"))


__all__ = ["EMPTY", "ID_SHAPE", "TOOL_NAME", "World", "byte_range", "statement_refusal", "tool_rows", "world_of"]
