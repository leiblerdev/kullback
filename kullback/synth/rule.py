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
from typing import Any, Iterable, Optional

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
# The punctuation stripped at the edges of a word token before two statements are compared.
_EDGE = "".join(chr(c) for c in range(33, 127) if not chr(c).isalnum())


@dataclass
class World:
    """What the generator and the review read of the Environment: the policy, the tools, the Starting state
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
    """Each tool of the Environment: name, read or write, argument names and its description where it has one."""
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


def id_word(text: str, spare: Iterable[str] = ()) -> str:
    """The first id shaped word of a text (a code, a number of five digits, an email, a phone number) that is
    not spared, or nothing. A spared word is a value a bound row holds outside its keys (D400)."""
    kept = {str(word).lower() for word in spare}
    found = [m.group(0) for m in _ID.finditer(text)] + [m.group(0) for m in _EMAIL.finditer(text)]
    found += [m.group(0).strip() for m in _PHONE.finditer(text)
              if sum(ch.isdigit() for ch in m.group(0)) >= PHONE_DIGITS]
    return next((word for word in found if word.lower() not in kept), "")


def statement_refusal(text: str, world: World, *, tools_only: bool = False, spare: Iterable[str] = ()) -> str:
    """Why a statement breaks the rule, as a drop reason, or nothing where it keeps it.

    Two checks only. A tool name (or its words with the underscores read as spaces) is the harness
    speaking; an id shape (a code, an order number, an email, a phone number) ties the statement to one
    row, where a scenario says the kind of thing. A word in `spare` is no id shape (id_word).
    """
    lowered = str(text or "").lower()
    for tool in world.tools:
        name = str(tool.get("name") or "").lower()
        if name and (_named(lowered, name) or ("_" in name and _named(lowered, name.replace("_", " ")))):
            return TOOL_NAME
    if not tools_only and id_word(str(text or ""), spare):
        return ID_SHAPE
    return ""


def _leaves(node: Any) -> Iterable[str]:
    """Every scalar a row holds at any depth, as text; a whole float as its integer."""
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for value in node:
            yield from _leaves(value)
    elif node is not None and not isinstance(node, bool):
        yield str(int(node)) if isinstance(node, float) and node.is_integer() else str(node)


def row_values(world: World, binding: dict) -> set[str]:
    """Every value the bound rows hold outside the world's keys, as words: the words the id rule spares (D400)."""
    keys = {str(key).lower() for rows in world.state.values() if isinstance(rows, dict) for key in rows}
    bound = [world.state.get(str(table), {}).get(str(key)) for table, key in (binding or {}).items()
             if isinstance(world.state.get(str(table)), dict)]
    out = {word for text in _leaves(bound) for word in (text.lower(), *(w.strip(_EDGE).lower() for w in text.split()))}
    return {word for word in out if word and word not in keys}


def nearest_line(text: str, lines: Iterable[str]) -> str:
    """The line most like the text by ROUGE-L, so a refusal can quote what the model nearly copied (D400)."""
    best = max(((rouge_l(text, line), line.strip()) for line in lines if str(line).strip()), default=(0.0, ""))
    return best[1]


def byte_range(text: str, span: str) -> Optional[tuple[int, int]]:
    """The UTF-8 byte range of the span in the page text, or nothing where the page does not hold it."""
    at = str(text).find(str(span)) if span else -1
    if at < 0 or not str(span).strip():
        return None
    start = len(str(text)[:at].encode("utf-8"))
    return start, start + len(str(span).encode("utf-8"))


def _words(text: str) -> list[str]:
    return [word for word in (token.strip(_EDGE) for token in str(text or "").lower().split()) if word]


def _lcs(a: list[str], b: list[str]) -> int:
    """The length of the longest common subsequence of two word lists, one row at a time."""
    row = [0] * (len(b) + 1)
    for word in a:
        last = 0
        for at, other in enumerate(b, start=1):
            last, row[at] = row[at], last + 1 if word == other else max(row[at], row[at - 1])
    return row[-1]


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L of two statements: the F1 of their longest common subsequence over lowercased words.

    Words are split on whitespace with punctuation stripped at their edges; 0.0 when either side is empty.
    """
    left, right = _words(a), _words(b)
    if not left or not right:
        return 0.0
    return 2 * _lcs(left, right) / (len(left) + len(right))


__all__ = ["EMPTY", "ID_SHAPE", "TOOL_NAME", "World", "byte_range", "id_word", "nearest_line", "rouge_l",
           "row_values", "statement_refusal", "tool_rows", "world_of"]
