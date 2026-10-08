"""The writer's only tools: world search, the policy, and the one tool that adds items; the bounded session.

The world tools (`WORLD_TOOLS`) close over one `items.World` (the Starting state, the policy, the
Intent's facts and the tool list) and nothing else, so a writer session can read no Run, Reference,
Verifier or End state: `refuse_run_inputs` refuses such records before a session starts. `add_items`
keeps an item only when code finds every value it names in the world, the policy or a user turn,
and says why when it does not ("find, do not invent"). The two lookups ported from the intentv
experiment (`READ_TOOLS`) stay for that frozen experiment only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback.spec.items import World

# The most read rounds a writer session may spend before it must write (intentv r1).
MAX_READ_ROUNDS = 4

READ_TOOLS = [
    {"name": "lookup_rows",
     "description": ("Read rows of the Starting state by table and key. Read-only. "
                     "With no key, returns the row count and sample keys."),
     "parameters": {"type": "object",
                    "properties": {"table": {"type": "string"},
                                   "key": {"type": "string"}},
                    "required": ["table"]}},
    {"name": "search_rows",
     "description": ("Find row keys whose dotted field equals a value. Read-only. "
                     "Returns at most 10 keys."),
     "parameters": {"type": "object",
                    "properties": {"table": {"type": "string"},
                                   "field": {"type": "string"},
                                   "value": {"type": "string"}},
                    "required": ["table", "field", "value"]}},
]

# Record types a writer input may never be: each would show a Run, its end, or a Verifier.
_FORBIDDEN_TYPES = ("Run", "Trace", "Verifier", "Atom", "Verdict", "Event")
# Keys that mark a plain dict as a Run, an End state or a Verifier rather than a Starting state
# (list-valued, since a Starting state table is a dict of rows; end_state in any form).
_FORBIDDEN_KEYS = ("events", "end_state", "atoms", "turns", "tool_calls")


class WriterInputError(ValueError):
    """A writer input that would show a Run, a Reference, a Verifier or an End state."""


def refuse_run_inputs(*inputs: Any) -> None:
    """Raises when any input is a Run-side record, or a dict carrying one at its top level."""
    for value in inputs:
        name = type(value).__name__
        if name in _FORBIDDEN_TYPES:
            raise WriterInputError(f"a writer input may not be a {name}")
        if isinstance(value, dict):
            found = sorted(key for key in _FORBIDDEN_KEYS
                           if isinstance(value.get(key), list) or (key == "end_state" and key in value))
            if found:
                raise WriterInputError(f"a writer input may not carry {', '.join(found)}")


def read_tools(state: dict) -> dict[str, Callable[..., str]]:
    """The two lookups, bound to one Starting state."""
    refuse_run_inputs(state)

    def lookup_rows(table: str, key: str = "") -> str:
        rows = state.get(table)
        if not isinstance(rows, dict):
            return "no such table"
        if not key:
            keys = sorted(rows)[:12]
            return json.dumps({"rows": len(rows), "sample_keys": keys})
        row = rows.get(key)
        if row is None:
            return "no such row"
        return json.dumps(row, default=str)[:2000]

    def search_rows(table: str, field: str, value: str) -> str:
        rows = state.get(table)
        if not isinstance(rows, dict):
            return "no such table"
        hits = []
        for row_key, row in rows.items():
            node = row
            for part in field.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if node is not None and str(node) == value:
                hits.append(row_key)
            if len(hits) >= 10:
                break
        return json.dumps(hits)

    return {"lookup_rows": lookup_rows, "search_rows": search_rows}


def _thinking_of(reply: object) -> list[str]:
    """Summarized thinking text off a reply, signatures dropped."""
    out = []
    for block in getattr(reply, "thinking_blocks", None) or []:
        if isinstance(block, dict):
            text = block.get("thinking") or block.get("summary") or block.get("text")
            if text:
                out.append(str(text))
    if getattr(reply, "thinking", None):
        out.append(reply.thinking)
    return out


def _usage_of(reply: object) -> dict:
    usage = getattr(reply, "usage", None)
    get = (lambda k: getattr(usage, k, 0) or 0) if usage is not None else (lambda k: 0)
    return {"input": get("input"), "output": get("output"),
            "cache_read": get("cache_read"), "cache_write": get("cache_write")}


def _assistant_tool_turn(reply: object) -> dict:
    return {"role": "assistant", "content": reply.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                           for c in reply.tool_calls]}


# The most tool rounds an item-writing session may spend: reads and adds share them.
MAX_ITEM_ROUNDS = 14

_OBJECT = {"type": "object"}
WORLD_TOOLS = [
    {"name": "read_schema",
     "description": "The world's tables (row count, fields) and the tools (kind, argument names). Read-only.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "find_rows",
     "description": ("Find rows holding a value: in one field (dotted path) or any field, in one table or "
                     "every table. Read-only. Returns at most 10 {table, key}."),
     "parameters": {"type": "object",
                    "properties": {"value": {"type": "string"}, "table": {"type": "string"},
                                   "field": {"type": "string"}},
                    "required": ["value"]}},
    {"name": "read_row",
     "description": "One row of the world by table and key. Read-only.",
     "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "key": {"type": "string"}},
                    "required": ["table", "key"]}},
    {"name": "read_policy",
     "description": "The policy's section headings, or with section, the text under that heading. Read-only.",
     "parameters": {"type": "object", "properties": {"section": {"type": "string"}}}},
    {"name": "add_items",
     "description": ("Add Verifier items. Each value must be found in the world, the policy or a user turn; "
                     "a refused item comes back with the reason, fix it and add it again under its id."),
     "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": _OBJECT}},
                    "required": ["items"]}},
    {"name": "drop_items",
     "description": "Remove items you added, by id.",
     "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ids"]}},
]


class ItemStore:
    """The items a session has kept, by id, and every refusal it made, each with its reason.

    `ruled` limits drops and replacements to the ruled items (a repair), and `max_adds` the new ids it
    may add; None lets the writer drop or replace any item it added.
    """

    def __init__(self, world: World, facts: dict, unwitnessed: Callable[[Any], bool] = lambda fact: False,
                 items: Iterable[dict] = (), ruled: Optional[set[str]] = None, max_adds: Optional[int] = None):
        self.world, self.facts, self.unwitnessed = world, facts, unwitnessed
        self.items: dict[str, dict] = {item["id"]: item for item in items}
        self.held = set(self.items)
        self.ruled, self.max_adds = ruled, max_adds
        self.refused: list[dict] = []
        self.dropped: list[str] = []
        self.offered = 0
        self.not_ruled = 0

    def _limit(self, item_id: str) -> str:
        """Why a repair may not add this id: an unruled item it held, or past the add cap; else empty."""
        if self.ruled is None:
            return ""
        if item_id in self.held and item_id not in self.ruled:
            self.not_ruled += 1
            return "not ruled: a repair may not replace an item the ruling does not name"
        added = {i for i in self.items if i not in self.held}
        if self.max_adds is not None and item_id not in self.held and item_id not in added \
                and len(added) >= self.max_adds:
            return f"at most {self.max_adds} items may be added in one repair"
        return ""

    def add(self, items: Any) -> str:
        if not isinstance(items, list):
            return "items is a list of item objects"
        lines = []
        for item in items:
            self.offered += 1
            item_id = str(item.get("id")) if isinstance(item, dict) else "?"
            limit = self._limit(item_id)
            kept, why = (None, limit) if limit else self.world.check(item, self.facts, self.unwitnessed)
            if kept is None:
                self.refused.append({"id": item_id, "kind": item.get("kind") if isinstance(item, dict) else None,
                                     "reason": why})
                lines.append(f"{item_id}: refused: {why}")
                continue
            self.items[kept["id"]] = kept
            lines.append(f"{item_id}: kept" + (f" (row {kept['row']})" if "row" in kept else ""))
        return "\n".join(lines) or "no items"

    def drop(self, ids: Any) -> str:
        out = []
        for item_id in [str(i) for i in ids or []]:
            if self.ruled is not None and item_id not in self.ruled and item_id in self.held:
                self.not_ruled += 1
                out.append(f"{item_id}: not ruled, kept")
            elif self.items.pop(item_id, None) is not None:
                self.dropped.append(item_id)
                out.append(f"{item_id}: dropped")
            else:
                out.append(f"{item_id}: no such item")
        return "\n".join(out) or "no ids"


def world_tools(store: ItemStore) -> dict[str, Callable[..., str]]:
    """The world tools bound to one store and its world."""
    world = store.world

    def read_schema() -> str:
        return json.dumps(world.schema())[:6000]

    def find_rows(value: str, table: str = "", field: str = "") -> str:
        if table and table not in world.state:
            return f"no such table; tables: {', '.join(sorted(world.state))}"
        return json.dumps(world.search(value, table, field))

    def read_row(table: str, key: str) -> str:
        row = world.row(table, key)
        return "no such row" if row is None else json.dumps(row, default=str)[:3000]

    def read_policy(section: str = "") -> str:
        return world.policy_section(section)[:6000]

    return {"read_schema": read_schema, "find_rows": find_rows, "read_row": read_row, "read_policy": read_policy,
            "add_items": lambda items=None: store.add(items), "drop_items": lambda ids=None: store.drop(ids)}


def _answer(impl: dict, call: Any) -> str:
    fn = impl.get(call.name)
    try:
        return str(fn(**(call.arguments or {}))) if fn else "unknown tool"
    except TypeError:
        return "bad arguments"


def run_session(model: Any, messages: list[dict], state: dict, config: Any,
                max_rounds: int = MAX_READ_ROUNDS,
                price: Optional[Callable[[dict], float]] = None, force_answer: bool = False,
                ceiling_usd: Optional[float] = None, tools: Optional[list[dict]] = None,
                impl: Optional[dict[str, Callable[..., str]]] = None) -> dict:
    """One bounded writer session: the tools, then a final answer.

    `tools` and `impl` default to the two read-only lookups over `state`. `price` turns summed usage
    into dollars; without one the session reports 0. A session that is still calling tools after
    `max_rounds`, or once its spend reaches `ceiling_usd`, ends unfinished with no text, unless
    `force_answer` makes that last call one where no tool may be called.
    """
    tools = READ_TOOLS if tools is None else tools
    impl = read_tools(state) if impl is None else impl
    working = list(messages)
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    rounds, thinking, uses = 0, [], []
    for number in range(max_rounds + 1):
        spent_out = _spent(price, usage, ceiling_usd)
        if spent_out and not force_answer:
            break
        last = force_answer and (number == max_rounds or spent_out)
        reply = model.query(working, tools=tools, config=_forced(config) if last else config)
        for key, value in _usage_of(reply).items():
            usage[key] += value
        thinking += _thinking_of(reply)
        if not reply.tool_calls:
            return _result(reply.content or "", usage, price, rounds, thinking, uses)
        if last:
            break
        rounds += 1
        working.append(_assistant_tool_turn(reply))
        for call in reply.tool_calls:
            uses.append(call.name)
            working.append({"role": "tool", "tool_call_id": call.id, "content": _answer(impl, call)})
    return _result("", usage, price, rounds, thinking, uses, unfinished=True)


def _spent(price: Optional[Callable[[dict], float]], usage: dict, ceiling_usd: Optional[float]) -> bool:
    return ceiling_usd is not None and price is not None and price(usage) >= ceiling_usd


def _forced(config: Any) -> Any:
    """The call config under which no tool may be called."""
    return config.model_copy(update={"tool_choice": "none"}) if config is not None else config


def _result(text: str, usage: dict, price: Optional[Callable[[dict], float]], rounds: int,
            thinking: list, uses: list, **extra: Any) -> dict:
    return {"text": text, "usage": usage, "usd": price(usage) if price else 0.0, "tool_rounds": rounds,
            "thinking": thinking, "tool_uses": uses, **extra}


def parse_list(text: str, key: str) -> tuple[list, str]:
    """The list under `key` in the one JSON object of a final answer, or why there is none."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, IndexError):
        return [], "no JSON object"
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], f"no {key} list"
    return items, ""


def _section(name: str) -> str:
    return "section:" + re.sub(r"\s+", " ", name).strip().casefold()


def policy_sections(policy_text: str) -> list[str]:
    """The headed sections of a policy text, as `section:<name>`, for a because to name."""
    out = []
    for line in (policy_text or "").splitlines():
        clean = line.strip().strip("#").replace("**", "").strip()
        if line.startswith("#") and len(clean.split()) >= 2:
            out.append(_section(clean))
    return out


def _policy_texts(workdir: Path) -> set[str]:
    """The Environment's policy.md, else every distinct system prompt its recordings carry."""
    shipped = Path(workdir) / "env" / "policy.md"
    if shipped.is_file():
        return {shipped.read_text(encoding="utf-8")}
    texts = set()
    for path in sorted((Path(workdir) / "traces").glob("*.json")):
        try:
            prompt = json.loads(path.read_text(encoding="utf-8")).get("system_prompt")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(prompt, str) and prompt:
            texts.add(prompt)
    return texts


def _rule_names(workdir: Path) -> list[str]:
    try:
        body = json.loads((Path(workdir) / "constraints_check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rules = body.get("constraints", []) if isinstance(body, dict) else body
    return [str(rule.get("name") or rule.get("id")) for rule in rules if isinstance(rule, dict)
            and (rule.get("name") or rule.get("id"))] if isinstance(rules, list) else []


def policy_sections_of(workdir: Path) -> list[str]:
    """Every policy name a because may cite in this workdir, the single source for all grounding.

    The headings of the policy text (policy_sections) and the compiled rule names of
    constraints_check.json, as one normalised, sorted set of `section:<name>`. The writer grounds with
    it and the trust gates (sp-trust) import it, so the two never ground against different vocabularies.
    """
    names = {section for text in _policy_texts(workdir) for section in policy_sections(text)}
    names |= {_section(name) for name in _rule_names(workdir)}
    return sorted(names)


def tool_rows(sigs: Iterable[Any]) -> list[dict]:
    """Name, kind and argument names per tool schema, sorted by name."""
    rows = []
    for sig in sigs:
        body = sig.model_dump() if hasattr(sig, "model_dump") else dict(sig)
        if body.get("name"):
            rows.append({"name": body["name"], "kind": body.get("kind", "read"),
                         "args": sorted(field["name"] for field in body.get("args_fields") or [])})
    return sorted(rows, key=lambda row: row["name"])
