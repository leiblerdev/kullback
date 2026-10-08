"""The writer's only tools: two read-only lookups over the Starting state, and the bounded session.

Ported from the intentv experiment (its `_read_impl` and `run_writer`). The tools close over one
Starting state dict and nothing else, so a writer session can read no Run, Reference, Verifier or
End state: `refuse_run_inputs` refuses such records before a session starts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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


def _answer(impl: dict, call: Any) -> str:
    fn = impl.get(call.name)
    try:
        return str(fn(**(call.arguments or {}))) if fn else "unknown tool"
    except TypeError:
        return "bad arguments"


def run_session(model: Any, messages: list[dict], state: dict, config: Any,
                max_rounds: int = MAX_READ_ROUNDS,
                price: Optional[Callable[[dict], float]] = None, force_answer: bool = False,
                ceiling_usd: Optional[float] = None) -> dict:
    """One bounded writer session: the read-only tools, then a final answer.

    `price` turns summed usage into dollars; without one the session reports 0. A session that
    is still reading after `max_rounds`, or once its spend reaches `ceiling_usd`, ends unfinished
    with no text, unless `force_answer` makes that last call one where no tool may be called.
    """
    impl = read_tools(state)
    working = list(messages)
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    rounds, thinking, uses = 0, [], []
    for number in range(max_rounds + 1):
        spent_out = _spent(price, usage, ceiling_usd)
        if spent_out and not force_answer:
            break
        last = force_answer and (number == max_rounds or spent_out)
        reply = model.query(working, tools=READ_TOOLS, config=_forced(config) if last else config)
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
