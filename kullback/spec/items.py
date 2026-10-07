"""The Verifier's items, written by the Spec writer from the Intent, the policy and the world (sh-writer).

An item is one typed thing a correct Run shows, never a Reference's diff:

    state : row_is    {table, find: {field: value}, expect: {field: value | [values] | {"one_of": [...]} |
                       {"not": [...]}}, free: [fields a tool sets to a value the writer cannot find]}
            row_new   {table, where: {field: value}, count}      a row the Run makes, its id minted, unnamed
            row_keeps {table, find, fields: [...]}               one per "do not change"
    event : called {tool, args?} | not_called {tool, args?} | before {first: "confirm_turn" | tool, then: tool}
    text  : said {values: [...], not_values: [...]}              containment, negation aware
    judge : {question, anchor: {answer, accepted, reject, row: {table, find, fields}},
             evidence: ["final_answer" | "after_call:<tool>" | "assistant:<k>" | "assistant_turns"]}

Every item carries gate, weight and its provenance: fact_ids (Intent facts) or policy_line (a verbatim
policy line). Code adds the sanity item (`nothing_else`): no row outside the declared rows changed.
"Find, do not invent": `World.check` refuses a value no world row, policy line or user turn holds, a
find that names no row or several, and a field the table does not have, each with the reason. A judge
anchor cites the row it was read from (one row by a world search) and its answer states those fields;
only an anchor whose answer quotes the item's policy line goes without a row.
State items may name an alternative end state (`alt`); one without belongs to every end state, and a
Run passes the state items when any one end state matches.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable, Optional

from kullback.runner.records import Atom, Conduct, EndState, ExpectedCell, Forbidden, ValueSource

STATE_KINDS = ("row_is", "row_new", "row_keeps")
EVENT_KINDS = ("called", "not_called", "before")
ITEM_KINDS = STATE_KINDS + EVENT_KINDS + ("said", "judge")
SANITY = "nothing_else"
SANITY_ID = "sanity"
CONFIRM_TURN = "confirm_turn"
EVIDENCE = ("final_answer", "assistant_turns")
AFTER_CALL = "after_call:"
ASSISTANT = "assistant:"
# Gate by default: what the world must hold and what the event log must show; text and judge items weigh.
DEFAULT_GATE = {kind: kind in STATE_KINDS + EVENT_KINDS for kind in ITEM_KINDS}
# The shortest policy line an item may rest on, as the because rule asks of an Intent quote (ground.MIN_QUOTE).
MIN_POLICY_LINE = 12
MAX_ALTERNATIVES = 8
FIND_LIMIT = 10


def _flat(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def norm(value: Any) -> str:
    """One comparable text for a scalar: numbers without a trailing .0, text casefolded and trimmed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number == int(number):
            return str(int(number))
        return repr(number)
    text = _flat(value)
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text
    return norm(number) if re.fullmatch(r"-?[\d,]*\.?\d+", text) else text


def _leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    elif value is not None:
        yield value


def path_value(row: Any, field: str) -> Any:
    node = row
    for part in str(field).split("."):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _contains(haystack: str, needle: str) -> bool:
    """`needle` in `haystack` as whole words (a number is never found inside a longer number)."""
    return bool(needle) and re.search(rf"(?<![\w.]){re.escape(needle)}(?![\w]|\.\d)", haystack) is not None


class World:
    """The Starting state, the policy and the user turns, for finding rows and sourcing values."""

    def __init__(self, state: dict, policy_text: str = "", facts: Iterable[Any] = (), tools: Iterable[dict] = (),
                 columns: Optional[dict] = None):
        self.state = {table: rows for table, rows in (state or {}).items() if isinstance(rows, dict)}
        # The schema's columns per table, for a table the Starting state holds no row of.
        self.declared = {table: list(names) for table, names in (columns or {}).items()}
        self.policy = policy_text or ""
        self.facts = list(facts)
        self.tools = {row["name"]: row for row in tools if isinstance(row, dict) and row.get("name")}
        self._values: Optional[dict[str, tuple[str, str]]] = None

    # --- reading -------------------------------------------------------------------------------
    def columns(self, table: str) -> list[str]:
        """The top-level fields of a table, the union over its rows."""
        names: dict[str, None] = dict.fromkeys(self.declared.get(table) or [])
        for row in (self.state.get(table) or {}).values():
            if isinstance(row, dict):
                names.update(dict.fromkeys(row))
        return sorted(names)

    def schema(self) -> dict:
        return {"tables": {table: {"rows": len(rows), "fields": self.columns(table)}
                           for table, rows in sorted(self.state.items())},
                "tools": sorted(self.tools.values(), key=lambda row: row["name"])}

    def find(self, table: str, where: dict) -> list[str]:
        """The keys of the rows of `table` whose every named field (dotted path) equals its value."""
        rows = self.state.get(table) or {}
        wanted = {field: norm(value) for field, value in (where or {}).items()}
        return [key for key, row in rows.items()
                if wanted and all(norm(path_value(row, field)) == value for field, value in wanted.items())]

    def search(self, value: Any, table: str = "", field: str = "") -> list[dict]:
        """Rows holding `value` in `field` (any field when empty), in `table` (every table when empty)."""
        target, hits = norm(value), []
        for name in [table] if table else sorted(self.state):
            for key, row in (self.state.get(name) or {}).items():
                leaves = [path_value(row, field)] if field else list(_leaves(row))
                if norm(key) == target or any(norm(leaf) == target for leaf in leaves if leaf is not None):
                    hits.append({"table": name, "key": key})
                if len(hits) >= FIND_LIMIT:
                    return hits
        return hits

    def row(self, table: str, key: str) -> Optional[dict]:
        row = (self.state.get(table) or {}).get(key)
        return row if isinstance(row, dict) else None

    def policy_section(self, name: str = "") -> str:
        """The policy text under the heading `name`; with no name, the headings."""
        lines = self.policy.splitlines()
        heads = [(n, line.strip().strip("#").replace("**", "").strip()) for n, line in enumerate(lines)
                 if line.startswith("#")]
        if not name:
            return json.dumps([head for _, head in heads if head])
        for index, (start, head) in enumerate(heads):
            if _flat(head) == _flat(name):
                end = heads[index + 1][0] if index + 1 < len(heads) else len(lines)
                return "\n".join(lines[start:end])
        return "no such section"

    def section_of(self, line: str) -> str:
        """The heading the policy line sits under, or empty."""
        at = _flat(self.policy).find(_flat(line))
        head = ""
        if at < 0:
            return head
        seen = 0
        for raw in self.policy.splitlines():
            seen += len(_flat(raw)) + 1
            if raw.startswith("#"):
                head = raw.strip().strip("#").replace("**", "").strip()
            if seen > at:
                return head
        return head

    # --- sourcing ------------------------------------------------------------------------------
    def _world_values(self) -> dict[str, tuple[str, str]]:
        if self._values is None:
            values: dict[str, tuple[str, str]] = {}
            for table, rows in self.state.items():
                for key, row in rows.items():
                    values.setdefault(norm(key), (table, key))
                    for leaf in _leaves(row):
                        values.setdefault(norm(leaf), (table, key))
            self._values = values
        return self._values

    def source(self, value: Any) -> Optional[ValueSource]:
        """Where a value is found: a user turn first, then a policy line, then a world row; None if nowhere."""
        target = norm(value)
        if not target:
            return None
        for fact in self.facts:
            if _contains(_flat(fact.text), target):
                return ValueSource(kind="user_turn", ptr={"recording": fact.source.recording, "turn": fact.source.turn,
                                                          "fact": fact.id})
        if _contains(_flat(self.policy), target):
            return ValueSource(kind="policy", ptr={"clause": self.section_of(target)})
        found = self._world_values().get(target)
        if found is not None:
            return ValueSource(kind="world", ptr={"table": found[0], "row": found[1]})
        return None

    # --- checking one item ---------------------------------------------------------------------
    def check(self, item: Any, known_facts: dict, unwitnessed: Callable[[Any], bool] = lambda fact: False
              ) -> tuple[Optional[dict], str]:
        """The item as code keeps it (row ids resolved, gate, weight, sources), or why it is refused."""
        if not isinstance(item, dict) or item.get("kind") not in ITEM_KINDS:
            return None, f"kind must be one of {', '.join(ITEM_KINDS)}"
        kept = {"id": str(item.get("id") or ""), "kind": item["kind"]}
        if not kept["id"] or kept["id"] == SANITY_ID:
            return None, "every item needs its own id (not 'sanity')"
        why = self._provenance(item, kept, known_facts, unwitnessed)
        if why:
            return None, why
        gate = item.get("gate", DEFAULT_GATE[item["kind"]])
        weight = item.get("weight", 1.0)
        if not isinstance(gate, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            return None, "gate is true or false and weight a positive number"
        kept.update(gate=gate, weight=float(weight))
        why = getattr(self, f"_{item['kind']}")(item, kept)
        return (None, why) if why else (kept, "")

    def _provenance(self, item: dict, kept: dict, known: dict, unwitnessed: Callable[[Any], bool]) -> str:
        fact_ids = [str(f) for f in item.get("fact_ids") or [] if str(f) in known]
        line = _flat(item.get("policy_line"))
        if line and (len(line) < MIN_POLICY_LINE or line not in _flat(self.policy)):
            return "policy_line must be a verbatim line of the policy, at least 12 characters"
        if not fact_ids and not line:
            return "name the Intent facts the item answers (fact_ids) or quote the policy line (policy_line)"
        if fact_ids and not line and all(unwitnessed(known[f]) for f in fact_ids):
            return "an item may not rest on unwitnessed facts alone; add a witnessed fact or a policy line"
        kept["fact_ids"] = fact_ids
        kept["policy_line"] = str(item.get("policy_line")).strip() if line else None
        return ""

    def _value(self, value: Any, what: str) -> tuple[Any, str]:
        """A value or value set with every value sourced, or why not."""
        if isinstance(value, dict):
            if len(value) != 1 or next(iter(value)) not in ("one_of", "not") or not isinstance(
                    next(iter(value.values())), list) or not next(iter(value.values())):
                return None, f"{what}: a value set is {{\"one_of\": [...]}} or {{\"not\": [...]}}"
            values = next(iter(value.values()))
        elif isinstance(value, list):
            if not value or any(isinstance(one, (dict, list)) for one in value):
                return None, (f"{what}: a list value lists scalars found in the world; a field holding records "
                              "a tool fills in goes in free")
            values = value
        else:
            values = [value]
        for one in values:
            if isinstance(one, (dict, list)):
                return None, f"{what}: name one scalar field per cell, not a nested value"
            if one is not None and self.source(one) is None:
                return None, (f"{what}: value {json.dumps(one)} is not in the world, the policy or a user turn; "
                              "find it with the tools, or demand another field")
        return value, ""

    def _table(self, item: dict) -> str:
        table = str(item.get("table") or "")
        return "" if table in self.state else f"no table {table!r}; tables: {', '.join(sorted(self.state))}"

    def _one_row(self, item: dict, kept: dict) -> str:
        why = self._table(item)
        if why:
            return why
        find = item.get("find")
        if not isinstance(find, dict) or not find:
            return "find names the row: {field: value} read from the world"
        keys = self.find(item["table"], find)
        if len(keys) != 1:
            return f"find matches {len(keys)} rows of {item['table']}; it must match exactly one"
        kept.update(table=item["table"], find=find, row=keys[0])
        return self._alt(item, kept)

    def _alt(self, item: dict, kept: dict) -> str:
        alt = item.get("alt")
        if alt is None:
            return ""
        if not isinstance(alt, int) or not 0 <= alt < MAX_ALTERNATIVES:
            return f"alt is an end state number from 0 to {MAX_ALTERNATIVES - 1}"
        kept["alt"] = alt
        return ""

    def _field(self, table: str, field: Any) -> str:
        """A field the table has; any field of a table with no row and no declared column is unknown, not refused."""
        top, columns = str(field or "").split(".")[0], self.columns(table)
        return "" if top in columns or (top and not columns) else f"{table} has no field {field!r}"

    def _row_is(self, item: dict, kept: dict) -> str:
        why = self._one_row(item, kept)
        expect, free = item.get("expect") or {}, item.get("free") or []
        if why or not isinstance(expect, dict) or not isinstance(free, list) or not (expect or free):
            return why or ("expect names the fields the row holds after a correct Run: {field: value}; free "
                           "names fields it may change to a value no tool lets you find")
        for field, value in expect.items():
            why = self._field(kept["table"], field) or self._value(value, f"{field}")[1]
            if why:
                return why
        for field in free:
            why = self._field(kept["table"], field)
            if why or str(field) in expect:
                return why or f"{field} is in expect and in free; name it once"
        kept["expect"] = dict(expect)
        if free:
            kept["free"] = [str(field) for field in free]
        return ""

    def _row_new(self, item: dict, kept: dict) -> str:
        why = self._table(item) or self._alt(item, kept)
        where = item.get("where")
        if why or not isinstance(where, dict) or not where:
            return why or "where names fields the new row holds: {field: value}"
        for field, value in where.items():
            why = self._field(item["table"], field) or self._value(value, field)[1]
            if why or isinstance(value, dict):
                return why or f"{field}: a new row's field holds one value"
        count = item.get("count", 1)
        if not isinstance(count, int) or count < 1:
            return "count is the number of new rows, at least 1"
        kept.update(table=item["table"], where=dict(where), count=count)
        return ""

    def _row_keeps(self, item: dict, kept: dict) -> str:
        why = self._one_row(item, kept)
        fields = item.get("fields")
        if why or not isinstance(fields, list) or not fields:
            return why or "fields lists the fields that must not change"
        for field in fields:
            why = self._field(kept["table"], field)
            if why:
                return why
        row = self.row(kept["table"], kept["row"]) or {}
        kept.update(fields=[str(f) for f in fields], expect={str(f): path_value(row, str(f)) for f in fields})
        return ""

    def _tool(self, name: Any) -> str:
        return "" if name in self.tools else f"no tool {name!r}"

    def _called(self, item: dict, kept: dict) -> str:
        why = self._tool(item.get("tool"))
        args = item.get("args") or {}
        if why or not isinstance(args, dict):
            return why or "args is {argument: value}"
        allowed = self.tools[item["tool"]].get("args") or []
        for name, value in args.items():
            if allowed and name not in allowed:
                return f"{item['tool']} takes {', '.join(allowed)}"
            why = self._value(value, name)[1]
            if why:
                return why
        kept.update(tool=item["tool"], args=dict(args))
        return ""

    _not_called = _called

    def _before(self, item: dict, kept: dict) -> str:
        first, then = item.get("first"), item.get("then")
        why = (self._tool(first) if first != CONFIRM_TURN else "") or self._tool(then)
        if why:
            return why
        kept.update(first=first, then=then)
        return ""

    def _said(self, item: dict, kept: dict) -> str:
        values, nots = item.get("values") or [], item.get("not_values") or []
        if not isinstance(values, list) or not isinstance(nots, list) or not (values or nots):
            return "said lists values (and not_values): the words the Candidate must or must not state"
        for value in values + nots:
            if isinstance(value, (dict, list)) or not norm(value):
                return "each said value is one short value, never a sentence"
            if len(str(value).split()) > 6:
                return f"said value {json.dumps(value)} is a sentence; list the value itself"
            why = self._value(value, "said")[1]
            if why:
                return why
        kept.update(values=list(values), not_values=list(nots))
        return ""

    def _evidence(self, evidence: Any) -> str:
        """Each evidence ref is final_answer, assistant_turns, assistant:<k> or after_call:<tool>."""
        if not isinstance(evidence, list) or not evidence:
            return "evidence is a list of refs"
        for ref in evidence:
            text = str(ref)
            if text in EVIDENCE or (text.startswith(ASSISTANT) and text[len(ASSISTANT):].isdigit()):
                continue
            if text.startswith(AFTER_CALL) and text[len(AFTER_CALL):] in self.tools:
                continue
            return (f"evidence is final_answer, {AFTER_CALL}<tool>, {ASSISTANT}<k> or assistant_turns; "
                    f"not {text!r}")
        return ""

    def _anchor_row(self, anchor: dict, answer: str) -> tuple[Optional[dict], str]:
        """The row the anchor cites, found by a world search: one row, its fields, each stated in the answer."""
        cited = anchor.get("row")
        if not isinstance(cited, dict):
            return None, ("anchor.row is {table, find: {field: value}, fields: [...]}: the row you found the answer "
                          "in with find_rows, and the fields the answer states")
        kept: dict = {}
        why = self._one_row(dict(cited, alt=None), kept)
        fields = cited.get("fields")
        if why or not isinstance(fields, list) or not fields:
            return None, f"anchor.row: {why or 'fields lists the fields the answer states'}"
        row, values = self.row(kept["table"], kept["row"]) or {}, {}
        for field in fields:
            why = self._field(kept["table"], field)
            if why:
                return None, f"anchor.row: {why}"
            values[str(field)] = path_value(row, str(field))
            if norm(values[str(field)]) not in norm(answer):
                return None, (f"anchor.answer does not state {kept['table']} row {kept['row']} {field} "
                              f"({json.dumps(values[str(field)])}); read the row again or cite the right one")
        return {"table": kept["table"], "find": kept["find"], "key": kept["row"], "fields": values}, ""

    def _judge(self, item: dict, kept: dict) -> str:
        question = str(item.get("question") or "").strip()
        anchor = item.get("anchor")
        if not question.endswith("?"):
            return "question is one yes or no question ending with '?'"
        if not isinstance(anchor, dict) or not str(anchor.get("answer") or "").strip():
            return "anchor is {answer, accepted: [...], reject: [...], row}: what a met checkpoint states"
        answer = str(anchor["answer"]).strip()
        evidence = item.get("evidence") or ["final_answer"]
        why = self._evidence(evidence)
        if why:
            return why
        cited, why = self._anchor_row(anchor, answer)
        policy = item.get("policy_line") and _flat(item["policy_line"]) in _flat(answer)
        if why and not (policy and anchor.get("row") is None):
            return why
        kept.update(question=question, anchor={"answer": answer,
                                               "accepted": [str(a) for a in anchor.get("accepted") or []],
                                               "reject": [str(r) for r in anchor.get("reject") or []],
                                               "row": cited},
                    evidence=[str(e) for e in evidence])
        return ""


# --- compiling items to the Verifier's parts -------------------------------------------------------

def because_of(item: dict, facts: dict, world: World) -> str:
    """The because a Check keeps: the facts' words, else the policy line under its section's name."""
    texts = [facts[f].text for f in item.get("fact_ids") or [] if f in facts]
    if texts:
        return " ".join(texts)
    section = world.section_of(item.get("policy_line") or "")
    return f"section:{section}: {item.get('policy_line')}" if section else str(item.get("policy_line"))


def _from(item: dict) -> dict:
    return {"fact_ids": list(item.get("fact_ids") or []), "policy_line": item.get("policy_line")}


def _describe(item: dict) -> str:
    kind = item["kind"]
    if kind in ("row_is", "row_keeps"):
        free = f", may change {sorted(item['free'])}" if item.get("free") else ""
        return f"{item['table']} row {item['row']} holds {sorted(item['expect'])}{free}"
    if kind == "row_new":
        return f"{item.get('count', 1)} new {item['table']} row(s) where {sorted(item['where'])}"
    if kind in ("called", "not_called"):
        return f"{item['tool']} is {'not ' if kind == 'not_called' else ''}called"
    if kind == "before":
        return f"{item['first']} comes before {item['then']}"
    if kind == "said":
        return f"the Candidate states {len(item['values'])} value(s), not {len(item['not_values'])}"
    return f"judge: {item['question']}"


def atom_of(item: dict) -> Atom:
    """One item as one atom: gate and weight on it, the item itself and its provenance as the target.

    The legacy scorer never fails a Run on these (kind allowed); the end state, the forbidden list and
    the conduct rules carry the gates, and runner/items.py scores every item.
    """
    target = {key: value for key, value in item.items() if key not in ("id", "gate", "weight", "fact_ids",
                                                                         "policy_line")}
    target["from"] = _from(item)
    return Atom(id=item["id"], kind="allowed", description=_describe(item), target=target,
                judge=item["kind"] == "judge", gate=bool(item["gate"]), weight=float(item["weight"]))


def _cell_source(value: Any, world: World) -> Optional[ValueSource]:
    found = None
    values = next(iter(value.values())) if isinstance(value, dict) else value if isinstance(value, list) else [value]
    for one in values:
        found = world.source(one) if one is not None else ValueSource(kind="world", ptr={"removed": True})
        if found is None:
            return None
    return found


def end_states_of(items: list[dict], world: World, action_tables: Iterable[str] = ()) -> list[EndState]:
    """One EndState per alternative: the state items without `alt` in each, those with `alt` in theirs.

    Every EndState allows the action tables (a call's record, not the world's change); nothing else
    outside its cells and new rows may move, which is the sanity item. With no state item at all there
    is one EndState with no cells: the Run must leave the world as it found it.
    """
    state = [item for item in items if item["kind"] in STATE_KINDS]
    alts = sorted({item["alt"] for item in state if item.get("alt") is not None}) or [None]
    allowed = [{"table": table} for table in sorted(set(action_tables))]
    out = []
    for alt in alts:
        cells: list[ExpectedCell] = []
        new_rows: list[dict] = []
        free: list[dict] = []
        for item in state:
            if item.get("alt") not in (None, alt):
                continue
            if item["kind"] == "row_new":
                new_rows.append({"table": item["table"], "where": item["where"], "count": item["count"],
                                 "item": item["id"]})
                continue
            free += [{"table": item["table"], "row_id": item["row"], "field": field, "why": f"free: {item['id']}"}
                     for field in item.get("free") or []]
            row_source = ValueSource(kind="world", ptr={"table": item["table"], "row": item["row"]})
            for field, value in item["expect"].items():
                cells.append(ExpectedCell(table=item["table"], row_id=item["row"], field=field, value=value,
                                          source=_cell_source(value, world), row_source=row_source))
        cells.sort(key=lambda cell: (cell.table, cell.row_id, cell.field))
        out.append(EndState(cells=cells, allowed=list(allowed) + free, new_rows=new_rows))
    return out


def sanity_item(end_states: list[EndState]) -> dict:
    """The generated sanity item: the declared rows of each end state, outside which nothing moves."""
    rows = [sorted({f"{cell.table}.{cell.row_id}" for cell in state.cells}
                   | {f"{rule['table']}.{rule['row_id']}" for rule in state.allowed if "row_id" in rule})
            for state in end_states]
    new = [[{"table": spec["table"], "count": spec["count"]} for spec in state.new_rows] for state in end_states]
    return {"id": SANITY_ID, "kind": SANITY, "gate": True, "weight": 1.0, "rows": rows, "new": new,
            "fact_ids": [], "policy_line": None}


def sanity_atom(end_states: list[EndState]) -> Atom:
    item = sanity_item(end_states)
    return Atom(id=SANITY_ID, kind="allowed", gate=True, weight=1.0,
                description="no row outside the declared rows changed (generated by code)",
                target={"kind": SANITY, "rows": item["rows"], "new": item["new"],
                        "from": {"fact_ids": [], "policy_line": None, "code": True}})


def _source_of(item: dict, facts: dict) -> ValueSource:
    fact = next((facts[f] for f in item.get("fact_ids") or [] if f in facts), None)
    if fact is not None:
        return ValueSource(kind="user_turn", ptr={"recording": fact.source.recording, "turn": fact.source.turn})
    return ValueSource(kind="policy", ptr={"clause": item.get("policy_line")})


def event_gates(items: list[dict], facts: dict, write_tools: Iterable[str]) -> tuple[list[Conduct], list[Forbidden]]:
    """The gate event items as the Verifier's conduct rules and forbidden calls, read by today's verdict.

    `before` with a confirmation turn first is confirm_before_write; `called` is a call that must
    happen (the conduct kind that reads `called`); `not_called` is a forbidden call of the tool. A
    tool-before-tool order has no conduct rule yet and is scored by runner/items.py alone.
    """
    writes = set(write_tools)
    conduct: list[Conduct] = []
    forbidden: list[Forbidden] = []
    for item in items:
        if not item.get("gate"):
            continue
        source = _source_of(item, facts)
        if item["kind"] == "before" and item["first"] == CONFIRM_TURN:
            conduct.append(Conduct(kind="confirm_before_write", tool=item["then"], source=source))
        elif item["kind"] == "called" and not item.get("args"):
            conduct.append(Conduct(kind="handoff", tool=item["tool"], source=source))
        elif item["kind"] == "not_called" and not item.get("args"):
            forbidden.append(Forbidden(kind="write" if item["tool"] in writes else "end_call", tool=item["tool"],
                                       source=source))
    return conduct, forbidden


def contradictions(items: list[dict]) -> list[tuple[str, str]]:
    """(item id, why) for items no Run can pass together: a tool both called and not, a cell both v and not v."""
    out: list[tuple[str, str]] = []
    needed = {}
    for item in items:
        if item["kind"] == "called" and not item.get("args"):
            needed.setdefault(item["tool"], item["id"])
        elif item["kind"] == "before":
            needed.setdefault(item["then"], item["id"])
    for item in items:
        if item["kind"] == "not_called" and not item.get("args") and item["tool"] in needed:
            out.append((item["id"], f"not_called {item['tool']} beside {needed[item['tool']]}, which needs the call"))
    cells: dict[tuple, tuple[str, str]] = {}
    for item in items:
        if item["kind"] != "row_is":
            continue
        for field, value in item["expect"].items():
            key = (item.get("alt"), item["table"], item["row"], field)
            text = json.dumps(value, sort_keys=True, default=str)
            if key in cells and cells[key][1] != text:
                out.append((item["id"], f"{field} of that row is also expected by {cells[key][0]}"))
            cells.setdefault(key, (item["id"], text))
    return out


def counts_of(items: list[dict]) -> dict:
    """Items by kind and by gate, the sanity item included, for the Spec record and the report."""
    by_kind: dict[str, int] = {}
    for item in items:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
    return {"items": len(items), "by_kind": dict(sorted(by_kind.items())),
            "gate": sum(1 for item in items if item.get("gate")),
            "scored": sum(1 for item in items if not item.get("gate")),
            "fact_backed": sum(1 for item in items if item.get("fact_ids")),
            "policy_backed": sum(1 for item in items if item.get("policy_line"))}


__all__ = ["CONFIRM_TURN", "DEFAULT_GATE", "EVENT_KINDS", "ITEM_KINDS", "SANITY", "SANITY_ID", "STATE_KINDS",
           "World", "atom_of", "because_of", "contradictions", "counts_of", "end_states_of", "event_gates",
           "norm", "sanity_atom", "sanity_item"]
