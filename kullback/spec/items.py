"""The Verifier's items, written by the Spec writer from the Intent, the policy and the world (sh-writer).

An item is one typed thing a correct Run shows, never a Reference's diff:

    state : row_is    {table, find: {field: value}, expect: {field: value | [values] | {"one_of": [...]} |
                       {"not": [...]}}, free: {field: why a tool sets it to a value the writer cannot find}}
            row_new   {table, where: {field: value}, count}      a row the Run makes, its id minted, unnamed
            row_keeps {table, find, fields: [...]}               one per "do not change"
    event : called {tool, args?} | not_called {tool, args?} | before {first: "confirm_turn" | tool, then: tool}
    text  : said {values: [...], not_values: [...]}              containment in an assistant message
    judge : {question, anchor: {answer, accepted, reject, row: {table, find, fields}},
             evidence: ["final_answer" | "after_call:<tool>" | "assistant:<k>" | "assistant_turns"]}

Every item carries gate, weight and its provenance: fact_ids (Intent facts) or policy_line (a verbatim
policy line). Code adds the sanity item (`nothing_else`): no row outside the declared rows changed.
"Find, do not invent": `World.check` refuses a value no world row, policy line or user turn holds, a
find that names no row or several, and a field the table does not have, each with the reason; a row_is
whose expect is empty or holds only the find key, a free field with no reason, and a free field holding
a value an Intent fact names elsewhere in the world (that value belongs in expect). A judge
anchor cites the row it was read from (one row by a world search) and its answer states those fields;
only an anchor whose answer quotes the item's policy line goes without a row, and that refusal item
always gates.
A gated value for a cell must be reachable (D387, `World.reach_of`): a tool writes it to that field (the
build's learned writes), or copies that field from an argument and a start row holds the value anywhere,
a write tool's result in a recording of the Task shows it there, the user's own words fix it, or a start
row holds it under that field name. A gated row_is or row_new naming any other
value is refused with the field's reachable values; a weighed one is kept with the same finding as a
warning. Nulls and `not` sets are never checked. A World built without a `Reach` checks nothing here.
State items may name an alternative end state (`alt`); one without belongs to every end state, and a
Run passes the state items when any one end state matches.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable, NamedTuple, Optional

from kullback.runner.atom_context import is_transfer
from kullback.runner.records import Atom, EndState, ExpectedCell, ValueSource
from kullback.spec.atoms import called_set_atom
from kullback.spec.effects import type_of
from kullback.spec.schema import is_reaction

STATE_KINDS = ("row_is", "row_new", "row_keeps")
EVENT_KINDS = ("called", "not_called", "before")
ITEM_KINDS = STATE_KINDS + EVENT_KINDS + ("said", "judge")
SANITY = "nothing_else"
SANITY_ID = "sanity"
CONFIRM_TURN = "confirm_turn"
EVIDENCE = ("final_answer", "assistant_turns")
# The default evidence: every agent turn up to the first transfer or hand-off call (judge/shape.py).
WINDOW = "assistant_turns"
AFTER_CALL = "after_call:"
ASSISTANT = "assistant:"
# Gate by default: what the world must hold and what the event log must show; text and judge items weigh.
DEFAULT_GATE = {kind: kind in STATE_KINDS + EVENT_KINDS for kind in ITEM_KINDS}
# The shortest policy line an item may rest on, as the because rule asks of an Intent quote (ground.MIN_QUOTE).
MIN_POLICY_LINE = 12
MAX_ALTERNATIVES = 8
FIND_LIMIT = 10
# What `World._arg_shape` answers for a list of objects the recordings show: check it on its key field.
OBJECTS = "objects"


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


def _keys(value: Any) -> Iterable[str]:
    """Every dict key nested in a value."""
    if isinstance(value, dict):
        for name, item in value.items():
            yield str(name)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    """The numbers a text states, thousands separators dropped."""
    return {float(match.replace(",", "")) for match in NUMBER_RE.findall(text or "")}


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


def _unfounded_gate(item: dict, facts: list) -> str:
    """Why a gate item may not rest on these facts: every one was ruled unfounded (the Examiner ruled the
    user never asked for it, spec/review.py); else empty. A scored item may."""
    gate = item.get("gate", DEFAULT_GATE[item["kind"]]) is True
    if gate and facts and all(getattr(fact, "unfounded", False) for fact in facts):
        return ("a gate may not rest on unfounded facts alone (the Examiner ruled the user never asked for "
                "them); cite a fact the user said or make the item scored (gate false)")
    return ""


# Why a gate on a side fact is weighed, kept on the item so a count can find it.
SIDE_WEIGHED = "side fact"


# Why a gate on a reaction is weighed (D366): the user said it in answer to the agent's own conduct (could
# not, declined citing no rule, stalled, proposed another way), so it is the recording's way out, not the ask.
AGENT_WEIGHED = "reaction"
# Why a gate on a conditional ask is weighed (D366, H2): the condition is the agent's claim about the world
# or the rules, and the item names neither the row nor the policy line that confirms it.
UNCONFIRMED_WEIGHED = "unconfirmed condition"


def _unconfirmed(kept: dict, facts: list) -> bool:
    """Whether a gate item cites a conditional ask (condition world_or_policy) and confirms the agent's
    claim neither by a policy line nor by the row it names in confirms."""
    return any(getattr(fact, "condition", "none") == "world_or_policy" for fact in facts) \
        and not kept.get("policy_line") and not kept.get("confirms")


# The item kinds that check what the agent told the user: one citing a side fact answers the side question.
TOLD_KINDS = ("judge", "said")


def _on_side(kept: dict, facts: list) -> bool:
    """Whether a gate item rests on a side fact (an ask only to be told something, beside the Task's ask):
    a told item citing one, or any item citing one and no fact asking for a change or for something left
    undone. A policy line does not lift it."""
    if not any(getattr(fact, "side", False) for fact in facts):
        return False
    return kept.get("kind") in TOLD_KINDS or not any(getattr(fact, "kind", None) in ("do", "keep") for fact in facts)


def _untold(kept: dict, facts: list) -> bool:
    """Whether a told item (judge, said) rests on no told fact and no policy line: nobody asked to be told
    it and no rule says the agent must tell it (D369)."""
    return kept.get("kind") in TOLD_KINDS and not kept.get("policy_line") \
        and not any(getattr(fact, "kind", None) == "told" for fact in facts)


# Why a told gate with no told fact and no policy line is weighed (D369).
TELL_WEIGHED = "untold"

# Each weighing rule in order: the test on (kept item, its facts), the reason kept on it, the line for the writer.
_WEIGHINGS = (
    (lambda kept, facts: any(is_reaction(fact) for fact in facts), AGENT_WEIGHED,
     "gate weighed: it cites a reaction, a fact the user said in answer to the agent's own conduct "
     "(could not, declined citing no rule, stalled, proposed another way), which a run whose agent "
     "meets the first ask never reaches; it is scored, never gated. If the item checks the Task's ask, "
     "add it again citing only the ask's facts"),
    (_unconfirmed, UNCONFIRMED_WEIGHED,
     "gate weighed: it cites a conditional ask whose condition is the agent's claim about the world or "
     "the rules; confirm it yourself (confirms: the row whose read shows it, or the policy_line that "
     "states it) and gate the branch the world selects, or leave it scored"),
    (_on_side, SIDE_WEIGHED,
     "gate weighed: it rests on a side fact, a question asked beside the Task's ask, which a run "
     "whose user never asks it cannot meet; it is scored, never gated. If the item checks the "
     "Task's ask, add it again citing only the ask's facts"),
    (_untold, TELL_WEIGHED,
     "gate weighed: it checks what the agent told the user, but no fact asks to be told it and no "
     "policy_line says the agent must tell it; it is scored, never gated. Gate it only citing a told "
     "fact or quoting the policy line that requires the telling"),
)


def _weighed_gate(kept: dict, known: dict) -> str:
    """One line for the writer when its gate is weighed, else empty: it cites a reaction (condition agent),
    or a conditional ask whose condition it does not confirm, or a side fact, or it is a told item with no
    told fact and no policy line. A scored item may rest on them. after_pushback and echo are evidence, not
    this rule (D366)."""
    if not kept.get("gate"):
        return ""
    facts = [known[f] for f in kept.get("fact_ids") or []]
    for applies, reason, line in _WEIGHINGS:
        if applies(kept, facts):
            kept.update(gate=False, weighed=reason)
            return line
    return ""


# The finding a gated value no source reaches carries (D387); a weighed item gets it as a warning.
UNREACHABLE = "is unreachable"
# The most reachable values a finding lists for a field.
MAX_CANDIDATES = 10


def reach_key(value: Any) -> str:
    """A value as the reach rule compares it: `norm`, a number rounded to 6 places first, so a tool's float
    noise (114.7299999999999) matches the value it stands for (114.73)."""
    text = norm(value)
    try:
        return norm(round(float(text), 6)) if re.fullmatch(r"-?\d*\.?\d+(e-?\d+)?", text) else text
    except ValueError:
        return text


def leaf_of(field: Any) -> str:
    """The field name a dotted path ends in, list positions skipped: what a tool's writes are keyed by."""
    parts = [part for part in str(field or "").split(".") if part and not part.isdigit()]
    return parts[-1] if parts else ""


class Reach(NamedTuple):
    """What tools can write, per field (D387), each value normalised to (its text, where it was seen):
    `tools` per table and field name, the build's learned writes (spec/effects.py); `recorded` per field
    name, what a write tool's result showed in a recording of the Task (its presence only, never its path);
    `fed` per table and field name, the tool arguments the field is copied from ("tool.arg")."""
    tools: dict = {}
    recorded: dict = {}
    fed: dict = {}


def effect_reach(effects: dict) -> tuple[dict, dict]:
    """Per table and field name, the values each write tool's learned effects say it writes there, and the
    tool arguments it copies there ("tool.arg"): the `tools` and `fed` of a Reach."""
    out: dict = {}
    fed: dict = {}
    for tool, learned in sorted((effects or {}).items()):
        for row in (learned or {}).get("writes") or []:
            table, field = str(row["table"]), str(row["field"])
            for value in row.get("values") or []:
                out.setdefault(table, {}).setdefault(field, {}).setdefault(reach_key(value), (value, f"tool:{tool}"))
            for arg in row.get("from_args") or []:
                fed.setdefault(table, {}).setdefault(field, []).append(f"{tool}.{arg}")
    return out, fed


def reach_of_inputs(effects: dict, results: Iterable[tuple[str, str, Any]]) -> Reach:
    """The Reach of one Task: the build's learned writes and its recordings' write results."""
    tools, fed = effect_reach(effects)
    return Reach(tools, recorded_reach(results), fed)


def recorded_reach(results: Iterable[tuple[str, str, Any]]) -> dict:
    """Per field name, the scalars write tools' results held, from (recording, tool, result) triples;
    a result recorded as JSON text is read as JSON."""
    out: dict = {}

    def walk(node: Any, key: str, seen: str) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, str(name), seen)
        elif isinstance(node, list):
            for value in node:
                walk(value, key, seen)
        elif node is not None and key and norm(node):
            out.setdefault(key, {}).setdefault(reach_key(node), (node, seen))
    for recording, tool, result in results:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                continue
        walk(result, "", f"recording:{recording}:{tool}")
    return out


def _members(value: Any) -> list:
    """The values a cell names that a Run must be able to write: each of a list or a one_of, never a not."""
    if isinstance(value, dict):
        return [] if "not" in value else list(value.get("one_of") or [])
    return list(value) if isinstance(value, list) else [value]


class World:
    """The Starting state, the policy and the user turns, for finding rows and sourcing values."""

    def __init__(self, state: dict, policy_text: str = "", facts: Iterable[Any] = (), tools: Iterable[dict] = (),
                 columns: Optional[dict] = None, signatures: Optional[dict] = None, reach: Optional[Reach] = None):
        self.state = {table: rows for table, rows in (state or {}).items() if isinstance(rows, dict)}
        # The schema's columns per table, for a table the Starting state holds no row of.
        self.declared = {table: list(names) for table, names in (columns or {}).items()}
        self.policy = policy_text or ""
        self.facts = list(facts)
        self.tools = {row["name"]: row for row in tools if isinstance(row, dict) and row.get("name")}
        # Per tool, what its recorded calls passed (spec/effects.py signatures): a called item no
        # recorded call could have made is refused.
        self.signatures = dict(signatures or {})
        # What tools can write (D387); None checks no gated value for reachability.
        self.reach = reach
        self._values: Optional[dict[str, tuple[str, str]]] = None
        self._by_name: Optional[dict[str, dict[str, tuple[str, str, Any]]]] = None
        # The item being checked, so a value can be read against its gate and its row.
        self._kept: dict = {}

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

    # --- reaching (D387) -----------------------------------------------------------------------
    def _start_values(self) -> dict[str, dict[str, tuple[str, str, Any]]]:
        """Per field name, every scalar some start row holds under it, at any depth: (table, row, value) first
        seen."""
        if self._by_name is None:
            out: dict[str, dict[str, tuple[str, str, Any]]] = {}

            def walk(node: Any, key: str, at: tuple) -> None:
                if isinstance(node, dict):
                    for name, value in node.items():
                        walk(value, str(name), at)
                elif isinstance(node, list):
                    for value in node:
                        walk(value, key, at)
                elif node is not None and norm(node):
                    out.setdefault(key, {}).setdefault(reach_key(node), at + (node,))
            for table, rows in sorted(self.state.items()):
                for key, row in rows.items():
                    walk(row, "", (table, key))
            self._by_name = out
        return self._by_name

    def reach_of(self, table: str, field: Any, value: Any) -> Optional[ValueSource]:
        """Where a cell's value is reachable from, or None: the user's own words (no echo fact), a tool's
        learned writes to that table and field, a write tool's result in a recording of the Task, a start
        row holding it under that field name. A null is always reachable."""
        if value is None:
            return ValueSource(kind="world", ptr={"removed": True})
        text, key, name = norm(value), reach_key(value), leaf_of(field)
        return self._user_reach(text) or self._tool_reach(table, name, key) or self._start_reach(name, key) \
            or self._arg_reach(table, name, text)

    def _user_reach(self, target: str) -> Optional[ValueSource]:
        """The user's own words fixing the value: a fact that is not an echo of the agent's proposal."""
        for fact in self.facts:
            if not getattr(fact, "echo", False) and _contains(_flat(fact.text), target):
                return ValueSource(kind="user_turn", ptr={"recording": fact.source.recording,
                                                          "turn": fact.source.turn, "fact": fact.id})
        return None

    def _tool_reach(self, table: str, name: str, target: str) -> Optional[ValueSource]:
        """A tool's learned writes to that table and field, else a recorded write result under that field."""
        reach = self.reach or Reach()
        for seen in ((reach.tools.get(table) or {}).get(name) or {}, reach.recorded.get(name) or {}):
            if target in seen:
                return ValueSource(kind="tool_result", ptr={"field": name, "seen": seen[target][1]})
        return None

    def _unfound(self, value: Any) -> bool:
        """A value no user turn, policy line, world row or tool write holds."""
        return self.source(value) is None and not self._tool_written(value)

    def _tool_written(self, value: Any) -> bool:
        """A value some tool's learned writes or a recorded write result hold under any field: found, so a
        gated value the policy spells another way can be re-gated to what the tool writes (D387)."""
        reach, key = self.reach or Reach(), reach_key(value)
        fields = [seen for table in reach.tools.values() for seen in table.values()] + list(reach.recorded.values())
        return any(key in seen for seen in fields)

    def _start_reach(self, name: str, target: str) -> Optional[ValueSource]:
        at = (self._start_values().get(name) or {}).get(target)
        return ValueSource(kind="world", ptr={"table": at[0], "row": at[1]}) if at else None

    def _arg_reach(self, table: str, name: str, target: str) -> Optional[ValueSource]:
        """A field a tool copies from an argument holds what the call passed: any value the world holds."""
        fed = ((self.reach or Reach()).fed.get(table) or {}).get(name)
        held = self._world_values().get(target) if fed else None
        return ValueSource(kind="tool_result", ptr={"field": name, "seen": f"arg:{fed[0]}", "table": held[0],
                                                    "row": held[1]}) if held else None

    def candidates(self, table: str, field: Any) -> list:
        """The reachable values of a field a finding lists: tools' writes, recorded results, start rows."""
        name, reach = leaf_of(field), self.reach or Reach()
        out: dict[str, Any] = {}
        for seen in ((reach.tools.get(table) or {}).get(name) or {}, reach.recorded.get(name) or {}):
            for key, (raw, _) in sorted(seen.items()):
                out.setdefault(key, raw)
        start = self._start_values().get(name) or {}
        for key in sorted(start, key=lambda key: (start[key][0] != table, key)):
            out.setdefault(key, start[key][2])
        return list(out.values())[:MAX_CANDIDATES]

    def unreachable(self, item: dict) -> list[tuple[str, Any]]:
        """(field, value) per value a row_is expects or a row_new holds that no source reaches; none
        without a Reach."""
        if self.reach is None or item.get("kind") not in ("row_is", "row_new"):
            return []
        cells = item.get("expect") if item["kind"] == "row_is" else item.get("where")
        return [(str(field), one) for field, value in sorted((cells or {}).items()) for one in _members(value)
                if one is not None and self.reach_of(item["table"], field, one) is None]

    def unreachable_line(self, item: dict, found: list[tuple[str, Any]]) -> str:
        """The finding for an item's unreachable values: the item, each field, the value and its candidates."""
        lines = []
        for field, value in found:
            listed = self.candidates(item["table"], field)
            lines.append(f"item {item['id']} {field}: value {json.dumps(value)} {UNREACHABLE}: no tool writes "
                         f"it there, no user turn fixes it, no start row holds it under {leaf_of(field)}; "
                         f"reachable values: {json.dumps(listed) if listed else 'none found'}")
        return "; ".join(lines)

    # --- checking one item ---------------------------------------------------------------------
    def check(self, item: Any, known_facts: dict, unwitnessed: Callable[[Any], bool] = lambda fact: False
              ) -> tuple[Optional[dict], str]:
        """The item as code keeps it (row ids resolved, gate, weight, sources), with a note when its
        gate is weighed, or why it is refused."""
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
        if gate and item["kind"] == "called" and is_transfer(str(item.get("tool") or "")) \
                and not is_transfer(kept["policy_line"] or ""):
            gate = False  # a hand-off gates only where its policy line names one; else it is weighed
        kept.update(gate=gate, weight=float(weight))
        self._kept = kept
        why = self._kind_rules(item, kept)
        if why:
            return None, why
        return self._reachable(kept, _weighed_gate(kept, known_facts))

    def _reachable(self, kept: dict, note: str) -> tuple[Optional[dict], str]:
        """The kept item and its note, or the refusal when a gated value no source reaches (D387); a weighed
        item naming one keeps the finding as a warning in its note."""
        found = self.unreachable(kept)
        if found and kept["gate"]:
            return None, (f"gate value unreachable: {self.unreachable_line(kept, found)}. A gated value must be "
                          "one a tool can write or the user fixed: gate one of the reachable values, or weigh the "
                          "item (gate false)")
        if found:
            warning = f"warning: {self.unreachable_line(kept, found)}; weighed, so it costs score, not the Verdict"
            note = f"{note}; {warning}" if note else warning
        return kept, note

    def _kind_rules(self, item: dict, kept: dict) -> str:
        """Why the item's confirms or its kind's own rules refuse it, else empty."""
        return self._confirms(item, kept) or getattr(self, f"_{item['kind']}")(item, kept)

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
        return _unfounded_gate(item, [known[f] for f in fact_ids])

    def _confirms(self, item: dict, kept: dict) -> str:
        """An item's confirms, when it has one: {table, find} naming the one row whose read confirms a
        conditional ask's claim about the world (H2); kept with the row resolved."""
        confirms = item.get("confirms")
        if confirms is None:
            return ""
        if not isinstance(confirms, dict) or not isinstance(confirms.get("find"), dict) or not confirms.get("table"):
            return "confirms is {table, find: {field: value}}: the one row whose read confirms the condition"
        table = str(confirms["table"])
        if table not in self.state:
            return f"confirms: no table {table}"
        rows = self.find(table, confirms["find"])
        if len(rows) != 1:
            return f"confirms: find matches {len(rows)} rows of {table}; name exactly one"
        kept["confirms"] = {"table": table, "row": rows[0]}
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
            if one is not None and self._unfound(one):
                return None, (f"{what}: value {json.dumps(one)} is not in the world, the policy or a user turn, "
                              "nor what a tool writes; find it with the tools, or demand another field")
            if one is not None and self._echo_only(one):
                return None, (f"{what}: value {json.dumps(one)} rests only on echo facts (the user said yes to the "
                              "agent's proposal); a gate value needs the user's own words, the policy or the row "
                              "the item names: make the item scored (gate false) or demand another field")
        return value, ""

    def _echo_only(self, value: Any) -> bool:
        """A gate item's value whose only user source is an echo fact, found in neither the policy nor
        the row the item names. A scored item may rest on an echo fact."""
        if not self._kept.get("gate"):
            return False
        target = norm(value)
        facts = [fact for fact in self.facts if _contains(_flat(fact.text), target)]
        if not facts or not all(getattr(fact, "echo", False) for fact in facts):
            return False
        if _contains(_flat(self.policy), target):
            return False
        row = self.row(self._kept["table"], self._kept["row"]) if self._kept.get("row") else None
        return not (row and any(norm(leaf) == target for leaf in _leaves(row)))

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
        expect, free = item.get("expect") or {}, item.get("free") or {}
        if why or not isinstance(expect, dict) or not set(expect) - set(kept["find"]):
            return why or ("expect names a field the row holds after a correct Run beside the find key: "
                           "{field: value}; a row checked on its find key alone checks nothing")
        if not isinstance(free, dict) or any(not isinstance(r, str) or not r.strip() or "\n" in r
                                             for r in free.values()):
            return "free is {field: one line on why a tool sets it to a value you cannot find}"
        for field, value in expect.items():
            why = self._field(kept["table"], field) or self._value(value, f"{field}")[1]
            if why:
                return why
        for field in free:
            why = self._field(kept["table"], field) or self._named_free(kept["table"], kept["row"], str(field))
            if why or str(field) in expect:
                return why or f"{field} is in expect and in free; name it once"
        kept["expect"] = dict(expect)
        if free:
            kept["free"] = {str(field): " ".join(reason.split()) for field, reason in free.items()}
        return ""

    def _named_free(self, table: str, row_key: str, field: str) -> str:
        """Why a free field is refused: an Intent fact names a value this field holds elsewhere in the world.

        The values are those under the field's own key, or a key nested in it, in any row of any table,
        less what the row holds there now; a fact naming one of them names the field's new value.
        """
        now = path_value((self.state.get(table) or {}).get(row_key) or {}, field)
        keys = {field.split(".")[-1]} | set(_keys(now))
        held = {norm(leaf) for leaf in _leaves(now)}
        values = {value for value in self._values_under(keys) if len(value) >= 3} - held
        for fact in self.facts:
            text = _flat(fact.text)
            named = next((value for value in sorted(values) if value in text and _contains(text, value)), None)
            if named is not None:
                return (f"free {field}: fact {fact.id} names a value this field holds elsewhere in the world; "
                        "put the value the Intent names in expect")
        return ""

    def _values_under(self, keys: set[str]) -> set[str]:
        """Every scalar the world holds under one of these keys, at any depth of any row."""
        out: set[str] = set()

        def walk(node: Any, key: Optional[str]) -> None:
            if isinstance(node, dict):
                for name, value in node.items():
                    walk(value, str(name))
            elif isinstance(node, list):
                for value in node:
                    walk(value, key)
            elif node is not None and key in keys:
                out.add(norm(node))
        for rows in self.state.values():
            walk(rows, None)
        return out

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
        recorded = (self.signatures.get(item["tool"]) or {}).get("args") or {}
        for name, value in args.items():
            if allowed and name not in allowed:
                return f"{item['tool']} takes {', '.join(allowed)}"
            why = self._arg_shape(item["tool"], name, value, recorded.get(name))
            if why == OBJECTS:
                why = self._objects(name, value, recorded[name])
            elif not why:
                why = self._value(value, name)[1]
            if why:
                return why
        why = self._never_equal(item["tool"], args)
        if why:
            return why
        kept.update(tool=item["tool"], args=dict(args))
        return ""

    def _arg_shape(self, tool: str, name: str, value: Any, recorded: Optional[dict]) -> str:
        """Why no recorded call of the tool passed an argument like this one; OBJECTS for a list of objects
        the recordings show, which is checked on its key field; empty when the shape is one recorded."""
        if not recorded or value_set_of(value):
            return ""
        why = _type_refusal(tool, name, value, set(recorded.get("types") or []))
        keys = recorded.get("required_keys") or []
        if why or not keys or not isinstance(value, (list, dict)):
            return why
        entries = value if isinstance(value, list) else list(value.values())
        if entries and all(isinstance(one, dict) for one in entries):
            return OBJECTS
        return (f"{tool} {name}: every recorded call passes objects with keys {', '.join(keys)}; "
                "write each entry as such an object")

    def _objects(self, name: str, value: Any, recorded: dict) -> str:
        """A list of objects compared on its key field: each object carries the keys every recorded one did,
        no key none did, and its key field names a value the world or the Intent holds."""
        need, known = set(recorded.get("required_keys") or []), set(recorded.get("keys") or [])
        for one in value if isinstance(value, list) else value.values():
            if not need <= set(one) or not set(one) <= known:
                return (f"{name}: each object carries the keys {', '.join(sorted(need))}"
                        + (f" (and may carry {', '.join(sorted(known - need))})" if known - need else ""))
            for key in recorded.get("key_fields") or []:
                why = self._value(one[key], f"{name}.{key}")[1]
                if why:
                    return why
        return ""

    def _never_equal(self, tool: str, args: dict) -> str:
        for one, two in (self.signatures.get(tool) or {}).get("never_equal") or []:
            if one in args and two in args and norm(args[one]) and norm(args[one]) == norm(args[two]):
                return f"{tool}: no recorded call gives {one} and {two} the same value"
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

    def _narrowed(self, evidence: list, reason: Any) -> str:
        """A ref narrower than the window (every agent turn up to the first transfer) needs its reason."""
        if all(str(ref) == WINDOW for ref in evidence):
            return ""
        if isinstance(reason, str) and reason.strip() and "\n" not in reason.strip():
            self._kept["evidence_why"] = " ".join(reason.split())
            return ""
        return (f"evidence narrower than {WINDOW} needs evidence_why: one line on why the answer can only "
                f"come there; else leave evidence out and every agent turn up to a transfer is read")

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

    def _computed(self, answer: str, cited: Optional[dict]) -> str:
        """Why an anchor answer is refused: it states a number that is no value of the cited row and is
        not in the policy, so the writer computed it. The answer is looked up, never worked out."""
        row = self.row(cited["table"], cited["key"]) or {} if cited else {}
        found = {n for leaf in _leaves(row) for n in _numbers(str(leaf))} | _numbers(self.policy)
        found |= _numbers(str(cited["key"])) if cited else set()
        made = sorted(_numbers(answer) - found)
        if made:
            return (f"anchor.answer states {', '.join(f'{n:g}' for n in made)}, which is no value of the cited row "
                    "and not in the policy; state the row's own value or a policy line, never a computed one")
        return ""

    def _judge(self, item: dict, kept: dict) -> str:
        question = str(item.get("question") or "").strip()
        anchor = item.get("anchor")
        if not question.endswith("?"):
            return "question is one yes or no question ending with '?'"
        if not isinstance(anchor, dict) or not str(anchor.get("answer") or "").strip():
            return "anchor is {answer, accepted: [...], reject: [...], row}: what a met checkpoint states"
        answer = str(anchor["answer"]).strip()
        evidence = item.get("evidence") or [WINDOW]
        why = self._evidence(evidence) or self._narrowed(evidence, item.get("evidence_why"))
        if why:
            return why
        cited, why = self._anchor_row(anchor, answer)
        policy = item.get("policy_line") and _flat(item["policy_line"]) in _flat(answer)
        if why and not (policy and anchor.get("row") is None):
            return why
        why = self._computed(answer, cited)
        if why:
            return why
        if cited is None:
            kept["gate"] = True  # a refusal is the item that decides: a silent Run must fail it (D330)
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


def _one_of_args(args: Any) -> bool:
    """Whether these called args carry a {"one_of": [...]} set the Verdict cannot match."""
    return isinstance(args, dict) and any(
        isinstance(value, dict) and len(value) == 1 and isinstance(value.get("one_of"), list)
        for value in args.values())


def atom_of(item: dict) -> Atom:
    """One item as one atom: gate and weight on it, the item itself and its provenance as the target.

    The atom's kind stays allowed, so no older reader takes it for a required write; the verdict
    decides it by its gate flag (runner/verdict.py): state items as the end state's cells, event
    items off the calls and the log, judge items by the judge's score. A called item naming a set
    of allowed values is the one exception: the Verdict exact-matches a called target, so the set
    compiles to a rule matching any member instead (spec/atoms.py, D352).
    """
    if item["kind"] in ("called", "not_called") and _one_of_args(item.get("args")):
        return called_set_atom(
            item["id"], item["tool"], dict(item.get("args") or {}), item["kind"] == "not_called",
            [item["tool"]], _describe(item), gate=bool(item["gate"]), weight=float(item["weight"]),
            extra={"item_kind": item["kind"], "from": _from(item)})
    target = {key: value for key, value in item.items() if key not in ("id", "gate", "weight", "fact_ids",
                                                                         "policy_line", "confirms")}
    target["from"] = _from(item)
    return Atom(id=item["id"], kind="allowed", description=_describe(item), target=target,
                judge=item["kind"] == "judge", gate=bool(item["gate"]), weight=float(item["weight"]))


def value_set_of(value: Any) -> bool:
    """A value set ({"one_of": [...]} or {"not": [...]}), not a value."""
    return isinstance(value, dict) and len(value) == 1 and next(iter(value)) in ("one_of", "not")


def _type_refusal(tool: str, name: str, value: Any, types: set) -> str:
    """Why a value's JSON type is none the recorded calls passed (a number stands for a number); else empty."""
    kind = type_of(value)
    numeric = {"int", "float"}
    if not types or kind in types or kind == "NoneType" or (kind in numeric and types & numeric):
        return ""
    return f"{tool} {name}: every recorded call passes {' or '.join(sorted(types))}, not {kind}"


def _cell_source(value: Any, world: World) -> Optional[ValueSource]:
    found = None
    values = next(iter(value.values())) if isinstance(value, dict) else value
    values = values if isinstance(values, list) else [values]
    for one in values:
        found = world.source(one) if one is not None else ValueSource(kind="world", ptr={"removed": True})
        if found is None:
            return None
    return found


def _source_of(item: dict, field: str, value: Any, world: World) -> Optional[ValueSource]:
    """A cell's source: where it is reachable from for a gated row_is (`_gate_source`), else `_cell_source`."""
    if item["gate"] and item["kind"] == "row_is":
        return _gate_source(item["table"], field, value, world)
    return _cell_source(value, world)


def _gate_source(table: str, field: str, value: Any, world: World) -> Optional[ValueSource]:
    """A gated cell's source where the world knows what tools can write (D387): where each value it names
    is reachable from, None when one is not; a `not` set, and a world with no Reach, as `_cell_source`."""
    if world.reach is None or (isinstance(value, dict) and (not value_set_of(value) or "not" in value)):
        return _cell_source(value, world)
    found = ValueSource(kind="world", ptr={"removed": True})
    for one in _members(value):
        found = world.reach_of(table, field, one)
        if found is None:
            return None
    return found


def _answers(item: dict, effects: dict) -> list[str]:
    """The tools that could answer one state item: for a row_is, every tool that always moves each column
    it expects on that table; for a row_new, every tool that always makes rows in that table."""
    if item["kind"] == "row_new":
        return [tool for tool, learned in effects.items() if item["table"] in (learned.get("makes") or [])]
    if item["kind"] != "row_is":
        return []
    wanted = {str(field).split(".")[0] for field in item.get("expect") or {}} - set(item.get("find") or {})
    return [tool for tool, learned in effects.items() if wanted and wanted <= {
        column["column"] for column in learned.get("columns") or [] if column["table"] == item["table"]}]


def expected_tools(items: list[dict], effects: dict) -> list[str]:
    """The write tools an end state's items expect to run.

    A called item names its tool. A state item implies the one tool that alone answers it (`_answers`);
    an item two tools could answer implies neither.
    """
    out = {item["tool"] for item in items if item["kind"] == "called" and item.get("tool") in effects}
    for item in items:
        fits = _answers(item, effects)
        if len(fits) == 1:
            out.add(fits[0])
    return sorted(out)


def _declared_rows(items: list[dict]) -> tuple[dict[str, set[str]], set[tuple]]:
    """(table to the rows a row_is declares, the (table, row, column) cells expect)."""
    declared: dict[str, set[str]] = {}
    for item in items:
        if item["kind"] == "row_is" and item.get("row") is not None:
            declared.setdefault(item["table"], set()).add(item["row"])
    expected = {(item["table"], item["row"], str(field).split(".")[0]) for item in items
                if item["kind"] in ("row_is", "row_keeps") for field in item.get("expect") or {}}
    return declared, expected


def effect_rules(items: list[dict], effects: Optional[dict], alt: Any = None) -> list[dict]:
    """Allowed rules for what each write tool an end state expects always moves (spec/effects.py).

    The items are those of the end state `alt` (those without `alt` and those with it). On a table whose
    rows the end state declares, the rule frees the column on those rows; on a table it declares no row
    of (a count on a related row), on every row of it. A column a cell expects stays checked by its
    cell; no rule is added there.
    """
    items = [item for item in items if item.get("alt") in (None, alt)]
    declared, expected = _declared_rows(items)
    rules: dict[tuple, dict] = {}
    for tool in expected_tools(items, effects or {}):
        for column in effects[tool].get("columns") or []:
            table, field = column["table"], column["column"]
            for row in sorted(declared.get(table) or ()) or [None]:
                rule = {"table": table, "field": field, "why": f"free: tool effect: {tool}"}
                if (table, row, field) not in expected:
                    rules.setdefault((table, row, field), dict(rule, row_id=row) if row is not None else rule)
    return [rules[key] for key in sorted(rules, key=lambda key: tuple(str(part) for part in key))]


def end_states_of(items: list[dict], world: World, action_tables: Iterable[str] = (),
                  effects: Optional[dict] = None) -> list[EndState]:
    """One EndState per alternative: the state items without `alt` in each, those with `alt` in theirs.

    Every EndState allows the action tables (a call's record, not the world's change) and what each write
    tool it expects always moves (`effect_rules`); nothing else outside its cells and new rows may move,
    which is the sanity item. With no state item at all there is one EndState with no cells: the Run
    must leave the world as it found it.
    """
    state = [item for item in items if item["kind"] in STATE_KINDS]
    alts = sorted({item["alt"] for item in state if item.get("alt") is not None}) or [None]
    allowed = [{"table": table} for table in sorted(set(action_tables))]
    out = []
    for alt in alts:
        cells: list[ExpectedCell] = []
        new_rows: list[dict] = []
        free: list[dict] = effect_rules(items, effects, alt)
        for item in state:
            if item.get("alt") not in (None, alt):
                continue
            if item["kind"] == "row_new":
                new_rows.append({"table": item["table"], "where": item["where"], "count": item["count"],
                                 "item": item["id"], "gate": bool(item["gate"]),
                                 "weight": float(item["weight"])})
                continue
            free += [{"table": item["table"], "row_id": item["row"], "field": field,
                      "why": f"free: {item['id']}: {reason}"} for field, reason in (item.get("free") or {}).items()]
            row_source = ValueSource(kind="world", ptr={"table": item["table"], "row": item["row"]})
            for field, value in item["expect"].items():
                cells.append(ExpectedCell(table=item["table"], row_id=item["row"], field=field, value=value,
                                          source=_source_of(item, field, value, world), row_source=row_source,
                                          gate=bool(item["gate"]), weight=float(item["weight"]) / len(item["expect"])))
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
            "free_fields": sum(len(item.get("free") or {}) for item in items),
            "end_states": len({item.get("alt", 0) for item in items if item.get("kind") in STATE_KINDS}) or
            int(any(item.get("kind") == SANITY for item in items)),
            "anchors_without_lookup": sum(1 for item in items if item.get("kind") == "judge"
                                          and not (item.get("anchor") or {}).get("row")),
            "policy_backed": sum(1 for item in items if item.get("policy_line"))}


__all__ = ["CONFIRM_TURN", "DEFAULT_GATE", "EVENT_KINDS", "ITEM_KINDS", "Reach", "SANITY", "SANITY_ID", "STATE_KINDS",
           "UNREACHABLE", "World", "effect_reach", "reach_of_inputs", "recorded_reach", "atom_of", "because_of", "contradictions", "counts_of", "effect_rules", "end_states_of",
           "expected_tools",
           "norm", "sanity_atom", "sanity_item"]
