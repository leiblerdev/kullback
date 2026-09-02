"""Everything an atom predicate may look at, and the confinement gate that certifies it before it
runs (D39, design section 7)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.runner.canon import canon_value, compare
from kullback.runner.confinement import SAFE_BUILTINS, confine
from kullback.runner.records import Run

CONFIRM_WORDS = ("yes", "confirm", "go ahead")


def _reply(payload: dict) -> dict:
    """The model reply a payload carries, whether nested under `reply` (loop.py's shape) or not."""
    reply = payload.get("reply")
    return reply if isinstance(reply, dict) else payload


def _text_of(payload: dict) -> str:
    """The text of a message event, in the shapes loop.py and RecordedModel both write."""
    for value in (payload.get(key) for key in ("content", "text", "message", "reply")):
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            return value
    return ""


# --- the world a predicate sees ---

def _scalar_canon(canon: Any) -> Any:
    """canon.py's rules by default (D39); a caller may pass the module or its own callable.

    There is one canonicalizer and route.py keys its recordings with it, so the default here is
    canon.py itself rather than the identity, which used to leave a Verdict comparing raw values.
    """
    if canon is None:
        return canon_value
    for attr in ("canon_value", "canonicalize"):
        function = getattr(canon, attr, None)
        if callable(function):
            return function
    return canon if callable(canon) else canon_value


class AtomContext:
    """Everything an atom predicate may look at: the writes, the messages, and the End state."""

    def __init__(self, run: Run, canon: Any = None, write_tools: Optional[Iterable[str]] = None,
                 schema: Any = None, rules: Any = None, equivalence: Any = None) -> None:
        self.run = run
        self._canon = _scalar_canon(canon)
        self.write_tools = set(write_tools) if write_tools else None
        self.exempt = {f"{c.table}.{c.name}" for c in getattr(schema, "columns", [])
                       if getattr(c, "class_", None) == "exempt"}
        self.semantic = {f"{c.table}.{c.name}" for c in getattr(schema, "columns", [])
                         if getattr(c, "class_", None) == "semantic"}
        self.rules = rules  # the customer's CanonRules, so a Verdict canonicalizes their way (D39)
        self.equivalence = equivalence  # the EquivalenceTable a semantic pair is settled by (D84)
        self.comparisons: list[Any] = []
        self.calls: list[dict] = []
        self.assistant: list[tuple[int, str]] = []
        self.user: list[tuple[int, str]] = []
        self.start_state: dict = {}
        self.end_state: dict = {}
        self.covered: set[int] = set()
        self.marking = True
        self._scan()

    def _scan(self) -> None:
        inline: list[dict] = []
        for event in self.run.events:
            payload = event.payload or {}
            if event.type == "tool_call":
                self.calls.append({"i": len(self.calls), "idx": event.idx, "id": payload.get("id"),
                                   "name": payload.get("name", ""), "error": None,
                                   "args": payload.get("args") or payload.get("arguments") or {}})
            elif event.type == "tool_result":
                for call in reversed(self.calls):
                    if payload.get("id") in (None, call["id"]):
                        call["error"] = payload.get("error")
                        break
            elif event.type == "model_call":
                text = _text_of(payload)
                if text:
                    self.assistant.append((event.idx, text))
                # A call carried on the model reply happened where that reply is, not before the Run
                # started: a sequence rule reads these positions (D43 case 3). loop.py nests the
                # reply (and its tool_calls) under payload["reply"], so the fallback has to unwrap
                # it the same way _text_of does, or a reply-only record of a call is silently lost.
                inline.extend((event.idx, made) for made in (_reply(payload).get("tool_calls") or []))
            elif event.type == "user_turn":
                self.user.append((event.idx, _text_of(payload)))
            elif event.type == "stop":
                self.start_state = payload.get("start_state") or self.start_state
                self.end_state = payload.get("end_state") or self.end_state
        if not self.calls:
            self.calls = [{"i": i, "idx": idx, "id": c.get("id"), "name": c.get("name", ""), "error": None,
                           "args": c.get("args") or c.get("arguments") or {}}
                          for i, (idx, c) in enumerate(inline)]

    # canonicalization, shared with route.py by construction (D39)
    def c(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.c(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.c(v) for v in value]
        return self._canon(value)

    def t(self, value: Any) -> str:
        return " ".join(str(self.c(value)).split()).lower()

    # helpers the Verifier's predicate_src calls
    def wrote(self, tool: str, **fields: Any) -> bool:
        hit = False
        for call in self.calls:
            if call["name"] != tool or call["error"]:
                continue
            args = call["args"] or {}
            if all(self.c(args.get(k)) == self.c(v) for k, v in fields.items()):
                hit = True
                if self.marking:
                    self.covered.add(call["i"])
        return hit

    def called(self, tool: str) -> bool:
        """A call the Environment rejected had no effect, so it is not a call that happened (D45)."""
        return any(call["name"] == tool and not call["error"] for call in self.calls)

    def attempted(self, tool: str) -> bool:
        """Every call to this tool, the rejected ones included, for a rule about what was tried."""
        return any(call["name"] == tool for call in self.calls)

    def asked(self, *words: str) -> bool:
        return any("?" in text and all(self.t(w) in self.t(text) for w in words)
                   for _, text in self.assistant)

    def communicated(self, *values: Any) -> bool:
        return any(all(self.t(v) in self.t(text) for v in values) for _, text in self.assistant)

    def user_said(self, *words: str) -> bool:
        return any(any(self.t(w) in self.t(text) for w in words) for _, text in self.user)

    def user_confirmed_before(self, tool: str, *words: str) -> bool:
        first = next((call["idx"] for call in self.calls if call["name"] == tool), None)
        if first is None:
            return True
        needles = words or CONFIRM_WORDS
        return any(idx < first and any(self.t(w) in self.t(text) for w in needles)
                   for idx, text in self.user)

    def value(self, table: str, row_id: str, field: Optional[str] = None) -> Any:
        """The End state value as it stands.

        Uncanonicalized on purpose: the canonicalizer turns 25 and True into canonical strings, so a
        canonicalized value could never equal a number or a boolean literal in a predicate. Use
        eq() to compare two values under the canonicalizer (D39).
        """
        row = (self.end_state.get(table) or {}).get(row_id)
        return row if field is None else (row or {}).get(field)

    def eq(self, left: Any, right: Any) -> bool:
        """Compare two values the way the rest of the Verdict compares them (D39)."""
        return self.c(left) == self.c(right)

    def same(self, column: str, before: Any, after: Any) -> bool:
        """Whether two values of one End state column count as the same value.

        A semantic column is not settled by string equality: D84 sends the pair to the customer's
        EquivalenceTable, and a pair the table does not hold comes back unresolved rather than
        judged here, because verdict.py never calls a model (D91). Every comparison is kept so the
        Verdict can say which pairs it rested on and which are still open.
        """
        if column not in self.semantic:
            return self.c(before) == self.c(after)
        comparison = compare(before, after, "semantic", rules=self.rules,
                             table=self.equivalence, column=column)
        self.comparisons.append(comparison)
        return comparison.equal

    def changed(self, table: str, row_id: str, field: str) -> bool:
        before = ((self.start_state.get(table) or {}).get(row_id) or {}).get(field)
        after = ((self.end_state.get(table) or {}).get(row_id) or {}).get(field)
        return not self.same(f"{table}.{field}", before, after)

    def diff(self) -> dict:
        """The End state diff after canonicalization, with exempt columns dropped (D39, D73)."""
        out: dict = {}
        for table in sorted(set(self.start_state) | set(self.end_state)):
            before_rows = self.start_state.get(table) or {}
            after_rows = self.end_state.get(table) or {}
            for row_id in sorted(set(before_rows) | set(after_rows), key=str):
                before, after = before_rows.get(row_id), after_rows.get(row_id)
                fields = {}
                for key in sorted(set(before or {}) | set(after or {}), key=str):
                    if f"{table}.{key}" in self.exempt:
                        continue
                    raw_before, raw_after = (before or {}).get(key), (after or {}).get(key)
                    was, now = self.c(raw_before), self.c(raw_after)
                    if not self.same(f"{table}.{key}", raw_before, raw_after):
                        fields[key] = {"before": was, "after": now}
                if fields or (before is None) != (after is None):
                    out[f"{table}.{row_id}"] = {"present_before": before is not None,
                                                "present_after": after is not None, "fields": fields}
        return out

    def write_calls(self) -> list[dict]:
        if self.write_tools is None:
            return [c for c in self.calls if c["i"] in self.covered]
        return [c for c in self.calls if c["name"] in self.write_tools and not c["error"]]

    def extra_writes(self) -> list[dict]:
        return [c for c in self.write_calls() if c["i"] not in self.covered]

    def writes_count(self) -> int:
        return len(self.write_calls())

    def transcript(self) -> list[dict]:
        """The Run as a policy predicate reads it: role, content and tool calls, in event order."""
        turns: list[tuple[int, dict]] = [
            (idx, {"role": "assistant", "content": text, "tool_calls": []}) for idx, text in self.assistant]
        turns += [(idx, {"role": "user", "content": text, "tool_calls": []}) for idx, text in self.user]
        turns += [(call["idx"], {"role": "assistant", "content": None,
                                 "tool_calls": [{"name": call["name"], "arguments": call["args"]}]})
                  for call in self.calls]
        return [turn for _, turn in sorted(turns, key=lambda pair: pair[0])]

    def env(self) -> dict:
        # A fresh copy of the builtins per evaluation: _evaluate's dict(env) is shallow, so handing
        # out the module-level dict would let one atom pop a name every later atom and every later
        # Verdict in this process would then miss.
        return {"__builtins__": dict(SAFE_BUILTINS), "wrote": self.wrote, "called": self.called,
                "attempted": self.attempted, "eq": self.eq,
                "asked": self.asked, "communicated": self.communicated, "user_said": self.user_said,
                "user_confirmed_before": self.user_confirmed_before, "value": self.value,
                "changed": self.changed, "diff": self.diff, "extra_writes": self.extra_writes,
                "writes_count": self.writes_count, "write_calls": self.write_calls,
                "calls": [dict(c) for c in self.calls], "transcript": self.transcript(),
                "messages": [t for _, t in self.assistant], "user_turns": [t for _, t in self.user],
                "start_state": self.start_state, "end_state": self.end_state, "canon": self.c}


def gate(source: str) -> list[str]:
    """Certify one predicate before it is compiled: no imports, no dunder walk, no denied name.

    Restricting `__builtins__` is not enough on its own. `wrote.__globals__` hands a predicate this
    module's globals, `().__class__.__base__.__subclasses__()` walks every loaded class, and a bound
    helper's `__self__` reaches the Run, so an atom could read or write anything the process can.
    The Builder gates a predicate the same way when it compiles one, but a Verifier can arrive from
    disk, so the Runner gates it again (design section 7: the Verdict is code without exception).
    runner/confinement.py holds the actual check, so gates/confinement.py's constraint gate cannot
    drift from this one.
    """
    return confine(source)


def _evaluate(source: str, env: dict) -> bool:
    """Run one atom predicate that gate() has already certified."""
    if "\n" in source.strip() or source.strip().startswith("def "):
        namespace = dict(env)
        exec(compile(source, "<atom>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        if check is None:
            raise ValueError("a multi-line predicate must define check()")
        return bool(check())
    return bool(eval(compile(source, "<atom>", "eval"), dict(env)))  # noqa: S307
