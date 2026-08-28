"""Decide one Run's pass or fail from its End state against a Task's Verifier, in code only (D43, D46, D94)."""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.shared.canon import canon_value, compare, record_use
from harness.shared.records import Atom, Run, Verdict, Verifier

VERDICT_VERSION = "1"
EVENT_TYPES = {"model_call", "tool_call", "tool_result", "user_turn", "error", "stop"}
MUST_HOLD = {"required", "question", "communicate", "hard"}
TRANSFER_HINTS = ("transfer", "escalate", "handoff", "hand_off")
GAVE_UP = {"transfer", "transferred", "agent_transfer", "gave_up", "no_action"}
ENV_ERROR_REASONS = {"env_error", "environment_error"}
CONFIRM_WORDS = ("yes", "confirm", "go ahead")
# The same names policy.py certifies for a compiled predicate at build time, so a predicate that
# passed its positive and negative test cannot NameError here and be skipped as a broken atom.
# The Runner may not import the Builder (D89), so the list is repeated; tests/test_verdict.py
# asserts the two stay a superset relationship.
SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
    "max", "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "KeyError", "TypeError", "ValueError",
)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
# The build-time gate policy.py runs, repeated here because a Verifier can reach the Runner from
# disk or from a Builder that edited itself (D69), so the only gate that counts is this one.
DENIED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr",
})
DENIED_ATTRS = frozenset({"format", "format_map"})  # "{0.__class__}".format(x) is an attribute walk
# How each judge use says "this holds", "this does not" and "I did not decide" (judge.py's _USES).
JUDGE_HOLDS = {"pass", "equivalent", "acceptable", "good_reference"}
JUDGE_FAILS = {"fail", "not_equivalent", "unacceptable", "bad_reference"}
CAUSES = {"candidate", "environment", "simulated_user", "undetermined"}


# --- loading a stored Run ---

def load_run(source: Any) -> Run:
    """Read a Run from a JSONL path, a dict or a Run; header lines, event lines and a footer all work."""
    if isinstance(source, Run):
        return source
    if isinstance(source, dict):
        return Run.model_validate(source)
    path = Path(source)
    head: dict = {}
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") in EVENT_TYPES:
            events.append(obj)
        else:
            events.extend(obj.pop("events", []) or [])
            head.update(obj)
    extra = {k: v for k, v in head.items() if k not in Run.model_fields}
    if extra:
        events.append({"type": "stop", "payload": extra})
    head = {k: v for k, v in head.items() if k in Run.model_fields}
    head.setdefault("run_id", path.stem)
    head["events"] = [dict(e, idx=e.get("idx", i)) for i, e in enumerate(events)]
    return Run.model_validate(head)


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
        self.tracked = False
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
                # started: a sequence rule reads these positions (D43 case 3).
                inline.extend((event.idx, made) for made in (payload.get("tool_calls") or []))
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
        if self.marking:
            self.tracked = True
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
        return {"__builtins__": SAFE_BUILTINS, "wrote": self.wrote, "called": self.called,
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
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"does not parse: {error.msg}"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad.append("imports a module")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENIED_ATTRS:
                bad.append(f"touches {node.attr}")
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            bad.append(f"uses {node.id}")
    return sorted(set(bad))


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


def _judge_says(result: Any) -> Optional[bool]:
    """Does this judge result say the atom holds? None means the judge did not decide (D76).

    A JudgeResult's verdict is a word, not a flag: `fail` and `abstain` are both non-empty strings,
    so reading one as a bool passed every judged Run. bool() is never the answer here, and an
    abstain is never a pass: D76 sends it to a person.
    """
    if isinstance(result, bool):
        return result
    data = result if isinstance(result, dict) else {
        key: getattr(result, key, None) for key in ("pass", "passed", "verdict")}
    for key in ("pass", "passed"):
        if isinstance(data.get(key), bool):
            return data[key]
    word = str(data.get("verdict") or "").strip().lower()
    if word in JUDGE_HOLDS:
        return True
    if word in JUDGE_FAILS:
        return False
    return None  # abstain, undetermined, an unknown word, or a shape this code does not read


def _env_marks(run: Run, flagged_tools: Iterable[str]) -> list[str]:
    """D88 code-first marks: assisted, fact_unavailable, a flagged or low-fidelity tool, an overlay miss."""
    flagged = set(flagged_tools or ())
    marks: list[str] = ["env_mark:assisted"] if run.assisted else []
    for event in run.events:
        payload = event.payload or {}
        found = []
        if event.assisted or payload.get("assisted") or event.route == "llm":
            found.append("env_mark:assisted")
        tags = payload.get("tags") or []
        if (payload.get("fact_unavailable") or payload.get("tag") == "fact_unavailable"
                or "fact_unavailable" in tags):
            found.append("env_mark:fact_unavailable")
        if payload.get("overlay_miss"):
            found.append("env_mark:overlay_miss")
        if payload.get("name") in flagged:
            found.append(f"env_mark:flagged_tool:{payload['name']}")
        marks.extend(m for m in found if m not in marks)
    return marks


def _termination(run: Run) -> str:
    if run.termination_reason:
        return run.termination_reason
    for event in reversed(run.events):
        if event.type == "stop":
            return str((event.payload or {}).get("termination_reason") or "")
    return ""


def _env_error(run: Run) -> bool:
    """An infrastructure failure inside the Environment, not a Candidate mistake."""
    if _termination(run).lower() in ENV_ERROR_REASONS:
        return True
    return any(event.type == "error" and ((event.payload or {}).get("environment")
               or (event.payload or {}).get("class") == "env_error") for event in run.events)


def _is_transfer(name: str) -> bool:
    return any(hint in (name or "").lower() for hint in TRANSFER_HINTS)


def _named_cause(cause_result: Any, notes: list[str]) -> Optional[str]:
    """The cause a judge named for this failure, with its cited spans kept beside it (D88)."""
    if cause_result is None:
        return None
    if isinstance(cause_result, str):
        word, spans = cause_result, []
    elif isinstance(cause_result, dict):
        word, spans = str(cause_result.get("verdict") or ""), cause_result.get("cited_spans") or []
    else:
        word = str(getattr(cause_result, "verdict", "") or "")
        spans = getattr(cause_result, "cited_spans", None) or []
    word = word.strip().lower()
    if word not in CAUSES:
        return None
    for span in spans:
        notes.append(f"judge_span:{span}")
    return word


def verdict(run_jsonl: Any, verifier: Verifier, canon: Any = None, judge_results: Optional[dict] = None,
            *, environment: Any = None, runner_version: Optional[str] = None,
            reference_path: Optional[Iterable[str]] = None, write_tools: Optional[Iterable[str]] = None,
            flagged_tools: Iterable[str] = (), schema: Any = None, cause_result: Any = None,
            rules: Any = None, equivalence: Any = None, workdir: Any = None,
            verdict_version: str = VERDICT_VERSION) -> Verdict:
    """Pass or fail one stored Run on its End state; never calls a model, judge atoms arrive as results (D76).

    `cause_result` is judge.py's answer for this Run's failure cause (D88); code marks the Run,
    the judge names the cause, and neither is computed here.
    """
    run = load_run(run_jsonl)
    context = AtomContext(run, canon, write_tools, schema, rules=rules, equivalence=equivalence)
    notes: list[str] = []
    failures: list[Atom] = []
    unevaluable: list[Atom] = []
    judge_used = False

    # Hard atoms run last: without a write-tool set write_calls() is what the other atoms covered,
    # so a hard atom placed first in the Verifier would see an empty list and hold vacuously.
    for atom in sorted(verifier.atoms, key=lambda a: a.kind == "hard"):
        holds: Optional[bool] = None
        refused = gate(atom.predicate_src) if atom.predicate_src and not atom.judge else []
        if atom.judge:
            if judge_results is None or atom.id not in judge_results:
                notes.append(f"judge_atom_unevaluated:{atom.id}")
            else:
                judge_used = True
                holds = _judge_says(judge_results[atom.id])
                if holds is None:
                    notes.append(f"judge_abstained:{atom.id}")
        elif not atom.predicate_src:
            notes.append(f"atom_without_predicate:{atom.id}")
        elif refused:
            notes.append(f"atom_rejected:{atom.id}:{refused[0]}")
        else:
            context.marking = atom.kind != "forbidden"
            try:
                holds = _evaluate(atom.predicate_src, context.env())
            except Exception as error:  # a broken atom is a Verifier defect, not a Candidate failure
                notes.append(f"atom_error:{atom.id}:{type(error).__name__}")
            finally:
                context.marking = True
        if holds is None:
            # An atom that had to hold and could not be checked leaves the Run not verdicted; a
            # counted pass here would hide a Verifier defect or an unrun judge (D76, D79).
            if atom.kind in MUST_HOLD:
                unevaluable.append(atom)
            continue
        if atom.kind in MUST_HOLD and not holds:
            failures.append(atom)
        elif atom.kind == "forbidden" and holds:
            failures.append(atom)

    for comparison in context.comparisons:
        # A semantic pair the judge settled makes this a judged Verdict (D84), and a pair nobody has
        # settled is named so the report can put it in front of a person rather than bury it.
        judge_used = judge_used or bool(getattr(comparison, "judge_used", False))
        if getattr(comparison, "route", None) == "unresolved":
            notes.append(f"semantic_unresolved:{comparison.key}")
        if workdir is not None:
            record_use(workdir, comparison, run.run_id, verifier.task_id)

    order = {atom.id: i for i, atom in enumerate(verifier.atoms)}
    failures.sort(key=lambda a: order.get(a.id, 0))
    unevaluable.sort(key=lambda a: order.get(a.id, 0))

    failing_atom = None
    not_verdicted = False
    if failures:
        first = failures[0]
        failing_atom = first.id
        notes.append(f"failing_atom:{first.id}: {first.description or first.predicate_src or first.kind}")
    elif unevaluable:
        first = unevaluable[0]
        failing_atom, not_verdicted = first.id, True
        notes.append(f"not_verdicted:{first.id}: a {first.kind} atom could not be evaluated")
    elif context.write_tools is not None:
        # A write on a Task whose Verifier asks for none is an extra write by definition, so the
        # check runs on the write-tool set alone and not on whether an atom happened to call wrote().
        extras = context.extra_writes()
        if extras:
            failing_atom = f"extra_write:{extras[0]['name']}"
            notes.append(f"failing_atom:{failing_atom}: write not required and not allowed by any atom")
    else:
        notes.append("side_effect_check_skipped")

    marks = _env_marks(run, flagged_tools)
    is_env_error = _env_error(run)
    passed = failing_atom is None and not is_env_error and not not_verdicted
    names = [call["name"] for call in context.calls]
    side_effects = context.writes_count()

    if is_env_error:
        klass, cause, suspected = "env_error", "environment", True
    elif not_verdicted:
        # An atom the Verifier could not evaluate is an immature Verifier, not a broken Environment:
        # design section 6 calls this state "Task not verdicted", so it is not blamed on the
        # Environment and does not count as a Candidate failure either.
        klass, cause, suspected = "not_verdicted", "undetermined", False
    elif passed:
        klass, cause, suspected = "pass", None, False
    else:
        # A transfer changes nothing even where mine.py classed the transfer tool as a write (D46).
        acting = [call for call in context.write_calls() if not _is_transfer(call["name"])]
        transferred = not acting and (
            any(_is_transfer(name) for name in names) or _termination(run).lower() in GAVE_UP
        )
        klass = "transferred_without_acting" if transferred else "fail"
        cause = _named_cause(cause_result, notes)  # code marks it, the judge names the cause (D88)
        suspected = bool(marks) or cause == "environment"
        if cause is None and not suspected:
            notes.append("cause_pending_judge")

    notes.extend(marks)
    notes.append(f"side_effects={side_effects}")
    notes.append(f"tool_calls={len(names)}")
    same_path = None if reference_path is None else names == list(reference_path)

    return Verdict(
        run_id=run.run_id,
        env_id=getattr(environment, "env_id", None) or run.env_id,
        # None, not "0": a placeholder string is truthy, so it would walk past regrade's presence
        # check and score a Run against versions nobody ever copied (D97).
        schema_version=getattr(environment, "schema_version", None),
        tools_version=getattr(environment, "tools_version", None),
        policy_version=getattr(environment, "policy_version", None),
        verifier_version=verifier.verifier_version,
        verdict_version=verdict_version,
        runner_version=runner_version,
        **{"pass": passed, "class": klass},
        failing_atom=failing_atom,
        same_path=same_path,
        cause=cause,
        judge_used=judge_used,
        environment_suspected=suspected,
        notes=notes,
    )
