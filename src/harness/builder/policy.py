"""Turns policy sentences into before-write Constraint predicates, with one rewrite pass for review and a judge atom when a rule must stay natural language (D43 case 3, D76)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.shared.records import (
    Constraint,
    ConstraintTests,
    GateResult,
    RawPtr,
    content_hash,
)

STAGE = "compile_policy"
_BULLET_CHARS = "-*+"
_CLOSERS = ('"', "'", ")", "]", "}")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "no.", "mr.", "mrs.", "ms.", "dr.", "approx.")
_DENIED_NAMES = frozenset(
    {"__import__", "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "locals", "vars", "getattr", "setattr", "delattr"}
)
# The gate is an allowlist, not a blocklist: a name or attribute nobody listed here is refused.
_SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
    "max", "min", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "KeyError", "TypeError", "ValueError",
)
_HELPER_NAMES = frozenset({"user_confirmed", "called_before", "said_before"})
_ALLOWED_ATTRS = frozenset({
    "add", "append", "copy", "count", "endswith", "extend", "find", "get", "index", "isalnum",
    "isalpha", "isdigit", "items", "join", "keys", "lower", "replace", "split", "startswith",
    "strip", "upper", "values",
})
_REFUSED_NODES = (ast.Lambda, ast.ClassDef, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)

# Every predicate runs with these helpers already defined, so a sequence rule stays one line.
HELPERS_SRC = '''
_YES = ("yes", "yeah", "yep", "confirm", "confirmed", "sure", "ok", "okay", "correct", "proceed")
_YES_PHRASES = ("go ahead", "please do", "that is right", "that's right")
_NO = ("no", "not", "never", "nope", "nah", "cannot", "cant", "dont", "stop", "wait", "wrong", "incorrect")

def _plain_words(text):
    return set("".join(c if (c.isalnum() or c.isspace()) else " " for c in text).split())

def user_confirmed(transcript):
    """True when the last user turn says yes to an action an assistant turn proposed (D43 case 3)."""
    messages = list(transcript or [])
    last = None
    for pos in range(len(messages) - 1, -1, -1):
        if messages[pos].get("role") == "user":
            last = pos
            break
    if last is None:
        return False
    proposal = ""
    for msg in reversed(messages[:last]):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            proposal = msg.get("content") or ""
            break
    if "?" not in proposal:
        return False  # nothing was proposed, so nothing was confirmed
    text = (messages[last].get("content") or "").lower()
    words = _plain_words(text)
    if words & set(_NO) or "n't" in text:
        return False  # a refusal that happens to carry a yes word is still a refusal
    return bool(words & set(_YES)) or any(phrase in text for phrase in _YES_PHRASES)

def called_before(transcript, *names):
    """True when the transcript already holds a tool call with one of these names."""
    wanted = set(names)
    for msg in list(transcript or []):
        for call in msg.get("tool_calls") or []:
            if call.get("name") in wanted:
                return True
    return False

def said_before(transcript, *needles):
    """True when an assistant turn already contains one of these strings, case-insensitive."""
    for msg in list(transcript or []):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").lower()
        if any(needle.lower() in text for needle in needles):
            return True
    return False
'''

_ALLOWED_BLOCK = "_ALLOWED = (\n" + "\n".join(
    "    " + " ".join(f'"{name}",' for name in _SAFE_BUILTINS[i : i + 6])
    for i in range(0, len(_SAFE_BUILTINS), 6)
) + "\n)"

_RUNNER_SRC = '''
import builtins
import json
import sys

''' + _ALLOWED_BLOCK + '''

def main():
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    namespace = {"__builtins__": {n: getattr(builtins, n) for n in _ALLOWED}}
    exec(compile(payload["src"], "<constraint>", "exec"), namespace)
    check = namespace.get("check")
    if not callable(check):
        print(json.dumps({"error": "predicate has no check function"}))
        return
    rows = []
    for case in payload["cases"]:
        row = {"label": case["label"], "expect": bool(case["expect"])}
        try:
            got = check(case.get("pre_state") or {}, case.get("write_call") or {}, case.get("transcript") or [])
            row["got"] = bool(got)
        except Exception as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        rows.append(row)
    print(json.dumps({"results": rows}))

main()
'''

_CONTRACT = """You turn one sentence of a customer's agent policy into a Python predicate.

Signature: def check(pre_state, write_call, transcript) -> bool. Return True when the rule is
satisfied and False when it is broken. pre_state is the database dict before the write. write_call
is {"name": tool name, "arguments": {...}}. transcript is the turns and tool calls before this write
and nothing after it, each
{"role": "user" or "assistant" or "tool", "content": str, "tool_calls": [{"name": str, "arguments": {}}]}.
A rule about what must happen before an action ("never X without a prior confirmation") is a
predicate over the transcript. A rule about no action returns True for calls it does not cover.
Already in scope: user_confirmed(transcript), which is True only when the last user turn says yes to
an action an assistant turn proposed; called_before(transcript, *names); said_before(transcript, *texts).
Plain Python only: no imports, no attribute other than the usual dict, list and str methods, no
lambda, no yield, no nested or class definitions, and no name you have not defined yourself.

Answer with JSON and nothing else:
{"compilable": true, "predicate_src": "...", "tests": {"pos": [case], "neg": [case]}}
or {"compilable": false, "reason": "..."} when no code can decide it.
A case is {"pre_state": {}, "write_call": {}, "transcript": []} and nothing else; the gate decides
what each case must return. Give exactly one pos case, which must return True, and one neg case,
which must return False."""

_REWRITE = """That predicate did not hold up: {reason}

Rewrite the rule into a checkable form that stays faithful to the sentence, then give the predicate
for the rewrite. Answer with JSON and nothing else:
{{"rewritten_text": "...", "predicate_src": "...", "tests": {{"pos": [case], "neg": [case]}}}}
or, when no code can decide it even after a rewrite,
{{"judge_atom": true, "rewritten_text": "...", "reason": "..."}}"""


@dataclass(frozen=True)
class PolicySentence:
    """One sentence of a policy file with the character span it came from."""

    index: int
    text: str
    start: int
    end: int
    section: Optional[str]
    file_hash: str

    def span(self) -> RawPtr:
        return RawPtr(file_hash=self.file_hash, msg_index=self.index)


# --- splitting ---


def split_policy(policy_md: str, file_hash: Optional[str] = None) -> list[PolicySentence]:
    """Policy markdown into sentences, headings and code fences dropped, spans kept."""
    file_hash = file_hash or content_hash(policy_md)
    out: list[PolicySentence] = []
    for pieces, section in _blocks(policy_md):
        text = "\n".join(t for _, t in pieces)
        for start, end in _sentence_spans(text):
            chunk = text[start:end]
            sentence = " ".join(chunk.split())
            if not sentence:
                continue
            lead = len(chunk) - len(chunk.lstrip())
            trail = len(chunk) - len(chunk.rstrip())
            out.append(
                PolicySentence(
                    index=len(out),
                    text=sentence,
                    start=_absolute(pieces, start + lead),
                    end=_absolute(pieces, end - trail),
                    section=section,
                    file_hash=file_hash,
                )
            )
    return out


def _blocks(policy_md: str) -> list[tuple[list[tuple[int, str]], Optional[str]]]:
    """Paragraphs and bullets as lists of (absolute offset, line text), with their section."""
    lines = policy_md.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    blocks: list[tuple[list[tuple[int, str]], Optional[str]]] = []
    buf: list[tuple[int, str]] = []
    section: Optional[str] = None
    fenced = False

    def flush() -> None:
        if buf:
            blocks.append((list(buf), section))
            buf.clear()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            fenced = not fenced
            continue
        if fenced:
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            section = stripped.lstrip("#").strip() or None
            continue
        body = line.lstrip()
        lead = len(line) - len(body)
        marker = _marker_length(body)
        if marker:
            flush()
            buf.append((offsets[i] + lead + marker, body[marker:].strip()))
        else:
            buf.append((offsets[i] + lead, body.strip()))
    flush()
    return blocks


def _marker_length(body: str) -> int:
    """How many characters of bullet or number marker start this line, 0 when it is a paragraph."""
    if body[:1] in _BULLET_CHARS and body[1:2].isspace():
        return 2
    head = body.split(" ", 1)[0]
    if len(head) > 1 and head[-1] in ".)" and head[:-1].isdigit():
        return len(head) + 1
    return 0


def _absolute(pieces: list[tuple[int, str]], rel: int) -> int:
    """A block-relative offset back to a file offset, across the lines the block spans."""
    for start, text in pieces:
        if rel <= len(text):
            return start + rel
        rel -= len(text) + 1
    start, text = pieces[-1]
    return start + len(text)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split on . ! ? that end a sentence; abbreviations, decimals and times stay whole."""
    spans, start = [], 0
    for i, char in enumerate(text):
        if char not in ".!?":
            continue
        end = i + 1
        while text[end : end + 1] in _CLOSERS:  # a sentence may end inside its own quotation
            end += 1
        following = text[end : end + 1]
        if following and not following.isspace():
            continue
        head = text[: i + 1].lower()
        if any(head.endswith(a) for a in _ABBREVIATIONS):
            continue
        rest = text[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in _BULLET_CHARS or rest[0] in "\"'("):
            continue
        if _has_letter(text[start:end]):
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if _has_letter(text[a:b])]


def _has_letter(chunk: str) -> bool:
    """A list marker on its own ("2.") is not a rule, so it never becomes a Constraint."""
    return any(c.isalpha() for c in chunk)


# --- the sandbox ---


def run_constraint_tests(constraint: Constraint, timeout_s: float = 5.0) -> GateResult:
    """Run a constraint's positive and negative cases in a subprocess with no imports and a timeout."""
    src = (constraint.predicate_src or "").strip()
    cases = _cases(constraint)
    metrics = {"pos": len(constraint.tests.pos), "neg": len(constraint.tests.neg), "ran": 0, "ok": 0}
    failures: list[str] = []
    if not src:
        failures.append("no predicate_src")
    if not constraint.tests.pos:
        failures.append("no positive test case")
    if not constraint.tests.neg:
        failures.append("no negative test case")
    if src:
        failures.extend(_static_check(src))
    if failures:
        return GateResult(stage=STAGE, passed=False, metrics=metrics, failures=failures)

    data = _sandbox(HELPERS_SRC + "\n" + src, cases, timeout_s)
    if data.get("error"):
        return GateResult(stage=STAGE, passed=False, metrics=metrics, failures=[data["error"]])
    for row in data.get("results", []):
        metrics["ran"] += 1
        if row.get("error"):
            failures.append(f"{row['label']} raised {row['error']}")
        elif row.get("got") != row.get("expect"):
            failures.append(f"{row['label']} returned {row.get('got')}, expected {row.get('expect')}")
        else:
            metrics["ok"] += 1
    return GateResult(stage=STAGE, passed=not failures, metrics=metrics, failures=failures)


def _cases(constraint: Constraint) -> list[dict]:
    """The gate owns the expected answer: a case the model wrote cannot declare its own outcome."""
    out = []
    for label, group, expect in (("pos", constraint.tests.pos, True), ("neg", constraint.tests.neg, False)):
        for i, case in enumerate(group):
            row = {k: v for k, v in dict(case or {}).items() if k not in ("expect", "label")}
            row["label"] = f"{label}[{i}]"
            row["expect"] = expect
            out.append(row)
    return out


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the predicate defines for itself: assignments, loop targets, functions and their parameters."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                    out.add(arg.arg)
                out.update(extra.arg for extra in (args.vararg, args.kwarg) if extra is not None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    return out


def _static_check(src: str) -> list[str]:
    """Certify a predicate before running it: only listed names, listed attributes, one top-level check()."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"predicate does not parse: {exc.msg} on line {exc.lineno}"]
    bad: list[str] = []
    known = _bound_names(tree) | _HELPER_NAMES | set(_SAFE_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad.append("predicate imports a module")
        elif isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRS:
                bad.append(f"predicate touches {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in _DENIED_NAMES:
                bad.append(f"predicate uses {node.id}")
            elif isinstance(node.ctx, ast.Load) and node.id not in known:
                bad.append(f"predicate uses an unknown name {node.id}")
        elif isinstance(node, _REFUSED_NODES):
            bad.append(f"predicate uses {type(node).__name__.lower()}, which the gate does not certify")
        elif isinstance(node, ast.FunctionDef) and node not in tree.body:
            bad.append("predicate defines a function inside a function")
    checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if not checks:
        bad.append("predicate has no top-level def check(pre_state, write_call, transcript)")
    elif len(checks[0].args.args) != 3:
        bad.append("check must take exactly pre_state, write_call, transcript")
    return sorted(set(bad))


def _sandbox(src: str, cases: list[dict], timeout_s: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "run_constraint.py"
        script.write_text(_RUNNER_SRC, encoding="utf-8")
        payload = root / "payload.json"
        payload.write_text(json.dumps({"src": src, "cases": cases}), encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(script), str(payload)],
                capture_output=True, text=True, timeout=timeout_s, cwd=str(root),
            )
        except subprocess.TimeoutExpired:
            return {"error": f"predicate timed out after {timeout_s} seconds"}
        if done.returncode != 0:
            return {"error": "predicate failed to load: " + (done.stderr.strip().splitlines() or ["no output"])[-1]}
        try:
            return json.loads(done.stdout.strip() or "{}")
        except json.JSONDecodeError:
            return {"error": "sandbox returned no result"}


# --- compiling ---


def compile_rule(model: Any, sentence: PolicySentence, timeout_s: float = 5.0) -> Constraint:
    """One sentence into a Constraint: compiled, or rewritten for review, or a judge atom, or residual."""
    constraint = Constraint(
        id="c_" + content_hash({"text": sentence.text, "file": sentence.file_hash})[:12],
        text=sentence.text,
        span=sentence.span(),
        span_text=sentence.text,
    )
    first, error = _ask(model, _CONTRACT, f"Section: {sentence.section or 'top level'}\nRule: {sentence.text}")
    if error:
        constraint.residual_reason = f"model call failed: {error}"
        return constraint
    reason = first.get("reason") or "the model gave no usable predicate"
    if first.get("compilable") and first.get("predicate_src"):
        _fill(constraint, first)
        gate = run_constraint_tests(constraint, timeout_s)
        if gate.passed:
            constraint.compiled = True
            return constraint
        reason = "; ".join(gate.failures)
    _clear(constraint)

    second, error = _ask(model, _CONTRACT, f"Rule: {sentence.text}\n\n" + _REWRITE.format(reason=reason))
    if error:
        constraint.residual_reason = f"model call failed: {error}"
        return constraint
    if not second:
        # A Builder output failure is not a decision about the rule, so it never becomes a judge atom (D76).
        constraint.residual_reason = "model output unparseable, the rewrite never happened"
        return constraint
    if second.get("rewritten_text"):
        constraint.rewritten_text = str(second["rewritten_text"]).strip()
    if second.get("judge_atom"):
        return _judge(constraint)
    if not constraint.rewritten_text:
        constraint.residual_reason = "the model gave no rewrite for review: " + reason
        return constraint
    if not second.get("predicate_src"):
        return _judge(constraint)
    _fill(constraint, second)
    if not run_constraint_tests(constraint, timeout_s).passed:
        _judge(constraint)
    return constraint


def _judge(constraint: Constraint) -> Constraint:
    """The rule stays natural language: judge.py evaluates it at Verdict time (D76)."""
    _clear(constraint)
    constraint.compiled = False
    constraint.judge_atom = True
    return constraint


def compile_policy(
    model: Any,
    policy: str | Iterable[PolicySentence],
    file_hash: Optional[str] = None,
    timeout_s: float = 5.0,
    limit: Optional[int] = None,
) -> list[Constraint]:
    """Every sentence of a policy, or a chosen list of sentences, compiled in order."""
    sentences = split_policy(policy, file_hash) if isinstance(policy, str) else list(policy)
    if limit is not None:
        sentences = sentences[:limit]
    return [compile_rule(model, sentence, timeout_s) for sentence in sentences]


def accept_rewrite(constraint: Constraint, timeout_s: float = 5.0) -> Constraint:
    """The reviewer accepted the rewrite (D76): it becomes a normal Constraint if its tests still pass."""
    out = constraint.model_copy(deep=True)
    if not out.rewritten_text:
        return out
    if run_constraint_tests(out, timeout_s).passed:
        out.compiled = True
        out.judge_atom = False
    else:
        _clear(out)
        out.compiled = False
        out.judge_atom = True
    return out


def reject_rule(constraint: Constraint, reason: str) -> Constraint:
    """The reviewer rejected the rule (D76): it is reported as not checked and never reaches a Verdict."""
    out = constraint.model_copy(deep=True)
    _clear(out)
    out.compiled = False
    out.judge_atom = False
    out.residual_reason = reason
    return out


def residual(constraints: Iterable[Constraint]) -> list[Constraint]:
    """The rules nothing checks, for the report."""
    return [c for c in constraints if c.residual_reason]


def pending_review(constraints: Iterable[Constraint]) -> list[Constraint]:
    """Rewrites waiting for the setup review (D76, D48): checked by nobody until accept_rewrite."""
    return [
        c for c in constraints
        if c.rewritten_text and not c.compiled and not c.judge_atom and not c.residual_reason
    ]


# --- the Reference's own path (the second half of the compile policy gate) ---


def reference_violations(
    constraints: Iterable[Constraint],
    reference_runs: Iterable[Any],
    write_tools: Optional[Iterable[str]] = None,
    timeout_s: float = 5.0,
) -> list[dict]:
    """Compiled rules a Reference's own path breaks, so the rule goes to residual, not into every Verdict."""
    views = [(str(getattr(run, "run_id", "?")), _run_cases(run, write_tools)) for run in reference_runs or ()]
    out: list[dict] = []
    for constraint in constraints or ():
        if not constraint.compiled or not constraint.predicate_src:
            continue
        for run_id, cases in views:
            if not cases:
                continue
            data = _sandbox(HELPERS_SRC + "\n" + constraint.predicate_src, cases, timeout_s)
            if data.get("error"):
                out.append({"run_id": run_id, "constraint_id": constraint.id, "tool": None,
                            "at": None, "reason": data["error"]})
                continue
            for row, case in zip(data.get("results", []), cases, strict=False):
                if row.get("error") or row.get("got") is not True:
                    out.append({"run_id": run_id, "constraint_id": constraint.id,
                                "tool": case["write_call"]["name"], "at": case["at"],
                                "reason": row.get("error") or "the Reference's own write breaks the rule"})
    return out


def _run_cases(run: Any, write_tools: Optional[Iterable[str]]) -> list[dict]:
    """Every write in a Run as a before-write case: the Starting state and the transcript up to that call."""
    events = list(getattr(run, "events", None) or [])
    wanted = set(write_tools) if write_tools else None
    start_state: dict = {}
    for event in events:
        payload = event.payload or {}
        if event.type == "stop" and isinstance(payload.get("start_state"), dict):
            start_state = payload["start_state"]
    transcript: list[dict] = []
    cases: list[dict] = []
    for event in events:
        payload = event.payload or {}
        if event.type == "user_turn":
            transcript.append({"role": "user", "content": str(payload.get("content") or payload.get("text") or ""),
                               "tool_calls": []})
        elif event.type == "model_call":
            reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else payload
            if reply.get("content"):
                transcript.append({"role": "assistant", "content": str(reply["content"]), "tool_calls": []})
        elif event.type == "tool_call":
            name = str(payload.get("name") or "")
            args = payload.get("args") or payload.get("arguments") or {}
            if wanted is None or name in wanted:
                cases.append({"label": f"{name}@{event.idx}", "expect": True, "at": event.idx,
                              "pre_state": start_state, "write_call": {"name": name, "arguments": args},
                              "transcript": [dict(m) for m in transcript]})
            transcript.append({"role": "assistant", "content": None,
                               "tool_calls": [{"name": name, "arguments": args}]})
    return cases


def _ask(model: Any, system: str, user: str) -> tuple[dict, Optional[str]]:
    """One model call; JSON out, or an error string when the provider itself failed."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        reply = model.query(messages)
    except Exception as exc:  # a provider failure is not a property of the rule
        return {}, f"{type(exc).__name__}: {exc}"
    return _parse(reply.content), None


def _parse(content: Optional[str]) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _fill(constraint: Constraint, data: dict) -> None:
    constraint.predicate_src = str(data.get("predicate_src") or "").strip() or None
    tests = data.get("tests") or {}
    constraint.tests = ConstraintTests(
        pos=[c for c in (tests.get("pos") or []) if isinstance(c, dict)],
        neg=[c for c in (tests.get("neg") or []) if isinstance(c, dict)],
    )


def _clear(constraint: Constraint) -> None:
    constraint.predicate_src = None
    constraint.tests = ConstraintTests()
