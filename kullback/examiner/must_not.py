"""The must-not side of an Intent-written Verifier: what the agent must not do.

`must_not` takes vf-write's must checks as input and never reads a Run. It returns
forbidden atoms, each carrying a `because`: the policy rule, or the Intent sentence
the write would go beyond. General classes only:

- scope: a write through a tool no must check covers fails, unless the policy or
  the Intent allows it. One forbidden whole-tool atom per uncovered write tool.
- policy prohibitions: the compiled Hard rules reused verbatim, none written by hand.
  A residual rule (neither compiled nor judged) is reported, never verdicted.
- claims: nothing added here. A claim that does not show in state already fails:
  the required write atom it names is missing, and the counts live in claims.py
  as measurement only, never as a ruling.
- find, do not invent: when the Intent says a record is missing or broken, that
  sentence becomes the scope atom's because, so creating the record fails under
  the sentence that forbids inventing it.
- one to one: every "do not" sentence of the Intent is named in the because of
  the scope atoms, and every forbidden atom names its Intent sentence or its
  policy rule. A "do not" no tool mapping can cover (every tool already allowed)
  gets no atom: a check that names no tool is one no Run can fail.

Forbidden values in messages (values read but not to be stated, negation handled)
would fit as forbidden communicate-value atoms scored against the answer text,
beside the communicate atoms; agreed for synthetic Tasks first, not built here.

`tier_of` maps an atom to its DESIGN.md tier from its kind only, never per Task.
After the vf-write merge it lives in intent_derive.py; this copy keeps vf-forbid
self-contained until then. Every atom this module emits is forbidden or hard,
so every one maps to sanity.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.gates.verifier_suite import atom_payload, make_atom
from kullback.runner.records import Atom
from kullback.runner.target import names_no_row

# A sentence that forbids: the plain prohibitions an instruction uses. General
# English, not per-corpus vocabulary (the claims module keeps the same kind of
# list for verbs any conversation uses).
_DO_NOT = re.compile(r"\bdo not\b|\bdon't\b|\bnever\b", re.IGNORECASE)
# A sentence that reports a record gone or unusable: creating it would invent it.
_MISSING_OR_BROKEN = re.compile(
    r"\bmissing\b|\bbroken\b|\bnot found\b|\bdoes not exist\b|\bdoesn't exist\b",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def tier_of(atom: Any) -> str:
    """The DESIGN.md tier of one atom, from its kind only.

    Required end-state writes are critical; forbidden checks, policy
    prohibitions and the claim check are sanity; questions, message facts and
    anything that never fails a Run are important. Gates fail reward 0 on
    critical and sanity; important never gates.
    """
    kind = atom.kind if isinstance(atom, Atom) else dict(atom).get("kind")
    if kind == "required":
        return "critical"
    if kind in ("forbidden", "hard"):
        return "sanity"
    return "important"


def must_not(verifier_checks: Iterable[Any], intent: Any, policy: Optional[Iterable[Any]],
             tools: Optional[Iterable[str]]) -> list[Atom]:
    """The forbidden atoms for one Task: scope plus policy prohibitions.

    `verifier_checks` is vf-write's must side (required and allowed atoms with
    structured write targets). `intent` is the Intent record, a dict carrying
    its text, or the text itself. `policy` is the compiled constraints.
    `tools` is the write tools from the tool schemas. No Run is read here.
    """
    must = [_as_atom(check) for check in verifier_checks or []]
    must = [atom for atom in must if atom is not None]
    text = _intent_text(intent)
    sentences = _sentences(text)
    tools = sorted({str(tool) for tool in tools or () if str(tool)})

    atoms = _scope_atoms(must, sentences, tools)
    atoms += _policy_atoms(policy, tools)
    return atoms


def _scope_atoms(must: list[Atom], sentences: list[str], tools: list[str]) -> list[Atom]:
    """One forbidden whole-tool atom per write tool no must check covers.

    The because prefers the prohibiting sentences (do-not, then missing/broken)
    and falls back to the Intent line itself. With no Intent text there is no
    quote to ground the check, so no scope atom is written.
    """
    if not sentences or not tools:
        return []
    covered_tools, covered_pairs = _coverage(must)
    partly = {tool for tool, _ in covered_pairs}
    because = _scope_because(sentences)
    atoms = []
    for tool in tools:
        if tool in covered_tools or tool in partly:
            # Any must coverage is the Intent's allowance on that tool; the
            # verdict's extra-write check still fails its uncovered rows.
            continue
        atoms.append(make_atom(
            f"forbidden.{tool}", "forbidden",
            {"kind": "write", "tool": tool, "because": {"intent": because}},
            description=f"the Run makes no write through {tool}"))
    return atoms

def _scope_because(sentences: list[str]) -> list[str]:
    """The Intent sentences a beyond-scope write goes against, most specific first."""
    prohibited = [sentence for sentence in sentences if _DO_NOT.search(sentence)]
    if prohibited:
        return prohibited
    missing = [sentence for sentence in sentences if _MISSING_OR_BROKEN.search(sentence)]
    return missing or list(sentences)


def _policy_atoms(policy: Optional[Iterable[Any]], tools: list[str]) -> list[Atom]:
    """One hard atom per compiled policy rule, its predicate reused verbatim.

    Residual rules are skipped: reported elsewhere, never verdicted. The reads
    a rule sees are unknown without a Run, so read_tools stays empty rather
    than guessed at.
    """
    atoms = []
    for rule in policy or []:
        rule = _as_rule(rule)
        if rule is None or not (rule.get("compiled") or rule.get("judge_atom")):
            continue
        rule_id = str(rule.get("id"))
        atoms.append(make_atom(
            f"hard.{rule_id}", "hard",
            {"kind": "hard", "constraint_id": rule_id,
             "judge": bool(rule.get("judge_atom")),
             "predicate_src": rule.get("predicate_src"),
             "write_tools": list(tools), "read_tools": [],
             "because": {"policy_rule": rule_id}},
            judge=bool(rule.get("judge_atom")),
            description=str(rule.get("text") or rule_id)))
    return atoms


def _coverage(must: list[Atom]) -> tuple[set[str], set[tuple[Any, Any]]]:
    """The tools wholly allowed and the (tool, entity) pairs allowed, by the must checks."""
    covered_tools: set[str] = set()
    covered_pairs: set[tuple[Any, Any]] = set()
    for atom in must:
        if atom.kind == "forbidden":
            continue
        payload = atom_payload(atom)
        if payload.get("kind") not in ("write", "write_value"):
            continue
        tool = payload.get("tool")
        if not tool:
            continue
        if names_no_row(payload):
            covered_tools.add(tool)
        else:
            covered_pairs.add((tool, payload.get("entity")))
    return covered_tools, covered_pairs


def _sentences(text: str) -> list[str]:
    """The Intent line as sentences, in order, blanks dropped."""
    return [sentence.strip() for sentence in _SPLIT.split(text.strip()) if sentence.strip()]


def _intent_text(intent: Any) -> str:
    """The Intent line out of a record, a dict, or a bare string."""
    if intent is None:
        return ""
    if isinstance(intent, str):
        return intent
    text = intent.text if hasattr(intent, "text") else dict(intent).get("text", "")
    return text or ""


def _as_atom(check: Any) -> Optional[Atom]:
    """One must check as an Atom; anything else is not a check and is dropped."""
    if isinstance(check, Atom):
        return check
    try:
        return Atom.model_validate(check)
    except Exception:
        return None


def _as_rule(rule: Any) -> Optional[dict]:
    """One policy rule as a dict; anything without an id is not a rule."""
    if rule is None:
        return None
    body = rule.model_dump() if hasattr(rule, "model_dump") else dict(rule)
    return body if body.get("id") else None
