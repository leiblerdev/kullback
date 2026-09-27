"""The must-not side of an Intent-written Verifier: what the agent must not do.

`must_not` takes the writer's must checks as input and never reads a Run. It returns
forbidden and hard atoms, each carrying a `because`: the policy rule, the Intent
sentence the write would go beyond, or the claim rule. General classes only:

- scope: a write through a tool no must check covers fails, unless the policy or
  the Intent allows it. One forbidden whole-tool atom per uncovered write tool.
- policy prohibitions: the compiled Hard rules reused verbatim, none written by hand.
  A residual rule (neither compiled nor judged) is reported, never verdicted.
- claims (D228): MEASUREMENT ONLY (round 2 bar missed: 5.5% and 16.9% of right
  Runs fail it on the two clones, over the 2% bar). `claim_gate_atoms` builds
  the one hard atom for all write tools, scored the same way, but `must_not`
  does not emit it, so no shipped Verifier carries it. The rule: a Run whose
  transcript claims a write while no effective write reached the state fails;
  silence passes; any effective write answers every claim.
- find, do not invent: when the Intent says a record is missing or broken, that
  sentence becomes the scope atom's because, so creating the record fails under
  the sentence that forbids inventing it.
- one to one: every "do not" sentence of the Intent is named in the because of
  the scope atoms, and every scope or policy atom names its Intent sentence or
  its policy rule. A "do not" no tool mapping can cover (every tool already allowed)
  gets no atom: a check that names no tool is one no Run can fail.

Forbidden values in messages (values read but not to be stated, negation handled)
would fit as forbidden communicate-value atoms scored against the answer text,
beside the communicate atoms; agreed for synthetic Tasks first, not built here.

`tier_of` maps an atom to its DESIGN.md tier from its kind only, never per Task;
spec/schema.py reads it from here. Every atom this module emits is forbidden or
hard, so every one maps to sanity.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback import claims as _claims
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
             tools: Optional[Iterable[Any]]) -> list[Atom]:
    """The must-not atoms for one Task: scope and policy prohibitions.

    `verifier_checks` is the writer's must side (required and allowed atoms with
    structured write targets). `intent` is the Intent record, a dict carrying
    its text, or the text itself. `policy` is the compiled constraints.
    `tools` is the write tools from the tool schemas: names, or dicts carrying
    the name and description the claim vocabulary is mined from. No Run is read here.
    """
    must = [_as_atom(check) for check in verifier_checks or []]
    must = [atom for atom in must if atom is not None]
    text = _intent_text(intent)
    sentences = _sentences(text)
    names, _ = _tool_inputs(tools)
    atoms = _scope_atoms(must, sentences, names)
    atoms += _policy_atoms(policy, names)
    return atoms


def _tool_inputs(tools: Optional[Iterable[Any]]) -> tuple[list[str], list[dict]]:
    """Write tool names, with the schema dicts the claim vocabulary is mined from.

    A bare name mines from the name alone; a dict carries its description too.
    """
    names: list[str] = []
    sigs: list[dict] = []
    for tool in tools or ():
        if isinstance(tool, str):
            if tool:
                names.append(tool)
                sigs.append({"name": tool, "kind": "write"})
            continue
        body = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        if body.get("name"):
            names.append(str(body["name"]))
            sigs.append({"name": str(body["name"]), "kind": "write",
                         "description": body.get("description") or ""})
    order = sorted(set(names))
    sigs = [next(sig for sig in sigs if sig["name"] == name) for name in order]
    return order, sigs


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

def claim_gate_atoms(names: list[str], sigs: list[dict]) -> list[Atom]:
    """The D228 claim gate as measurement only: one hard atom for all write tools.

    Round 2 bar missed (over 2% of right Runs fail on both clones), so `must_not`
    does not emit this atom and no shipped Verifier carries it. Score it beside
    the Verdict the way claims.py is scored: the atom fails a Run whose
    transcript claims a write while the state received no effective write at
    all. Claims are read with D223's own sentence rule over stems mined by
    kullback/claims.py from the tools' own names and descriptions, with no
    general verbs and no per-tool attribution. Silence passes; any effective
    write answers every claim. The failure names the atom, never the sentence.
    """
    if not names:
        return []
    vocabulary = _claims.tool_vocabulary(sigs)
    vocab = {tool: tuple(sorted(vocabulary.get(tool, ()))) for tool in names}
    source = _CLAIM_SRC.replace("@@VOCAB@@", repr(vocab))
    source = source.replace("@@VERBS@@", repr(sorted(_claims.GENERAL_VERBS)))
    source = source.replace("@@FUTURE@@", repr(_future_cues()))
    source = source.replace("@@TOOLS@@", repr(tuple(names)))
    return [Atom(id="claim_without_effect", kind="hard",
                 target={"kind": "hard", "write_tools": list(names), "read_tools": [],
                         "because": {"claim_rule": "D228"}},
                 predicate_src=source,
                 description="every write the transcript claims took effect")]


def _future_cues() -> tuple[str, ...]:
    """D223's future cues as plain words and phrases, read off its own pattern.

    The pattern is one alternation of words (`will`) and phrases (`going to`,
    `i'?ll` meaning both `i'll` and `ill`), so splitting it names exactly what
    the matcher skips, with no second copy to drift.
    """
    pattern = _claims.FUTURE_CUES.pattern
    group = pattern[len(r"\b("):-len(r")\b")]
    cues: list[str] = []
    for cue in group.split("|"):
        if cue.endswith("'?ll"):
            cues.extend([cue.replace("?", ""), cue.replace("'?", "")])
        else:
            cues.append(cue)
    return tuple(cues)


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

# The D228 predicate one claim atom carries, scored by hard_holds over the Run's
# transcript and calls. It mirrors kullback/claims.py with no imports and no
# attribute walks, so the confinement gate lets it through: sentence split,
# past-tense stems, future cues and the mined vocabulary below are D223's rule,
# and only the precision guard fails the Run. Placeholders are filled per tool
# at build time; check() takes no arguments, which is what hard_holds calls.
_CLAIM_SRC = '''_VOCAB = @@VOCAB@@
_VERBS = @@VERBS@@
_FUTURE = @@FUTURE@@
_TOOLS = @@TOOLS@@


def _bare(word):
    if len(word) > 3 and word.endswith("e"):
        return word[:-1]
    return word


def _stem(word):
    word = word.lower()
    for ending, cut in (("ied", 3), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if len(word) > cut + 2 and word.endswith(ending):
            base = word[:-cut]
            if ending == "ied":
                return base + "y"
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
                return base[:-1]
            return _bare(base)
    return _bare(word)


def _words(text):
    spaced = []
    for pos in range(len(text)):
        char = text[pos]
        if pos and char.isupper() and (text[pos - 1].islower() or text[pos - 1].isdigit()):
            spaced.append(" ")
        spaced.append(char)
    words = []
    word = []
    for char in "".join(spaced):
        if char.isalpha():
            word.append(char)
        else:
            if len(word) > 2:
                words.append("".join(word))
            word = []
    if len(word) > 2:
        words.append("".join(word))
    return words


def _sentences(text):
    out = []
    for part in text.split("\\n"):
        start = 0
        pos = 0
        while pos < len(part):
            if part[pos] in ".?!" and (pos + 1 == len(part) or part[pos + 1] in " \\t"):
                cut = part[start:pos + 1].strip()
                if cut:
                    out.append(cut)
                pos += 1
                while pos < len(part) and part[pos] in " \\t":
                    pos += 1
                start = pos
            else:
                pos += 1
        tail = part[start:].strip()
        if tail:
            out.append(tail)
    return out


def _past_stems(sentence):
    if "?" in sentence:
        return set()
    flat = []
    for char in sentence.lower():
        flat.append(char if (char.isalnum() or char == "'") else " ")
    flat = " " + "".join(flat) + " "
    for cue in _FUTURE:
        if " " + cue + " " in flat:
            return set()
    stems = set()
    for word in _words(sentence):
        lowered = word.lower()
        if lowered in _VERBS or lowered.endswith("ed") or lowered.endswith("ied"):
            stems.add(_stem(word))
    return stems


def _effective_write():
    for call in calls:
        if (call.get("name") or "") in _TOOLS and not call.get("error"):
            return True
    return False


def check():
    claimed = False
    for turn in transcript:
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content") or ""
        if not isinstance(content, str):
            continue
        for sentence in _sentences(content):
            stems = _past_stems(sentence)
            if not stems:
                continue
            for tool in _TOOLS:
                mined = _VOCAB.get(tool) or ()
                for stem in stems:
                    if stem in mined:
                        claimed = True
                        break
                if claimed:
                    break
            if claimed:
                break
        if claimed:
            break
    if not claimed:
        return True
    return _effective_write()
'''
