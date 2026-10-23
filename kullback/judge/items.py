"""Judge items (D328): one model call per item, score 0 or 1, on evidence cut from the Run's event log.

A judge item is a Verifier atom with `judge=True` whose target is {question, anchor, evidence}. The
question is one objective checkpoint (one right answer, not taste); the anchor says what meets it:
the answer, the equivalents to accept and the wrong answers to reject; the evidence names what the
judge reads: the Candidate's final answer by default, or named assistant turns. The prompt is a
strict checkpoint prompt: no partial credit, nothing guessed, and a target that falls in cut
evidence scores 0 with "evidence truncated".

The anchor lives on the Verifier, which the Candidate never sees. The judge sees the anchor and the
evidence and nothing else: no Reference, no expected end state, no user goal.

A call that fails or a reply that cannot be read gives score None, never a false 0: the reward is
masked for that Run, the same rule the grader follows.

The cap fits a model that reasons before it answers (D400): a reasoning model spends its output
tokens on thinking first, so a cap sized for the JSON alone comes back empty with stop reason
max_tokens. A reply that is empty or cut is asked once more at a larger cap; empty again, the item
is a judge outage (status "outage"), still unscored, so a report can tell an outage from a no.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from kullback.ai.provider import Model, ModelConfig
from kullback.judge.shape import NO_EVIDENCE, anchor_of, evidence_refs, has_evidence, is_item, resolve
from kullback.runner.budget import call_cost
from kullback.runner.records import Atom, Run

ITEM_JUDGE_VERSION = "2"
JUDGE_TOKENS = 4000  # thinking plus a one-line JSON answer
RETRY_TOKENS = 16000  # the one retry of an empty or cut reply
CUT_STOPS = ("max_tokens", "length")  # the stop reasons of a reply cut by the cap
TURN_CHARS = 4000  # each turn of evidence, clamped and marked when cut
EVIDENCE_CHARS = 24000  # all the evidence of one item; a long multi-turn Run fits
WHY_CHARS = 25
CUT_MARK = "[truncated]"
TRUNCATED_WHY = "evidence truncated"

SYSTEM = (
    "You are checking whether a conversation outcome meets one objective checkpoint (one right "
    "answer, not taste). Below is the evidence and the checkpoint. Judge strictly: met = 1, not met "
    "= 0, no partial credit. Rely only on the evidence given; if you cannot see it, score 0, do not "
    "guess. If the evidence is marked truncated where the checkpoint's target falls, score 0 and "
    f'write "{TRUNCATED_WHY}" in why. An answer listed under reject scores 0 even when it also '
    "says the right answer as one of several options it leaves the user to pick from.\n"
    "Answer with the JSON object first, nothing before it, and nothing else: "
    '{"results": {"<id>": {"score": 0 or 1, '
    f'"why": "<= {WHY_CHARS} chars"}}}}'
)


class JudgeScore(BaseModel):
    """One judge item judged on one Run: the Verdict reads its score into the item's holds (D328)."""
    id: str
    kind: str = "judge"
    gate: bool = True
    weight: float = 1.0
    score: Optional[int] = None  # 0 or 1; None when the judge failed or was not asked
    why: str = ""
    truncated: bool = False  # some evidence the judge read was cut
    evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)  # evidence refs the Run had nothing for
    error: Optional[str] = None
    status: str = ""  # "answered" (also a no-evidence 0, D408), "unreadable", or "outage" when no answer came back (D400)
    stop_reason: Optional[str] = None  # of the last call
    retries: int = 0
    cost_usd: float = 0.0
    model: Optional[str] = None
    version: str = ITEM_JUDGE_VERSION


def _clamp(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f" {CUT_MARK}", True


def build_evidence(run: Run, refs: Iterable[str], *, turn_chars: int = TURN_CHARS,
                   total_chars: int = EVIDENCE_CHARS) -> tuple[str, bool, list[str]]:
    """The evidence block, whether any of it was cut, and the refs the Run had nothing for.

    A turn named by two refs is shown once. Each turn is clamped to `turn_chars`; once the block
    reaches `total_chars` the remaining turns are dropped and the block ends marked truncated.
    """
    blocks: list[str] = []
    seen: set[int] = set()
    cut, missing, used = False, [], 0
    for ref in refs:
        found = resolve(run, ref)
        if not found:
            missing.append(ref)
        for label, idx, text in found:
            if idx in seen:
                continue
            seen.add(idx)
            body, clipped = _clamp(text, turn_chars)
            block = f"<{label}>\n{body}\n</{label}>"
            if used + len(block) > total_chars:
                blocks.append(f"{CUT_MARK} (further turns not shown)")
                return "\n\n".join(blocks), True, missing
            cut = cut or clipped
            used += len(block)
            blocks.append(block)
    if not blocks:
        blocks.append("(no evidence: the Run has no assistant message for the named turns)")
    return "\n\n".join(blocks), cut, missing


def checkpoint_line(item_id: str, target: dict) -> str:
    anchor = anchor_of(target)
    line = f"- [{item_id}] {target.get('question')} (met = 1: {anchor['answer']}"
    if anchor["accepted"]:
        line += "; also accept: " + "; ".join(anchor["accepted"])
    if anchor["reject"]:
        line += "; reject (score 0): " + "; ".join(anchor["reject"])
    return line + ")"


def judge_message(item_id: str, target: dict, evidence: str) -> str:
    return f"[Evidence]\n{evidence}\n\n[Checkpoint]\n{checkpoint_line(item_id, target)}"


def _read(content: Optional[str], item_id: str) -> tuple[Optional[int], str, Optional[str]]:
    """(score, why, key) from the reply; score None when the reply is not the asked-for object.

    Each call asks one item (D328), so a reply with exactly one result is read whatever its key; with
    several the key must be the item id (D424). `key` is the key read when it is not the item id.
    """
    text = (content or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    try:
        body = json.loads(match.group(0)) if match else None
    except ValueError:
        body = None
    results = body.get("results") if isinstance(body, dict) else None
    key = None
    if isinstance(results, dict) and item_id not in results and len(results) == 1:
        key = next(iter(results))
    row = results.get(item_id if key is None else key) if isinstance(results, dict) else None
    if not isinstance(row, dict):
        return None, "unreadable reply", None
    score = row.get("score")
    if isinstance(score, bool) or score not in (0, 1):
        return None, "score not 0 or 1", key
    return int(score), str(row.get("why") or "")[:WHY_CHARS * 2], key


def _parse(content: Optional[str], item_id: str) -> tuple[Optional[int], str]:
    """(score, why) from the reply; score None when the reply is not the asked-for object."""
    score, why, _ = _read(content, item_id)
    return score, why


def judge_item(model: Model, atom: Atom, run: Run, *, model_id: Optional[str] = None) -> JudgeScore:
    """One judge item on one Run: one call at temperature 0, a second when the first comes back empty or
    cut, priced; score 0, 1 or None."""
    target = atom.target or {}
    refs = evidence_refs(target)
    evidence, cut, missing = build_evidence(run, refs)
    name = model_id or getattr(model, "name", None)
    result = JudgeScore(id=atom.id, gate=bool(getattr(atom, "gate", True)),
                        weight=float(getattr(atom, "weight", 1.0)), truncated=cut,
                        evidence=refs, missing=missing, model=name)
    if not has_evidence(run, target):  # nothing to read: 0 by code, no call (D328)
        # An answer, not an outage: every reader scores it 0 rather than leaving the Run unscored (D408).
        result.score, result.why, result.status = 0, NO_EVIDENCE, "answered"
        return result
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": judge_message(atom.id, target, evidence)}]
    for cap in (JUDGE_TOKENS, RETRY_TOKENS):
        try:
            reply = model.query(messages, config=ModelConfig(temperature=0, max_tokens=cap))
        except Exception as error:  # any failed call masks the item; it is never a 0
            result.error = f"{type(error).__name__}: {str(error)[:200]}"
            result.status = "outage"
            return result
        result.cost_usd += call_cost(reply.usage, name) if name else 0.0
        result.stop_reason = reply.stop_reason
        result.score, result.why, key = _read(reply.content, atom.id)
        if key is not None:  # the one result came back under another key; say so (D424)
            result.why += f" [read under key {key[:WHY_CHARS]!r}, not the item id]"
        if result.score is not None or not _empty_or_cut(reply) or cap == RETRY_TOKENS:
            break
        result.retries += 1
    if result.score is not None:
        result.status = "answered"
    else:
        result.status = "outage" if not (reply.content or "").strip() else "unreadable"
        result.error = f"{result.why}: {(reply.content or '')[:200]!r}"
    return result


def _empty_or_cut(reply) -> bool:
    """A reply worth one retry: the cap cut it, or it holds no JSON object at all."""
    return reply.stop_reason in CUT_STOPS or not re.search(r"\{.*\}", reply.content or "", re.S)


def judge_items(model: Model, atoms: Iterable[Atom], run: Run, **kwargs) -> dict[str, JudgeScore]:
    """Every item-shaped judge atom of a Verifier on one Run, one call each, keyed by atom id."""
    return {atom.id: judge_item(model, atom, run, **kwargs) for atom in atoms if is_item(atom)}


def tally(results: Iterable[JudgeScore]) -> dict:
    """Counts a report prints: judged, scores, unjudged, truncated, spend."""
    rows = list(results)
    return {"items": len(rows),
            "met": sum(1 for r in rows if r.score == 1),
            "not_met": sum(1 for r in rows if r.score == 0),
            "unjudged": sum(1 for r in rows if r.score is None),
            "outage": sum(1 for r in rows if r.status == "outage"),
            "truncated": sum(1 for r in rows if r.truncated),
            "said_truncated": sum(1 for r in rows if TRUNCATED_WHY in r.why.lower()),
            "cost_usd": round(sum(r.cost_usd for r in rows), 4)}

