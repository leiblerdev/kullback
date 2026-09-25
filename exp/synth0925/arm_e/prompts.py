"""Arm E synthesiser prompts: one opening request (glued) or one long blueprint (direct).

Only the construction differs: both get the same goal set. Tool names and
shapes are read off the workdir at run time, never fixed here.
"""

from __future__ import annotations

import json
from typing import Any

GLUED_PROMPT = (
    "A person is writing to a support agent about several problems they want solved. "
    "Write their first message.\n\n"
    "Their goals, each in plain words:\n{goals}\n\n"
    "Write the symptoms and the outcomes they want, in their own words, in a few sentences. "
    "Do not name any tool, any system, any table or any step, and do not say how the agent "
    "should do it. Do not include any value that is not in the list below.\n\n"
    "The only values this person knows and may state:\n{facts}\n\n"
    "Reply as JSON with two keys: request (the message and nothing else inside it) and "
    "must_tell (a short list of things the agent must tell the person at the end, may be empty)."
)

DIRECT_PROMPT = (
    "A person is writing to a support agent about several problems they want solved. "
    "Plan the exact steps that fulfil every goal below.\n\n"
    "Their goals, each in plain words with the values the world holds for it:\n{goals}\n\n"
    "The steps the agent may take, each with the arguments it requires:\n{tools}\n\n"
    "Rules: use only the steps named above; fill every required argument; an argument "
    "that names a row must reuse a value from the goal data above, never an invented one; "
    "the request must state symptoms and outcomes in the person's own words in a few sentences, "
    "naming no tool, system, table or step, and carrying no value outside this list:\n{facts}\n\n"
    "Reply as JSON with three keys: request (the person's first message), writes (the exact "
    "steps in order, each with name and args), and must_tell (a short list of things the agent "
    "must tell the person at the end, may be empty)."
)


def _query(model: Any, prompt: str) -> tuple[str, str]:
    try:
        reply = model.query([{"role": "user", "content": prompt}])
    except Exception as error:  # noqa: BLE001 - provider failure is a revision reason
        return "", f"the synthesiser failed: {type(error).__name__}"
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return "", "the synthesiser answered nothing"
    return text, ""


def _parse_json(text: str) -> tuple[dict, str]:
    body = text.strip()
    if body.startswith("```"):
        lines = body.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        body = "\n".join(lines)
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return {}, "the synthesiser did not reply in JSON"
        try:
            parsed = json.loads(body[start : end + 1])
        except (ValueError, TypeError):
            return {}, "the synthesiser did not reply in JSON"
    if not isinstance(parsed, dict):
        return {}, "the synthesiser did not reply in JSON"
    return parsed, ""


def goal_line(unit: dict, askable: set[str]) -> tuple[str, str, list[dict], list[dict]]:
    """One goal in plain words: the step name plus the values a user could state.

    Returns the goal line, the full value block for domain data, and the
    said/held split of this unit's values.
    """
    words = str(unit["tool"]).replace("_", " ")
    said = [{"field": k, "value": v} for k, v in unit["args"].items() if k in askable]
    held = [{"field": k, "value": v} for k, v in unit["args"].items() if k not in askable]
    line = words
    if said:
        line += ": " + ", ".join(f"{r['field']} {r['value']}" for r in said)
    block = words + " " + json.dumps(unit["args"], default=str, sort_keys=True)
    return line, block, said, held


def glued_request(model: Any, goals: list[str], facts: list[dict], note: str = "") -> tuple[dict, str]:
    lines = "\n".join(f"- {g}" for g in goals) or "- nothing"
    fact_lines = "\n".join(f"- {r['field']}: {r['value']}" for r in facts) or "- nothing"
    prompt = GLUED_PROMPT.format(goals=lines, facts=fact_lines)
    if note:
        prompt += f"\n\nThe previous draft failed for this reason, fix it: {note}"
    text, reason = _query(model, prompt)
    if reason:
        return {}, reason
    parsed, reason = _parse_json(text)
    if reason:
        return {}, reason
    request = str(parsed.get("request") or "").strip()
    if not request:
        return {}, "the synthesiser gave no request"
    must = parsed.get("must_tell") or []
    must_tell = [str(m) for m in must if str(m).strip()][:5]
    return {"request": request, "must_tell": must_tell}, ""


def direct_blueprint(
    model: Any, goals: list[str], blocks: list[str], tools: list[str], facts: list[dict], note: str = ""
) -> tuple[dict, str]:
    glines = "\n".join(f"- {g}\n  data: {b}" for g, b in zip(goals, blocks, strict=True)) or "- nothing"
    tlines = "\n".join(f"- {t}" for t in tools) or "- nothing"
    fact_lines = "\n".join(f"- {r['field']}: {r['value']}" for r in facts) or "- nothing"
    prompt = DIRECT_PROMPT.format(goals=glines, tools=tlines, facts=fact_lines)
    if note:
        prompt += f"\n\nThe previous blueprint failed for this reason, fix it: {note}"
    text, reason = _query(model, prompt)
    if reason:
        return {}, reason
    parsed, reason = _parse_json(text)
    if reason:
        return {}, reason
    request = str(parsed.get("request") or "").strip()
    writes = parsed.get("writes") or []
    if not request or not isinstance(writes, list) or not writes:
        return {}, "the synthesiser gave no usable blueprint"
    clean = []
    for w in writes:
        if not isinstance(w, dict):
            continue
        name = str(w.get("name") or "").strip()
        args = w.get("args") or {}
        if name and isinstance(args, dict):
            clean.append({"name": name, "args": args})
    if not clean:
        return {}, "the synthesiser gave no usable blueprint"
    must = parsed.get("must_tell") or []
    must_tell = [str(m) for m in must if str(m).strip()][:5]
    return {"request": request, "writes": clean, "must_tell": must_tell}, ""


def leaked(request: str, held: list[dict]) -> list[str]:
    return [row["field"] for row in held if str(row.get("value") or "") and str(row["value"]) in request]
