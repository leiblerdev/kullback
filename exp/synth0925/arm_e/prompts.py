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

DIRECT_TASK = (
    "A person is writing to a support agent about several problems they want solved. "
    "You plan the exact steps that fulfil every goal below. This person is {anchor}.\n\n"
    "Their goals, each in plain words:\n{goals}\n\n"
    "The only values this person knows and may state:\n{facts}\n\n"
    "Look up rows with the read steps below (each names the arguments it requires):\n{reads}\n\n"
    "When the plan is ready, use the write steps below (each names the arguments it "
    "requires):\n{writes}\n\n"
    "Rules: use only the steps named above; fill every required argument from values "
    "your reads returned, never an invented one; reads never change anything; the request "
    "must state symptoms and outcomes in the person's own words in a few sentences, naming "
    "no tool, system, table or step, and carrying no value outside the list above.\n\n"
    "{note}"
    "Reply as JSON: either {{\"read\": {{\"name\": ..., \"args\": {{...}}}}}} to look up "
    "rows, or {{\"blueprint\": {{\"request\": ..., \"writes\": [{{\"name\": ..., \"args\": "
    "{{...}}}}], \"must_tell\": [...]}}}} when every goal has its steps."
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


def direct_blueprint_loop(
    model: Any,
    goals: list[str],
    facts: list[dict],
    anchor: str,
    read_tools: list[str],
    write_tools: list[str],
    note: str,
    executor: Any,
    max_reads: int = 12,
) -> tuple[dict, str, int]:
    """One long blueprint from scratch: goals in words, rows found by reading.

    The model sees no recorded call and no args JSON, only the goals, the anchor
    id, the user facts, and the tool shapes. It may call read steps (executed
    here against a fresh world, results truncated) up to max_reads times, then
    returns the blueprint. Returns the blueprint, the failure reason, and the
    reads used.
    """
    glines = "\n".join(f"- {g}" for g in goals) or "- nothing"
    fact_lines = "\n".join(f"- {r['field']}: {r['value']}" for r in facts) or "- nothing"
    rlines = "\n".join(f"- {t}" for t in read_tools) or "- nothing"
    wlines = "\n".join(f"- {t}" for t in write_tools) or "- nothing"
    history = DIRECT_TASK.format(
        anchor=anchor, goals=glines, facts=fact_lines, reads=rlines, writes=wlines, note=note
    )
    used = 0
    for _ in range(max_reads + 1):
        text, reason = _query(model, history)
        if reason:
            return {}, reason, used
        parsed, reason = _parse_json(text)
        if reason:
            history += f"\n\nThat reply was not usable ({reason}). Reply as JSON, a read or a blueprint."
            continue
        if isinstance(parsed.get("blueprint"), dict):
            return _clean_blueprint(parsed["blueprint"], used)
        want = parsed.get("read") or {}
        if not isinstance(want, dict) or not str(want.get("name") or "").strip():
            history += "\n\nThat reply was neither a read nor a blueprint. Reply as JSON, one or the other."
            continue
        if used >= max_reads:
            history += "\n\nNo read calls are left. Reply with the blueprint now."
            continue
        observation = executor(str(want.get("name")), want.get("args") or {})
        used += 1
        history += f"\n\nThe read returned:\n{observation}\n\nReply as JSON, a read or a blueprint."
    return {}, "the synthesiser never returned a blueprint", used


def _clean_blueprint(raw: Any, used: int) -> tuple[dict, str, int]:
    if not isinstance(raw, dict):
        return {}, "the synthesiser gave no usable blueprint", used
    request = str(raw.get("request") or "").strip()
    writes = raw.get("writes") or []
    if not request or not isinstance(writes, list) or not writes:
        return {}, "the synthesiser gave no usable blueprint", used
    clean = []
    for w in writes:
        if not isinstance(w, dict):
            continue
        name = str(w.get("name") or "").strip()
        args = w.get("args") or {}
        if name and isinstance(args, dict):
            clean.append({"name": name, "args": args})
    if not clean:
        return {}, "the synthesiser gave no usable blueprint", used
    must = raw.get("must_tell") or []
    must_tell = [str(m) for m in must if str(m).strip()][:5]
    return {"request": request, "writes": clean, "must_tell": must_tell}, "", used


def leaked(request: str, held: list[dict]) -> list[str]:
    return [row["field"] for row in held if str(row.get("value") or "") and str(row["value"]) in request]
