"""Step 2 of fx-examwhy: replay an Examiner session's own history, ask it why.

Usage:
  source /Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env
  python3 scripts/experiments/ask_examiner.py --bus BUS --session IDX --out OUT.json

Sends the session's agent_end messages + one retrospective user message (no tools),
then one counterfactual (history + retro Q + answer + counterfactual Q) chaining the
first reply so the shared prefix can hit prompt cache. Spend accumulates in
scripts/experiments/ask_spend.json; the $5 ceiling is refused before any call
that would cross it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

RETRO_Q = (
    "Looking back at this session: after the edits that were refused or dry-run refused, "
    "why did you do what you did next? What in the tool results was unclear or missing? "
    "Which tool did you not use that you could have, and why? What one change to the "
    "tools or instructions would have let you repair more Tasks?"
)

COUNTER_Q = (
    "Counterfactual: here again is the first ruling that refused one of your edits "
    "(copied from the tool result above). What exact edit do you try next, and why? "
    "Name the tool, the atom id, and the fields you would change."
)

SPEND_FILE = WORKTREE / "scripts" / "experiments" / "ask_spend.json"
CEILING_USD = 5.0
MODEL_ID = "bedrock/global.anthropic.claude-opus-5-5"


def load_session_messages(bus: Path, idx: int) -> tuple[list[dict], list[str]]:
    """The idx-th examiner session's agent_end messages plus its refusal texts."""
    from kullback.ai import provider as provider_mod

    provider_mod.load_dotenv(Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env"))
    ends = []
    with open(bus) as f:
        for line in f:
            e = json.loads(line)
            if e["agent"] == "examiner" and e["event"].get("type") == "agent_end":
                ends.append(e)
    session = ends[idx]["event"]
    refusals = []
    for m in session["messages"]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str) \
                and "refused" in m["content"][:400].lower():
            refusals.append(m["content"])
    return session["messages"], refusals


def to_wire_messages(raw: list[dict]) -> list[dict]:
    from kullback.ai.messages import (
        AssistantMessage,
        ToolCall,
        ToolResultMessage,
        UserMessage,
        to_wire,
    )

    parsed = []
    for m in raw:
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict))
            parsed.append(UserMessage(role="user", content=content or ""))
        elif role == "assistant":
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")
            calls = [
                ToolCall(id=c.get("id", ""), name=c.get("name", ""),
                         arguments=c.get("arguments") or {})
                for c in (m.get("tool_calls") or [])
            ]
            parsed.append(AssistantMessage(
                role="assistant", content=content,
                thinking_blocks=m.get("thinking_blocks"),
                tool_calls=calls,
                stop_reason=m.get("stop_reason") or "stop",
                model=m.get("model"),
                error_message=m.get("error_message")))
        elif role == "tool":
            parsed.append(ToolResultMessage(
                role="tool", tool_call_id=m.get("tool_call_id", ""),
                tool_name=m.get("tool_name", ""),
                content=m.get("content", "") or "",
                is_error=bool(m.get("is_error"))))
    return to_wire(parsed)


def save(out: dict, path: str) -> None:
    Path(path).write_text(json.dumps(out, indent=1))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from kullback.ai.provider import live_model
    from kullback.runner.budget import BudgetedModel, Ceiling, call_cost

    raw, refusals = load_session_messages(Path(args.bus), args.session)
    wire = to_wire_messages(raw)
    retro = [{"role": "user", "content": RETRO_Q}]

    spent = json.loads(SPEND_FILE.read_text()) if SPEND_FILE.exists() else {"usd": 0.0}
    ceiling = Ceiling(CEILING_USD, spent=spent["usd"], workdir="/tmp/examwhy-ask")
    ceiling.require_priced(MODEL_ID)
    model = BudgetedModel(live_model(MODEL_ID, None), stage="examwhy",
                          workdir="/tmp/examwhy-ask", model_id=MODEL_ID, ceiling=ceiling)

    out: dict = {"model": MODEL_ID, "bus": args.bus, "session": args.session,
                 "n_history_messages": len(wire), "refusals_seen": len(refusals)}
    reply = model.query(wire + retro, tools=None)
    out["retrospective"] = {"content": reply.content,
                            "usage": reply.usage.model_dump(),
                            "cost_usd": call_cost(reply.usage, MODEL_ID)}
    save(out, args.out)  # never lose a paid reply to a later failure

    first_refusal = refusals[0] if refusals else "(no refused edit in this session)"
    follow = [{"role": "assistant", "content": reply.content or ""},
              {"role": "user",
               "content": f"{COUNTER_Q}\n\nRefused ruling:\n{first_refusal}"}]
    try:
        reply2 = model.query(wire + retro + follow, tools=None)
        out["counterfactual"] = {"content": reply2.content,
                                 "usage": reply2.usage.model_dump(),
                                 "cost_usd": call_cost(reply2.usage, MODEL_ID)}
    except Exception as exc:  # ceiling reached: record, keep the retrospective
        out["counterfactual"] = {"error": f"{type(exc).__name__}: {exc}"}
    SPEND_FILE.write_text(json.dumps({"usd": ceiling.spent}))
    save(out, args.out)
    print(f"wrote {args.out} spend_total={ceiling.spent:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
