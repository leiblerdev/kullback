"""Did the recorded user hold to their own opening asks? One judge question per Task, before the writer.

Every recording was made with a model-simulated user, and that user sometimes strays from what it came
for: it withholds what the agent needs, swaps two values, stops after the first confirmation, or asks
for something it then drops. The miner records that faithfully, so a Verifier written from it is right
for the conversation and wrong for the ask. This check reads the Intent and the recordings only (never
a Run, never the Reference) and asks, for each opening ask (a volunteered do or told fact of condition
none), what became of it by the end of the recordings: carried out, refused on a policy ground or
withdrawn by the user in their own words, or ended by someone, with the ender named: the user (changed
what they wanted, asked for the opposite, withheld what the policy requires when asked), the agent (failed,
stopped or handed off without acting) or the world (a condition the user set did not hold, a call failed).
Only an ask ended by the user, quoted in the user's own words, sets the Task aside as intent_strayed, with
no Verifier; an agent or world ending is kept on the record and sets nothing aside.

Code decides first: a Task with no opening ask has nothing to check and no call is made. The judge
answers the rest in one call, each answer with the line it rests on; code keeps an answer only when that
quote is in the recording it names, as the miner checks in_reply_to; for the user ender the quote must sit in
a user line and carry the user's words once any control marker (###NAME###) is taken out, so a marker alone
is never evidence of the user straying. A failed call or an unreadable reply
sets nothing aside: the Task goes on to the writer and the Spec says why.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from kullback.ai.provider import ModelConfig
from kullback.runner.budget import call_cost
from kullback.runner.records import Trace
from kullback.spec.intent_tools import content_tokens, task_of, trace_index

CARRIED_OUT, REFUSED_OR_WITHDRAWN, ENDED_BY = "carried_out", "refused_or_withdrawn", "ended_by"
ANSWERS = (CARRIED_OUT, REFUSED_OR_WITHDRAWN, ENDED_BY)
USER, AGENT, WORLD = "user", "agent", "world"
ENDERS = (USER, AGENT, WORLD)
# A simulator's control marker in a user line (a stop, a hand-off): never the user's own words.
CONTROL_MARK_RE = re.compile(r"#{2,}[A-Z][A-Z_-]*#{2,}")
USER_PREFIX = "user: "
CODE, JUDGE = "code", "judge"
RESULT_CHARS = 400  # each tool result shown, clamped
MAX_TOKENS = 3000

SYSTEM = """\
You read the recorded conversations of one Task, in which a simulated user asked an agent for something,
and the user's opening asks, mined from what the user said first. A simulated user sometimes strays from
what it came for. For each opening ask, answer what became of it by the end of the recordings:
- carried_out: the agent did it (for a change, a call that makes it succeeded; for something to be told,
  the agent told the user).
- refused_or_withdrawn: the agent declined it on a ground of policy, or the user withdrew it in their own
  words.
- ended_by: neither of those; name who ended the ask (ender):
  - user: the user, in their own words, changed what they wanted, asked for the opposite, or withheld what
    the policy requires the agent to have (an identity detail, a confirmation) when the agent asked for it;
  - agent: the agent failed, stopped, or handed off without acting, also after the user confirmed;
  - world: a condition the user set did not hold (an option that does not exist, a price not met), or a
    tool call failed.
Where the Task has several recordings, answer carried_out or refused_or_withdrawn when any recording shows
it, and ended_by only when none does. A stop or hand-off marker on its own is never the user changing
course: the user ender needs the user's own words.
For each answer quote, copied word for word from one line of the recording you name, the line or call
that shows it: the call that made the change, the agent's refusal, the user's own withdrawal, the user's
words that changed course, the agent's line or the failed call.
Output one JSON object and nothing else:
{"facts": {"<fact id>": {"answer": "carried_out" | "refused_or_withdrawn" | "ended_by",
 "ender": "user" | "agent" | "world" (ended_by only), "recording": "<recording label>",
 "quote": "<words copied from one line>", "why": "<one short sentence>"}}}"""


def opening_asks(intent: Any) -> list[Any]:
    """The opening asks of an Intent: volunteered do or told facts the user raised unprompted (condition none)."""
    return [fact for fact in intent.facts
            if fact.stance == "volunteered" and fact.kind in ("do", "told") and fact.condition == "none"]


def _clamp(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [cut]"


def recording_lines(trace: Trace) -> list[str]:
    """One recording as lines in order: what the user and the agent said, and each call the agent made with
    its arguments and its result or error (clamped). Tool turns are shown on their call, never twice."""
    calls = {call.id: call for call in trace.tool_calls if call.id}
    lines: list[str] = []
    for turn in trace.turns:
        text = (turn.content or "").strip()
        if turn.role == "user" and text:
            lines.append(f"user: {text}")
        elif turn.role == "assistant":
            if text:
                lines.append(f"agent: {text}")
            for call_id in turn.tool_call_ids:
                call = calls.get(call_id)
                if call is None:
                    continue
                answer = f"error: {call.error.payload}" if call.error is not None else str(call.result)
                lines.append(f"agent calls {call.name} {json.dumps(call.args, sort_keys=True)} -> "
                             f"{_clamp(answer, RESULT_CHARS)}")
    return lines


def _squash(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def quote_in(quote: str, lines: list[str]) -> bool:
    """Whether the quote carries content words and sits, word for word, inside one line of the recording."""
    span = _squash(quote)
    return bool(content_tokens(quote)) and any(span in _squash(line) for line in lines)


def user_words_in(quote: str, lines: list[str]) -> bool:
    """Whether the quote is the user's own words: in a user line, with content words once any control
    marker is taken out of it."""
    words = CONTROL_MARK_RE.sub(" ", quote)
    users = [line for line in lines if line.startswith(USER_PREFIX)]
    return quote_in(quote, users) and bool(content_tokens(words))


def judge_message(asks: list[Any], recordings: dict[str, list[str]]) -> str:
    facts = "\n".join(f"- [{fact.id}] ({fact.kind}) {fact.text}" for fact in asks)
    blocks = "\n\n".join(f"<recording {label}>\n" + "\n".join(lines) + f"\n</recording {label}>"
                         for label, lines in recordings.items())
    return f"[Opening asks]\n{facts}\n\n[Recordings]\n{blocks}"


def _reply_rows(content: Optional[str]) -> Optional[dict]:
    match = re.search(r"\{.*\}", (content or "").strip(), re.S)
    try:
        body = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    rows = body.get("facts") if isinstance(body, dict) else None
    return rows if isinstance(rows, dict) else None


def checked_answers(rows: dict, asks: list[Any], recordings: dict[str, list[str]],
                    labels: dict[str, str]) -> dict[str, dict]:
    """Each opening ask's answer as code kept it: the answer, its ender (ended_by only), the recording it
    names, the quote, whether the quote holds (in that recording; for the user ender, the user's own words
    in a user line), and the why. An answer or ender outside the listed ones, or none, is kept as None."""
    out: dict[str, dict] = {}
    for fact in asks:
        row = rows.get(fact.id) if isinstance(rows.get(fact.id), dict) else {}
        answer = row.get("answer") if row.get("answer") in ANSWERS else None
        ender = row.get("ender") if answer == ENDED_BY and row.get("ender") in ENDERS else None
        label = str(row.get("recording") or "")
        quote = " ".join(str(row.get("quote") or "").split())
        lines = recordings.get(label)
        held = lines is not None and (user_words_in(quote, lines) if ender == USER else quote_in(quote, lines))
        out[fact.id] = {"answer": answer, "ender": ender, "recording": labels.get(label), "quote": quote,
                        "quote_ok": held, "why": str(row.get("why") or "")[:300]}
    return out


def strayed_facts(answers: dict[str, dict]) -> list[str]:
    """The opening asks the judge said the user ended, in the user's own words as code found them: any one
    sets the Task aside. An ask ended by the agent or the world is kept on the record only."""
    return [fact_id for fact_id, row in answers.items()
            if row["answer"] == ENDED_BY and row["ender"] == USER and row["quote_ok"]]


def check_coherence(workdir: Path, task_id: str, intent: Any, model: Any,
                    traces: Optional[dict[str, Trace]] = None) -> tuple[dict, float]:
    """(the coherence record for the Spec, what the call cost). The record says who decided (code or
    judge), the per-fact answers, the strayed fact ids, the quotes that failed and any error."""
    asks = opening_asks(intent)
    if not asks:
        return {"decided_by": CODE, "facts": {}, "strayed": [], "why": "no opening ask"}, 0.0
    traces = trace_index(workdir) if traces is None else traces
    run_ids = [run_id for run_id in task_of(workdir, task_id).run_ids if run_id in traces]
    labels = {f"r{i + 1}": run_id for i, run_id in enumerate(run_ids)}
    recordings = {label: recording_lines(traces[run_id]) for label, run_id in labels.items()}
    if not any(recordings.values()):
        return {"decided_by": CODE, "facts": {}, "strayed": [], "why": "no recording"}, 0.0
    name = getattr(model, "name", None)
    record: dict = {"decided_by": JUDGE, "model": name, "facts": {}, "strayed": [], "unquoted": 0}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": judge_message(asks, recordings)}]
    try:
        reply = model.query(messages, config=ModelConfig(temperature=0, max_tokens=MAX_TOKENS))
    except Exception as error:  # a failed call sets nothing aside; the writer goes on
        return dict(record, error=f"{type(error).__name__}: {str(error)[:200]}"), 0.0
    spent = call_cost(reply.usage, name) if name and reply.usage is not None else 0.0
    rows = _reply_rows(reply.content)
    if rows is None:
        return dict(record, error="unreadable reply", usd=spent), spent
    answers = checked_answers(rows, asks, recordings, labels)
    return dict(record, facts=answers, strayed=strayed_facts(answers), usd=spent,
                unquoted=sum(1 for row in answers.values() if row["answer"] and not row["quote_ok"])), spent


__all__ = ["AGENT", "ANSWERS", "CARRIED_OUT", "CODE", "ENDED_BY", "ENDERS", "JUDGE", "REFUSED_OR_WITHDRAWN", "USER",
           "WORLD", "check_coherence", "checked_answers", "opening_asks", "quote_in", "recording_lines",
           "strayed_facts", "user_words_in"]
