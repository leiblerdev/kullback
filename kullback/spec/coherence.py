"""Did the recorded user hold to their own opening asks? Answered by the miner, checked by code.

Every recording was made with a model-simulated user, and that user sometimes strays from what it came
for: it withholds what the agent needs, or contradicts its own ask. The miner records that faithfully, so a
Verifier written from it is right for the conversation and wrong for the ask. The miner already reads every
recording, so it answers the question in its own add_facts call, with no second model call (D388): for each
opening ask (a volunteered do or told fact of condition none), what became of it by the end of the
recordings: carried out, refused on a policy ground or withdrawn by the user in their own words, or ended by
someone, with the ender named: the user (contradicted their own ask, or withheld what the policy requires
when asked), the agent (failed, stopped or handed off without acting) or the world (a condition the user set
did not hold, a call failed). Only an ask ended by the user, quoted in the user's own words, sets the Task
aside as intent_strayed, with no Verifier; every other answer is kept on the record and sets nothing aside.

This module holds the definitions the miner's prompt quotes and the checks code runs on its answers. A Task
with no opening ask is decided by code. Code keeps an answer only when its quote is in the recording it
names, as the miner checks in_reply_to; for the user ender the quote must sit in a user line and carry the
user's words once any control marker (###NAME###) is taken out, so a marker alone is never evidence of the
user straying. An opening ask the miner gave no answer for sets nothing aside and is counted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from kullback.runner.records import Trace
from kullback.spec.intent_tools import content_tokens, task_of, trace_index

CARRIED_OUT, REFUSED_OR_WITHDRAWN, ENDED_BY = "carried_out", "refused_or_withdrawn", "ended_by"
ANSWERS = (CARRIED_OUT, REFUSED_OR_WITHDRAWN, ENDED_BY)
USER, AGENT, WORLD = "user", "agent", "world"
ENDERS = (USER, AGENT, WORLD)
# A simulator's control marker in a user line (a stop, a hand-off): never the user's own words.
CONTROL_MARK_RE = re.compile(r"#{2,}[A-Z][A-Z_-]*#{2,}")
USER_PREFIX = "user: "
CODE, MINER = "code", "miner"
RESULT_CHARS = 400  # each tool result shown, clamped

# The question as the miner's prompt puts it (spec/intent.py), the definitions code checks against.
OUTCOME_RULE = """\
For each opening ask (a do or told fact you mark volunteered, condition none) give its outcome: what became
of it by the end of the recordings.
- carried_out: the agent did it (made the change, or told the user what was asked).
- refused_or_withdrawn: the agent declined it on a ground of policy, or the user withdrew it in their own
  words. When the user drops or declines an ask in their own words after hearing a price, a condition or a
  refusal, that is refused_or_withdrawn, not a user ending.
- ended_by: neither of those; name who ended the ask (ender):
  - user: the user contradicted their own ask (asked for the opposite of it), or withheld what the policy
    requires the agent to have (an identity detail, a confirmation) when the agent asked for it; nothing else;
  - agent: the agent failed, stopped, or handed off without acting, also after the user confirmed;
  - world: a condition the user set did not hold (an option that does not exist, a price not met), or what
    the agent tried failed.
Where the Task has several recordings, answer carried_out or refused_or_withdrawn when any recording shows
it, and ended_by only when none does. A stop or hand-off marker on its own is never the user's ending: the
user ender needs the user's own words. Quote, copied word for word from one line of the recording you name,
the line that shows the outcome; for the user ender, the user's own words."""


def opening_asks(intent: Any) -> list[Any]:
    """The opening asks of an Intent: volunteered do or told facts the user raised unprompted (condition none);
    never a fact of a scenario's end, which no user said (D414)."""
    from kullback.spec.items import is_end_fact

    return [fact for fact in intent.facts if fact.stance == "volunteered" and fact.kind in ("do", "told")
            and fact.condition == "none" and not is_end_fact(fact)]


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


def checked_answers(rows: dict, asks: list[Any], recordings: dict[str, list[str]]) -> dict[str, dict]:
    """Each opening ask's answer as code kept it: the answer, its ender (ended_by only), the recording it
    names, the quote, whether the quote holds (in that recording; for the user ender, the user's own words
    in a user line), and the why. An answer or ender outside the listed ones, or none, is kept as None."""
    out: dict[str, dict] = {}
    for fact in asks:
        row = rows.get(fact.id) if isinstance(rows.get(fact.id), dict) else {}
        answer = row.get("answer") if row.get("answer") in ANSWERS else None
        ender = row.get("ender") if answer == ENDED_BY and row.get("ender") in ENDERS else None
        recording = str(row.get("recording") or "")
        quote = " ".join(str(row.get("quote") or "").split())
        lines = recordings.get(recording)
        held = lines is not None and (user_words_in(quote, lines) if ender == USER else quote_in(quote, lines))
        out[fact.id] = {"answer": answer, "ender": ender, "recording": recording if lines is not None else None,
                        "quote": quote, "quote_ok": held, "why": str(row.get("why") or "")[:300]}
    return out


def strayed_facts(answers: dict[str, dict]) -> list[str]:
    """The opening asks the miner said the user ended, in the user's own words as code found them: any one
    sets the Task aside. An ask ended by the agent or the world is kept on the record only."""
    return [fact_id for fact_id, row in answers.items()
            if row["answer"] == ENDED_BY and row["ender"] == USER and row["quote_ok"]]


def coherence_of(workdir: Path, task_id: str, intent: Any, rows: dict,
                 traces: Optional[dict[str, Trace]] = None) -> dict:
    """The coherence record for the Spec from the miner's outcome rows ({fact id: row}): who decided (code or
    miner), the per-fact answers as code kept them, the strayed fact ids, the quotes that failed, and the
    opening asks left with no answer."""
    asks = opening_asks(intent)
    if not asks:
        return {"decided_by": CODE, "facts": {}, "strayed": [], "why": "no opening ask"}
    traces = trace_index(workdir) if traces is None else traces
    recordings = {run_id: recording_lines(traces[run_id])
                  for run_id in task_of(workdir, task_id).run_ids if run_id in traces}
    answers = checked_answers(rows, asks, recordings)
    return {"decided_by": MINER, "facts": answers, "strayed": strayed_facts(answers),
            "unquoted": sum(1 for row in answers.values() if row["answer"] and not row["quote_ok"]),
            "unanswered": sum(1 for row in answers.values() if row["answer"] is None)}


__all__ = ["AGENT", "ANSWERS", "CARRIED_OUT", "CODE", "ENDED_BY", "ENDERS", "MINER", "OUTCOME_RULE",
           "REFUSED_OR_WITHDRAWN", "USER", "WORLD", "checked_answers", "coherence_of", "opening_asks", "quote_in",
           "recording_lines", "strayed_facts", "user_words_in"]
