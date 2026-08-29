"""Writes a Task's one-line Intent and refuses it unless every noun phrase points to a span in
every member Run (D47, D83)."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal, Optional, Sequence

from harness.builder.cluster import APOSTROPHE_RE, STOPWORDS
from harness.builder.cluster import first_line as _shared_first_line
from harness.shared.provider import Model
from harness.shared.records import RawPtr, Record, Task, ToolCall, Trace

MAX_INTENT_CHARS = 200
MAX_PROMPT_RUNS = 5  # D65: the prompt samples the Task's Runs, the grounding still reads all of them
MAX_PROMPT_SPAN_CHARS = 300
PHRASE_RE = re.compile(r"[a-z0-9_]+")
MATCH_RE = re.compile(r"[a-z0-9]+")  # splits credit_card_1234 into credit, card, 1234, so a phrase can reach it
CLAUSE_RE = re.compile(r"[.,;:!?\n]")
# The frame every model writes the Intent in ("The user wanted help to ...") is not evidence of anything
# and grounded on no span in any domain; it is stripped before the noun phrases are read.
FRAME_RE = re.compile(r"^\s*(?:the\s+)?(?:user|customer|caller|client)?(?:'s)?\s*(?:wanted|wants|asked|asks|"
                      r"needed|needs|requested|requests|would like|is asking|was asking)?"
                      r"(?:\s+(?:help|to|for|with|about))*\s*", re.IGNORECASE)
# A phrase the user ruled out in the same breath is not evidence that the user wanted it.
NEGATIONS = frozenset("not no never dont cannot cant wont without nor neither".split())

SpanSource = Literal["user_utterance", "tool_arg", "written_value"]


class IntentSpan(Record):
    """Where one noun phrase of the Intent is evidenced: a user utterance, a tool argument, or a written value."""
    phrase: str = ""
    trace_id: str
    source: SpanSource
    text: str
    label: Optional[str] = None  # the tool and argument the text came from; shown, never matched against
    raw_ptr: Optional[RawPtr] = None


class Intent(Record):
    """The one-line Intent of a Task with the span behind every noun phrase (D47); ungrounded means no Verdict."""
    task_id: str
    text: str = ""
    spans: list[IntentSpan] = []
    grounded: bool = False
    unguarded: bool = False
    ungrounded_phrases: list[str] = []
    run_coverage: dict[str, list[str]] = {}  # phrase -> every member Run that evidences it
    reason: Optional[str] = None
    model: Optional[str] = None


def noun_phrases(text: str) -> list[str]:
    """Runs of non-stopword tokens, in order, deduplicated. A cheap stand-in for a noun-phrase parser."""
    phrases: list[str] = []
    current: list[str] = []
    low = (text or "").lower()
    end = 0
    for match in PHRASE_RE.finditer(low):
        broken = low[end : match.start()].strip() != ""  # punctuation between two words ends a phrase
        end = match.end()
        token = match.group()
        if (broken or token in STOPWORDS) and current:
            phrases.append(" ".join(current))
            current = []
        if token not in STOPWORDS:
            current.append(token)
    if current:
        phrases.append(" ".join(current))
    seen: set[str] = set()
    return [p for p in phrases if not (p in seen or seen.add(p))]


def _flat(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _earlier_view(calls: Sequence[ToolCall], index: int) -> Optional[dict]:
    """The latest earlier result describing the same thing this write returned, or nothing.

    Same thing means the two results share keys and agree on at least one of the shared ids, so a
    read of another order is not mistaken for the before picture of this one.
    """
    result = calls[index].result
    for earlier in reversed(list(calls[:index])):
        other = earlier.result
        if not isinstance(other, dict):
            continue
        shared = set(other) & set(result)
        if not shared:
            continue
        ids = [k for k in sorted(shared) if k.endswith("_id")]
        if ids and not any(other[k] == result[k] for k in ids):
            continue
        return other
    return None


def _written_value(calls: Sequence[ToolCall], index: int) -> Optional[Any]:
    """What the write changed, not everything it echoed back.

    A tau2 write returns the whole Order, so the untouched fields grounded phrases about parts of
    the world the write never touched ("return" off an exchange). Against the read that came
    before it, the span is the fields that actually differ; a write that changed nothing is no
    evidence at all (D47).
    """
    result = calls[index].result
    if not isinstance(result, dict):
        return result
    prior = _earlier_view(calls, index)
    if prior is None:
        return result
    changed = {k: v for k, v in result.items() if k not in prior or prior[k] != v}
    return changed or None


def span_candidates(trace: Trace, write_tools: Optional[set[str]] = None) -> list[IntentSpan]:
    """Every place in one Run an Intent phrase is allowed to point at.

    The tool's own name is not one of them: it is our schema, not the customer's evidence, and
    `exchange_delivered_order_items` otherwise grounds the phrase "delivered items" in a Run where
    nobody said it (D47).
    """
    writes = set(write_tools or ())
    out: list[IntentSpan] = []
    for turn in trace.turns:
        if turn.role == "user" and turn.content:
            out.append(
                IntentSpan(
                    trace_id=trace.trace_id,
                    source="user_utterance",
                    text=turn.content,
                    raw_ptr=turn.raw_ptr,
                )
            )
    for index, call in enumerate(trace.tool_calls):
        if call.error is not None:
            continue
        for key in sorted(call.args):
            out.append(
                IntentSpan(
                    trace_id=trace.trace_id,
                    source="tool_arg",
                    text=_flat(call.args[key]),
                    label=f"{call.name}.{key}",
                    raw_ptr=call.raw_ptr,
                )
            )
        if call.name in writes and call.result is not None:
            written = _written_value(trace.tool_calls, index)
            if written is not None:
                out.append(
                    IntentSpan(
                        trace_id=trace.trace_id,
                        source="written_value",
                        text=_flat(written),
                        label=call.name,
                        raw_ptr=call.raw_ptr,
                    )
                )
    return out


def _clauses(text: str) -> list[list[str]]:
    """The span's text as clauses of tokens; punctuation ends a clause, as it ends a thought."""
    lowered = APOSTROPHE_RE.sub("", (text or "").lower())
    return [MATCH_RE.findall(clause) for clause in CLAUSE_RE.split(lowered)]


def _covers(clauses: Sequence[Sequence[str]], phrase: str, honour_negation: bool) -> bool:
    """Does one clause of the span say this phrase, in these words, in this order?

    Set containment said yes to "late delivery" for "the delivery was fine but the card was late",
    so the phrase has to be a contiguous run of tokens inside one clause (D47).
    """
    wanted = MATCH_RE.findall(phrase)
    if not wanted:
        return False
    for tokens in clauses:
        for start in range(len(tokens) - len(wanted) + 1):
            if list(tokens[start : start + len(wanted)]) != wanted:
                continue
            if honour_negation and any(t in NEGATIONS for t in tokens[:start]):
                break  # this clause rules the phrase out; another clause may still say it
            return True
    return False


def _matching_spans(
    phrases: Sequence[str],
    traces: Sequence[Trace],
    write_tools: Optional[set[str]] = None,
) -> dict[str, list[IntentSpan]]:
    """Every span that evidences each phrase, in trace order, at most one span per Run."""
    candidates = [
        (span, _clauses(span.text), span.source == "user_utterance")
        for trace in traces
        for span in span_candidates(trace, write_tools)
    ]
    found: dict[str, list[IntentSpan]] = {}
    for phrase in phrases:
        seen: set[str] = set()
        spans: list[IntentSpan] = []
        for span, clauses, honour_negation in candidates:
            if span.trace_id in seen or not _covers(clauses, phrase, honour_negation):
                continue
            seen.add(span.trace_id)
            spans.append(span.model_copy(update={"phrase": phrase}))
        found[phrase] = spans
    return found


def _choose_spans(found: dict[str, list[IntentSpan]]) -> tuple[list[IntentSpan], list[str]]:
    """One span per phrase for the record, preferring a Run not shown yet so a reviewer sees spread."""
    spans: list[IntentSpan] = []
    ungrounded: list[str] = []
    used: set[str] = set()
    for phrase, matches in found.items():
        if not matches:
            ungrounded.append(phrase)
            continue
        chosen = next((m for m in matches if m.trace_id not in used), matches[0])
        used.add(chosen.trace_id)
        spans.append(chosen)
    return spans, ungrounded


def ground_phrases(
    phrases: Sequence[str],
    traces: Sequence[Trace],
    write_tools: Optional[set[str]] = None,
) -> tuple[list[IntentSpan], list[str]]:
    """One span per phrase, preferring a Run not used yet so the record shows real spread."""
    return _choose_spans(_matching_spans(phrases, traces, write_tools))


def _intent_prompt(traces: Sequence[Trace], write_tools: Optional[set[str]]) -> str:
    """The evidence from a bounded sample of the Task's Runs (D65: no call may grow with the corpus)."""
    shown = list(traces[:MAX_PROMPT_RUNS])
    lines = [
        "Write one line saying what the user wanted, common to all these runs.",
        "Start with the verb (as in: cancel order #W1). Do not write 'the user wanted'.",
        "Use only words evidenced below; every noun must appear in the evidence.",
    ]
    if len(traces) > len(shown):
        lines.append(f"These are {len(shown)} of the task's {len(traces)} runs; say only what all of them show.")
    lines.append("")
    for trace in shown:
        for span in span_candidates(trace, write_tools):
            text = " ".join(span.text.split())[:MAX_PROMPT_SPAN_CHARS]
            label = f" {span.label}" if span.label else ""
            lines.append(f"{trace.trace_id} {span.source}{label}: {text}")
    lines += ["", "Reply with the one line only."]
    return "\n".join(lines)


def strip_frame(text: str) -> str:
    """"The user wanted help to cancel order #W1" is "cancel order #W1"; a line that is only the frame stays."""
    stripped = FRAME_RE.sub("", text or "", count=1).strip()
    return stripped if stripped else (text or "").strip()


def _first_line(text: Optional[str]) -> str:
    return strip_frame(_shared_first_line(text, MAX_INTENT_CHARS) or "")


def write_intent(
    model: Model,
    task: Task,
    traces: Iterable[Trace],
    *,
    write_tools: Optional[set[str]] = None,
) -> Intent:
    """The Task's Intent, grounded phrase by phrase; a single-Run Task skips the cross-run check (D97).

    The cross-run check asks whether the Task's Runs all show the Intent, which is what makes it the
    Task's Intent (D83), not whether the spans the chooser picked happen to name two Runs. That
    counted spans, so one phrase both Runs said was refused, while the union of two Runs' different
    intents was accepted because it had one span from each.
    """
    wanted = list(task.run_ids)
    members = [t for t in traces if t.trace_id in set(wanted)]
    if not members:
        raise ValueError(f"task {task.id} has no member traces among the traces given")
    missing_traces = [rid for rid in wanted if rid not in {t.trace_id for t in members}]
    if missing_traces:
        raise ValueError(f"task {task.id} is missing the traces for {', '.join(missing_traces)}")
    members.sort(key=lambda t: wanted.index(t.trace_id))

    reply = model.query([{"role": "user", "content": _intent_prompt(members, write_tools)}])
    text = _first_line(reply.content)
    found = _matching_spans(noun_phrases(text), members, write_tools)
    spans, ungrounded = _choose_spans(found)
    coverage = {phrase: sorted(s.trace_id for s in matches) for phrase, matches in found.items() if matches}
    member_ids = sorted(t.trace_id for t in members)
    gaps = {
        phrase: [rid for rid in member_ids if rid not in set(runs)]
        for phrase, runs in coverage.items()
        if len(runs) < len(member_ids)
    }

    intent = Intent(
        task_id=task.id,
        text=text,
        spans=spans,
        ungrounded_phrases=ungrounded,
        run_coverage=coverage,
        model=getattr(model, "name", None),
    )
    if not text:
        intent.reason = "the model returned no intent"
    elif ungrounded:
        intent.reason = "noun phrases with no span: " + ", ".join(ungrounded)
    elif not spans:
        intent.reason = "no noun phrase to ground"
    elif len(wanted) == 1:
        intent.grounded = True
        intent.unguarded = True  # D81: one Run cannot cross-check itself
    elif gaps:
        listed = [f"{phrase} (not in {', '.join(runs[:3])})" for phrase, runs in list(gaps.items())[:5]]
        intent.reason = "noun phrases not evidenced in every Run: " + "; ".join(listed)
    else:
        intent.grounded = True
    return intent


def apply_intent(task: Task, intent: Intent) -> Task:
    """A copy of the Task carrying a grounded Intent as its name (D47); an ungrounded one is not applied."""
    if not intent.grounded:
        return task.model_copy(deep=True)
    return task.model_copy(
        deep=True,
        update={
            "intent": intent.text,
            "name": task.name or intent.text,
            "unguarded": task.unguarded or intent.unguarded,
        },
    )
