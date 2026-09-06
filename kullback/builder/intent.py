"""Writes a Task's one-line Intent and refuses it unless every noun phrase points to a span in
every member Run (D47, D83). A refused line is not the end of it: the phrases with no span are read
back to the model, with the words of each phrase the evidence does and does not show, and it writes
again, up to MAX_INTENT_ATTEMPTS times, and the best attempt is kept. The Intent record, its spans
and `apply_intent` live in kullback.runner.records so the Examiner reads an Intent without importing
the Builder (D123); they are re-exported here under the names this module always had.

One tokeniser and one clause splitter read both sides, the Intent's words and the span's, so the two
agree by construction: a price, a time and an id stay whole ("$17.99" is the token "17.99", "#W1234"
is "w1234"), a hyphen, a slash and an underscore join rather than break, and `normalise` folds simple
inflections together so "earbud" and "earbuds" are one word. A phrase grounds either whole, as a
contiguous run inside one clause, or word by word within one Run when the customer said the same
words in another arrangement. `IntentSpan` is in the frozen `kullback.runner.records`, so which of
the two a span used is kept in the span's display-only `label` (read it with `span_mode`) rather than
in a new field.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional, Sequence

from kullback.ai.provider import Model
from kullback.builder.cluster import APOSTROPHE_RE, STOPWORDS
from kullback.builder.cluster import first_line as _shared_first_line
from kullback.runner.records import (  # noqa: F401 - Intent, IntentSpan, SpanSource and apply_intent are re-exported
    Intent,
    IntentSpan,
    SpanSource,
    Task,
    ToolCall,
    Trace,
    apply_intent,
)

MAX_INTENT_CHARS = 200
MAX_PROMPT_RUNS = 5  # D65: the prompt samples the Task's Runs, the grounding still reads all of them
MAX_INTENT_ATTEMPTS = 3  # the first write plus two rewrites with the ungrounded phrases named
MAX_LISTED_PHRASES = 5
MAX_PROMPT_SPAN_CHARS = 300
MAX_LISTED_WORDS = 8
# One token: letters and digits, with a dot, comma or colon kept only between two digits, so "$17.99"
# is "17.99", "12:30" is "12:30" and "1,000" is "1,000", while "order 2.Next" is still two tokens.
# An underscore is not part of a token, so credit_card_1234 is credit, card, 1234 and a phrase saying
# "credit card" reaches it; the phrase "credit_card_1234" splits the same way, so both forms match.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:(?<=[0-9])[.,:][0-9]+)*")
# A gap of only these between two tokens keeps them in one phrase: "gift-card", "e-mail", "paypal/venmo"
# and credit_card_1234 read as the words they spell, either way round.
JOINER_RE = re.compile("[\\s_/\\-\u2010-\u2015]*")
# Punctuation ends a clause, as it ends a thought, unless it sits between two digits and is part of a
# number: splitting "$17.99" at the point is what put the phrase "99 price difference" in a live build.
CLAUSE_RE = re.compile(r"[;!?\n]|(?<![0-9])[.,:]|[.,:](?![0-9])")
# English customer-service verbs. A run of words is cut at one of these and the verb stands alone, so
# "change wireless earbuds" asks the evidence for "change" and for "wireless earbuds" rather than for
# the three words in that order in one clause, which no customer ever writes.
ACTION_VERB_WORDS = """
    activate add adjust apply approve arrange assign book cancel change charge check close combine
    compare complete confirm contact correct credit deactivate delete deliver disable dispatch
    downgrade enable escalate exchange extend find fix issue locate modify move notify open pause pay
    place process rebook redeem refund reimburse remove renew repair replace report request
    reschedule reset resend reserve resolve restart restock resume return schedule send ship split
    submit subscribe swap switch track transfer unsubscribe update upgrade verify
""".split()
# The frame every model writes the Intent in ("The user wanted help to ...") is not evidence of anything
# and grounded on no span in any domain; it is stripped before the noun phrases are read.
FRAME_RE = re.compile(r"^\s*(?:the\s+)?(?:user|customer|caller|client)?(?:'s)?\s*(?:wanted|wants|asked|asks|"
                      r"needed|needs|requested|requests|would like|is asking|was asking)?"
                      r"(?:\s+(?:help|to|for|with|about))*\s*", re.IGNORECASE)
# A phrase the user ruled out in the same breath is not evidence that the user wanted it.
NEGATIONS = frozenset("not no never dont cannot cant wont without nor neither".split())

def _undouble(stem: str) -> str:
    """"cancell" is "cancel" and "shipp" is "ship"; a doubled consonant is spelling, not a word."""
    doubled = len(stem) > 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou"
    return stem[:-1] if doubled else stem


def _strip_suffix(word: str) -> str:
    """One inflection off the end of a word, and only where four letters are left standing."""
    for suffix, stem in (("ies", word[:-3] + "y"), ("es", word[:-2]), ("s", word[:-1]),
                         ("ing", _undouble(word[:-3])), ("ed", _undouble(word[:-2])), ("e", word[:-1])):
        if not word.endswith(suffix) or len(word) - len(suffix) < 4:
            continue
        if suffix == "s" and word.endswith("ss"):
            continue  # "address" is not the plural of "addres"
        return stem
    return word


def normalise(word: str) -> str:
    """One spelling for a word's simple inflections, so "earbud" and "earbuds" are the same word.

    A suffix stripper, not a lemmatiser: plurals, past tense and -ing, stripped until nothing more
    comes off, so "addresses", "address" and "changes", "changed", "change" each land on one stem. It
    never touches a word carrying a digit, so an order id and a price keep their spelling. A real
    lemmatiser (spaCy, nltk) would beat it on irregulars, and is not worth a model download and a new
    dependency in the Builder's hot path for the handful of words a customer-service line uses.
    """
    if not word.isalpha():
        return word
    while True:
        stem = _strip_suffix(word)
        if stem == word:
            return word
        word = stem


ACTION_VERBS = frozenset(normalise(word) for word in ACTION_VERB_WORDS)


def _tokens(text: Optional[str]) -> list[str]:
    """The words of one piece of text, folded to their stems; the same reading on both sides."""
    lowered = APOSTROPHE_RE.sub("", (text or "").lower())
    return [normalise(token) for token in TOKEN_RE.findall(lowered)]


def content_words(phrase: str) -> list[str]:
    """The words of a phrase that have to be evidenced; a stopword is evidence of nothing."""
    lowered = APOSTROPHE_RE.sub("", (phrase or "").lower())
    words = TOKEN_RE.findall(lowered)
    return [w for w in words if w not in STOPWORDS] or words


def noun_phrases(text: str) -> list[str]:
    """Runs of non-stopword tokens, in order, deduplicated. A cheap stand-in for a noun-phrase parser.

    A run also ends at an action verb, and the verb stands as a phrase of its own: a customer writes
    "order change" and "wireless earbuds" in two breaths, so asking the evidence for the model's
    "change wireless earbuds" in those words in that order asks for a sentence nobody wrote. Both
    halves still have to be evidenced, so no word of the Intent goes unchecked (D47).
    """
    phrases: list[str] = []
    current: list[str] = []
    low = APOSTROPHE_RE.sub("", (text or "").lower())
    end = 0
    for match in TOKEN_RE.finditer(low):
        broken = JOINER_RE.fullmatch(low[end : match.start()]) is None  # punctuation ends a phrase
        end = match.end()
        token = match.group()
        # An -ing form is a modifier here, not the action: "shipping address" and "tracking number"
        # are one thing each, while "change", "changed" and "changes" all take an object.
        verb = normalise(token) in ACTION_VERBS and not token.endswith("ing")
        if (broken or verb or token in STOPWORDS) and current:
            phrases.append(" ".join(current))
            current = []
        if verb:
            phrases.append(token)
        elif token not in STOPWORDS:
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
    """The span's text as clauses of tokens; punctuation ends a clause, as it ends a thought.

    Not a point between two digits: "$17.99" is one number and one token, whatever else the sentence
    around it does.
    """
    lowered = APOSTROPHE_RE.sub("", (text or "").lower())
    return [[normalise(t) for t in TOKEN_RE.findall(clause)] for clause in CLAUSE_RE.split(lowered)]


def _covers(clauses: Sequence[Sequence[str]], phrase: str, honour_negation: bool) -> bool:
    """Does one clause of the span say this phrase, in these words, in this order?

    Set containment said yes to "late delivery" for "the delivery was fine but the card was late",
    so the phrase has to be a contiguous run of tokens inside one clause (D47). This is the strict
    reading; `_ground_in_run` falls back to the words on their own when it fails.
    """
    wanted = _tokens(phrase)
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


def _run_evidence(
    traces: Sequence[Trace], write_tools: Optional[set[str]] = None
) -> list[tuple[str, list[tuple[IntentSpan, list[list[str]], bool]]]]:
    """Each Run's spans with their clauses read once, in trace order."""
    return [
        (trace.trace_id,
         [(span, _clauses(span.text), span.source == "user_utterance")
          for span in span_candidates(trace, write_tools)])
        for trace in traces
    ]


def _ground_in_run(
    spans: Sequence[tuple[IntentSpan, list[list[str]], bool]], phrase: str
) -> tuple[Optional[IntentSpan], str]:
    """The span this Run evidences the phrase with, and how it did it.

    Whole phrase first, in one clause, in these words, in this order. When no span says it that way,
    every content word of the phrase has to be evidenced somewhere in this same Run instead, in any
    clause and in any order: the model wrote "change wireless earbuds on order #W5061109" for a
    customer who said "order change" and "wireless earbuds" a sentence apart, and those are the
    customer's own words, not invented ones. Negation still bites, word by word, so a word the
    customer ruled out grounds nothing either way (D47).
    """
    for span, clauses, honour_negation in spans:
        if _covers(clauses, phrase, honour_negation):
            return span, "phrase"
    chosen: Optional[IntentSpan] = None
    for word in content_words(phrase):
        hit = next((s for s, clauses, negation in spans if _covers(clauses, word, negation)), None)
        if hit is None:
            return None, ""
        chosen = chosen if chosen is not None else hit
    return (chosen, "tokens") if chosen is not None else (None, "")


def span_mode(span: IntentSpan) -> str:
    """How this span was grounded: the whole "phrase", or its words as "tokens".

    `IntentSpan` is in the frozen `kullback.runner.records` (D61), so the mode rides in `label`, the
    one field on it that is shown and never matched against, rather than in a field of its own.
    """
    label = span.label or ""
    return "tokens" if label == "tokens" or label.endswith(" (tokens)") else "phrase"


def _with_mode(span: IntentSpan, phrase: str, mode: str) -> IntentSpan:
    label = span.label
    if mode == "tokens":
        label = f"{label} (tokens)" if label else "tokens"
    return span.model_copy(update={"phrase": phrase, "label": label})


def _matching_spans(
    phrases: Sequence[str],
    traces: Sequence[Trace],
    write_tools: Optional[set[str]] = None,
) -> dict[str, list[IntentSpan]]:
    """Every span that evidences each phrase, in trace order, at most one span per Run."""
    runs = _run_evidence(traces, write_tools)
    found: dict[str, list[IntentSpan]] = {}
    for phrase in phrases:
        spans: list[IntentSpan] = []
        for _, run_spans in runs:
            span, mode = _ground_in_run(run_spans, phrase)
            if span is not None:
                spans.append(_with_mode(span, phrase, mode))
        found[phrase] = spans
    return found


def _word_evidence(
    phrases: Sequence[str], traces: Sequence[Trace], write_tools: Optional[set[str]] = None
) -> dict[str, tuple[list[str], list[str]]]:
    """Per phrase, the words some Run does say and the words no Run says, for the rewrite prompt."""
    runs = _run_evidence(traces, write_tools)
    out: dict[str, tuple[list[str], list[str]]] = {}
    for phrase in phrases:
        shown, missing = [], []
        for word in content_words(phrase):
            anywhere = any(_covers(clauses, word, negation)
                           for _, spans in runs for _, clauses, negation in spans)
            (shown if anywhere else missing).append(word)
        out[phrase] = (shown, missing)
    return out


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


def _intent_prompt(traces: Sequence[Trace], write_tools: Optional[set[str]], *,
                   hint: Optional[str] = None, feedback: Optional[str] = None) -> str:
    """The evidence from a bounded sample of the Task's Runs (D65: no call may grow with the corpus).

    `hint` is what a repair asked for, `feedback` what the last attempt got wrong; both are short
    lines beside the same evidence, never more of it.
    """
    shown = list(traces[:MAX_PROMPT_RUNS])
    lines = [
        "Write one line saying what the user wanted, common to all these runs.",
        "Start with the verb (as in: cancel order #W1). Do not write 'the user wanted'.",
        "Use only words evidenced below; every noun must appear in the evidence.",
    ]
    if len(traces) > len(shown):
        lines.append(f"These are {len(shown)} of the task's {len(traces)} runs; say only what all of them show.")
    if hint:
        lines.append(f"A repair asks for this: {hint}")
    lines.append("")
    for trace in shown:
        for span in span_candidates(trace, write_tools):
            text = " ".join(span.text.split())[:MAX_PROMPT_SPAN_CHARS]
            label = f" {span.label}" if span.label else ""
            lines.append(f"{trace.trace_id} {span.source}{label}: {text}")
    if feedback:
        lines += ["", feedback]
    lines += ["", "Reply with the one line only."]
    return "\n".join(lines)


def strip_frame(text: str) -> str:
    """"The user wanted help to cancel order #W1" is "cancel order #W1"; a line that is only the frame stays."""
    stripped = FRAME_RE.sub("", text or "", count=1).strip()
    return stripped if stripped else (text or "").strip()


def _first_line(text: Optional[str]) -> str:
    return strip_frame(_shared_first_line(text, MAX_INTENT_CHARS) or "")


def _gaps(coverage: dict[str, list[str]], member_ids: Sequence[str]) -> dict[str, list[str]]:
    """Per phrase, the member Runs that do not evidence it; a phrase every Run shows is not in here."""
    return {
        phrase: [rid for rid in member_ids if rid not in set(runs)]
        for phrase, runs in coverage.items()
        if len(runs) < len(member_ids)
    }


def _graded(task: Task, text: str, members: Sequence[Trace], write_tools: Optional[set[str]],
            model: Model) -> Intent:
    """One line as an Intent record: the spans behind its phrases and the reason it is refused, if it is.

    Nothing here asks the model anything; it is the grounding `write_intent` runs on each attempt.
    """
    found = _matching_spans(noun_phrases(text), members, write_tools)
    spans, ungrounded = _choose_spans(found)
    coverage = {phrase: sorted(s.trace_id for s in matches) for phrase, matches in found.items() if matches}
    member_ids = sorted(t.trace_id for t in members)
    gaps = _gaps(coverage, member_ids)

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
    elif len(member_ids) == 1:
        intent.grounded = True
        intent.unguarded = True  # D81: one Run cannot cross-check itself
    elif gaps:
        listed = [f"{phrase} (not in {', '.join(runs[:3])})"
                  for phrase, runs in list(gaps.items())[:MAX_LISTED_PHRASES]]
        intent.reason = "noun phrases not evidenced in every Run: " + "; ".join(listed)
    else:
        intent.grounded = True
    return intent


def _score(intent: Intent, member_ids: Sequence[str]) -> tuple:
    """How good one attempt is, smallest first: a line at all, then fewest ungrounded phrases, then grounded.

    An empty reply grounds nothing and has no phrase to be ungrounded, so it would otherwise score
    better than a line with one phrase out of place.
    """
    return (0 if intent.text else 1,
            len(intent.ungrounded_phrases),
            0 if intent.grounded else 1,
            len(_gaps(intent.run_coverage, member_ids)))


def _feedback(intent: Intent, member_ids: Sequence[str], members: Sequence[Trace],
              write_tools: Optional[set[str]] = None) -> str:
    """What to tell the model about the attempt it just made: the phrases that are not evidenced, named.

    The first live builds refused 160 of 205 Tasks on phrases like "99 price difference" that no Run
    says in those words, and the model was never told: it was asked once and its answer graded in
    silence. Naming the phrase is still not enough to rewrite from, so each one is broken into the
    words the evidence does show and the words it does not; the rewrite then has to change one word,
    not guess at a whole line.
    """
    lines = [f'Your last line was: "{intent.text}"' if intent.text else "Your last reply held no line."]
    if intent.ungrounded_phrases:
        listed = intent.ungrounded_phrases[:MAX_LISTED_PHRASES]
        lines.append("These phrases have no span in the evidence: " + ", ".join(listed) + ".")
        words = _word_evidence(listed, members, write_tools)
        for phrase in listed:
            shown, missing = words[phrase]
            parts = []
            if shown:
                parts.append("the evidence shows " + ", ".join(shown[:MAX_LISTED_WORDS]))
            if missing:
                parts.append("no run says " + ", ".join(missing[:MAX_LISTED_WORDS]))
            lines.append(f'In "{phrase}": ' + "; ".join(parts) + ".")
    gaps = _gaps(intent.run_coverage, member_ids)
    if gaps:
        listed = [f"{phrase} (not in {', '.join(runs[:3])})"
                  for phrase, runs in list(gaps.items())[:MAX_LISTED_PHRASES]]
        lines.append("These phrases are not evidenced in every run: " + "; ".join(listed) + ".")
    lines.append("Rewrite the intent using only words that appear in the evidence, for every run.")
    return "\n".join(lines)


def write_intent(
    model: Model,
    task: Task,
    traces: Iterable[Trace],
    *,
    write_tools: Optional[set[str]] = None,
    hint: Optional[str] = None,
) -> Intent:
    """The Task's Intent, grounded phrase by phrase; a single-Run Task skips the cross-run check (D97).

    The cross-run check asks whether the Task's Runs all show the Intent, which is what makes it the
    Task's Intent (D83), not whether the spans the chooser picked happen to name two Runs. That
    counted spans, so one phrase both Runs said was refused, while the union of two Runs' different
    intents was accepted because it had one span from each.

    An attempt that grounds is the answer. One that does not is fed back to the model with its
    unevidenced phrases named, up to `MAX_INTENT_ATTEMPTS` writes in all, and the best of them is
    kept: a line at all before none, then the fewest phrases with no span. `hint` is what a repair
    asked for and rides on every attempt's prompt.
    """
    wanted = list(task.run_ids)
    members = [t for t in traces if t.trace_id in set(wanted)]
    if not members:
        raise ValueError(f"task {task.id} has no member traces among the traces given")
    missing_traces = [rid for rid in wanted if rid not in {t.trace_id for t in members}]
    if missing_traces:
        raise ValueError(f"task {task.id} is missing the traces for {', '.join(missing_traces)}")
    members.sort(key=lambda t: wanted.index(t.trace_id))
    member_ids = sorted(t.trace_id for t in members)

    best: Optional[Intent] = None
    feedback: Optional[str] = None
    for _ in range(MAX_INTENT_ATTEMPTS):
        prompt = _intent_prompt(members, write_tools, hint=hint, feedback=feedback)
        reply = model.query([{"role": "user", "content": prompt}])
        intent = _graded(task, _first_line(reply.content), members, write_tools, model)
        if best is None or _score(intent, member_ids) < _score(best, member_ids):
            best = intent
        if intent.grounded:
            break
        feedback = _feedback(intent, member_ids, members, write_tools)
    return best
