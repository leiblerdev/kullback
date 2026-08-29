"""The Simulated user: rules taken exactly from one trace (D44), answers from the trace or the
world, never invented (D77). What a fact is, how a turn states it and how an agent asks for it come
from the Vocabulary the build derived (D115); the generic core is the default."""

from __future__ import annotations

import re
from typing import Any, Iterable, NamedTuple, Optional

from harness.builder.vocabulary import GENERIC, Vocabulary
from harness.shared.records import DisclosureRule, Event, Trace, UserFact, UserRules

# Whole sentences the recorded user said, kept verbatim: the goal it opened with, the confirmations
# a write needs, the choices it stated and the line it closed on (D44). The wording model never
# touches these, so a confirmation is repeated exactly as it was given.
GOAL, CONFIRMATION, CHOICE, CLOSING = "goal", "confirmation", "choice", "closing"
SPOKEN_FIELDS = (GOAL, CONFIRMATION, CHOICE, CLOSING)

# A sentence is an ask only when it is a question or asks in so many words.
REQUEST_CUE = re.compile(
    r"\b(please provide|provide me|could you|can you|would you|may i (?:have|get)|"
    r"what(?:'s| is) your|let me know|tell me|confirm|share|send me|need your|give me)\b"
)
CONFIRM_REQUEST = re.compile(
    r"\b(confirm|reply yes|shall i|should i (?:proceed|go ahead)|would you like me to|"
    r"do you want me to|is (?:that|this) correct|are you sure|proceed\?|go ahead\?)", re.I
)
OPEN_REQUEST = re.compile(
    r"\b(which (?:item|items|one|option|options|product)|what would you like|"
    r"which would you like|how would you like|what can i help|how can i help|"
    r"what brings you|what do you need)", re.I
)
CLOSE_CUE = re.compile(
    r"\b(anything else|else i can help|have a (?:great|nice|good) day|glad i could help|"
    r"you're (?:very )?welcome|thanks for contacting|thank you for contacting|goodbye)", re.I
)
AGENT_REFUSAL = re.compile(r"\b(i cannot|i can't|i am not able|i'm not able|unable to|not allowed)", re.I)

AFFIRM_CUE = re.compile(
    r"^(yes|yeah|yep|sure|correct|that'?s correct|that'?s right|ok|okay|confirmed|"
    r"please (?:go ahead|proceed)|go ahead|let'?s (?:just )?(?:go ahead|do)|i confirm)\b", re.I
)
CLOSING_CUE = re.compile(
    r"###stop###|\bthat'?s all\b|\bthat is all\b|\bthat covers everything\b|\bnothing else\b|"
    r"\bno other questions\b|\bgoodbye\b|\bhave a (?:great|nice|good) day\b|"
    r"\bthanks? (?:you )?for (?:your help|all your help|making)\b|\bthank you for your help\b", re.I
)

REFUSAL_CUES = (
    "do not remember", "don't remember", "do not have", "don't have", "rather not",
    "prefer not", "cannot share", "can't share", "not comfortable", "will not give",
    "won't give", "no, thank",
)
WALK_AWAY_CUES = ("never mind", "nevermind", "forget it", "do it myself", "cancel this chat")

GENERIC_CLOSE = "No, that is all. Thank you."


def _norm(text: Optional[str]) -> str:
    """Lower case with curly apostrophes flattened, and 'email address' kept out of the address cue."""
    plain = (text or "").replace("’", "'").lower()
    return plain.replace("email address", "email")


def _sentences(body: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.?!:])\s+|\n+", body) if part.strip()]


def _is_request(sentence: str) -> bool:
    return "?" in sentence or bool(REQUEST_CUE.search(sentence))


def _value_of(prefix: str, value: str) -> str:
    """The stored form of a value: an id keeps the mark the customer's tools take (retail's '#')."""
    text = value.strip()
    return text if not prefix or text.startswith(prefix) else prefix + text


def _group(match: re.Match) -> str:
    return match.group(1) if match.groups() else match.group(0)


def extracted_values(text: Optional[str], asked: Iterable[str] = (),
                     vocab: Vocabulary = GENERIC) -> list[tuple[str, str]]:
    """Every (field, value) a user turn states; `asked` allows the bare values that answer an ask."""
    found: list[tuple[str, str]] = []
    body = (text or "").replace("’", "'")
    for spec in vocab.fields:
        if spec.pattern:
            for match in re.finditer(spec.pattern, body):
                found.append((spec.field, _value_of(spec.prefix, _group(match))))
    for field in asked:
        spec = vocab.get(field)
        if spec is None or not spec.asked_only or any(seen == field for seen, _ in found):
            continue
        match = re.search(spec.asked_only, body)
        if match is not None:
            found.append((field, _value_of(spec.prefix, _group(match))))
    return found


def asked_fields(text: Optional[str], vocab: Vocabulary = GENERIC) -> list[str]:
    """The fields an agent turn asks for: request sentences only, in vocabulary order."""
    requests = [sentence for sentence in _sentences(_norm(text)) if _is_request(sentence)]
    if not requests:
        return []
    return [spec.field for spec in vocab.fields
            if any(re.search(cue, sentence) for sentence in requests for cue in spec.cues)]


def derive_user_rules(trace: Trace, vocab: Vocabulary = GENERIC) -> UserRules:
    """Facts, disclosure, refusals and walk-away for one Run, read off the trace's user turns (D44)."""
    rules = UserRules(style_sample=[trace.trace_id])
    seen_values: set[tuple[str, str]] = set()
    disclosed: set[str] = set()
    asked_anywhere: list[str] = []
    answered: set[str] = set()
    pending: list[str] = []
    saw_user_turn = False
    agent_refused = False
    for turn in trace.turns:
        if turn.role == "assistant":
            stated = {field for field, _ in extracted_values(turn.content, vocab=vocab)}
            pending = [field for field in asked_fields(turn.content, vocab=vocab) if field not in stated]
            asked_anywhere += [field for field in pending if field not in asked_anywhere]
            agent_refused = bool(AGENT_REFUSAL.search(turn.content or ""))
            continue
        if turn.role != "user":
            continue
        text = (turn.content or "").strip()
        said = _norm(text)
        values = extracted_values(text, asked=pending, vocab=vocab)
        for field, value in values:
            if (field, value) in seen_values:
                continue
            seen_values.add((field, value))
            rules.facts.append(UserFact(field=field, value=value, span=turn.raw_ptr))
            answered.add(field)
            if field not in disclosed:
                disclosed.add(field)
                on_request = field in pending
                rules.disclosure.append(DisclosureRule(
                    field=field,
                    on_request=on_request,
                    condition=None if on_request else "volunteered",
                ))
        refusing = bool(pending) and any(cue in said for cue in REFUSAL_CUES)
        _spoken_fact(rules, turn, text, said, values, refusing,
                     first=not saw_user_turn and not pending)
        if refusing:
            for field in pending:
                if field not in rules.refusals and field not in answered:
                    rules.refusals.append(field)
        if any(cue in said for cue in WALK_AWAY_CUES) or (agent_refused and CLOSING_CUE.search(said)):
            rules.walk_away.append(text)
        saw_user_turn = True
        pending = []
    rules.refusals = [field for field in rules.refusals if field not in answered]
    if not saw_user_turn:
        rules.incomplete_reasons.append("no user turns in the trace")
    else:
        rules.incomplete_reasons += [
            f"the agent asked for {field} and the trace records no answer"
            for field in asked_anywhere
            if field not in answered and field not in rules.refusals
        ]
    return rules


def _spoken_fact(rules: UserRules, turn: Any, text: str, said: str, values: list,
                 refusing: bool, first: bool) -> None:
    """The goal, a confirmation, a choice or the closing line, kept as the recorded sentence."""
    if not text:
        return
    if CLOSING_CUE.search(said):
        field = CLOSING
    elif first:
        field = GOAL
    elif values or refusing:
        return  # this turn answered a field ask, or refused it; both are recorded already
    elif AFFIRM_CUE.search(said):
        field = CONFIRMATION
    else:
        field = CHOICE
    if field in (GOAL, CLOSING) and any(fact.field == field for fact in rules.facts):
        return
    rules.facts.append(UserFact(field=field, value=text, span=turn.raw_ptr))


class FactLookup(NamedTuple):
    """What the Starting state holds for a field; synthetic rows make the turn Assisted (D40, D77)."""
    value: Any
    synthetic: bool = False


def dict_reader(mapping: dict):
    """A Starting state reader over a plain dict, for tests and small callers."""

    class _DictReader:
        def get(self, field: str):
            return mapping.get(field)
    return _DictReader()


def _words(field: str) -> str:
    return field.replace("_", " ")


def _field_of(message: Any, key: str) -> str:
    if isinstance(message, dict):
        return message.get(key) or ""
    return getattr(message, key, "") or ""


def _flatten(row: Any) -> dict[str, list]:
    """Every leaf value seen under each key, nested dicts included, so 'zip' inside 'address' is
    found. A key seen more than once (two payment methods that both carry 'source') keeps every
    value it was seen with, instead of the first one a dict happened to iterate to, so a caller can
    tell a genuine single answer from an ambiguous one (row 11).
    """
    flat: dict[str, list] = {}
    if not isinstance(row, dict):
        return flat
    for key, value in row.items():
        if isinstance(value, dict):
            for inner_key, inner_values in _flatten(value).items():
                flat.setdefault(inner_key, []).extend(inner_values)
        elif not isinstance(value, list):
            flat.setdefault(key, []).append(value)
    return flat


def _one(values: Optional[list]) -> Any:
    """The value at one key, only when every occurrence the row carries for it agrees; otherwise the
    field is unavailable rather than answered from whichever occurrence came first."""
    if not values:
        return None
    return values[0] if len({str(v) for v in values}) == 1 else None


def _row_value(row: dict, field: str) -> Any:
    """One field of this user's row, in the customer's own column names."""
    flat = _flatten(row)
    if field == "name":
        first, last = _one(flat.get("first_name")), _one(flat.get("last_name"))
        return f"{first} {last}" if first and last else _one(flat.get("name"))
    if field == "address":
        address = row.get("address") if isinstance(row.get("address"), dict) else None
        if address:
            parts = [address.get(key) for key in ("address1", "address2", "city", "state", "zip")]
            return ", ".join(str(part) for part in parts if part)
        return _one(flat.get("address"))
    for key in {"card_last4": ("card_last4", "last_four", "last4"),
                "payment_method": ("payment_method", "source"),
                "phone": ("phone", "phone_number")}.get(field, (field,)):
        value = _one(flat.get(key))
        if value is not None:
            return value
    return None


class SimulatedUser:
    """Replies as the recorded user: facts from the rules, then the Starting state, then nothing (D77)."""

    def __init__(self, rules: UserRules, starting_state_reader: Any = None, model: Any = None,
                 identity: Optional[dict] = None, vocab: Vocabulary = GENERIC):
        self.rules = rules
        self.reader = starting_state_reader
        self.model = model
        self.vocab = vocab
        self.events: list[Event] = []
        self.done = False
        identity_fields = vocab.by_kind("identity")
        self.identity = dict(identity) if identity else {
            fact.field: fact.value for fact in rules.facts if fact.field in identity_fields}
        self._row: Any = None
        self._row_read = False
        self._used: dict[str, int] = {CONFIRMATION: 0, CHOICE: 0}
        self._silent = 0

    def reply(self, transcript: list) -> str:
        question = ""
        for message in transcript:
            if _field_of(message, "role") == "assistant":
                question = _field_of(message, "content")
        answers: dict[str, Any] = {}
        sources: dict[str, str] = {}
        spoken: list[str] = []
        unavailable: list[str] = []
        assisted = False
        for field in asked_fields(question, vocab=self.vocab):
            fact = self._fact(field)
            if fact is not None:
                answers[field], sources[field] = fact.value, "rules"
                continue
            if field in self.rules.refusals:
                sources[field] = "refused"
                continue
            found = self._from_world(field)
            if found is None:
                unavailable.append(field)
                sources[field] = "unavailable"
                continue
            answers[field], sources[field] = found.value, "world"
            assisted = assisted or found.synthetic
        if not self.events:
            self._open(answers, sources, spoken)
        if not (answers or unavailable or spoken):
            self._respond(question, sources, spoken, unavailable)
        self._silent = 0 if (answers or unavailable or spoken) else self._silent + 1
        text = self._say(question, answers, sources, spoken, unavailable)
        self.events.append(Event(
            idx=len(self.events),
            type="user_turn",
            payload={
                "text": text,
                "fields": list(sources),
                "sources": sources,
                "unavailable_fields": unavailable,
                "tags": ["fact_unavailable"] if unavailable else [],
            },
            assisted=assisted,
        ))
        return text

    def _open(self, answers: dict, sources: dict, spoken: list) -> None:
        """The opening reply: the goal the recorded user stated, then what it volunteered (D44)."""
        goal = self._fact(GOAL)
        if goal is not None:
            spoken.append(str(goal.value))
            sources[GOAL] = "rules"
        self._volunteer(answers, sources)

    def _respond(self, question: str, sources: dict, spoken: list, unavailable: list) -> None:
        """Nothing was asked by name: a confirmation, a stated choice, or the close of the Run."""
        for field, reuse_last, cue in ((CONFIRMATION, True, CONFIRM_REQUEST),
                                       (CHOICE, False, OPEN_REQUEST)):
            if not cue.search(question or ""):
                continue
            fact = self._next(field, reuse_last=reuse_last)
            if fact is not None:
                spoken.append(str(fact.value))
                sources[field] = "rules"
            else:  # the trace holds no answer to this request, and nothing is invented (D77)
                unavailable.append(field)
                sources[field] = "unavailable"
            return
        if CLOSE_CUE.search(question or "") or self._silent >= 1:
            closing = self._fact(CLOSING)
            spoken.append(str(closing.value) if closing is not None else GENERIC_CLOSE)
            sources[CLOSING] = "rules" if closing is not None else "generic_close"
            self.done = True

    def _fact(self, field: str) -> Optional[UserFact]:
        for fact in self.rules.facts:
            if fact.field == field:
                return fact
        return None

    def _next(self, field: str, reuse_last: bool = False) -> Optional[UserFact]:
        """Recorded free-text answers are used in the order the recorded user gave them."""
        facts = [fact for fact in self.rules.facts if fact.field == field]
        if not facts:
            return None
        index = self._used.get(field, 0)
        if index >= len(facts):
            return facts[-1] if reuse_last else None
        self._used[field] = index + 1
        return facts[index]

    def _from_world(self, field: str) -> Optional[FactLookup]:
        """Read the Task's Starting state as this user: the overlay first, then the shared world (D74)."""
        if self.reader is None:
            return None
        if hasattr(self.reader, "shared") or hasattr(self.reader, "overlay"):
            row = self._user_row()
            if row is None:
                return None
            value = _row_value(row[1], field)
            return None if value is None else FactLookup(value, self._is_synthetic(row[0]))
        found = self.reader.get(field)
        if found is None:
            return None
        return found if isinstance(found, FactLookup) else FactLookup(found)

    def _user_row(self) -> Optional[tuple[str, dict]]:
        """This user's own row, matched on the identity facts; no identity means no row (D77)."""
        if self._row_read:
            return self._row
        self._row_read = True
        if self.identity:
            for tables in (getattr(self.reader, "overlay", {}) or {},
                           getattr(self.reader, "shared", {}) or {}):
                matches = _matching_rows(tables, self.identity)
                if len(matches) > 1:  # an id names one row; other tables only carry it
                    named = {str(value) for value in self.identity.values()}
                    matches = [match for match in matches if match[0] in named]
                if len(matches) == 1:
                    self._row = matches[0]
                    break
        return self._row

    def _is_synthetic(self, row_id: str) -> bool:
        synthetic = getattr(self.reader, "synthetic_rows", ()) or ()
        return row_id in synthetic

    def _volunteer(self, answers: dict, sources: dict) -> None:
        """On the opening reply, give the facts the recorded user gave without being asked (D44)."""
        for rule in self.rules.disclosure:
            if rule.on_request or rule.field in sources or rule.field in self.rules.refusals:
                continue
            fact = self._fact(rule.field)
            if fact is not None:
                answers[rule.field], sources[rule.field] = fact.value, "volunteered"

    def _say(self, question: str, answers: dict, sources: dict, spoken: list, unavailable: list) -> str:
        plain = list(spoken)
        plain += [f"My {_words(field)} is {value}." for field, value in answers.items()]
        plain += [f"I would rather not share my {_words(f)}."
                  for f, source in sources.items() if source == "refused"]
        plain += ["I do not have an answer for that." if field in SPOKEN_FIELDS
                  else f"I do not have my {_words(field)}." for field in unavailable]
        sentence = " ".join(plain) or "Okay, thank you."
        if self.model is None or unavailable or spoken or not answers:
            return sentence
        return self._word(question, sentence, answers, sources)

    def _word(self, question: str, sentence: str, answers: dict, sources: dict) -> str:
        """The model fills wording only; a wording that drops or adds a fact is thrown away (D44)."""
        reply = self.model.query([
            {"role": "system", "content": "You are the customer in a support chat. Say what you are "
                                          "told to say, in your own words, briefly. Add no new facts."},
            {"role": "user", "content": f"The agent said: {question}\nSay: {sentence}"},
        ])
        text = (reply.content or "").strip()
        if not text or any(str(value) not in text for value in answers.values()):
            return sentence
        allowed = [str(value).lower() for value in answers.values()]
        for _, value in extracted_values(text, asked=sources, vocab=self.vocab):
            spoken_value = value.lower()
            if not any(spoken_value in known or known in spoken_value for known in allowed):
                return sentence  # the model added a fact the rules and the world never gave
        return text


def _matching_rows(tables: dict, identity: dict) -> list[tuple[str, dict]]:
    """Rows that agree with every identity fact they carry, and carry enough of them to be this user."""
    needed = min(2, len(identity))
    matches: list[tuple[str, dict]] = []
    for rows in (tables or {}).values():
        for row_id, row in (rows.items() if isinstance(rows, dict) else []):
            if not isinstance(row, dict):
                continue
            hits = 0
            for field, value in identity.items():
                held = _row_value(row, field)
                if held is None:
                    continue
                if str(held).strip().lower() != str(value).strip().lower():
                    hits = -1
                    break
                hits += 1
            if hits >= needed:
                matches.append((str(row_id), row))
    return matches
