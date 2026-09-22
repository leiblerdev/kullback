"""The rule-driven Simulated user: rules taken exactly from one trace (D44), answers from the trace
or the world, never invented (D77).

It moved here from kullback/builder/user_sim.py in D214, unchanged: the Simulated user is an agent
of the harness in its own right, beside the Builder and the Examiner, and it is the floor the agent
user (agent.py) has to beat before it drives a Task and the fallback whenever a guard drops the
agent's turn. `kullback.builder.user_sim` re-exports every name, so a caller that has always read
them from there still does. What a fact is, how a turn states it and how an agent asks for it come
from the Vocabulary the build derived (D115); the generic core is the default.

It ends by protocol and answers from typed facts (D210). Every end carries one of four reason
kinds, so a report can tell a Run that finished from one that ran dry; a fact the user itself never
gave is a record fact it points at rather than speaks; and every answer goes through the same
class-generic strip an Intent goes through (D196), so no value only the system knew is handed over.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, NamedTuple, Optional  # noqa: F401  - the surface holds

from kullback.runner.records import DisclosureRule, Event, Trace, UserFact, UserRules  # noqa: F401  - as above
from kullback.user.vocabulary import GENERIC, Vocabulary

# Whole sentences the recorded user said, kept verbatim: the goal it opened with, the confirmations
# a write needs, the choices it stated and the line it closed on (D44). The wording model never
# touches these, so a confirmation is repeated exactly as it was given.
GOAL, CONFIRMATION, CHOICE, CLOSING = "goal", "confirmation", "choice", "closing"
SPOKEN_FIELDS = (GOAL, CONFIRMATION, CHOICE, CLOSING)

# A sentence is an ask only when it is a question or asks in so many words.
REQUEST_CUE = re.compile(
    r"\b(please provide|provide me|could you|can you|would you|may i (?:have|get)|"
    r"what(?:'s| is) your|let me know|tell me|confirm|share|send me|need your|give me|"
    r"need the following)\b"
)
# A request that ends in a colon goes on to name what it asks for, one list item per line.
LIST_ITEM = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
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
    r"\bthat'?s it\b|\bthat is it\b|\ball set\b|\bnothing further\b|\bnothing more\b|"
    r"\bthanks? (?:you )?for (?:your help|all your help|making)\b|\bthank you for your help\b", re.I
)

# Two values of one field: the one the account holds and the one the user is moving to. A recorded
# sentence that names a change states the second; a question that names the stored one asks for the
# first. Both are the words any domain's users and agents use, not any one customer's.
CHANGE_CUE = re.compile(
    r"\b(new|newly|change|changed|changing|update|updated|updating|instead|different|another|"
    r"moving|move|switch|switching|replace|ship it to|send it to)\b", re.I
)
STORED_CUE = re.compile(
    r"\b(on file|on record|on the account|on my account|on your account|in our system|"
    r"current|currently|existing|original|registered|already have|we have)\b", re.I
)
# A word that names a field points at something else when the next word re-points it: "the name of
# the product" is not the user's name.
REPOINT = r"(?!\s+of\b)"
STOP_WORDS = frozenset(
    "a an the my your our their this that these those is are was were do does did can could would "
    "will shall may i you we it me us to for of on in at and or with please what which who whose "
    "how thanks thank yes no ok okay so if be been am".split()
)

REFUSAL_CUES = (
    "do not remember", "don't remember", "do not have", "don't have", "rather not",
    "prefer not", "cannot share", "can't share", "not comfortable", "will not give",
    "won't give", "no, thank",
)
WALK_AWAY_CUES = ("never mind", "nevermind", "forget it", "do it myself", "cancel this chat")

# Markdown emphasis and the runs of whitespace a recorded turn wraps its values in: what the eye
# skips over when it reads "my name is **Chen Johnson**" and a regex does not.
EMPHASIS = re.compile(r"[*_`~]+")
# The two parts of a name a customer's tools take separately. An agent asking for "your name" is
# answered from one `name` fact, so the parts of one call are joined, in this order.
NAME_PARTS = ("first_name", "last_name")
NAME_ASK = "What is your full name?"

# --- how a conversation ends, in one vocabulary (D210) -------------------------------------
# Four kinds, read in this order wherever more than one is true at once. They live here and not in
# the Runner because the Runner is frozen; the Runner's own end reasons are read below and mapped
# into them, so a replayed Run and a Simulated one are reported in one vocabulary.
GOAL_SATISFIED = "goal_satisfied"        # every write the goal implies is confirmed done
SCENARIO_EXHAUSTED = "scenario_exhausted"  # the user has nothing left that answers this Candidate
HANDED_OFF = "handed_off"                # the Candidate closed the conversation or passed it on
GAVE_UP = "gave_up"                      # the turn limit, which the loop holds and the user never sees
USER_END_KINDS = (GOAL_SATISFIED, SCENARIO_EXHAUSTED, HANDED_OFF, GAVE_UP)
# The Runner's own end reasons, read for the Runs the Simulated user did not end itself.
TURN_LIMIT_REASONS = frozenset(("max_turns", "max_steps"))
STOP_REASONS = frozenset(("transfer", "user_stop", "agent_stop"))
# The Runner's handoff token, spelled here so this module reads it without importing the loop.
TRANSFER_MARKER = "###TRANSFER###"
HANDOFF_CUE = re.compile(
    r"###TRANSFER###|\b(transfer(?:ring)? you|transfer this|connect(?:ing)? you (?:with|to)|"
    r"hand(?:ing)? (?:you |this )?(?:over|off)|escalat(?:e|ing) (?:this|you)|"
    r"pass(?:ing)? you (?:on|to)|a (?:human|colleague|specialist) (?:agent )?will)\b", re.I)
# How many turns the user can have nothing at all for before its scenario has run out, and how many
# turns of nothing to say. Both were already the inputs the close was taken on; they are inputs to
# a kind now, never the end itself.
UNANSWERABLE_LIMIT = 2
SILENCE_LIMIT = 1

# --- what class a fact is in (D210) --------------------------------------------------------
# Askable facts are typed apart from record facts, so a leak cannot happen by the user answering.
ASKABLE = "askable"  # a user of this Task said it, so this user says it once asked
RECORD = "record"    # only the world holds it, so the user points the Candidate at the tools
# The source names a turn records for the two answers that are not a value.
RECORD_SOURCE = "record"
STRIPPED_SOURCE = "stripped"
RECORD_LINE = "I do not have my {} to hand, you can look it up."
# Tags the turn carries so its end and its non-answers reach the Run's own file through the loop.
IN_RECORD_TAG = "fact_in_record"
STRIPPED_TAG = "user_answer_stripped"

GENERIC_CLOSE = "No, that is all. Thank you."
# The source name a restated goal carries, so a report can tell the opening request from the same
# sentence said a second time because the agent had not acted on it.
GOAL_RESTATED = "goal_restated"
# D44: a question the recorded user was never asked gets a representative answer. For "do you
# confirm" the representative answer of a user who asked for the action is yes, unless the
# recording holds a no; the source names it so a report can tell the yes from a recorded one.
GENERIC_CONFIRM = "Yes, please go ahead."
NEGATIVE_CUE = re.compile(r"^(no|nope|don'?t|do not|not yet|hold on|wait|stop|cancel that)\b", re.I)


def _norm(text: Optional[str]) -> str:
    """Lower case with curly apostrophes flattened, and 'email address' kept out of the address cue."""
    plain = (text or "").replace("’", "'").lower()
    return plain.replace("email address", "email")


def _sentences(body: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.?!:])\s+|\n+", body) if part.strip()]


def _is_request(sentence: str) -> bool:
    return "?" in sentence or bool(REQUEST_CUE.search(sentence))


def _value_of(prefix: str, value: str) -> str:
    """The stored form of a value: an id keeps the mark the customer's tools take (a leading '#', say)."""
    text = value.strip()
    return text if not prefix or text.startswith(prefix) else prefix + text


def _group(match: re.Match) -> str:
    return match.group(1) if match.groups() else match.group(0)


def extracted_values(text: Optional[str], asked: Iterable[str] = (),
                     vocab: Vocabulary = GENERIC) -> list[tuple[str, str]]:
    """Every (field, value) a user turn states; `asked` allows the bare values that answer an ask."""
    found: list[tuple[str, str]] = []
    claimed: list[tuple[int, int]] = []
    body = (text or "").replace("’", "'")
    for spec in vocab.fields:
        if spec.pattern:
            for match in re.finditer(spec.pattern, body):
                found.append((spec.field, _value_of(spec.prefix, _group(match))))
                claimed.append(match.span())
    for field in asked:
        spec = vocab.get(field)
        if spec is None or not spec.asked_only or any(seen == field for seen, _ in found):
            continue
        # A bare value is read only where no shaped field already read one: build 8's Simulated
        # user answered an address question with the order id the same turn had named.
        for match in re.finditer(spec.asked_only, body):
            if any(start < match.end() and match.start() < end for start, end in claimed):
                continue
            found.append((field, _value_of(spec.prefix, _group(match))))
            break
    return found


def _request_sentences(body: str) -> list[str]:
    """The request sentences, and the list items a request that ends in a colon goes on to name.

    "Could you please provide me with: - your first name - your last name - your zip code" asks for
    three fields, and none of the three lines is a question on its own. Build 8's Simulated user
    read the ask and not the list, and answered "I do not have my name" on 63 re-rolls.
    """
    out: list[str] = []
    listing = False
    for sentence in _sentences(body):
        if _is_request(sentence):
            out.append(sentence)
            listing = sentence.endswith(":")
        elif listing and LIST_ITEM.match(sentence):
            out.append(sentence)
        else:
            listing = False
    return out


def asked_fields(text: Optional[str], vocab: Vocabulary = GENERIC) -> list[str]:
    """The fields an agent turn asks for: request sentences only, in vocabulary order."""
    requests = _request_sentences(_norm(text))
    if not requests:
        return []
    return [spec.field for spec in vocab.fields
            if any(re.search(cue, sentence) for sentence in requests for cue in spec.cues)]


def named_fields(text: Optional[str], fields: Iterable[str], vocab: Vocabulary = GENERIC) -> list[str]:
    """The fields of `fields` a request names by their own words, for wording no cue carries.

    The cues are the words this corpus and the web showed (D115); an agent that asks in words
    neither showed still names the field itself ("the name on the account"), and a fact the
    recording holds is answered whatever the wording (D44). Build 8's Simulated user answered
    "I do not have my name" to 63 such asks, with the name in its own rules.
    """
    requests = _request_sentences(_norm(text))
    if not requests:
        return []
    out: list[str] = []
    for field in fields:
        spec = vocab.get(field)
        words = [_words(field)] + list(spec.aliases if spec is not None else [])
        cues = [r"\b" + re.escape(word) + r"\b" + REPOINT for word in words if word]
        if field not in out and any(re.search(cue, sentence) for sentence in requests for cue in cues):
            out.append(field)
    return out


def _flattened(text: Optional[str]) -> str:
    """A turn as the eye reads it: emphasis marks dropped, runs of whitespace one space, lower case,
    so "**Chen Johnson**" and a name broken over two lines both read as "chen johnson"."""
    plain = EMPHASIS.sub("", (text or "").replace("’", "'"))
    return re.sub(r"\s+", " ", plain).strip().lower()


def _said_in(text: Optional[str], value: str) -> bool:
    """This turn said this value: whole word, however it was emphasised or wrapped, and with or
    without the leading mark an id carries, since users drop the '#' their orders are stored with."""
    body = _flattened(text)
    for form in (value, _bare_value(value)):
        needle = _flattened(form)
        if needle and re.search(r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])", body):
            return True
    return False


def _field_for(arg: str, vocab: Vocabulary) -> str:
    """The field an argument name states, and where the vocabulary knows no such argument, the
    argument's own name, which is the name `vocabulary.derive` would have given it (D115)."""
    return vocab.field_for(arg) or arg


def _name_field(vocab: Vocabulary) -> str:
    """The field an ask for a name reaches. The Simulated user looks up the field `asked_fields`
    returns, so a mined name filed under anything else is a name it will never find."""
    fields = asked_fields(NAME_ASK, vocab=vocab)
    return fields[0] if fields else "name"


def _spoken_args(call: Any) -> list[tuple[str, str]]:
    """The (argument, value) pairs of one call a user could have said out loud: strings and whole
    numbers, and the strings inside a list argument, since a basket of item ids is a list of facts."""
    out: list[tuple[str, str]] = []
    for arg, value in (call.args or {}).items():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                continue
            text = str(item).strip()
            if text:
                out.append((arg, text))
    return out


def _name_rows(parts: dict[str, tuple[str, Any]], vocab: Vocabulary) -> list[tuple[str, str, Any]]:
    """A first and a last name said in one user turn, as the one name fact an agent's ask reaches.

    Two facts of one field are told apart by the sentence they were said in, and the identity a
    world row is matched on takes one value per field, so "Chen" and "Johnson" filed separately
    would authenticate as neither. Parts said in different turns stay the facts they are.
    """
    first, last = parts.get("first_name"), parts.get("last_name")
    if first and last and first[1].idx == last[1].idx:
        return [(_name_field(vocab), f"{first[0]} {last[0]}", first[1])]
    return [(_field_for(arg, vocab), value, turn) for arg, (value, turn) in parts.items()]


def _argument_facts(trace: Trace, vocab: Vocabulary) -> list[tuple[str, str, Any]]:
    """Every (field, value, user turn) the recorded calls show the user gave, in call order.

    The cues will never cover every phrasing, and the trace already holds the ground truth: a
    scalar argument of a call that returned no error, said by the user in a turn at or before that
    call, is a fact that user gave, whatever words it was said in. A value only the tools knew (the
    user id a lookup returned, the payment method the agent read off the account) never appears in a
    user turn, so it is never mined (D77: nothing invented); a value a user turn holds only after
    the call may be the user repeating what the agent just told it, so later turns do not count.
    Build 12: 108 of 456 traces gave a first and last name the recorded agent authenticated with,
    in wording no cue carried ("my name is **Chen Johnson**"), no name fact was mined, and 149 of
    600 re-rolls ended without authentication and made no write.
    """
    user_turns = [turn for turn in trace.turns if turn.role == "user"]
    found: list[tuple[str, str, Any]] = []
    for call in trace.tool_calls:
        if call.error is not None:
            continue  # the call the world refused carries whatever the agent guessed, not a fact
        # A call the trace ties to no turn is ordered by nothing, so every user turn can hold it.
        cut = next((turn.idx for turn in trace.turns
                    if call.id and call.id in (turn.tool_call_ids or [])), None)
        before = [turn for turn in user_turns if cut is None or turn.idx <= cut]
        rows: list[tuple[str, str, Any]] = []
        parts: dict[str, tuple[str, Any]] = {}
        for arg, value in _spoken_args(call):
            turn = next((turn for turn in before if _said_in(turn.content, value)), None)
            if turn is None:
                continue
            if arg in NAME_PARTS:
                parts.setdefault(arg, (value, turn))
                continue
            rows.append((_field_for(arg, vocab), value, turn))
        found += _name_rows(parts, vocab) + rows
    return found


def derive_user_rules(trace: Trace, vocab: Vocabulary = GENERIC,
                      writes: Iterable[str] = ()) -> UserRules:
    """Facts, disclosure, refusals and walk-away for one Run, read off the trace's user turns (D44).

    `writes` names the tools that change the world (`ToolSig.kind`). A write the recorded Run made
    after it asked the user to confirm is the evidence that user agreed, whatever it said elsewhere.

    The user turns are read twice: once for the values the vocabulary's patterns state, then once
    for the values the recorded agent's own successful calls carry (`_argument_facts`). The second
    pass only ever appends, so the facts the patterns found keep their order and their spans.
    """
    rules = UserRules(style_sample=[trace.trace_id])
    seen_values: set[tuple[str, str]] = set()
    disclosed: set[str] = set()
    asked_anywhere: list[str] = []
    answered: set[str] = set()
    pending: list[str] = []
    asks: dict[int, list[str]] = {}
    saw_user_turn = False
    agent_refused = False
    confirm_asked = False
    confirmed_at: Optional[int] = None
    for turn in trace.turns:
        if turn.role == "assistant":
            stated = {field for field, _ in extracted_values(turn.content, vocab=vocab)}
            pending = [field for field in asked_fields(turn.content, vocab=vocab) if field not in stated]
            asked_anywhere += [field for field in pending if field not in asked_anywhere]
            agent_refused = bool(AGENT_REFUSAL.search(turn.content or ""))
            confirm_asked = bool(CONFIRM_REQUEST.search(turn.content or ""))
            continue
        if turn.role != "user":
            continue
        text = (turn.content or "").strip()
        said = _norm(text)
        values = extracted_values(text, asked=pending, vocab=vocab)
        if confirm_asked and confirmed_at is None and text and not NEGATIVE_CUE.search(said):
            confirmed_at = turn.idx
        for field, value in values:
            if (field, value) in seen_values:
                continue
            seen_values.add((field, value))
            rules.facts.append(UserFact(field=field, value=value, span=turn.raw_ptr,
                                        context=_stated_in(text, value)))
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
        answered_ask = any(field in pending for field, _ in values)
        _spoken_fact(rules, turn, text, said, values, refusing,
                     first=not saw_user_turn and not pending, answered_ask=answered_ask,
                     confirming=confirm_asked)
        if refusing:
            for field in pending:
                if field not in rules.refusals and field not in answered:
                    rules.refusals.append(field)
        if any(cue in said for cue in WALK_AWAY_CUES) or (agent_refused and CLOSING_CUE.search(said)):
            rules.walk_away.append(text)
        saw_user_turn = True
        asks[turn.idx] = pending
        pending = []
        confirm_asked = False
    _add_argument_facts(rules, trace, vocab, asks, disclosed, answered)
    rules.confirmed_by_write = _wrote_after(trace, confirmed_at, writes)
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


def _add_argument_facts(rules: UserRules, trace: Trace, vocab: Vocabulary,
                        asks: dict[int, list[str]], disclosed: set[str], answered: set[str]) -> None:
    """Append the facts the recorded calls show, with the disclosure the pattern pass would give.

    A fact the patterns already hold is left alone, so this never doubles a value or moves one; a
    field first given here is disclosed on request when the agent turn before that user turn asked
    for it, and volunteered otherwise, and counts as answered so it is neither a refusal nor a
    reason the rules are incomplete.
    """
    held = {(fact.field, _flattened(str(fact.value))) for fact in rules.facts}
    for field, value, turn in _argument_facts(trace, vocab):
        key = (field, _flattened(str(value)))
        if key in held:
            continue
        held.add(key)
        text = (turn.content or "").strip()
        rules.facts.append(UserFact(field=field, value=value, span=turn.raw_ptr,
                                    context=_stated_in(text, value)))
        answered.add(field)
        if field in disclosed:
            continue
        disclosed.add(field)
        on_request = field in asks.get(turn.idx, ())
        rules.disclosure.append(DisclosureRule(field=field, on_request=on_request,
                                               condition=None if on_request else "volunteered"))


def _spoken_fact(rules: UserRules, turn: Any, text: str, said: str, values: list,
                 refusing: bool, first: bool, answered_ask: bool = False,
                 confirming: bool = False) -> None:
    """The goal, a confirmation, a choice or the closing line, kept as the recorded sentence.

    A yes that goes on to name the order or the item is still the yes the write needed; only a yes
    that answers a field the agent asked for ("yes, my zip is 19122") is the field's answer alone.
    An answer to a question that asked the user to confirm is that question's confirmation whatever
    words it opens with, so a re-roll asking the same question gets the sentence it was given.
    """
    if not text:
        return
    if CLOSING_CUE.search(said):
        field = CLOSING
    elif first:
        field = GOAL
    elif (confirming or AFFIRM_CUE.search(said)) and not (answered_ask or refusing):
        field = CONFIRMATION
    elif values or refusing:
        return  # this turn answered a field ask, or refused it; both are recorded already
    else:
        field = CHOICE
    if field in (GOAL, CLOSING) and any(fact.field == field for fact in rules.facts):
        return
    rules.facts.append(UserFact(field=field, value=text, span=turn.raw_ptr, context=text))


def _stated_in(text: str, value: str) -> str:
    """The recorded sentence a value was said in, which is how two values of one field are told
    apart; the whole turn when no one sentence holds it."""
    for sentence in _sentences(text):
        if str(value) in sentence or _bare_value(str(value)) in sentence:
            return sentence
    return text


def _bare_value(value: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+", "", value)


def _wrote_after(trace: Trace, confirmed_at: Optional[int], writes: Iterable[str]) -> bool:
    """The recorded Run changed the world after the user answered a question that asked it to
    confirm. That write is the evidence the recorded user agreed (D44)."""
    names = set(writes)
    if confirmed_at is None or not names:
        return False
    ids = {call.id for call in trace.tool_calls if call.name in names and call.id}
    return any(turn.idx > confirmed_at and ids.intersection(turn.tool_call_ids or ())
               for turn in trace.turns)


def _closes(text: Optional[str]) -> bool:
    """This Candidate turn closed the conversation or passed it on, which is a handoff (D210)."""
    said = text or ""
    return bool(CLOSE_CUE.search(said) or HANDOFF_CUE.search(said))


def _names_change(text: Optional[str]) -> bool:
    """This sentence states a value the user is moving to rather than the one it holds today."""
    return bool(CHANGE_CUE.search(text or ""))


def _intent(question: Optional[str]) -> Optional[str]:
    """Which of two values of a kind a question names: the one being moved to, the one the account
    holds today, or neither, when it names both or says nothing either way."""
    said = _norm(question)
    change, stored = CHANGE_CUE.search(said), STORED_CUE.search(said)
    if change and not stored:
        return "change"
    if stored and not change:
        return "stored"
    return None


def _asks_stored(question: Optional[str]) -> bool:
    """This question asks for the value the account holds today, not the one being moved to."""
    return _intent(question) == "stored"


def _agrees(asked: str, context: Optional[str]) -> int:
    """+1 when the question and the sentence a value was stated in agree about which value it is,
    -1 when they disagree, 0 when the question names neither."""
    intent = _intent(asked)
    if intent is None:
        return 0
    changed = _names_change(context)
    return 1 if (intent == "change") == changed else -1


def _tokens(text: Optional[str]) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", _norm(text)) if word not in STOP_WORDS]


def _shared(asked: str, context: Optional[str], own: set) -> int:
    """Words the question and the recorded sentence share, the field's own words apart: what the
    question says about the value it wants ("the order with the lamp") picks between two of them."""
    return len((set(_tokens(asked)) & set(_tokens(context))) - own)


def fact_class(rules: UserRules, field: str, value: Any = None) -> str:
    """Which class an answer is in: askable when the recording holds this user saying it, record
    when only the world does (D210).

    The class is not a new thing to mine. D77 already says a fact is mined from a user turn or from
    an argument a user turn said, and nothing else; so the rules are exactly the askable facts, and
    a value the Starting state answers with that the rules do not hold is a record fact. Typing them
    apart is what keeps a value only the system knew out of the user's mouth structurally, rather
    than by a check run afterwards. The class is of the value and not of the field, because one
    field holds two values where a user is moving from one to the other: it said the one it is
    moving to, so that one is askable and the one on the account is the world's.
    """
    if value is None:
        return ASKABLE if any(fact.field == field for fact in rules.facts) else RECORD
    return ASKABLE if any(fact.field == field and str(fact.value) == str(value)
                          for fact in rules.facts) else RECORD


def goal_write_set(trace: Optional[Trace], writes: Iterable[str]) -> set[str]:
    """The writes the Task's goal implies: the write-kind tools the Reference called and the world
    did not refuse (D210).

    `writes` names the tools that change the world (`ToolSig.kind`), which is the only class read
    here; which tools those are is the build's answer, not this module's. A recording that changed
    nothing gives the empty set, and a goal that implies no write is satisfied as soon as the
    Candidate has nothing left to do, which is what the empty set says.
    """
    names = set(writes)
    return {call.name for call in (trace.tool_calls if trace is not None else ())
            if call.name in names and call.error is None}


def _words(field: str) -> str:
    return field.replace("_", " ")


# The two readings the end protocol takes off a Candidate turn, public because the agent user's own
# end protocol (kullback/user/guards.py) decides the same four kinds over the turns a model drove
# and must read them exactly as the rule-driven user does, not with a second pair of cues (D214).
closes = _closes
names_change = _names_change
