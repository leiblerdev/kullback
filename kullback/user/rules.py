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
from typing import Any, Callable, Iterable, NamedTuple, Optional

from kullback.runner.records import DisclosureRule, Event, Trace, UserFact, UserRules
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


def end_kind_of(payload: Any) -> Optional[str]:
    """The end kind one recorded user turn carries, in either shape the two users write it in.

    The rule-driven user tags the turn it ends on and the Runner copies the tags into the Run's own
    file; the agent user writes the kind under `user_end`. Both are the same fact, and a reader that
    knows only one of the two shapes silently reads every Run written in the other as a Run that
    ended in no kind at all. One function, so there is one answer.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("user_end") in USER_END_KINDS:
        return str(payload["user_end"])
    for tag in payload.get("tags") or ():
        if tag in USER_END_KINDS:
            return str(tag)
    return None


def end_of_run(run: Any) -> Optional[str]:
    """How this Run ended, in the four kinds (D210), or nothing where it ended in neither vocabulary.

    The Simulated user tags the turn it ends on, and the loop copies a turn's tags into the Run's
    own file, so the kind survives into the record without the Runner having to know about it. A
    Run the user never ended is read from the Runner's own end reason instead: the turn limit is
    `gave_up`, and a Candidate that stopped or was handed over is `handed_off`. A Run that crashed
    ended in neither vocabulary and is left unclassified rather than guessed at.
    """
    for event in reversed(list(getattr(run, "events", None) or [])):
        if getattr(event, "type", None) != "user_turn":
            continue
        kind = end_kind_of(getattr(event, "payload", None))
        if kind is not None:
            return kind
    reason = getattr(run, "termination_reason", None) or ""
    if reason in TURN_LIMIT_REASONS:
        return GAVE_UP
    return HANDED_OFF if reason in STOP_REASONS else None


def ends_by_kind(rows: Iterable[Any]) -> dict[str, int]:
    """How many Runs ended each way, over rows carrying a `user_end` (D210). Every kind is named,
    zero included, so a round that stopped ending one way says so rather than dropping the line."""
    counts = {kind: 0 for kind in USER_END_KINDS}
    for row in rows or ():
        kind = row.get("user_end") if isinstance(row, dict) else getattr(row, "user_end", None)
        if kind in counts:
            counts[kind] += 1
    return counts


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


def _calls_of(message: Any) -> list:
    """The tool calls one transcript message carries, whichever shape the caller builds it in."""
    calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    return list(calls or [])


def _attr(message: Any, key: str) -> Any:
    """One field of a transcript message, whichever shape it is in, with None kept as None.

    `_field_of` folds a missing field to the empty string because its callers want a name to
    compare. A marker's absence and a marker holding nothing are different things here, so this
    keeps them apart.
    """
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def write_took_effect(message: Any) -> bool:
    """Whether one write call's result shows the write took effect (D227).

    The Runner records a tool result as `result` plus, where the world refused or failed the call,
    `error`: the D67 ToolCallError carrying that refusal's `class` in the recorded taxonomy. That
    marker is the whole of the refusal reading, because it is the one field the Runner writes on
    every route it can refuse on, whatever payload the tool answered with. A call carrying it
    changed nothing, which is the reading the Verifier suite's own write_effects has always taken.

    Where the call also carries D215's write effect record, the record decides: the effect has to
    show a column of the world moved. A record naming no moved column is a write that returned
    cleanly and left the world where it was. A call carrying no record at all is answered on the
    error marker alone, so a Runner on a route that records no effect gets the reading its own
    records support rather than a goal refused for want of evidence.
    """
    if _attr(message, "error") is not None:
        return False
    effect = _attr(message, "write_effect")
    if effect is None:
        return True
    return _effect_moved_a_column(effect)


def _effect_moved_a_column(effect: Any) -> bool:
    """Whether one D215 write effect record shows a column of the world moved.

    D215 writes two records per write, and both are read here because either may be the one riding
    with the call. The evidence the Builder mined off the recording is a list of rows, one per
    column, each naming the table, the row, the column path and what that column held before and
    after the write (`builder/effects.replay_evidence`); such a column moved when its after value
    differs from its before value. The replay's own record of running those checks counts the
    columns it checked and names the ones that never reached their after value
    (`runner/replay.ScoredRouter._check_effects`); a write moved a column there when it was checked
    on at least one and failed none of them, since a failure is a column the write left where it was.

    Neither record is read for its column names, only for whether any column moved, so a record
    written in either shape answers the same question and a shape neither knows falls back to
    whether the record holds anything at all.
    """
    if isinstance(effect, dict) and _is_replay_check(effect):
        checked = _count(effect.get("effect_checks"))
        failed = _count(effect.get("effect_failures_total")) or len(effect.get("effect_failures") or ())
        return checked > 0 and failed == 0
    rows = [effect] if isinstance(effect, dict) else list(effect or ())
    return any(_column_moved(row) for row in rows)


def _is_replay_check(effect: dict) -> bool:
    """Whether this record is the replay's own count of the effect checks it ran, not one column."""
    return any(key in effect for key in ("effect_checks", "effect_failures", "effect_failures_total"))


def _column_moved(row: Any) -> bool:
    """Whether one mined effect row shows its column holding something else after the write."""
    if not isinstance(row, dict):
        return bool(row)
    if "before" not in row and "after" not in row:
        return bool(row)
    return row.get("before") != row.get("after")


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def writes_made(transcript: Any, write_tools: Iterable[str]) -> set[str]:
    """The write-kind tools this Run has made, counting only the calls whose result took effect (D227).

    One function, read by both Simulated users, so the rule-driven user's `_goal_done` and the agent
    user's `end_run` cannot disagree about whether the goal is done. Before D227 this counted the
    write-kind tools the transcript showed called, so a write the world refused still ended the
    conversation `goal_satisfied` while the same Run was graded as claiming a write it never made.

    A tool result message names the call it answered, so it is the message an effect is read from.
    An assistant message's `tool_calls` are requests and not results, and a request whose result
    never came back is not a write that happened, so a name seen only there no longer counts.
    """
    names = frozenset(write_tools or ())
    made: set[str] = set()
    if not names:
        return made
    for message in transcript or ():
        if _field_of(message, "name") in names and write_took_effect(message):
            made.add(_field_of(message, "name"))
    return made


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
                 identity: Optional[dict] = None, vocab: Vocabulary = GENERIC,
                 write_tools: Iterable[str] = (), goal_writes: Optional[Iterable[str]] = None,
                 answer_strip: Optional[Callable[[str], tuple[str, list]]] = None):
        self.rules = rules
        self.reader = starting_state_reader
        self.model = model
        self.vocab = vocab
        # The tools that change the world (`ToolSig.kind`), so this user can tell a Run that has
        # done what it came for from one that has not. The transcript `reply` is handed already
        # carries the calls and their results, so nothing new has to be plumbed to the user; the
        # vocabulary knows fields and not tool kinds, which is why the names are passed in here.
        self.write_tools = frozenset(write_tools or ())
        # The writes the Task's goal implies (D210, `goal_write_set`). None means the caller named
        # none, and the end then falls back to D158's reading, that any write is the Run acting; an
        # empty set is a goal that implies no write and is satisfied without one.
        self.goal_writes = None if goal_writes is None else frozenset(goal_writes)
        # D196's strip, prepared over this Task's own evidence and applied to what this user is
        # about to say. Without one the user speaks its facts unchecked, as it did before D210.
        self.answer_strip = answer_strip
        self.events: list[Event] = []
        self.done = False
        # Which of the four kinds ended this Run, set on the turn that ends it (D210).
        self.end_reason: Optional[str] = None
        self.stripped = 0
        identity_fields = vocab.by_kind("identity")
        # A value the recorded user stated as the one it is moving to is not the value the world
        # keys its row by, so it never joins the identity a row is matched on.
        self.identity = dict(identity) if identity else {
            fact.field: fact.value for fact in rules.facts
            if fact.field in identity_fields and not _names_change(fact.context)}
        self._row: Any = None
        self._row_read = False
        self._used: dict[str, int] = {CONFIRMATION: 0, CHOICE: 0}
        self._silent = 0
        self._refused = 0
        self._restated = False
        # Turns this user had nothing at all for what was asked on: the scenario running out,
        # counted (D210). One turn, however many fields it named: the rule is a Candidate asking
        # twice, so a single turn naming two unknown fields is one ask and not two.
        self._unanswerable = 0
        self._end_tagged = False

    def reply(self, transcript: list) -> str:
        question = ""
        for message in transcript:
            if _field_of(message, "role") == "assistant":
                question = _field_of(message, "content")
        answers: dict[str, Any] = {}
        sources: dict[str, str] = {}
        spoken: list[str] = []
        unavailable: list[str] = []
        record: list[str] = []
        assisted = False
        for field in self._asked(question):
            fact = self._fact(field, question)
            if fact is not None and _asks_stored(question) and _names_change(fact.context):
                found = self._from_world(field)  # the question asks for the value on the account
                if found is not None:
                    self._speak(found.value, field, answers, sources, record)
                    assisted = assisted or found.synthetic
                    continue
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
            self._speak(found.value, field, answers, sources, record)
            assisted = assisted or found.synthetic
        self._strip_answers(answers, sources, unavailable)
        if not self.events:
            self._open(answers, sources, spoken)
            self._strip_answers(answers, sources, unavailable)
        said_something = bool(answers or unavailable or record or spoken)
        if not said_something:
            self._respond(question, sources, spoken, unavailable, self._writes_made(transcript))
            said_something = bool(spoken or unavailable)
        self._silent = 0 if said_something else self._silent + 1
        self._unanswerable += int(bool(unavailable))
        # D210, Greptile P1 (PR 25): a turn whose only content is that this user has no record of
        # what was asked is not the user having something to say, it is the exhaustion signal
        # itself. Without this the unanswerable limit is reachable only on the turns where nothing
        # was asked by name, so a Candidate that keeps asking for fields nobody ever told this user
        # about runs to the turn limit and is reported as gave_up instead of scenario_exhausted.
        if not self.done and unavailable and not (answers or record or spoken):
            self._close(question, sources, spoken, self._writes_made(transcript))
        text = self._say(question, answers, sources, spoken, unavailable, record)
        # How many of this turn's asks went unanswered, and how many the Run has left unanswered so
        # far: the refusal rate a build reports, read off the Simulated user's own turns. A record
        # fact counts here too: the Candidate asked and got no value, whoever holds it.
        refused = sum(1 for source in sources.values()
                      if source in ("refused", "unavailable", RECORD_SOURCE, STRIPPED_SOURCE))
        self._refused += refused
        tags = (["fact_unavailable"] if unavailable else []) + ([IN_RECORD_TAG] if record else [])
        if any(source == STRIPPED_SOURCE for source in sources.values()):
            tags.append(STRIPPED_TAG)
        if self.end_reason is not None and not self._end_tagged:
            tags.append(self.end_reason)
            self._end_tagged = True
        self.events.append(Event(
            idx=len(self.events),
            type="user_turn",
            payload={
                "text": text,
                "fields": list(sources),
                "sources": sources,
                "unavailable_fields": unavailable,
                "record_fields": record,
                "tags": tags,
                "refused": refused,
                "refused_so_far": self._refused,
                "user_end": self.end_reason,
                "stripped_so_far": self.stripped,
            },
            assisted=assisted,
        ))
        return text

    def _asked(self, question: str) -> list[str]:
        """The fields this question asks for: the vocabulary's cues, plus the facts the recording
        holds that the question names in words no cue carries.

        A field the question states itself is not an ask for it. An agent that lists the action it
        is about to take and asks "do you confirm" names the order it is confirming, and answering
        with that order id instead of confirming is what build 8's Simulated user did.
        """
        stated = {field for field, _ in extracted_values(question, vocab=self.vocab)}
        fields = [field for field in asked_fields(question, vocab=self.vocab) if field not in stated]
        if CONFIRM_REQUEST.search(question or ""):
            return fields
        held: list[str] = []
        for fact in self.rules.facts:
            if fact.field not in SPOKEN_FIELDS and fact.field not in stated and fact.field not in held:
                held.append(fact.field)
        return fields + [field for field in named_fields(question, held, vocab=self.vocab)
                         if field not in fields]

    def _open(self, answers: dict, sources: dict, spoken: list) -> None:
        """The opening reply: the goal the recorded user stated, then what it volunteered (D44)."""
        goal = self._fact(GOAL)
        if goal is not None:
            spoken.append(str(goal.value))
            sources[GOAL] = "rules"
        self._volunteer(answers, sources)

    def _respond(self, question: str, sources: dict, spoken: list, unavailable: list,
                 made: Optional[set] = None) -> None:
        """Nothing was asked by name: a confirmation, a stated choice, the goal again, or the close.

        A real user whose request has not been acted on says it again before it leaves. Build 12's
        Simulated user left instead: of 381 re-rolls that made no write, 261 ended on the user's
        closing line, 248 of those after a turn where nothing was asked of it and it had nothing to
        say, and the dead turn came second of all its turns in 140 of them. So the first dead turn,
        and a "was there anything else" asked while the Run has not written yet, get the goal the
        recording opened with, once (D44: the recorded sentence, D77: nothing invented). The second
        such turn closes, which is why the close is gated on `self._restated` and not on `_silent`
        alone: the restatement speaks, so it resets the silence counter the turn it happens.

        D210: the close is a protocol and not a cue. `_end_kind` says which of the four kinds this
        turn is, and a turn that is none of them does not end the Run, so a Candidate that says
        goodbye over a goal whose writes are not done is answered rather than agreed with.
        """
        for field, reuse_last, cue in ((CONFIRMATION, True, CONFIRM_REQUEST),
                                       (CHOICE, False, OPEN_REQUEST)):
            if not cue.search(question or ""):
                continue
            fact = self._next(field, reuse_last=reuse_last)
            if fact is not None:
                spoken.append(str(fact.value))
                sources[field] = "rules"
            elif field == CONFIRMATION and not self._declined():
                spoken.append(GENERIC_CONFIRM)  # D44: the representative answer to an ask never recorded
                sources[field] = "generic_confirm"
            else:  # the trace holds no answer to this request, and nothing is invented (D77)
                unavailable.append(field)
                sources[field] = "unavailable"
            return
        satisfied = self._goal_done(made or set())
        goal = self._fact(GOAL)
        if goal is not None and not self._restated and not satisfied:
            spoken.append(str(goal.value))
            sources[GOAL] = GOAL_RESTATED
            self._restated = True
            return
        self._close(question, sources, spoken, made or set())

    def _close(self, question: str, sources: dict, spoken: list, made: set) -> None:
        """End the Run where this turn is one of the four kinds, and say nothing where it is not (D210).

        The one place a Run ends, so the protocol reads the same whether the turn asked for fields
        this user has no record of or asked nothing by name at all.
        """
        kind = self._end_kind(question, self._goal_done(made))
        if kind is None:
            return
        closing = self._fact(CLOSING)
        spoken.append(str(closing.value) if closing is not None else GENERIC_CLOSE)
        sources[CLOSING] = "rules" if closing is not None else "generic_close"
        self.end_reason = kind
        self.done = True

    def _end_kind(self, question: str, satisfied: bool) -> Optional[str]:
        """Which of the four kinds this end is, or nothing where the user has not ended (D210).

        More than one can hold at once, so they are read in one order. A Run whose goal writes are
        all confirmed is done whatever the Candidate said next. A Candidate that twice asks for
        what nobody ever told this user has run the scenario out, whatever it says while doing it.
        A Candidate that then closes or passes the conversation on ended it, and that is a handoff
        rather than the user running dry. Last comes the user with nothing left to say and no close
        to answer, which is the scenario out in the other way. The closing cue and the silence
        counter are still read, but each is an input to a kind and neither is the end on its own.
        `gave_up` is never reached here: it is the turn limit, which the loop holds and the user
        never sees.
        """
        if satisfied:
            return GOAL_SATISFIED
        if self._unanswerable >= UNANSWERABLE_LIMIT:
            return SCENARIO_EXHAUSTED
        if _closes(question):
            return HANDED_OFF
        return SCENARIO_EXHAUSTED if (self._restated or self._silent >= SILENCE_LIMIT) else None

    def _goal_done(self, made: set) -> bool:
        """Every write the Task's goal implies has been made in this Run (D210).

        `made` is the write-kind tools this transcript shows called. A caller that named the goal's
        own writes is answered against them; one that named none falls back to D158's reading, that
        a Run which wrote at all has acted, so a caller passing nothing gets the user it had.
        """
        if self.goal_writes is not None:
            return self.goal_writes <= made
        return bool(self.write_tools) and bool(made)

    def _writes_made(self, transcript: list) -> set[str]:
        """The write-kind tools this Run has made so far, so the user can tell a Run that has done
        what it came for from one that has not.

        Read off the transcript `reply` is handed, through the one shared `writes_made` the agent
        user reads too (D227), so a write the world refused leaves the goal open for both. No other
        state is kept, so a caller that passes no `write_tools` gets a user that restates its goal
        once whatever the Run has done.
        """
        return writes_made(transcript, self.write_tools)

    def _declined(self) -> bool:
        """The recording holds a no where a yes was asked for; then no yes is representative.

        A recorded Run that went on to the write said yes with the write, whatever a later line
        said no to: build 8 read one such no and refused every confirmation for the rest of the Run.
        """
        if self.rules.confirmed_by_write:
            return False
        return any(fact.field in (CONFIRMATION, CHOICE) and NEGATIVE_CUE.search(_norm(str(fact.value)))
                   for fact in self.rules.facts)

    def _fact(self, field: str, question: str = "") -> Optional[UserFact]:
        """The recorded value for this field, and where the recording holds more than one, the one
        the agent's question names (D44)."""
        facts = [fact for fact in self.rules.facts if fact.field == field]
        if len(facts) <= 1:
            return facts[0] if facts else None
        asked = " ".join(_request_sentences(_norm(question))) or _norm(question)
        own = set(_tokens(_words(field)))
        best, score = facts[0], None
        for order, fact in enumerate(facts):
            here = (_agrees(asked, fact.context), _shared(asked, fact.context, own), -order)
            if score is None or here > score:
                best, score = fact, here
        return best

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

    def _speak(self, value: Any, field: str, answers: dict, sources: dict, record: list) -> None:
        """A value the world holds, said or pointed at by its class (D210).

        Askable is a value this user itself gave, so it is said. Record exists only in the world,
        and the Candidate has the tools that read it: the user says so and names no value, which is
        the leak D79 check 7 fails a Task for, closed where the answer is chosen rather than after.
        """
        if fact_class(self.rules, field, value) == ASKABLE:
            answers[field], sources[field] = value, "world"
            return
        record.append(field)
        sources[field] = RECORD_SOURCE

    def _strip_answers(self, answers: dict, sources: dict, unavailable: list) -> None:
        """D196's strip, run again on what this user is about to say (D210).

        A value the strip takes out is one the system held and no user of this Task ever said, so it
        is not this user's to give however it reached the rules; the fact is answered as absent and
        the removal counted, which is what stops the leak check being passed by the user handing the
        value over. The check is per value and not per line: a line is many facts, and only the fact
        whose value the strip would take is lost.
        """
        if self.answer_strip is None:
            return
        for field in [field for field, value in answers.items() if self._removed(str(value))]:
            answers.pop(field)
            sources[field] = STRIPPED_SOURCE
            unavailable.append(field)
            self.stripped += 1

    def _removed(self, text: str) -> bool:
        """The strip would take something out of this text, so no user of this Task ever said it."""
        if self.answer_strip is None or not text:
            return False
        _, taken = self.answer_strip(text)
        return bool(taken)

    def _say(self, question: str, answers: dict, sources: dict, spoken: list, unavailable: list,
             record: Iterable[str] = ()) -> str:
        plain = list(spoken)
        plain += [f"My {_words(field)} is {value}." for field, value in answers.items()]
        plain += [f"I would rather not share my {_words(f)}."
                  for f, source in sources.items() if source == "refused"]
        plain += [RECORD_LINE.format(_words(field)) for field in record]
        plain += ["I do not have an answer for that." if field in SPOKEN_FIELDS
                  else f"I do not have my {_words(field)}." for field in unavailable]
        sentence = " ".join(plain) or "Okay, thank you."
        if self.model is None or unavailable or spoken or list(record) or not answers:
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
        if self._removed(text):
            return sentence  # D210: the wording carried a value no user of this Task ever said
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


# The two readings the end protocol takes off a Candidate turn, public because the agent user's own
# end protocol (kullback/user/guards.py) decides the same four kinds over the turns a model drove
# and must read them exactly as the rule-driven user does, not with a second pair of cues (D214).
closes = _closes
names_change = _names_change
