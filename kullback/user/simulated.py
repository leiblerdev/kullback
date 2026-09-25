"""The rule-driven Simulated user: it replies as the recorded user, from the rules and the world.

Moved out of kullback/user/rules.py unchanged, so deriving the rules and speaking them come apart.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from kullback.runner.records import Event, UserFact, UserRules
from kullback.user.ends import FactLookup, _field_of, _row_value, goal_done, tool_called, writes_made
from kullback.user.rules import (
    ASKABLE,
    CHOICE,
    CLOSING,
    CONFIRM_REQUEST,
    CONFIRMATION,
    GENERIC_CLOSE,
    GENERIC_CONFIRM,
    GOAL,
    GOAL_RESTATED,
    GOAL_SATISFIED,
    HANDED_OFF,
    IN_RECORD_TAG,
    NEGATIVE_CUE,
    OPEN_REQUEST,
    RECORD_LINE,
    RECORD_SOURCE,
    SCENARIO_EXHAUSTED,
    SILENCE_LIMIT,
    SPOKEN_FIELDS,
    STRIPPED_SOURCE,
    STRIPPED_TAG,
    UNANSWERABLE_LIMIT,
    _agrees,
    _asks_stored,
    _closes,
    _names_change,
    _norm,
    _request_sentences,
    _shared,
    _tokens,
    _words,
    asked_fields,
    extracted_values,
    fact_class,
    named_fields,
)
from kullback.user.vocabulary import GENERIC, Vocabulary


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
        self._acted = False  # the Candidate has used a tool in this Run (p3)
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
        self._acted = tool_called(transcript)
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
        satisfied = self._goal_done(made or set(), question)
        goal = self._fact(GOAL)
        # A goal naming no writes has nothing left to restate once the Candidate closes (p3).
        no_write_close = self.goal_writes is not None and not self.goal_writes and _closes(question)
        if goal is not None and not self._restated and not satisfied and not no_write_close:
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
        kind = self._end_kind(question, self._goal_done(made, question))
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

    def _goal_done(self, made: set, question: str = "") -> bool:
        """The Task's goal is done in this Run, through the one predicate both users read (D210).

        `made` is the write-kind tools this transcript shows made. A goal naming no writes is done
        only when the Candidate has used a tool and its latest turn closes; D158's reading holds
        where the caller named no goal writes at all.
        """
        return goal_done(self.goal_writes, self.write_tools, made,
                         acted=self._acted, closed=_closes(question))

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
