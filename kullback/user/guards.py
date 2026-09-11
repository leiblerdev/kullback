"""The guards that run in code after the model, on every turn, never waived (D214 rule 4).

A language model asked to play someone's customer will invent a value when it is short of one and
will hand over a value it was never given when the other side asks nicely. Neither is a prompting
problem, so neither is answered in the prompt: the model writes a turn, code reads it, and a turn
that fails a guard never reaches the Candidate. What reaches it instead is the rule-driven user's
turn for the same beat (agent.py), which is why the floor is never removed.

Five guards, in the order they run. A value the world holds and no user of this Task ever said is
replaced by the sentence that points at it (D210). The D196 strip runs on what is left. Any
value-shaped token that is still not one of this user's own words is an invention and the turn is
dropped. Volunteering past the point the recorded user had reached is cut, counted by facts and not
by sentences, so a Candidate cannot farm facts out of this user by asking many questions. And the
end is code's: the model may request a kind and `EndProtocol` decides whether one holds.

Every drop carries a reason, and the reasons are counted per round, because a guard that fires on
every turn is a context that is wrong and not a model that is bad.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional, Sequence

from kullback.runner.records import Record
from kullback.user import rules as rules_mod

# The reasons a turn is dropped or changed. They are the counters a round reports.
INVENTED = "agent_user_invented"
RECORD_FACT = "agent_user_record_fact"
STRIPPED = "agent_user_stripped"
OVER_DISCLOSED = "agent_user_over_disclosed"
SAID_NOTHING = "agent_user_said_nothing"
DROP_REASONS = (INVENTED, RECORD_FACT, STRIPPED, OVER_DISCLOSED, SAID_NOTHING)

# What counts as a value-shaped token: the things a customer gets wrong by inventing them. Anything
# carrying a digit (an amount, a count, a date, an id with a number in it), a run of capitals or
# digits long enough to be a code, and an address. Ordinary words are not values, because a user who
# says the wrong ordinary word has said something odd and not something false, and the fidelity
# score is what reads that.
VALUE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@#/-]*")
CODE_TOKEN = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,}$")
# Numbers small enough that any sentence carries them: "two of them", "the 3rd option". A guard that
# drops a turn for saying "one" drops every turn.
SMALL_NUMBER = 10


def value_tokens(text: Optional[str]) -> list[str]:
    """Every value-shaped token of a turn, in the order it says them."""
    out: list[str] = []
    for match in VALUE_TOKEN.finditer(text or ""):
        token = match.group(0).strip("._-#/")
        if not token or token in out:
            continue
        if token.isdigit() and int(token) < SMALL_NUMBER:
            continue  # "two of them", "the 3rd option": every sentence carries these
        if any(ch.isdigit() for ch in token):
            out.append(token)
        elif CODE_TOKEN.match(token):
            out.append(token)
    return out


def canonical(token: str) -> str:
    """One value as two spellings of it compare: case, spacing and punctuation dropped.

    A user that says its postal code without the hyphen and one that says it with have said the same
    value, and a guard that reads them as two values drops every second turn.
    """
    return re.sub(r"[^a-z0-9]", "", (token or "").lower())


class Grounding:
    """What this user is known to have said, as the guard compares against it.

    The askable facts' values, and the recorded sentences they were said in: a token inside a
    sentence the recorded user actually spoke is that user's own word, whatever the fact list calls
    it. Nothing else is in here, so nothing else may be spoken.
    """

    def __init__(self, values: Iterable[Any] = (), sentences: Iterable[str] = (),
                 by_field: Optional[dict[str, list[str]]] = None):
        self.values = [canonical(str(v)) for v in values if str(v).strip()]
        # Which field each value answers, so a turn that answers the question it was asked is told
        # apart from a turn that volunteers something nobody asked for (D214 rule 4).
        self.by_field: dict[str, list[str]] = {field: [canonical(str(v)) for v in vals]
                                               for field, vals in (by_field or {}).items()}
        self.tokens: set[str] = {c for value in self.values if (c := value)}
        for sentence in sentences:
            self.tokens.update(canonical(token) for token in value_tokens(sentence))
        self.tokens.discard("")

    def answering(self, fields: Iterable[str]) -> set[str]:
        """The canonical values that answer these fields: what a turn may say for free."""
        wanted = {str(field) for field in fields or ()}
        return {value for field, values in self.by_field.items() if field in wanted
                for value in values if value}

    def allows(self, token: str) -> bool:
        """This token is one of the user's own values, or a part of one, or holds one whole.

        Both directions, because a fact stored as one string is often said in pieces (a full address
        said street first) and a fact stored in pieces is often said as one.
        """
        canon = canonical(token)
        if not canon or canon in self.tokens:
            return True
        return any(canon in value or value in canon for value in self.values if value)


class GuardOutcome(Record):
    """What the guards made of one turn: the text to speak, or nothing and the reason why."""

    text: str = ""
    dropped: bool = False
    reason: Optional[str] = None
    changed: list[str] = []   # the guards that rewrote the turn without dropping it


def _sentences(text: str) -> list[str]:
    parts = rules_mod._sentences(text or "")
    return parts or ([text] if text else [])


class Guards:
    """The five guards over one Task's turns.

    `record_values` are the values only the world holds, so a turn that speaks one is answered with
    the sentence that points at it instead. `strip` is D196's, prepared over this Task's own
    evidence. `grounding` is what this user may say.
    """

    def __init__(self, grounding: Grounding, record_values: Optional[dict[str, Any]] = None,
                 strip: Optional[Callable[[str], tuple[str, list]]] = None):
        self.grounding = grounding
        self.record_values = dict(record_values or {})
        self.strip = strip
        self.counts: dict[str, int] = {reason: 0 for reason in DROP_REASONS}

    def check(self, text: str, *, facts_allowed: int = -1, facts_said: int = 0,
              asked: Iterable[str] = ()) -> GuardOutcome:
        """One model turn through every guard, in order.

        `facts_allowed` is how many facts the recorded user had volunteered by this point and
        `facts_said` how many this Run's user has volunteered already; a negative allowance is no
        limit, which is the Task with no recording to measure against. `asked` are the fields the
        question asked for: answering the question is always allowed, and only what a turn adds
        beyond the answer is a disclosure counted against the allowance.
        """
        said = (text or "").strip()
        if not said:
            return self._drop(SAID_NOTHING)
        changed: list[str] = []
        said = self._replace_record_facts(said, changed)
        said = self._strip_values(said, changed)
        if not said.strip():
            return self._drop(STRIPPED if STRIPPED in changed else RECORD_FACT)
        invented = [t for t in value_tokens(said) if not self.grounding.allows(t)]
        if invented:
            return self._drop(INVENTED)
        said = self._cut_surplus(said, facts_allowed, facts_said, changed,
                                 free=self.grounding.answering(asked))
        if not said.strip():
            return self._drop(OVER_DISCLOSED)
        return GuardOutcome(text=said, dropped=False, changed=changed)

    def _drop(self, reason: str) -> GuardOutcome:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return GuardOutcome(text="", dropped=True, reason=reason)

    def _replace_record_facts(self, text: str, changed: list[str]) -> str:
        """A sentence naming a value only the world holds becomes the sentence that points at it."""
        if not self.record_values:
            return text
        out: list[str] = []
        hit = False
        for sentence in _sentences(text):
            field = next((f for f, value in self.record_values.items()
                          if value is not None and rules_mod._said_in(sentence, str(value))), None)
            if field is None:
                out.append(sentence)
                continue
            hit = True
            out.append(rules_mod.RECORD_LINE.format(rules_mod._words(field)))
        if hit:
            changed.append(RECORD_FACT)
            self.counts[RECORD_FACT] = self.counts.get(RECORD_FACT, 0) + 1
        return " ".join(out)

    def _strip_values(self, text: str, changed: list[str]) -> str:
        """D196's strip: a sentence it takes something out of is a sentence this user cannot say."""
        if self.strip is None:
            return text
        kept = []
        for sentence in _sentences(text):
            _, taken = self.strip(sentence)
            if taken:
                changed.append(STRIPPED)
                self.counts[STRIPPED] = self.counts.get(STRIPPED, 0) + 1
                continue
            kept.append(sentence)
        return " ".join(kept)

    def _cut_surplus(self, text: str, allowed: int, said: int, changed: list[str],
                     free: Iterable[str] = ()) -> str:
        """Sentences past the point the recorded user had volunteered by are cut (D214 rule 4)."""
        if allowed < 0:
            return text
        budget = max(0, allowed - said)
        out: list[str] = []
        spent = 0
        cut = False
        free = set(free or ())
        for sentence in _sentences(text):
            carried = self.facts_in(sentence, free)
            if carried and spent + carried > budget:
                cut = True
                continue
            spent += carried
            out.append(sentence)
        if cut:
            changed.append(OVER_DISCLOSED)
            self.counts[OVER_DISCLOSED] = self.counts.get(OVER_DISCLOSED, 0) + 1
        return " ".join(out)

    def facts_in(self, text: str, free: Iterable[str] = ()) -> int:
        """How many of this user's own values a sentence carries, which is what a disclosure is counted in.

        `free` are the values that answer the question asked, which is not a disclosure.
        """
        canon = [canonical(token) for token in value_tokens(text)]
        skip = set(free or ())
        return sum(1 for value in self.grounding.values
                   if value and value not in skip
                   and any(value in token or token in value for token in canon))


class EndProtocol:
    """The four kinds (D210), decided by code over the conversation a driver actually produced.

    The rule-driven user decides its own end inside `reply`, off counters that advance with the turns
    it spoke. An agent user's turns are not those turns, so its counters are not those counters: a
    Candidate whose question the rules had nothing for and the agent answered has not run the
    scenario out, and a protocol reading the rules' counters would end it as if it had. The kinds,
    their order and the two limits are the rules' own, read from that module, so the two drivers end
    in one vocabulary and a build that compares them compares like with like.
    """

    def __init__(self, goal_writes: Optional[Iterable[str]] = None, write_tools: Iterable[str] = ()):
        self.goal_writes = None if goal_writes is None else frozenset(goal_writes)
        self.write_tools = frozenset(write_tools or ())
        self.unanswerable = 0
        self.silent = 0

    def goal_done(self, made: Iterable[str]) -> bool:
        """Every write the Task implied has been made, on D227's effect set (D251).

        Empty `goal_writes` no longer vacuously satisfies when `write_tools` is non-empty. A
        communicate-only Task (neither set) keeps D210's empty-set satisfy.
        """
        return rules_mod.goal_writes_done(made, self.goal_writes, self.write_tools)

    def kind(self, question: Optional[str], *, said_anything: bool, had_nothing: bool,
             made: Iterable[str] = (), requested: Optional[str] = None) -> Optional[str]:
        """Which kind this turn ends on, or nothing where it does not end the Run.

        `requested` is what the model asked for through `end_run`; it is read as evidence that the
        user had nothing left, never as the answer, because a model that would rather stop than keep
        asking is exactly the failure this protocol exists to catch. Implied writes still unmade
        block `scenario_exhausted` (D251): the kind stays open or becomes `gave_up` at the turn
        limit, never a pass.
        """
        self.unanswerable += int(bool(had_nothing))
        self.silent = 0 if said_anything else self.silent + 1
        if self.goal_done(made):
            return rules_mod.GOAL_SATISFIED
        owed = rules_mod.writes_owed(made, self.goal_writes, self.write_tools)
        if not owed and self.unanswerable >= rules_mod.UNANSWERABLE_LIMIT:
            return rules_mod.SCENARIO_EXHAUSTED
        if rules_mod.closes(question):
            return rules_mod.HANDED_OFF
        if not owed and self.silent >= rules_mod.SILENCE_LIMIT:
            return rules_mod.SCENARIO_EXHAUSTED
        if requested == rules_mod.HANDED_OFF and rules_mod.closes(question):
            return rules_mod.HANDED_OFF
        return None


def grounding_for(facts: Iterable[Any], sentences: Iterable[str] = ()) -> Grounding:
    """The grounding of one Task, from its askable facts and the sentences they were said in."""
    rows = [f for f in facts if getattr(f, "value", None) is not None]
    by_field: dict[str, list[str]] = {}
    for fact in rows:
        by_field.setdefault(str(getattr(fact, "field", "")), []).append(str(fact.value))
    return Grounding([f.value for f in rows], sentences, by_field)


def writes_made(transcript: Sequence[Any], write_tools: Iterable[str]) -> set[str]:
    """The writes this transcript shows made, read through the rules' own function (D227).

    This used to be a second copy of the reading, which is how the two Simulated users came to be
    able to disagree about whether the goal was done. There is one function now and it lives with
    the rules; this name stays so the agent user's callers keep reading it from its own package.
    """
    return rules_mod.writes_made(transcript, write_tools)
