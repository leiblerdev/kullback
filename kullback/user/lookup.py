"""The user's fact store and the one lookup both Simulated users answer from (shape-1007, D332).

The store is the facts the recorded user gave (the rules facts and the argument facts), keyed by
field name, with the vocabulary's aliases beside each name; where no build Vocabulary exists the
aliases come from the tool signatures' argument names (D326, `vocabulary.from_signatures`). The
lookup takes question text and gives back the facts it asks for, matched three ways in one order:
the field's own words, then an alias, then plain-word overlap with the field's content words. The
agent user calls it through `my_facts(question=...)`; the rule user calls the same function on the
request sentences of the Candidate's turn. Nothing here reads a cue list: a field is heard by its
own name and the names users and tools gave it.

A field the question names that the store holds no value for comes back as `missing`. That list is
the seam for the user reading its own rows in the world: today the rule user hands it to its
Starting state reader, and a later store answers it from the user's own rows itself.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kullback.runner.records import Record, UserFact
from kullback.user import rules as rules_mod
from kullback.user.vocabulary import GENERIC, Vocabulary

FIELD = "field"
ALIAS = "alias"
OVERLAP = "overlap"
# Words that say which kind of value a field holds and never which thing it is about: an overlap on
# one of these alone ("number", "new") names no field. They are the words of any domain's ids.
KIND_WORDS = frozenset("id ids number numbers no code codes new old current total".split())
# The shortest word an overlap reads, so "or" and "to" never name a field, and the shortest stem
# two words may share ("item" and "items"; never "use" and "user").
MIN_WORD = 3
MIN_STEM = 4
# What a question points at: up to four words after "which", "what" or "your" (stop words drop).
POINTED = re.compile(r"\b(?:which|what|your)\s+((?:[a-z]+\s+){0,3}[a-z]+)")


class Hit(Record):
    """One fact the question asks for, and how it was heard."""
    field: str
    value: Any = None
    how: str = FIELD


class Found(Record):
    """What a question asks for: the facts held, best first per field, and the names held nowhere."""
    question: str = ""
    hits: list[Hit] = []
    missing: list[str] = []

    def fields(self) -> list[str]:
        out: list[str] = []
        for field in [hit.field for hit in self.hits] + list(self.missing):
            if field not in out:
                out.append(field)
        return out


class FactStore:
    """The user's facts by field name, each with its aliases, and the names it knows of but holds
    no value for (a refusal, a record field, a vocabulary field nobody stated)."""

    def __init__(self, facts: Iterable[UserFact], vocab: Vocabulary = GENERIC,
                 names: Iterable[str] = ()):
        self.facts = [fact for fact in facts or ()
                      if fact.field not in rules_mod.SPOKEN_FIELDS and fact.value is not None]
        self.vocab = vocab
        known = [fact.field for fact in self.facts] + list(names or ()) + [spec.field for spec in vocab.fields]
        self.names: list[str] = []
        for field in known:
            if field and field not in self.names and field not in rules_mod.SPOKEN_FIELDS:
                self.names.append(field)

    @classmethod
    def from_rules(cls, rules: Any, vocab: Vocabulary = GENERIC, names: Iterable[str] = ()) -> "FactStore":
        return cls(getattr(rules, "facts", None) or (), vocab,
                   [*(getattr(rules, "refusals", None) or ()), *names])

    def held(self) -> list[str]:
        out: list[str] = []
        for fact in self.facts:
            if fact.field not in out:
                out.append(fact.field)
        return out

    def phrases(self, field: str) -> list[str]:
        """The field's own words, then its aliases, normalised as the question is."""
        spec = self.vocab.get(field)
        words = [_words(field)] + list(spec.aliases if spec is not None else [])
        out: list[str] = []
        for word in words:
            word = rules_mod._norm(word).strip()
            # A name of kind words alone (an argument called "id") would be heard in every "item id".
            if word and word not in out and not set(word.split()) <= KIND_WORDS:
                out.append(word)
        return out

    def lookup(self, question: Optional[str]) -> Found:
        """The facts this question asks for, by field name, alias and plain-word overlap.

        A fact whose value the question already states is not asked for: an agent that reads back
        the record it is about to change names that record and is not asking for it. Where a field has
        two values, the one whose recorded sentence agrees with the question comes first (D44).
        Overlap is read only where no held name or alias was heard, and only over what the question
        points at: "confirm the items on this account" asks for agreement, not for the items.
        """
        text = rules_mod._norm(question)
        if not text.strip():
            return Found(question=question or "")
        heard: dict[str, str] = {}
        for field in self.names:
            for index, phrase in enumerate(self.phrases(field)):
                if re.search(r"\b" + re.escape(phrase) + r"\b" + rules_mod.REPOINT, text):
                    heard[field] = FIELD if index == 0 else ALIAS
                    break
        held = self.held()
        if not any(field in held for field in heard):
            if heard:
                # A name heard that holds no value ("address") reaches the held facts sharing its
                # words ("address 1"), and nothing else the turn happens to point at.
                words = {word for field in heard for phrase in self.phrases(field) for word in _content(phrase)}
            else:
                words = self._pointed(text)
            over = self._overlapping(words, held) or ([] if heard else self._overlapping(words, self.names))
            if over:
                heard = {field: OVERLAP for field in over}
        hits: list[Hit] = []
        missing: list[str] = []
        for field, how in heard.items():
            facts = [fact for fact in self.facts if fact.field == field]
            if facts and all(rules_mod._said_in(question, str(fact.value)) for fact in facts):
                continue  # the question states every value held: nothing is asked
            if not facts:
                missing.append(field)
                continue
            for fact in _best_first(facts, text, field):
                if not rules_mod._said_in(question, str(fact.value)):
                    hits.append(Hit(field=field, value=fact.value, how=how))
        return Found(question=question or "", hits=hits, missing=missing)

    @staticmethod
    def _pointed(text: str) -> set[str]:
        """The content words a question points at with "which", "what" or "your" ("which plan").

        Only those words count, never in a closing question, and "your" not in an offer: on live
        Candidate turns the whole sentence was too loose, "the only item you want to change" asks
        for agreement, "use the card" is not an ask for a user id, "anything else, like your other
        accounts?" closes, and "shall I add a cover to your plan?" asks for a yes.
        """
        words: set[str] = set()
        for sentence in rules_mod._sentences(text) or [text]:
            if rules_mod.CLOSE_CUE.search(sentence):
                continue
            offer = bool(rules_mod.CONFIRM_REQUEST.search(sentence))
            # A word the next word re-points ("the name of the product") names no field of ours.
            sentence = re.sub(r"\b[a-z]+\s+of\b", " ", sentence)
            for match in POINTED.finditer(sentence):
                if offer and match.group(0).startswith("your"):
                    continue  # "shall I add a cover to your plan?" offers, it does not ask
                words |= set(_content(match.group(1)))
        return words

    def _overlapping(self, words: set[str], fields: Iterable[str]) -> list[str]:
        """Of `fields`, those sharing the most content words with `words`."""
        scored: dict[str, int] = {}
        for field in fields:
            own = {word for phrase in self.phrases(field) for word in _content(phrase)}
            score = sum(1 for word in own if any(_same_word(word, seen) for seen in words))
            if score:
                scored[field] = score
        top = max(scored.values(), default=0)
        return [field for field, score in scored.items() if score == top]

    def everything(self) -> Found:
        return Found(hits=[Hit(field=fact.field, value=fact.value) for fact in self.facts])


def asked_in(turn: Optional[str]) -> str:
    """The request sentences of a turn, joined: what a Candidate's turn asks, or empty."""
    return " ".join(rules_mod._request_sentences(rules_mod._norm(turn)))


def _words(field: str) -> str:
    """A field name as words: underscores and camel case split, digits apart ("address1")."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", field.replace("_", " "))
    return re.sub(r"(?<=[A-Za-z])(?=\d)", " ", spaced).lower()


def _content(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z]+", text)
            if len(word) >= MIN_WORD and word not in rules_mod.STOP_WORDS and word not in KIND_WORDS]


def _same_word(a: str, b: str) -> bool:
    """One word, allowing a plural and a shared stem: "items" and "item", "plans" and "plan"."""
    if a == b:
        return True
    a, b = a.rstrip("s") or a, b.rstrip("s") or b
    short, long = sorted((a, b), key=len)
    return len(short) >= MIN_STEM and long.startswith(short)


def _best_first(facts: list[UserFact], text: str, field: str) -> list[UserFact]:
    """Two values of one field, the one the question names first (D44, `SimulatedUser._fact`)."""
    if len(facts) <= 1:
        return facts
    own = set(rules_mod._tokens(_words(field)))
    ranked = sorted(enumerate(facts), key=lambda pair: (
        -rules_mod._agrees(text, pair[1].context), -rules_mod._shared(text, pair[1].context, own), pair[0]))
    return [fact for _, fact in ranked]
