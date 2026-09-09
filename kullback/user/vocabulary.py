"""What a fact is, in the shape the Simulated user reads it (D115, D214).

The vocabulary has two halves. This one is the shape and the generic core: a FieldSpec says how a
user turn states a value, how an agent turn asks for it and how it is stored, and the five fields
every domain has (email, name, phone, postal code, address) are code because they are the same
everywhere. The other half is the mining, which reads a corpus and the web and stays with the
Builder, because reading a corpus is the Builder's work.

They are apart because the user package may not import the Builder (D214, the import contract): the
user speaks the vocabulary, the Builder derives it, and a package that speaks one has no business
depending on the package that mines it. `kullback.builder.vocabulary` re-exports every name here,
so a caller that has always read them from there still does.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Optional

from kullback.runner.records import Record

Kind = Literal["identity", "reference", "value"]


class FieldSpec(Record):
    """One fact users state: how a turn states it, how an agent asks for it, how it is stored."""
    field: str
    kind: Kind = "value"
    pattern: Optional[str] = None      # over a user turn; group 1 when there is one, else the whole match
    asked_only: Optional[str] = None   # a bare value, read only from a turn that answers an ask for the field
    prefix: str = ""                   # the stored form's leading mark that users drop, such as a '#'
    cues: list[str] = []               # an agent request matching one of these asks for the field
    aliases: list[str] = []            # the plain words the cues were made from
    sources: list[str] = []            # signature:<tool>, schema:<table.column>, trace:<n> values, web:<url>
    examples: list[str] = []


class Vocabulary(Record):
    domain: str = ""
    fields: list[FieldSpec] = []
    searched: list[dict] = []          # one row per query: what was asked, which pages were read
    notes: list[str] = []

    def by_kind(self, kind: Kind) -> list[str]:
        return [f.field for f in self.fields if f.kind == kind]

    def get(self, field: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.field == field), None)

    def field_for(self, arg: str) -> Optional[str]:
        """The field a tool argument states: its own FieldSpec, else the generic field whose words
        already ask for it, else None.

        `derive` names a derived field after the argument it came from and folds the rest into the
        generic core (`first_name` into `name`), so this is that same mapping read the other way
        round: what a reader of a recorded call should call the value it carries.
        """
        spec = self.get(arg) or _folds_into(arg, self.fields)
        return spec.field if spec is not None else None


def _cue(words: str) -> str:
    return r"\b" + re.escape(words) + r"\b"


def _field_words(field: str) -> str:
    return field.replace("_", " ").strip()


def _base_cues(field: str) -> tuple[list[str], list[str]]:
    """Cues from the argument's own name: `order_id` is asked for as an order id, number or #."""
    words = _field_words(field)
    aliases = [words]
    cues = [_cue(words)]
    if field.endswith("_id"):
        entity = _field_words(field[:-3])
        cues += [r"\b" + re.escape(entity) + r" (?:id|number|no\.?|#|reference|code)\b", r"\bwhich " + re.escape(entity) + r"\b"]
        aliases += [f"{entity} number"]
    return cues, aliases


def _folds_into(field: str, generic: Iterable[FieldSpec]) -> Optional[FieldSpec]:
    """A derived field whose name a generic field already asks for is that field (first_name is name)."""
    words = _field_words(field)
    for spec in generic:
        if words == spec.field or any(re.search(cue, words) for cue in spec.cues):
            return spec
    return None


# The generic core: the same words in every domain, so they are code and not evidence.
GENERIC_FIELDS: list[FieldSpec] = [
    FieldSpec(field="email", kind="identity", pattern=r"[\w.+-]+@[\w-]+\.[\w.-]*\w",
              cues=[r"\bemail\b"], aliases=["email"], sources=["generic"]),
    FieldSpec(field="name", kind="identity",
              pattern=r"\b(?i:my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
              asked_only=r"^[^A-Za-z]*(?:it is |it's |sure[!,.]*\s*)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
              cues=[r"\byour (?:first |last |full )?name\b", r"\b(?:first|last|full) name\b"],
              aliases=["name", "first name", "last name", "full name"], sources=["generic"]),
    FieldSpec(field="phone", kind="identity", pattern=r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b",
              cues=[r"\bphone\b"], aliases=["phone"], sources=["generic"]),
    FieldSpec(field="zip", kind="identity",
              pattern=r"(?i:\b(?:zip|zipcode|postal code|postcode)\D{0,12})(\d{5}(?:-\d{4})?)\b",
              asked_only=r"\b(\d{5}(?:-\d{4})?)\b",
              cues=[r"\bzip\b", r"\bpostal code\b", r"\bpostcode\b"],
              aliases=["zip", "postal code", "postcode"], sources=["generic"]),
    FieldSpec(field="address", kind="identity", pattern=r"(?i:(?:my address is|the address is)\s+)(.+?)(?:\.|$)",
              cues=[r"\byour address\b", r"\b(?:shipping|billing|delivery|mailing) address\b"],
              aliases=["address", "shipping address", "billing address"], sources=["generic"]),
]

GENERIC = Vocabulary(domain="generic", fields=[f.model_copy(deep=True) for f in GENERIC_FIELDS])
