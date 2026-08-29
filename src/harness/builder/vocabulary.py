"""The facts a customer's users state and the words agents ask for them with (D115).

What a fact is comes from the corpus, in code: a tool argument the recorded users stated in their
own turns, its value shape from the values themselves and the schema's id patterns, its kind from
where the argument sits (the tool that opens a Run names the user; a value the world keys a row by
is a reference). The web adds one thing, through the model: the words an agent uses to ask for a
field that this corpus never showed ("booking reference" for `reservation_id`), and every such
word has to appear on a page that was fetched, or it is dropped. A generic core (email, name,
phone, address, postal code) is the same in every domain and ships as code.

`user_sim.py` reads the result: the value patterns say what a user turn states, the cues say what
an agent turn asks for, the prefix says how a stated id is stored.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Literal, Optional

from harness.builder.mine import id_pattern
from harness.shared.records import EntitySchema, Record, ToolSig, Trace

Kind = Literal["identity", "reference", "value"]

STATED_SHARE = 0.5     # a tool argument is a user fact when this share of its recorded values was said by the user
STATED_MIN = 3         # ... and at least this many were
OPENER_SHARE = 0.5     # a tool is the one that names the user when this share of its calls open a Run
ENUM_MAX = 12          # a field with at most this many distinct values is matched by listing them
LOOSE_SHARE = 0.01     # a value pattern that matches this share of ordinary user words is asked-only
USELESS_SHARE = 0.10   # ... and one that matches this share is no pattern at all (two letters is any word)
ENUM_PRECISION = 0.5   # an enumerated value has to go with its call at least this often ("no" goes with everything)
DIGIT_SHARE = 0.8      # values that carry a digit this often are matched only where a digit is
MAX_WEB_FIELDS = 12
MAX_ALIASES = 8
MAX_ALIAS_WORDS = 4
PAGES_PER_FIELD = 3
PAGE_CHARS = 3000
WORD_RE = re.compile(r"[A-Za-z0-9#][A-Za-z0-9#_@.'-]*")
ALIAS_RE = re.compile(r"^[a-z0-9#][a-z0-9# ]*$")
STOP_ALIASES = frozenset("the a an your my this that it id number value information details".split())


class FieldSpec(Record):
    """One fact users state: how a turn states it, how an agent asks for it, how it is stored."""
    field: str
    kind: Kind = "value"
    pattern: Optional[str] = None      # over a user turn; group 1 when there is one, else the whole match
    asked_only: Optional[str] = None   # a bare value, read only from a turn that answers an ask for the field
    prefix: str = ""                   # the stored form's leading mark that users drop, retail's '#'
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


def _cue(words: str) -> str:
    return r"\b" + re.escape(words) + r"\b"


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


# --- reading the corpus -------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("’", "'")).strip().lower()


def _bare(value: str) -> str:
    """A value without the leading mark users drop when they say it."""
    return re.sub(r"^[^A-Za-z0-9]+", "", value.strip())


def _prefix_of(value: str) -> str:
    return value.strip()[:len(value.strip()) - len(_bare(value))]


def _user_text_before(trace: Trace, call_idx: int) -> str:
    """Everything the user said before the turn that made the call; nothing after it can have caused it."""
    turns = [t for t in trace.turns if t.role == "user"]
    cut = next((t.idx for t in trace.turns if call_idx in (t.tool_call_ids or [])), None)
    if cut is None:
        return _norm(" ".join(t.content or "" for t in turns))
    return _norm(" ".join(t.content or "" for t in turns if t.idx < cut))


def _stated_values(traces: Iterable[Trace]) -> dict[tuple[str, str], dict]:
    """(tool, argument) -> the recorded values and how many the user had said before the call."""
    out: dict[tuple[str, str], dict] = defaultdict(lambda: {"values": [], "stated": 0, "prefixes": Counter()})
    for trace in traces:
        by_id = {c.id: c for c in trace.tool_calls if c.id}
        for call in trace.tool_calls:
            said = _user_text_before(trace, call.id) if call.id in by_id else _user_text_before(trace, "")
            for arg, value in (call.args or {}).items():
                if not isinstance(value, str) or not value.strip() or len(value) > 80:
                    continue
                cell = out[(call.name, arg)]
                cell["values"].append(value.strip())
                cell["prefixes"][_prefix_of(value)] += 1
                if _norm(_bare(value)) and _norm(_bare(value)) in said:
                    cell["stated"] += 1
    return out


def _openers(traces: Iterable[Trace]) -> set[str]:
    """Tools whose calls open a Run at least OPENER_SHARE of the time: the ones that name the user."""
    first: Counter = Counter()
    total: Counter = Counter()
    for trace in traces:
        if trace.tool_calls:
            first[trace.tool_calls[0].name] += 1
        total.update(c.name for c in trace.tool_calls)
    return {name for name, n in total.items() if n and first[name] / n >= OPENER_SHARE}


def _ordinary_words(traces: Iterable[Trace], values: set[str]) -> list[str]:
    """The words users said that are not a recorded value: what a value pattern must not match."""
    bare = {_norm(_bare(v)) for v in values}
    words = [w for t in traces for turn in t.turns if turn.role == "user"
             for w in WORD_RE.findall(turn.content or "")]
    return [w for w in words if _norm(_bare(w)) not in bare]


def _scan_pattern(pattern: str, prefix: str) -> str:
    """A schema id pattern as a regex over a user turn: anchors off, the prefix optional, `.` narrowed."""
    body = pattern.strip("^$").replace(".", "[A-Za-z0-9]")
    quoted = re.escape(prefix)
    for lead in (quoted, prefix):  # the schema writes the mark bare or escaped; either way it comes off
        if lead and body.startswith(lead):
            body = body[len(lead):]
            break
    return (f"(?:{quoted})?" if quoted else "") + r"\b" + body + r"\b"


def _with_digit(pattern: str, values: list[str]) -> str:
    """Where the recorded values carry a digit, so must a match: six letters is a word, ZFA14B is an id."""
    if r"\d" in pattern or not values:
        return pattern
    if sum(any(c.isdigit() for c in v) for v in values) / len(values) < DIGIT_SHARE:
        return pattern
    return pattern.replace(r"\b", r"\b(?=[A-Za-z0-9]*\d)", 1)


def _value_pattern(values: list[str], id_patterns: dict[str, str], arg: str, prefix: str,
                   precise: Optional[dict[str, bool]] = None) -> Optional[str]:
    """A regex for the values as users state them: the schema's id shape for that column, else the
    values listed when there are few and each goes with its call, else the shape the values share;
    None when they share nothing."""
    bare = [_bare(v) for v in values]
    for column, pattern in id_patterns.items():
        if column.rsplit(".", 1)[-1] == arg and sum(bool(re.fullmatch(pattern, v)) for v in values) >= 0.9 * len(values):
            return _with_digit(_scan_pattern(pattern, prefix), bare)
    distinct = sorted(set(bare), key=str.lower)
    if (len(distinct) <= ENUM_MAX and all(len(v) >= 3 for v in distinct)
            and all((precise or {}).get(_norm(v), True) for v in distinct)):
        return r"\b(?i:" + "|".join(re.escape(_norm(v)) for v in distinct) + r")\b"
    if len(distinct) <= ENUM_MAX and not all((precise or {}).get(_norm(v), True) for v in distinct):
        return None  # a few values and one of them is said in Runs that never carry it: a word, not a fact
    shape = id_pattern(bare)
    if not shape or not re.search(r"\\d|[A-Za-z0-9]", shape.strip("^$")):
        return None
    return _with_digit(_scan_pattern(shape, ""), bare)


def _enum_precision(traces: list[Trace], stated: dict[tuple[str, str], dict]) -> dict[str, bool]:
    """Whether a value goes with a call that carries it: "economy" is said when a booking asks for
    it, "no" is said in every Run. Per normalized value: Runs whose calls carried it over Runs
    whose user said it, at least ENUM_PRECISION."""
    said_in: Counter = Counter()
    carried_in: dict[str, set] = defaultdict(set)
    values = {_norm(_bare(v)) for cell in stated.values() for v in cell["values"]}
    for trace in traces:
        text = _norm(" ".join(t.content or "" for t in trace.turns if t.role == "user"))
        for value in values:
            if value and re.search(r"\b" + re.escape(value) + r"\b", text):
                said_in[value] += 1
        for call in trace.tool_calls:
            for value in (call.args or {}).values():
                if isinstance(value, str) and _norm(_bare(value)) in values:
                    carried_in[_norm(_bare(value))].add(trace.trace_id)
    return {v: (len(carried_in[v]) / said_in[v] >= ENUM_PRECISION) if said_in[v] else True for v in values}


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


def _folds_into(field: str, generic: list[FieldSpec]) -> Optional[FieldSpec]:
    """A derived field whose name a generic field already asks for is that field (first_name is name)."""
    words = _field_words(field)
    for spec in generic:
        if words == spec.field or any(re.search(cue, words) for cue in spec.cues):
            return spec
    return None


def domain_name(policy_text: str, tables: Iterable[str]) -> str:
    """What the domain is called, for a search query: the policy's title line, else its tables."""
    for line in (policy_text or "").splitlines():
        text = line.strip().lstrip("#").strip()
        if text:
            return text[:80]
    return ", ".join(sorted(tables))[:80] or "customer support"


def derive(traces: list[Trace], schema: EntitySchema, sigs: list[ToolSig], policy_text: str = "") -> Vocabulary:
    """The fields of this corpus, from code alone: generic core plus every argument users stated."""
    vocab = Vocabulary(domain=domain_name(policy_text, schema.tables),
                       fields=[f.model_copy(deep=True) for f in GENERIC_FIELDS])
    stated = _stated_values(traces)
    openers = _openers(traces)
    known_args = {(s.name, f.name) for s in sigs for f in s.args_fields} | {(s.name, a) for s in sigs for a in (s.args_schema.get("properties") or {})}
    candidates: dict[str, dict] = {}
    for (tool, arg), cell in stated.items():
        if known_args and (tool, arg) not in known_args:
            continue
        if cell["stated"] < STATED_MIN or cell["stated"] / len(cell["values"]) < STATED_SHARE:
            continue
        spot = candidates.setdefault(arg, {"values": [], "tools": [], "prefixes": Counter(), "identity": False})
        spot["values"] += cell["values"]
        spot["tools"].append(tool)
        spot["prefixes"].update(cell["prefixes"])
        spot["identity"] = spot["identity"] or tool in openers
    all_values = {v for spot in candidates.values() for v in spot["values"]}
    ordinary = _ordinary_words(traces, all_values)
    precise = _enum_precision(traces, {k: c for k, c in stated.items() if k[1] in candidates})
    for arg, spot in sorted(candidates.items()):
        sources = [f"signature:{t}" for t in sorted(set(spot["tools"]))] + [f"trace:{len(spot['values'])} values"]
        generic = _folds_into(arg, vocab.fields)
        if generic is not None:
            generic.sources += [s for s in sources if s not in generic.sources]
            continue
        prefix = spot["prefixes"].most_common(1)[0][0] if spot["prefixes"] else ""
        column = next((c for c in schema.id_patterns if c.rsplit(".", 1)[-1] == arg), None)
        # Who the user is: an argument of the tool that opens a Run, holding a value the world keys a row by
        # (airline's user_id); a choice an opening write takes (a cabin) is a value like any other.
        kind: Kind = "identity" if spot["identity"] and column else "reference" if column else "value"
        pattern = _value_pattern(spot["values"], schema.id_patterns, arg, prefix, precise)
        asked_only = None
        if pattern is not None and ordinary:
            loose = sum(bool(re.fullmatch(pattern, w)) for w in ordinary) / len(ordinary)
            if loose > USELESS_SHARE:  # any two letters, any run of letters: no shape to read at all
                pattern = None
            elif loose > LOOSE_SHARE:
                pattern, asked_only = None, pattern
        cues, aliases = _base_cues(arg)
        vocab.fields.append(FieldSpec(
            field=arg, kind=kind, pattern=pattern, asked_only=asked_only, prefix=prefix, cues=cues, aliases=aliases,
            sources=sources + ([f"schema:{column}"] if column else []),
            examples=[v for v, _ in Counter(spot["values"]).most_common(3)]))
    return vocab


# --- the web, for wording only --------------------------------------------------

def _queries(domain: str, spec: FieldSpec) -> list[str]:
    """What to search for a field. The domain and the field's own words only: a corpus value is a
    customer's datum and never goes onto the web."""
    words = _field_words(spec.field)
    return [f"{domain} customer support {words} also called",
            f"where to find your {words} {domain} help"]


def alias_prompt(domain: str, spec: FieldSpec, pages: list[Any]) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {p.url}\n{p.text[:PAGE_CHARS]}" for i, p in enumerate(pages))
    return (f"These pages are about {domain}.\n\n{excerpts}\n\n"
            f"A support tool takes an argument `{spec.field}` with values like {spec.examples or ['(none shown)']}. "
            f"List the words a support agent uses when asking a customer for this value, exactly as they appear "
            f"in the pages above, lowercase, at most {MAX_ALIAS_WORDS} words each. Reply with JSON only: "
            f'{{"aliases": ["...", "..."]}}. If the pages do not name it, reply {{"aliases": []}}.')


def parse_aliases(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    rows = data.get("aliases") if isinstance(data, dict) else None
    return [str(a).strip().lower() for a in rows if isinstance(a, str)] if isinstance(rows, list) else []


def grounded_aliases(aliases: Iterable[str], pages: list[Any], spec: FieldSpec) -> list[tuple[str, str]]:
    """The aliases that appear on a fetched page, with the page: the model names, the page vouches."""
    out: list[tuple[str, str]] = []
    for alias in aliases:
        words = alias.split()
        if not ALIAS_RE.match(alias) or not words or len(words) > MAX_ALIAS_WORDS or alias in STOP_ALIASES:
            continue
        if alias in spec.aliases or alias in [a for a, _ in out]:
            continue
        if any(re.fullmatch(shape, alias, re.I) for shape in (spec.pattern, spec.asked_only) if shape):
            continue  # "virginia" is a state, not a word for asking one; a value is never a cue
        page = next((p for p in pages if p.text and alias in p.text.lower()), None)
        if page is not None:
            out.append((alias, page.url))
        if len(out) >= MAX_ALIASES:
            break
    return out


def enrich(vocab: Vocabulary, search: Any, model: Any, pages_per_field: int = PAGES_PER_FIELD) -> Vocabulary:
    """Ask the web how this domain's agents word an ask for each derived field; wording only."""
    out = vocab.model_copy(deep=True)
    if search is None or model is None:
        out.notes.append("no web search: the cues are the field names alone")
        return out
    derived = [f for f in out.fields if "generic" not in f.sources][:MAX_WEB_FIELDS]
    for spec in derived:
        queries = _queries(out.domain, spec)
        row: dict = {"field": spec.field, "queries": queries, "urls": [], "aliases": []}
        try:
            urls: list[str] = []
            for query in queries:
                urls += [h.url for h in search.search(query, limit=pages_per_field) if h.url not in urls]
            pages = [p for p in search.fetch(urls[:pages_per_field * len(queries)]) if p.text and not p.error]
        except Exception as exc:  # the web being down is a note on the vocabulary, not a dead build
            row["error"] = f"{type(exc).__name__}: {exc}"
            out.searched.append(row)
            continue
        row["urls"] = [p.url for p in pages]
        if pages:
            reply = model.query([{"role": "user", "content": alias_prompt(out.domain, spec, pages)}])
            for alias, url in grounded_aliases(parse_aliases(reply.content or ""), pages, spec):
                spec.aliases.append(alias)
                spec.cues.append(_cue(alias))
                spec.sources.append(f"web:{url}")
                row["aliases"].append(alias)
        out.searched.append(row)
    failed = [r for r in out.searched if r.get("error")]
    if failed and len(failed) == len(out.searched):
        out.notes.append(f"search unavailable: {failed[0]['error']}")
    return out
