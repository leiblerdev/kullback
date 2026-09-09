"""What the domain's own public material says people ask for, read as task archetypes (D225).

D224 showed what the graph alone can buy. A walk is a sequence of tools the recordings happened to
chain, so the Tasks it makes are shaped by what a handful of agents did and by nothing else: the
count goes up, the pool does not, and a whole band of walks turned out to be worth nothing. What the
graph cannot know is what a person in this domain actually writes in when they need something. The
public material knows: a help centre, a product manual and a blog say, in the customer's own words,
what people come with and what they expect back.

So this module reads that material and extracts task archetypes: a goal in the customer's voice,
the situation they are in, the effects they expect in the words of the world rather than the names
of tools, and whatever the page states as a rule. An archetype is not a Task. It becomes one only
after `synthesise.py` finds a walk of the mined graph that realises its effects, runs it in the
rebuilt world and puts it through every rung of the realism bar.

Four things keep this honest.

  Nothing about a customer domain is in this code. A source is a URL a caller hands over, a search
  the caller asked for, or a page a model named for a one line description of the domain the caller
  wrote. No domain name, no host and no product is spelled anywhere here, so the same command reads
  any domain's material.

  Nothing read is kept where it could be published. Fetched text lives in a content addressed cache
  under the workdir, never in the package and never in git, and an archetype record carries a byte
  range into that cache rather than the bytes.

  The benchmark cannot read itself. A source under the corpus's own repository or paper is refused
  and counted, and so is every page the crawl or the model would otherwise reach under it. Reading
  a published task list back in would make an archetype that attests nothing.

  Nothing of the page is copied. A record that repeats a sentence of the page it came from is
  dropped, and so is one that repeats a string of the corpus's own Task list, which is
  contamination rather than a leak: the record would then be the benchmark restating itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from html import unescape
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urljoin, urlsplit

from kullback.gates.artifacts import leak_gate
from kullback.runner.records import read_json, write_json
from kullback.sampling import sample_key

# Where an archetype and a gap live, and the shape's version, bumped when a record's fields change.
DIR = "domain"
ARCHETYPES = "archetypes.json"
GAPS = "gaps.json"
FORMAT = 1

# The cache fetched text is kept in, content addressed, under the workdir's own cache directory. It
# is never in the package and never in git: a page of a customer's help centre is the customer's.
CACHE = ("cache", "domain")

# The environment variable a search key is read from. Its value is never printed and never stored.
SEARCH_KEY_VAR = "KULLBACK_SEARCH_KEY"

# How many pages one crawl reads however wide the linking is, and how much of a page a reader is
# shown. Both are about wall clock and cost, not about correctness.
PAGE_LIMIT = 24
PAGE_CHARS = 12000
# A string shorter than this cannot identify a Task of the corpus, so it is not contamination.
MIN_CORPUS_STRING = 12
# How many words in a row an archetype and a corpus Intent have to share before the archetype is
# the Intent restated rather than two texts about one domain.
SHINGLE_WORDS = 6

# Why a source was refused. Counted per reason so a reader can tell a benchmark page from a page
# that simply did not answer.
CORPUS_URL = "under the source corpus's own repository or paper"
EXCLUDED = "under an excluded URL the caller named"
UNREACHABLE = "the page did not answer"

# How a source came to be read. The record says so, because a page a model named and a page the
# caller handed over are different evidence.
GIVEN = "given"
LINKED = "linked"
SEARCHED = "searched"
NAMED = "named"


# --- URLs, hosts and the exclusion --------------------------------------------------

def host_of(url: str) -> str:
    """The host one URL sits on, lowercased and without a port, or nothing where there is none."""
    parts = urlsplit(str(url or ""))
    return (parts.hostname or "").lower()


def _path_of(url: str) -> str:
    parts = urlsplit(str(url or ""))
    return parts.path or "/"


def under(url: str, other: str) -> bool:
    """Whether one URL sits under another: the same host and a path the other's path starts.

    A repository URL and a paper URL both name a host and a path, and everything below that path is
    the same publication. Host alone would refuse a whole code host; path alone would refuse an
    unrelated site that happens to spell the same words.
    """
    if not url or not other:
        return False
    if host_of(url) != host_of(other):
        return False
    base = _path_of(other).rstrip("/")
    return not base or _path_of(url).rstrip("/").startswith(base)


def refusal(url: str, corpus_url: Optional[str], exclude: Iterable[str]) -> str:
    """Why this URL may not be read, or nothing where it may.

    The corpus's own repository and paper are refused whoever proposed the URL: the caller, a link
    on a page, a search result or a model naming pages for a domain description. A benchmark that
    reads its own published task list back in has attested nothing.
    """
    if corpus_url and under(url, corpus_url):
        return CORPUS_URL
    for other in exclude or ():
        if under(url, str(other)):
            return EXCLUDED
    return ""


# --- fetching and the content addressed cache ---------------------------------------

def cache_dir(workdir: Any) -> Path:
    return Path(workdir).joinpath(*CACHE)


def content_key(text: str) -> str:
    """The name a page's text is filed under: its own content and nothing about where it came from."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:32]


def store_text(workdir: Any, text: str) -> str:
    """Put a fetched page in the cache and answer the key it is filed under."""
    key = content_key(text)
    path = cache_dir(workdir) / f"{key}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(str(text), encoding="utf-8")
    return key


def cached_text(workdir: Any, key: str) -> str:
    """The text one cache key holds, or nothing where the cache has been cleared."""
    path = cache_dir(workdir) / f"{str(key)}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK = re.compile(r"\n{3,}")
_HREF = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"'#][^"']*)["']""", re.I)


def page_text(html: str) -> str:
    """A page as a reader is shown it: the words, with the markup and the scripts gone."""
    body = _TAGS.sub(" ", str(html or ""))
    body = _TAG.sub("\n", body)
    body = unescape(body)
    body = _SPACE.sub(" ", body)
    return _BLANK.sub("\n\n", "\n".join(line.strip() for line in body.splitlines())).strip()


def links(html: str, base: str) -> list[str]:
    """Every link a page carries, resolved against the page's own URL, in the order they appear."""
    out: list[str] = []
    for href in _HREF.findall(str(html or "")):
        target = urljoin(str(base), href.strip())
        if target.split("#")[0] not in out and urlsplit(target).scheme in ("http", "https"):
            out.append(target.split("#")[0])
    return out


def http_fetch(url: str, timeout: float = 20.0) -> str:
    """One page over the network, as text. The only place this module opens a socket.

    Callers that must not touch the network hand their own reader in, which is what the tests do:
    the crawl takes a fetch function and never reaches for this one itself.
    """
    from urllib.request import Request, urlopen  # imported here: nothing offline pays for it

    request = Request(str(url), headers={"User-Agent": "kullback-domain-reader"})  # noqa: S310
    with urlopen(request, timeout=timeout) as answer:  # noqa: S310
        raw = answer.read()
    return raw.decode("utf-8", errors="replace")


def crawl(workdir: Any, sources: Iterable[dict], *, fetch: Callable[[str], str],
          depth: int = 1, corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
          limit: int = PAGE_LIMIT) -> dict:
    """Read the sources and the pages they link to on the same host, up to `depth`.

    A link off the source's own host is never followed: a help centre links out to a payment
    processor and a social network, and neither says what this domain's customers ask for. Depth is
    counted from each source, so depth zero reads the sources alone and depth one reads them and
    their own links. Every page read is filed in the cache by its content, and the record carries
    the key rather than the text.
    """
    pages: list[dict] = []
    refused: list[dict] = []
    seen: set[str] = set()
    queue = [(str(row.get("url")), int(row.get("depth") or 0), str(row.get("via") or GIVEN))
             for row in sources or () if row.get("url")]
    while queue and len(pages) < int(limit):
        url, at, via = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        why = refusal(url, corpus_url, exclude)
        if why:
            refused.append({"url": url, "reason": why, "via": via})
            continue
        try:
            html = fetch(url)
        except Exception as error:  # noqa: BLE001 - any transport failure is one unread page
            refused.append({"url": url, "reason": UNREACHABLE, "via": via,
                            "error": type(error).__name__})
            continue
        text = page_text(html)
        if not text:
            refused.append({"url": url, "reason": UNREACHABLE, "via": via})
            continue
        pages.append({"url": url, "key": store_text(workdir, text), "chars": len(text),
                      "via": via, "depth": at})
        if at >= int(depth):
            continue
        for target in links(html, url):
            if host_of(target) == host_of(url) and target not in seen:
                queue.append((target, at + 1, LINKED))
    return {"pages": pages, "refused": refused}


# --- where sources come from ---------------------------------------------------------

def search_sources(query: str, *, search: Optional[Callable[[str, str], list[str]]] = None,
                   env: Optional[dict] = None) -> tuple[list[str], str]:
    """The pages a search names for a query, but only where a key is set in the environment.

    The variable's name is said out loud and its value is never read into a record, a log or a
    report. Without a key this does nothing and says so, rather than half running.
    """
    values = os.environ if env is None else env
    key = str(values.get(SEARCH_KEY_VAR) or "")
    if not key:
        return [], f"no search key in {SEARCH_KEY_VAR}, so no search was run"
    if search is None:
        return [], f"{SEARCH_KEY_VAR} is set but no search reader was given"
    try:
        return [str(url) for url in search(str(query), key) if url], ""
    except Exception as error:  # noqa: BLE001 - a search that fails leaves the given sources alone
        return [], f"the search failed: {type(error).__name__}"


SOURCE_PROMPT = (
    "You are helping someone find the public pages that say what customers of one kind of business "
    "write in about.\n\n"
    "The business is described in one line:\n{domain}\n\n"
    "Name up to {count} public web pages that are worth reading to learn what those customers ask "
    "for: help centre articles, support documentation, product manuals, policy pages, or blog posts "
    "about handling those requests. Prefer pages written for customers over pages written for "
    "investors or press.\n\n"
    "Do not name any page of an academic paper, a benchmark, a dataset or a code repository.\n\n"
    'Answer with JSON only, as {{"pages": [{{"url": "https://...", "why": "one short line"}}]}}.')

# How many pages a domain description is allowed to name, and how many of those are read.
NAMED_LIMIT = 8


def _json_reply(model: Any, prompt: str) -> Any:
    """One model answer parsed as JSON, or nothing where the model failed or wrote prose.

    Every model call in this module goes through here, so a provider failure and a draft that is not
    JSON are one thing to the callers: no records from that page, counted, and never an exception
    thrown out of a read.
    """
    if model is None:
        return None
    try:
        reply = model.query([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 - a provider failure is a page that yielded nothing
        return None
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def named_sources(model: Any, description: str, *, fetch: Callable[[str], str],
                  corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
                  count: int = NAMED_LIMIT) -> dict:
    """The pages a model names for a one line description of a domain, and which of them resolved.

    Domain reading is research and not only transcription: a caller who can describe the business
    should not also have to know its help centre's URL. What comes back is a proposal and nothing
    more. A named page is read only if it is not under the corpus's own publication and only if it
    answers, and the record says which were used and which were dropped, so a reader can see how
    much of the reading the model found rather than the caller.
    """
    body = _json_reply(model, SOURCE_PROMPT.format(domain=str(description), count=int(count)))
    proposed = [str(row.get("url")) for row in ((body or {}).get("pages") or [])
                if isinstance(row, dict) and row.get("url")][:int(count)]
    kept: list[str] = []
    dropped: list[dict] = []
    for url in proposed:
        why = refusal(url, corpus_url, exclude)
        if why:
            dropped.append({"url": url, "reason": why})
            continue
        try:
            if not page_text(fetch(url)):
                dropped.append({"url": url, "reason": UNREACHABLE})
                continue
        except Exception as error:  # noqa: BLE001 - a page that does not resolve is not a source
            dropped.append({"url": url, "reason": UNREACHABLE, "error": type(error).__name__})
            continue
        kept.append(url)
    return {"proposed": proposed, "used": kept, "dropped": dropped}


# --- reading one page into archetypes ------------------------------------------------

READER_PROMPT = (
    "Below is the text of one public page about a business. Read it as evidence of what that "
    "business's own customers write in about.\n\n"
    "Write down each distinct thing a customer of this business would ask for, as a record. Write "
    "every record from the customer's side and never from the operator's side: the goal line is "
    "what the customer would say about themselves, in the first person, beginning with I or We, "
    "saying what they wanted, what they already know, and what went wrong for them. Never write it "
    "as an instruction to staff and never write it as a feature of the product.\n\n"
    "For each record give:\n"
    "  goal: one line, first person, in the customer's own words.\n"
    "  preconditions: the situation this customer is already in, as short lines.\n"
    "  effects: what this customer expects to be different afterwards, in the words of the world "
    "and never as the name of a system, a button or a step.\n"
    "  constraints: any rule or policy the page states about this request, as short lines.\n\n"
    "Write in your own words. Do not copy any sentence of the page.\n\n"
    'Answer with JSON only, as {{"archetypes": [{{"goal": "...", "preconditions": [], "effects": '
    '[], "constraints": []}}]}}.\n\n'
    "The page:\n{text}")

FIELDS = ("preconditions", "effects", "constraints")


def _lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in (value or ()) if str(item).strip()]


def archetype_id(source: str, goal: str) -> str:
    """A name for one archetype: a function of where it came from and what it says, and nothing else."""
    return "arch_" + hashlib.sha256(f"{source}\n{goal}".encode("utf-8")).hexdigest()[:12]


def read_page(model: Any, url: str, text: str, *, chars: int = PAGE_CHARS) -> list[dict]:
    """The archetypes one page attests, as records with a byte range back into the page.

    The range is where the reader was looking, which is the whole span it was shown: a record whose
    provenance cannot be checked against the cached text is a record nobody can audit. No sentence
    of the page travels with the record.
    """
    body = _json_reply(model, READER_PROMPT.format(text=str(text)[:int(chars)]))
    rows = (body or {}).get("archetypes") or []
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        goal = str(row.get("goal") or "").strip()
        if not goal:
            continue
        record = {"id": archetype_id(url, goal), "goal": goal, "source": str(url),
                  "start": 0, "end": min(len(str(text)), int(chars))}
        for field in FIELDS:
            record[field] = _lines(row.get(field))
        out.append(record)
    return out


# The words a person uses about themselves. A goal line written without one of them is written from
# the operator's side, whatever it says: "process the refund" is an instruction to staff and "I want
# my money back" is a customer. The archetype is evidence about customers, so the operator's side is
# dropped rather than rewritten, and the count says how much of a page was written that way.
CUSTOMER_WORDS = frozenset({"i", "i'm", "i've", "i'd", "i'll", "im", "me", "my", "mine", "myself",
                            "we", "we're", "we've", "we'd", "we'll", "us", "our", "ours"})


def customer_voice(goal: str) -> bool:
    """Whether a goal line is written in the first person of the customer rather than at them."""
    words = {word.strip(" .,:;!?\"'()") for word in str(goal or "").lower().split()}
    return bool(words & CUSTOMER_WORDS)


def record_strings(record: dict) -> list[str]:
    """Every string one archetype record carries, which is what the two scans read."""
    return [str(record.get("goal") or "")] + [line for field in FIELDS
                                              for line in _lines(record.get(field))]


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .,:;!?")


_SENTENCE = re.compile(r"[^.!?\n]+")

# A phrase shorter than this is a turn of speech and not a copied sentence.
MIN_SENTENCE_WORDS = 6


def sentences(text: str) -> set[str]:
    """The page's own sentences, normalised, as the copy scan compares against."""
    return {_norm(part) for part in _SENTENCE.findall(str(text or ""))
            if len(_norm(part).split()) >= MIN_SENTENCE_WORDS}


def copies_page(record: dict, text: str) -> str:
    """The sentence of the page this record repeats, or nothing where it repeats none.

    Whole sentences and not phrases: an archetype written from a page will share the domain's words
    with it, which is the point, and a record that shares a sentence is a transcription.
    """
    owned = sentences(text)
    for line in record_strings(record):
        normalised = _norm(line)
        if len(normalised.split()) >= MIN_SENTENCE_WORDS and normalised in owned:
            return line
    return ""


def corpus_strings(workdir: Any, limit: int = 4000) -> list[str]:
    """The strings of the corpus's own Task list, which an archetype may never repeat.

    An Intent is what the benchmark says a Task is, in words. A record that repeats one is the
    benchmark restating itself through a public page, which is contamination: the archetype attests
    nothing, whether the page copied the benchmark or the reader was shown the benchmark.
    """
    root = Path(workdir)
    out: list[str] = []
    body = read_json(root / "tasks.json", {}) or {}
    rows = body.get("tasks") if isinstance(body, dict) else body
    for row in rows or ():
        for field in ("intent", "name"):
            value = str((row or {}).get(field) or "").strip()
            if len(value) >= MIN_CORPUS_STRING:
                out.append(value)
    for path in sorted((root / "intents").glob("*.json"))[:int(limit)]:
        record = read_json(path, {}) or {}
        value = str(record.get("text") or record.get("intent") or "").strip()
        if len(value) >= MIN_CORPUS_STRING:
            out.append(value)
    return out[:int(limit)]


def shingles(text: str, width: int = SHINGLE_WORDS) -> set[str]:
    """Every run of `width` consecutive words in a text, normalised.

    A run of words is what says two texts were written from each other. One word is a domain's
    vocabulary and every archetype will share that with the corpus, which is the point of reading
    the domain at all; six in a row is a sentence being carried across.
    """
    words = _norm(text).split()
    return {" ".join(words[at:at + width]) for at in range(max(len(words) - width + 1, 0))}


def contaminated(record: dict, strings: Iterable[str]) -> str:
    """The corpus string this record repeats, or nothing. The D89 leak scan, read the other way.

    The leak scan asks whether a Verifier's constant reached a Candidate. This asks whether a
    corpus's own words reached an archetype, which is the same match on the same token boundaries
    with the two sides swapped, plus one thing a constant scan cannot see: a run of words shared
    with a corpus Intent, which is that Intent being restated rather than quoted whole.
    """
    strings = [str(text) for text in strings or () if len(str(text)) >= MIN_CORPUS_STRING]
    if not strings:
        return ""
    lines = record_strings(record)
    result = leak_gate(lines, strings, min_length=MIN_CORPUS_STRING)
    for failure in result.failures or ():
        for text in strings:
            if str(text) in str(failure):
                return text
    held = {shingle for line in lines for shingle in shingles(line)}
    for text in strings:
        if held & shingles(text):
            return text
    return ""


# --- deduplication by goal ------------------------------------------------------------

SAME_PROMPT = (
    "Two people each described something they want done. Decide whether they are asking for the "
    "same thing.\n\n"
    "A: {first}\nB: {second}\n\n"
    "Answer with JSON only. If they are the same request, answer "
    '{{"same": true, "citation": "the words both of them use for it"}}. The citation must be words '
    "that appear in both lines; without one, answer {{\"same\": false}}. If they differ in what "
    'would have to change in the world, answer {{"same": false}}.')


def same_goal(model: Any, first: str, second: str) -> tuple[bool, str]:
    """Whether two goals are one, and the words both of them use for it.

    Equal only with a citation: a judge that answers same and cannot point at the words both lines
    use has not read them, and two archetypes collapsing on that answer would lose a real request.
    Without a model, only two goals that read the same after normalising are one.
    """
    if _norm(first) == _norm(second):
        return True, _norm(first)
    body = _json_reply(model, SAME_PROMPT.format(first=str(first), second=str(second)))
    if not isinstance(body, dict) or not body.get("same"):
        return False, ""
    citation = _norm(body.get("citation") or "")
    if not citation or citation not in _norm(first) or citation not in _norm(second):
        return False, ""
    return True, citation


def dedup(records: Iterable[dict], model: Any = None) -> tuple[list[dict], list[dict]]:
    """The archetypes with the repeats folded in, and the folds, each with the citation that made it.

    A fold keeps the first record and adds the second's source to it, so an archetype two pages
    attest says so and is not counted twice.
    """
    kept: list[dict] = []
    folded: list[dict] = []
    for record in records or ():
        match = None
        for held in kept:
            same, citation = same_goal(model, held.get("goal"), record.get("goal"))
            if same:
                match = (held, citation)
                break
        if match is None:
            kept.append(dict(record, sources=[str(record.get("source"))]))
            continue
        held, citation = match
        if str(record.get("source")) not in held["sources"]:
            held["sources"].append(str(record.get("source")))
        folded.append({"id": record.get("id"), "into": held.get("id"), "citation": citation})
    return kept, folded


# --- mapping an archetype onto the tool graph -----------------------------------------

MAP_PROMPT = (
    "A person wants something done. Below are the tools one system has, with the arguments they "
    "take and the fields they answer with.\n\n"
    "What the person wants:\n  goal: {goal}\n  they expect afterwards:\n{effects}\n\n"
    "The tools:\n{tools}\n\n"
    "For each expected effect, name the tools that would make it true, using the tool names exactly "
    "as they are spelled above. If no tool would make an effect true, say so for that effect.\n\n"
    'Answer with JSON only, as {{"effects": [{{"effect": "...", "tools": ["..."]}}]}}. An effect no '
    "tool realises carries an empty list.")


def tool_lines(sigs: Iterable[Any], graph: dict) -> list[str]:
    """Every tool as the mapper is shown it: its name, its arguments and the fields it answers with.

    The result fields come from the graph's own edges, which are the paths the recordings showed a
    value travelling out of this tool under. Nothing here reads a customer's value.
    """
    columns: dict[str, list[str]] = {}
    for edge in graph.get("edges") or ():
        column = str(edge.get("column") or "")
        if column and column not in columns.setdefault(str(edge.get("from")), []):
            columns[str(edge.get("from"))].append(column)
    out = []
    for sig in sigs or ():
        name = str(getattr(sig, "name", "") or "")
        if not name:
            continue
        schema = getattr(sig, "args_schema", None) or {}
        args = sorted((schema.get("properties") or {}) if isinstance(schema, dict) else {})
        kind = str(getattr(sig, "kind", "") or "")
        out.append(f"- {name} ({kind}); arguments: {', '.join(args) or 'none'}; "
                   f"answers with: {', '.join(sorted(columns.get(name, []))[:8]) or 'nothing recorded'}")
    return out


def _nodes(graph: dict) -> dict[str, dict]:
    return {str(node.get("name")): node for node in graph.get("nodes") or ()}


def reachable_from_start(graph: dict) -> set[str]:
    """Every tool a walk could arrive at, starting where a recorded Run started and following edges.

    A write no recorded path reaches is a write the graph cannot walk to, so a mapping that names it
    is a mapping no Task can be shaped from, however plainly the page describes the effect.
    """
    nodes = _nodes(graph)
    frontier = {name for name, node in nodes.items() if node.get("starts")}
    seen = set(frontier)
    edges = [(str(edge.get("from")), str(edge.get("to"))) for edge in graph.get("edges") or ()]
    while frontier:
        step = {to for src, to in edges if src in frontier and to not in seen}
        seen |= step
        frontier = step
    return seen


def check_mapping(named: Iterable[str], graph: dict) -> tuple[list[str], list[str]]:
    """The tools a mapping named that the graph holds and can reach, and the reasons the rest fell.

    Code and not the model: a mapper that names a tool the Environment does not have, or a write no
    recorded start reaches, has proposed a Task that cannot run, and a name is cheap to check.
    """
    nodes = _nodes(graph)
    reachable = reachable_from_start(graph)
    kept: list[str] = []
    reasons: list[str] = []
    for name in named or ():
        name = str(name)
        node = nodes.get(name)
        if node is None:
            reasons.append(f"no tool named {name}")
            continue
        if node.get("write") and name not in reachable:
            reasons.append(f"the write {name} is not reachable from a recorded start")
            continue
        if name not in kept:
            kept.append(name)
    return kept, reasons


def map_archetype(model: Any, record: dict, sigs: Iterable[Any], graph: dict) -> dict:
    """One archetype against the tool graph: which tools realise which effect, checked in code."""
    effects = _lines(record.get("effects"))
    body = _json_reply(model, MAP_PROMPT.format(
        goal=str(record.get("goal") or ""),
        effects="\n".join(f"  - {line}" for line in effects) or "  - nothing stated",
        tools="\n".join(tool_lines(sigs, graph)) or "none"))
    rows = (body or {}).get("effects") or []
    writes = {str(node.get("name")) for node in graph.get("nodes") or () if node.get("write")}
    tools: list[str] = []
    unrealised: list[str] = []
    reasons: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kept, why = check_mapping(row.get("tools") or (), graph)
        reasons += why
        if not kept:
            unrealised.append(str(row.get("effect") or ""))
            continue
        tools += [name for name in kept if name not in tools]
    return {"id": record.get("id"), "goal": record.get("goal"), "source": record.get("source"),
            "sources": record.get("sources") or [str(record.get("source"))],
            "tools": tools, "write_tools": [name for name in tools if name in writes],
            "effects_unrealised": unrealised, "reasons": sorted(set(reasons))}


def split_mapped(mappings: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """The archetypes this Environment can execute, and the gaps: what it cannot.

    A gap is not a failure of the reading. It is what this domain does that this Environment does
    not, in the words of the page, and it is the first input to a specification environment.
    """
    mapped: list[dict] = []
    gaps: list[dict] = []
    for row in mappings or ():
        (mapped if row.get("write_tools") else gaps).append(row)
    return mapped, gaps


# --- the store --------------------------------------------------------------------------

def store_dir(workdir: Any) -> Path:
    return Path(workdir) / DIR


def write_domain(workdir: Any, body: dict, gaps: Iterable[dict]) -> Path:
    """archetypes.json and gaps.json, written together because one is the other's complement."""
    write_json(store_dir(workdir) / GAPS, {"format": FORMAT, "gaps": list(gaps)})
    return write_json(store_dir(workdir) / ARCHETYPES, body)


def read_domain(workdir: Any) -> dict:
    body = read_json(store_dir(workdir) / ARCHETYPES, None)
    return body if isinstance(body, dict) else {"format": FORMAT, "archetypes": [], "counts": {}}


def read_gaps(workdir: Any) -> list[dict]:
    body = read_json(store_dir(workdir) / GAPS, None)
    return list((body or {}).get("gaps") or []) if isinstance(body, dict) else []


def counts_of(workdir: Any) -> dict:
    """What the reading holds, as a round line and a report read it: never a value of any world."""
    body = read_domain(workdir)
    counts = dict(body.get("counts") or {})
    rows = body.get("archetypes") or []
    counts.update({"archetypes_extracted": int(counts.get("archetypes_extracted") or len(rows)),
                   "archetypes_mapped": sum(1 for row in rows if row.get("write_tools")),
                   "archetype_gaps": len(read_gaps(workdir))})
    return counts


def per_source(body: dict) -> list[dict]:
    """How many archetypes each source yielded, by URL, which is what says a page was worth reading."""
    held: dict[str, dict] = {}
    for row in body.get("archetypes") or ():
        for url in row.get("sources") or [row.get("source")]:
            row_held = held.setdefault(str(url), {"source": str(url), "archetypes": 0, "mapped": 0})
            row_held["archetypes"] += 1
            row_held["mapped"] += 1 if row.get("write_tools") else 0
    return [held[key] for key in sorted(held)]


# --- one reading -------------------------------------------------------------------------

def read(workdir: Any, *, sources: Iterable[str] = (), fetch: Callable[[str], str] = http_fetch,
         depth: int = 1, corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
         reader: Any = None, mapper: Any = None, judge: Any = None,
         description: str = "", search_query: str = "",
         search: Optional[Callable[[str, str], list[str]]] = None,
         env: Optional[dict] = None, limit: int = PAGE_LIMIT) -> dict:
    """Read a domain's public material into archetypes and gaps, and write both down.

    The order is fixed and every stage counts what it dropped: sources are gathered, the crawl reads
    them, a reader extracts records per page, the two scans drop the records that copy, the judge
    folds the repeats, the mapper puts the rest against the graph and the split files the gaps.
    """
    from kullback.synthesise import World, load_graph

    workdir = Path(workdir)
    given = [{"url": str(url), "depth": 0, "via": GIVEN} for url in sources or () if url]
    found: list[str] = []
    notes: list[str] = []
    named: dict = {}
    if description:
        named = named_sources(reader, description, fetch=fetch, corpus_url=corpus_url,
                              exclude=exclude)
        found += named.get("used") or []
        given += [{"url": url, "depth": 0, "via": NAMED} for url in named.get("used") or []]
    if search_query:
        results, note = search_sources(search_query, search=search, env=env)
        if note:
            notes.append(note)
        given += [{"url": url, "depth": 0, "via": SEARCHED} for url in results]
    read_pages = crawl(workdir, given, fetch=fetch, depth=depth, corpus_url=corpus_url,
                       exclude=exclude, limit=limit)
    strings = corpus_strings(workdir)
    records: list[dict] = []
    copied = contaminated_count = operator_side = 0
    for page in read_pages["pages"]:
        text = cached_text(workdir, page["key"])
        for record in read_page(reader, page["url"], text):
            if not customer_voice(record.get("goal")):
                operator_side += 1
                continue
            repeat = copies_page(record, text)
            if repeat:
                copied += 1
                continue
            if contaminated(record, strings):
                contaminated_count += 1
                continue
            records.append(record)
    extracted = len(records)
    kept, folded = dedup(records, judge)
    graph = load_graph(workdir)
    sigs = World(workdir).sigs if (workdir / "tool_sigs.json").is_file() else []
    mappings = [map_archetype(mapper, record, sigs, graph) for record in kept]
    mapped, gaps = split_mapped(mappings)
    for record in kept:
        for row in mappings:
            if row.get("id") == record.get("id"):
                record.update({"tools": row["tools"], "write_tools": row["write_tools"],
                               "effects_unrealised": row["effects_unrealised"]})
    body = {"format": FORMAT, "archetypes": kept,
            "sources": {"given": [str(url) for url in sources or ()],
                        "named": named, "notes": notes,
                        "pages": read_pages["pages"], "refused": read_pages["refused"]},
            "counts": {"sources_read": len(read_pages["pages"]),
                       "sources_refused": len(read_pages["refused"]),
                       "sources_named": len(found),
                       "archetypes_extracted": extracted,
                       "archetypes_copied": copied,
                       "archetypes_operator_side": operator_side,
                       "archetypes_contaminated": contaminated_count,
                       "archetypes_folded": len(folded),
                       "archetypes_kept": len(kept),
                       "archetypes_mapped": len(mapped),
                       "archetype_gaps": len(gaps)},
            "folded": folded}
    write_domain(workdir, body, gaps)
    return body


# --- the plausibility judges (D225 rule 5) -------------------------------------------------

PLAUSIBLE_PROMPT = (
    "{persona}\n\n"
    "Someone wrote down a request a customer might send in, and the steps a support agent would "
    "take for it. The request was supposed to be an instance of this archetype, which was read off "
    "the business's own public material:\n"
    "  the customer wants: {goal}\n"
    "  they are in this situation: {preconditions}\n"
    "  they expect afterwards: {effects}\n"
    "  the material states these rules: {constraints}\n\n"
    "The request as written:\n{intent}\n\n"
    "The steps taken: {steps}\n\n"
    "Your only power is to reject. If a real customer would not send this, or the steps do not do "
    'what the archetype says they expect, answer {{"reject": true, "citation": "the line of the '
    'archetype above that it contradicts"}}. The citation must be one of the archetype lines you '
    'were shown. Otherwise answer {{"reject": false}}. You cannot approve anything; a false answer '
    "means only that you found no reason to remove it.")

SHAPE_JUDGE = ("You read requests for whether their shape is one a person would write: what they "
               "say, what they leave out, and whether the outcome they ask for is one they would "
               "want.")
CHECKS_JUDGE = ("You read requests for whether the work described actually delivers what the "
                "person asked for, step by step, and whether anything asked for is left undone.")
PERSONAS = (SHAPE_JUDGE, CHECKS_JUDGE)


def rejection(model: Any, persona: str, record: dict, intent: str,
              steps: Iterable[str]) -> str:
    """The archetype line this Task contradicts, in one judge's reading, or nothing.

    A judge can never pass a Task. It can only remove one, and only by citing a line of the
    archetype it was shown: an unciting rejection is a judge disliking a sentence, and a generator
    that lets that through is one whose Task count is a model's mood.
    """
    lines = [str(record.get("goal") or "")] + [line for field in FIELDS
                                               for line in _lines(record.get(field))]
    body = _json_reply(model, PLAUSIBLE_PROMPT.format(
        persona=str(persona), goal=record.get("goal"),
        preconditions="; ".join(_lines(record.get("preconditions"))) or "nothing stated",
        effects="; ".join(_lines(record.get("effects"))) or "nothing stated",
        constraints="; ".join(_lines(record.get("constraints"))) or "nothing stated",
        intent=str(intent), steps=", ".join(str(step) for step in steps or ()) or "none"))
    if not isinstance(body, dict) or not body.get("reject"):
        return ""
    citation = _norm(body.get("citation") or "")
    for line in lines:
        if citation and _norm(line) == citation:
            return line
    return ""


def judged(models: Iterable[Any], record: dict, intent: str, steps: Iterable[str]) -> list[dict]:
    """Every rejection the judges made, one row each, with the persona and the line it cited."""
    out = []
    for model, persona in zip(list(models or ()), PERSONAS, strict=False):
        cited = rejection(model, persona, record, intent, steps)
        if cited:
            out.append({"persona": persona, "citation": cited})
    return out


def pick(records: Iterable[dict], ident: str) -> Optional[dict]:
    """One archetype of a list, under the keyed draw the rest of the harness draws with (D212)."""
    rows = list(records or ())
    if not rows:
        return None
    return rows[sample_key("domain-archetype", str(ident), "") % len(rows)]


__all__ = ["ARCHETYPES", "CACHE", "CORPUS_URL", "CUSTOMER_WORDS", "DIR", "EXCLUDED", "FIELDS",
           "FORMAT", "GAPS", "GIVEN", "LINKED", "NAMED", "PAGE_LIMIT", "PERSONAS", "SEARCHED",
           "SEARCH_KEY_VAR", "UNREACHABLE", "archetype_id", "cache_dir", "cached_text",
           "check_mapping", "contaminated", "copies_page", "corpus_strings", "counts_of", "crawl",
           "customer_voice", "dedup",
           "host_of", "http_fetch", "judged", "links", "map_archetype", "named_sources",
           "page_text", "per_source", "pick", "read", "read_domain", "read_gaps", "read_page",
           "reachable_from_start", "record_strings", "refusal", "rejection", "same_goal",
           "search_sources", "sentences", "shingles", "split_mapped", "store_dir", "store_text", "under",
           "write_domain"]
