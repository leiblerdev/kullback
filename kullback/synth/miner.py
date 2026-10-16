"""The scenario miner: scenarios from the recordings, a cited search and a customer's lines (D372).

Three sources, each a function returning records. The recordings source reads every Task's opening
ask per recording and groups a batch of them into statements with one model call. The search source
writes queries from the Environment's one-line description and its tool descriptions, reads the pages
the search names through the D225 crawl, and keeps a statement only where the page holds the span it
says it rests on, whose byte range becomes its citation. The ask source rewrites each line a customer
typed as a scenario under the same rule, one model call per batch, and keeps the line as typed in its
source: a line in the customer's own shape would tell the blind review where it came from.

Every statement is held to the same rule by code: no tool name of the Environment, no id shape and
no identifying value of a Starting state row. A recordings or search statement that breaks it is sent
back once with the reason, then dropped and counted. Repeats across sources fold under the D219 judge
(`domain.dedup`), recordings first, so a search or ask scenario the recordings already show keeps the
recordings record and adds its own source under `also_from`. A record keeps the id it was born with,
also after a fix rewrites its statement.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback import domain
from kullback.runner.records import read_json, write_json
from kullback.spec import intent_tools
from kullback.synth import records as R
from kullback.synth import search as search_mod
from kullback.synth.prompt import wrap

# How many opening asks one model call groups, and how many search queries one call writes.
BATCH = 25
QUERIES = 4
# A row value is identifying when it is at least this long and carries a digit, an underscore, an
# at sign, or two words with a capital: an id, a code, an address, a name. A plain word is not.
MIN_VALUE_CHARS = 4
# How many identifying values one Starting state yields to the check, at most.
VALUE_CAP = 50000

TOOL_NAME = "tool_name"
ID_SHAPE = "id_shape"
ROW_VALUE = "row_value"
NO_SPAN = "no_verified_span"
COPIES_PAGE = "copies_page"
CONTAMINATED = "contaminated"
NO_OPENING = "no_opening_grouped"
EMPTY = "empty_statement"

# A letter-and-digit token of four characters or more, a run of five digits, or a number after a hash.
_ID = re.compile(r"#\d+|\b(?=\w*\d)(?=\w*[A-Za-z])\w{4,}\b|\b\d{5,}\b")


@dataclass
class World:
    """What the miner and the review read of the Environment: the policy, the tools, the Starting state
    and the one-line description of the business."""

    workdir: Path
    policy: str = ""
    tools: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    description: str = ""
    values: list[str] = field(default_factory=list)


def _policy(workdir: Path) -> str:
    path = workdir / "env" / "policy.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    for trace in intent_tools.trace_index(workdir).values():
        if trace.system_prompt:
            return str(trace.system_prompt)
    return ""


def tool_rows(workdir: Path) -> list[dict]:
    """Each mined tool: name, read or write, argument names and its description where it has one."""
    sigs = read_json(workdir / "tool_sigs.json", None) or []
    sigs = sigs.get("sigs", []) if isinstance(sigs, dict) else sigs
    out = []
    for sig in sigs if isinstance(sigs, list) else []:
        if isinstance(sig, dict) and sig.get("name"):
            out.append({"name": str(sig["name"]), "kind": sig.get("kind", "read"),
                        "args": sorted(str(f.get("name")) for f in sig.get("args_fields") or []
                                       if isinstance(f, dict)),
                        "description": sig.get("description")})
    return out


def _state(workdir: Path) -> dict:
    for path in (workdir / "db.json", workdir / "env" / "db.json"):
        body = read_json(path, None)
        if isinstance(body, dict):
            return body
    return {}


def identifying(value: Any) -> bool:
    text = str(value)
    if len(text) < MIN_VALUE_CHARS:
        return False
    return (any(ch.isdigit() for ch in text) or "_" in text or "@" in text
            or (" " in text.strip() and any(ch.isupper() for ch in text)))


def _leaves(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value)
    elif isinstance(node, str):
        yield node


def row_values(state: dict, cap: int = VALUE_CAP) -> list[str]:
    """The identifying row keys and string values of the Starting state, lowercased, at most `cap`."""
    found: set[str] = set()
    for rows in state.values():
        if not isinstance(rows, dict):
            continue
        for key, row in rows.items():
            for text in (str(key), *_leaves(row)):
                if identifying(text):
                    found.add(text.strip().lower())
            if len(found) >= cap:
                return sorted(found)
    return sorted(found)


def world_of(workdir: Any, description: str = "") -> World:
    root = Path(workdir)
    state = _state(root)
    meta = read_json(root / "environment.json", {}) or {}
    line = description or (str(meta.get("description") or "") if isinstance(meta, dict) else "")
    return World(workdir=root, policy=_policy(root), tools=tool_rows(root), state=state,
                 description=line, values=row_values(state))


def _named(lowered: str, needle: str) -> bool:
    return needle in lowered and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lowered) is not None


def statement_refusal(text: str, world: World, *, tools_only: bool = False) -> str:
    """Why a statement breaks the rule, as a drop reason, or nothing where it keeps it.

    A tool name (or its words with the underscores read as spaces) is the harness speaking; an id shape
    or an identifying row value ties the statement to one row, where a scenario says the kind of thing.
    """
    lowered = str(text or "").lower()
    for tool in world.tools:
        name = str(tool.get("name") or "").lower()
        if name and (_named(lowered, name) or ("_" in name and _named(lowered, name.replace("_", " ")))):
            return TOOL_NAME
    if tools_only:
        return ""
    if _ID.search(str(text or "")):
        return ID_SHAPE
    if any(_named(lowered, value) for value in world.values):
        return ROW_VALUE
    return ""


# --- prompts ------------------------------------------------------------------------

RULE = wrap("Write each scenario as one or two sentences in the third person saying what happens, in general words: "
        "the kind of person, the kind of thing and the step it has reached. Never name a system, a tool, a "
        "button, an id, a number, a name, a place or any other value taken from a message; say the kind of "
        "thing instead. The shapes, over placeholders:\n"
        "  a <kind of customer> wants to <change> a <thing> that has already <reached a later step>\n"
        "  a <kind of customer> asks for <something> that the rules allow only <under a condition>\n"
        "  a <kind of customer> reports that <a thing> <went wrong> and wants <a remedy>")

RECORDINGS_PROMPT = wrap(
    "Below are the opening messages customers sent to the support agent of one business, numbered.\n\n"
    "Group them into scenarios, one per distinct situation, and say which numbered openings each groups. "
    + RULE + "\n\nQuote any line of the policy a scenario touches, word for word.\n\n"
    "The policy:\n{policy}\n\nThe openings:\n{openings}\n\n"
    'Answer with JSON only, as {{"scenarios": [{{"statement": "...", "openings": [0], "policy_lines": ["..."]}}]}}.')

SEND_BACK_PROMPT = wrap(
    "These scenario statements break the rule below, each with why. Rewrite each one so it keeps the "
    "situation and follows the rule.\n\n" + RULE + "\n\nThe statements:\n{rows}\n\n"
    'Answer with JSON only, as {{"statements": [{{"index": 0, "statement": "..."}}]}}.')

QUERY_PROMPT = wrap(
    "One business is described in one line:\n{description}\n\nWhat its support agent can do:\n{tools}\n\n"
    "Write up to {count} web search queries that find what customers of this kind of business ask for, how "
    "its workflows go and what goes wrong. Never search for an academic paper, a benchmark, a dataset or a "
    'code repository.\n\nAnswer with JSON only, as {{"queries": ["..."]}}.')

PAGE_PROMPT = wrap(
    "Below is the text of one public page about the kind of business described in one line:\n{description}\n\n"
    "Write each distinct situation a customer of this kind of business is in, as the page shows it. " + RULE
    + "\n\nFor each scenario copy the span of the page it rests on, word for word, one or two sentences. A "
    "scenario the page does not show is left out. Quote any line of the policy it touches, word for word.\n\n"
    "The policy:\n{policy}\n\nThe page:\n{text}\n\n"
    'Answer with JSON only, as {{"scenarios": [{{"statement": "...", "span": "...", "policy_lines": ["..."]}}]}}.')

ASK_PROMPT = wrap(
    "Below are lines a person typed, numbered, each naming a situation they want covered, in their own words. "
    "Rewrite each line as one scenario of the situation it names, one per line. " + RULE + "\n\n"
    "The lines:\n{lines}\n\n"
    'Answer with JSON only, as {{"scenarios": [{{"line": 0, "statement": "..."}}]}}.')

FIX_PROMPT = wrap(
    "A reviewer sent this scenario back once, with the reason. Rewrite it from the same source, following "
    "the reason: rewrite the words, rebind it to what the source shows, or narrow it.\n\n" + RULE
    + "\n\nThe scenario:\n{statement}\n\nThe reason:\n{why}\n\nThe source it was written from:\n{source}\n\n"
    'Answer with JSON only, as {{"statement": "..."}}.')


def _ask(model: Any, prompt: str) -> dict:
    body = domain._json_reply(model, prompt)
    return body if isinstance(body, dict) else {}


# --- the recordings ---------------------------------------------------------------------


def _opening_index(workdir: Path, task_id: str) -> dict[str, int]:
    """Per recording, the opening ask the Intent miner marked; none where the Task has no Spec yet."""
    from kullback.spec.schema import load_spec

    try:
        spec = load_spec(workdir, task_id)
    except (OSError, ValueError):
        return {}
    return intent_tools.opening_turns(spec.intent.facts) if spec is not None else {}


def openings(workdir: Any) -> list[dict]:
    """Every recording's opening ask, one row each: the Task, the recording and the user's words.

    The opening is the turn the Intent miner marked as the first ask where a Spec exists, else the
    recording's first user turn.
    """
    root = Path(workdir)
    traces = intent_tools.trace_index(root)
    out: list[dict] = []
    for path in sorted((root / "tasks").glob("*.json")):
        try:
            found = intent_tools.recordings_of(root, path.stem, traces)
        except ValueError:
            continue
        marked = _opening_index(root, path.stem)
        for recording, turns in sorted(found.recordings.items()):
            turn = next((t for t in turns if t.index == marked.get(recording)), turns[0])
            out.append({"task_id": path.stem, "recording": recording, "text": turn.text})
    return out


def _send_back(model: Any, rows: list[tuple[dict, str]]) -> dict[int, str]:
    """The rewrites of the statements that broke the rule, by their index in `rows`."""
    listed = "\n".join(f"[{at}] {row['statement']} (why: {why.replace('_', ' ')})"
                       for at, (row, why) in enumerate(rows))
    body = _ask(model, SEND_BACK_PROMPT.format(rows=listed))
    out: dict[int, str] = {}
    for row in body.get("statements") or ():
        if isinstance(row, dict) and isinstance(row.get("index"), int):
            out[row["index"]] = str(row.get("statement") or "")
    return out


def checked(model: Any, drafts: list[dict], world: World, dropped: Counter) -> list[dict]:
    """The drafts that keep the rule, after one send-back of those that break it; the rest dropped."""
    kept: list[dict] = []
    broken: list[tuple[dict, str]] = []
    for draft in drafts:
        why = statement_refusal(draft["statement"], world) if draft["statement"] else EMPTY
        if why:
            broken.append((draft, why))
        else:
            kept.append(draft)
    if not broken:
        return kept
    rewrites = _send_back(model, broken) if model is not None else {}
    for at, (draft, why) in enumerate(broken):
        again = " ".join(str(rewrites.get(at) or "").split())
        later = statement_refusal(again, world) if again else why
        if later:
            dropped[later] += 1
        else:
            kept.append(dict(draft, statement=again))
    return kept


def _recording_drafts(body: dict, batch: list[dict], policy: str, counts: Counter) -> list[dict]:
    drafts = []
    for row in body.get("scenarios") or ():
        if not isinstance(row, dict):
            continue
        picked = sorted({at for at in row.get("openings") or () if isinstance(at, int) and 0 <= at < len(batch)})
        if not picked:
            counts[NO_OPENING] += 1
            continue
        lines, lost = R.policy_lines_in(row.get("policy_lines") or (), policy)
        counts["policy_lines_dropped"] += lost
        drafts.append({"statement": " ".join(str(row.get("statement") or "").split()), "policy_lines": lines,
                       "task_ids": sorted({batch[at]["task_id"] for at in picked}), "frequency": len(picked)})
    return drafts


def from_recordings(world: World, model: Any, *, batch: int = BATCH,
                    dropped: Optional[Counter] = None) -> list[R.Scenario]:
    """Scenarios grouping the recordings' opening asks, one model call per batch."""
    dropped = Counter() if dropped is None else dropped
    rows = openings(world.workdir)
    out: list[R.Scenario] = []
    for start in range(0, len(rows), max(int(batch), 1)) if model is not None else ():
        chunk = rows[start:start + max(int(batch), 1)]
        listed = "\n".join(f"[{at}] {' '.join(row['text'].split())}" for at, row in enumerate(chunk))
        body = _ask(model, RECORDINGS_PROMPT.format(policy=world.policy or "none recorded", openings=listed))
        drafts = checked(model, _recording_drafts(body, chunk, world.policy, dropped), world, dropped)
        out += [R.born(d["statement"], R.Source(kind="recordings", task_ids=d["task_ids"]), d["policy_lines"],
                       d["frequency"]) for d in drafts]
    return out


# --- the search -------------------------------------------------------------------------


def _tool_lines(world: World) -> str:
    return "\n".join(f"- {row['name']}: {row.get('description') or 'no description'}"
                     for row in world.tools) or "none recorded"


def byte_range(text: str, span: str) -> Optional[tuple[int, int]]:
    """The UTF-8 byte range of the span in the page text, or nothing where the page does not hold it."""
    at = str(text).find(str(span)) if span else -1
    if at < 0 or not str(span).strip():
        return None
    start = len(str(text)[:at].encode("utf-8"))
    return start, start + len(str(span).encode("utf-8"))


def _page_drafts(model: Any, world: World, page: dict, text: str, strings: list[str],
                 dropped: Counter) -> list[dict]:
    body = _ask(model, PAGE_PROMPT.format(description=world.description, policy=world.policy or "none recorded",
                                          text=text[:domain.PAGE_CHARS]))
    drafts = []
    for row in body.get("scenarios") or ():
        if not isinstance(row, dict):
            continue
        statement = " ".join(str(row.get("statement") or "").split())
        where = byte_range(text, str(row.get("span") or ""))
        if where is None:
            dropped[NO_SPAN] += 1
            continue
        if domain.copies_page({"goal": statement}, text):
            dropped[COPIES_PAGE] += 1
            continue
        if domain.contaminated({"goal": statement}, strings):
            dropped[CONTAMINATED] += 1
            continue
        lines, lost = R.policy_lines_in(row.get("policy_lines") or (), world.policy)
        dropped["policy_lines_dropped"] += lost
        drafts.append({"statement": statement, "policy_lines": lines, "url": page["url"], "key": page["key"],
                       "start": where[0], "end": where[1]})
    return drafts


def search_pages(world: World, model: Any, *, search: Optional[Callable[[str, str], list[str]]] = None,
                 env: Optional[dict] = None, fetch: Callable[[str], str] = domain.http_fetch,
                 corpus_url: Optional[str] = None, exclude: Iterable[str] = (), count: int = QUERIES,
                 notes: Optional[list] = None) -> dict:
    """The pages the search names for queries a model wrote, read through the D225 crawl."""
    notes = [] if notes is None else notes
    if not world.description:
        notes.append("no one-line description of the business, so no search was run")
        return {"pages": [], "refused": []}
    body = _ask(model, QUERY_PROMPT.format(description=world.description, tools=_tool_lines(world), count=count))
    urls: list[str] = []
    for query in [str(q) for q in body.get("queries") or () if str(q).strip()][:int(count)]:
        found, note = search_mod.run(query, world.workdir, search=search, env=env)
        if note and note not in notes:
            notes.append(note)
        urls += [url for url in found if url not in urls]
    sources = [{"url": url, "depth": 0, "via": domain.SEARCHED} for url in urls]
    return domain.crawl(world.workdir, sources, fetch=fetch, depth=0, corpus_url=corpus_url, exclude=exclude)


def from_search(world: World, model: Any, *, dropped: Optional[Counter] = None, **reading: Any) -> list[R.Scenario]:
    """Scenarios read off search pages, each citing the byte range of the span it rests on."""
    dropped = Counter() if dropped is None else dropped
    if model is None:
        return []
    pages = search_pages(world, model, **reading)
    strings = domain.corpus_strings(world.workdir)
    out: list[R.Scenario] = []
    for page in pages["pages"]:
        text = domain.cached_text(world.workdir, page["key"])
        drafts = checked(model, _page_drafts(model, world, page, text, strings, dropped), world, dropped)
        out += [R.born(d["statement"], R.Source(kind="search", url=d["url"], key=d["key"], start=d["start"],
                                                end=d["end"]), d["policy_lines"]) for d in drafts]
    dropped["pages_refused"] += len(pages["refused"])
    return out


# --- the customer's lines ---------------------------------------------------------------


def ask_lines(path: Any) -> list[str]:
    target = Path(path)
    if not target.is_file():
        return []
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _ask_drafts(body: dict, chunk: list[str]) -> list[dict]:
    """One draft per line the model rewrote, the first answer for each line index, the line as typed kept."""
    drafts: dict[int, dict] = {}
    for row in body.get("scenarios") or ():
        at = row.get("line") if isinstance(row, dict) else None
        if isinstance(at, int) and 0 <= at < len(chunk) and at not in drafts:
            drafts[at] = {"statement": " ".join(str(row.get("statement") or "").split()), "line": chunk[at]}
    return [drafts[at] for at in sorted(drafts)]


def from_asks(world: World, lines: Iterable[str], model: Any, *, batch: int = BATCH,
              dropped: Optional[Counter] = None, notes: Optional[list] = None) -> list[R.Scenario]:
    """Each line a customer typed, rewritten as a scenario under the same rule, one model call per batch.

    The line as typed stays in the source. A line naming a tool of the Environment is dropped and counted
    before any call; a rewrite that breaks the rule is sent back once, then dropped and counted.
    """
    dropped = Counter() if dropped is None else dropped
    notes = [] if notes is None else notes
    kept = []
    for line in lines:
        why = statement_refusal(line, world, tools_only=True)
        if why:
            dropped[why] += 1
        else:
            kept.append(line)
    if model is None:
        if kept:
            notes.append("no model, so no ask line was rewritten as a scenario")
        return []
    out: list[R.Scenario] = []
    size = max(int(batch), 1)
    for start in range(0, len(kept), size):
        chunk = kept[start:start + size]
        listed = "\n".join(f"[{at}] {line}" for at, line in enumerate(chunk))
        drafts = checked(model, _ask_drafts(_ask(model, ASK_PROMPT.format(lines=listed)), chunk), world, dropped)
        out += [R.born(d["statement"], R.Source(kind="ask", line=d["line"])) for d in drafts]
    return out


# --- folding and coverage ---------------------------------------------------------------


def fold(records: list[R.Scenario], judge: Any = None) -> tuple[list[R.Scenario], list[dict]]:
    """The records with repeats folded in under the D219 judge, in the order given, and the folds."""
    by_id = {record.id: record for record in records}
    rows = [{"id": record.id, "goal": record.statement, "source": record.id} for record in records]
    kept, folded = domain.dedup(rows, judge)
    for row in folded:
        held, other = by_id[row["into"]], by_id[row["id"]]
        held.also_from += [other.source, *other.also_from]
        if held.source.kind == other.source.kind == "recordings":
            held.frequency += other.frequency
            held.source.task_ids = sorted({*held.source.task_ids, *other.source.task_ids})
    return [by_id[row["id"]] for row in kept], [dict(row, kind=by_id[row["id"]].source.kind) for row in folded]


def coverage(kept: list[R.Scenario], folded: list[dict], dropped: Counter) -> dict:
    """Scenarios per source kind, what the recordings never show, the asks nothing else covers, drops."""
    per_kind = Counter(record.source.kind for record in kept)
    return {"per_kind": {kind: per_kind.get(kind, 0) for kind in R.SOURCE_KINDS},
            "search_not_in_recordings": per_kind.get("search", 0),
            "asks_not_covered": per_kind.get("ask", 0),
            "folded": Counter(row["kind"] for row in folded),
            "dropped": dict(sorted(dropped.items()))}


def mine(workdir: Any, *, model: Any, judge: Any = None, asks: Any = None, searching: bool = True,
         batch: int = BATCH, description: str = "", **reading: Any) -> dict:
    """Run the three sources, fold the repeats, write the new records and the counts, return the counts.

    A record already in the store keeps its status and its review: mining again adds, never resets.
    """
    world = world_of(workdir, description)
    dropped: Counter = Counter()
    notes: list[str] = []
    found = from_recordings(world, model, batch=batch, dropped=dropped)
    if searching:
        found += from_search(world, model, dropped=dropped, notes=notes, **reading)
    else:
        notes.append("search turned off")
    path = Path(asks) if asks else R.store_dir(workdir) / R.ASKS
    found += from_asks(world, ask_lines(path), model, batch=batch, dropped=dropped, notes=notes)
    kept, folded = fold(found, judge)
    held = {record.id for record in R.load_scenarios(workdir)}
    R.write_all(workdir, [record for record in kept if record.id not in held])
    body = {"format": R.FORMAT, "notes": notes, "coverage": coverage(kept, folded, dropped),
            "born": len([record for record in kept if record.id not in held]),
            "folded": folded}
    write_json(R.store_dir(workdir) / R.MINED, body)
    return body


# --- the fix step -----------------------------------------------------------------------


def source_text(world: World, record: R.Scenario) -> str:
    """What a record was written from: its grouped openings, its cited span, or the line as typed."""
    source = record.source
    if source.kind == "recordings":
        wanted = set(source.task_ids)
        return "\n".join(row["text"] for row in openings(world.workdir) if row["task_id"] in wanted)
    if source.kind == "search":
        raw = domain.cached_text(world.workdir, source.key).encode("utf-8")
        return raw[int(source.start or 0):int(source.end or 0)].decode("utf-8", errors="replace")
    return source.line


def fix(world: World, record: R.Scenario, why: str, model: Any) -> R.Scenario:
    """Rewrite the statement once from the same source with the reason; a rewrite that breaks the rule,
    or none at all, ends the record unsure with why."""
    body = _ask(model, FIX_PROMPT.format(statement=record.statement, why=why, source=source_text(world, record)))
    statement = " ".join(str(body.get("statement") or "").split())
    broken = statement_refusal(statement, world) if statement else EMPTY
    if broken:
        row = {"kind": "fix_failed", "why": why, "refused": broken}
        return record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
    row = {"kind": "fixed", "why": why, "was": record.statement}
    return record.model_copy(update={"statement": statement, "status": "born", "fixes": record.fixes + 1,
                                     "review": [*record.review, row]})


__all__ = ["BATCH", "QUERIES", "World", "byte_range", "checked", "coverage", "fix", "fold", "from_asks",
           "from_recordings", "from_search", "mine", "openings", "row_values", "search_pages", "source_text",
           "statement_refusal", "tool_rows", "world_of"]
