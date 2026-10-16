"""The synthetic generator's miner: scenarios from the recordings, a cited search and a customer's lines (D372).

The miner is a role on the shared agent harness and event stream, as the Examiner and the scenario
review are (mine_tools.py): one session per batch of one source, on the workdir bus under the agent
name synth, its calls booked under the stage synth_mining by the budget subscriber. The recordings
source hands a session a batch of opening asks, one per recording, and the agent groups them into
statements; frequency is the count of openings grouped. The search source hands one session the
harness's web reader as the shared web tools (D375) over search.Crawled: the agent searches, reads pages
through the D225 crawl, and writes a statement only with the span of a page it read, whose byte range
becomes the citation. The ask source
hands a session a batch of lines a customer typed and the agent rewrites each as a scenario in the third
person; the line as typed stays in the source only, since the customer's own shape would tell the blind
review where it came from. A line naming a tool is dropped before any session.

The rule lives in the write tools, which refuse with the reason so the agent writes again; a second
refusal drops the scenario, counted per reason. Repeats fold by code only, statements equal up to case,
punctuation and spacing, recordings first, so a search or ask scenario the recordings already show keeps
the recordings record and adds its own source under `also_from`; no model judges a repeat. The agent
lists what is already written before it writes, and the review may rule a scenario a repeat of another. A record keeps the id it was born with,
also after a fix rewrites its statement. The fix step is a session too, under the stage synth_fix.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from kullback import domain
from kullback.runner import budget
from kullback.runner.records import write_json
from kullback.spec import intent_tools
from kullback.synth import mine_prompt
from kullback.synth import mine_tools as T
from kullback.synth import records as R
from kullback.synth import search as search_mod
from kullback.synth.rule import EMPTY, ID_SHAPE, TOOL_NAME, World, statement_refusal, world_of

# How many opening asks or typed lines one session mines.
BATCH = 25
STAGE = "synth_mining"
FIX_STAGE = "synth_fix"
CEILING_NOTE = "stopped at the spend ceiling; mine again to continue"
_PUNCT = re.compile(r"[^\w\s]")


@dataclass
class Mining:
    """One mining call: the world, the model, what every session found and dropped, the stream."""

    world: World
    model: Any
    found: list = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    notes: list = field(default_factory=list)
    bus: Any = None
    ceiling: Optional[budget.Ceiling] = None
    subscribers: tuple = ()
    capped: Counter = field(default_factory=Counter)

    def over(self) -> bool:
        if self.ceiling is not None and self.ceiling.remaining() <= 0:
            if CEILING_NOTE not in self.notes:
                self.notes.append(CEILING_NOTE)
            return True
        return False

    def session(self, kind: str, opening: str, stage: str = STAGE, **root: Any) -> T.MineRoot:
        made = T.MineRoot(world=self.world, kind=kind, found=self.found, dropped=self.dropped, bus=self.bus, **root)
        if T.run(made, self.model, opening, stage, self.ceiling, self.subscribers):
            self.capped[kind] += 1
        return made


def _entries(workdir: Path) -> list[str]:
    folder = R.store_dir(workdir)
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


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


def _batched(mining: Mining, kind: str, items: list[dict], batch: int) -> None:
    """One session per batch of items of one source, stopping at the ceiling."""
    size = max(int(batch), 1)
    for start in range(0, len(items), size):
        if mining.over():
            return
        chunk = items[start:start + size]
        opening = mine_prompt.opening(kind, _entries(mining.world.workdir), count=len(chunk))
        mining.session(kind, opening, items=chunk)


def from_recordings(mining: Mining, *, batch: int = BATCH) -> None:
    """Scenarios grouping the recordings' opening asks, one session per batch."""
    if mining.model is None:
        return
    _batched(mining, "recordings", openings(mining.world.workdir), batch)


# --- the search -------------------------------------------------------------------------


def from_search(mining: Mining, *, engine: Any = None, env: Optional[dict] = None,
                corpus_url: Optional[str] = None, exclude: Iterable[str] = (),
                refuse: Optional[Callable[[str], str]] = None) -> None:
    """Scenarios read off pages the harness's reader names, each citing the byte range of its span.

    With nothing live and no memo there is no reader: no session runs, no page is read, and the note
    names the switch.
    """
    if mining.model is None:
        return
    if not mining.world.description:
        mining.notes.append("no one-line description of the business, so no search was run")
        return
    engine = engine if engine is not None else search_mod.reader(mining.world.workdir, env)
    if engine is None:
        mining.notes.append(search_mod.OFF_NOTE)
        return
    if mining.over():
        return
    pages: dict = {}
    reader = search_mod.Crawled(engine, mining.world.workdir, keys=pages, corpus_url=corpus_url, exclude=exclude,
                                notes=mining.notes)
    opening = mine_prompt.opening("search", _entries(mining.world.workdir))
    mining.session("search", opening, reader=reader, pages=pages, refuse=refuse or domain.destination_refusal)


# --- the customer's lines ---------------------------------------------------------------


def ask_lines(path: Any) -> list[str]:
    target = Path(path)
    if not target.is_file():
        return []
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def from_asks(mining: Mining, lines: Iterable[str], *, batch: int = BATCH) -> None:
    """Each line a customer typed, rewritten as a scenario under the rule, one session per batch.

    A line naming a tool of the Environment is dropped and counted before any session.
    """
    kept = []
    for line in lines:
        why = statement_refusal(line, mining.world, tools_only=True)
        if why:
            mining.dropped[why] += 1
        else:
            kept.append({"text": line})
    if mining.model is None:
        if kept:
            mining.notes.append("no model, so no ask line was rewritten as a scenario")
        return
    _batched(mining, "ask", kept, batch)


# --- folding and coverage ---------------------------------------------------------------


def canonical(text: str) -> str:
    """A statement as the fold compares it: lowercase, punctuation dropped, spacing collapsed."""
    return " ".join(_PUNCT.sub(" ", str(text or "").lower()).split())


def fold(records: list[R.Scenario]) -> tuple[list[R.Scenario], list[dict]]:
    """The records with repeats folded in by code, in the order given, and the folds.

    Two statements fold when they are equal up to case, punctuation and spacing; no model judges it.
    The first record keeps the others' sources under also_from, and two recordings records add up.
    """
    first: dict[str, R.Scenario] = {}
    folded: list[dict] = []
    for record in records:
        held = first.setdefault(canonical(record.statement), record)
        if held is record:
            continue
        held.also_from += [record.source, *record.also_from]
        if held.source.kind == record.source.kind == "recordings":
            held.frequency += record.frequency
            held.source.task_ids = sorted({*held.source.task_ids, *record.source.task_ids})
        folded.append({"id": record.id, "into": held.id, "kind": record.source.kind})
    return list(first.values()), folded


def coverage(kept: list[R.Scenario], folded: list[dict], dropped: Counter) -> dict:
    """Scenarios per source kind, what the recordings never show, the asks nothing else covers, drops."""
    per_kind = Counter(record.source.kind for record in kept)
    return {"per_kind": {kind: per_kind.get(kind, 0) for kind in R.SOURCE_KINDS},
            "search_not_in_recordings": per_kind.get("search", 0),
            "asks_not_covered": per_kind.get("ask", 0),
            "folded": Counter(row["kind"] for row in folded),
            "dropped": dict(sorted(dropped.items()))}


def mine(workdir: Any, *, model: Any, asks: Any = None, searching: bool = True,
         batch: int = BATCH, description: str = "", engine: Any = None, env: Optional[dict] = None,
         corpus_url: Optional[str] = None, exclude: Iterable[str] = (), ceiling: Optional[budget.Ceiling] = None,
         subscribers: Iterable[Any] = (), refuse: Optional[Callable[[str], str]] = None) -> dict:
    """Run the three sources as sessions, fold the repeats, write the new records and the counts.

    A record already in the store keeps its status and its review: mining again adds, never resets.
    """
    world = world_of(workdir, description)
    mining = Mining(world=world, model=model, bus=T.bus_for(workdir), ceiling=ceiling,
                    subscribers=tuple(subscribers))
    from_recordings(mining, batch=batch)
    if searching:
        from_search(mining, engine=engine, env=env, corpus_url=corpus_url, exclude=exclude, refuse=refuse)
    else:
        mining.notes.append("search turned off")
    path = Path(asks) if asks else R.store_dir(workdir) / R.ASKS
    from_asks(mining, ask_lines(path), batch=batch)
    kept, folded = fold(list(mining.found))
    held = {canonical(record.statement) for record in R.load_scenarios(workdir)}
    new = [record for record in kept if canonical(record.statement) not in held]
    R.write_all(workdir, new)
    body = {"format": R.FORMAT, "notes": mining.notes, "coverage": coverage(kept, folded, mining.dropped),
            "born": len(new), "folded": folded, "capped": dict(sorted(mining.capped.items()))}
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


def _fix_items(world: World, sent: list[tuple[R.Scenario, str]]) -> str:
    return "\n\n".join(f"{record.id}: {record.statement}\nreason: {why}\nwritten from: {source_text(world, record)}"
                       for record, why in sent)


def _fixed(record: R.Scenario, why: str, statement: Optional[str]) -> R.Scenario:
    if not statement:
        row = {"kind": "fix_failed", "why": why}
        return record.model_copy(update={"status": "unsure", "review": [*record.review, row]})
    row = {"kind": "fixed", "why": why, "was": record.statement}
    return record.model_copy(update={"statement": statement, "status": "born", "fixes": record.fixes + 1,
                                     "review": [*record.review, row]})


def fix(world: World, sent: list[tuple[R.Scenario, str]], model: Any, *, bus: Any = None,
        ceiling: Optional[budget.Ceiling] = None, subscribers: Iterable[Any] = ()) -> list[R.Scenario]:
    """Each scenario sent back, rewritten once from the same source with the reason, in one session.

    A scenario the session does not rewrite under the rule ends unsure, with why.
    """
    if not sent:
        return []
    mining = Mining(world=world, model=model, bus=bus if bus is not None else T.bus_for(world.workdir),
                    ceiling=ceiling, subscribers=tuple(subscribers))
    if model is None or mining.over():
        return [_fixed(record, why, None) for record, why in sent]
    opening = mine_prompt.opening("fix", _entries(world.workdir), items=_fix_items(world, sent))
    root = mining.session("fix", opening, stage=FIX_STAGE, fixing={record.id: record for record, _ in sent})
    return [_fixed(record, why, root.fixed.get(record.id)) for record, why in sent]


__all__ = ["BATCH", "EMPTY", "FIX_STAGE", "ID_SHAPE", "Mining", "STAGE", "TOOL_NAME", "World", "canonical", "coverage",
           "fix", "fold", "from_asks", "from_recordings", "from_search", "mine", "openings", "source_text",
           "statement_refusal", "world_of"]
