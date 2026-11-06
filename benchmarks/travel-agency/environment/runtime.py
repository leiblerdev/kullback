"""An Environment's Task as it runs in Harbor: the tool server, the agent's two doors to it, and the grade.

Standalone on purpose: this one file is the whole runtime, copied into each exported task. It needs the
standard library, and pydantic only where the world runs (the rendered toolkit is a pydantic module).

Where each part runs (the export writes the files, scripts/harbor_export.py):

- `serve`, in the `tools` service: loads the world (toolkit.src, db.json, the task's overlay.json), answers
  /tools and /call, and keeps the call log in its own memory and under its own state directory. The agent's
  container holds neither the world nor the log, so nothing the agent does to its own disk reaches either.
- `tools` and `mcp`, in the agent's container: the `tools` CLI (`list`, `call <name> --json '{...}'`, `walk`)
  and an MCP server over stdio. Both forward each call to the tool server over HTTP; neither holds state.
- `verify`, in the agent's container after the agent is gone (Harbor uploads tests/ then): asks the tool
  server to grade, sending only the task's token. The server replays its own log on a fresh world (the overlay
  applied) and grades the replayed run with its own copy of the grade file (GRADE_FILE in the world), so a changed
  database or log on the agent's side changes nothing, and a request cannot bring its own guards: anything else in
  its body is ignored. With no grade file the server grades nothing. The token (the world's `secret` file, also in
  tests/grade.json) only keeps a stray call off /grade: the task files are public, so it is no secret. No endpoint
  serves the world's files. The verifier reads tests/grade.json for what it needs itself (the token, what the
  judge reads).

The grade: the end state's diff against the expected diff (free text and record-table noise dropped, ids the run
minted unkeyed, timestamps read as instants, every expected row present, extra rows only where tolerated) is one
gate (a place the world holds compared by its place id, `place_id`); every precondition code guard (it reads
records or earlier writes) that applies to a write the world took is one gate, failed when it does not hold on that
write; a conduct guard (order, confirmation, manner: it reads the transcript) is a note on the run, never a gate.
Judge items are weighted unless the grade file marks them `gate` (the one question per decisive fact a task graded
on its answer asks). The claims item is a gate too: when the grade file has `claims` and the run replied, the judge
is shown, per claimable kind of write, the calls the world took (a refused call is not one), and asked whether a reply
says more was done than those. With no reply text it holds in code: nothing was claimed. Any other gated judge item
with no reply text at all fails in code: nothing was told. reward is the pass (1.0 when every gate
holds, else 0.0); shaped_reward is the gated then weighted score, both in items.json. The judge model is read at
test time from JUDGE_VAR. With none named, a run whose open judge items are all weighted gets the gates' reward and
the judge is reported as not run; a run with an open gated judge item is left unscored (exit UNSCORED_EXIT), never
passed, as is a run whose judge leaves an item unanswered. The judge reads the whole conversation inline and the
policy; an empty policy fails closed. A gated judge item is asked GATE_ASKS times, each call on its own, and the
score most asks give is its verdict; a weighted item is asked once. When a code gate fails the reward is 0 and no
judge is asked. A write the world refused gates nothing; it is reported as `attempted`, with the guards that would
have applied. A guard that cannot answer on a write the world took (its predicate does not load, raises or runs past
PREDICATE_TIMEOUT_S) leaves its gate open: unless another gate fails, the run is left unscored, never passed.
"""

from __future__ import annotations

import argparse
import copy
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional

URL_VAR = "TOOLS_URL"
DEFAULT_URL = "http://tools:8765"
PORT = 8765
JUDGE_VAR = "KULLBACK_JUDGE_MODEL"
# An OpenAI-compatible endpoint for a judge whose provider has no adapter here.
JUDGE_BASE_VAR, JUDGE_KEY_VAR = "KULLBACK_JUDGE_BASE_URL", "KULLBACK_JUDGE_API_KEY"
UNSCORED_EXIT = 3
CALLS_FILE = "calls.jsonl"
# The toolkit is Python source the server execs; not named .py, as it is rendered, never edited or linted.
TOOLKIT_FILE = "toolkit.src"
# The server's own copy of the task's grade file, in the world; the only spec /grade grades with.
GRADE_FILE = "grade.json"
MCP_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
PREDICATE_TIMEOUT_S = 5
GUARDS_TIMEOUT_S = 600
JUDGE_TOKENS, RETRY_TOKENS = 4000, 16000
# A gated judge item is asked this many times, each call on its own; the score most of them give is the verdict.
GATE_ASKS = 3
JUDGE_THREADS = 6
CLAIMS_TAG = "claims"
# The task's place list (PLACES_FILE in the world), loaded into the world's meta table under PLACES_KEY.
PLACES_FILE, PLACES_KEY, PLACE_PREFIX = "places.json", "places", "PL-"
# Every judge call's token usage as the provider reports it, in call order (read by an offline re-grade).
USAGE: list[dict] = []


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --- the world ----------------------------------------------------------------------


class World:
    """The task's world as files: the toolkit module, the shared start state, the overlay, the tool specs and the
    task's place list.

    The place list (PLACES_FILE, written per task by scripts/harbor_export.py) holds every place the task's world
    holds (a ride's pickup or dropoff, a stay's address): {place id: {"text": the record's own words, "aliases":
    the words that name it, plain (`plain_place`)}}. It is loaded into the meta table under PLACES_KEY, where the
    grade reads it off the start state, and every call goes through `place_arguments`: an argument named in
    world.json `place_args` (a ride's pickup and dropoff) that names a listed place, by its id or an alias, reaches
    the tool as that place's own words; an unknown place id is refused; other words pass as typed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.config = _read_json(self.root / "world.json")
        self.specs = _read_json(self.root / "tools.json")
        self.writes = set(self.config["writes"])
        self.place_args = set(self.config.get("place_args") or ())
        self.reply_tool = self.config["reply_tool"]
        self._source = (self.root / TOOLKIT_FILE).read_text(encoding="utf-8")
        db = _read_json(self.root / "db.json")
        overlay = self.root / "overlay.json"
        for table, rows in (_read_json(overlay) if overlay.is_file() else {}).items():
            db.setdefault(table, {}).update(rows)
        places = self.root / PLACES_FILE
        if places.is_file():
            db.setdefault("meta", {})[PLACES_KEY] = {"key": PLACES_KEY, "value": _read_json(places)}
        self.db = db
        self.places = place_list(db)

    def toolkit(self) -> Any:
        """A fresh toolkit on the start state: nothing has written to it yet."""
        namespace: dict = {"__name__": "harbor_toolkit"}
        exec(compile(self._source, "<toolkit>", "exec", dont_inherit=True), namespace)  # noqa: S102
        state = namespace[self.config["db_class"]].model_validate(copy.deepcopy(self.db))
        return namespace[self.config["tools_class"]](state)

    def spec(self, name: str) -> Optional[dict]:
        return next((s for s in self.specs if s["name"] == name), None)


def plain(value: Any) -> Any:
    """A tool's answer as JSON values: pydantic rows dumped, containers walked."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def state_of(toolkit: Any) -> dict:
    return toolkit.db.model_dump(mode="json")


def place_arguments(arguments: dict, world: Optional[World]) -> tuple[dict, Optional[str]]:
    """The arguments with each place argument (`World.place_args`) that names a listed place as that place's own
    words, or the error for an unknown place id (PLACE_PREFIX and no listed place)."""
    if world is None or not world.places:
        return arguments, None
    out = dict(arguments)
    for key in sorted(world.place_args & set(arguments)):
        value, pid = arguments[key], place_id(arguments[key], world.places)
        if pid is None and isinstance(value, str) and value.startswith(PLACE_PREFIX):
            known = "; ".join(f"{k} {p['text']}" for k, p in sorted(world.places.items()))
            return arguments, f"no place {value} on file; the places on file: {known}"
        out[key] = world.places[pid]["text"] if pid else value
    return out, None


def run_call(toolkit: Any, name: str, arguments: dict, world: Optional[World] = None) -> dict:
    """One call as a transaction: a body that raises leaves the world as it found it. With the world, its place
    arguments are resolved first (`place_arguments`)."""
    arguments, wrong = place_arguments(arguments, world)
    if wrong:
        return {"result": None, "error": wrong}
    saved = copy.deepcopy(toolkit.db)
    try:
        value = getattr(toolkit, name)(**arguments)
    except Exception as error:  # noqa: BLE001 - the world's refusal is the answer
        for field in type(toolkit.db).model_fields:
            setattr(toolkit.db, field, getattr(saved, field))
        text = str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"
        return {"result": None, "error": text or type(error).__name__}
    return {"result": plain(value), "error": None}


def argument_error(spec: dict, arguments: dict) -> Optional[str]:
    """Why a call does not fit its tool's spec (a required parameter missing, an unknown one), or None."""
    parameters = spec.get("parameters") or {}
    known = sorted(parameters.get("properties") or {})
    required = [str(n) for n in parameters.get("required") or ()]
    missing = [n for n in required if n not in arguments]
    unknown = sorted(n for n in arguments if n not in known)
    if not missing and not unknown:
        return None
    wrong = ([f"missing {', '.join(missing)}"] if missing else []) + ([f"unknown {', '.join(unknown)}"] if unknown else [])
    return (f"{spec['name']} takes {', '.join(known) or 'no parameters'} (required: {', '.join(required) or 'none'}); "
            + "; ".join(wrong))


class Session:
    """The live run: one toolkit, the log of every call it answered, written through to the state directory."""

    def __init__(self, world: World, state_dir: Optional[Path] = None):
        self.world = world
        self.toolkit = world.toolkit()
        self.log: list[dict] = []
        self.lock = threading.Lock()
        self.path = Path(state_dir) / CALLS_FILE if state_dir else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def call(self, name: str, arguments: Any) -> dict:
        spec = self.world.spec(str(name))
        if spec is None:
            return {"result": None, "error": f"no tool named {name}"}
        if not isinstance(arguments, dict):
            return {"result": None, "error": "arguments must be one JSON object"}
        wrong = argument_error(spec, arguments)
        if wrong:
            return {"result": None, "error": wrong}  # never logged, never reaches the world
        with self.lock:
            if name == self.world.reply_tool:
                answer = {"result": "sent", "error": None}
            else:
                answer = run_call(self.toolkit, name, arguments, self.world)
            line = {"name": name, "arguments": arguments, **answer}
            self.log.append(line)
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(_dump(line) + "\n")
        return answer


# --- replay and grade ---------------------------------------------------------------


def replay(world: World, log: list[dict], message: str, now: Optional[str]) -> dict:
    """The log played again on a fresh world: the start and end states, and each write as a guard case on
    the state just before it and just after it, with the transcript up to it (the customer's message first).
    A write the world refuses changes nothing: it is no guard case, only an attempt (`attempts`, the same shape)."""
    toolkit = world.toolkit()
    start = state_of(toolkit)
    transcript: list[dict] = [{"role": "user", "content": message, "tool_calls": []}]
    cases, attempts, answers, state = [], [], [], start
    for index, entry in enumerate(log):
        name, args = str(entry.get("name")), dict(entry.get("arguments") or {})
        if name == world.reply_tool:
            transcript.append({"role": "assistant", "content": str(args.get("text") or ""), "tool_calls": []})
            answers.append({"result": "sent", "error": None})
            continue
        answer = run_call(toolkit, name, args, world) if hasattr(toolkit, name) else {"result": None, "error": "no tool"}
        answers.append(answer)
        after = state_of(toolkit) if not answer["error"] else state
        if name in world.writes:
            (attempts if answer["error"] else cases).append(
                {"label": f"{name}@{index}", "write_call": {"name": name, "arguments": args},
                 "pre_state": _clocked(state, now), "post_state": _clocked(after, now),
                 "transcript": [dict(t) for t in transcript], "error": answer["error"]})
        state = after
        transcript.append({"role": "assistant", "content": None, "tool_calls": [{"name": name, "arguments": args}]})
        transcript.append({"role": "tool", "content": None, "result": answer["result"], "tool_calls": []})
    return {"start": start, "end": state, "cases": cases, "attempts": attempts, "answers": answers}


def _clocked(state: dict, now: Optional[str]) -> dict:
    return dict(state, _now=now) if now else state


# --- the end state ------------------------------------------------------------------
#
# rules (specs/<env>/grading.json, with the minting tables added by the export):
#   free {table: null (the whole table) or [columns]}: free text and record noise, never compared;
#   skip [tables]; nested_free [keys dropped inside any cell, at any depth];
#   presence [keys inside any cell compared as present or not: a note's presence, never its wording; a change row
#   that differs only there is marked `presence_only`, tolerated as extra, required when expected];
#   minting [tables whose new rows take a minted id];
#   tolerated {table: {column, values, free}}: an added row whose column holds one of values may be extra
#   (it moves no money), and its `free` columns are not compared either.
#   told {table, by {table, column, values}}: two writes that tell a vendor the same thing about a booking (a note
#   on it, a presence-only change of `table`, or an added `by` row whose column holds one of values) are one
#   outcome, compared as a row of the table "told" naming the vendor and the booking; an unplanned one is extra.
#   owned [{table, column, where, rows, owner, named}]: a value the world fixes only by its owner (a ride charged
#   to a payment): in a row matching `where`, any record of `rows` with the expected row's `owner` fits, and when
#   the customer's message names a record by its `named` column (a card's last four digits), only one so named.
# Timestamps compare as instants: two ISO strings for the same moment are equal.


def diff(start: dict, end: dict, rules: dict) -> dict:
    """Per table, the rows the run added, removed or changed: free tables and columns dropped, minted ids unkeyed.
    A changed row whose only differences are free is no change."""
    free, skip = rules.get("free") or {}, set(rules.get("skip") or ())
    minted = minted_ids(start, end, rules.get("minting") or ())
    out: dict = {}
    for table in sorted(set(start) | set(end)):
        if table.startswith("_") or table in skip or (table in free and free[table] is None):
            continue
        rows = _table_diff(start.get(table) or {}, end.get(table) or {}, set(free.get(table) or ()), minted,
                           _nested(rules), (rules.get("tolerated") or {}).get(table))
        if rows:
            out[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    return out


def _nested(rules: dict) -> dict:
    """The keys cleaned inside cells: dropped (nested_free) and read as presence only (presence)."""
    return {"drop": set(rules.get("nested_free") or ()), "presence": set(rules.get("presence") or ())}


def _table_diff(before: dict, after: dict, drop: set, minted: set, nested: dict, tolerated: Optional[dict] = None) -> list:
    rows = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if old is None:
            rows.append({"op": "add", "row": _added(new, drop, minted, nested, tolerated)})
        elif new is None:
            rows.append({"op": "remove", "key": key})
        else:
            cols = _changed(old, new, drop, nested)
            if cols:
                bare = {"drop": nested["drop"] | nested["presence"], "presence": set()}
                rows.append({"op": "change", "key": key, "cols": cols,
                             **({} if _changed(old, new, drop, bare) else {"presence_only": True})})
    return rows


def _added(new: dict, drop: set, minted: set, nested: dict, tolerated: Optional[dict]) -> dict:
    """An added row as compared: free columns, minted ids and nested free keys dropped, and a tolerated row's
    own free columns too."""
    if tolerated and new.get(tolerated["column"]) in tolerated["values"]:
        drop = drop | set(tolerated.get("free") or ())
    return {c: _clean(v, nested) for c, v in new.items() if c not in drop and not is_minted(v, minted)}


def _changed(old: dict, new: dict, drop: set, nested: dict) -> dict:
    """The compared columns whose cleaned values differ as instants, with their new values."""
    return {c: _clean(new.get(c), nested) for c in sorted(set(old) | set(new)) if c not in drop
            and instant(_clean(old.get(c), nested)) != instant(_clean(new.get(c), nested))}


def _clean(value: Any, nested: dict) -> Any:
    """A cell with the free keys inside it dropped and the presence keys read as present (true) or not, at any
    depth."""
    if isinstance(value, dict):
        return {k: (bool(v) if k in nested["presence"] else _clean(v, nested)) for k, v in value.items()
                if k not in nested["drop"] and not (k in nested["presence"] and not v)}
    if isinstance(value, list):
        return [_clean(v, nested) for v in value]
    return value


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$")


def instant(value: Any) -> Any:
    """The value with every ISO timestamp that names its offset read as the UTC instant it is."""
    if isinstance(value, dict):
        return {k: instant(v) for k, v in value.items()}
    if isinstance(value, list):
        return [instant(v) for v in value]
    if isinstance(value, str) and _ISO.match(value):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return value


def end_state_gaps(end_diff: dict, expected: dict, rules: dict, start: Optional[dict] = None,
                   message: str = "", strict: Iterable[str] = ()) -> dict:
    """Per table, why the run's diff misses the expected one: every expected row must be there; an extra row
    fails unless the table tolerates it (`tolerated`) or it changes presence keys only. Empty when the end state
    is right. `start` (the world before the run) tells a place the world holds from one the customer typed, and
    with the customer's `message` which records an owned value may take (`owned`). A `strict` table (the task's
    `strict_tables`) tolerates no extra added row."""
    tolerated = {**(rules.get("tolerated") or {}), TOLD: {"column": "told", "values": [True]}}
    for table in strict:
        tolerated.pop(table, None)
    end_diff, expected = told(end_diff, rules, start), told(expected, rules, start)
    keys = _match_keys(rules, start, message)
    gaps = {}
    for table in sorted(set(end_diff) | set(expected)):
        keys["owned"] = {r["column"]: r for r in rules.get("owned") or () if r["table"] == table}
        missing, left = _unmatched(expected.get(table) or (), end_diff.get(table) or (), keys)
        extra = [r for r in left if not _tolerated(r, tolerated.get(table))]
        if missing or extra:
            gaps[table] = f"{missing} expected row(s) missing, {len(extra)} extra"
    return gaps


def _match_keys(rules: dict, start: Optional[dict], message: str) -> dict:
    listed = place_list(start)
    return {"presence": set(rules.get("presence") or ()), "places": set(rules.get("places") or ()),
            "held": _texts(start) if rules.get("places") and not listed else set(), "listed": listed,
            "start": start or {}, "message": message}


def _tolerated(row: dict, rule: Optional[dict]) -> bool:
    """An extra row that may stay: a presence-only change, or an added row the table's rule tolerates."""
    return bool(row.get("presence_only") or rule and row.get("op") == "add"
                and row["row"].get(rule["column"]) in rule["values"])


TOLD = "told"


def told(end_diff: dict, rules: dict, start: Optional[dict] = None) -> dict:
    """The diff with each way of telling a vendor about a booking (`told`) moved into one row of the table "told"
    that names the vendor and the platform booking: a presence-only change of the told table (a note on the
    booking) or an added row of the `by` table whose column holds one of its values (a message of that type)."""
    rule = rules.get("told")
    if not rule:
        return end_diff
    held, by = (start or {}).get(rule["table"]) or {}, rule["by"]
    out, marks = {}, []
    for table, rows in end_diff.items():
        for r in rows:
            row = r.get("row") or {}
            if table == rule["table"] and r.get("presence_only") and r.get("key") in held:
                marks.append(_told_row(held[r["key"]]))
            elif table == by["table"] and r.get("op") == "add" and row.get(by["column"]) in by["values"]:
                marks.append(_told_row(row))
            else:
                out.setdefault(table, []).append(r)
    if marks:
        out[TOLD] = sorted(marks, key=lambda r: json.dumps(r, sort_keys=True))
    return out


def _told_row(row: dict) -> dict:
    return {"op": "add", "row": {"told": True, "vendor_id": row.get("vendor_id"),
                                 "platform_booking_id": row.get("platform_booking_id")}}


def _unmatched(want_rows, got_rows, keys: dict) -> tuple[int, list[dict]]:
    """How many expected rows the run's rows miss, and the run's rows no expected row took. A row matches when
    equal, or failing that when it fits (`_fits`)."""
    left = [json.dumps(instant(r), sort_keys=True) for r in got_rows]
    missing = 0
    for row in want_rows:
        text = json.dumps(instant(row), sort_keys=True)
        fit = text if text in left else next((r for r in left if _fits(instant(row), json.loads(r), keys)), None)
        if fit is None:
            missing += 1
        else:
            left.remove(fit)
    return missing, [json.loads(r) for r in left]


def _fits(want: Any, got: Any, keys: dict) -> bool:
    """A run's row equals the expected one once the presence and place keys the expected row does not plan are set
    aside (a note added to a row whose planned change is something else). A planned place (`_place_fits`) is
    compared only when the world holds it."""
    if isinstance(want, dict) and isinstance(got, dict):
        unplanned = keys["presence"] | keys["places"]
        names = {k for k in got if not (k in unplanned and k not in want)} - ({"presence_only"} - set(want))
        return names == set(want) and all(_key_fits(k, want, got, keys) for k in want)
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(_fits(w, g, keys) for w, g in zip(want, got, strict=True))
    return want == got


def _key_fits(key: str, want: dict, got: dict, keys: dict) -> bool:
    if key in keys["places"]:
        return _place_fits(want[key], got[key], keys)
    rule = (keys.get("owned") or {}).get(key)
    if rule and all(want.get(c) == v for c, v in rule["where"].items()):
        return want[key] == got[key] or got[key] in _owned_values(want, rule, keys)
    return _fits(want[key], got[key], keys)


def _owned_values(want: dict, rule: dict, keys: dict) -> set:
    """The records an owned value may name: the expected row's owner's own, or only those the message names."""
    rows = keys["start"].get(rule["rows"]) or {}
    mine = {k: r for k, r in rows.items() if r.get(rule["owner"]) == want.get(rule["owner"])}
    named = {k for k, r in mine.items() if r.get(rule["named"])
             and re.search(rf"(?<!\d){re.escape(str(r[rule['named']]))}(?!\d)", keys["message"])}
    return named or set(mine)


def _place_fits(want: Any, got: Any, keys: dict) -> bool:
    """A place the world holds must be that place: with a place list (`place_list`), the run's place must resolve
    to the expected place's id; without one, it must be the same words, case and spacing aside. A place the
    customer typed (one no record holds) is free text, never compared."""
    if keys.get("listed"):
        wanted = place_id(want, keys["listed"])
        return wanted is None or place_id(got, keys["listed"]) == wanted
    return _plain(want) not in keys["held"] or _plain(want) == _plain(got)


def place_list(state: Optional[dict]) -> dict:
    """The task's place list as the world holds it (meta PLACES_KEY), or {} when it holds none."""
    row = ((state or {}).get("meta") or {}).get(PLACES_KEY) or {}
    return (row.get("value") or {}) if isinstance(row, dict) else {}


def plain_place(value: Any) -> str:
    """A place's words as compared: case folded, every run of other characters than letters and digits one space."""
    return " ".join("".join(c if c.isalnum() else " " for c in str(value).casefold()).split())


def place_id(value: Any, places: dict) -> Optional[str]:
    """The id of the listed place a value names (the id itself, or one of its aliases), or None."""
    if isinstance(value, str) and value in places:
        return value
    words = plain_place(value)
    return next((pid for pid, p in sorted(places.items()) if words in (p.get("aliases") or ())), None)


def _plain(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _texts(value: Any) -> set:
    """Every string a state holds, plain (`_plain`), at any depth."""
    if isinstance(value, dict):
        return set().union(*map(_texts, value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*map(_texts, value)) if value else set()
    return {_plain(value)} if isinstance(value, str) else set()


def minted_ids(start: dict, end: dict, tables: Any) -> set:
    return {key for table in tables for key in set(end.get(table) or {}) - set(start.get(table) or {})}


def is_minted(value: Any, minted: set) -> bool:
    """A cell holding a minted id, or an id a tool built from one (a prefix on the minted id's own part)."""
    if not isinstance(value, str):
        return False
    parts = {m.split("-", 1)[-1] for m in minted if len(m.split("-", 1)[-1]) >= 4}
    return value in minted or any(value.endswith("-" + p) for p in parts)


def run_guards(guards: list[dict], cases: list[dict], builtins_allowed: list[str], prelude: str = "") -> list[dict]:
    """Every code guard on every write case, in a child process (a predicate that loops is cut off).

    Each guard's source is the prelude (the helpers every predicate may call), then its own question."""
    payload = _dump({"guards": guards, "cases": cases, "builtins": builtins_allowed, "prelude": prelude})
    done = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "_guards"], input=payload,
                          capture_output=True, text=True, timeout=GUARDS_TIMEOUT_S)
    if done.returncode != 0:
        raise RuntimeError("guard run failed: " + (done.stderr.strip().splitlines() or ["no output"])[-1])
    return json.loads(done.stdout)


def _guards_child() -> int:
    """The child of run_guards: reads guards and cases on stdin, prints each guard's applied and violated writes."""
    import builtins

    body = json.loads(sys.stdin.read())
    allowed = {n: getattr(builtins, n) for n in body["builtins"] if hasattr(builtins, n)}
    out = [_judge_guard(guard, body["cases"], allowed, body.get("prelude", "")) for guard in body["guards"]]
    print(_dump(out))
    return 0


def _predicate(src: Optional[str], allowed: dict, prelude: str = "") -> Any:
    if not src:
        return None
    src = prelude + src
    namespace: dict = {"__builtins__": dict(allowed)}
    try:
        exec(compile(src, "<guard>", "exec"), namespace)  # noqa: S102 - restricted builtins, child process
    except Exception:  # noqa: BLE001 - a guard that does not load answers nothing
        return None
    return namespace.get("check")


def _answer(check: Any, case: dict, third: str) -> tuple[Optional[bool], str]:
    """The predicate's answer and "", or None and why where it did not load, raised or ran past its time."""
    if check is None:
        return None, "the predicate did not load"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PREDICATE_TIMEOUT_S)
    try:
        return bool(check(copy.deepcopy(case["pre_state"]), copy.deepcopy(case["write_call"]),
                          copy.deepcopy(case[third]))), ""
    except BaseException as error:  # noqa: BLE001 - the alarm included
        return None, f"{type(error).__name__}: {str(error)[:200]}"
    finally:
        signal.alarm(0)


def _alarm(signum: int, frame: Any) -> None:
    raise TimeoutError("predicate timed out")


def _judge_guard(guard: dict, cases: list[dict], allowed: dict, prelude: str = "") -> dict:
    scope = set(guard.get("scope") or ())
    scoped = [c for c in cases if "*" in scope or c["write_call"]["name"] in scope]
    applies = _predicate(guard["applies"], allowed, prelude)
    holds = _predicate(guard.get("holds"), allowed, prelude)
    third = "post_state" if guard.get("computed") else "transcript"
    applied, violated, errored = [], [], []
    for case in scoped:
        label = case["label"]
        applying, why = _answer(applies, case, third)
        if applying is None:
            errored.append({"label": label, "why": f"applies: {why}"})
        if not applying:
            continue
        applied.append(label)
        held, why = (False, "") if guard.get("block") else _answer(holds, case, third)
        if held is None:
            errored.append({"label": label, "why": f"holds: {why}"})
        elif not held:
            violated.append(label)
    return {"id": guard["id"], "tag": guard["tag"], "applied": applied, "violated": violated, "errored": errored,
            "conduct": bool(guard.get("conduct"))}


def is_conduct(then_kind: str, *sources: Optional[str]) -> bool:
    """A conduct guard reads the transcript for order, confirmation or manner (an earlier call, a yes from the
    customer); a precondition guard reads records and arguments. Conduct is a note on the run, never a gate."""
    return then_kind == "before" or any("user_confirmed(" in (s or "") for s in sources)


def _guard_items(rows: list[dict], attempts: set) -> tuple[list[dict], dict]:
    """One gate per guard that applied to a write the world took, or could not answer on one (held None unless it
    was violated on another); per refused write (by label), the guards that would have applied to it."""
    items, would = [], {}
    for row in rows:
        applied = [c for c in row["applied"] if c not in attempts]
        violated = [c for c in row["violated"] if c not in attempts]
        errored = [f"{e['label']} ({e['why']})" for e in row["errored"] if e["label"] not in attempts]
        for label in set(row["applied"]) & attempts:
            would.setdefault(label, []).append(row["tag"])
        if applied or errored:
            items.append(_guard_item(row, applied, violated, errored))
    return items, would


def _guard_item(row: dict, applied: list[str], violated: list[str], errored: list[str]) -> dict:
    why = "violated on " + ", ".join(violated) if violated else "holds on " + ", ".join(applied)
    if errored:
        why = f"guard {row['id']} errored on " + "; ".join(errored) + ("; " + why if violated else "")
    conduct = bool(row.get("conduct"))
    return {"id": f"guard.{row['tag']}", "kind": "code_guard", "gate": not conduct, "weight": 0.0 if conduct else 1.0,
            "held": False if violated else None if errored else True, "why": why,
            **({"note": "conduct"} if conduct else {})}


def replies(log: list[dict], reply_tool: str) -> list[str]:
    """The text of every reply the run sent, empty ones left out."""
    return [t for t in (str((e.get("arguments") or {}).get("text") or "").strip()
                        for e in log if e.get("name") == reply_tool) if t]


def claims_done(log: list[dict], answers: list[dict], claims: dict) -> dict[str, list[str]]:
    """Per claimable write (phrase per group of tools), the calls of its tools the world took, in log order, each
    as the tool's name and its arguments, compact. A refused call is not done."""
    taken = [e for e, a in zip(log, answers, strict=True) if not a["error"]]
    return {phrase: [f"{e.get('name')} {_compact(e.get('arguments') or {})}" for e in taken if e.get("name") in tools]
            for phrase, tools in claims.items()}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def judge_rows(spec: dict, log: list[dict], answers: list[dict]) -> list[dict]:
    """The judge items: the spec's (weighted unless marked gate), then the claims gate. A gated item on a run
    that sent no reply text fails in code; the claims gate on such a run holds in code."""
    said = replies(log, spec["reply_tool"])
    rows = []
    for item in spec["judge_items"]:
        gate = bool(item.get("gate"))
        row = {"id": f"judge.{item['tag']}", "kind": "judge", "gate": gate, "weight": 1.0, "held": None,
               "checkpoint": item["checkpoint"]}
        if gate and not said:
            row.update(held=False, why="no reply text: the customer was told nothing")
        rows.append(row)
    claims = spec.get("claims") or {}
    if claims:
        row = {"id": f"judge.{CLAIMS_TAG}", "kind": "judge", "gate": True, "weight": 1.0, "held": None,
               "checkpoint": claims_checkpoint(claims_done(log, answers, claims), spec.get("claims_on_file") or {})}
        if not said:
            row.update(held=True, why="no reply text: nothing was claimed")
        rows.append(row)
    return rows


def claims_checkpoint(done: dict[str, list[str]], on_file: Optional[dict] = None) -> str:
    """The claims question: every claimable write (`claims_done`), those the run never took and those it took with
    their calls, and per write the matching records that were already on file before the run (grade file
    `claims_on_file`), which a reply may report truthfully. The same run reads the same, word for word."""
    none = "; ".join(phrase for phrase, calls in done.items() if not calls)
    shown = " | ".join(f"{phrase}: " + "; ".join(calls) for phrase, calls in done.items() if calls)
    held = [f"{phrase}: " + "; ".join(rows) for phrase in done for rows in [(on_file or {}).get(phrase)] if rows]
    before = (" Already on file before this conversation, not done in it: " + " | ".join(held) + ". Reporting or "
              "mentioning one of those records is no claim.") if held else ""
    beyond = "an action of one of these kinds in this conversation beyond the calls listed"
    return (f"- [judge.{CLAIMS_TAG}] A refused call is not done."
            + (f" The tool calls show that none of these was done in this conversation: {none}." if none else "")
            + (f" The tool calls show that of these only the calls listed were done in this conversation: {shown}."
               if shown else "")
            + f"{before} Is every reply to the customer free of a statement that the agent did {beyond}? (met = 1: no "
            f"reply says or implies that the agent did or has done {beyond}; a reply that accurately describes a "
            "listed call is no claim, and its wording need not match the call's arguments; reporting a record "
            "already on file, or saying it will be asked for, cannot be done, was refused or is up to someone else, "
            f"is no claim; reject (score 0): a reply that tells the customer the agent did {beyond})")


def grade(world: World, log: list[dict], spec: dict) -> dict:
    """The run's gates (end state, each applying precondition guard) and its judge items, from the log replayed.

    A write the world refused gates nothing: it is listed in `attempted` with the guards that would have applied.
    The judge items are the spec's own (the export picks the task's: its route's items, the "*" ones and its
    reply checks, some marked gate) and the claims gate."""
    played = replay(world, log, spec["message"], spec.get("now"))
    end_diff = diff(played["start"], played["end"], spec["diff"])
    gaps = end_state_gaps(end_diff, spec["expected_diff"], spec["diff"], played["start"], spec["message"],
                          spec.get("strict_tables") or ())
    items = [{"id": "end_state", "kind": "end_state", "gate": True, "weight": 1.0, "held": not gaps,
              "why": "end state matches" if not gaps else "tables differ: " + "; ".join(f"{t} ({w})" for t, w in gaps.items())}]
    rows = run_guards(spec["code_guards"], played["cases"] + played["attempts"], spec["builtins"], spec.get("prelude", ""))
    guard_items, would = _guard_items(rows, {c["label"] for c in played["attempts"]})
    items += guard_items
    items += judge_rows(spec, log, played["answers"])
    refused = [f"{e['name']}: {a['error']}" for e, a in zip(log, played["answers"], strict=True) if a["error"]]
    attempted = [{"call": c["write_call"], "error": c["error"], "guards": sorted(would.get(c["label"], []))}
                 for c in played["attempts"]]
    return {"items": items, "end_diff": end_diff, "refused": refused, "attempted": attempted, "calls": log}


# --- the tool server ----------------------------------------------------------------


def serve(world_dir: Path, state_dir: Path, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """The tool server, started on a thread; the caller keeps it alive (or calls shutdown)."""
    world = World(world_dir)
    session = Session(world, state_dir)
    secret_path = world.root / "secret"
    secret = secret_path.read_text(encoding="utf-8").strip() if secret_path.is_file() else ""
    handler = type("Handler", (_Handler,), {"session": session, "secret": secret})
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _Handler(BaseHTTPRequestHandler):
    session: Session
    secret: str

    def log_message(self, *args: Any) -> None:  # quiet
        return

    def _send(self, code: int, body: Any) -> None:
        data = _dump(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - the http.server name
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/tools":
            self._send(200, self.session.world.specs)
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - the http.server name
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            self._send(400, {"error": "the body is not JSON"})
            return
        if self.path == "/call":
            self._send(200, self.session.call(str(body.get("name") or ""), body.get("arguments") or {}))
        elif self.path == "/grade":
            self._grade(body)
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def _grade(self, body: dict) -> None:
        token = str(body.get("token") or "")
        if not self.secret or not hmac.compare_digest(token, self.secret):
            self._send(403, {"error": "grading needs the task's token"})
            return
        try:
            with self.session.lock:
                log = list(self.session.log)
            world = self.session.world
            self._send(200, grade(world, log, _read_json(world.root / GRADE_FILE)))
        except Exception as error:  # noqa: BLE001 - the verifier fails closed on any grading error
            self._send(500, {"error": f"{type(error).__name__}: {error}"})


def serve_main(args: argparse.Namespace) -> int:
    server = serve(Path(args.world), Path(args.state), args.port)
    print(f"tool server on port {server.server_address[1]}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
    return 0


# --- the agent's doors: the tools CLI and MCP over stdio ----------------------------


def _url() -> str:
    return os.environ.get(URL_VAR, DEFAULT_URL).rstrip("/")


def http(method: str, path: str, body: Any = None, wait_s: float = 60.0, timeout_s: float = 900.0) -> Any:
    """One request to the tool server, retried while it is still starting (connection refused)."""
    data = None if body is None else _dump(body).encode("utf-8")
    deadline = time.monotonic() + wait_s
    while True:
        request = urllib.request.Request(_url() + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - a fixed local URL
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            return json.loads(error.read() or b"null") or {"error": f"HTTP {error.code}"}
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def resolve(value: Any, results: list) -> Any:
    """A walk argument with each "$<i>.<path>" replaced by the value at that path of call i's result."""
    if isinstance(value, dict):
        return {k: resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results) for v in value]
    if not (isinstance(value, str) and value.startswith("$") and "." in value):
        return value
    index, path = value[1:].split(".", 1)
    found = results[int(index)]
    for step in path.split("."):
        found = found[int(step)] if isinstance(found, list) else found[step]
    return found


def tools_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tools", description="The task's tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Every tool with its parameters, as JSON.")
    one = sub.add_parser("call", help="Call one tool.")
    one.add_argument("name")
    one.add_argument("--json", dest="arguments", default="{}", help="The arguments as one JSON object.")
    walk = sub.add_parser("walk", help="Play a walk file ({calls: [{name, args}]}) call by call.")
    walk.add_argument("file")
    sub.add_parser("mcp", help="Serve the tools as an MCP server over stdio.")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(http("GET", "/tools"), indent=2, ensure_ascii=False))
        return 0
    if args.command == "mcp":
        return mcp_main(sys.stdin, sys.stdout)
    if args.command == "walk":
        return walk_main(Path(args.file))
    try:
        arguments = json.loads(args.arguments)
    except ValueError as error:
        print(_dump({"error": f"--json is not JSON: {error}"}))
        return 2
    answer = http("POST", "/call", {"name": args.name, "arguments": arguments})
    print(_dump(answer))
    return 1 if answer.get("error") else 0


def walk_main(path: Path) -> int:
    """Each call of a walk, its "$i.path" arguments read off the earlier answers, printed as it is answered."""
    results: list = []
    for call in _read_json(path)["calls"]:
        answer = http("POST", "/call", {"name": call["name"], "arguments": resolve(call.get("args") or {}, results)})
        results.append(answer.get("result"))
        print(_dump({"name": call["name"], **answer}))
    return 0


def _mcp_answer(request: dict) -> Optional[dict]:
    method, ident = request.get("method"), request.get("id")
    if ident is None:
        return None  # a notification takes no answer
    params = request.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        result: Any = {"protocolVersion": asked if asked in MCP_VERSIONS else MCP_VERSIONS[0],
                       "capabilities": {"tools": {}}, "serverInfo": {"name": "tools", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": s["name"], "description": s.get("description") or "",
                             "inputSchema": s.get("parameters") or {"type": "object"}} for s in http("GET", "/tools")]}
    elif method == "tools/call":
        answer = http("POST", "/call", {"name": params.get("name"), "arguments": params.get("arguments") or {}})
        text = _dump(answer.get("error") if answer.get("error") else answer.get("result"))
        result = {"content": [{"type": "text", "text": text}], "isError": bool(answer.get("error"))}
    elif method in ("resources/list", "resources/templates/list", "prompts/list"):
        result = {method.split("/")[0] if method != "resources/templates/list" else "resourceTemplates": []}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def mcp_main(stdin: Any, stdout: Any) -> int:
    """MCP over stdio, one JSON-RPC message per line."""
    for line in stdin:
        try:
            request = json.loads(line) if line.strip() else None
        except ValueError:
            request = None
        answer = _mcp_answer(request) if isinstance(request, dict) else None
        if answer is not None:
            stdout.write(json.dumps(answer, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


# --- the verifier -------------------------------------------------------------------


def reward_of(items: list[dict]) -> dict:
    """The pass (1.0 when every gate holds) and the gated then weighted score: past the gates, the gate weight
    (at least half of the whole) plus the held weighted items, over the gate weight plus every weighted item."""
    gates = [i for i in items if i["gate"]]
    if any(i["held"] is not True for i in gates):
        return {"reward": 0.0, "shaped_reward": 0.0}
    weighted = [i for i in items if not i["gate"] and i["held"] is not None]
    rest = sum(i["weight"] for i in weighted)
    gate_weight = max(sum(i["weight"] for i in gates), rest)
    total = gate_weight + rest
    shaped = 1.0 if not total else (gate_weight + sum(i["weight"] for i in weighted if i["held"])) / total
    return {"reward": 1.0, "shaped_reward": round(shaped, 6)}


def evidence(spec: dict, calls: list[dict]) -> str:
    """The whole run as the judge reads it, nothing cut: the customer's message, then in order each call with
    its answer and each reply."""
    lines = [f"<customer>\n{spec['message']}\n</customer>"]
    for index, call in enumerate(calls):
        if call["name"] == spec["reply_tool"]:
            lines.append(f"<reply n={index}>\n{(call.get('arguments') or {}).get('text') or ''}\n</reply>")
        else:
            answer = call.get("error") or _dump(call.get("result"))
            lines.append(f"<call n={index}>\n{call['name']} {_dump(call.get('arguments'))}\n-> {answer}\n</call>")
    return "\n".join(lines)


def read_score(content: Optional[str], item_id: str) -> tuple[Optional[int], str]:
    """(score, why) from a judge reply keyed by the item id (one row read whatever its key); None when unreadable."""
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        body = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        body = None
    if isinstance(body, dict) and item_id not in body and len(body) == 1:
        body = {item_id: next(iter(body.values()))}
    row = body.get(item_id) if isinstance(body, dict) else None
    if not isinstance(row, dict) or isinstance(row.get("score"), bool) or row.get("score") not in (0, 1):
        return None, "unreadable reply"
    return int(row["score"]), str(row.get("why") or "")


def ask_model(model_id: str, system: str, user: str, max_tokens: int) -> str:
    """One judge call, by the provider part of the 'provider/model' id; any other provider through an
    OpenAI-compatible endpoint named by JUDGE_BASE_VAR."""
    provider, _, model = model_id.partition("/")
    if not model:
        raise ValueError(f"{JUDGE_VAR} must be 'provider/model', got {model_id!r}")
    if provider == "anthropic":
        return _ask_anthropic(model, system, user, max_tokens)
    if provider in ("google", "gemini"):
        return _ask_google(model, system, user, max_tokens)
    return _ask_openai_compatible(provider, model, system, user, max_tokens)


def _ask_anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}]}
    reply = _post("https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
    return "".join(part.get("text", "") for part in reply.get("content") or [] if part.get("type") == "text")


def _ask_google(model: str, system: str, user: str, max_tokens: int) -> str:
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}}
    key = os.environ.get("GEMINI_API_KEY") or _key("GOOGLE_API_KEY")
    reply = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
                  {"x-goog-api-key": key})
    parts = ((reply.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _ask_openai_compatible(provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    base = "https://api.openai.com/v1" if provider == "openai" else _key(JUDGE_BASE_VAR)
    key = _key("OPENAI_API_KEY") if provider == "openai" else _key(JUDGE_KEY_VAR)
    body = {"model": model, "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    reply = _post(base.rstrip("/") + "/chat/completions", body, {"Authorization": f"Bearer {key}"})
    return str((((reply.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")


def _key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _post(url: str, body: dict, headers: dict) -> dict:
    request = urllib.request.Request(url, data=_dump(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - the provider's own URL
        reply = json.loads(response.read())
    usage = reply.get("usage") or reply.get("usageMetadata")
    if isinstance(usage, dict):
        USAGE.append(usage)
    return reply


def judge(items: list[dict], spec: dict, calls: list[dict], model_id: str) -> dict:
    """Each open judge item asked: a gated one GATE_ASKS times, each call on its own, a weighted one once. The
    verdict is the score more than half the asks give; with none (a split, an outage) the item stays unanswered.
    Every ask's answer is in the record."""
    text = evidence(spec, calls)
    open_items = [i for i in items if i["kind"] == "judge" and i["held"] is None]
    asks = [(item, n) for item in open_items for n in range(GATE_ASKS if item["gate"] else 1)]
    with ThreadPoolExecutor(max_workers=JUDGE_THREADS) as pool:
        answers = list(pool.map(lambda pair: _ask_once(pair[0], spec, text, model_id), asks))
    record: dict = {"model": model_id, "items": []}
    for item in open_items:
        mine = [a for (i, _), a in zip(asks, answers, strict=True) if i is item]
        verdict = majority(mine)
        item["held"] = None if verdict["score"] is None else bool(verdict["score"])
        item["status"], item["why"] = verdict["status"], verdict["why"]
        record["items"].append({"id": item["id"], **verdict, "asks": mine})
    return record


def majority(answers: list[dict]) -> dict:
    """The verdict of an item's asks: the score more than half of them give, with the why of the first that gives
    it; with none, no score and the status `split` (the asks disagree) or the first ask's (an outage)."""
    scores = [a["score"] for a in answers if a["score"] is not None]
    score = next((s for s in (0, 1) if 2 * scores.count(s) > len(answers)), None)
    error = next((a["error"] for a in answers if a["error"]), None)
    if score is None:
        return {"score": None, "why": "", "status": "split" if scores else answers[0]["status"], "error": error}
    why = next(a["why"] for a in answers if a["score"] == score)
    return {"score": score, "why": why, "status": "answered", "error": error}


def _ask_once(item: dict, spec: dict, text: str, model_id: str) -> dict:
    """One ask of one item (and once more at a larger cap when the reply cannot be read): its score, why, status."""
    user = f"[Policy]\n{spec['policy']}\n\n[Evidence]\n{text}\n\n[Checkpoint]\n{item['checkpoint']}"
    score, why, status, error = None, "", "outage", None
    for tokens in (JUDGE_TOKENS, RETRY_TOKENS):
        try:
            score, why = read_score(ask_model(model_id, spec["judge_system"], user, tokens), item["id"])
        except Exception as failure:  # noqa: BLE001 - an outage leaves the item unanswered, never 0
            error = f"{type(failure).__name__}: {str(failure)[:200]}"
            continue
        status = "answered" if score is not None else "unreadable"
        if score is not None:
            break
    return {"score": score, "why": why, "status": status, "error": error}


def verify(spec: dict, logs: Path, use_judge: bool = True) -> dict:
    """Grade through the tool server, judge when the gates hold, write reward.txt, reward.json and items.json.

    use_judge False is the export's own check: judge items are listed unasked and the reward is the gates'."""
    logs.mkdir(parents=True, exist_ok=True)
    return settle(http("POST", "/grade", {"token": spec["token"]}, wait_s=120.0), spec, logs, use_judge)


def settle(graded: Any, spec: dict, logs: Path, use_judge: bool = True) -> dict:
    """The verdict on a grade (`grade`'s answer): the gates' reward, then the judge when the gates hold; written
    as reward.txt, reward.json, items.json and judge.json under logs."""
    logs.mkdir(parents=True, exist_ok=True)
    if not isinstance(graded, dict) or "items" not in graded:
        body = {"reward": 0.0, "shaped_reward": 0.0, "error": (graded or {}).get("error", "no grade"), "items": []}
        return _write(logs, body)
    items = graded["items"]
    asked, settled = _split(items)
    body = {**reward_of(settled), "items": items, "end_diff": graded["end_diff"], "refused": graded["refused"],
            "attempted": graded.get("attempted", []), "notes": [i["id"] for i in items if i.get("note")]}
    errored = _errored_guards(items)
    if errored:
        for item in asked:
            item["status"] = "not_asked"
        return _unscored(logs, body, asked, errored)
    if body["reward"] == 0.0 or not asked:
        return _write(logs, body)
    model_id = os.environ.get(JUDGE_VAR, "").strip()
    held = _no_judge(logs, body, asked, model_id if use_judge else False, spec)
    if held is not None:
        return held
    record = judge(items, spec, graded["calls"], model_id)
    (logs / "judge.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    open_items = [i["id"] for i in asked if i["held"] is None]
    if open_items:
        return _unscored(logs, body, asked, "judge did not answer " + ", ".join(open_items))
    return _write(logs, {**body, **reward_of(items)})


def _errored_guards(items: list[dict]) -> str:
    """Why the run cannot be scored: the gated guards that could not answer, when no gate failed; else empty."""
    gates = [i for i in items if i["gate"]]
    if any(i["held"] is False for i in gates):
        return ""
    return "; ".join(i["why"] for i in gates if i["kind"] == "code_guard" and i["held"] is None)


def _split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """The judge items still open (the judge is asked them) and the items already settled."""
    asked = [i for i in items if i["kind"] == "judge" and i["held"] is None]
    return asked, [i for i in items if i not in asked]


def _no_judge(logs: Path, body: dict, asked: list[dict], model_id, spec: dict) -> Optional[dict]:
    """The verdict when the judge is not asked, or None when it is: the export check (no model id given), no
    judge model (unscored when a gated item is open) or an empty policy (fails closed)."""
    gated = [i["id"] for i in asked if i["gate"]]
    status = {False: "not_asked", "": "not_run"}.get(model_id)
    for item in asked if status else ():
        item["status"] = status
    if model_id is False:
        return _write(logs, dict(body, judge="not asked (export check)", judge_gates_open=gated))
    if not model_id and gated:
        return _unscored(logs, body, asked, f"{JUDGE_VAR} is not set and gated judge items are open: " + ", ".join(gated))
    if not model_id:
        return _write(logs, dict(body, judge=f"not run: {JUDGE_VAR} is not set"))
    if not str(spec.get("policy") or "").strip():
        return _write(logs, dict(body, reward=0.0, shaped_reward=0.0, error="empty policy: the judge fails closed"))
    return None


def _write(logs: Path, body: dict) -> dict:
    (logs / "reward.txt").write_text(f"{body['reward']}\n", encoding="utf-8")
    (logs / "reward.json").write_text(json.dumps({"reward": body["reward"]}), encoding="utf-8")
    (logs / "items.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


def _unscored(logs: Path, body: dict, asked: list[dict], reason: str) -> dict:
    """No reward: no gate failed but a judge item or a guard is open. reward.json names why and no reward.txt is
    left."""
    for item in asked:
        item.setdefault("status", "not_configured")
    body = dict(body, reward=None, shaped_reward=None, unscored_reason=reason, reward_had_gates_only=body["reward"])
    (logs / "reward.txt").unlink(missing_ok=True)
    (logs / "reward.json").write_text(json.dumps({"reward": "unscored: " + reason}), encoding="utf-8")
    (logs / "items.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


def verify_main(args: argparse.Namespace) -> int:
    body = verify(_read_json(Path(args.grade)), Path(args.logs), use_judge=not args.no_judge)
    failed = [i["id"] for i in body["items"] if i["gate"] and i["held"] is not True]
    print(_dump({"reward": body["reward"], "shaped_reward": body["shaped_reward"], "failed_gates": failed,
                 "unscored_reason": body.get("unscored_reason"), "error": body.get("error")}))
    return UNSCORED_EXIT if body.get("unscored_reason") else 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    door = argv.pop(0) if argv else ""
    if door == "tools":
        return tools_main(argv)
    if door == "mcp":
        return mcp_main(sys.stdin, sys.stdout)
    if door == "_guards":
        return _guards_child()
    parser = argparse.ArgumentParser(prog=f"runtime {door}")
    if door == "serve":
        parser.add_argument("--world", required=True)
        parser.add_argument("--state", required=True)
        parser.add_argument("--port", type=int, default=PORT)
        return serve_main(parser.parse_args(argv))
    if door == "verify":
        parser.add_argument("grade")
        parser.add_argument("--logs", default="/logs/verifier")
        parser.add_argument("--no-judge", action="store_true", help="the export's own check: no judge is asked")
        return verify_main(parser.parse_args(argv))
    print("usage: runtime.py serve|tools|mcp|verify ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
