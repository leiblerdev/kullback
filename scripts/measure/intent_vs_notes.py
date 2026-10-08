"""Scores the intent miner against the benchmark's own user note. Measurement only.

The note (each raw task's `user_scenario`: persona and the instructions' reason_for_call,
known_info, unknown_info, task_instructions) is never an input to the miner: `mine` runs the miner
and never opens the raw files; `score` reads the note afterwards and prints counts, never text.

    python scripts/measure/intent_vs_notes.py mine  <workdir> --split vf_split.json --corpus c1 \
        --n 20 --seed 0 --exclude <pilot ids> --ceiling-usd 0.25 --total-usd 3 --out mine.json
    python scripts/measure/intent_vs_notes.py score <workdir> --split vf_split.json --corpus c1 \
        --n 20 --seed 0 --exclude <pilot ids> [--witness-share 0.6 --carried-share 0.5] --out score.json

Only the split's dev ids are read; a test id is refused. `--n` draws that many dev ids with `--seed`
after `--exclude` drops ids; mine and score draw the same ids from the same arguments.
The pilot ids, which shaped the miner's prompt in round 0 and are left out of every scored draw:
task_018a626ff6c9, task_01f6b83d556b, task_023e88aadca9 (corpus c1).

`--witness-share` and `--carried-share` re-apply the miner's two code checks to Intents already
mined: witnesses are found again at the given share, and a kept fact its own turn carries below the
carried share is left out, as the miner would have refused it. A carried share below the one the
miner ran with changes nothing, since those facts were refused and never kept. A workdir Task maps to its raw task ids
through its recordings' sidecar rows (grader/<hash>.json: trace_id and fields.task_id).

Per Task, with the one word reading the miner's own witness check uses (content words, normalised):
- recall: a note sentence is recalled when half its content words are in one IntentFact; split by
  note field, since persona and unknown_info say what a user never volunteers.
- precision: an IntentFact is in the note when half its words are in the note's words.
- rule 5: accepted facts with no volunteering witness, in the note and not.
- steering leak: an IntentFact whose words the agent said before any user turn of its source
  recording said them.
Rates carry their denominators and Wilson 95% intervals. Output holds ids and counts only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kullback.spec.intent_tools import (  # noqa: E402
    CARRIED_SHARE,
    WITNESS_SHARE,
    TaskRecordings,
    carried_share,
    content_tokens,
    counts,
    recordings_of,
    share_in,
    task_of,
    trace_index,
    witnesses,
)
from kullback.spec.schema import IntentFact, SpecIntent, load_spec  # noqa: E402

MATCH_SHARE = 0.5
NOTE_FIELDS = ("reason_for_call", "known_info", "unknown_info", "task_instructions")
# The fields a user is expected to say; persona and unknown_info are scored but kept apart.
SAID_FIELDS = ("reason_for_call", "known_info", "task_instructions")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# --- shared readers ------------------------------------------------------------------------------

def dev_ids(split_path: Path, corpus: str) -> list[str]:
    split = json.loads(Path(split_path).read_text())
    return list((split.get("dev") or {}).get(corpus) or [])


def draw(ids: list[str], n: Optional[int], seed: int, exclude: set[str]) -> list[str]:
    """n dev ids drawn with a fixed seed after the excluded ids are dropped; all of them without n."""
    pool = sorted(set(ids) - exclude)
    return pool if n is None or n >= len(pool) else sorted(random.Random(seed).sample(pool, n))


def test_ids(split_path: Path, corpus: str) -> set[str]:
    split = json.loads(Path(split_path).read_text())
    return set((split.get("test") or {}).get(corpus) or [])


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list[float]]:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


# --- the note (score only) -----------------------------------------------------------------------

def raw_tasks(workdir: Path) -> dict[str, dict]:
    """Every raw task by id, from every raw file under workdir/raw."""
    out: dict[str, dict] = {}
    for path in sorted((Path(workdir) / "raw").glob("*.json")):
        body = json.loads(path.read_text())
        for task in (body.get("tasks") or []) if isinstance(body, dict) else []:
            out[str(task.get("id"))] = task
    return out


def trace_to_raw_task(workdir: Path) -> dict[str, str]:
    """trace_id -> the raw task id its sidecar row names."""
    out: dict[str, str] = {}
    for path in sorted((Path(workdir) / "grader").glob("*.json")):
        body = json.loads(path.read_text())
        raw_id = (body.get("fields") or {}).get("task_id")
        if body.get("trace_id") and raw_id is not None:
            out[str(body["trace_id"])] = str(raw_id)
    return out


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [t for item in value for t in _texts(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _texts(item)]
    return [str(value)]


def note_sentences(scenario: dict) -> dict[str, list[set[str]]]:
    """Field -> the content words of each sentence of that field of one note."""
    instructions = scenario.get("instructions") or {}
    if isinstance(instructions, str):
        instructions = {"task_instructions": instructions}
    fields = {"persona": scenario.get("persona"), **{f: instructions.get(f) for f in NOTE_FIELDS}}
    out: dict[str, list[set[str]]] = {}
    for name, value in fields.items():
        rows = [content_tokens(s) for text in _texts(value) for s in SENTENCE_RE.split(text)]
        out[name] = [words for words in rows if words]
    return out


# --- per Task scoring ------------------------------------------------------------------------------

def _leaked(fact: Any, trace: Any) -> bool:
    """Whether an agent turn said the fact's words before any user turn of the recording did."""
    words = content_tokens(fact.text)
    for turn in trace.turns if trace is not None else ():
        if turn.role not in ("user", "assistant") or not turn.content:
            continue
        if share_in(words, content_tokens(turn.content)) >= MATCH_SHARE:
            return turn.role == "assistant"
    return False


def _sentences(notes: list[dict]) -> dict[str, list[set[str]]]:
    """Field -> sentence word sets, over the union of a Task's notes."""
    sentences: dict[str, list[set[str]]] = {}
    for scenario in notes:
        for name, rows in note_sentences(scenario).items():
            sentences.setdefault(name, []).extend(rows)
    return sentences


def _recalled(sentences: dict[str, list[set[str]]], fact_words: list[set[str]]) -> dict[str, int]:
    return {name: sum(1 for s in rows if any(share_in(s, fw) >= MATCH_SHARE for fw in fact_words))
            for name, rows in sentences.items()}


@dataclass(frozen=True)
class Shares:
    """The two word shares the miner's code checks use, as the score re-applies them."""
    witness: float = WITNESS_SHARE
    carried: float = CARRIED_SHARE


def _carried(fact: IntentFact, found: TaskRecordings) -> float:
    """The share of a kept fact's words its own turn carries; 0 when the turn is gone."""
    turn = next((t for t in found.recordings.get(fact.source.recording) or () if t.index == fact.source.turn), None)
    return carried_share(fact.text, turn) if turn is not None else 0.0


def _fact_rows(facts: list[IntentFact], note_words: set[str], traces: dict, found: TaskRecordings,
               shares: Shares) -> list[dict]:
    """Per fact kept under the carried share: in the note, rule 5 marked under the witness share, leaked."""
    rows = []
    for f in facts:
        seen = bool(witnesses(f.text, f.source.recording, found, shares.witness))
        rows.append({"in_note": share_in(content_tokens(f.text), note_words) >= MATCH_SHARE,
                     "marked": f.stance == "accepted" and not seen, "volunteered": f.stance == "volunteered",
                     "witnessed": seen, "leaked": _leaked(f, traces.get(f.source.recording))})
    return rows


def score_task(intent: SpecIntent, notes: list[dict], traces: dict, found: Optional[TaskRecordings] = None,
               shares: Optional[Shares] = None) -> dict:
    """Counts for one Task against the union of its notes."""
    shares = shares or Shares()
    found = found if found is not None else TaskRecordings(task_id=intent.task_id)
    sentences = _sentences(notes)
    note_words = set().union(*[w for rows in sentences.values() for w in rows])
    kept = [f for f in intent.facts if not found.recordings or _carried(f, found) >= shares.carried]
    facts = _fact_rows(kept, note_words, traces, found, shares)

    def n(**want: bool) -> int:
        return sum(all(row[k] == v for k, v in want.items()) for row in facts)

    return {
        "facts": len(facts), "volunteered": n(volunteered=True), "accepted": n(volunteered=False),
        "with_witness": n(witnessed=True), "dropped_by_carried": len(intent.facts) - len(kept),
        "refusals": intent.refusals,
        "note_sentences": {name: len(rows) for name, rows in sentences.items()},
        "recalled": _recalled(sentences, [content_tokens(f.text) for f in kept]),
        "facts_in_note": n(in_note=True),
        "unwitnessed_accepted_in_note": n(marked=True, in_note=True),
        "unwitnessed_accepted_not_in_note": n(marked=True, in_note=False),
        "leaked": n(leaked=True), "leaked_volunteered": n(leaked=True, volunteered=True),
    }


def _sum(rows: list[dict], key: str, sub: Optional[tuple[str, ...]] = None) -> int:
    if sub is None:
        return sum(int(r.get(key) or 0) for r in rows)
    return sum(int((r.get(key) or {}).get(name) or 0) for r in rows for name in sub)


def aggregate(rows: list[dict]) -> dict:
    """The corpus rates, each with its denominator and Wilson interval."""
    said = SAID_FIELDS
    facts = _sum(rows, "facts")
    unwitnessed = _sum(rows, "unwitnessed_accepted_in_note") + _sum(rows, "unwitnessed_accepted_not_in_note")
    out = {
        "tasks": len(rows),
        "facts": facts,
        "refusals": _sum(rows, "refusals"),
        "dropped_by_carried": _sum(rows, "dropped_by_carried"),
        "witnessed": _sum(rows, "with_witness"),
        "recall_said_fields": rate(_sum(rows, "recalled", said), _sum(rows, "note_sentences", said)),
        "precision": rate(_sum(rows, "facts_in_note"), facts),
        "volunteered_share": rate(_sum(rows, "volunteered"), facts),
        "witnessed_share": rate(_sum(rows, "with_witness"), facts),
        "unwitnessed_accepted_in_note": rate(_sum(rows, "unwitnessed_accepted_in_note"), unwitnessed),
        "steering_leak": rate(_sum(rows, "leaked"), facts),
        "steering_leak_volunteered": rate(_sum(rows, "leaked_volunteered"), _sum(rows, "volunteered")),
    }
    for name in ("persona", *NOTE_FIELDS):
        out[f"recall_{name}"] = rate(_sum(rows, "recalled", (name,)), _sum(rows, "note_sentences", (name,)))
    return out


def score(workdir: Path, task_ids: list[str], shares: Optional[Shares] = None) -> dict:
    """Per Task counts and the corpus rates, for the Tasks that have a mined Intent."""
    shares = shares or Shares()
    raw = raw_tasks(workdir)
    by_trace = trace_to_raw_task(workdir)
    traces = trace_index(workdir)
    rows = []
    for task_id in task_ids:
        spec = load_spec(workdir, task_id)
        if spec is None:
            continue
        raw_ids = sorted({by_trace[r] for r in task_of(workdir, task_id).run_ids if r in by_trace})
        notes = [raw[i].get("user_scenario") or {} for i in raw_ids if i in raw]
        found = recordings_of(workdir, task_id, traces)
        rows.append({"task_id": task_id, "raw_tasks": len(notes),
                     **score_task(spec.intent, notes, traces, found, shares)})
    return {"shares": {"witness": shares.witness, "carried": shares.carried},
            "rates": aggregate(rows), "per_task": rows}


# --- mining (never reads the note) ---------------------------------------------------------------

def mine(workdir: Path, task_ids: list[str], model_id: str, ceiling_usd: float, total_usd: float,
         workers: int = 4) -> dict:
    """Mine each Task in turn until the total is spent; one row of counts per Task."""
    from kullback.ai.provider import live_model
    from kullback.spec.intent import mine_intent

    model = live_model(model_id)
    traces = trace_index(workdir)
    spent = {"usd": 0.0}
    rows: list[dict] = []

    def one(task_id: str) -> Optional[dict]:
        if spent["usd"] >= total_usd:
            return None
        started = time.time()
        out = mine_intent(workdir, task_id, model, ceiling_usd=min(ceiling_usd, total_usd - spent["usd"]),
                          traces=traces)
        spent["usd"] += out.spent_usd
        facts = out.intent.facts
        return {"task_id": task_id, "tool_calls": out.tool_calls, "spent_usd": round(out.spent_usd, 4),
                "stopped": out.stopped, "facts": len(facts), "refusals": out.refusals,
                "volunteered": sum(f.stance == "volunteered" for f in facts),
                "accepted": sum(f.stance == "accepted" for f in facts),
                "with_witness": sum(bool(f.witnesses) for f in facts),
                "counted": sum(counts(f) for f in facts), "seconds": round(time.time() - started, 1)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for row in pool.map(one, task_ids) if row is not None]
    return {"model": model_id, "spent_usd": round(spent["usd"], 4), "tasks": rows}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("mine", "score"))
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--corpus", required=True, help="the split's corpus label")
    parser.add_argument("--n", type=int, default=None, help="draw n dev ids with --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", nargs="*", default=[], help="dev ids to leave out (the pilot ids)")
    parser.add_argument("--witness-share", type=float, default=WITNESS_SHARE)
    parser.add_argument("--carried-share", type=float, default=CARRIED_SHARE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ceiling-usd", type=float, default=0.25)
    parser.add_argument("--total-usd", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = draw(dev_ids(args.split, args.corpus), args.n, args.seed, set(args.exclude))
    if set(ids) & test_ids(args.split, args.corpus):
        parser.error("a test id is in the dev list; the sealed test ids are never read")
    if args.command == "mine":
        from kullback.ai.provider import DEFAULT_MODEL
        body = mine(args.workdir, ids, args.model or DEFAULT_MODEL, args.ceiling_usd, args.total_usd)
    else:
        body = score(args.workdir, ids, Shares(witness=args.witness_share, carried=args.carried_share))
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body.get("rates") or {"spent_usd": body.get("spent_usd"), "tasks": len(body["tasks"])},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
