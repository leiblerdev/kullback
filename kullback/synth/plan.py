"""The plan of situations the recordings' openings show, written before any scenario (D380).

A plan session reads a batch of numbered opening asks and puts each one in exactly one place: a
situation it writes, a situation already in the plan it attaches to, or the skipped list with the
reason. A situation is one sentence in general words saying one thing that happens, a one word kind,
the openings that show it and the variants that change what the agent must do. Each opening keeps the
Task and the recording it came from, so provenance holds by construction: after the plan no model
lists an opening number again.

The plan lives at <workdir>/scenarios/plan.json, written after every session. The tools refuse with
why, as the generator's write tools do: an opening another situation holds (naming it), an empty
statement, a tool name or an id shape, and a hedged statement, which says two things at once and is
split into variants instead.

A plan_search session (D410) writes situations from the web instead: each rests on the span of a page it
read, kept under "source" in the Source shape a search scenario carries, with no openings and no variants.
Its write is refused for a kind the plan does not hold and for a statement that nearly repeats a situation
of the plan, naming the holder.

A kind sharing its stem with an earlier kind of the plan is refused in both sessions, naming that kind, so
one thing is never counted under two words (D411). A refusal quotes the words that failed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from kullback.runner.records import read_json, write_json
from kullback.synth import records as R
from kullback.synth.rule import EMPTY, ID_SHAPE, World, id_word, rouge_l, statement_refusal

PLAN = "plan.json"
FORMAT = 1
# How many opening asks one plan session reads.
PLAN_BATCH = 150
HEDGED = "hedged_statement"
HELD = "opening_already_held"
NEAR = "near_repeat"
KIND = "kind_not_in_plan"
_HEDGES = re.compile(r"\bsometimes\b| or ", re.IGNORECASE)
TWIN = "kind_twin"
# Endings cut off a kind before two kinds are compared by stem (D411).
_ENDINGS = ("ations", "ation", "ings", "ing", "ments", "ment", "ed", "es", "s")


def situation_id(statement: str) -> str:
    """A situation's name: a content hash of its statement, nothing else."""
    return "si_" + hashlib.sha256(str(statement).encode("utf-8")).hexdigest()[:12]


def opening_key(row: dict) -> tuple[str, str]:
    return str(row.get("task_id") or ""), str(row.get("recording") or "")


def _opening(row: dict) -> dict:
    return {"task_id": str(row.get("task_id") or ""), "recording": str(row.get("recording") or ""),
            "text": str(row.get("text") or "")}


@dataclass
class Plan:
    """Every situation written so far and every opening skipped, across the sessions of one store."""

    situations: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def holder(self, row: dict) -> str:
        """The id of the situation holding this opening, "skipped", or nothing."""
        key = opening_key(row)
        for situation in self.situations:
            if any(opening_key(held) == key for held in situation["openings"]):
                return situation["id"]
        return "skipped" if any(opening_key(held) == key for held in self.skipped) else ""

    def find(self, situation_id_: str) -> dict:
        found = next((s for s in self.situations if s["id"] == situation_id_), None)
        if found is None:
            raise ValueError(f"{situation_id_!r} is not a situation of the plan; list_plan names them")
        return found

    def rows(self) -> list[dict]:
        """The plan as list_plan shows it: id, statement, kind, how many openings, variants."""
        return [{"id": s["id"], "statement": s["statement"], "kind": s["kind"], "openings": len(s["openings"]),
                 "variants": list(s["variants"])} for s in self.situations]

    def kinds(self) -> list[str]:
        return sorted({s["kind"] for s in self.situations})

    def per_kind(self) -> dict[str, int]:
        """Situations per kind, fewest first, ties by name."""
        counts: dict[str, int] = {}
        for situation in self.situations:
            counts[situation["kind"]] = counts.get(situation["kind"], 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (item[1], item[0])))

    def body(self) -> dict:
        return {"format": FORMAT, "situations": self.situations, "skipped": self.skipped}


def plan_path(workdir: Any) -> Path:
    return R.store_dir(workdir) / PLAN


def load_plan(workdir: Any) -> Plan:
    body = read_json(plan_path(workdir), None) or {}
    return Plan(situations=list(body.get("situations") or []), skipped=list(body.get("skipped") or []))


def write_plan(workdir: Any, plan: Plan) -> Path:
    return write_json(plan_path(workdir), plan.body())


def unplanned(plan: Plan, rows: Iterable[dict]) -> list[dict]:
    """The openings no situation holds and none skipped, in the order given."""
    return [row for row in rows if not plan.holder(row)]


# --- the checks the plan tools hold ---------------------------------------------------------


def _picked(items: list[dict], numbers: Iterable[int]) -> list[tuple[int, dict]]:
    picked = sorted(set(int(at) for at in numbers))
    outside = [at for at in picked if not 0 <= at < len(items)]
    if outside:
        raise ValueError(f"item {', '.join(map(str, outside))} is not a numbered item of this session")
    return [(at, items[at]) for at in picked]


def _free(plan: Plan, items: list[dict], numbers: Iterable[int]) -> list[dict]:
    """The picked openings, refused naming the holder where any already sits in the plan."""
    picked = _picked(items, numbers)
    held = [f"{at} (in {plan.holder(row)})" for at, row in picked if plan.holder(row)]
    if held:
        raise ValueError(f"refused ({HELD.replace('_', ' ')}): item {', '.join(held)} already has its place; "
                         "pick the rest in a new call")
    return [row for _, row in picked]


def _statement_why(text: str, world: World) -> str:
    if not text:
        return EMPTY
    return statement_refusal(text, world) or (HEDGED if _HEDGES.search(f" {text} ") else "")


def _refused(why: str, text: str = "") -> ValueError:
    """The refusal, quoting the words that failed: the hedging word or the id shaped word (D411)."""
    if why == HEDGED:
        found = _HEDGES.search(f" {text} ")
        word = found.group(0).strip() if found else ""
        return ValueError(f"refused (hedged statement: \"{word}\"): it says two things at once; write one "
                          "situation and give the other as a variant")
    word = id_word(text) if why == ID_SHAPE else ""
    named = why.replace("_", " ") + (f": \"{word}\"" if word else "")
    return ValueError(f"refused ({named}): write it again, following the rule")


def stem(kind: str) -> str:
    """A kind without its ending and a doubled last letter, so book and booking, cancel and cancellation meet."""
    word = str(kind or "").lower()
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[:-len(ending)]
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word


def kind_twin(plan: Plan, kind: str) -> str:
    """The plan's first kind sharing a stem with this one where it is another word, or nothing (D411).

    Two stems meet when they are equal or one starts the other at four letters or more."""
    for situation in plan.situations:
        other = str(situation["kind"])
        if other == kind:
            return ""
        if meet(kind, other):
            return other
    return ""


def meet(one: str, other: str) -> bool:
    """Whether two kinds are one by stem: equal stems, or one starting the other at four letters or more."""
    short, long_ = sorted((stem(one), stem(other)), key=len)
    return short == long_ or (len(short) >= 4 and long_.startswith(short))


def canonical_kind(plan: Plan, kind: str) -> str:
    """The plan's first kind one with this one by stem, or the kind itself: kinds merge before they spread (D420)."""
    word = str(kind or "")
    return next((str(s["kind"]) for s in plan.situations if meet(word, str(s["kind"]))), word) if word else word


def recorded_kinds(plan: Plan) -> Counter:
    """Per kind of the plan, merged by stem, the recorded Tasks whose openings its situations hold."""
    tasks: dict[str, set] = {}
    for situation in plan.situations:
        held = tasks.setdefault(canonical_kind(plan, situation["kind"]), set())
        held.update(opening["task_id"] for opening in situation["openings"])
    return Counter({kind: len(ids) for kind, ids in tasks.items()})


def by_distance(want: dict, have: Counter) -> str:
    """The choice furthest under its wanted share: wanted share less the share it holds, largest first, ties by
    name (D420). A choice wanted by nothing is wanted evenly with the rest where nothing is wanted at all."""
    choices = sorted(want)
    if not choices:
        return ""
    total = sum(want.values())
    shares = {c: (want[c] / total if total else 1 / len(choices)) for c in choices}
    held = sum(have.get(c, 0) for c in choices)
    return max(choices, key=lambda c: (shares[c] - (have.get(c, 0) / held if held else 0.0), -choices.index(c)))


def _no_twin(plan: Plan, kind: str) -> None:
    twin = kind_twin(plan, kind)
    if twin:
        raise ValueError(f"refused ({TWIN.replace('_', ' ')}): the plan holds the kind {twin}; give {twin}")


def write_situation(plan: Plan, world: World, items: list[dict], statement: str, kind: str,
                    openings: Iterable[int], variants: Iterable[str]) -> dict:
    """One situation into the plan, or refused with why."""
    text = " ".join(str(statement or "").split())
    why = _statement_why(text, world)
    lines = [" ".join(str(line).split()) for line in variants or () if str(line).strip()]
    failed = next((line for line in lines if statement_refusal(line, world)), "")
    if why:
        raise _refused(why, text)
    if failed:
        raise _refused(statement_refusal(failed, world), failed)
    word = str(kind or "").strip().lower()
    if not word or len(word.split()) != 1:
        raise ValueError("kind is one word: change, cancel, return, exchange, question, complaint, refusal or your own")
    _no_twin(plan, word)
    made = situation_id(text)
    if any(s["id"] == made for s in plan.situations):
        raise ValueError(f"{made} already says this; attach the openings to it with attach_openings")
    picked = _free(plan, items, openings)
    situation = {"id": made, "statement": text, "kind": word, "openings": [_opening(row) for row in picked],
                 "variants": lines}
    plan.situations.append(situation)
    return situation


def near_holder(plan: Plan, text: str, near: float) -> str:
    """The first situation of the plan whose statement the text nearly repeats, at ROUGE-L `near` or over."""
    return next((s["id"] for s in plan.situations if rouge_l(text, s["statement"]) >= near), "")


def write_searched(plan: Plan, world: World, statement: str, kind: str, source: dict, span: str,
                   near: float) -> dict:
    """One situation from a page span into the plan (D410), or refused with why."""
    text = " ".join(str(statement or "").split())
    why = _statement_why(text, world)
    if why:
        raise _refused(why, text)
    word = str(kind or "").strip().lower()
    _no_twin(plan, word)
    kinds = plan.kinds()
    if kinds and word not in kinds:
        raise ValueError(f"refused ({KIND.replace('_', ' ')}): give one of {', '.join(kinds)}")
    if not word or len(word.split()) != 1:
        raise ValueError("kind is one word")
    holder = near_holder(plan, text, near)
    if holder:
        raise ValueError(f"refused ({NEAR.replace('_', ' ')}): {holder} already says this; write another situation")
    situation = {"id": situation_id(text), "statement": text, "kind": word, "openings": [], "variants": [],
                 "source": dict(source), "span": span}
    plan.situations.append(situation)
    return situation


def attach_openings(plan: Plan, items: list[dict], situation_id_: str, openings: Iterable[int]) -> dict:
    """Openings added to a situation of this or any earlier session, or refused with why."""
    situation = plan.find(situation_id_)
    picked = _free(plan, items, openings)
    situation["openings"] += [_opening(row) for row in picked]
    return situation


def skip_opening(plan: Plan, items: list[dict], openings: Iterable[int], why: str) -> list[dict]:
    """Items that are no opening ask, marked with the reason, or refused with why."""
    reason = " ".join(str(why or "").split())
    if not reason:
        raise ValueError("skip_opening needs why: a turn from the middle of a conversation, identification only, no ask")
    picked = _free(plan, items, openings)
    rows = [{"task_id": row["task_id"], "recording": row["recording"], "why": reason} for row in picked]
    plan.skipped += rows
    return rows


__all__ = ["by_distance", "canonical_kind", "meet", "recorded_kinds", "HEDGED", "HELD", "KIND", "NEAR", "TWIN", "kind_twin", "stem", "near_holder", "write_searched", "PLAN", "PLAN_BATCH", "Plan", "attach_openings", "load_plan", "opening_key",
           "plan_path", "situation_id", "skip_opening", "unplanned", "write_plan", "write_situation"]
