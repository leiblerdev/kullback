"""The Spec records and their one file on disk: workdir/spec/<task_id>.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, model_serializer, model_validator

from kullback.runner.records import GATE_KINDS, AtomKind, Item, Record
from kullback.spec.must_not import tier_of as _atom_tier
from kullback.user.rules import spec_goal

# fallback: the user asks for it only in reply to an agent line that refused, could not, or proposed
# another way; code keeps the stance only when the quoted agent line is there (spec/intent_tools.py).
Stance = Literal["volunteered", "accepted", "fallback"]
# What a fact asks for, as the miner reads it: a change made, something left undone or unchanged,
# only to be told something, or a detail (who the user is, a value or condition shaping another ask).
FactKind = Literal["do", "keep", "told", "detail"]
# Spec fields retired with the Examiner's edit kinds (D325 to D331): dropped when an older file loads.
RETIRED_KEYS = ("atom_edits", "end_state_edits")
Tier = Literal["critical", "sanity", "important"]
# What prompted a fact, read from the agent line it answers (in_reply_to): none for a fact the user
# raised in their first turn; asked when the agent asked for it; world_or_policy when the agent stated a
# fact about the world or the rules (a conditional ask, gated only once the writer confirms the claim);
# agent when the agent could not, declined with no policy line, stalled or proposed another way (a
# reaction: never gated, never a mutant key, never counted as an ask). spec/intent_tools.py checks it.
Condition = Literal["none", "asked", "world_or_policy", "agent"]

# The atom kind a demand compiles to when it is not the check's own kind (compile.py does the same).
_COMPILED_KIND = {"say": "communicate", "ask": "question"}


class FactSource(Record):
    """Where a fact was said: one recording and the index of the user turn in it."""
    recording: str
    turn: int


class ReplyQuote(Record):
    """The agent line a fallback fact answers: the user turn it stands before, and a span copied from it."""
    turn: int
    quote: str


class IntentFact(Record):
    """One thing the user wants, with its provenance (DESIGN.md rule 5)."""
    id: str
    text: str
    source: FactSource
    stance: Stance
    witnesses: list[str] = Field(default_factory=list)
    # The fact carries the agent's proposal: the user only said yes to it, or a value of it (a token
    # with a digit) was in the agent's line before and not in the user's turn (spec/intent_tools.is_echo).
    echo: bool = False
    # The user said it in reply to the agent declining, failing, or offering another way, before
    # what the user first asked for was met: no gate item may rest on such facts alone (spec/items.py).
    after_pushback: bool = False
    # The Examiner ruled the user never asked for it (an intent_unfounded ruling the writer applied):
    # no gate item may rest on it alone (spec/items.py).
    unfounded: bool = False
    # The miner's reading of what the fact asks for; None on a fact mined before kinds were read.
    kind: Optional[FactKind] = None
    # Set by code (spec/intent_tools.mark_side), never by a model: an ask only to be told something in a
    # Task that also asks for a change or for something left undone. It may be scored, never gated.
    side: bool = False
    # On a fact said after the user's first turn: the agent line it answers, checked by code against the
    # recording.
    in_reply_to: Optional[ReplyQuote] = None
    condition: Condition = "none"

    @model_serializer(mode="wrap")
    def _without_unread_kind(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            for key in ("kind", "in_reply_to"):
                if data.get(key) is None:
                    data.pop(key, None)
            if not data.get("side"):
                data.pop("side", None)
            if data.get("condition") == "none":
                data.pop("condition", None)
        return data


def is_reaction(fact: Any) -> bool:
    """A fact the user said in reaction to the agent's conduct: condition agent, or a fallback fact mined
    before conditions were read. It never gates, never keys a mutant and never counts as an ask."""
    condition = getattr(fact, "condition", "none")
    return condition == "agent" or (getattr(fact, "stance", None) == "fallback" and condition == "none")


def is_ask(fact: Any) -> bool:
    """A do or keep fact that is not a reaction: what a gate must hold (unchecked_asks)."""
    return getattr(fact, "kind", None) in ("do", "keep") and not is_reaction(fact)


def unchecked_asks(spec: Any) -> list[str]:
    """The ids of the Spec's asks (is_ask) that are not filed gaps and that no gate item cites (D370): a
    write, a forbid, a kept row or a judge refusal all count; a weighed item does not."""
    gated = {str(f) for check in spec.checks if (check.demand or {}).get("gate") is True
             for f in (check.demand or {}).get("fact_ids") or ()}
    gaps = set(spec.gaps)
    return [fact.id for fact in spec.intent.facts if is_ask(fact) and fact.id not in gaps and fact.id not in gated]


class SpecIntent(Record):
    """The mined Intent of one Task: its facts, and the facts as text, one line each."""
    task_id: str
    facts: list[IntentFact] = Field(default_factory=list)
    text: str = ""
    # Facts the miner offered that code refused (words not in the named turn, a repeat, and so on).
    refusals: int = 0
    model: Optional[str] = None
    written_at: Optional[str] = None
    # Facts the miner marked fallback with no agent line carrying its quote: kept as volunteered.
    fallback_unquoted: int = 0
    # The position check (spec/intent_tools.py): facts said after the user's first turn sent back for
    # want of the agent line they answer, and those kept as reactions when sent again without it.
    late_sent_back: int = 0
    late_unplaced: int = 0
    # Reactions promoted to none: another recording's user volunteered them before any agent tool call.
    promoted: int = 0

    @model_serializer(mode="wrap")
    def _without_zero_unquoted(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            for key in ("fallback_unquoted", "late_sent_back", "late_unplaced", "promoted"):
                if not data.get(key):
                    data.pop(key, None)
        return data


def intent_text(facts: list[IntentFact]) -> str:
    """The facts joined, one line each: the text a because must quote."""
    return "\n".join(fact.text for fact in facts)


def tier_of(kind: str, demand: dict) -> Tier:
    """The DESIGN.md rule 7 tier of a check, from the atom kind it compiles to."""
    compiled = _COMPILED_KIND.get(str((demand or {}).get("demand")), kind)
    return _atom_tier({"kind": compiled})


class Check(Item):
    """One item of the Verifier, with the Intent facts (`fact_ids`) or the policy line it rests on.

    `demand` holds the item as code kept it (spec/items.py), or a demand of the older grammar.
    `gate` and `weight` go onto every atom it compiles to (D329); left unsaid, the gate follows
    the kind as an atom's does. A failing gate check zeroes a Run's reward; the others are weighed.
    """
    id: str
    kind: AtomKind
    demand: dict
    because: str
    tier: Tier
    fact_ids: list[str] = Field(default_factory=list)
    policy_line: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _gate_from_kind(cls, data: Any) -> Any:
        if isinstance(data, dict) and "gate" not in data:
            data = dict(data, gate=data.get("kind") in GATE_KINDS)
        return data

    def _gate_default(self) -> bool:
        return self.kind in GATE_KINDS

    def _drop_defaults(self, data: dict) -> dict:
        if data.get("policy_line") is None:
            data.pop("policy_line", None)
        return super()._drop_defaults(data)


class Spec(Record):
    """The single source of truth for one Task: the Intent and the checks written from it."""
    task_id: str
    intent: SpecIntent
    checks: list[Check] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rulings_open: int = 0
    # Open not_enough_truth rulings: the Task stays untrusted for that reason (spec/trust.py).
    truth_open: int = 0
    round: int = 0
    # Review rounds whose Examiner session ran over this Task, oldest first.
    examined_rounds: list[int] = Field(default_factory=list)
    version: int = 0
    # Open blocking rulings (spec/rulings.py) and review rounds run; requests left by the retired router,
    # and why the Task is set aside.
    pending_repair: list[dict] = Field(default_factory=list)
    set_aside: Optional[str] = None
    # The writer's session counts: demands, refusals, gaps, spend, read rounds, attempts, sent back.
    writer: dict = Field(default_factory=dict)
    # What the Verifier came to: its items by kind and gate, end states, cells (spec/items.py counts_of).
    end_state: dict = Field(default_factory=dict)


    @model_validator(mode="before")
    @classmethod
    def _retired(cls, data: Any) -> Any:
        """A Spec written before D331 loads: the Examiner's edit records are retired, never applied."""
        if isinstance(data, dict) and any(key in data for key in RETIRED_KEYS):
            data = {key: value for key, value in data.items() if key not in RETIRED_KEYS}
        return data


# Checks that specify a Task with no expected write: a Spec keeping one is never empty.
_NO_WRITE_DEMANDS = ("no_write", "cap")


def is_empty(spec: Spec) -> bool:
    """A Spec that keeps no check, or whose every Intent fact is a gap, unless it keeps a no_write or cap check.

    A Spec of items whose checks rest on the policy alone is not empty: a refusal is the sanity item and
    one judge item on a policy line.
    """
    if any(check.demand.get("demand") in _NO_WRITE_DEMANDS for check in spec.checks):
        return False
    if any(check.policy_line for check in spec.checks):
        return False
    gaps = set(spec.gaps)
    facts = spec.intent.facts
    return not spec.checks or (bool(facts) and all(fact.id in gaps for fact in facts))


def spec_path(workdir: Path, task_id: str) -> Path:
    return Path(workdir) / "spec" / f"{task_id}.json"


def spec_verifier_path(workdir: Path, task_id: str) -> Path:
    return Path(workdir) / "spec" / "verifiers" / f"{task_id}.json"


def spec_ids(workdir: Path) -> list[str]:
    """The ids of every Spec on disk, sorted; the spec folder holds Specs only, one file each."""
    folder = spec_path(workdir, "_").parent
    return sorted(path.stem for path in folder.glob("*.json")) if folder.is_dir() else []


def save_spec(workdir: Path, spec: Spec) -> Path:
    """Writes the Spec to its one file and returns the path."""
    path = spec_path(workdir, spec.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return path


def load_spec(workdir: Path, task_id: str) -> Optional[Spec]:
    """The Task's Spec, or None when it has none yet."""
    path = spec_path(workdir, task_id)
    if not path.exists():
        return None
    return Spec.model_validate_json(path.read_text())


def goal_of(workdir: Path, task_id: str, writes: Any) -> tuple[Optional[set], Optional[dict]]:
    """The user's goal label from the Task's Spec (`user.rules.spec_goal`): its writes and counts.
    With no Spec yet the goal is any write (None), never the Reference's calls."""
    spec = load_spec(Path(workdir), task_id) if workdir is not None else None
    if spec is None:
        return None, None
    return spec_goal([check.demand for check in spec.checks], writes)
