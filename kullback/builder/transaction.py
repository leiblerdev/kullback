"""Every repair is a transaction: it lands only if it fixed its target and cost nothing else (D201).

A repair verb used to land on its own ruling. The verb rewrote one Intent, one body or one table,
the repairing stage ruled on that one target, and whether the rest of the corpus moved under it was
nobody's question. Rounds where the red lights closed while the trusted count fell are the signature
of a repair that fixed one Task and cost two, and no record said which repair it was.

So a repair is opened, applied and closed. `open_transaction` records, for the set of Tasks the
repair can touch, the four lights they stand on now: whether the Task's recorded calls replay,
whether it keeps a Reference, whether it passes the suite, and whether it is trusted. `close`
reads the same four back off the artifacts the stage just wrote and rules:

  accepted                  the target moved up and no Task in the set lost a light
  reverted: regression      a Task in the set lost a light; the ruling names the Tasks and the light
  reverted: no effect       the target did not move

A reverted repair is undone, byte for byte, over the files its own kind writes, so the next attempt
starts from the artifacts the last accepted repair left. The ruling is the lesson the next attempt
reads, which is D191's shape: a sentence saying what this attempt bought and what it cost.

Three things are deliberately not here. Nothing calls a model: the lights are read off the artifacts
the repairing stage already wrote, and the scoped replay the rule needs is the one that stage ran
over the tool it recompiled, not a second one bought on top. Nothing is measured outside the set the
repair can touch, because a repair on one tool cannot be held to a Task that never calls it. And the
kinds that decide rather than act (refusing a Task, escalating it, recording a finding) open no
transaction at all: they own no artifact, so there is nothing to revert and nothing they can break.

The body scorer of D184 is one instance of this rule and not a second path: the set is the tool's own
recorded calls, the target's state is the score the compiler ranks its attempts by, and a rewritten
body that scores higher while dropping a call the kept body answered is reverted the same way an
Intent that costs another Task its Reference is (`rule`, called from the compile stage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

# The four lights a Task stands on, in the order a Task earns them. A repair is allowed to leave a
# light red that was already red; it is never allowed to turn a green one red.
LIGHTS: tuple[str, ...] = ("replay", "reference", "suite", "trusted")

ACCEPTED = "accepted"
REVERTED_REGRESSION = "reverted_regression"
REVERTED_NO_EFFECT = "reverted_no_effect"
DEFERRED = "deferred"

# How many repairs of a round may be checked over the whole corpus before the next one waits. A kind
# whose set cannot be worked out is held against every Task, and holding a round's every repair
# against every Task costs the round more than the repairs buy. The fourth such repair applies with
# its check deferred to the round's end, where the gates rule over the whole corpus anyway, and the
# ruling says so rather than reading as though it had been checked.
WHOLE_CORPUS_LIMIT = 3

# How many broken Tasks a ruling names before it counts the rest. A sentence naming forty ids is not
# a lesson, and the count is what says how bad it was.
NAMED = 6

# The files each acting kind's stage rewrites, and which a revert therefore has to put back. A kind
# absent from here owns no artifact and opens no transaction.
KIND_FILES: dict[str, tuple[str, ...]] = {
    "repair_recompile": ("bodies.json", "tool_builds.json", "tool_call_outcomes.json",
                         "tool_fidelity.json", "gates.json"),
    "repair_grow": ("db.json", "gates.json"),
    "repair_intent": ("gates.json",),
}
# The one extra path a kind writes under a directory, named after the target it was called on.
KIND_TARGET_FILE: dict[str, str] = {"repair_intent": "intents/{target}.json"}
# The file a kind keeps its stage's own ruling in, per target. It is a record of what the run
# decided and not an input of it, so a revert leaves it alone and stamps the outcome on it instead:
# putting it back would take away the score pair the next hint is written against (D191), which is
# the one thing a reverted recompile still has to say.
KIND_RECORD: dict[str, str] = {"repair_recompile": "kept_bodies.json"}


# --- reading the workdir ------------------------------------------------------


def _json_at(workdir: Any, relative: Any, default: Any) -> Any:
    """One record file of a workdir, or the default when it is missing or half written."""
    try:
        return json.loads((Path(workdir) / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _trusted_ids(workdir: Any) -> Optional[set[str]]:
    """The Tasks the trust gate last ruled trusted, or None where nothing has ruled yet.

    The ruling in `gates.json` is the freshest one; before the first Examiner beat of a build there
    is none, and the last round's own count is what stands. None is not an empty set: a light nobody
    has ruled on cannot be lost, and telling the two apart is what stops a repair being blamed for a
    gate that has never run.
    """
    rows = _json_at(workdir, "gates.json", [])
    for row in reversed(rows if isinstance(rows, list) else []):
        if isinstance(row, dict) and row.get("stage") == "trusted":
            ids = (row.get("metrics") or {}).get("trusted")
            if isinstance(ids, list):
                return {str(task) for task in ids}
    rounds = _json_at(workdir, "rounds.json", [])
    for record in reversed(rounds if isinstance(rounds, list) else []):
        ids = ((record or {}).get("counts") or {}).get("trusted_ids") if isinstance(record, dict) else None
        if isinstance(ids, list):
            return {str(task) for task in ids}
    return None


def corpus_tasks(workdir: Any) -> list[str]:
    """Every Task of the build, off the frozen list where there is one and the status rows otherwise."""
    frozen = _json_at(workdir, "tasks_frozen.json", [])
    if isinstance(frozen, dict):
        frozen = frozen.get("tasks") or frozen.get("task_ids") or []
    if isinstance(frozen, list) and frozen:
        return sorted(str(task) for task in frozen)
    status = _json_at(workdir, "task_status.json", {})
    return sorted(status) if isinstance(status, dict) else []


def task_lights(workdir: Any, tasks: Optional[Iterable[str]] = None) -> dict[str, dict[str, Optional[bool]]]:
    """The four lights per Task, read off the artifacts as they stand; no model call and no replay.

    `replay` is the Task's own fidelity attribution: every recorded call of every tool it calls
    replayed, which is the reading the compile stage writes for each Task and the one a body repair
    moves. `reference` and `suite` are the Examiner's own rows. `trusted` is the trust gate's last
    ruling. A light no artifact has ruled on is None, which no comparison counts either way: a
    repair answers for the lights that were on when it started, not for gates that never ran.
    """
    fidelity = _json_at(workdir, "tool_fidelity.json", {})
    fidelity = fidelity.get("tasks") if isinstance(fidelity, dict) else None
    fidelity = fidelity if isinstance(fidelity, dict) else {}
    status = _json_at(workdir, "task_status.json", {})
    status = status if isinstance(status, dict) else {}
    trusted = _trusted_ids(workdir)
    wanted = list(tasks) if tasks is not None else sorted(set(fidelity) | set(status))
    out: dict[str, dict[str, Optional[bool]]] = {}
    for task in wanted:
        row = status.get(task) if isinstance(status.get(task), dict) else None
        per_tool = fidelity.get(task) if isinstance(fidelity.get(task), dict) else None
        out[str(task)] = {
            "replay": None if per_tool is None else
                      all(not int((cell or {}).get("differing") or 0) for cell in per_tool.values()),
            "reference": None if row is None else bool(row.get("reference_confirmed")),
            "suite": None if row is None else bool(row.get("verifier_passed")),
            "trusted": None if trusted is None else task in trusted,
        }
    return out


def call_lights(outcomes: Iterable[dict]) -> dict[str, dict[str, Optional[bool]]]:
    """The same shape over one tool's recorded calls: whether each call replayed (D184's instance).

    A body is repaired against a set of calls the way an Intent is repaired against a set of Tasks,
    so the calls go through the same rule: a rewritten body that answers a call the kept body did
    not, while losing one the kept body did, has cost something and is not an improvement.
    """
    out: dict[str, dict[str, Optional[bool]]] = {}
    for row in outcomes or ():
        if not isinstance(row, dict):
            continue
        call_id = str(row.get("call_id") or "")
        if call_id:
            out[call_id] = {"replay": bool(row.get("replayed"))}
    return out


# --- what one repair can touch -------------------------------------------------


def scope_of(workdir: Any, kind: str, target: str) -> tuple[list[str], bool]:
    """The Tasks one repair can touch, and whether that set is narrower than the whole corpus.

    A repair on one Task's Intent touches that Task. A repair on one tool's body touches the Tasks
    whose own recorded calls that tool answers, which the fidelity attribution already names. A
    repair on the Starting state moves rows every Task may read, and no artifact says which Tasks
    those are, so the set is the corpus and the second value is False: that is the fallback the
    round's limit is about.
    """
    if kind == "repair_intent":
        return [target], True
    if kind == "repair_recompile":
        fidelity = _json_at(workdir, "tool_fidelity.json", {})
        per_task = fidelity.get("tasks") if isinstance(fidelity, dict) else None
        if isinstance(per_task, dict) and per_task:
            return sorted(task for task, row in per_task.items()
                          if isinstance(row, dict) and target in row), True
    return corpus_tasks(workdir), False


def target_state(workdir: Any, kind: str, target: str) -> Any:
    """How the one target stands, as a value that compares: a repair moved it when the value went up.

    A red light going green and a score going up are the same question asked of different targets,
    so both are answered with something ordered and `improved` is the whole of the comparison.
    """
    if kind == "repair_intent":
        record = _json_at(workdir, Path("intents") / f"{target}.json", {})
        return bool(record.get("grounded")) if isinstance(record, dict) else False
    if kind == "repair_recompile":
        builds = _json_at(workdir, "tool_builds.json", {})
        row = builds.get(target) if isinstance(builds, dict) else None
        row = row if isinstance(row, dict) else {}
        return (not bool(row.get("assisted")), tuple(row.get("score") or ()))
    if kind == "repair_grow":
        db = _json_at(workdir, "db.json", {})
        rows = db.get(target) if isinstance(db, dict) else None
        return len(rows) if isinstance(rows, dict) else 0
    return None


def improved(before: Any, after: Any) -> bool:
    """True when the target moved up. Two values that cannot be compared never count as a move."""
    if before is None or after is None:
        return False
    try:
        return bool(after > before)
    except TypeError:
        return False


def regressions(before: dict[str, dict[str, Optional[bool]]],
                after: dict[str, dict[str, Optional[bool]]]) -> list[str]:
    """Every green light in `before` that is red in `after`, as "<id> <light>", in a stable order.

    Only lights that were measured on both sides count. A Task that had no row before the repair and
    has one after gained a reading, not a regression, and a gate that has never run leaves None on
    both sides.
    """
    lost = []
    for entity in sorted(before):
        was, now = before.get(entity) or {}, after.get(entity) or {}
        for light in LIGHTS:
            if was.get(light) is True and now.get(light) is False:
                lost.append(f"{entity} {light}")
    return lost


# --- the ruling ----------------------------------------------------------------


class RepairRuling(BaseModel):
    """What one transaction decided, in the shape the round report and the next attempt read."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    target: str
    status: str
    moved: bool = False
    broke: list[str] = Field(default_factory=list)
    scope: int = 0
    scoped: bool = True
    green_before: int = 0
    green_after: int = 0

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def line(self) -> str:
        """The sentence the repair result opens with, which is the lesson the next attempt reads."""
        where = f"{self.scope} Task{'' if self.scope == 1 else 's'}" + ("" if self.scoped else ", the whole corpus")
        counts = f"{self.green_before} to {self.green_after} lights green over {where}"
        if self.status == DEFERRED:
            return (f"unchecked: the set could not be narrowed and this round has already held "
                    f"{WHOLE_CORPUS_LIMIT} repairs against the whole corpus, so this one is applied "
                    f"and its check waits for the round's own gates at round end")
        if self.status == ACCEPTED:
            return f"accepted: fixed {self.target} and broke nothing ({counts})"
        if self.status == REVERTED_REGRESSION:
            named = ", ".join(self.broke[:NAMED])
            more = len(self.broke) - NAMED
            fixed = f"fixed {self.target}" if self.moved else f"did not fix {self.target}"
            return (f"reverted: {fixed}, broke {len(self.broke)}: {named}"
                    + (f" and {more} more" if more > 0 else "") + f" ({counts})")
        return f"reverted: no effect, {self.target} stands where it was ({counts})"

    def as_row(self) -> dict:
        """The fields this ruling puts on the repair's own request record."""
        return {"outcome": self.status, "moved": self.moved, "broke": list(self.broke),
                "scope": self.scope, "scoped": self.scoped,
                "green_before": self.green_before, "green_after": self.green_after}


def _green(lights: dict[str, dict[str, Optional[bool]]]) -> int:
    return sum(1 for row in lights.values() for light in LIGHTS if (row or {}).get(light) is True)


def rule(kind: str, target: str, moved: bool,
         before: dict[str, dict[str, Optional[bool]]],
         after: dict[str, dict[str, Optional[bool]]], scoped: bool = True) -> RepairRuling:
    """The one rule every repair goes through: it lands if it moved its target and broke nothing.

    A repair that broke something is reverted whether or not it moved its target, and the ruling
    says which it was: a repair that fixed its target and cost two Tasks is a different lesson from
    one that fixed nothing and cost two. A repair that broke nothing and moved nothing is reverted
    for no effect, because the round has to be told that the hint bought nothing rather than reading
    an unchanged artifact as a repair.
    """
    broke = regressions(before, after)
    status = REVERTED_REGRESSION if broke else (ACCEPTED if moved else REVERTED_NO_EFFECT)
    return RepairRuling(kind=kind, target=target, status=status, moved=bool(moved), broke=broke,
                        scope=len(before), scoped=scoped,
                        green_before=_green(before), green_after=_green(after))


# --- the transaction ------------------------------------------------------------


def recorded_repairs(workdir: Any, round_no: Optional[int] = None) -> list[dict]:
    """Every repair request this workdir has recorded, oldest file first; one round when asked."""
    folder = Path(workdir) / "repairs"
    rows: list[dict] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (round_no is None or int(row.get("round") or 0) == int(round_no)):
                rows.append(row)
    return rows


def whole_corpus_checks(workdir: Any, round_no: int) -> int:
    """How many repairs of this round were already held against the whole corpus."""
    return sum(1 for row in recorded_repairs(workdir, round_no)
               if row.get("scoped") is False and row.get("outcome") != DEFERRED)


def repair_files(workdir: Any, kind: str, target: str) -> list[Path]:
    """The files a revert of this kind has to put back, whether or not they exist yet."""
    root = Path(workdir)
    paths = [root / name for name in KIND_FILES.get(kind, ())]
    pattern = KIND_TARGET_FILE.get(kind)
    if pattern:
        paths.append(root / pattern.format(target=target))
    return paths


def snapshot(paths: Iterable[Path]) -> dict[str, Optional[bytes]]:
    """The bytes of each file now, with None for one that does not exist yet."""
    out: dict[str, Optional[bytes]] = {}
    for path in paths:
        try:
            out[str(path)] = path.read_bytes()
        except OSError:
            out[str(path)] = None
    return out


def restore(taken: dict[str, Optional[bytes]]) -> list[str]:
    """Put back every file whose bytes have moved since the snapshot; the names of the ones put back.

    A file that did not exist when the snapshot was taken and does now is removed, because that is
    what "as it was" means for it.
    """
    put_back = []
    for name, body in taken.items():
        path = Path(name)
        try:
            now = path.read_bytes() if path.is_file() else None
        except OSError:
            now = None
        if now == body:
            continue
        if body is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        put_back.append(path.name)
    return put_back


@dataclass
class Transaction:
    """One repair held open: what it can touch, where those Tasks stood, and how to put them back."""

    workdir: Any
    kind: str
    target: str
    scope: list[str] = field(default_factory=list)
    scoped: bool = True
    deferred: bool = False
    before_state: Any = None
    before_lights: dict[str, dict[str, Optional[bool]]] = field(default_factory=dict)
    taken: dict[str, Optional[bytes]] = field(default_factory=dict)


def opens(kind: str) -> bool:
    """Whether this kind of repair owns an artifact, and so has anything to hold open."""
    return kind in KIND_FILES


def open_transaction(workdir: Any, kind: str, target: str, round_no: int = 1) -> Transaction:
    """Record where the Tasks this repair can touch stand, and take the files it may have to put back."""
    scope, scoped = scope_of(workdir, kind, target)
    deferred = not scoped and whole_corpus_checks(workdir, round_no) >= WHOLE_CORPUS_LIMIT
    if deferred:
        return Transaction(workdir=workdir, kind=kind, target=target, scope=scope, scoped=scoped,
                           deferred=True)
    return Transaction(workdir=workdir, kind=kind, target=target, scope=scope, scoped=scoped,
                       before_state=target_state(workdir, kind, target),
                       before_lights=task_lights(workdir, scope),
                       taken=snapshot(repair_files(workdir, kind, target)))


def stamp_record(workdir: Any, kind: str, target: str, ruling: RepairRuling) -> None:
    """Write the transaction's outcome onto the stage's own ruling for this target, where it keeps one.

    The stage recorded what it decided about this target before the transaction ruled on it, so
    without this the record would go on saying a body was released after the release was put back.
    """
    name = KIND_RECORD.get(kind)
    if not name:
        return
    rows = _json_at(workdir, name, {})
    row = rows.get(target) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        return
    row["reverted"] = ruling.status
    row["reverted_tasks"] = list(ruling.broke[:NAMED])
    (Path(workdir) / name).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def close(txn: Transaction) -> RepairRuling:
    """Rule on the repair now the stage has run, and put the artifacts back where it is not accepted."""
    if txn.deferred:
        return RepairRuling(kind=txn.kind, target=txn.target, status=DEFERRED,
                            scope=len(txn.scope), scoped=txn.scoped)
    after = task_lights(txn.workdir, txn.scope)
    ruling = rule(txn.kind, txn.target,
                  improved(txn.before_state, target_state(txn.workdir, txn.kind, txn.target)),
                  txn.before_lights, after, scoped=txn.scoped)
    if not ruling.accepted:
        restore(txn.taken)
        stamp_record(txn.workdir, txn.kind, txn.target, ruling)
    return ruling


# --- what a round says about its repairs ------------------------------------------


def round_outcomes(repairs: Iterable[dict]) -> dict[str, int]:
    """The counts a round carries: how many repairs landed, and how many were put back and why."""
    rows = [row for row in repairs if isinstance(row, dict)]
    return {
        "repairs_accepted": sum(1 for row in rows if row.get("outcome") == ACCEPTED),
        "repairs_reverted_regression": sum(1 for row in rows if row.get("outcome") == REVERTED_REGRESSION),
        "repairs_reverted_no_effect": sum(1 for row in rows if row.get("outcome") == REVERTED_NO_EFFECT),
        "repairs_deferred": sum(1 for row in rows if row.get("outcome") == DEFERRED),
    }


def reverted_by_kind(repairs: Iterable[dict]) -> str:
    """The reverted repairs of a round, by kind, for the round's own report; "" when none were.

    By kind and not by target, because what a round has to be told is which verb keeps buying
    nothing: five recompiles reverted for no effect is a hint that is not working, and one Intent
    reverted for a regression is a Task that two Tasks depend on.
    """
    counts: dict[str, dict[str, int]] = {}
    for row in repairs:
        if not isinstance(row, dict):
            continue
        status = str(row.get("outcome") or "")
        if status not in (REVERTED_REGRESSION, REVERTED_NO_EFFECT):
            continue
        counts.setdefault(str(row.get("verb") or "?"), {})[status] = \
            counts.setdefault(str(row.get("verb") or "?"), {}).get(status, 0) + 1
    if not counts:
        return ""
    parts = []
    for kind in sorted(counts):
        broke = counts[kind].get(REVERTED_REGRESSION, 0)
        nothing = counts[kind].get(REVERTED_NO_EFFECT, 0)
        said = ", ".join(part for part in
                         (f"{broke} for a regression" if broke else "",
                          f"{nothing} for no effect" if nothing else "") if part)
        parts.append(f"{kind} {broke + nothing} ({said})")
    return "reverted repairs: " + "; ".join(parts)
