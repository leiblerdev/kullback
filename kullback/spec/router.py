"""Rulings become moves, routed by code over the workdir bus (DESIGN.md rule 4).

A Bus subscriber. Every ruling.filed is read back from the bus log in sequence order and moved on
by its target: a wrong Run is re-rolled (and excluded first when it is a Reference), a wrong check
or Intent goes to the Spec writer's repair, and an Environment that cannot support the Task goes to
the Builder as a note. Each move is one round on the Task; a ruling filed after ROUNDS_CAP rounds
is no agreement, so the Task is set aside with its rulings attached and task.set_aside published.

The router keeps the last sequence it read in workdir/spec/state/router.json, so replaying the bus
from any sequence handles each ruling once. Moves it cannot make yet are recorded, never dropped:
a repair with no writer lands on the Spec as pending_repair and publishes spec.repair_deferred, a
re-roll with no Runner as a queued request in the state file.

ROUNDS_CAP and `is_set_aside` are the one source of the set-aside rule: sp-trust's gates read them
from here rather than keep their own.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from kullback.agent.events import CustomEvent
from kullback.runner.records import EXAM_DIR, read_json, write_json
from kullback.spec import events
from kullback.spec.schema import Spec, load_spec, save_spec

log = logging.getLogger(__name__)

ROUNDS_CAP = 2
REROLL_COUNT = 2
STATE_FILE = "state/router.json"
NO_AGREEMENT = events.NO_AGREEMENT
WRITER_MODULE = "kullback.spec.writer"
WRITER_ENTRY = "repair_for_ruling"
# The events that can change what the router owes a Task.
HANDLED = (events.RULING_FILED, events.SPEC_REPAIRED, events.SPEC_DEFENDED)


def state_path(workdir: Any) -> Path:
    return Path(workdir) / "spec" / STATE_FILE


def load_state(workdir: Any) -> dict:
    body = read_json(state_path(workdir), {}) or {}
    return {"last_seq": int(body.get("last_seq") or 0), "rounds": dict(body.get("rounds") or {}),
            "queue": list(body.get("queue") or []), "actions": list(body.get("actions") or [])}


def is_set_aside(spec: Spec) -> bool:
    """Is the Task set aside? Marked so, or at ROUNDS_CAP rounds with a ruling still open. sp-trust reads it."""
    return spec.set_aside is not None or (spec.round >= ROUNDS_CAP and spec.rulings_open > 0)


def writer_repair() -> Optional[Callable]:
    """kullback.spec.writer.repair_for_ruling(workdir, task_id, ruling, model, ceiling_usd) -> Spec, else None.

    The writer saves the Spec, rewrites the Verifier file and publishes spec.repaired or spec.defended.
    """
    try:
        return getattr(importlib.import_module(WRITER_MODULE), WRITER_ENTRY, None)
    except ImportError:
        return None


def ruling_body(workdir: Any, payload: dict) -> dict:
    """The whole ruling from its file (the reason unredacted), else the bus payload."""
    path = Path(workdir) / EXAM_DIR / "rulings" / str(payload.get("task_id")) / f"{payload.get('number')}.json"
    body = read_json(path, None) if path.is_file() else None
    return body if isinstance(body, dict) else dict(payload)


class Router:
    """Moves each ruling on the bus once. `reroll(task_id, count)` and `note(task_id, ruling)` come from the
    session that attaches the router, the repair from the Spec writer with the session's `model` and
    `ceiling_usd`; a missing callee records the move as owed."""

    def __init__(self, workdir: Any, bus: Any, *, reroll: Optional[Callable] = None,
                 note: Optional[Callable] = None, model: Any = None, ceiling_usd: Optional[float] = None,
                 reroll_count: int = REROLL_COUNT, reject_reference: Optional[Callable] = None,
                 reference_ids_of: Optional[Callable] = None):
        self.workdir = Path(workdir)
        self.bus = bus
        self.reroll, self.note = reroll, note
        # The Examiner's Reference exclusion, injected by the Examiner so the Spec never imports it.
        self.reject_reference, self.reference_ids_of = reject_reference, reference_ids_of
        self.model, self.ceiling_usd = model, ceiling_usd
        self.reroll_count = reroll_count
        self.state = load_state(self.workdir)
        self._busy = False

    # --- the subscriber ---

    def __call__(self, event: Any) -> None:
        if isinstance(event, CustomEvent) and event.name in HANDLED:
            self.catch_up()

    def attach(self) -> Callable[[], None]:
        """Subscribe to the bus and read whatever it already holds past the last sequence."""
        unsubscribe = self.bus.subscribe(self)
        self.catch_up()
        return unsubscribe

    def catch_up(self) -> int:
        """Handle every record after the last sequence read; returns how many were handled."""
        if self._busy:
            return 0
        self._busy = True
        handled = 0
        try:
            while True:
                records = self.bus.replay(self.state["last_seq"])
                if not records:
                    return handled
                for record in records:
                    handled += self._handle(record)
                    self.state["last_seq"] = record.seq
                    self._save()
        finally:
            self._busy = False

    def _handle(self, record: Any) -> int:
        event = record.event
        if not isinstance(event, CustomEvent) or event.name not in HANDLED:
            return 0
        if event.name == events.RULING_FILED:
            self._ruling(record.seq, dict(event.payload))
        else:
            self._answered(str(event.payload.get("task_id")))
        return 1

    # --- rulings ---

    def _ruling(self, seq: int, payload: dict) -> None:
        task_id = str(payload.get("task_id"))
        target = str(payload.get("target"))
        spec = load_spec(self.workdir, task_id)
        if spec is not None and spec.set_aside:
            return self._act(seq, task_id, target, "ignored_set_aside")
        rounds = spec.round if spec is not None else int(self.state["rounds"].get(task_id, 0))
        if rounds >= ROUNDS_CAP:
            return self._set_aside(seq, task_id, target, spec, rounds)
        body = ruling_body(self.workdir, payload)
        action, still_open = self._move(target, task_id, body, spec, seq)
        self._next_round(task_id, rounds + 1, int(still_open))
        self._act(seq, task_id, target, action)

    def _move(self, target: str, task_id: str, body: dict, spec: Optional[Spec], seq: int) -> tuple[str, bool]:
        if target == "run":
            return self._rerun(task_id, body, seq)
        if target in ("check", "intent"):
            return self._repair(task_id, body, spec, seq)
        if target == "environment":
            if self.note is None:
                self.state["queue"].append({"seq": seq, "task_id": task_id, "move": "builder_note"})
                return "builder_note_queued", True
            self.note(task_id, body)
            return "builder_note", True
        return "unknown_target", False

    def _rerun(self, task_id: str, body: dict, seq: int) -> tuple[str, bool]:
        run_id = str(body.get("run_id") or "")
        action = "reroll"
        if self.reject_reference is None or self.reference_ids_of is None:
            log.warning("router: no Reference exclusion injected; ruling on run %s of %s rerolls only", run_id, task_id)
        elif run_id and run_id in self.reference_ids_of(self.workdir, task_id):
            self.reject_reference(self.workdir, task_id, run_id, str(body.get("reason") or ""))
            action = "reference_rejected_and_reroll"
        if self.reroll is None:
            self.state["queue"].append({"seq": seq, "task_id": task_id, "move": "reroll",
                                        "count": self.reroll_count})
            return action + "_queued", True
        self.reroll(task_id, self.reroll_count)
        return action, False

    def _repair(self, task_id: str, body: dict, spec: Optional[Spec], seq: int) -> tuple[str, bool]:
        repair = writer_repair()
        if repair is not None:
            repair(self.workdir, task_id, body, self.model, self.ceiling_usd)
            return "repair", True
        log.warning("no %s.%s: ruling %s on task %s waits as pending_repair",
                    WRITER_MODULE, WRITER_ENTRY, body.get("number"), task_id)
        events.spec_repair_deferred(self.bus, task_id, body.get("number"))
        request = {"seq": seq, "number": body.get("number"), "target": body.get("target"),
                   "check_ids": list(body.get("check_ids") or [])}
        if spec is not None:
            spec.pending_repair.append(request)
            save_spec(self.workdir, spec)
        else:
            self.state["queue"].append(dict(request, task_id=task_id, move="repair"))
        return "pending_repair", True

    def _next_round(self, task_id: str, rounds: int, opened: int) -> None:
        """Count the round on the Spec as it is now on disk, after whatever the move wrote to it."""
        self.state["rounds"][task_id] = rounds
        spec = load_spec(self.workdir, task_id)
        if spec is None:
            return
        spec.round = rounds
        spec.rulings_open += opened
        save_spec(self.workdir, spec)

    def _set_aside(self, seq: int, task_id: str, target: str, spec: Optional[Spec], rounds: int) -> None:
        if spec is not None:
            spec.set_aside = NO_AGREEMENT
            spec.rulings_open += 1
            save_spec(self.workdir, spec)
        events.task_set_aside(self.bus, task_id, NO_AGREEMENT, rounds)
        self._act(seq, task_id, target, "set_aside")

    def _answered(self, task_id: str) -> None:
        """A repair or a defence answers the oldest ruling the writer owed the Task."""
        spec = load_spec(self.workdir, task_id)
        if spec is None:
            return
        spec.rulings_open = max(spec.rulings_open - 1, 0)
        spec.pending_repair = spec.pending_repair[1:]
        save_spec(self.workdir, spec)

    def _act(self, seq: int, task_id: str, target: str, action: str) -> None:
        self.state["actions"].append({"seq": seq, "task_id": task_id, "target": target, "action": action})

    def _save(self) -> None:
        write_json(state_path(self.workdir), self.state)


__all__ = ["HANDLED", "NO_AGREEMENT", "REROLL_COUNT", "ROUNDS_CAP", "WRITER_ENTRY", "WRITER_MODULE", "Router",
           "is_set_aside", "load_state", "ruling_body", "state_path", "writer_repair"]
