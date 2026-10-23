"""The hand-built Runs the Verifier tests share: one Reference, its re-runs and the Runs that must fail.

`derive` calls the Examiner's `derive_verifier` (the derivation moved there in phase 5, D123);
everything else is a Run record and a way to put one on disk the way loop.py does.
"""

from __future__ import annotations

import json
from pathlib import Path

from kullback import derive as V
from kullback.gates import verifier_suite as S
from kullback.runner.canon import CanonRules
from kullback.runner.records import Run, Task, Verifier, as_dict
from tests.helpers.records import assistant, call, make_run, result, user

WRITE_TOOLS = {"cancel_pending_order"}
ORDER = {"order_id": "#W123", "status": "pending", "total": 150.0}
TASK = Task(id="t1", intent="cancel the pending order and record the reason")


# --- tiny hand-built Runs -------------------------------------------------

def reference_events(reason: str = "no longer needed",
                     final: str = "Your order #W123 is cancelled and 150.0 is refunded.",
                     written: str = "#W123", answered: str | None = None) -> list[dict]:
    """The Reference's events; `written` is the id the cancel names, `answered` the id its result names."""
    return [
        user("Please cancel my order #W123."),
        assistant("Sure. Why do you want to cancel it?"),
        user(reason),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        call("cancel_pending_order", {"order_id": written, "reason": reason}, cid="c1"),
        result({"order_id": answered or written, "status": "cancelled"}, cid="c1"),
        assistant(final),
    ]


def reference_run() -> Run:
    return make_run("ref", reference_events())


def alt_path_run() -> Run:
    """Same writes, different reads and different wording (D46 different path)."""
    return make_run("alt", [
        user("Please cancel my order #W123."),
        call("get_order_details", {"order_id": "#W123"}, kind="read", cid="c0"),
        result(ORDER, cid="c0"),
        assistant("Happy to help. May I ask the reason?"),
        user("no longer needed"),
        call("cancel_pending_order", {"order_id": "#W123", "reason": "no longer needed"}, cid="c1"),
        result({"order_id": "#W123", "status": "cancelled"}, cid="c1"),
        assistant("Done: #W123 is cancelled, 150.0 goes back to your card."),
    ])


def other_reason_run() -> Run:
    return make_run("rr2", reference_events(reason="changed my mind"))


def extra_write_run() -> Run:
    """A successful re-run that writes one entity more than the Reference did."""
    return make_run("extra", reference_events() + [
        call("cancel_pending_order", {"order_id": "#W888", "reason": "no longer needed"}, cid="c2"),
        result({"order_id": "#W888", "status": "cancelled"}, cid="c2"),
    ])


def failed_run() -> Run:
    return make_run("bad", [
        user("Please cancel my order #W123."),
        call("cancel_pending_order", {"order_id": "#W999", "reason": "no longer needed"}, cid="c1"),
        result({"error": "not found"}, cid="c1"),
        assistant("I could not find that order."),
    ], termination_reason="max_steps")


def wrong_run() -> Run:
    """Plausible but wrong: the wrong entity, everything else in place."""
    return make_run("wrong", reference_events(written="#W999"))


def empty_run() -> Run:
    return make_run("empty", [], termination_reason="max_steps")


def write_events_jsonl(run: Run, path: Path) -> str:
    """A Run as one header line plus one line per event."""
    head = as_dict(run)
    events = head.pop("events")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(head) + "\n")
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return str(path)


def write_run_json(run: Run, path: Path) -> str:
    """A whole Run as a single JSON line."""
    path.write_text(json.dumps(as_dict(run)) + "\n", encoding="utf-8")
    return str(path)


def write_run_with_footer(run: Run, path: Path, **footer) -> Path:
    """A Run as one line per event plus the trailing footer line loop.py writes (state included)."""
    with path.open("w", encoding="utf-8") as handle:
        for event in as_dict(run)["events"]:
            handle.write(json.dumps(event) + "\n")
        handle.write(json.dumps(footer) + "\n")
    return path


def derive(tmp_path: Path, reruns=None, **kwargs) -> Verifier:
    reruns = reruns if reruns is not None else [alt_path_run(), other_reason_run()]
    paths = [write_events_jsonl(r, tmp_path / f"{r.run_id}.jsonl") for r in reruns]
    kwargs.setdefault("write_tools", WRITE_TOOLS)
    return V.derive_verifier(TASK, reference_run(), paths, kwargs.pop("canon", CanonRules()), **kwargs)


def payloads(verifier: Verifier) -> list[dict]:
    return [S.atom_payload(a) for a in verifier.atoms]

