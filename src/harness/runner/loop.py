"""One Run, one turn at a time: query the model, route its tool calls, write one JSONL line per event (D90)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from harness.shared.records import Cost, Event, Run, as_dict

TRANSFER = "###TRANSFER###"
STOP_MARKERS = ("###STOP###", TRANSFER)


@dataclass
class RunState:
    """A Run in progress: the record, the transcript the model sees, and where its JSONL goes."""
    run: Run
    messages: list[dict] = field(default_factory=list)
    path: Optional[Path] = None
    user: Any = None
    max_turns: int = 20
    turn: int = 0
    stopped: bool = False
    finished: bool = False


def new_run_state(run_id: str, *, workdir: Any = None, path: Any = None, system_prompt: Optional[str] = None,
                  first_user: Optional[str] = None, user: Any = None, max_turns: int = 20,
                  **run_fields: Any) -> RunState:
    """A fresh Run with its opening transcript and an empty JSONL file under the workdir."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if first_user:
        messages.append({"role": "user", "content": first_user})
    if path is None and workdir is not None:
        path = Path(workdir) / f"{run_id}.jsonl"
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return RunState(run=Run(run_id=run_id, **run_fields), messages=messages, path=path,
                    user=user, max_turns=max_turns)


def emit(state: RunState, event_type: str, payload: dict, route: Optional[str] = None,
         cost: Optional[Cost] = None, assisted: bool = False) -> Event:
    """Append one event to the Run and one line to its JSONL. No clock: the idx is the only order."""
    event = Event(idx=len(state.run.events), type=event_type, payload=payload,
                  route=route, cost=cost, assisted=assisted)
    state.run.events.append(event)
    if assisted:
        state.run.assisted = True
    if state.path is not None:
        with state.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(as_dict(event), ensure_ascii=False, default=str) + "\n")
    return event


def step(state: RunState, model: Any, tools: Optional[list[dict]] = None, router: Any = None) -> RunState:
    """Advance one turn: one model call, each tool call it made, then the user or a stop."""
    if state.stopped:
        return state
    state.turn += 1
    try:
        reply = model.query(state.messages, tools)
        emit(state, "model_call", {"reply": reply.model_dump(mode="json")},
             cost=Cost(model=reply.model or getattr(model, "name", None), usage=reply.usage))
        state.messages.append(_assistant_message(reply))
        for call in reply.tool_calls:
            _tool_call(state, call, router)
        if not reply.tool_calls:
            _user_turn(state, reply.content or "")
    except Exception as exc:  # the model or the Simulated user fell over, which is not the Run's verdict
        _crashed(state, exc, router)
        raise
    if not state.stopped and state.turn >= state.max_turns:
        _stop(state, "max_turns")
    return state


def run(state: RunState, model: Any, tools: Optional[list[dict]] = None, router: Any = None,
        max_steps: Optional[int] = None) -> RunState:
    """Call step until the Run stops: no tool calls and the user is done, or max turns."""
    steps = 0
    while not state.stopped:
        step(state, model, tools, router)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            if not state.stopped:  # a Run left by the caller's cap still says how it ended (D90)
                _stop(state, "max_steps")
            break
    return finish(state, router)


def _crashed(state: RunState, exc: Exception, router: Any) -> None:
    """An environment failure ends the Run in the file too: an error event, a reason and a footer."""
    emit(state, "error", {"class": "env_error", "message": f"{type(exc).__name__}: {exc}"})
    _stop(state, "env_error")
    finish(state, router)


def finish(state: RunState, router: Any = None) -> RunState:
    """Close the Run: the End state hash comes from the router's state, not from a tool."""
    if state.finished:
        return state
    state.finished = True
    if router is not None and hasattr(router, "state_hash"):
        state.run.end_state_hash = router.state_hash()
    _footer(state, router)
    return state


def _footer(state: RunState, router: Any) -> None:
    """One trailing line naming the Run and its Start and End state, which is what a Verdict reads."""
    if state.path is None:
        return
    footer = {"run_id": state.run.run_id, "task_id": state.run.task_id, "env_id": state.run.env_id,
              "trace_id": state.run.trace_id, "model": state.run.model, "seed": state.run.seed,
              "termination_reason": state.run.termination_reason,
              "end_state_hash": state.run.end_state_hash}
    if router is not None and hasattr(router, "world"):
        footer["start_state"] = getattr(router, "start_world", None) or {}
        footer["end_state"] = router.world()
    with state.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(footer, ensure_ascii=False, default=str) + "\n")


def _tool_call(state: RunState, call: Any, router: Any) -> None:
    args = dict(call.arguments or {})
    emit(state, "tool_call", {"id": call.id, "name": call.name, "args": args})
    if router is None:
        raise ValueError(f"the model called {call.name} but the loop was given no router")
    outcome = router.route(call.name, args)
    payload: dict = {"id": call.id, "name": call.name, "result": outcome.result}
    if outcome.error is not None:
        payload["error"] = as_dict(outcome.error)
    if getattr(outcome, "overlay_miss", None):  # D74, D88: a Starting state that could not be pinned
        payload["overlay_miss"] = outcome.overlay_miss
    emit(state, "tool_result", payload, route=outcome.route, assisted=outcome.assisted)
    state.run.route_counts[outcome.route] = state.run.route_counts.get(outcome.route, 0) + 1
    state.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                           "content": _as_text(outcome.result)})


def _user_turn(state: RunState, said: str) -> None:
    """No tool calls, so the Simulated user answers, or the Run stops here."""
    if state.user is None or _ends(state.user, said):
        _stop(state, _reason(said, "user_stop" if state.user is not None else "agent_stop"))
        return
    seen = len(getattr(state.user, "events", None) or [])
    answer = state.user.reply(state.messages)
    payload, assisted = _user_payload(state.user, answer, seen)
    emit(state, "user_turn", payload, assisted=assisted)
    if answer is None or _ends(state.user, answer):
        _stop(state, _reason(answer or "", "user_stop"))
        return
    state.messages.append({"role": "user", "content": answer})


def _user_payload(user: Any, answer: Optional[str], seen: int) -> tuple[dict, bool]:
    """The turn as the Simulated user recorded it: its tags, the fields it lacked, its assisted mark.

    Only an event the user appended during this turn is read (D77): a fact it could not give one
    turn ago does not tag this one, and a read that hit a synthetic row makes this turn Assisted.
    """
    payload: dict = {"text": answer or ""}
    events = getattr(user, "events", None) or []
    if len(events) <= seen:
        return payload, False
    recorded = events[-1]
    carried = recorded.payload or {}
    for key in ("unavailable_fields", "sources"):
        if carried.get(key):
            payload[key] = carried[key]
    tags = list(carried.get("tags") or [])
    if tags:
        payload.update({"tags": tags}, **{tag: True for tag in tags})
    return payload, bool(getattr(recorded, "assisted", False))


def _stop(state: RunState, reason: str) -> None:
    state.stopped = True
    state.run.termination_reason = reason
    emit(state, "stop", {"reason": reason, "termination_reason": reason, "turns": state.turn})


def _ends(user: Any, text: str) -> bool:
    return bool(getattr(user, "done", False)) or _marker(text) is not None


def _marker(text: str) -> Optional[str]:
    return next((marker for marker in STOP_MARKERS if marker in (text or "")), None)


def _reason(text: str, default: str) -> str:
    """D46: a Run handed to a human is a transfer by name, which is the class a Verdict reads."""
    return "transfer" if _marker(text) == TRANSFER else default


def _assistant_message(reply: Any) -> dict:
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
    }


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
