"""Snapshot for the step split (G2): full serialised RunState after every step.

Builds 30+ scripted Runs on an invented widget domain (a scripted model returning fixed
replies, a real Router over two invented tools, a scripted user) and writes the full
serialised RunState after every step to one JSON file. Run on origin/main code and on the
patched code; the diff must be byte for byte empty.

Usage: uv run python scripts/step_split_snapshot.py --out <file>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import kullback.runner.loop as loop_mod
from kullback.ai.provider import RetryExhausted, TestModel
from kullback.runner import loop
from kullback.runner.records import Event, ToolSig, as_dict
from kullback.runner.route import Router
from kullback.runner.state import StateView

DOMAIN_TOOLS = [{"name": "describe_widget"}, {"name": "rename_widget"}]


def tools_module() -> types.ModuleType:
    """Two invented widget tools: one read, one write that refuses an empty label."""
    module = types.ModuleType("snapshot_tools")

    def describe_widget(state, widget_id):
        row = state.row("widgets", widget_id)
        if row is None:
            raise KeyError(widget_id)
        return row

    def rename_widget(state, widget_id, label):
        if not label:
            raise ValueError("a widget needs a label")
        row = dict(state.row("widgets", widget_id) or {})
        row["label"] = label
        state.shared.setdefault("widgets", {})[str(widget_id)] = row
        return row

    module.describe_widget = describe_widget
    module.rename_widget = rename_widget
    return module


def sigs() -> list[ToolSig]:
    return [ToolSig(name="describe_widget"), ToolSig(name="rename_widget")]


def shared() -> dict:
    return {"widgets": {"w1": {"id": "w1", "label": "plain", "status": "on shelf"}}}


def make_router(**kwargs) -> Router:
    return Router(
        env_tools_module=tools_module(),
        recordings=None,
        starting_state=StateView(shared=copy.deepcopy(shared())),
        overlay=None,
        tool_sigs=sigs(),
        **kwargs,
    )


class ScriptedUser:
    """A scripted Simulated user: fixed answers, optional recorded payload per turn."""

    def __init__(self, answers, payload=None, assisted=False, explode_on=0):
        self.answers = list(answers)
        self.payload = dict(payload or {})
        self.assisted = assisted
        self.explode_on = explode_on
        self.turns = 0
        self.events: list[Event] = []
        self.done = False

    def reply(self, transcript):
        self.turns += 1
        if self.explode_on and self.turns >= self.explode_on:
            raise RuntimeError("user sim down")
        answer = self.answers.pop(0) if self.answers else "###STOP###"
        if self.payload:
            self.events.append(Event(idx=len(self.events), type="user_turn",
                                     payload=dict(self.payload, text=answer),
                                     assisted=self.assisted))
        return answer


class BoomModel(TestModel):
    def __init__(self, replies, error):
        super().__init__(replies)
        self.error = error

    def query(self, messages, tools=None, config=None):
        if self.index >= len(self.replies):
            raise self.error
        return super().query(messages, tools, config)


def dump_state(state) -> dict:
    return {
        "run_id": state.run.run_id,
        "messages": copy.deepcopy(state.messages),
        "tools": copy.deepcopy(state.tools),
        "turn": state.turn,
        "stopped": state.stopped,
        "finished": state.finished,
        "termination_reason": state.run.termination_reason,
        "end_state_hash": state.run.end_state_hash,
        "route_counts": dict(sorted(state.run.route_counts.items())),
        "assisted": state.run.assisted,
        "events": [as_dict(e) for e in state.run.events],
    }


def drive(make_state, model, router, use_run=False, max_steps=None) -> list[dict]:
    """Step (or run) while recording the full state after every step, crash included."""
    state = make_state()
    dumps = [dump_state(state)]
    original = loop_mod.step

    def recording_step(s, m, t=None, r=None):
        try:
            return original(s, m, t, r)
        finally:
            dumps.append(dump_state(s))

    loop_mod.step = recording_step
    try:
        if use_run:
            loop.run(state, model, DOMAIN_TOOLS, router, max_steps=max_steps)
        else:
            for _ in range(30):
                if state.stopped:
                    break
                loop.step(state, model, DOMAIN_TOOLS, router)
            loop.finish(state, router)
    except Exception as exc:  # noqa: BLE001 - the crash is part of the snapshot
        dumps.append({"crashed": f"{type(exc).__name__}: {exc}"})
    finally:
        loop_mod.step = original
    return dumps


def fresh(run_id, **kwargs):
    def make():
        return loop.new_run_state(run_id, first_user="what is on shelf w1", **kwargs)
    return make


def scenarios() -> list[tuple[str, list[dict]]]:
    out = []

    def add(name, dumps):
        out.append((name, dumps))

    # 1-3. plain text turns, no tools.
    add("text_then_agent_stop", drive(
        fresh("text-stop"), TestModel([{"content": "w1 is on the shelf."}]), make_router()))
    add("text_transfer_to_human", drive(
        fresh("text-transfer"),
        TestModel([{"content": "I will hand you over. ###TRANSFER###"}]), make_router()))
    add("text_then_user_stop", drive(
        fresh("text-user", user=ScriptedUser(["###STOP###"])),
        TestModel([{"content": "anything else?"}], loop=True), make_router()))

    # 4-6. tool traffic: one call, several calls in one turn, unknown tool error.
    one = [{"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]},
        {"content": "w1 is on the shelf."}]
    add("one_tool_call", drive(fresh("one-tool"), TestModel(one), make_router()))
    several = [{"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}},
        {"id": "c2", "name": "rename_widget",
         "arguments": {"widget_id": "w1", "label": "shiny"}},
        {"id": "c3", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]},
        {"content": "renamed and rechecked."}]
    add("several_tool_calls_one_turn", drive(
        fresh("several-tools"), TestModel(several), make_router()))
    unknown = [{"content": None, "tool_calls": [
        {"id": "c1", "name": "levitate_widget", "arguments": {}}]},
        {"content": "sorry, I cannot do that."}]
    add("unknown_tool_error_then_carry_on", drive(
        fresh("unknown-tool"), TestModel(unknown), make_router()))

    # 7-8. a write the body refuses, and a write that lands.
    refused = [{"content": None, "tool_calls": [
        {"id": "c1", "name": "rename_widget", "arguments": {"widget_id": "w1", "label": ""}}]},
        {"content": "the shelf refused an empty label."}]
    add("refused_write_then_carry_on", drive(
        fresh("refused-write"), TestModel(refused), make_router()))
    write = [{"content": None, "tool_calls": [
        {"id": "c1", "name": "rename_widget", "arguments": {"widget_id": "w1", "label": "shiny"}}]},
        {"content": "renamed."}]
    add("write_lands_end_state_moves", drive(
        fresh("write-lands"), TestModel(write), make_router()))

    # 9-13. the Simulated user: plain end, transfer, tagged turns, refusal counts, assisted read.
    add("user_answers_then_ends", drive(
        fresh("user-chat", user=ScriptedUser(["w1 please.", "###STOP###"])),
        TestModel([{"content": "which one?"}], loop=True), make_router()))
    add("user_hands_to_human", drive(
        fresh("user-transfer", user=ScriptedUser(["###TRANSFER###"])),
        TestModel([{"content": "anything else?"}], loop=True), make_router()))
    add("user_turn_carries_sources", drive(
        fresh("user-sources",
              user=ScriptedUser(["w1."], payload={"sources": {"widget": "world"}}, assisted=True)),
        TestModel([{"content": "which one?"}], loop=True), make_router(), ))
    add("user_turn_carries_refusals", drive(
        fresh("user-refused", user=ScriptedUser(["w1."], payload={"refused": 0, "refused_so_far": 2})),
        TestModel([{"content": "which one?"}], loop=True), make_router()))
    add("user_turn_carries_unavailable", drive(
        fresh("user-unavail", user=ScriptedUser(
            ["no label."],
            payload={"tags": ["fact_unavailable"], "unavailable_fields": ["label"]})),
        TestModel([{"content": "what label?"}], loop=True), make_router()))

    # 14-16. multi-turn mixes of tools and user turns.
    mixed = [
        {"content": None, "tool_calls": [
            {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]},
        {"content": "it is plain; want it renamed?"},
        {"content": None, "tool_calls": [
            {"id": "c2", "name": "rename_widget",
             "arguments": {"widget_id": "w1", "label": "shiny"}}]},
        {"content": "done."},
    ]
    add("mixed_tools_and_user_turns", drive(
        fresh("mixed", user=ScriptedUser(["yes please.", "thanks."])),
        TestModel(mixed), make_router()))
    add("empty_content_tool_turn", drive(
        fresh("empty-content"),
        TestModel([{"tool_calls": [
            {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]},
            {"content": "here it is."}]), make_router()))
    add("two_errors_then_answer", drive(
        fresh("two-errors"), TestModel([
            {"content": None, "tool_calls": [
                {"id": "c1", "name": "levitate_widget", "arguments": {}},
                {"id": "c2", "name": "rename_widget",
                 "arguments": {"widget_id": "w1", "label": ""}}]},
            {"content": "neither worked."}]), make_router()))

    # 17-19. assisted runs through the stand-in.
    add("stand_in_marks_assisted", drive(
        fresh("assisted"), TestModel([
            {"tool_calls": [{"id": "c1", "name": "polish_widget", "arguments": {}}]},
            {"content": "done"}]),
        make_router(stand_in_model=TestModel(['{"label": "shiny"}']))))
    add("stand_in_then_user", drive(
        fresh("assisted-user", user=ScriptedUser(["thanks."])), TestModel([
            {"tool_calls": [{"id": "c1", "name": "polish_widget", "arguments": {}}]},
            {"content": "polished."}]),
        make_router(stand_in_model=TestModel(['{"label": "shiny"}']))))

    # 20-23. stops by count: max turns, turn budget exhaustion, caller cap, noop on stopped.
    looping_tools = TestModel(
        [{"tool_calls": [{"id": "c1", "name": "describe_widget",
                          "arguments": {"widget_id": "w1"}}]}],
        loop=True)
    add("max_turns_stops_tool_loop", drive(
        fresh("max-turns", max_turns=3), looping_tools, make_router()))
    looping_text = TestModel([{"content": "still here."}], loop=True)
    add("turn_budget_exhausted_on_text", drive(
        fresh("turn-budget", user=ScriptedUser(["go on."]), max_turns=4),
        looping_text, make_router()))
    add("caller_cap_max_steps", drive(
        fresh("caller-cap", max_turns=50), TestModel(
            [{"tool_calls": [{"id": "c1", "name": "describe_widget",
                              "arguments": {"widget_id": "w1"}}]}],
            loop=True),
        make_router(), use_run=True, max_steps=2))

    def stopped_noop():
        state = loop.new_run_state("stopped-noop", first_user="what is on shelf w1")
        model, router = TestModel([{"content": "w1 is on the shelf."}]), make_router()
        dumps = [dump_state(state)]
        original = loop_mod.step
        loop_mod.step = lambda s, m, t=None, r=None: (original(s, m, t, r), dumps.append(dump_state(s)))[0]
        try:
            loop.run(state, model, DOMAIN_TOOLS, router)
            before = dump_state(state)
            loop.step(state, model, DOMAIN_TOOLS, router)
            dumps.append(dump_state(state))
            assert dumps[-1] == before
        finally:
            loop_mod.step = original
        dumps.append(dump_state(state))
        return dumps

    add("step_on_stopped_run_is_noop", stopped_noop())

    # 24-27. crashes: model raises, provider error, user raises, tool without router.
    add("model_exception_ends_run", drive(
        fresh("model-boom"),
        BoomModel([{"content": None, "tool_calls": [
            {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]}],
            RuntimeError("provider down")),
        make_router()))
    add("provider_error_carries_status", drive(
        fresh("provider-status"),
        BoomModel([], RetryExhausted("invented/test: 5 attempts failed", status=503, attempts=5)),
        make_router()))
    add("user_exception_ends_run", drive(
        fresh("user-boom", user=ScriptedUser(["hi."], explode_on=1)),
        TestModel([{"content": "anything else?"}], loop=True), make_router()))
    add("tool_call_without_router_is_refused", drive(
        fresh("no-router"), TestModel(one), None))

    # 28-32. bookkeeping variants: system prompt, tool specs, seed and ids, second write.
    add("system_prompt_and_tool_specs", drive(
        lambda: loop.new_run_state("prompt-tools", system_prompt="you shelve widgets",
                                   first_user="what is on shelf w1"),
        TestModel(one), make_router()))
    add("bare_run_no_prompt_no_user", drive(
        lambda: loop.new_run_state("bare"), TestModel(one), make_router()))
    add("run_record_fields_carried", drive(
        lambda: loop.new_run_state("fields", first_user="what is on shelf w1",
                                   env_id="e1", task_id="t1", seed=7, model="m"),
        TestModel(one), make_router()))
    add("second_write_same_run", drive(
        fresh("two-writes"), TestModel([
            {"content": None, "tool_calls": [
                {"id": "c1", "name": "rename_widget",
                 "arguments": {"widget_id": "w1", "label": "shiny"}}]},
            {"content": None, "tool_calls": [
                {"id": "c2", "name": "rename_widget",
                 "arguments": {"widget_id": "w1", "label": "matte"}}]},
            {"content": "renamed twice."}]), make_router()))
    add("run_driven_whole_then_compared", drive(
        fresh("whole-run", user=ScriptedUser(["yes please."])), TestModel(mixed),
        make_router(), use_run=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step-split-snapshot.json")
    args = ap.parse_args()

    runs = [{"scenario": name, "steps": dumps} for name, dumps in scenarios()]
    payload = json.dumps({"runs": runs}, sort_keys=True, indent=1, default=str)
    Path(args.out).write_text(payload + "\n", encoding="utf-8")
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"runs={len(runs)} steps={sum(len(r['steps']) for r in runs)} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
