"""The intent miner: a Task's SpecIntent from the user's words only, provenance on every fact.

One agent session per Task on the Examiner's model path (the adapter `provider.live_model` gives
the CLI). It reads the user turns of every recording through `intent_tools` and keeps facts with
`add_fact`; code checks each fact against the turn it names and finds its witnesses. The session
stops on the model's stop, on the call cap, or when its spend reaches the Task's ceiling in USD.
Whatever facts were kept by then are the Intent, written through `schema.save_spec`; a Spec that
already holds checks keeps them. The benchmark's own note is never read here (DESIGN.md rule 6).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kullback.agent.context import ContextConfig
from kullback.agent.harness import AgentHarness
from kullback.agent.loop import Hooks
from kullback.runner import budget
from kullback.runner.records import Usage
from kullback.spec.intent_tools import MinerState, miner_tools, recordings_of
from kullback.spec.schema import Spec, SpecIntent, intent_text, load_spec, save_spec

STAGE = "spec"
MAX_CALLS = 40
MAX_TURNS = 30

WHAT_YOU_RECEIVE = """\
# What you receive
One Task: the recorded conversations in which one user asked an agent for the same thing. You see
only what the users said, each line with the agent's line just before it, so you know what a user
was answering. You never see what the agent did with its tools. Your job is the Intent: what the
user came for, as short facts in the user's own words, which a new agent must meet to serve them."""

TOOLS_TEXT = """\
# Tools
- list_recordings: the Task's recordings and how many user turns each has.
  Example: list_recordings({"task_id": "task_1"})
- user_turns: one recording's user turns, each with its index and the agent's line before it.
  Example: user_turns({"recording": "rec-a"})
- add_facts: keep facts, each naming the recording and user turn that say it, and its stance.
  Example: add_facts({"facts": [{"text": "wants the second shipment sent to the new flat",
  "recording": "rec-a", "turn": 3, "stance": "volunteered"}]})"""

EXAMPLES = """\
# Examples
- The user says "I moved, please send the parcel to my new flat". Fact: "wants the parcel sent to
  the new flat", volunteered.
- The agent says "Shall I also refund the fee to your card?" and the user says "Yes please".
  Fact: "agrees to the fee refunded to the card", accepted, on the user's turn.
- The user says "don't cancel the other booking". Fact: "does not want the other booking
  cancelled", volunteered. A refusal is a fact too.
- The agent explains a delay and the user asks "so when will it arrive then?". That reacts to the
  agent's answer; it is not what the user came for. No fact."""

CHOICE_RULE = """\
# Choosing facts
Read every recording before you keep anything; different users of the same Task say different
parts of it. Keep what the user came for: each thing to be done, each value or condition that
shapes it (which item, which option, how to pay), each refusal, and each question the user called
to have answered. Keep the user's values exactly (names, numbers, dates, codes); paraphrase only to
make the line stand alone. Keep the details a user gives to identify themselves too: a new user
acting from the Intent needs them. Leave out follow-up questions that react to the agent's last
answer, greetings and thanks. Mark a fact accepted
when the user only said yes to the agent's line; volunteered when the user raised it or supplied
the value. One fact per thing, kept once, at the turn where a user first says it."""

FEEDBACK_SHAPE = """\
# What the tools answer
add_facts answers one row per fact: its id and the other recordings whose user volunteered the same
words, or why it was refused (the turn does not carry those words, the user only repeated the
agent's line, the fact is kept already, the turn is no user turn). Correct a refused fact and send
it again, or drop it when the refusal shows it is not the user's."""

STOP_RULE = """\
# When to stop
Keep all facts in one add_facts call once every recording has been read, then resend only the
refused ones you can correct. Stop there and answer in one line: how many facts you kept."""


def intent_system_prompt() -> str:
    """The miner's prompt in the repo's order, the stop rule last."""
    return "\n\n".join([WHAT_YOU_RECEIVE, TOOLS_TEXT, EXAMPLES, CHOICE_RULE, FEEDBACK_SHAPE, STOP_RULE])


def intent_message(task_id: str, recordings: int) -> str:
    return (f"Task {task_id} has {recordings} recordings. Read them all with your tools and keep "
            "the facts of what the user wants.")


@dataclass
class MineResult:
    """One mining session: the Intent it wrote and what it spent to write it."""
    intent: SpecIntent
    tool_calls: dict[str, int] = field(default_factory=dict)
    refusals: int = 0
    spent_usd: float = 0.0
    stopped: Optional[str] = None
    path: Optional[Path] = None


class _Watch:
    """Counts tool calls and prices every model call; stops the session at the cap or the ceiling."""

    def __init__(self, model_id: Optional[str], ceiling_usd: float, max_calls: int):
        self.model_id = model_id
        self.ceiling_usd = ceiling_usd
        self.max_calls = max_calls
        self.harness: Optional[AgentHarness] = None
        self.calls: dict[str, int] = {}
        self.spent = 0.0
        self.stopped: Optional[str] = None

    def _stop(self, why: str) -> None:
        self.stopped = self.stopped or why
        if self.harness is not None:
            self.harness.cancel()

    def cap(self, call: Any) -> None:
        """The tool_call hook: the call past the cap is blocked before it runs."""
        if sum(self.calls.values()) >= self.max_calls:
            self._stop(f"call cap {self.max_calls} reached")
            raise RuntimeError(self.stopped)

    def __call__(self, event: Any) -> None:
        if event.type == "tool_execution_end":
            self.calls[event.tool_name] = self.calls.get(event.tool_name, 0) + 1
            return
        message = getattr(event, "message", None)
        usage = getattr(message, "usage", None)
        if event.type != "message_end" or getattr(message, "role", None) != "assistant" \
                or not isinstance(usage, Usage):
            return
        self.spent += budget.call_cost(usage, budget.sent_wire_id(self.model_id, getattr(message, "model", None)))
        if self.spent >= self.ceiling_usd:
            self._stop(f"ceiling {self.ceiling_usd:.2f} USD reached")


def _save(workdir: Path, intent: SpecIntent) -> Path:
    """The Intent part of the Task's Spec file; checks already on the Spec stay."""
    spec = load_spec(workdir, intent.task_id)
    spec = Spec(task_id=intent.task_id, intent=intent) if spec is None \
        else spec.model_copy(update={"intent": intent, "version": spec.version + 1})
    return save_spec(workdir, spec)


def mine_intent(workdir: Path, task_id: str, model: Any, ceiling_usd: float,
                max_calls: int = MAX_CALLS, traces: Optional[dict] = None) -> MineResult:
    """Mine one Task's Intent, write it to workdir/spec/<task_id>.json, and say what it cost."""
    workdir = Path(workdir)
    state = MinerState(found=recordings_of(workdir, task_id, traces))
    model_id = getattr(model, "name", None)
    watch = _Watch(model_id, ceiling_usd, max_calls)
    harness = AgentHarness(model=model, system=intent_system_prompt(), tools=miner_tools(state),
                           max_turns=MAX_TURNS, hooks=Hooks(tool_call=[watch.cap]),
                           context=ContextConfig(window=budget.window_for(model_id)))
    watch.harness = harness
    harness.subscribe(watch)
    harness.subscribe(budget.subscriber(workdir, STAGE, model_id))

    async def go() -> None:
        async for _ in harness.prompt(intent_message(task_id, len(state.found.recordings))):
            pass

    if state.found.recordings:
        asyncio.run(go())
    else:
        watch.stopped = "no recording with a user turn"
    intent = SpecIntent(task_id=task_id, facts=list(state.facts), text=intent_text(state.facts),
                        refusals=state.refused,
                        model=model_id, written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return MineResult(intent=intent, tool_calls=dict(watch.calls), refusals=state.refused, spent_usd=watch.spent,
                      stopped=watch.stopped, path=_save(workdir, intent))


__all__ = ["MAX_CALLS", "MineResult", "STAGE", "intent_message", "intent_system_prompt", "mine_intent"]
