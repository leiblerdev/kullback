"""`AgentUser`: one Simulated user driven by a model, one turn per call, with the rules as its floor.

The harness is built the way the Builder's and the Examiner's are (`kullback.agent`), with the user
extension over one Task's curated context. What is different is the beat: the Builder runs one
session over a whole build, and this one runs one session per turn, because a turn is what the
Runner asks for and a user that keeps a transcript of its own thinking between turns is a user with
a second memory the recorded person did not have. So `reply` builds a harness, prompts it once, and
lets it call the user's tools until it answers with no tool call; that answer is the turn.

Then code takes over. The guards read the turn (guards.py) and a turn that fails one never leaves
this object: the rule-driven user answers the same beat instead, which is why the floor is never
removed and why a build that turns the model off is byte-identical to the build before D214. The
end is decided by the protocol and never by the model, which may only request a kind.

The object presents exactly the interface the Runner's loop asks of a Simulated user (`reply`,
`events`, `done`), so nothing in kullback/runner changes to run one (D214 rule 7).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Optional, Sequence

from kullback.agent.context import ContextConfig
from kullback.agent.events import MessageEnd
from kullback.agent.extensions import load_extensions
from kullback.agent.harness import AgentHarness
from kullback.ai.provider import Model
from kullback.runner import budget
from kullback.runner.records import Event
from kullback.user import guards as guards_mod
from kullback.user import rules as rules_mod
from kullback.user.context import TaskContext, volunteer_allowance
from kullback.user.extension import user_extension
from kullback.user.tools import Toolbox
from kullback.user.vocabulary import GENERIC, Vocabulary

# The one message a turn's harness is sent. It says what beat this is and nothing about the domain.
TURN_MESSAGE = ("Write your next turn in this conversation. Read what they just said, use your tools "
                "for anything about yourself you are not sure of, and answer with the turn itself and "
                "no tool call.")
OPENING_MESSAGE = ("Open the conversation: say why you got in touch, in your own words, and whatever "
                   "you would say without being asked. Answer with the turn itself and no tool call.")
# A turn is one model answer, however many tool calls it took to get there. The cap is on the tool
# calls, so a model that loops on its own tools cannot spend a build's ceiling on one turn.
MAX_TURNS_PER_REPLY = 6

# Why a turn was not the model's. Counted per round beside the guard reasons (D214 rule 3).
NO_MODEL = "agent_user_no_model"
MODEL_FAILED = "agent_user_model_failed"


def context_config(model: Optional[Model]) -> ContextConfig:
    """The context settings of a turn on this model: its own window, passed in because the agent core
    may not import `runner.budget` (D121, D124)."""
    return ContextConfig(window=budget.window_for(getattr(model, "name", None)))


class AgentUser:
    """A Simulated user a model drives, with the rule-driven one underneath it (D214 rule 3).

    `fallback` is the rule-driven user of the same Task, built from the same rules over the same
    world; it answers every beat the model's turn is dropped on, and its end decision stands on those
    beats because it is code that made it.
    """

    def __init__(self, ctx: TaskContext, fallback: Any, model: Optional[Model] = None, *,
                 vocab: Vocabulary = GENERIC, write_tools: Iterable[str] = (),
                 goal_writes: Optional[Iterable[str]] = None,
                 answer_strip: Optional[Callable[[str], tuple[str, list]]] = None,
                 record_values: Optional[dict] = None, trace: Any = None,
                 max_tool_turns: int = MAX_TURNS_PER_REPLY):
        self.ctx = ctx
        self.fallback = fallback
        self.model = model
        self.vocab = vocab
        self.trace = trace
        self.max_tool_turns = max_tool_turns
        self.guards = guards_mod.Guards(
            guards_mod.grounding_for(ctx.askable(), ctx.grounding()),
            record_values=record_values, strip=answer_strip)
        self.protocol = guards_mod.EndProtocol(goal_writes, write_tools)
        self.write_tools = frozenset(write_tools or ())
        self.box = Toolbox(ctx)
        self.events: list[Event] = []
        self.done = False
        self.end_reason: Optional[str] = None
        # What a round reports about this user (D214 rule 3): turns the model drove, turns dropped
        # and why, and the tools it reached for.
        self.counts: dict[str, int] = {"agent_user_turns": 0, "fallback_turns": 0,
                                       NO_MODEL: 0, MODEL_FAILED: 0}
        self._facts_said = 0
        self._turn = 0
        self._end_tagged = False

    # --- the Runner's interface ---------------------------------------------------------------

    def reply(self, transcript: list) -> str:
        """One turn: the model's if it passes every guard, the rule-driven user's if it does not."""
        self._turn += 1
        question = _last_assistant(transcript)
        self.box.requested = None
        asked = rules_mod.asked_fields(question, vocab=self.vocab)
        text = self._model_turn(transcript, question, asked)
        if text is None:
            return self._from_fallback(transcript)
        outcome = self.guards.check(
            text, facts_allowed=volunteer_allowance(self.trace, _rules_of(self.fallback), self._turn),
            facts_said=self._facts_said, asked=asked)
        if outcome.dropped:
            return self._from_fallback(transcript, dropped=outcome.reason)
        return self._speak(outcome.text, question, transcript, outcome.changed, asked)

    # --- the model's turn ----------------------------------------------------------------------

    def _model_turn(self, transcript: list, question: str, asked: Sequence[str] = ()) -> Optional[str]:
        """One model answer over a harness built for this beat, or None when there is no model."""
        if self.model is None:
            self.counts[NO_MODEL] += 1
            return None
        harness = self.harness(transcript, asked)
        opening = OPENING_MESSAGE if not self.events else TURN_MESSAGE
        try:
            return _last_text(harness, opening)
        except Exception:  # a turn the provider could not answer is a beat the rules take
            self.counts[MODEL_FAILED] += 1
            return None

    def harness(self, transcript: Sequence = (), asked: Iterable[str] = ()) -> AgentHarness:
        """The harness of one turn: the user extension over this Task's curated context."""
        harness = AgentHarness(model=self.model, max_turns=self.max_tool_turns,
                               context=context_config(self.model))
        load_extensions(harness, [user_extension(self.ctx, self.box, transcript, asked)])
        return harness

    # --- what is actually said -------------------------------------------------------------------

    def _from_fallback(self, transcript: list, dropped: Optional[str] = None) -> str:
        """The rule-driven user answers this beat, and its end decision stands (D214 rule 3)."""
        self.counts["fallback_turns"] += 1
        seen = len(self.fallback.events)
        text = self.fallback.reply(transcript)
        self.box.said.append(text)
        event = self.fallback.events[-1] if len(self.fallback.events) > seen else None
        payload = dict(event.payload or {}) if event is not None else {"text": text}
        payload["driver"] = "rules"
        if dropped:
            payload["agent_turn_dropped"] = dropped
        self.done = bool(getattr(self.fallback, "done", False))
        self.end_reason = getattr(self.fallback, "end_reason", None)
        self._record(payload, assisted=bool(getattr(event, "assisted", False)))
        return text

    def _speak(self, text: str, question: str, transcript: list, changed: Sequence[str],
               asked: Sequence[str] = ()) -> str:
        """The model's turn, guarded, with the end decided in code (D210, D214 rule 4)."""
        self.counts["agent_user_turns"] += 1
        self.box.said.append(text)
        # Counted the way the allowance is: what the turn volunteered past the question it answered.
        carried = self.guards.facts_in(text, self.guards.grounding.answering(asked))
        self._facts_said += carried
        made = guards_mod.writes_made(transcript, self.write_tools)
        kind = self.protocol.kind(question, said_anything=bool(text.strip()),
                                  had_nothing=self._had_nothing(text, question),
                                  made=made, requested=self.box.requested)
        tags = list(changed)
        if kind is not None:
            self.done = True
            self.end_reason = kind
            if not self._end_tagged:
                tags.append(kind)
                self._end_tagged = True
        self._record({"text": text, "driver": "agent", "facts": carried, "tags": tags,
                      "requested_end": self.box.requested, "user_end": kind}, assisted=False)
        return text

    def _had_nothing(self, text: str, question: str) -> bool:
        """This turn answered a question with no fact of its own, which is the scenario running out."""
        if not rules_mod._request_sentences(rules_mod._norm(question)):
            return False
        return self.guards.facts_in(text) == 0

    def _record(self, payload: dict, assisted: bool) -> None:
        payload.setdefault("tags", [])
        payload.setdefault("user_end", self.end_reason)
        self.events.append(Event(idx=len(self.events), type="user_turn", payload=payload,
                                 assisted=assisted))


def _rules_of(fallback: Any):
    return getattr(fallback, "rules", None)


def _last_assistant(transcript: Sequence) -> str:
    question = ""
    for message in transcript or ():
        if rules_mod._field_of(message, "role") == "assistant":
            question = rules_mod._field_of(message, "content")
    return question or ""


def _last_text(harness: AgentHarness, message: str) -> str:
    """Run one prompt to the end and answer with the last assistant text the model wrote."""
    said: list[str] = []

    async def go() -> None:
        async for event in harness.prompt(message):
            if isinstance(event, MessageEnd) and getattr(event.message, "role", "") == "assistant":
                text = (getattr(event.message, "content", "") or "").strip()
                if text:
                    said.append(text)

    asyncio.run(go())
    return said[-1] if said else ""


def run_user(ctx: TaskContext, fallback: Any, model: Optional[Model] = None, **kwargs) -> AgentUser:
    """One agent user over one Task, ready for the Runner's loop to ask turns of."""
    return AgentUser(ctx, fallback, model, **kwargs)
