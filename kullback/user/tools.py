"""The agent user's own tools: what a person on the phone can reach, and nothing else (D214 rule 2).

The Builder reads the build's artifacts and the Examiner reads the Runs; this one reads its own
memory. `my_facts` is what it knows about itself, filtered by what it was actually asked, so a Task
with forty facts does not put forty in front of a model answering a question about one. `my_goal` is
why it called. `what_i_said` is its own earlier turns, which is the cheapest way to stop a model
contradicting itself two turns later. `end_run` is the only way to ask for an ending, and code
decides whether the ending holds (guards.EndProtocol). `consult` exists only on a Task whose
recording shows this user going and looking something up between its own turns, and where the
recording shows none the tool is not registered at all.

No tool here reads the Environment's tables. That is the whole design: the user sees what a person
on the phone sees, so a Verifier cannot be passed by a user that reads the world and hands the
Candidate the row it was supposed to look up itself.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, NoArgs
from kullback.user import rules as rules_mod
from kullback.user.context import TaskContext

NO_FACT = "You have no record of that."
NOT_CONSULTED = "You have nothing to look that up in."


class FactsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asked: list[str] = Field(default_factory=list,
                             description="The fields the other side asked for. Empty is everything you hold.")


class EndArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="One of " + ", ".join(rules_mod.USER_END_KINDS) + ".")


class ConsultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="The field you go and look up.")


class TextOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""


class EndOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: str = ""
    accepted: bool = False
    note: str = ""


class Toolbox:
    """The state one turn's tools read and write: the context, what has been said, what was requested.

    One instance per Run, so `what_i_said` grows with the conversation and `requested` is cleared at
    the start of each turn by the caller (agent.py).
    """

    def __init__(self, ctx: TaskContext, said: Optional[list[str]] = None):
        self.ctx = ctx
        self.said: list[str] = list(said or [])
        self.requested: Optional[str] = None
        self.consulted: list[str] = []

    def facts(self, asked: list[str]) -> str:
        wanted = [field for field in asked if field]
        rows = [f for f in self.ctx.askable() if not wanted or f.field in wanted]
        if not rows:
            return NO_FACT
        lines = [f"{f.field}: {f.value}" for f in rows]
        missing = [field for field in wanted if not any(f.field == field for f in rows)]
        for field in missing:
            lines.append(f"{field}: " + (rules_mod.RECORD_LINE.format(rules_mod._words(field))
                                         if field in self.ctx.record_fields else NO_FACT))
        return "\n".join(lines)

    def goal(self) -> str:
        return self.ctx.goal or NO_FACT

    def what_i_said(self) -> str:
        return "\n".join(f"{index + 1}. {text}" for index, text in enumerate(self.said)) or NO_FACT

    def end(self, kind: str) -> EndOut:
        """Record the request. Code decides; the tool says so, so the model does not read a
        successful call as the conversation being over and stop answering."""
        if kind not in rules_mod.USER_END_KINDS:
            return EndOut(requested=kind, accepted=False,
                          note="that is not one of " + ", ".join(rules_mod.USER_END_KINDS))
        self.requested = kind
        return EndOut(requested=kind, accepted=False,
                      note="recorded; whether the conversation ends is decided in code from what has "
                           "actually happened, so answer this turn as if it continues")

    def consult(self, field: str) -> str:
        """Look one field up where the recording shows this user looking things up, else nothing."""
        if field not in self.ctx.consultations:
            return NOT_CONSULTED
        fact = next((f for f in self.ctx.askable() if f.field == field), None)
        if fact is None:
            return NO_FACT
        self.consulted.append(field)
        return f"{field}: {fact.value}"


def user_tools(box: Toolbox) -> list[AgentTool]:
    """The tools of one Run's user; `consult` only where the recording justified it (D214 rule 2)."""

    async def facts(args: FactsArgs) -> TextOut:
        return TextOut(text=box.facts(list(args.asked)))

    async def goal(_: NoArgs) -> TextOut:
        return TextOut(text=box.goal())

    async def said(_: NoArgs) -> TextOut:
        return TextOut(text=box.what_i_said())

    async def end(args: EndArgs) -> EndOut:
        return box.end(args.kind)

    async def consult(args: ConsultArgs) -> TextOut:
        return TextOut(text=box.consult(args.field))

    tools = [
        AgentTool("my_facts", "What you know about yourself. Pass the fields the other side asked for; "
                              "pass nothing to see everything you hold.",
                  FactsArgs, TextOut, facts, render=_text),
        AgentTool("my_goal", "Why you got in touch, in the words you used.", NoArgs, TextOut, goal,
                  render=_text),
        AgentTool("what_i_said", "Your own earlier turns in this conversation, in order.",
                  NoArgs, TextOut, said, render=_text),
        AgentTool("end_run", "Ask to end the conversation with one of the four kinds. Code decides "
                             "whether it ends; answer this turn either way.",
                  EndArgs, EndOut, end, render=_end),
    ]
    if box.ctx.consultations:
        tools.append(AgentTool(
            "consult", "Go and look one field up, the way you did in this conversation before: "
                       + ", ".join(box.ctx.consultations) + ".",
            ConsultArgs, TextOut, consult, render=_text))
    return tools


def _text(result: Any) -> str:
    return getattr(result, "text", "") or ""


def _end(result: EndOut) -> str:
    return f"{result.requested}: {result.note}"
