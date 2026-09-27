"""The agent user's own tools: what a person on the phone can reach, and nothing else (D214 rule 2).

The Builder reads the build's artifacts and the Examiner reads the Runs; this one reads its own
memory. `my_facts` is what it knows about itself, filtered by what it was actually asked, so a Task
with forty facts does not put forty in front of a model answering a question about one. `my_goal` is
why it called. `what_i_said` is its own earlier turns, which is the cheapest way to stop a model
contradicting itself two turns later. `end_run` is the only way to ask for an ending, and code
decides whether the ending holds (guards.EndProtocol). `consult` exists only on a Task whose
recording shows this user going and looking something up between its own turns, and where the
recording shows none the tool is not registered at all. `my_choices` answers a choice from what was
recorded, nearest first, and says which source answered (account.py).

`my_account` reads its own account, under these rules, all enforced in code and none in the prompt:
read only, over the Starting state as the Run opened; only this customer's rows; only the columns
some recorded user ever stated; and a value it read may be spoken only on a turn whose question
asked for it (guards.account_leak), else the turn goes to the floor. Beyond that no tool reads the
Environment's tables: the user sees what a person on the phone sees, so a Verifier cannot be passed
by a user that reads the world and hands the Candidate the row it was supposed to look up itself.
Both tools are registered only where the caller built them, so a user without them is the user it
was.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, NoArgs
from kullback.user import rules as rules_mod
from kullback.user.account import AccountView, ChoiceBook
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


class ChoiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="The kind of option you are asked to choose, in its own words.")


class AccountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what: str = Field(default="", description="What you look for on your account. Empty is all of it.")


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

    def __init__(self, ctx: TaskContext, said: Optional[list[str]] = None, *,
                 choices: Optional[ChoiceBook] = None, account: Optional[AccountView] = None):
        self.ctx = ctx
        self.said: list[str] = list(said or [])
        self.requested: Optional[str] = None
        self.consulted: list[str] = []
        self.choices = choices
        self.account = account
        # What the two readings handed out this Run, by value: a choice with its kind and source,
        # an account value with its column. The guards and the turn record read these (agent.py).
        self.chosen: dict[str, tuple[str, str]] = {}
        self.account_given: dict[str, str] = {}
        # Every tool call this turn, as names and argument keys, never values: the turn
        # record says whether the model read before it wrote, even where the turn fell
        # to the floor. Each recorded turn takes the list and clears it (agent.py).
        self.tool_calls: list[dict[str, Any]] = []

    def _used(self, tool: str, args: Iterable[str]) -> None:
        self.tool_calls.append({"tool": tool, "args": sorted(args)})

    def facts(self, asked: list[str]) -> str:
        self._used("my_facts", ("asked",) if asked else ())
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
        self._used("my_goal", ())
        return self.ctx.goal or NO_FACT

    def what_i_said(self) -> str:
        self._used("what_i_said", ())
        return "\n".join(f"{index + 1}. {text}" for index, text in enumerate(self.said)) or NO_FACT

    def end(self, kind: str) -> EndOut:
        """Record the request. Code decides; the tool says so, so the model does not read a
        successful call as the conversation being over and stop answering."""
        self._used("end_run", ("kind",))
        if kind not in rules_mod.USER_END_KINDS:
            return EndOut(requested=kind, accepted=False,
                          note="that is not one of " + ", ".join(rules_mod.USER_END_KINDS))
        self.requested = kind
        return EndOut(requested=kind, accepted=False,
                      note="recorded; whether the conversation ends is decided in code from what has "
                           "actually happened, so answer this turn as if it continues")

    def my_choices(self, kind: str) -> str:
        self._used("my_choices", ("kind",))
        if self.choices is None:
            return NO_FACT
        value, source, line = self.choices.answer(kind)
        if value is not None and source is not None:
            self.chosen[value] = (kind, source)
        return line

    def my_account(self, what: str = "") -> str:
        self._used("my_account", ("what",) if what else ())
        if self.account is None:
            return NO_FACT
        text, given = self.account.read(what)
        self.account_given.update(given)
        return text

    def consult(self, field: str) -> str:
        """Look one field up where the recording shows this user looking things up, else nothing."""
        self._used("consult", ("field",))
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

    async def choices(args: ChoiceArgs) -> TextOut:
        return TextOut(text=box.my_choices(args.kind))

    async def account(args: AccountArgs) -> TextOut:
        return TextOut(text=box.my_account(args.what))

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
    if box.choices is not None:
        tools.append(AgentTool(
            "my_choices", "What you chose, or would choose, when offered this kind of option, and "
                          "where that comes from. No record means you have no preference on file.",
            ChoiceArgs, TextOut, choices, render=_text))
    if box.account is not None:
        tools.append(AgentTool(
            "my_account", "Read your own account, the way you would on your own screen. Say a value "
                          "from it only when the other side asks for it.",
            AccountArgs, TextOut, account, render=_text))
    return tools


def _text(result: Any) -> str:
    return getattr(result, "text", "") or ""


def _end(result: EndOut) -> str:
    return f"{result.requested}: {result.note}"
