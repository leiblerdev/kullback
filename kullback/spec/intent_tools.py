"""The intent miner's view of a Task: the user turns of every recording, and nothing else.

A recording is one Trace of the Task (tasks/<task_id>.json names them in `run_ids`). The miner is
shown, for each user turn, its index, its text and the agent's last spoken line before it, so it
knows what the user was answering. No tool call, tool result, write or end state is ever shown:
that is held here, in what `user_turns_of` returns, and not in the prompt (DESIGN.md rule 2).

Three tools over one Task: `list_recordings` and `user_turns` read, `add_facts` keeps facts in
memory. `add_facts` refuses a fact whose words the named turn does not carry, so every fact stays in
the user's words and the agent's line before it, and a volunteered fact in which the user added no
word of their own to that line.
Witnesses are code: another recording whose user volunteers the same content words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool, RetryableToolError
from kullback.runner.records import Task, Trace
from kullback.spec.schema import FactSource, IntentFact
from kullback.spec.text import APOSTROPHE_RE, STOPWORDS, TOKEN_RE, normalise

# A fact is carried by a turn when this share of its content words is in the turn.
CARRIED_SHARE = 0.5
# Another recording witnesses a fact when one of its user turns volunteers this share of its words.
WITNESS_SHARE = 0.6


@dataclass(frozen=True)
class UserTurn:
    """One user turn: its index in the recording, its text, and the agent's last line before it."""
    index: int
    text: str
    agent_before: str = ""


@dataclass
class TaskRecordings:
    """Every recording of one Task, as user turns only, keyed by recording id."""
    task_id: str
    recordings: dict[str, list[UserTurn]] = field(default_factory=dict)


def content_tokens(text: Optional[str]) -> set[str]:
    """The normalised content words of a text: the one reading both sides of every match use."""
    lowered = APOSTROPHE_RE.sub("", (text or "").lower())
    return {normalise(token) for token in TOKEN_RE.findall(lowered) if token not in STOPWORDS}


def share_in(words: set[str], other: set[str]) -> float:
    """The share of `words` found in `other`; 0 when there are no words to find."""
    return len(words & other) / len(words) if words else 0.0


def user_turns_of(trace: Trace) -> list[UserTurn]:
    """The user turns of a recording, each with the agent's last spoken text before it.

    Only a turn's `content` is read, and only for user and assistant turns: tool turns and the
    assistant's tool calls are never part of what this returns.
    """
    out: list[UserTurn] = []
    agent_said = ""
    for turn in trace.turns:
        text = (turn.content or "").strip()
        if turn.role == "assistant" and text:
            agent_said = text
        elif turn.role == "user" and text:
            out.append(UserTurn(index=turn.idx, text=text, agent_before=agent_said))
            agent_said = ""
    return out


def trace_index(workdir: Path) -> dict[str, Trace]:
    """Every recording under workdir/traces, by its trace id."""
    folder = Path(workdir) / "traces"
    out: dict[str, Trace] = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        trace = Trace.model_validate(json.loads(path.read_text(encoding="utf-8")))
        out[trace.trace_id] = trace
    return out


def task_of(workdir: Path, task_id: str) -> Task:
    """The Task record at workdir/tasks/<task_id>.json; a ValueError when there is none."""
    path = Path(workdir) / "tasks" / f"{task_id}.json"
    if "/" in task_id or task_id.startswith(".") or not path.is_file():
        raise ValueError(f"unknown Task {task_id!r}: it is not one of the Tasks under tasks/")
    return Task.model_validate(json.loads(path.read_text(encoding="utf-8")))


def recordings_of(workdir: Path, task_id: str, traces: Optional[dict[str, Trace]] = None) -> TaskRecordings:
    """The user turns of every recording of the Task that has at least one."""
    traces = trace_index(workdir) if traces is None else traces
    found = TaskRecordings(task_id=task_id)
    for run_id in task_of(workdir, task_id).run_ids:
        trace = traces.get(run_id)
        turns = user_turns_of(trace) if trace is not None else []
        if turns:
            found.recordings[run_id] = turns
    return found


def volunteers(turn: UserTurn, words: set[str], share: float = WITNESS_SHARE) -> bool:
    """Whether the user raised these words in this turn rather than echoing the agent's line."""
    own = content_tokens(turn.text) - content_tokens(turn.agent_before)
    return share_in(words, own) >= share


def _value_tokens(text: Optional[str]) -> set[str]:
    return {token for token in content_tokens(text) if any(ch.isdigit() for ch in token)}


def is_echo(text: str, stance: str, turn: UserTurn) -> bool:
    """Whether a fact carries the agent's proposal rather than the user's words: the user only
    accepted it, or a value of it (a token with a digit) was in the agent's line and not the user's."""
    if stance == "accepted":
        return True
    return bool(_value_tokens(text) & (_value_tokens(turn.agent_before) - _value_tokens(turn.text)))


def mark_echo(facts: list[IntentFact], found: TaskRecordings) -> list[IntentFact]:
    """The facts with `echo` set from their own turns; a fact whose turn is gone keeps its flag."""
    out = []
    for fact in facts:
        turn = next((t for t in found.recordings.get(fact.source.recording) or () if t.index == fact.source.turn),
                    None)
        out.append(fact if turn is None else fact.model_copy(update={"echo": is_echo(fact.text, fact.stance, turn)}))
    return out


def carried_share(text: str, turn: UserTurn) -> float:
    """The share of a fact's words its turn and the agent's line before it carry."""
    return share_in(content_tokens(text), content_tokens(turn.text) | content_tokens(turn.agent_before))


def witnesses(text: str, source: str, found: TaskRecordings, share: float = WITNESS_SHARE) -> list[str]:
    """The other recordings whose user volunteers the fact's words (DESIGN.md rule 5)."""
    words = content_tokens(text)
    return [recording for recording, turns in sorted(found.recordings.items())
            if recording != source and any(volunteers(turn, words, share) for turn in turns)]


def counts(fact: IntentFact) -> bool:
    """Rule 5: a volunteered fact counts; an accepted one only with a volunteering witness."""
    return fact.stance == "volunteered" or bool(fact.witnesses)


# --- the tools -------------------------------------------------------------------------------------

class ListRecordingsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str


class RecordingRow(BaseModel):
    recording: str
    user_turns: int


class ListRecordingsResult(BaseModel):
    summary: str
    recordings: list[RecordingRow] = Field(default_factory=list)


class UserTurnsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording: str


class TurnRow(BaseModel):
    turn: int
    agent_before: str
    user: str


class UserTurnsResult(BaseModel):
    summary: str
    recording: str
    turns: list[TurnRow] = Field(default_factory=list)


class FactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One thing the user wants, in the user's own words.")
    recording: str = Field(description="The recording the fact was said in.")
    turn: int = Field(description="The index of the user turn that says it, as user_turns gives it.")
    stance: Literal["volunteered", "accepted"] = Field(
        description="volunteered: the user raised it. accepted: the user agreed to what the agent's "
                    "line before the turn offered.")


class AddFactsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[FactArgs] = Field(description="The facts to keep; each is checked on its own.")


class FactRow(BaseModel):
    text: str
    fact_id: Optional[str] = None
    witnesses: list[str] = Field(default_factory=list)
    refused: Optional[str] = None


class AddFactsResult(BaseModel):
    summary: str
    rows: list[FactRow] = Field(default_factory=list)


def _render(result: BaseModel) -> str:
    rows = {k: v for k, v in result.model_dump(mode="json", exclude_none=True).items() if k != "summary"}
    return getattr(result, "summary", "") + "\n" + json.dumps(rows, ensure_ascii=False)


@dataclass
class MinerState:
    """What one mining session holds: the Task's recordings and the facts kept so far."""
    found: TaskRecordings
    facts: list[IntentFact] = field(default_factory=list)
    refused: int = 0


def _turn(state: MinerState, recording: str, index: int) -> UserTurn:
    turns = state.found.recordings.get(recording)
    if turns is None:
        raise ValueError(f"{recording!r} is not a recording of this Task; name one list_recordings gives")
    for turn in turns:
        if turn.index == index:
            return turn
    raise ValueError(f"turn {index} is not a user turn of {recording}; name a turn user_turns lists")


def _check_words(args: FactArgs, turn: UserTurn) -> None:
    """A fact must be in the words of its turn and the agent's line before it, and a volunteered
    fact must carry at least one word the user said that the agent's line did not."""
    words = content_tokens(args.text)
    said = content_tokens(turn.text)
    offered = content_tokens(turn.agent_before)
    if carried_share(args.text, turn) < CARRIED_SHARE:
        raise ValueError(f"turn {turn.index} of {args.recording} does not carry this fact; words not "
                         "said there: " + ", ".join(sorted(words - said - offered)[:12]))
    if args.stance == "volunteered" and not words & (said - offered):
        raise ValueError("every word of it the user said was in the agent's line before: mark it "
                         "accepted, or name the turn where the user raised it first")


def _keep(state: MinerState, args: FactArgs) -> IntentFact:
    """One fact kept, or a ValueError saying why not."""
    turn = _turn(state, args.recording, args.turn)
    text = " ".join(args.text.split())
    words = content_tokens(text)
    if not words:
        raise ValueError("the fact has no content words")
    if any(content_tokens(f.text) == words for f in state.facts):
        raise ValueError("this fact is already kept")
    _check_words(args, turn)
    fact = IntentFact(id=f"f{len(state.facts) + 1}", text=text, stance=args.stance,
                      source=FactSource(recording=args.recording, turn=turn.index),
                      witnesses=witnesses(text, args.recording, state.found),
                      echo=is_echo(text, args.stance, turn))
    state.facts.append(fact)
    return fact


def _add_facts(state: MinerState):
    async def add_facts(args: AddFactsArgs) -> AddFactsResult:
        rows: list[FactRow] = []
        for item in args.facts:
            try:
                fact = _keep(state, item)
            except ValueError as refusal:
                state.refused += 1
                rows.append(FactRow(text=item.text, refused=str(refusal)))
                continue
            rows.append(FactRow(text=fact.text, fact_id=fact.id, witnesses=fact.witnesses))
        kept = [r for r in rows if r.fact_id]
        marked = sum(1 for f in state.facts if f.id in {r.fact_id for r in kept} and not counts(f))
        summary = (f"kept {len(kept)} of {len(rows)} facts, {len(state.facts)} in all; "
                   f"{marked} accepted with no volunteering witness, kept and marked")
        if len(kept) < len(rows):
            summary += f"; {len(rows) - len(kept)} refused, each row says why: correct and send those again"
        return AddFactsResult(summary=summary, rows=rows)

    return add_facts


def _list_recordings(state: MinerState):
    async def list_recordings(args: ListRecordingsArgs) -> ListRecordingsResult:
        if args.task_id != state.found.task_id:
            raise RetryableToolError(f"this session mines {state.found.task_id}, not {args.task_id}",
                                     f"call list_recordings with task_id {state.found.task_id}")
        rows = [RecordingRow(recording=r, user_turns=len(t)) for r, t in sorted(state.found.recordings.items())]
        return ListRecordingsResult(summary=f"{len(rows)} recordings of {args.task_id}", recordings=rows)

    return list_recordings


def _user_turns(state: MinerState):
    async def user_turns(args: UserTurnsArgs) -> UserTurnsResult:
        turns = state.found.recordings.get(args.recording)
        if turns is None:
            raise RetryableToolError(f"{args.recording!r} is not a recording of this Task",
                                     "name a recording from list_recordings")
        rows = [TurnRow(turn=t.index, agent_before=t.agent_before, user=t.text) for t in turns]
        return UserTurnsResult(summary=f"{len(rows)} user turns of {args.recording}",
                               recording=args.recording, turns=rows)

    return user_turns


def miner_tools(state: MinerState) -> list[AgentTool]:
    """The miner's three tools over one Task."""
    return [
        AgentTool("list_recordings", "The recordings of the Task, each with its number of user turns.",
                  ListRecordingsArgs, ListRecordingsResult, _list_recordings(state), render=_render),
        AgentTool("user_turns", "The user turns of one recording: index, the agent's line just before, "
                  "and what the user said.", UserTurnsArgs, UserTurnsResult, _user_turns(state), render=_render),
        AgentTool("add_facts", "Keep facts of what the user wants, each with the recording and user turn "
                  "that say it and whether the user volunteered it or accepted an offer.",
                  AddFactsArgs, AddFactsResult, _add_facts(state), render=_render),
    ]


def tool_names() -> tuple[str, ...]:
    return ("list_recordings", "user_turns", "add_facts")


__all__ = ["CARRIED_SHARE", "WITNESS_SHARE", "MinerState", "TaskRecordings", "UserTurn", "carried_share", "content_tokens",
           "counts", "miner_tools", "recordings_of", "share_in", "task_of", "tool_names", "trace_index",
           "user_turns_of", "volunteers", "witnesses"]

