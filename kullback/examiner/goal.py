"""The Examiner's own expected end state for one Task (the goal), from other evidence than the writer.

One model session per Task says what the world should look like after a right solve: the cells a
right solve must write, the cells where more than one value is right, what stays untouched, what the
agent must tell the user, whether the policy says to refuse, and what this evidence cannot decide.

What the session sees is held here, in `load_goal_inputs`: the user's own words in every recording
(never the agent's lines, its tool calls or their results), the policy, the Starting state through
read-only lookups, and the tool signatures. It never sees the Intent or the facts the miner wrote,
the writer's items, a Verifier, a Run, a Reference or an End state: none of them is loaded, and
`writer_tools.refuse_run_inputs` refuses a Starting state that carries one.

Report only: nothing reads exam/goal/ for trust, rulings or the writer yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from kullback.spec import items as I
from kullback.spec import writer_tools

# The read tools of the writer's world (writer_tools.WORLD_TOOLS), and no tool that adds anything.
READS = ("read_schema", "find_rows", "read_row", "read_policy")
GOAL_TOOLS = [tool for tool in writer_tools.WORLD_TOOLS if tool["name"] in READS]
# The most read rounds one goal session may spend before it must answer.
MAX_GOAL_ROUNDS = 16
GOAL_DIR = Path("exam") / "goal"
STAGE = "goal"
NEW_ROW = "new:"
ANY_VALID = "any valid"
KEYS = ("required", "choice", "untouched", "told", "refuse", "unknown")

WHAT_YOU_RECEIVE = """\
You decide what the world should look like after a right solve of one customer request. You receive \
what the user said in every recording of this request (the user's words only, never the agent's), the \
policy the agent works under, and the tools the agent has (name, kind, argument names). Different \
recordings are the same user with the same wishes; one may say more than another. The user's replies \
may answer agent lines you do not see: a bare yes or a reply to an offer is not a wish of its own."""

TOOLS_TEXT = """\
Your tools read the Starting state, the world before the agent acts, and the policy. They change nothing.
- read_schema: the tables (row count, fields) and the agent's tools.
- find_rows: rows holding a value, in one field or any field, in one table or every table.
- read_row: one row by table and key.
- read_policy: the policy's headings, or the text under one heading.
One example call: find_rows {"value": "<a value the user said>"} lists every {table, key} holding it."""

RULES = """\
How to decide:
1. Start from what the user asked for in their own words, before any refusal, failure or offer you can \
infer. A wish the user only reached after the agent declined or proposed something is not the goal.
2. Hold each request against the policy. A request the policy forbids is refused, not done; a request the \
policy allows is done the way the policy says, with the values the user fixed.
3. Find every row a request touches in the Starting state. Derive each written value from the user's words, \
the policy and the rows you read: compute what a tool would set (a status, an amount, a list after the \
change) from the row as it stands. Never invent a value you did not read or compute.
4. Be honest about what you cannot settle. An unknown beats a guess."""

CHOICE_RULE = """\
Where each cell goes:
- required: a right solve writes exactly this value, and the user's words or a policy line fix it.
- choice: a right solve writes this cell, but more than one value is right (the user left it open, or \
the policy allows several); give the allowed set, or "any valid" when any value a tool accepts is right.
- unknown: what this evidence cannot decide (the user's wish is unclear, a row you need is not there, \
the user would choose only after hearing something you cannot see). Set cannot_tell true when the goal \
of the request itself is unclear; then list what you can still say."""

OUTPUT = """\
Answer with one JSON object and nothing else:
{"cannot_tell": false, "why": "<the goal in one or two lines>",
 "required": [{"table": "", "row": "<key, or new:1 for a row the solve creates>", "field": "<field or \
dotted path>", "value": <the value after the solve>, "user": "<the user's words it rests on>", "policy": \
"<or the policy line>"}],
 "choice": [{"table": "", "row": "", "field": "", "allowed": [<values>] or "any valid", "why": ""}],
 "untouched": ["everything else"] or [{"table": "", "row": "", "field": "<optional>"}],
 "told": [{"item": "<a yes or no question: did the agent tell the user ...>", "user": "<the user's words>"}],
 "refuse": [{"what": "<the request>", "policy": "<the line>"}],
 "unknown": ["<what cannot be decided, and why>"]}
For a list field give the whole list after the solve, or {"append": [<elements>]} for what the solve adds. \
A value null means the solve removes the field. A value may name a created row as new:1. Quote user \
words and policy lines exactly, short."""

STOP_RULE = """\
Stop when every request the user made is placed (required, choice, refused or unknown) and every cell \
you name rests on what you read. Then answer with the JSON object alone."""

SYSTEM_PROMPT = "\n\n".join((WHAT_YOU_RECEIVE, TOOLS_TEXT, RULES, CHOICE_RULE, OUTPUT, STOP_RULE)) + "\n"


class GoalInputs(NamedTuple):
    """Everything one goal session may see: no Intent, item, Verifier, Run, Reference or End state."""
    user_turns: dict
    policy_text: str
    state: dict
    tools: list
    columns: dict = {}


def user_words(workdir: Path, task_id: str) -> dict[str, list[dict]]:
    """Every recording's user turns by recording id, the user's text alone (no agent line before it)."""
    from kullback.spec.intent_tools import recordings_of

    found = recordings_of(Path(workdir), task_id)
    return {recording: [{"turn": turn.index, "user": turn.text} for turn in turns]
            for recording, turns in sorted(found.recordings.items())}


def load_goal_inputs(workdir: Path, task_id: str) -> GoalInputs:
    """The user's words, the policy, the Starting state and the tool signatures of one Task.

    The policy and the Starting state are read as the writer reads them (spec/writer.py load_inputs):
    the recorded agent's own system prompt, the shared world with the Task's overlay over it. The
    Task's Spec, Verifier, Runs and Reference index are never opened.
    """
    from kullback.runner.state import StateView
    from kullback.runner.world.environment import BuiltEnvironment
    from kullback.spec.writer import _columns

    env = BuiltEnvironment(workdir)
    overlay, rows = env.overlay(task_id)
    state = StateView(env.db, overlay, rows).shared
    writer_tools.refuse_run_inputs(state)
    return GoalInputs(user_turns=user_words(workdir, task_id),
                      policy_text=env.system_prompt(env.task(task_id)) or "",
                      state=state, tools=writer_tools.tool_rows(env.sigs), columns=_columns(env.schema))


def goal_messages(inputs: GoalInputs) -> list[dict]:
    body = {"user_turns": inputs.user_turns or "no recording with a user turn",
            "tools": inputs.tools, "policy_text": inputs.policy_text or "no policy found"}
    return [{"role": "user", "content": SYSTEM_PROMPT + "\n" + json.dumps(body)}]


def world_of(inputs: GoalInputs) -> I.World:
    """The world the read tools close over: no Intent fact is put in it."""
    return I.World(inputs.state, inputs.policy_text, (), inputs.tools, inputs.columns)


def read_tools(inputs: GoalInputs) -> dict:
    tools = writer_tools.world_tools(writer_tools.ItemStore(world_of(inputs), {}))
    return {name: tools[name] for name in READS}


def _squash(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _quoted_in(quote: Any, texts: Iterable[str]) -> bool:
    words = _squash(quote).strip(" .\"'")
    return bool(words) and any(words in _squash(text) for text in texts)


def _cell_problems(cell: dict, world: I.World, said: list[str], policy: str, kind: str) -> list[str]:
    where = f"{kind} {cell.get('table')}.{cell.get('row')}.{cell.get('field')}"
    table, row = str(cell.get("table") or ""), str(cell.get("row") or "")
    out = []
    if table not in world.state and table not in world.declared:
        out.append(f"{where}: no such table")
    elif row and not row.startswith(NEW_ROW) and world.row(table, row) is None:
        out.append(f"{where}: no such row in the Starting state")
    if not cell.get("field"):
        out.append(f"{where}: no field")
    return out + (_ground_problems(cell, where, said, policy) if kind == "required" else [])


def _ground_problems(cell: dict, where: str, said: list[str], policy: str) -> list[str]:
    """What a required cell rests on: some user words or a policy line, each found where it is quoted from."""
    user, line = cell.get("user"), cell.get("policy")
    out = [] if user or line else [f"{where}: rests on no user words or policy line"]
    if user and not _quoted_in(user, said):
        out.append(f"{where}: user words not found in any user turn")
    if line and not _quoted_in(line, [policy]):
        out.append(f"{where}: policy line not found in the policy")
    return out


def last_goal_object(text: str) -> Optional[dict]:
    """The last JSON object in the text that holds a goal key: an answer may restate a broken first try."""
    decoder, found, start = json.JSONDecoder(), None, text.find("{")
    while start != -1:
        try:
            data, length = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(data, dict) and set(data) & set(KEYS + ("cannot_tell",)):
            found = data
        start = text.find("{", start + length)
    return found


def parse_goal(text: str, inputs: GoalInputs) -> dict:
    """The goal object of a final answer, every key present, and the problems code finds in it.

    Code refuses nothing here (report only): it names a cell on a table or row the Starting state
    does not hold, a required cell resting on nothing, and a quote no user turn or policy holds. An
    answer with no goal object is no answer, never a cannot_tell the session did not say.
    """
    data = last_goal_object(text or "")
    if data is None:
        return {"answered": False, "cannot_tell": False, "why": "", "problems": ["no goal object"],
                **{key: [] for key in KEYS}}
    goal = {"answered": True, "cannot_tell": bool(data.get("cannot_tell")), "why": str(data.get("why") or "")}
    for key in KEYS:
        value = data.get(key)
        goal[key] = value if isinstance(value, list) else ([] if value in (None, False) else [value])
    world = world_of(inputs)
    said = [turn["user"] for turns in inputs.user_turns.values() for turn in turns]
    problems = []
    for kind in ("required", "choice"):
        for cell in goal[kind]:
            problems += _cell_problems(cell, world, said, inputs.policy_text, kind) if isinstance(cell, dict) \
                else [f"{kind}: not an object"]
    goal["problems"] = problems
    return goal


def _config(config: Any) -> Any:
    from kullback.ai.provider import ModelConfig

    return config or ModelConfig(thinking={"type": "adaptive", "display": "summarized"}, max_tokens=16000)


def goal_path(workdir: Path, task_id: str, folder: Path = GOAL_DIR) -> Path:
    return Path(workdir) / folder / f"{task_id}.json"


def write_goal(task_id: str, inputs: GoalInputs, model: Any, *, model_id: Optional[str] = None,
               config: Any = None, ceiling_usd: Optional[float] = None) -> dict:
    """One goal session for one Task: the reads, then the goal object, with what it spent."""
    from kullback.spec.writer import _price

    writer_tools.refuse_run_inputs(inputs.state)
    price = _price(model_id)
    session = writer_tools.run_session(model, goal_messages(inputs), inputs.state, _config(config), MAX_GOAL_ROUNDS,
                                       price, force_answer=True, ceiling_usd=ceiling_usd, tools=GOAL_TOOLS,
                                       impl=read_tools(inputs))
    goal = parse_goal(session.get("text") or "", inputs)
    return dict(goal, task_id=task_id, model=model_id, usd=float(session.get("usd") or 0.0),
                tool_rounds=session.get("tool_rounds", 0), tool_uses=session.get("tool_uses", []),
                unfinished=bool(session.get("unfinished")), text=session.get("text") or "",
                thinking=session.get("thinking", []))


def goals(workdir: Any, task_ids: Iterable[str], model: Any, *, folder: Path = GOAL_DIR,
          ceiling_usd: Optional[float] = None, stop_usd: Optional[float] = None) -> dict:
    """One goal per named Task that has none under `folder`, priced under the goal stage of the ledger.

    Stops before a Task once the spend of this call reaches `stop_usd`. Returns written, skipped and
    failed counts and the spend.
    """
    from kullback.runner.budget import BudgetedModel

    workdir = Path(workdir)
    model_id = getattr(model, "name", None)
    priced = BudgetedModel(model, stage=STAGE, workdir=workdir, model_id=model_id)
    counts: dict = {"written": 0, "skipped": 0, "failed": {}, "usd": 0.0, "stopped": False}
    for task_id in task_ids:
        path = goal_path(workdir, task_id, folder)
        if path.is_file():
            counts["skipped"] += 1
            continue
        if stop_usd is not None and counts["usd"] >= stop_usd:
            counts["stopped"] = True
            break
        try:
            goal = write_goal(task_id, load_goal_inputs(workdir, task_id), priced, model_id=model_id,
                              ceiling_usd=ceiling_usd)
        except Exception as error:  # one Task's failure is counted; the others still run
            counts["failed"][task_id] = type(error).__name__
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(goal, indent=1, default=str), encoding="utf-8")
        counts["written"] += 1
        counts["usd"] += goal["usd"]
    return counts


__all__ = ["GOAL_DIR", "GOAL_TOOLS", "GoalInputs", "SYSTEM_PROMPT", "goal_messages", "goals", "load_goal_inputs",
           "parse_goal", "user_words", "write_goal"]
