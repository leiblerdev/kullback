"""The reference judge as an agent with a bounded look, and the one-shot judge as its fallback (D12, D92).

`reference.judge_groups` is one model call over the Intent, the policy and the End states, and it
fails recordings the corpus itself rewarded: on two corpora 22 of the joinable judge-failed
recordings on the first and 18 of 22 on the second, and it abstained not once in 126 judgements.
Its reasons say a state made no writes and so did not carry out the request; the request was often
one the policy does not allow, or one the world makes impossible, and refusing was the right end
state. Neither the rule the refusal rests on nor the row it rests on is in front of the one-shot
judge, and no amount of prompt wording puts them there (D182 measured the last try: showing what
each state told the user moved agreement 79.4 to 79.6 percent).

So the judge looks. It is an agent on the shared core (D121) with six read tools, each answering
the same masked forms the prompt already used: the Intent and its grounded phrases, the policy
sections matching a query, the Starting state rows the group's own calls named, what the group's
answers told the user, the compiled constraints' outcomes per Run, and one Run's final answer
clamped to a few hundred characters. What stays withheld is the conversation (D93): a judgement
that needs the transcript is an abstention, and `answer` is the one line of it the judge may see,
because a state that resolved the Task by answering has its outcome there.

The cap is six tool calls. Over it, or on a judgement that will not parse, the one-shot judge rules
and the record says so, so the agent can only add a look and never take away the answer that was
there before. Every call and the ruling go to the Task's reference record (`judge_calls`), so a
reader can see what the judge looked at before it failed a state.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.context import ContextConfig
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import AgentTool, NoArgs, TextResult
from kullback.examiner import reference as reference_mod
from kullback.gates import verifier_suite
from kullback.runner import budget

MAX_CALLS = 6            # the look is bounded; the seventh call is over the cap and the fallback rules
MAX_TURNS = MAX_CALLS + 4  # a backstop under the cap, so a model that answers nothing still stops
CALL_SUMMARY_CHARS = 100  # how much of each tool result the reference record keeps
POLICY_HITS = 8
POLICY_CHARS = 1600
PHRASE_CHARS = 400
ROWS_SHOWN = 4
ROW_CHARS = 400
ROWS_CHARS = 1600
STATED_CHARS = 1200
CONSTRAINT_RULES = 10
CONSTRAINT_CHARS = 1200
ANSWER_CHARS = 600
RULE_TEXT_CHARS = 160

OVER_THE_CAP = f"the judge asked for more than {MAX_CALLS} tool calls"
UNREADABLE = "the agent's judgement could not be read"

ANSWER_NOW = (f"You have used your {MAX_CALLS} tool calls. Answer now with the JSON judgement and call "
              "nothing else.")


def _clamp(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ..."


def _line(text: Any, limit: int = RULE_TEXT_CHARS) -> str:
    return _clamp(" ".join(str(text or "").split()), limit)


# --- what the judge may look at ---------------------------------------------

@dataclass
class JudgeCase:
    """One Task's judgement as the tools answer it: the Intent, the policy, the groups and their Runs.

    The groups are `reference.group()`'s rows, members included, so every tool reads the same Runs
    the ruling is about and nothing else is loaded for it.
    """
    intent: str = ""
    phrases: tuple[str, ...] = ()
    policy_lines: tuple[str, ...] = ()
    groups: list[dict] = field(default_factory=list)
    write_tools: frozenset = frozenset()
    read_tools: frozenset = frozenset()
    constraints: tuple = ()
    fn: Optional[Callable] = None

    def labels(self) -> list[str]:
        return [str(g["label"]) for g in self.groups]

    def group(self, label: str) -> Optional[dict]:
        wanted = str(label).strip().upper()
        return next((g for g in self.groups if str(g["label"]).upper() == wanted), None)

    def runs(self, label: str) -> list[Any]:
        row = self.group(label)
        return [verifier_suite.as_run(rec.path) for rec in (row or {}).get("members", [])]


def _unknown(label: str, case: JudgeCase) -> str:
    return f"no state is labelled {label!r}; the labels are {', '.join(case.labels()) or '(none)'}"


def _named_values(run: Any) -> list[str]:
    """Every id value the Run's own calls named, in call order, without repeats.

    An id field is the same one `write_effects` keys on, so the rows the judge sees are the rows the
    End state is about and not whatever else the world holds.
    """
    out: list[str] = []
    for call in verifier_suite.run_calls(run):
        for name in sorted(call.get("args") or {}):
            if name not in ("id", "ids") and not name.endswith(("_id", "_ids")):
                continue
            value = (call["args"] or {})[name]
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                if isinstance(item, (str, int, float)) and str(item) not in out:
                    out.append(str(item))
    return out


def _rows_text(case: JudgeCase, label: str) -> str:
    """The Starting state rows the group's calls named, as the world held them before the Run."""
    runs = case.runs(label)
    if not runs:
        return _unknown(label, case)
    wanted: list[str] = []
    for run in runs:
        for value in _named_values(run):
            if value not in wanted:
                wanted.append(value)
    lines: list[str] = []
    missing: list[str] = []
    for value in wanted:
        found = False
        for run in runs:
            for collection, rows in sorted((verifier_suite.start_state(run) or {}).items()):
                if isinstance(rows, dict) and value in rows:
                    lines.append(f"{collection} {value}: "
                                 + _clamp(json.dumps(rows[value], sort_keys=True, default=str), ROW_CHARS))
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(value)
        if len(lines) >= ROWS_SHOWN:
            break
    head = f"state {label}: its Runs named {len(wanted)} value(s) of the world; showing {len(lines)}"
    tail = (f"no row of the Starting state is keyed by: {', '.join(missing[:ROWS_SHOWN])}" if missing else "")
    return _clamp("\n".join([head, *lines] + ([tail] if tail else [])), ROWS_CHARS)


def _stated_text(case: JudgeCase, label: str) -> str:
    """What the group's answers told the user, and how each of its Runs ended."""
    row = case.group(label)
    if row is None:
        return _unknown(label, case)
    facts = list(row.get("told") or [])
    lines = [f"state {label}: {reference_mod.told_line(row)}"]
    if facts:
        lines.append("every value stated: " + ", ".join(str(f) for f in facts))
    for rec in row.get("members", []):
        if rec.transferred:
            ended = "handed the conversation on"
        elif rec.end_state and rec.end_state != reference_mod.ANSWERED:
            ended = "wrote and finished"
        elif rec.stated:
            ended = "answered with values it had read"
        else:
            ended = "finished without writing and without stating a value it had read"
        lines.append(f"run {rec.run_id} ({rec.kind}): {ended}")
    return _clamp("\n".join(lines), STATED_CHARS)


def _constraints_text(case: JudgeCase, label: str) -> str:
    """Per compiled rule with code, what it answered about each Run of the group (D173's `judged`)."""
    runs = case.runs(label)
    if not runs:
        return _unknown(label, case)
    atoms = reference_mod.coded_atoms(case.constraints, case.write_tools, case.read_tools)
    if not atoms:
        return f"state {label}: no rule of the policy compiled to code, so none was checked"
    text_of = {c.id: c.text for c in case.constraints}
    lines: list[str] = []
    silent = 0
    for atom in atoms:
        rule_id = atom.target["constraint_id"]
        judged = failed = 0
        for run in runs:
            held = verifier_suite.hard_holds(atom, run, set(case.write_tools), case.fn)
            if held is None:
                continue
            judged += 1
            failed += int(held is False)
        if not judged:
            silent += 1
            continue
        lines.append(f"{rule_id}: {_line(text_of.get(rule_id))} | judged {judged} of {len(runs)} run(s), "
                     f"failed {failed}")
        if len(lines) >= CONSTRAINT_RULES:
            break
    head = (f"state {label}: {len(atoms)} rule(s) of the policy have code; {silent} of them judged no call "
            "of these Runs, so the code says nothing about them")
    return _clamp("\n".join([head, *lines]), CONSTRAINT_CHARS)


def _answer_text(case: JudgeCase, label: str) -> str:
    """The final assistant turn of one Run of the group, text only, clamped. The rest stays withheld."""
    runs = case.runs(label)
    if not runs:
        return _unknown(label, case)
    said = ""
    for event in runs[0].events:
        text = verifier_suite._assistant_text(event)
        if text.strip():
            said = text
    if not said.strip():
        return f"state {label}: its first Run ended without an answer in words"
    return f"state {label}, the last thing its first Run said: " + _clamp(" ".join(said.split()), ANSWER_CHARS)


def _intent_text(case: JudgeCase) -> str:
    phrases = ", ".join(case.phrases)
    return "\n".join([
        "Intent, the ground truth of what the user wanted by the end of the Run: "
        + (_line(case.intent, reference_mod.MAX_REQUEST_CHARS) or "(not recorded)"),
        "The phrases of it that are grounded in the Runs: " + (_clamp(phrases, PHRASE_CHARS) or "(none)"),
    ])


def _policy_text(case: JudgeCase, query: str) -> str:
    """The policy sections matching a query, the Examiner's search shape (D178), clamped."""
    needle = " ".join(str(query or "").split()).lower()
    if not needle:
        return "policy needs a phrase to look for"
    hits = [text for text in case.policy_lines if needle in text.lower()]
    head = (f"policy search {needle!r}: {len(hits)} of {len(case.policy_lines)} sections match, "
            f"showing {min(len(hits), POLICY_HITS)}")
    lines = [f"- {_line(text, reference_mod.MAX_LINE_CHARS)}" for text in hits[:POLICY_HITS]]
    if not hits:
        lines = ["no section holds that phrase; search a shorter one, or a word the request uses"]
    return _clamp("\n".join([head, *lines]), POLICY_CHARS)


# --- the tools ---------------------------------------------------------------

class GroupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str = Field(description="The label of one End state, as the states are lettered above.")


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="A phrase to look for in the policy, case insensitive. Short is better: "
                                   "one noun of the request, not a sentence.")


def judge_tools(case: JudgeCase) -> list[AgentTool]:
    """The six read tools of one judgement. Every one of them answers about this Task and no other."""

    async def intent(_: NoArgs) -> TextResult:
        return TextResult(text=_intent_text(case))

    async def policy(args: QueryArgs) -> TextResult:
        return TextResult(text=_policy_text(case, args.query))

    async def rows(args: GroupArgs) -> TextResult:
        return TextResult(text=_rows_text(case, args.group))

    async def stated(args: GroupArgs) -> TextResult:
        return TextResult(text=_stated_text(case, args.group))

    async def constraints(args: GroupArgs) -> TextResult:
        return TextResult(text=_constraints_text(case, args.group))

    async def answer(args: GroupArgs) -> TextResult:
        return TextResult(text=_answer_text(case, args.group))

    def render(result: TextResult) -> str:
        return result.text

    return [
        AgentTool("intent", "The Intent line and the phrases of it that are grounded in the Runs.",
                  NoArgs, TextResult, intent, render=render),
        AgentTool("policy", "The policy sections holding a phrase. Use it before failing a state that "
                            "refused or wrote nothing: the rule it refused under is here or nowhere.",
                  QueryArgs, TextResult, policy, render=render),
        AgentTool("rows", "The Starting state rows one state's Runs named, as the world held them. Use it "
                          "when whether the change was possible or allowed turns on what the row holds.",
                  GroupArgs, TextResult, rows, render=render),
        AgentTool("stated", "What one state's answers told the user, every value, and how each of its Runs "
                            "ended.", GroupArgs, TextResult, stated, render=render),
        AgentTool("constraints", "Per rule of the policy that has code, what it answered about each Run of "
                                 "one state. A rule that judged no call says nothing either way.",
                  GroupArgs, TextResult, constraints, render=render),
        AgentTool("answer", "The last thing one state's first Run said, in words, clamped. The rest of the "
                            "conversation stays withheld.", GroupArgs, TextResult, answer, render=render),
    ]


# --- the prompt (GEPA order) --------------------------------------------------

WHAT_YOU_RECEIVE = (
    "Recordings of one Task ended in different states, lettered A, B, C. Say which of the states did NOT do "
    "what the Intent asked, or did something the policy does not allow.\n"
    "You receive the Intent and one line per state saying what its Runs wrote and what their answers told the "
    "user. You do not receive the policy, the rows of the world or the conversation: the first two are behind "
    "tools, and the conversation stays withheld. Whether the agent authenticated the user, asked for and was "
    "given a confirmation, or called its tools in the right order cannot be seen from here, and no state is "
    "ever failed for want of evidence you were not given."
)

TOOLS_TEXT = (
    "Your tools, all read only, one example call each:\n"
    '- intent {} : the Intent line and its grounded phrases.\n'
    '- policy {"query": "refund"} : the policy sections holding that phrase.\n'
    '- rows {"group": "A"} : the Starting state rows that state\'s Runs named.\n'
    '- stated {"group": "A"} : every value that state told the user, and how each of its Runs ended.\n'
    '- constraints {"group": "A"} : what each coded rule of the policy answered about that state\'s Runs.\n'
    '- answer {"group": "A"} : the last thing that state\'s first Run said, clamped.'
)

EXAMPLES = (
    "Two worked examples, from other Tasks than this one.\n"
    "\n"
    "One. The Intent is to move a booking to the following week. State A wrote nothing and told the user two "
    "values it had read; state B moved the booking. Failing A on the writes alone would be wrong, so the judge "
    "calls policy with the noun of the request and finds a section saying a booking inside its notice window "
    "may not be moved, then calls rows on A and sees the booking's date inside that window. A refused what the "
    "policy forbids and B did what it forbids, so the ruling fails B, on policy and rows.\n"
    "\n"
    "Two. The Intent is to add a second holder to an account. State A added one; state B wrote nothing and told "
    "the user nothing it had read. The judge calls rows on B and the account is there and open, calls policy "
    "for the noun and finds no rule against a second holder, and calls stated on B, which handed nothing on and "
    "stated no value. Nothing excuses B, so the ruling fails B. Had B handed the conversation on, or had the "
    "row been closed, the ruling would have been that it cannot tell, and nothing would be failed."
)

CHOICE_RULE = (
    "The rule for choosing.\n"
    "- Fail a state only when you are sure it did not do what the Intent asked and the policy allows. A pass is "
    "not yours to award: what you do not fail stays in.\n"
    "- A state that wrote nothing is not failed on that alone. Before failing one, look: the policy may forbid "
    "the change, the row may make it impossible, and refusing is then the right End state.\n"
    "- A state that told the user no value read from the world may still have answered in words that carry "
    "none, so that on its own is not a state failed either.\n"
    "- When your reason would need the conversation, the opening request, an authentication or a spoken "
    "confirmation, fail nothing and name what you needed instead.\n"
    "- Judge the Intent, never the opening request: the user may have revised it during the Run."
)

ANSWER_SHAPE = (
    'Answer with JSON only, on its own turn, no tool call beside it: '
    '{"failed": ["A"], "evidence": ["intent", "policy", "end_states"], "reason": "..."}. '
    'Every name in evidence must be one of ' + ", ".join(reference_mod.AVAILABLE_SOURCES)
    + '; when your answer needs anything else, name that instead and fail nothing. When you cannot tell, '
    'answer {"failed": [], "reason": "cannot tell"}.'
)

STOP_RULE = (
    f"Stop rule. You have at most {MAX_CALLS} tool calls for the whole judgement. Spend them on the states you "
    "are considering failing, then answer with the JSON. Answering earlier is right whenever you already know; "
    f"asking for a {MAX_CALLS + 1}th call ends the session and your judgement is lost."
)


def judge_system_prompt() -> str:
    return "\n\n".join([WHAT_YOU_RECEIVE, TOOLS_TEXT, EXAMPLES, CHOICE_RULE, ANSWER_SHAPE, STOP_RULE])


def judge_message(case: JudgeCase) -> str:
    """The one message the judge answers: the Intent and one line per state, the rest behind the tools."""
    lines = ["Intent: " + (_line(case.intent, reference_mod.MAX_REQUEST_CHARS) or "(not recorded)"), "",
             "End states:"]
    for row in case.groups:
        runs = len(row.get("runs") or [])
        lines.append(f"{row['label']} ({runs} run{'s' if runs != 1 else ''}): {row['state']}")
        lines.append("    " + reference_mod.told_line(row))
    lines += ["", "Look at what you need, then rule."]
    return "\n".join(lines)


# --- the agent ----------------------------------------------------------------

class AgentJudge:
    """The reference judge with a bounded look. `rule_groups` is what `reference.confirm` calls.

    Held by the derivation for the whole build and asked once per Task with more than one End state,
    so it carries what every judgement shares (the model, the constraints, the tool kinds, the canon)
    and takes what one judgement is about (the Intent, the policy, the groups) as arguments.
    """

    def __init__(self, model: Any, *, constraints: Iterable[Any] = (), write_tools: Iterable[str] = (),
                 read_tools: Iterable[str] = (), fn: Optional[Callable] = None,
                 max_calls: int = MAX_CALLS, max_turns: int = MAX_TURNS):
        self.model = model
        self.name = getattr(model, "name", "model")
        self.constraints = tuple(constraints)
        self.write_tools = frozenset(write_tools)
        self.read_tools = frozenset(read_tools)
        self.fn = fn or verifier_suite.canon_fn(None)
        self.max_calls = max_calls
        self.max_turns = max_turns

    def case(self, intent: str, policy_lines: Iterable[str], groups: list[dict],
             phrases: Iterable[str] = ()) -> JudgeCase:
        return JudgeCase(intent=intent or "", phrases=tuple(phrases), policy_lines=tuple(policy_lines),
                         groups=list(groups), write_tools=self.write_tools, read_tools=self.read_tools,
                         constraints=self.constraints, fn=self.fn)

    def rule_groups(self, intent: str, policy_lines: Iterable[str], groups: list[dict],
                    phrases: Iterable[str] = ()) -> reference_mod.Judgement:
        """One judgement: the agent's, or the one-shot judge's when the agent went over the cap or
        answered something that will not parse. The calls it made ride on the Judgement either way."""
        case = self.case(intent, policy_lines, groups, phrases)
        text, calls, over_cap = self._session(case)
        if not over_cap:
            judgement = reference_mod.parse_judgement(text, {str(g["label"]) for g in groups})
            if judgement.reason != reference_mod.UNREADABLE_REPLY:
                judgement.calls = calls
                return judgement
        fallback = reference_mod.judge_groups(self.model, intent, policy_lines, groups)
        fallback.calls = calls
        fallback.fell_back = OVER_THE_CAP if over_cap else UNREADABLE
        return fallback

    def _session(self, case: JudgeCase) -> tuple[str, list[dict], bool]:
        """The agent's last spoken turn, the calls it made, and whether it asked past the cap."""
        harness = AgentHarness(model=self.model, system=judge_system_prompt(), tools=judge_tools(case),
                               max_turns=self.max_turns,
                               context=ContextConfig(window=budget.window_for(self.name)))
        calls: list[dict] = []
        over_cap = False

        def watch(event: Any) -> None:
            nonlocal over_cap
            if event.type != "tool_execution_end":
                return
            calls.append({"tool": event.tool_name,
                          "summary": _clamp(" ".join((event.result.content or "").split()),
                                            CALL_SUMMARY_CHARS)})
            if len(calls) > self.max_calls:
                over_cap = True
                harness.cancel()
            elif len(calls) == self.max_calls:
                harness.steer(ANSWER_NOW)

        harness.subscribe(watch)
        said = ""

        async def go() -> None:
            nonlocal said
            async for event in harness.prompt(judge_message(case)):
                if event.type != "message_end":
                    continue
                message = event.message
                if getattr(message, "role", "") == "assistant" and message.stop_reason != "error" \
                        and (message.content or "").strip():
                    said = message.content or ""

        try:
            asyncio.run(go())
        except Exception:  # noqa: BLE001 - a judge that could not run rules nothing; the fallback does
            return "", calls, False
        return said, calls, over_cap


__all__ = ["MAX_CALLS", "MAX_TURNS", "OVER_THE_CAP", "UNREADABLE", "AgentJudge", "JudgeCase", "judge_message",
           "judge_system_prompt", "judge_tools"]
