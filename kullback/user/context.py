"""The context curated for the agent user, per Task, all of it mined (D214).

The Builder and the Examiner are given a curated context and so is this one: an ungrounded model
asked to be someone's customer invents a booking it never had and agrees to whatever the Candidate
proposes, and what keeps it honest is that everything it may say was mined from the recording. So
the sections here are the typed facts with their classes (D210), the goal in the recorded user's own
words, a persona read off the recorded user's turns as counted classes and never as quotes, the
values this user chooses when asked to choose (D151), the end protocol, the conversation so far, and
whatever the last round's fidelity taught about this Task (lesson.py).

Every item is a PromptSection with a stable tag, so the ContextManager can cut one and recall it by
name the way it does for the other two agents, and so a section that is empty on a Task is absent
rather than present and blank.

Nothing here names a corpus, a tool or a column: a persona is counts and bands, a fact is whatever
the Task's own rules hold, and the only English in the file is the frame the values are read in.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from kullback.agent.harness import PromptSection
from kullback.runner.records import Record, Trace, UserFact, UserRules
from kullback.user import rules as rules_mod
from kullback.user.vocabulary import GENERIC, Vocabulary

# The tags every section is filed under. They are stable across rounds and across Tasks, because a
# lesson that says "your facts section was wrong" is worth nothing if the section is named after the
# round it was written in.
FACTS_TAG = "user_facts"
GOAL_TAG = "user_goal"
PERSONA_TAG = "user_persona"
CHOICES_TAG = "user_choices"
PROTOCOL_TAG = "user_protocol"
PREFIX_TAG = "user_prefix"
LESSONS_TAG = "user_lessons"
SECTION_TAGS = (GOAL_TAG, FACTS_TAG, PERSONA_TAG, CHOICES_TAG, PROTOCOL_TAG, PREFIX_TAG, LESSONS_TAG)

# How a persona's two free numbers are banded. Bands and not the numbers themselves, because a
# turn length of nine words and one of eleven are the same person and a prompt that says "eleven"
# invites a model to count.
SHORT_TURN = 12          # words; at or under this the recorded user answers in fragments
LONG_TURN = 40           # words; at or over this it writes paragraphs
COURTEOUS_SHARE = 0.5    # this share of turns carrying a politeness cue is a courteous register

POLITE_CUE = re.compile(r"\b(please|thanks|thank you|could you|would you|sorry|appreciate)\b", re.I)
# How the recorded user answered a Candidate that would not do what was asked. Three classes, read
# in this order; a recording where the Candidate never refused is its own class, so a persona never
# claims to know something the recording could not show.
ACCEPTS, INSISTS, LEAVES, NEVER_REFUSED = "accepts", "insists", "leaves", "never_refused"
INSIST_CUE = re.compile(r"\b(but|still|please (?:check|try|look)|are you sure|can you (?:check|try)|"
                        r"i need|i really|why not|that is not|that's not)\b", re.I)
CLOSING_LINE, NO_CLOSE = "closing_line", "no_close"
COURTEOUS, PLAIN = "courteous", "plain"
FRAGMENTS, SENTENCES, PARAGRAPHS = "fragments", "sentences", "paragraphs"


class Persona(Record):
    """How the recorded user of one Task behaves, as counted classes (D214 rule 1).

    Not one quoted sentence: a persona that carries the recording's wording is a way for the model
    to say a value it was never given, and a persona that carries counts is not.
    """

    turns: int = 0
    length: str = SENTENCES          # fragments, sentences or paragraphs
    tone: str = PLAIN                # courteous or plain; named tone because "register" is taken
    volunteered: int = 0             # facts stated with no ask in the turn before them
    restated: int = 0                # fields stated in more than one turn, so it repeats when asked twice
    on_refusal: str = NEVER_REFUSED  # accepts, insists, leaves, or the recording never showed one
    ends: str = NO_CLOSE             # closing_line or no_close


class TaskContext(Record):
    """Everything the agent user of one Task may know, mined, with nothing else in it."""

    task_id: str = ""
    facts: list[UserFact] = []            # the askable facts (D210): what this user itself said
    record_fields: list[str] = []         # fields only the world holds, which it points at
    goal: str = ""                        # the recorded user's own opening sentence
    persona: Persona = Persona()
    choices: dict[str, str] = {}          # the Reference's write arguments (D151)
    consultations: list[str] = []         # fields a Trace shows the user consulting something for
    lessons: list[str] = []               # what the last rounds' fidelity taught (lesson.py)

    def askable(self) -> list[UserFact]:
        """The facts an agent user may speak: the spoken sentences are the persona's, not values."""
        return [f for f in self.facts if f.field not in rules_mod.SPOKEN_FIELDS]

    def fact_values(self) -> list[str]:
        return [str(f.value) for f in self.askable() if f.value is not None]

    def grounding(self) -> list[str]:
        """Every sentence this user is known to have said: the facts' own contexts and the spoken
        ones. A value-shaped token in one of these is the recorded user's word and not an invention,
        which is what the guard compares against (guards.py)."""
        out = [str(f.value) for f in self.facts if f.field in rules_mod.SPOKEN_FIELDS and f.value]
        out += [f.context for f in self.facts if f.context]
        return [text for text in out if text]


def _words(text: Optional[str]) -> int:
    return len(re.findall(r"\S+", text or ""))


def _length_class(turns: Sequence[Any]) -> str:
    if not turns:
        return SENTENCES
    average = sum(_words(t.content) for t in turns) / len(turns)
    if average <= SHORT_TURN:
        return FRAGMENTS
    return PARAGRAPHS if average >= LONG_TURN else SENTENCES


def _refusal_class(trace: Trace) -> str:
    """How this user answered the first Candidate turn that refused it, in one of three classes."""
    turns = list(trace.turns)
    for index, turn in enumerate(turns):
        if turn.role != "assistant" or not rules_mod.AGENT_REFUSAL.search(turn.content or ""):
            continue
        answer = next((t for t in turns[index + 1:] if t.role == "user"), None)
        if answer is None:
            return LEAVES
        said = rules_mod._norm(answer.content)
        if any(cue in said for cue in rules_mod.WALK_AWAY_CUES) or rules_mod.CLOSING_CUE.search(said):
            return LEAVES
        return INSISTS if INSIST_CUE.search(answer.content or "") else ACCEPTS
    return NEVER_REFUSED


def mine_persona(trace: Optional[Trace], user_rules: Optional[UserRules] = None) -> Persona:
    """The persona of one recording, as counts and classes (D214 rule 1).

    Read off the user turns only: how many there were, how long they run, whether they are courteous,
    how often a fact was given before anyone asked, how often a field was said more than once, what
    the user did when refused, and whether it closed the conversation. A recording with no user turns
    gives the default persona, which claims nothing.
    """
    if trace is None:
        return Persona()
    turns = [t for t in trace.turns if t.role == "user"]
    if not turns:
        return Persona()
    polite = sum(1 for t in turns if POLITE_CUE.search(t.content or ""))
    volunteered = sum(1 for rule in (user_rules.disclosure if user_rules else ()) if not rule.on_request)
    fields = [f.field for f in (user_rules.facts if user_rules else ())
              if f.field not in rules_mod.SPOKEN_FIELDS]
    restated = sum(1 for field in dict.fromkeys(fields) if fields.count(field) > 1)
    closed = bool(rules_mod.CLOSING_CUE.search(rules_mod._norm(turns[-1].content)))
    return Persona(
        turns=len(turns),
        length=_length_class(turns),
        tone=COURTEOUS if polite >= COURTEOUS_SHARE * len(turns) else PLAIN,
        volunteered=volunteered,
        restated=restated,
        on_refusal=_refusal_class(trace),
        ends=CLOSING_LINE if closed else NO_CLOSE,
    )


def volunteer_allowance(trace: Optional[Trace], user_rules: Optional[UserRules], turn_number: int) -> int:
    """How many facts the recorded user had given by its `turn_number`-th turn (D214 rule 4).

    The disclosure limit is a count and not a list, so a Candidate cannot farm facts by asking many
    questions: what it may have by turn n is what the recorded user had said by turn n, whatever it
    asked to get there. A Task with no recording has no limit, which is the user it had before.
    """
    if trace is None or user_rules is None:
        return -1
    turns = [t for t in trace.turns if t.role == "user"][:max(0, turn_number)]
    if not turns:
        return 0
    last = turns[-1].idx
    # By value and not by fact row: one value the recorded build filed under two field names (the
    # write's argument name beside the vocabulary's, D151) is still one thing this user has said.
    values = {str(f.value).strip().lower() for f in user_rules.facts
              if f.field not in rules_mod.SPOKEN_FIELDS and str(f.value).strip()}
    said = 0
    for value in values:
        turn = next((t for t in trace.turns
                     if t.role == "user" and t.idx <= last and rules_mod._said_in(t.content, value)), None)
        said += int(turn is not None)
    return said


def mine_consultations(trace: Optional[Trace], user_rules: Optional[UserRules] = None,
                       vocab: Vocabulary = GENERIC) -> list[str]:
    """The fields a Trace shows this user consulting something for, between its own turns (D214 rule 2).

    The class, read off the recording and nothing else. A recorded tool call's result carries values
    the world produced and the call's own arguments did not: nobody in the conversation put them
    there. Those values split in two. The ones no user turn ever says are the record class, which is
    what the user cannot give (`mine_record_values`). The ones a user turn does say, before any turn
    in the conversation said them, are this class: the user produced a value only the world's records
    hold and no one had told them, so they went and looked while the conversation was open, and a
    user who does that has somewhere to look.

    Where the recording shows none the tool is absent, which is the point: a tool a recording never
    justified is a tool the model uses to invent.
    """
    from kullback.user.guards import value_tokens
    if trace is None:
        return []
    passed_in = {str(value).strip() for call in trace.tool_calls for value in _leaves(call.args)}
    world: list[tuple[str, str]] = []
    for call in trace.tool_calls:
        for key, value in _keyed_leaves(call.result):
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue
            text = str(value).strip()
            if len(text) < MIN_RECORD_VALUE or text in passed_in or not value_tokens(text):
                continue
            world.append((key, text))
    out: list[str] = []
    said: list[str] = []
    for turn in trace.turns:
        text = turn.content or ""
        if turn.role == "user":
            for field, value in world:
                if field in out or not rules_mod._said_in(text, value):
                    continue
                if any(rules_mod._said_in(earlier, value) for earlier in said):
                    continue
                out.append(field)
        said.append(text)
    return out


def _leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for inner in value.values():
            yield from _leaves(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _leaves(inner)
    elif value is not None:
        yield value


def reference_choices(trace: Optional[Trace], write_tools: Iterable[str] = ()) -> dict[str, str]:
    """The values this user chooses when asked to choose: the Reference's write arguments (D151).

    A Candidate that offers three ways to do the thing gets the one the recording took, because the
    recording is what the Task is scored against; a user that picks another is a Run that fails a
    Verifier for the user's taste. Scalars only, and only from calls the world did not refuse.
    """
    names = set(write_tools)
    out: dict[str, str] = {}
    for call in (trace.tool_calls if trace is not None else ()):
        if call.error is not None or (names and call.name not in names):
            continue
        for arg, value in rules_mod._spoken_args(call):
            out.setdefault(arg, str(value))
    return out


def curate(task_id: str, user_rules: Optional[UserRules], trace: Optional[Trace] = None, *,
           vocab: Vocabulary = GENERIC, write_tools: Iterable[str] = (),
           record_fields: Iterable[str] = (), lessons: Iterable[str] = ()) -> TaskContext:
    """Everything mined for one Task, in one record (D214 rule 1)."""
    user_rules = user_rules or UserRules()
    goal = next((str(f.value) for f in user_rules.facts if f.field == rules_mod.GOAL), "")
    return TaskContext(
        task_id=task_id,
        facts=list(user_rules.facts),
        record_fields=[field for field in dict.fromkeys(record_fields)],
        goal=goal,
        persona=mine_persona(trace, user_rules),
        choices=reference_choices(trace, write_tools),
        consultations=mine_consultations(trace, user_rules, vocab),
        lessons=[str(text) for text in lessons if str(text).strip()],
    )


# --- the sections ---------------------------------------------------------------------------

def _facts_text(ctx: TaskContext, asked: Iterable[str] = ()) -> str:
    wanted = set(asked)
    rows = [f for f in ctx.askable() if not wanted or f.field in wanted]
    if not rows:
        return ""
    lines = [f"- {f.field}: {f.value}" for f in rows]
    if ctx.record_fields:
        lines.append("Fields you do not hold, which the other side can look up: "
                     + ", ".join(ctx.record_fields) + ".")
    return "What you know about yourself, and nothing else:\n" + "\n".join(lines)


def _persona_text(persona: Persona) -> str:
    return (f"How you talk, counted from how you talked before: {persona.turns} turn(s), "
            f"in {persona.length}, {persona.tone} in register. You gave "
            f"{persona.volunteered} fact(s) before anyone asked and repeated {persona.restated} "
            f"field(s) when asked a second time. Refused something, you {persona.on_refusal}. "
            f"You end with a {persona.ends.replace('_', ' ')}.")


def _protocol_text(kinds: Sequence[str] = rules_mod.USER_END_KINDS) -> str:
    return ("Ending. You never end the conversation yourself: call end_run with the kind you think "
            "holds and code decides whether it does. The kinds are " + ", ".join(kinds) + ".")


def _prefix_text(prefix: Sequence[Any]) -> str:
    lines = []
    for message in prefix or ():
        role = rules_mod._field_of(message, "role")
        content = (rules_mod._field_of(message, "content") or "").strip()
        if role in ("user", "assistant") and content:
            lines.append(f"{'you' if role == 'user' else 'them'}: {content}")
    return "The conversation so far.\n" + "\n".join(lines) if lines else ""


def sections(ctx: TaskContext, prefix: Sequence[Any] = (), asked: Iterable[str] = ()) -> list[PromptSection]:
    """The curated context as prompt sections, each under its stable tag, empty ones dropped."""
    built = [
        (GOAL_TAG, f"What you came for, in the words you used: {ctx.goal}" if ctx.goal else ""),
        (FACTS_TAG, _facts_text(ctx, asked)),
        (PERSONA_TAG, _persona_text(ctx.persona)),
        (CHOICES_TAG, ("When you are asked to choose, you choose these: "
                       + ", ".join(f"{k}={v}" for k, v in sorted(ctx.choices.items())) + ".")
         if ctx.choices else ""),
        (PROTOCOL_TAG, _protocol_text()),
        (PREFIX_TAG, _prefix_text(prefix)),
        (LESSONS_TAG, ("What the last rounds got wrong on this conversation:\n"
                       + "\n".join(f"- {line}" for line in ctx.lessons)) if ctx.lessons else ""),
    ]
    return [PromptSection(tag, text) for tag, text in built if text]


def content_key(ctx: TaskContext) -> str:
    """What the stall rule watches (D214 rule 6): the facts and the persona of this Task.

    A Task that stopped paying for the agent user starts paying again when its facts or its persona
    change, and neither a new lesson nor a new round is such a change; so the key is over those two
    and over nothing else.
    """
    from kullback.runner.records import content_hash
    return content_hash([[f.field, str(f.value)] for f in ctx.facts] + [ctx.persona.model_dump(mode="json")])


# --- the record facts, mined offline from the recording itself --------------------------------
# How many record values one Task keeps. The replacement guard reads every one of them on every
# sentence, and a Task whose recording returned a thousand rows would otherwise cost a turn a
# thousand comparisons for no gain: the ones that matter are the values an agent asked about.
RECORD_VALUE_CAP = 40
MIN_RECORD_VALUE = 3   # characters; shorter than this and a "value" is a flag or a code word


def _keyed_leaves(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for inner_key, inner in value.items():
            yield from _keyed_leaves(inner, str(inner_key))
    elif isinstance(value, list):
        for inner in value:
            yield from _keyed_leaves(inner, key)
    elif key and value is not None:
        yield key, value


def mine_record_values(trace: Optional[Trace], cap: int = RECORD_VALUE_CAP) -> dict[str, Any]:
    """The values only the world holds, read off the recording (D210's record class).

    A value a recorded tool call returned that no user turn in the same recording ever said is a
    value that user did not have: that is the class, and it is mined rather than declared, so a
    corpus whose users happen to know their own account numbers has fewer record facts and not a
    wrong list. Value-shaped only, because an ordinary word returned by a tool ("pending") is not
    something a user leaks by saying it.
    """
    from kullback.user.guards import value_tokens
    if trace is None:
        return {}
    said = [turn.content for turn in trace.turns if turn.role == "user"]
    out: dict[str, Any] = {}
    for call in trace.tool_calls:
        for key, value in _keyed_leaves(call.result):
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue
            text = str(value).strip()
            if len(text) < MIN_RECORD_VALUE or key in out or not value_tokens(text):
                continue
            if any(rules_mod._said_in(turn, text) for turn in said):
                continue
            out[key] = text
            if len(out) >= cap:
                return out
    return out
