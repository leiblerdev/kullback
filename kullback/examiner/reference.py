"""References by the D111 rule, and the constraints checked against them before they gate anything.

Which recorded Runs a Verifier may be derived from is settled before the Verifier exists, because
the Verifier is derived from those Runs and cannot judge them (D43). The rule is what the Intent
plus the policy say the End state should hold: a recording that broke a compiled Hard constraint is
a failed recording; the rest are grouped by End state, and the References are the one group that
agrees. When more than one group is left, code cannot tell which of them carried out the request, so
a judge may mark groups as failed, never as passed (D110); a Task whose groups still disagree gets
no Reference and no Verdict. That judge is an agent with a bounded look
(judge.py): it may ask for the Intent's phrases, the policy sections matching a query, the Starting
state rows its Runs named, what each state told the user, what the coded rules answered about each
Run, and one Run's last answer, six calls in all. The conversation is still withheld, so a ruling of
its that rests on anything else, authentication or a spoken confirmation or the opening request, is
an abstention and fails nothing (D93); over the cap the one-shot judge rules and the record says so.
Naming its sources is not enough on its own: the judge is asked only where the states disagree, so a
failure of a state must cite a key the states differ on, the value that state holds at it and what the
value should have been, and code checks all three against the material the prompt showed (D186). A
failure that cites nothing they differ on is uncited and the whole judgement abstains.
The End state it is handed has two halves, what the Runs wrote and what their
answers told the user (D43): with the second half missing a Task the recorded agent resolved by
answering read as a Task nobody acted on, and the judge failed the recordings the corpus itself had
rewarded. Re-rolls (D112) enter the same rule
as recordings of a lower standing: the Reference is a recording whenever the agreeing group holds
one, since the recording is the only Run that touched the customer's real system.

A fail-only judge with three or more states in front of it may drop some and still leave two, and
that residue was the largest single reason a fidelity-passing Task kept no Reference on all three
live builds. So a judgement that leaves two or more states in is asked once more, with only the
survivors and the keys those survivors differ on, under a stop rule saying exactly one may remain:
either it fails every other survivor with a cited failure, or it says the survivors cannot be told
apart on what they differ on, which is an abstention with that reason (D193). A second pass that
still leaves two or more, or that cites nothing, abstains the same way, and the Task falls where an
abstention already fell. When instead every state is failed, the failures are read against each
other: failures agreeing on one key and on one required value from the Intent or a numbered policy
line are the judge saying no recording did what was required, which is `no_correct_recording`, a
reason of its own for a later beat to file as an Environment or an Intent finding; failures
disagreeing among themselves are an abstention, "failed all on differing grounds".

A residue the judge will not settle is settled by code instead (D198). Asking a fail-only judge twice
is asking it the question it has already declined, so the states it left in are handed to the
derivation: one Verifier per surviving End state, each derived from that state as if it were the
Reference, each put through the D79 suite and the held-out false-rejection check, and the survivor
whose Verifier stands up best is the Reference. This module's part of that is only to say which
states survived, in the order the recordings first reached them; the deriving and the choosing are
the derive stage's, since the suite lives above this file.

A compiled constraint that fails on a large share of the confirmed recordings corpus-wide is demoted
first. The recordings are the frontier under the customer's own policy, and a rule they break that
often is a miscompiled rule, not a corpus of violations (D76); on the first retail build four such
rules failed the oracle check on most Tasks, two of them by naming tools the corpus never shows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from kullback.examiner import derive as verifier_mod
from kullback.gates import verifier_suite
from kullback.gates.verifier_suite import _key
from kullback.runner.judge import sources_not_given
from kullback.runner.records import Atom, Constraint
from kullback.runner.verdict import TRANSFER_HINTS

RECORDING = "recording"
REROLL = "reroll"
# The End state of a Run that wrote nothing and whose answer stated facts it had read from the world.
# It is shaped like a write effect so `group()` and `describe()` need no second kind of state, and it
# carries no fact of its own: which facts were stated is derive's business, this only says some were.
ANSWERED = (("answer", "", (("stated", "facts read from the world"),)),)
# A rule the confirmed recordings break at this share is miscompiled, not violated. Calibrated once on
# the second retail build against tau2's reward (D112 scaffolding, D114): the recordings failed by
# rules firing at 2.7% and up carried reward 1 at 82 to 93%, above the corpus rate of 72%, so those
# rules were eating the good recordings; the two rules at 1.4% carried 60%, consistent with real
# violations. Re-check on airline and telecom before the scaffold is deleted.
MISCOMPILED_SHARE = 0.02
MIN_RUNS_TO_DEMOTE = 3
MAX_POLICY_LINES = 40  # D65: the judge prompt is bounded whatever the policy's length
MAX_LINE_CHARS = 200
MAX_REQUEST_CHARS = 600
MAX_FACTS = 12  # how many of a group's stated facts the prompt lists before it says how many are left
MAX_DIFFERING_KEYS = 12  # how many differing keys a group's block names before it says how many are left
# What this judge is handed, and the whole of what a ruling of its may rest on. It sees the Intent,
# the policy and the End states, and never a transcript, so authentication, a spoken confirmation and
# the opening request are absent by construction; a ruling that rests on one of them abstains (D93).
AVAILABLE_SOURCES = ("intent", "policy", "end_states")
# D186's vocabulary. A key the judge may cite is a column of the settled End state (`tool.entity.column`,
# with `WRITTEN` for the write itself) or a value an answer stated (`STATED_PREFIX` and the value); a
# group that holds no value at a key holds `NO_VALUE`, which is a value the states can differ on too.
WRITTEN = "written"
STATED_PREFIX = "stated:"
NO_VALUE = "(no value in this state)"
NO_KEY = "no key"
# What `expected` may name beside another state's label: the Intent, or one numbered policy line.
INTENT_EXPECTED = "intent"
POLICY_EXPECTED = "policy:"
# What `parse_judgement` says about a reply it could not read. Named, because the agent judge (judge.py)
# falls back to the one-shot judge on exactly this answer and must not compare against a sentence.
UNREADABLE_REPLY = "unreadable reply"
# D193's three readings of what a judgement left behind. The first two are abstention reasons and the
# third is a reason of its own, distinct from the residue, so a reader can tell "the states were never
# told apart" from "the judge says none of them was right".
SURVIVORS_NOT_TOLD_APART = "survivors not told apart"
DIFFERING_GROUNDS = "failed all on differing grounds"
NO_CORRECT_RECORDING = "no correct recording"
# D198's three answers, once the derivation has put a Verifier of every survivor through the suite.
# They are named here because the reference record carries them and this module owns that record.
SURVIVOR_CHOSEN = "survivor_chosen"
SURVIVORS_EQUIVALENT = "survivors_equivalent"
SURVIVORS_ALL_FAIL = "survivors_all_fail"
# D218 rule 3: a residue that ends with nothing to derive from still says why. Every
# surviving End state held no Run the derivation could read, so there was no survivor to
# rule on and the Task keeps no Reference for that reason and not for an unrecorded one.
SURVIVORS_EMPTY = "survivors_empty"
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Recording:
    """One Run the D111 rule sees: where it is, what it wrote, what it told the user, and which Hard
    constraints it broke."""
    run_id: str
    path: str
    kind: str = RECORDING
    trace_id: Optional[str] = None
    end_state: tuple = ()
    violated: list[str] = field(default_factory=list)
    stated: tuple = ()          # the facts read from the world this Run's answers stated back
    transferred: bool = False   # it handed the conversation on rather than finishing it
    settled: tuple = ()         # the same writes with list arguments in order, beside what they answered


@dataclass
class Confirmation:
    """The rule's answer for one Task."""
    references: list[Recording] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # run_id -> why it is not a Reference
    groups: list[dict] = field(default_factory=list)
    reason: Optional[str] = None  # why the Task has no Reference, when it has none
    judged: bool = False
    judge_reason: Optional[str] = None
    judge_abstained: bool = False  # the judge rested on something it was not given, so it failed nothing
    judge_uncited: bool = False    # a failure of its cited nothing the states differ on (D186)
    abstain_reason: Optional[str] = None  # why it abstained, when it did
    judge_calls: list[dict] = field(default_factory=list)  # what the judge looked at, in order
    judge_fallback: Optional[str] = None  # why the one-shot judge ruled instead of the agent
    recordings: list[Recording] = field(default_factory=list)  # every Run the rule saw
    # D193. How many times the judge was asked (0 when it was not), what the second pass replied,
    # kept the way the first pass's reply is, and what the two passes together settled.
    judge_passes: int = 0
    judge_second: Optional[dict] = None
    judge_residue_resolved: bool = False   # the second pass left exactly one state in
    judge_residue_abstained: bool = False  # it left two or more, or cited nothing
    no_correct_recording: bool = False     # every state was failed, and the failures agree on why
    no_correct_key: Optional[str] = None   # the key they agree on, when they do
    # Per failed state, what its failure cited: the key, the value held there and what it should have
    # been. Kept across both passes, because the failed-all reading is over all of them at once.
    judge_citations: dict[str, dict] = field(default_factory=dict)
    # D198. The End states still in when the judgement was done, with their members, which is what
    # the derivation derives a Verifier from one by one. Empty on a Task the rule settled on its own.
    # The fields below it are the derivation's answer, written back onto this record.
    survivors: list[dict] = field(default_factory=list)
    residue_derived: bool = False           # a Verifier was derived per survivor
    survivor_reason: Optional[str] = None   # SURVIVOR_CHOSEN, SURVIVORS_EQUIVALENT or SURVIVORS_ALL_FAIL
    survivor_chosen: Optional[str] = None   # the label of the state that became the Reference
    survivor_scores: list[dict] = field(default_factory=list)  # per survivor, how its Verifier fared
    survivors_capped: bool = False          # more states survived than a Task may derive

    def as_dict(self) -> dict:
        return {"references": [{"run_id": r.run_id, "trace_id": r.trace_id, "kind": r.kind}
                               for r in self.references],
                "recordings": [{"run_id": r.run_id, "trace_id": r.trace_id, "kind": r.kind}
                               for r in self.recordings],
                "failed": dict(self.failed), "groups": list(self.groups), "reason": self.reason,
                "judged": self.judged, "judge_reason": self.judge_reason,
                "judge_abstained": self.judge_abstained, "judge_uncited": self.judge_uncited,
                "abstain_reason": self.abstain_reason, "judge_calls": list(self.judge_calls),
                "judge_fallback": self.judge_fallback,
                "judge_passes": self.judge_passes, "judge_second": self.judge_second,
                "judge_residue_resolved": self.judge_residue_resolved,
                "judge_residue_abstained": self.judge_residue_abstained,
                "no_correct_recording": self.no_correct_recording,
                "no_correct_key": self.no_correct_key,
                "judge_citations": dict(self.judge_citations),
                # The survivors by label only: the members are Recordings, and the record keeps
                # run ids, which `groups` above already carries per label.
                "survivor_labels": [str(g["label"]) for g in self.survivors],
                "residue_derived": self.residue_derived,
                "survivor_reason": self.survivor_reason,
                "survivor_chosen": self.survivor_chosen,
                "survivor_scores": list(self.survivor_scores),
                "survivors_capped": self.survivors_capped}


@dataclass
class Judgement:
    """One judge reply, read: the End states it failed, why, and any source it rested on and lacked."""
    failed: set[str] = field(default_factory=set)
    reason: str = ""
    # Per failed state, the words that failure gave for itself, which is what the record keeps of it.
    reasons: dict[str, str] = field(default_factory=dict)
    # Per failed state, what its failure pointed at: the key, the value that state holds there and
    # what the judge said it should have been. Read back when every state was failed (D193).
    citations: dict[str, dict] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    # The abstain reason when a failure of this ruling cited nothing the states differ on (D186).
    uncited: Optional[str] = None
    # What the judge looked at before it ruled (judge.py), one entry per tool call, and why the
    # one-shot judge ruled when the agent judge could not. Empty on the one-shot judge's own rulings.
    calls: list[dict] = field(default_factory=list)
    fell_back: Optional[str] = None

    @property
    def abstained(self) -> bool:
        return bool(self.unavailable) or bool(self.uncited)


# --- what a Run wrote -------------------------------------------------------

def end_state(run: Any, write_tools: Iterable[str], fn: Callable) -> tuple:
    """The Run's writes as one comparable value: tool, entity and canonical argument values, sorted.

    A Run that wrote nothing still has an outcome, and reducing every such Run to the same empty
    value hides it. When the user asked a question, what the Run did is whether its answer stated
    facts it had read from the world; a refusal such as "I'm unable to authenticate your account"
    read the world and stated none of it back. On the last build, 26 Tasks had recordings that made
    no writes, and every one of them landed in a single group: an answered recording and a refusal
    agreed, `confirm()` never called the judge, the refusals stayed References, and `derive`'s
    intersection of `communicate_values` over them came out empty, so those Verifiers were hard
    constraints plus "at most 0 writes" that an empty Run passes, and five of the nine D79 checks
    failed together. Of the 122 recordings behind those Tasks, 68 read something and stated none of
    it back, 11 read nothing, 43 stated facts. So a no-write Run whose answer states facts read from
    the world gets the ANSWERED state, a no-write Run that states nothing keeps the empty state, the
    two are separate groups, and the judge decides which of them the Intent asked for. A Run that
    wrote something is untouched: its writes are its outcome, whatever it said.
    """
    loaded = verifier_suite.as_run(run)
    effects = verifier_suite.write_effects(loaded, set(write_tools), fn)
    if not effects:
        return ANSWERED if verifier_suite.communicate_values(loaded, fn) else ()
    return tuple(sorted((e["tool"], e["entity"] or "", tuple(sorted(e["values"].items())))
                        for e in effects.values()))


def _ordered(values: Iterable[tuple], fn: Callable) -> tuple:
    """One write's arguments with the members of every list-valued one put in order.

    A list argument is a set of things the user named, and the order the recorded agent happened to
    list them in is not part of what the Run did to the world.
    """
    out = []
    for name, key in values:
        value = _decoded(key)
        if isinstance(value, list):
            key = json.dumps(sorted(_key(fn, item) for item in value), sort_keys=True)
        out.append((name, key))
    return tuple(out)


def _decoded(key: str) -> Any:
    """The value behind a canonical key. A customer's canon rules may render a list as text, so the
    key is read again while it still reads as JSON and the list inside it is not missed."""
    value: Any = key
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            break
    return value


def _results(run: Any, fn: Callable) -> dict[int, str]:
    """What each tool call was answered with, by the position of the call in the Run."""
    out: dict[int, str] = {}
    pending: list[tuple[int, Any]] = []
    for pos, event in enumerate(run.events):
        payload = dict(getattr(event, "payload", None) or {})
        if event.type == "tool_call":
            pending.append((pos, payload.get("id")))
        elif event.type == "tool_result":
            for at, call_id in reversed(pending):
                if payload.get("id") in (None, call_id) and at not in out:
                    out[at] = _key(fn, payload.get("result"))
                    break
    return out


def settled_state(run: Any, write_tools: Iterable[str], fn: Callable) -> tuple:
    """The End state with list arguments in order, each write beside what the tool answered.

    Two Runs that made the same call with the same items in another order reached the same state
    when the tool answered them the same way; when the answers differ the order mattered to the
    world and the two states stay apart. On one build 12 of 25 Tasks whose recordings
    disagreed differed by nothing else, and each of them lost its Reference over it.
    """
    loaded = verifier_suite.as_run(run)
    effects = verifier_suite.write_effects(loaded, set(write_tools), fn)
    if not effects:
        return ()
    answers = _results(loaded, fn)
    return tuple(sorted((e["tool"], e["entity"] or "", _ordered(sorted(e["values"].items()), fn),
                         answers.get(e["pos"], "")) for e in effects.values()))


def describe(state: tuple) -> str:
    if state == ANSWERED:
        return "no writes; the answer states facts read from the world"
    if not state:
        return "no writes; the answer states nothing read from the world"
    parts = []
    for tool, entity, values in state:
        args = ", ".join(f"{k}={_plain(v)}" for k, v in values if k != "")
        parts.append(f"{tool} on {entity or 'an entity'}" + (f" ({args})" if args else ""))
    return "; ".join(parts)


def _plain(canon_key: str) -> str:
    """A canonical JSON key as the judge should read it: the string without its quotes."""
    try:
        value = json.loads(canon_key)
    except (TypeError, ValueError):
        return str(canon_key)
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def hard_atoms(constraints: Iterable[Constraint], write_tools: Iterable[str],
               read_tools: Iterable[str] = ()) -> list[Atom]:
    """The compiled constraints as the hard atoms `_hard_holds` evaluates, over the Run's write calls only."""
    return [verifier_mod._atom(f"hard.{c.id}", "hard",
                               {"kind": "hard", "constraint_id": c.id, "judge": False,
                                "predicate_src": c.predicate_src, "write_tools": sorted(write_tools),
                                "read_tools": sorted(read_tools)}, description=c.text)
            for c in constraints if c.compiled and c.predicate_src and not c.judge_atom]


def violations(run: Any, atoms: Iterable[Atom], write_tools: Iterable[str], fn: Callable) -> list[str]:
    """The constraint ids whose predicate does not hold over the Run."""
    loaded = verifier_suite.as_run(run)
    return [atom.target["constraint_id"] for atom in atoms
            if verifier_suite.hard_holds(atom, loaded, set(write_tools), fn) is False]


def stated_facts(run: Any, fn: Callable) -> tuple:
    """The values this Run's answers stated back to the user, sorted, as the judge should read them.

    Part of the End state and not of the transcript (D43): what the user was told is an effect of the
    Run the same way a write is, and it is the half of the End state a Task resolved by an answer
    lives in. The values alone, never the sentences they were said in, so the judge still cannot read
    what the agent said.
    """
    return tuple(sorted(fact["text"] for fact in
                        verifier_suite.communicate_values(verifier_suite.as_run(run), fn).values()))


def transferred(run: Any) -> bool:
    """Did the Run hand the conversation on instead of finishing it? The hints are verdict.py's, so
    the rule that reads a Run as given up and the one that describes it to the judge agree."""
    loaded = verifier_suite.as_run(run)
    names = [call["name"].lower() for call in verifier_suite.run_calls(loaded)]
    return any(hint in name for name in names for hint in TRANSFER_HINTS)


def load(path: str, kind: str, *, run_id: Optional[str] = None, trace_id: Optional[str] = None,
         write_tools: Iterable[str], fn: Callable, atoms: Iterable[Atom] = ()) -> Recording:
    run = verifier_suite.as_run(path)
    return Recording(run_id=run_id or run.run_id, path=str(path), kind=kind, trace_id=trace_id,
                     end_state=end_state(run, write_tools, fn),
                     violated=violations(run, atoms, write_tools, fn),
                     stated=stated_facts(run, fn), transferred=transferred(run),
                     settled=settled_state(run, write_tools, fn))


# --- constraints against the corpus ---------------------------------------

def coded_atoms(constraints: Iterable[Constraint], write_tools: Iterable[str],
                read_tools: Iterable[str] = ()) -> list[Atom]:
    """Every constraint that has code, whether or not its own tests passed, as an atom to run.

    `hard_atoms` is what a Verdict may check and holds the compiled rules only. This is what the
    recordings are asked about, and a rule whose tests failed is asked too: its rate against the
    frontier is the evidence that says whether the code is wrong or the tests were.
    """
    return [verifier_mod._atom(f"hard.{c.id}", "hard",
                               {"kind": "hard", "constraint_id": c.id, "judge": False,
                                "predicate_src": c.predicate_src, "write_tools": sorted(write_tools),
                                "read_tools": sorted(read_tools)}, description=c.text)
            for c in constraints if c.predicate_src and not c.judge_atom]


def constraint_rates(constraints: Iterable[Constraint], runs: Iterable[Any], write_tools: Iterable[str],
                     fn: Callable, read_tools: Iterable[str] = ()) -> dict[str, dict]:
    """Per constraint with code: how many Runs it fails on, how many it saw, and how many it judged.

    A rule that judged nothing read as a rule the frontier passed, because `failed` stayed 0 while
    `runs` counted every recording; on two builds a quarter to a third of the rules with code had no
    rate at all and the rest could not be told apart from rules that had been checked and held. So
    each row now says how many Runs the rule actually answered about, and a rule that answered about
    none, or that has no code to run, carries the reason it was not checked.
    """
    atoms = coded_atoms(constraints, write_tools, read_tools)
    compiled = {c.id for c in constraints if c.compiled}
    rates = {atom.target["constraint_id"]: {"failed": 0, "runs": 0, "judged": 0} for atom in atoms}
    for run in runs:
        loaded = verifier_suite.as_run(run)
        for atom in atoms:
            row = rates[atom.target["constraint_id"]]
            row["runs"] += 1
            held = verifier_suite.hard_holds(atom, loaded, set(write_tools), fn)
            if held is None:
                continue
            row["judged"] += 1
            if held is False:
                row["failed"] += 1
    for rule in constraints:
        row = rates.get(rule.id)
        if row is None:
            rates[rule.id] = {"failed": 0, "runs": 0, "judged": 0,
                              "skipped": "the rule is a judge atom" if rule.judge_atom
                                         else "the rule compiled to no code"}
        elif not row["judged"]:
            row["skipped"] = ("no recording made a call this rule judges"
                              if row["runs"] else "there was no confirmed recording to check it against")
        if rule.id in rates and rule.id not in compiled:
            rates[rule.id]["gates"] = False  # the rate is reported; a rule whose tests failed gates nothing
    return rates


def demote(constraints: Iterable[Constraint], rates: dict[str, dict], share: float = MISCOMPILED_SHARE,
           min_runs: int = MIN_RUNS_TO_DEMOTE) -> tuple[list[Constraint], list[dict]]:
    """Keep the constraints the confirmed recordings mostly satisfy; the rest are reported, not gates.

    A rule is demoted when it fails on at least `share` of at least `min_runs` confirmed recordings.
    Below `min_runs` there is no evidence either way and the rule stays, since a build over a handful
    of traces must not lose its policy to a single odd recording.
    """
    kept, demoted = [], []
    for rule in constraints:
        row = rates.get(rule.id)
        if row and row["runs"] >= min_runs and row["failed"] / row["runs"] >= share:
            demoted.append({"id": rule.id, "text": rule.text, "failed": row["failed"], "runs": row["runs"],
                            "reason": f"fails on {row['failed']} of {row['runs']} confirmed recordings; "
                                      "a rule the frontier breaks that often is miscompiled (D76)"})
            continue
        kept.append(rule)
    return kept, demoted


# --- the rule ---------------------------------------------------------------

def _label(index: int) -> str:
    """One label per group, unique past the alphabet.

    confirm() both marks a group's members and keeps the groups the judge did not name by label, so
    two groups sharing a label would be failed and kept as one; a Task with 27 End states is rare
    and this is the whole of what it costs to be right about it.
    """
    return _LABELS[index % len(_LABELS)] + ("" if index < len(_LABELS) else str(index // len(_LABELS)))


def group(recordings: Iterable[Recording]) -> list[dict]:
    """The Runs by End state, recordings before re-rolls, in order of first appearance.

    Two states the settled state cannot tell apart are one state: the writes are the same and the
    order of a list argument is the only difference, and the tool answered both the same way.
    """
    ordered = sorted(recordings, key=lambda r: (r.kind != RECORDING, r.run_id))
    groups: list[dict] = []
    by_state: dict[tuple, dict] = {}
    by_settled: dict[tuple, dict] = {}
    for rec in ordered:
        row = by_state.get(rec.end_state)
        if row is None and rec.settled:
            row = by_settled.get(rec.settled)  # the same writes in another order, answered the same
        if row is None:
            row = {"label": _label(len(groups)), "state": describe(rec.end_state),
                   "runs": [], "members": []}
            groups.append(row)
        by_state.setdefault(rec.end_state, row)
        by_settled.setdefault(rec.settled, row)
        row["runs"].append(rec.run_id)
        row["members"].append(rec)
    for row in groups:
        row["told"] = sorted({fact for rec in row["members"] for fact in rec.stated})
        row["transferred"] = sum(1 for rec in row["members"] if rec.transferred)
    return groups


def _apply(out: Confirmation, groups: list[dict], judgement: Judgement) -> list[dict]:
    """Record one judgement's failures on the Confirmation and answer with the states it left in."""
    for g in groups:
        if g["label"] not in judgement.failed:
            continue
        why = (judgement.reasons.get(g["label"]) or judgement.reason
               or "did not reach the End state the Intent and the policy require")
        for rec in g["members"]:
            out.failed[rec.run_id] = f"judge: {why}"
        cited = judgement.citations.get(g["label"])
        if cited is not None:
            out.judge_citations[str(g["label"])] = dict(cited)
    return [g for g in groups if g["label"] not in judgement.failed]


def residue_keys(groups: list[dict]) -> list[str]:
    """The keys the states of a residue differ on, which is the whole of what could tell them apart."""
    return sorted({key for keys in differing_keys(groups).values() for key in keys})


def not_told_apart(groups: list[dict]) -> str:
    """The abstention a residue ends in: the survivors, and what a ruling had to choose between."""
    keys = residue_keys(groups)
    shown = ", ".join(keys[:MAX_DIFFERING_KEYS]) or "nothing they differ on"
    more = f" and {len(keys) - MAX_DIFFERING_KEYS} more" if len(keys) > MAX_DIFFERING_KEYS else ""
    return f"{SURVIVORS_NOT_TOLD_APART}: {shown}{more}"


def required_of_all(citations: dict[str, dict]) -> Optional[tuple[str, str]]:
    """The key and the required value every failure agrees on, when a judgement failed every state.

    Two or more failures citing one key and one `expected` that names the Intent or a numbered policy
    line are one claim, not several: no recorded End state held at that key what the Task required.
    That is a fact about the Environment or about the Intent and not about a disagreement between the
    recordings, so it is recorded under its own reason and a later beat can file it (D193). An
    `expected` naming another state cannot carry the claim, since that state was failed too, and
    failures pointing at different keys are the judge failing everything for different reasons, which
    settles nothing.
    """
    rows = list(citations.values())
    if len(rows) < 2:
        return None
    if len({_as_shown(row.get("key")) for row in rows}) != 1:
        return None
    if len({_as_shown(row.get("expected")) for row in rows}) != 1:
        return None
    want = _as_shown(rows[0].get("expected"))
    if want != INTENT_EXPECTED and not want.startswith(POLICY_EXPECTED):
        return None
    return str(rows[0].get("key")), str(rows[0].get("expected"))


def confirm(recordings: Iterable[Recording], *, intent: str = "", policy_lines: Iterable[str] = (),
            judge: Any = None, phrases: Iterable[str] = ()) -> Confirmation:
    """D111 over one Task's Runs: constraint violations out, then one agreeing End state, judge as residue.

    The judge is fail-only (D110), so one pass over three or more states may drop some and leave two
    behind. A residue is asked once more, with only the survivors and under the stop rule that one of
    them may remain (D193); what that pass leaves is the answer, and anything but one state left is
    an abstention naming the keys the survivors were not told apart on. A judgement that fails every
    state is read once more too, through what its failures cited.
    """
    out = Confirmation()
    recordings = list(recordings)
    out.recordings = list(recordings)
    for rec in recordings:
        if rec.violated:
            out.failed[rec.run_id] = "violates " + ", ".join(rec.violated)
    live = [r for r in recordings if not r.violated]
    if not recordings:
        out.reason = "no Run to confirm"
        return out
    if not live:
        out.reason = "every recording broke a Hard constraint"
        return out
    groups = group(live)
    out.groups = [{k: v for k, v in g.items() if k != "members"} for g in groups]
    remaining = groups
    if len(groups) > 1 and judge is not None:
        judgement = judge_groups(judge, intent, policy_lines, groups, phrases)
        out.judged, out.judge_passes, out.judge_reason = True, 1, judgement.reason
        out.judge_abstained = judgement.abstained
        out.judge_uncited = bool(judgement.uncited)
        out.abstain_reason = judgement.uncited or (judgement.reason if judgement.abstained else None)
        out.judge_calls = list(judgement.calls)
        out.judge_fallback = judgement.fell_back
        remaining = _apply(out, groups, judgement)
        if len(remaining) > 1 and not judgement.abstained:
            remaining = _second_pass(out, judge, intent, policy_lines, remaining, phrases)
    if len(remaining) == 1:
        out.references = list(remaining[0]["members"])
    elif not remaining:
        out.reason = "the judge failed every End state"
        required = required_of_all(out.judge_citations)
        if required is not None:
            out.no_correct_recording, out.no_correct_key = True, required[0]
            out.reason = (f"{NO_CORRECT_RECORDING}: every End state was failed on {required[0]}, "
                          f"and {required[1]} says what it should have held")
        elif out.judged:
            out.judge_abstained, out.abstain_reason = True, DIFFERING_GROUNDS
            out.reason += f"; the judge abstained, {DIFFERING_GROUNDS}"
            # Failed all, on grounds that do not agree, settles nothing: every state is still a
            # candidate and the derivation is asked which of them a Verifier can stand on (D198).
            out.survivors = list(groups)
    else:
        out.reason = (f"recordings disagree on the End state ({len(remaining)} states: "
                      + "; ".join(f"{g['label']} {g['state']}" for g in remaining) + ")")
        if out.judge_abstained:
            out.reason += f"; the judge abstained, {out.abstain_reason or out.judge_reason}"
        # A residue is what a judgement left behind, so these are survivors only where a judgement
        # was made: with no judge in the build nothing has been asked yet, and D111 keeps no
        # Reference for a Task whose recordings disagree.
        out.survivors = list(remaining) if out.judged else []
    return out


def _second_pass(out: Confirmation, judge: Any, intent: str, policy_lines: Iterable[str],
                 survivors: list[dict], phrases: Iterable[str]) -> list[dict]:
    """The one extra call a residue gets, and what it left; anything but one state left abstains.

    The survivors keep the labels the first pass gave them, so a failure of the second pass cites the
    same state the record already names, and the keys are recomputed over the survivors alone: a key
    that only told a failed state apart is no longer a difference and a failure resting on it would
    be resting on nothing in front of this pass.
    """
    if not residue_keys(survivors):
        # Nothing separates them, so there is nothing to ask about and no call is spent.
        out.judge_residue_abstained = True
        out.judge_abstained, out.abstain_reason = True, not_told_apart(survivors)
        return survivors
    judgement = judge_groups(judge, intent, policy_lines, survivors, phrases, True)
    out.judge_passes = 2
    out.judge_second = {"reason": judgement.reason, "failed": sorted(judgement.failed),
                        "calls": list(judgement.calls), "fallback": judgement.fell_back,
                        "abstained": judgement.abstained, "uncited": judgement.uncited}
    left = _apply(out, survivors, judgement)
    if len(left) == 1:
        out.judge_residue_resolved = True
        return left
    if not left:
        return left  # every survivor failed: the failed-all reading answers this one
    out.judge_residue_abstained = True
    out.judge_abstained, out.abstain_reason = True, not_told_apart(left)
    out.judge_uncited = out.judge_uncited or bool(judgement.uncited)
    return left


# --- what the states disagree on (D186) -------------------------------------

def state_values(group: dict) -> dict[str, str]:
    """One group's End state as keys and values a ruling can cite, rendered as the prompt renders them.

    A key is a column of the settled End state, `tool.entity.column`, and one key per value an answer
    stated, so the half of the End state a Task resolved by answering is citable too (D43, D182). A
    write with no named column gets `tool.entity.written` instead, so two states that called the same
    tool on different rows still differ on something nameable. The rendering is `describe`'s, so the
    value a failure quotes is the value the judge was shown and not one computed a second way.
    """
    values: dict[str, str] = {}
    for rec in group.get("members") or ():
        for tool, entity, entries in (rec.end_state or ()):
            columns = [(name, key) for name, key in entries if name]
            if not columns:
                values.setdefault(f"{tool}.{entity or '-'}.{WRITTEN}", "yes")
            for name, key in columns:
                values.setdefault(f"{tool}.{entity or '-'}.{name}", _plain(key))
    for fact in group.get("told") or ():
        values.setdefault(f"{STATED_PREFIX}{fact}", "told to the user")
    return values


def differing_keys(groups: list[dict]) -> dict[str, list[str]]:
    """Per group, the keys whose value in it is not the value every other group holds.

    Holding no value at a key is a value (`NO_VALUE`), so a state that wrote nothing differs from one
    that wrote on every column of that write, and a state that stated a fact differs from one that
    did not on that fact. This is the whole of what the judge was asked about: the states agree
    everywhere else, and a failure resting somewhere else is resting on nothing in front of it.
    """
    values = {str(g["label"]): state_values(g) for g in groups}
    out: dict[str, list[str]] = {label: [] for label in values}
    for key in sorted({key for row in values.values() for key in row}):
        held = {label: row.get(key, NO_VALUE) for label, row in values.items()}
        if len(set(held.values())) < 2:
            continue
        for label in held:
            out[label].append(key)
    return out


def differs_line(keys: Iterable[str], values: Optional[dict[str, str]] = None) -> str:
    """The line every group's block opens with: the keys a failure may cite and the value held at each.

    The value is on the line because a failure has to quote it back and is checked against the string
    the prompt used, so a judge that never saw the string could only guess it. A state that holds
    nothing at a key it differs on reads as `NO_VALUE`, which is the value it holds.
    """
    keys = list(keys)
    if not keys:
        return "differs from the other states on: nothing; it holds the values the others hold"
    held = dict(values or {})
    shown = [f"{key} = {held.get(key, NO_VALUE)}" if values is not None else key
             for key in keys[:MAX_DIFFERING_KEYS]]
    more = f" and {len(keys) - MAX_DIFFERING_KEYS} more" if len(keys) > MAX_DIFFERING_KEYS else ""
    return "differs from the other states on: " + ", ".join(shown) + more


def numbered_policy(policy_lines: Iterable[str]) -> list[tuple[int, str]]:
    """The policy as the judge cites it: one number per non-empty line, from 1, whitespace collapsed.

    The number is the line's place in the whole policy and not in whatever slice was shown, so the
    one-shot judge's first forty lines and the agent's search hits number the same rule alike.
    """
    return [(n, " ".join(text.split()))
            for n, text in enumerate((t for t in policy_lines if t and t.strip()), start=1)]


def shown_policy(policy_lines: Iterable[str]) -> list[int]:
    """The numbers of the policy lines the one-shot prompt prints, which are the ones it may cite."""
    return [n for n, _ in numbered_policy(policy_lines)[:MAX_POLICY_LINES]]


def _as_shown(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split()).casefold()


def _expected_holds(said: Any, key: str, held: str, values: dict[str, dict[str, str]], label: str,
                    policy_shown: set[str]) -> bool:
    """Does `expected` name something that was shown and that says the value should have been other?

    Another state, by label, whose value at this key is not the one the failed state holds; or the
    Intent; or one of the numbered policy lines the judge was shown. Anything else is nothing the
    judge can be held to, and the failure resting on it is uncited.
    """
    want = _as_shown(said)
    if not want:
        return False
    if want == INTENT_EXPECTED:
        return True
    if want.startswith(POLICY_EXPECTED):
        return want[len(POLICY_EXPECTED):].strip() in policy_shown
    other = want.removeprefix("state ").removeprefix("group ").strip().upper()
    row = values.get(other)
    return row is not None and other != label and _as_shown(row.get(key, NO_VALUE)) != _as_shown(held)


def read_failures(entries: Any, groups: list[dict],
                  policy_shown: Iterable[Any] = ()) -> tuple[dict[str, str], dict[str, dict], Optional[str]]:
    """The states a reply fails, what each failure cited, or the key that carried nothing.

    The judge is called only where the states disagree, so its judgement is which of the differences
    is the wrong one, and a failure is a claim about one named value: the state, the key, the value
    that state holds and what it should have been. All four are checked here against the material
    the prompt showed, the value against the rendered string the prompt used. On two corpora the
    reason the judge gave most often for a wrong failure was a confirmation it was never handed, and
    the `evidence` list it wrote itself never caught it, because it named allowed sources and rested
    on an absent one. A key nothing in front of it differs on is that same failure, in code.

    The citations come back beside the reasons because a judgement that failed every state is read
    again through them: failures agreeing on one key and one required value say something a residue
    does not (D193), and the reason text alone cannot be compared that way.
    """
    values = {str(g["label"]).upper(): state_values(g) for g in groups}
    differing = {label.upper(): keys for label, keys in differing_keys(groups).items()}
    numbers = {str(n).strip() for n in policy_shown}
    failed: dict[str, str] = {}
    cited: dict[str, dict] = {}
    for entry in entries or ():
        if not isinstance(entry, dict):
            return {}, {}, NO_KEY  # a bare label cites nothing, whatever else the reply said
        label = str(entry.get("group") or "").strip().upper()
        if label not in values:
            continue  # it names no state of this Task, so it fails nothing, as an unknown label always did
        said = " ".join(str(entry.get("key") or "").split())
        # The key is matched without case, because it is a name the judge copies back out of the
        # prompt; that it is one of the keys the states differ on is what is being checked here.
        key = {k.casefold(): k for k in differing[label]}.get(said.casefold())
        if key is None:
            return {}, {}, said or NO_KEY
        held = values[label].get(key, NO_VALUE)
        if _as_shown(entry.get("value")) != _as_shown(held):
            return {}, {}, key
        if not _expected_holds(entry.get("expected"), key, held, values, label, numbers):
            return {}, {}, key
        failed.setdefault(label, str(entry.get("reason") or "")[:MAX_LINE_CHARS])
        cited.setdefault(label, {"key": key, "value": held,
                                 "expected": str(entry.get("expected") or "")[:MAX_LINE_CHARS]})
    return failed, cited, None


# --- the judge: fails, never passes ----------------------------------------

def told_line(group: dict) -> str:
    """One group's answer side: the facts its Runs stated back, and how many gave the Run away.

    Without it a Task the recorded agent resolved by answering reads as a Task nobody did anything
    about: the judge saw "no writes" and failed the state, and on two builds it failed 15 of 59 and
    18 of 22 recordings the corpus itself had rewarded.
    """
    facts = list(group.get("told") or [])
    shown = ", ".join(str(f) for f in facts[:MAX_FACTS])
    more = f" and {len(facts) - MAX_FACTS} more" if len(facts) > MAX_FACTS else ""
    said = f"told the user: {shown}{more}" if facts else "told the user no fact read from the world"
    handed = int(group.get("transferred") or 0)
    total = len(group.get("runs") or [])
    if handed:
        return f"{said}; {handed} of {total} handed the conversation on"
    return f"{said}; none handed the conversation on"


def cited_shape(sources: Iterable[str]) -> str:
    """The shape of a judgement, said once, with one general example (D144's order, D186's rule).

    The example is a booking desk, a domain no corpus we build on has, so nothing here teaches a
    judge what to say about the traces it is judging. Both judges render this block: the one-shot
    judge from its three sources, the agent judge from its own.
    """
    return "\n".join([
        "Reply with JSON only:",
        '{"failed": [{"group": "A", "key": "book_slot.SLOT-7.starts_at", "value": "09:00", '
        '"expected": "B", "reason": "the Intent asked for the later slot and B holds it"}], '
        '"evidence": ["' + '", "'.join(list(sources)[:2]) + '"], "reason": "..."}',
        "Every name in evidence must be one of " + ", ".join(sources) + "; when your answer needs "
        "anything else, name that instead and fail nothing.",
        'Every failure names four things. group: the state you fail. key: one key off that state\'s '
        '"differs from the other states on" line, and nothing else. value: what that state holds at '
        'that key, copied from that same line word for word, which is "' + NO_VALUE + '" where the '
        "state holds nothing there. expected: what the value should have been, written as the label "
        "of another state, or as intent, or as policy:<n> for one of the numbered policy lines. "
        "reason: your own words, kept for the record.",
        "All four are checked in code. A failure whose key is not one the states differ on, or whose "
        "value is not the value that state holds, or whose expected names nothing you were shown, is "
        "not a failure, and a judgement holding one is dropped whole: nothing is failed and the Task "
        "keeps no Reference. So fail a state only where you can point at a value it holds and say "
        "what it should have been instead.",
        "Mark a state failed only when you are sure it did not do what the Intent asked and the policy "
        'allows; when you cannot tell, reply {"failed": [], "reason": "cannot tell"}.',
    ])


def one_state_left(groups: list[dict]) -> str:
    """The stop rule of a second pass: one of these states may remain, or you could not tell (D193).

    Said last, after the shape of a judgement, because it is the rule for stopping and not for
    reading the material. It names the two answers that settle the Task and says plainly that any
    other answer settles nothing, so a judge that keeps hedging is not left thinking it has helped.
    """
    labels = ", ".join(str(g["label"]) for g in groups)
    return "\n".join([
        f"This is the second and last time you are asked about this Task. The states left are {labels}, "
        "the ones your first ruling did not fail, and exactly one of them may remain.",
        "So there are two answers. Either fail every one of them but the one that did what the Intent "
        "asked, each failure citing a key these states differ on, as above; or, when the keys they "
        "differ on do not tell you which one did, say so and fail nothing: reply "
        '{"failed": [], "reason": "' + SURVIVORS_NOT_TOLD_APART + '"}.',
        "Leaving two or more of them in, or failing all of them, settles nothing: the Task then keeps "
        "no Reference at all, which is worse than saying you could not tell.",
    ])


def _judge_head() -> str:
    """The fixed prologue, the same bytes for every Task and every pass (D245).

    Sent as the system message so the provider reads a stable prefix; the per Task case rides in
    the user turn. Every sentence here also stands in `judge_prompt`, moved and never reworded.
    The second pass stop rule names the surviving states, so it stays in the user turn.
    """
    return "\n".join([
        "Recordings of one Task ended in different states. Say which of the states did NOT do what "
        "the Intent asked, or did something the policy does not allow.",
        "",
        "You have three sources and your answer may rest on nothing else: "
        + ", ".join(AVAILABLE_SOURCES) + ".",
        "You do not have the transcript. Whether the agent authenticated the user, asked for and was "
        "given a confirmation, said the right thing or called its tools in the right order cannot be "
        "seen from here, and no state is ever failed for want of evidence you were not given.",
        "",
        "The user may have revised the opening request during the Run, so judge the Intent and never "
        "the opening request.",
        "", "Policy, by line number:",
        "", "End states. Each says what the state's Runs wrote, which keys it differs from the other "
        "states on, what their answers told the user and whether they handed the conversation on. "
        "What the user was told is an effect of the Run and part of the End state (D43); the values "
        "are given without the sentences they were said in, and you still do not have the transcript. "
        "A state that told the user no value read from the world may still have answered in words "
        "that carry none, so that on its own is not a state failed.",
        "", cited_shape(AVAILABLE_SOURCES),
    ])


def _judge_body(intent: str, policy_lines: Iterable[str], groups: list[dict],
               final: bool = False) -> str:
    """The per Task case: the intent, the policy lines and the end states (D245).

    Sent as the user message under `_judge_head`. The stop rule of a second pass names the
    survivors, so it rides here too.
    """
    lines = ["Intent, the ground truth of what the user wanted by the end of the Run: "
             + (' '.join((intent or '').split())[:MAX_REQUEST_CHARS] or '(not recorded)')]
    lines += [f"{n}. {text[:MAX_LINE_CHARS]}"
              for n, text in numbered_policy(policy_lines)[:MAX_POLICY_LINES]]
    differing = differing_keys(groups)
    for g in groups:
        lines.append(f"{g['label']} ({len(g['runs'])} run{'s' if len(g['runs']) != 1 else ''}): {g['state']}")
        lines.append("    " + differs_line(differing.get(str(g["label"])) or (), state_values(g)))
        lines.append("    " + told_line(g))
    if final:
        lines += ["", one_state_left(groups)]
    return "\n".join(lines)


def _judge_messages(intent: str, policy_lines: Iterable[str], groups: list[dict],
                   final: bool = False) -> list[dict]:
    """The request messages: the stable prologue as system, the case as user (D245)."""
    return [{"role": "system", "content": _judge_head()},
            {"role": "user", "content": _judge_body(intent, policy_lines, groups, final)}]


def judge_prompt(intent: str, policy_lines: Iterable[str], groups: list[dict], final: bool = False) -> str:
    lines = ["Recordings of one Task ended in different states. Say which of the states did NOT do what "
             "the Intent asked, or did something the policy does not allow.",
             "",
             "You have three sources and your answer may rest on nothing else: "
             + ", ".join(AVAILABLE_SOURCES) + ".",
             "You do not have the transcript. Whether the agent authenticated the user, asked for and was "
             "given a confirmation, said the right thing or called its tools in the right order cannot be "
             "seen from here, and no state is ever failed for want of evidence you were not given.",
             "",
             "Intent, the ground truth of what the user wanted by the end of the Run: "
             + (' '.join((intent or '').split())[:MAX_REQUEST_CHARS] or '(not recorded)'),
             "The user may have revised the opening request during the Run, so judge the Intent and never "
             "the opening request.",
             "", "Policy, by line number:"]
    lines += [f"{n}. {text[:MAX_LINE_CHARS]}"
              for n, text in numbered_policy(policy_lines)[:MAX_POLICY_LINES]]
    lines += ["", "End states. Each says what the state's Runs wrote, which keys it differs from the other "
              "states on, what their answers told the user and whether they handed the conversation on. "
              "What the user was told is an effect of the Run and part of the End state (D43); the values "
              "are given without the sentences they were said in, and you still do not have the transcript. "
              "A state that told the user no value read from the world may still have answered in words "
              "that carry none, so that on its own is not a state failed."]
    differing = differing_keys(groups)
    for g in groups:
        lines.append(f"{g['label']} ({len(g['runs'])} run{'s' if len(g['runs']) != 1 else ''}): {g['state']}")
        lines.append("    " + differs_line(differing.get(str(g["label"])) or (), state_values(g)))
        lines.append("    " + told_line(g))
    lines += ["", cited_shape(AVAILABLE_SOURCES)]
    if final:
        lines += ["", one_state_left(groups)]
    return "\n".join(lines)


def judge_groups(model: Any, intent: str, policy_lines: Iterable[str], groups: list[dict],
                 phrases: Iterable[str] = (), final: bool = False) -> Judgement:
    """What the judge said about the End states; an unreadable reply fails nothing (D110).

    A judge that can look answers `rule_groups` and rules through its own tools (judge.py); anything
    else is one model call over this prompt, which is also what the agent judge falls back to. The
    dispatch is on the object rather than an import, so this module stays the one the agent judge
    reads its prompt pieces and its ruling parser from and not the other way round.

    `final` is the second pass over a residue (D193): the same dispatch, the same parser, the same
    checks on a failure, and one stop rule more saying exactly one of the states in front of it may
    remain. Both judges take it through here, so the opt-in agent gets the second pass too.
    """
    ruler = getattr(model, "rule_groups", None)
    if callable(ruler):
        return ruler(intent, policy_lines, groups, phrases, final)
    try:
        reply = model.query(_judge_messages(intent, policy_lines, groups, final))
    except Exception as exc:
        return Judgement(reason=f"judge call failed: {type(exc).__name__}")
    return parse_judgement(getattr(reply, "content", None) or "", groups,
                           policy_shown=shown_policy(policy_lines))


def parse_judgement(text: str, groups: list[dict], available: Iterable[str] = AVAILABLE_SOURCES,
                    policy_shown: Iterable[Any] = ()) -> Judgement:
    """The judge's reply as a ruling. A ruling resting on a source it was not handed fails nothing (D93).

    `available` is what this judge was given, and the one-shot judge's three are the default. The
    agent judge (judge.py) hands its own list, its six tools, because a ruling of its that rests on
    the rows it read rests on something it was given and abstaining on that would be D93 read
    backwards; what is absent by construction for either of them, the conversation, the opening
    request, an authentication, a spoken confirmation, is absent from both lists.

    The groups themselves come in rather than their labels alone, because D186's check is against
    what the states differ on and the labels do not carry it; `policy_shown` is the numbers of the
    policy lines this judge was actually shown, which are the ones an `expected` of its may cite.
    """
    match = _JSON_RE.search(text or "")
    if not match:
        return Judgement(reason=UNREADABLE_REPLY)
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Judgement(reason=UNREADABLE_REPLY)
    failed = body.get("failed") if isinstance(body, dict) else None
    if not isinstance(failed, list):
        return Judgement(reason=UNREADABLE_REPLY)
    said = str(body.get("reason") or "")[:MAX_LINE_CHARS]
    missing = sources_not_given(body.get("evidence"), available)
    if missing:
        return Judgement(reason=f"the ruling rests on {', '.join(missing)}, which this judge was not given"
                                + (f"; the judge said: {said}" if said else ""),
                         unavailable=missing)
    per_state, cited, uncited = read_failures(failed, list(groups), policy_shown)
    if uncited is not None:
        why = f"a failure rested on nothing the states differ on: {uncited}"
        return Judgement(reason=why + (f"; the judge said: {said}" if said else ""), uncited=why)
    return Judgement(failed=set(per_state), reason=said, reasons=per_state, citations=cited)


__all__ = ["RECORDING", "REROLL", "ANSWERED", "MISCOMPILED_SHARE", "AVAILABLE_SOURCES", "UNREADABLE_REPLY",
           "MAX_DIFFERING_KEYS", "NO_KEY", "NO_VALUE", "STATED_PREFIX", "WRITTEN",
           "SURVIVORS_NOT_TOLD_APART", "DIFFERING_GROUNDS", "NO_CORRECT_RECORDING",
           "SURVIVOR_CHOSEN", "SURVIVORS_EMPTY", "SURVIVORS_EQUIVALENT", "SURVIVORS_ALL_FAIL",
           "Recording", "Confirmation",
           "Judgement", "end_state", "settled_state", "describe", "stated_facts", "transferred",
           "told_line", "state_values", "differing_keys", "differs_line", "numbered_policy",
           "shown_policy", "read_failures", "cited_shape",
           "hard_atoms", "coded_atoms", "violations", "load", "constraint_rates", "demote", "group",
           "confirm", "residue_keys", "not_told_apart", "required_of_all",
           "one_state_left", "judge_prompt", "judge_groups", "parse_judgement"]
