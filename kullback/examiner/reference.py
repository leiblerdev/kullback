"""References by the D111 rule, and the constraints checked against them before they gate anything.

Which recorded Runs a Verifier may be derived from is settled before the Verifier exists, because
the Verifier is derived from those Runs and cannot judge them (D43). The rule is what the Intent
plus the policy say the End state should hold: a recording that broke a compiled Hard constraint is
a failed recording; the rest are grouped by End state, and the References are the one group that
agrees. When more than one group is left, code cannot tell which of them carried out the request, so
a judge may mark groups as failed, never as passed (D110); a Task whose groups still disagree gets
no Reference and no Verdict. That judge is handed the Intent, the policy and the End states and no
transcript, so a ruling of its that rests on anything else, authentication or a spoken confirmation
or the opening request, is an abstention and fails nothing (D93). Re-rolls (D112) enter the same rule
as recordings of a lower standing: the Reference is a recording whenever the agreeing group holds
one, since the recording is the only Run that touched the customer's real system.

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
from kullback.runner.judge import sources_not_given
from kullback.runner.records import Atom, Constraint

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
# What this judge is handed, and the whole of what a ruling of its may rest on. It sees the Intent,
# the policy and the End states, and never a transcript, so authentication, a spoken confirmation and
# the opening request are absent by construction; a ruling that rests on one of them abstains (D93).
AVAILABLE_SOURCES = ("intent", "policy", "end_states")
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Recording:
    """One Run the D111 rule sees: where it is, what it wrote, and which Hard constraints it broke."""
    run_id: str
    path: str
    kind: str = RECORDING
    trace_id: Optional[str] = None
    end_state: tuple = ()
    violated: list[str] = field(default_factory=list)


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
    recordings: list[Recording] = field(default_factory=list)  # every Run the rule saw

    def as_dict(self) -> dict:
        return {"references": [{"run_id": r.run_id, "trace_id": r.trace_id, "kind": r.kind}
                               for r in self.references],
                "recordings": [{"run_id": r.run_id, "trace_id": r.trace_id, "kind": r.kind}
                               for r in self.recordings],
                "failed": dict(self.failed), "groups": list(self.groups), "reason": self.reason,
                "judged": self.judged, "judge_reason": self.judge_reason,
                "judge_abstained": self.judge_abstained}


@dataclass
class Judgement:
    """One judge reply, read: the End states it failed, why, and any source it rested on and lacked."""
    failed: set[str] = field(default_factory=set)
    reason: str = ""
    unavailable: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return bool(self.unavailable)


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


def load(path: str, kind: str, *, run_id: Optional[str] = None, trace_id: Optional[str] = None,
         write_tools: Iterable[str], fn: Callable, atoms: Iterable[Atom] = ()) -> Recording:
    run = verifier_suite.as_run(path)
    return Recording(run_id=run_id or run.run_id, path=str(path), kind=kind, trace_id=trace_id,
                     end_state=end_state(run, write_tools, fn),
                     violated=violations(run, atoms, write_tools, fn))


# --- constraints against the corpus ---------------------------------------

def constraint_rates(constraints: Iterable[Constraint], runs: Iterable[Any], write_tools: Iterable[str],
                     fn: Callable, read_tools: Iterable[str] = ()) -> dict[str, dict]:
    """Per compiled constraint: how many of the given Runs it fails on, and how many it saw."""
    atoms = hard_atoms(constraints, write_tools, read_tools)
    rates = {atom.target["constraint_id"]: {"failed": 0, "runs": 0} for atom in atoms}
    for run in runs:
        loaded = verifier_suite.as_run(run)
        for atom in atoms:
            row = rates[atom.target["constraint_id"]]
            row["runs"] += 1
            if verifier_suite.hard_holds(atom, loaded, set(write_tools), fn) is False:
                row["failed"] += 1
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
    """The Runs by End state, recordings before re-rolls, in order of first appearance."""
    ordered = sorted(recordings, key=lambda r: (r.kind != RECORDING, r.run_id))
    groups: list[dict] = []
    by_state: dict[tuple, dict] = {}
    for rec in ordered:
        row = by_state.get(rec.end_state)
        if row is None:
            row = by_state[rec.end_state] = {"label": _label(len(groups)),
                                             "state": describe(rec.end_state), "runs": [], "members": []}
            groups.append(row)
        row["runs"].append(rec.run_id)
        row["members"].append(rec)
    return groups


def confirm(recordings: Iterable[Recording], *, intent: str = "", policy_lines: Iterable[str] = (),
            judge: Any = None) -> Confirmation:
    """D111 over one Task's Runs: constraint violations out, then one agreeing End state, judge as residue."""
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
        judgement = judge_groups(judge, intent, policy_lines, groups)
        out.judged, out.judge_reason = True, judgement.reason
        out.judge_abstained = judgement.abstained
        for g in groups:
            if g["label"] in judgement.failed:
                why = judgement.reason or "did not reach the End state the Intent and the policy require"
                for rec in g["members"]:
                    out.failed[rec.run_id] = f"judge: {why}"
        remaining = [g for g in groups if g["label"] not in judgement.failed]
    if len(remaining) == 1:
        out.references = list(remaining[0]["members"])
    elif not remaining:
        out.reason = "the judge failed every End state"
    else:
        out.reason = (f"recordings disagree on the End state ({len(remaining)} states: "
                      + "; ".join(f"{g['label']} {g['state']}" for g in remaining) + ")")
        if out.judge_abstained:
            out.reason += f"; the judge abstained, {out.judge_reason}"
    return out


# --- the judge: fails, never passes ----------------------------------------

def judge_prompt(intent: str, policy_lines: Iterable[str], groups: list[dict]) -> str:
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
             "", "Policy:"]
    lines += [f"- {' '.join(text.split())[:MAX_LINE_CHARS]}"
              for text in list(policy_lines)[:MAX_POLICY_LINES] if text and text.strip()]
    lines += ["", "End states:"]
    lines += [f"{g['label']} ({len(g['runs'])} run{'s' if len(g['runs']) != 1 else ''}): {g['state']}"
              for g in groups]
    lines += ["", 'Reply with JSON only: {"failed": ["A"], "evidence": ["intent", "end_states"], '
              '"reason": "..."}. Every name in evidence must be one of ' + ", ".join(AVAILABLE_SOURCES)
              + '; when your answer needs anything else, name that instead and fail nothing. Mark a state '
              'failed only when you are sure it did not do what the Intent asked and the policy allows; '
              'when you cannot tell, reply {"failed": [], "reason": "cannot tell"}.']
    return "\n".join(lines)


def judge_groups(model: Any, intent: str, policy_lines: Iterable[str], groups: list[dict]) -> Judgement:
    """What the judge said about the End states; an unreadable reply fails nothing (D110)."""
    try:
        reply = model.query([{"role": "user", "content": judge_prompt(intent, policy_lines, groups)}])
    except Exception as exc:
        return Judgement(reason=f"judge call failed: {type(exc).__name__}")
    return parse_judgement(getattr(reply, "content", None) or "", {g["label"] for g in groups})


def parse_judgement(text: str, labels: set[str]) -> Judgement:
    """The judge's reply as a ruling. A ruling resting on a source it was not handed fails nothing (D93)."""
    match = _JSON_RE.search(text or "")
    if not match:
        return Judgement(reason="unreadable reply")
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Judgement(reason="unreadable reply")
    failed = body.get("failed") if isinstance(body, dict) else None
    if not isinstance(failed, list):
        return Judgement(reason="unreadable reply")
    said = str(body.get("reason") or "")[:MAX_LINE_CHARS]
    missing = sources_not_given(body.get("evidence"), AVAILABLE_SOURCES)
    if missing:
        return Judgement(reason=f"the ruling rests on {', '.join(missing)}, which this judge was not given"
                                + (f"; the judge said: {said}" if said else ""),
                         unavailable=missing)
    return Judgement(failed={str(x).strip().upper() for x in failed} & labels, reason=said)


__all__ = ["RECORDING", "REROLL", "ANSWERED", "MISCOMPILED_SHARE", "AVAILABLE_SOURCES", "Recording", "Confirmation",
           "Judgement", "end_state", "describe",
           "hard_atoms", "violations", "load", "constraint_rates", "demote", "group", "confirm",
           "judge_prompt", "judge_groups", "parse_judgement"]
