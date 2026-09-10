"""User fidelity: how close a driver's turns are to the ones the recording holds (D214 rule 5).

Tool fidelity asks whether a compiled body answers a recorded call the way the recording did. This
asks the same question of the other half of the Environment, which nothing was asking: given the
recorded conversation up to a point, does the Simulated user say what that user said next. Four
readings per turn, all of them code over the Task's own records:

  recall      the askable facts the recorded turn carried that this turn carries too
  precision   no fact the recorded turn did not carry, so a user that reads out its whole file
              scores worse than one that answers the question
  record      no value only the world holds (D210), which is the leak read as a number
  end         the same decision about ending, on the turn where the recorded user ended

The Task's score is the mean over its recorded turns after the first, in [0, 1]. The rule-driven
user is scored exactly this way and costs nothing, and that is the baseline: an agent user earns a
Task by beating it there and nowhere else. It is a measurement in this decision and not a gate; it
becomes a gate in a later one, once a corpus has shown what it is worth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from kullback.runner.records import Record, Trace, UserFact, read_json, write_json
from kullback.user import rules as rules_mod

FILE_NAME = "user_fidelity.json"
# The shape of the file. Bumped when a reading's meaning changes, so a file written under an older
# meaning reads as an older file rather than as today's numbers.
FORMAT = 1

RULES_DRIVER = "rules"
AGENT_DRIVER = "agent"
# The bands a difficulty record carries a Task's user fidelity in (D209's shape): a band means the
# same on every corpus, and a number to three decimals invites a reader to compare two corpora's
# means as if they were the same measurement.
BANDS = ((0.9, "high"), (0.7, "fair"), (0.4, "low"))
NO_BAND = "poor"


def band(score: Optional[float]) -> Optional[str]:
    """One score as its band, or None where the Task has no score."""
    if score is None:
        return None
    return next((name for floor, name in BANDS if score >= floor), NO_BAND)


class TurnScore(Record):
    """One recorded turn and one driver's answer to it, in four readings."""

    recall: float = 1.0
    precision: float = 1.0
    record_clean: float = 1.0
    end_match: float = 1.0
    score: float = 1.0
    missed: list[str] = []          # facts the recorded turn carried and this one did not
    invented: list[str] = []        # facts this turn carried that the recorded turn did not
    record_spoken: list[str] = []   # fields only the world holds that this turn said out loud
    end_wrong: bool = False


class TaskScore(Record):
    """One driver over one Task: the mean, the turns, and what it got wrong (the lesson's input)."""

    task_id: str = ""
    driver: str = RULES_DRIVER
    score: float = 0.0
    turns: int = 0
    turn_scores: list[TurnScore] = []
    missed: list[str] = []
    invented: list[str] = []
    record_spoken: list[str] = []
    ends_wrong: int = 0


def facts_carried(text: Optional[str], facts: Iterable[UserFact]) -> list[str]:
    """The askable facts one turn says out loud, by field, however the turn spells them.

    `_said_in` is the rule-driven user's own reading of whether a turn stated a value, so the two
    drivers are scored by one rule and a difference between them is a difference in what they said.
    """
    out: list[str] = []
    for fact in facts:
        if fact.field in rules_mod.SPOKEN_FIELDS or fact.value is None:
            continue
        if fact.field not in out and rules_mod._said_in(text, str(fact.value)):
            out.append(fact.field)
    return out


def record_spoken(text: Optional[str], record_values: Optional[dict] = None) -> list[str]:
    """The fields only the world holds that this turn said a value of (D210's leak, counted)."""
    return [field for field, value in (record_values or {}).items()
            if value is not None and rules_mod._said_in(text, str(value))]


def _ratio(hit: int, total: int) -> float:
    return 1.0 if total == 0 else hit / total


def score_turn(said: Optional[str], recorded: Optional[str], facts: Iterable[UserFact], *,
               record_values: Optional[dict] = None, ended: bool = False,
               recorded_ended: bool = False) -> TurnScore:
    """One turn of one driver against the recorded turn it stands in for."""
    facts = list(facts)
    mine = facts_carried(said, facts)
    theirs = facts_carried(recorded, facts)
    missed = [field for field in theirs if field not in mine]
    invented = [field for field in mine if field not in theirs]
    spoken = record_spoken(said, record_values)
    recall = _ratio(len(theirs) - len(missed), len(theirs))
    precision = _ratio(len(mine) - len(invented), len(mine))
    clean = 0.0 if spoken else 1.0
    end_match = 1.0 if bool(ended) == bool(recorded_ended) else 0.0
    return TurnScore(recall=recall, precision=precision, record_clean=clean, end_match=end_match,
                     score=(recall + precision + clean + end_match) / 4,
                     missed=missed, invented=invented, record_spoken=spoken,
                     end_wrong=end_match == 0.0)


def score_of(turns: Sequence[TurnScore]) -> float:
    return sum(t.score for t in turns) / len(turns) if turns else 0.0


def as_task_score(task_id: str, driver: str, turns: Sequence[TurnScore]) -> TaskScore:
    return TaskScore(
        task_id=task_id, driver=driver, score=score_of(turns), turns=len(turns),
        turn_scores=list(turns),
        missed=list(dict.fromkeys(f for t in turns for f in t.missed)),
        invented=list(dict.fromkeys(f for t in turns for f in t.invented)),
        record_spoken=list(dict.fromkeys(f for t in turns for f in t.record_spoken)),
        ends_wrong=sum(1 for t in turns if t.end_wrong))


def recorded_turns(trace: Trace) -> list[Any]:
    """The recorded user turns a driver is scored on: every one after the first.

    The first is the opening, which every driver produces from the goal alone and which therefore
    separates no two drivers; scoring it would put the same number on both and flatten the mean.
    """
    return [t for t in trace.turns if t.role == "user"][1:]


def prefix_messages(trace: Trace, before_idx: int) -> list[dict]:
    """The conversation as it stood before one recorded turn, in the shape a driver's `reply` takes."""
    return [{"role": turn.role, "content": turn.content or ""}
            for turn in trace.turns if turn.idx < before_idx and turn.role in ("user", "assistant")]


def score_driver(driver: Any, trace: Trace, facts: Iterable[UserFact], *, task_id: str = "",
                 driver_name: str = RULES_DRIVER, record_values: Optional[dict] = None) -> TaskScore:
    """One driver over one recorded conversation, turn by turn, in order.

    The driver is asked for its turn at each recorded user turn with the recorded prefix in front of
    it, so a driver that carries state (which turn it is on, what it has already given away) carries
    the state the recorded conversation would have given it, and not the state its own answers would.
    """
    facts = list(facts)
    turns = recorded_turns(trace)
    last_idx = turns[-1].idx if turns else -1
    scored: list[TurnScore] = []
    for turn in turns:
        prefix = prefix_messages(trace, turn.idx)
        try:
            said = driver.reply(prefix)
        except Exception:  # a driver that fell over on a turn scores that turn as nothing said
            said = ""
        recorded_ended = turn.idx == last_idx and bool(
            rules_mod.CLOSING_CUE.search(rules_mod._norm(turn.content)))
        scored.append(score_turn(said, turn.content, facts, record_values=record_values,
                                 ended=bool(getattr(driver, "done", False)),
                                 recorded_ended=recorded_ended))
        if getattr(driver, "done", False):
            break
    return as_task_score(task_id, driver_name, scored)


# --- the corpus table -------------------------------------------------------------------------

def summarise(rows: Iterable[dict]) -> dict:
    """The corpus line: the mean per driver, the Tasks at 1.0, and where the agent beats the rules."""
    rows = [row for row in rows or () if isinstance(row, dict)]
    out: dict[str, Any] = {"tasks": len(rows)}
    for name in (RULES_DRIVER, AGENT_DRIVER):
        scores = [float(row[name]) for row in rows if row.get(name) is not None]
        out[name] = {
            "tasks": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
            "at_one": sum(1 for s in scores if s >= 1.0),
            "bands": _band_counts(scores),
        }
    both = [row for row in rows if row.get(AGENT_DRIVER) is not None and row.get(RULES_DRIVER) is not None]
    out["agent_beats_rules"] = sum(1 for row in both if row[AGENT_DRIVER] > row[RULES_DRIVER])
    out["compared"] = len(both)
    return out


def _band_counts(scores: Sequence[float]) -> dict[str, int]:
    counts = {name: 0 for _, name in BANDS}
    counts[NO_BAND] = 0
    for score in scores:
        counts[band(score) or NO_BAND] += 1
    return counts


def write_scores(workdir: Any, rows: Iterable[dict], round: int = 0) -> Path:
    """user_fidelity.json beside tool_fidelity.json: the per-Task rows and the corpus summary."""
    rows = [dict(row) for row in rows or ()]
    body = {"format": FORMAT, "round": round, "tasks": rows, "summary": summarise(rows)}
    return write_json(Path(workdir) / FILE_NAME, body)


def load_scores(workdir: Any) -> dict:
    body = read_json(Path(workdir) / FILE_NAME, {}) or {}
    return body if isinstance(body, dict) and body.get("format") == FORMAT else {}


def markdown_table(body: dict) -> list[str]:
    """The per-corpus table the CLI prints and a report embeds."""
    summary = body.get("summary") or {}
    lines = ["| driver | Tasks | mean | at 1.0 | high | fair | low | poor |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in (RULES_DRIVER, AGENT_DRIVER):
        row = summary.get(name) or {}
        bands = row.get("bands") or {}
        mean = row.get("mean")
        lines.append(f"| {name} | {row.get('tasks', 0)} | {'-' if mean is None else f'{mean:.3f}'} | "
                     f"{row.get('at_one', 0)} | {bands.get('high', 0)} | {bands.get('fair', 0)} | "
                     f"{bands.get('low', 0)} | {bands.get('poor', 0)} |")
    lines.append(f"agent beats rules on {summary.get('agent_beats_rules', 0)} of "
                 f"{summary.get('compared', 0)} Tasks scored both ways")
    return lines


# --- one workdir's Tasks ------------------------------------------------------------------------
# The records a score is read off. They are the harness's own file names and not any customer's, and
# they are read here rather than handed in because the CLI verb and the round driver both want the
# same answer and only one of them has a build plan in hand.
REPLAYS = "replays.json"
RULES_DIR = "user_rules"
TRACES_DIR = "traces"
TASKS_DIR = "tasks"
SIGS = "tool_sigs.json"
VOCABULARY = "vocabulary.json"


def _json(path: Path, default: Any) -> Any:
    return read_json(path, default)


def write_tools_of(workdir: Any) -> list[str]:
    """The tools that change the world, off the mined signatures; empty when there are none yet."""
    rows = _json(Path(workdir) / SIGS, []) or []
    return [str(row.get("name")) for row in rows
            if isinstance(row, dict) and row.get("kind") == "write" and row.get("name")]


def vocabulary_of(workdir: Any):
    """This build's Vocabulary, or the generic core where the build has not derived one."""
    from kullback.user.vocabulary import GENERIC, Vocabulary
    body = _json(Path(workdir) / VOCABULARY, None)
    if not isinstance(body, dict):
        return GENERIC
    try:
        return Vocabulary.model_validate(body)
    except ValueError:
        return GENERIC


def references(workdir: Any) -> dict[str, str]:
    """The Reference recording of each Task: {task_id: trace_id}, off replays.json.

    The confirmed replay is the Reference (D79), and a Task with none has no recorded conversation
    to be scored against, so it is absent rather than scored on nothing.
    """
    body = _json(Path(workdir) / REPLAYS, {}) or {}
    out: dict[str, str] = {}
    for task_id, rows in (body.items() if isinstance(body, dict) else ()):
        for trace_id, row in sorted((rows or {}).items()):
            if isinstance(row, dict) and row.get("confirmed"):
                out[task_id] = str(row.get("trace_id") or trace_id)
                break
    return out


def trace_index(workdir: Any) -> dict[str, Trace]:
    """Every Trace on disk, by trace_id; the file names are content hashes, not ids."""
    out: dict[str, Trace] = {}
    folder = Path(workdir) / TRACES_DIR
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        body = _json(path, None)
        if not isinstance(body, dict):
            continue
        try:
            trace = Trace.model_validate(body)
        except ValueError:
            continue
        out[trace.trace_id] = trace
    return out


def rules_for(workdir: Any, trace_id: str):
    """The Simulated user rules mined for one recording, or None."""
    from kullback.runner.records import UserRules
    body = _json(Path(workdir) / RULES_DIR / f"{trace_id}.json", None)
    if not isinstance(body, dict):
        return None
    try:
        return UserRules.model_validate(body)
    except ValueError:
        return None


def contexts_of(workdir: Any, lessons_for: Any = None) -> dict:
    """One curated TaskContext per Task with a Reference (D214 rule 1), keyed by Task id."""
    from kullback.user.context import curate, mine_record_values
    traces = trace_index(workdir)
    writes = write_tools_of(workdir)
    vocab = vocabulary_of(workdir)
    out: dict[str, Any] = {}
    for task_id, trace_id in references(workdir).items():
        user_rules = rules_for(workdir, trace_id)
        trace = traces.get(trace_id)
        if user_rules is None or trace is None:
            continue
        record = mine_record_values(trace)
        lessons = list(lessons_for(task_id) if lessons_for is not None else ())
        ctx = curate(task_id, user_rules, trace, vocab=vocab, write_tools=writes,
                     record_fields=sorted(record), lessons=lessons)
        out[task_id] = (ctx, user_rules, trace, record)
    return out


def task_ids(workdir: Any, limit: Optional[int] = None) -> Optional[list[str]]:
    """The Tasks a score would cover, in id order, cut to `limit`.

    A live scoring pays per recorded turn, so a first reading on a corpus is taken on a slice of it
    rather than on all of it. Without a limit this answers None, which is every Task.
    """
    if not limit or limit <= 0:
        return None
    return sorted(references(workdir))[:limit]


def score_workdir(workdir: Any, *, make_agent: Any = None, tasks: Optional[Iterable[str]] = None,
                  round: int = 0, write: bool = True) -> dict:
    """Both drivers over every Task of a workdir that has a Reference (D214 rule 5).

    The rule-driven user is scored with no world reader and no model, which is the baseline every
    corpus is compared on: it answers from the Task's own rules, which is what it does in a Run when
    the world holds nothing more. `make_agent(ctx, fallback, record)` builds the agent user for one
    Task and costs one model call per recorded turn; without it only the baseline is computed, and
    that is the whole offline measurement.
    """
    from kullback.user import rules as rules_module
    wanted = set(tasks or ())
    vocab = vocabulary_of(workdir)
    rows: list[dict] = []
    scores: dict[str, dict] = {}
    for task_id, (ctx, user_rules, trace, record) in sorted(contexts_of(workdir).items()):
        if wanted and task_id not in wanted:
            continue
        facts = ctx.askable()
        floor = rules_module.SimulatedUser(user_rules, vocab=vocab)
        rules_score = score_driver(floor, trace, facts, task_id=task_id,
                                   driver_name=RULES_DRIVER, record_values=record)
        row = {"task_id": task_id, RULES_DRIVER: rules_score.score, AGENT_DRIVER: None,
               "turns": rules_score.turns, "band": band(rules_score.score)}
        agent_score = None
        if make_agent is not None:
            driver = make_agent(ctx, rules_module.SimulatedUser(user_rules, vocab=vocab), record)
            if driver is not None:
                agent_score = score_driver(driver, trace, facts, task_id=task_id,
                                           driver_name=AGENT_DRIVER, record_values=record)
                row[AGENT_DRIVER] = agent_score.score
                row["agent_band"] = band(agent_score.score)
                row["guards"] = dict(getattr(driver, "guards", None).counts) if getattr(
                    driver, "guards", None) is not None else {}
                row["driver_counts"] = dict(getattr(driver, "counts", {}) or {})
        rows.append(row)
        scores[task_id] = {RULES_DRIVER: rules_score, AGENT_DRIVER: agent_score}
    body = {"format": FORMAT, "round": round, "tasks": rows, "summary": summarise(rows)}
    if write:
        write_json(Path(workdir) / FILE_NAME, body)
    return {"body": body, "scores": scores}


# --- the dry run: how often the rule-driven user runs out of scenario --------------------------
RUNS_DIR = "runs"
REROLL_RECORD = "rerolls.json"
REFUSED_WRITE_ENDS = "user_ended_on_refused_write"


def refused_write_ends(workdir: Any, write_tools: Optional[Iterable[str]] = None) -> dict:
    """How often the old goal rule would have ended a Run on a write the world refused (D227).

    The count a live build reads to see this decision working: a Run lands here when its transcript
    shows a write-kind tool called, every one of those calls came back carrying the D67 error marker,
    and the user ended the Run anyway. Under the rule this replaces those Runs ended goal_satisfied
    while the same Run was graded as claiming a write it never made; under the new one the goal stays
    open, the user restates it once and the Run runs on to its own end kind.

    Off the stored Runs and nothing else, so it costs no model call and reads workdirs written
    before this decision as readily as after it.
    """
    names = frozenset(write_tools if write_tools is not None else write_tools_of(workdir))
    out = {REFUSED_WRITE_ENDS: 0, "runs_read": 0, "runs_with_a_write": 0, "runs_with_no_end_kind": 0}
    folder = Path(workdir) / RUNS_DIR
    if not names or not folder.is_dir():
        return out
    for path in sorted(folder.glob("*/*.jsonl")):
        called, took_effect, satisfied, classified = False, False, False, False
        for event in _events(path):
            kind, payload = event.get("type"), event.get("payload") or {}
            if kind == "tool_result" and payload.get("name") in names:
                called = True
                took_effect = took_effect or payload.get("error") is None
            elif kind == "user_turn" and rules_mod.end_kind_of(payload) is not None:
                classified = True
                # Only the one kind the old rule reached over a refused write. A Run the Candidate
                # closed or the user ran out of scenario on ended the way it would have ended
                # anyway, so counting those would say this decision moved Runs it never touched.
                satisfied = satisfied or rules_mod.end_kind_of(payload) == rules_mod.GOAL_SATISFIED
        out["runs_read"] += 1
        out["runs_with_a_write"] += int(called)
        # A Run written before the end kinds existed carries none, and its stop reason is `user_stop`
        # whichever way the user ended it, so it cannot be classified either way and is counted here
        # rather than folded into the refusal count on the strength of a reason that says nothing.
        out["runs_with_no_end_kind"] += int(called and not classified)
        out[REFUSED_WRITE_ENDS] += int(called and not took_effect and satisfied)
    return out


ENDS_BY_DRIVER = "user_ends_by_kind"


def ends_by_driver(workdir: Any) -> dict[str, dict[str, int]]:
    """How the stored Runs ended, in the kinds of D210, under the user that ended each of them (D231).

    The turn a Run ends on names its own driver, so the split is read off the same payload the end
    kind is read off and never off which driver the Task was assigned. A Task the agent user drives
    still ends on a rules turn wherever that turn was dropped or the model could not answer, and a
    split taken off the assignment would file those Runs under the user that did not speak them. A
    turn that names no driver is the rule-driven user's own, which is the only one that writes none.

    Every kind is named for every driver that ended a Run, zero included, so a driver that stopped
    ending one way says so rather than dropping the line, and a driver that ended nothing is absent
    rather than a row of zeros: two thirds of Runs running out of scenario under one user is the
    reading this exists for, and it cannot be had from a count that has already summed the two.

    Off the stored Runs and nothing else, so it costs no model call and no Run. Those are the Runs
    the workdir holds now, not a pile that grows with each round: a stage replaces a Task's Runs
    whole before it writes its own (`_discard_runs`), and a Task this round left alone keeps the
    Runs it was last given, which is the same reading runs.json and the scorecard's Task coverage
    are taken off.
    """
    out: dict[str, dict[str, int]] = {}
    folder = Path(workdir) / RUNS_DIR
    for path in sorted(folder.glob("*/*.jsonl")) if folder.is_dir() else ():
        ended: Optional[tuple[str, str]] = None
        for event in _events(path):
            if event.get("type") != "user_turn":
                continue
            payload = event.get("payload") or {}
            kind = rules_mod.end_kind_of(payload)
            if kind is not None:
                ended = (str(payload.get("driver") or RULES_DRIVER), kind)
        if ended is None:
            continue
        driver, kind = ended
        counts = out.setdefault(driver, {name: 0 for name in rules_mod.USER_END_KINDS})
        counts[kind] = counts.get(kind, 0) + 1
    return {driver: counts for driver, counts in sorted(out.items())}


def _events(path: Path) -> Iterator[dict]:
    """One stored Run's events, skipping a line the file was cut off in the middle of writing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def dry_run_counts(workdir: Any) -> dict:
    """How the Runs of every Task ended, off the re-roll records the build already wrote (D210).

    The trigger for D214: a corpus whose Runs mostly end `scenario_exhausted` is a corpus whose
    Simulated user runs out of things to say, and that is the number an agent user has to move. No
    model call and no Run: this reads what is on disk.
    """
    from kullback.user.rules import USER_END_KINDS
    counts = {kind: 0 for kind in USER_END_KINDS}
    counts["unclassified"] = 0
    reasons: dict[str, int] = {}
    tasks = 0
    folder = Path(workdir) / RUNS_DIR
    for path in sorted(folder.glob(f"*/{REROLL_RECORD}")) if folder.is_dir() else ():
        body = _json(path, {}) or {}
        rows = body.get("runs") or [] if isinstance(body, dict) else []
        if rows:
            tasks += 1
        for row in rows:
            kind = row.get("user_end") if isinstance(row, dict) else None
            counts[kind if kind in counts else "unclassified"] += 1
            reason = str((row or {}).get("termination_reason") or "none")
            reasons[reason] = reasons.get(reason, 0) + 1
    total = sum(counts.values())
    # A workdir built before the end kinds existed (D210) carries none, and every Run reads as
    # unclassified. The Runner's own termination reasons are reported beside them for that case, so
    # a reader of an older workdir is told the kinds are missing rather than shown four zeros.
    return {"tasks": tasks, "runs": total, "ends": counts,
            "termination_reasons": dict(sorted(reasons.items())),
            "classified": total - counts["unclassified"],
            "exhausted_share": round(counts["scenario_exhausted"] / total, 4) if total else None,
            **refused_write_ends(workdir)}
