"""Decide one Run's pass or fail from its End state against a Task's Verifier, in code only (D43, D46, D94)."""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

from kullback.judge.shape import NO_EVIDENCE, has_evidence, is_item
from kullback.runner import target as _target
from kullback.runner.atom_context import AtomContext, _evaluate, gate, is_transfer
from kullback.runner.canon import UNRESOLVED, Unresolved, record_use
from kullback.runner.expected import Match, cell_results, collateral
from kullback.runner.records import (
    VERDICT_VERSION,
    Atom,
    Conduct,
    EndState,
    Forbidden,
    ItemResult,
    Run,
    Verdict,
    Verifier,
    load_run_jsonl,
)

# AtomContext, gate and _evaluate live in runner/atom_context.py: they are what a Verdict evaluates
# an atom's predicate against, and the confinement gate they share with gates/confinement.py's
# constraint gate (runner/confinement.py) is easier to find next to that world model than buried in
# this module's own top.

MUST_HOLD = {"required", "question", "communicate", "hard"}
GAVE_UP = {"transfer", "transferred", "agent_transfer", "gave_up", "no_action"}
ENV_ERROR_REASONS = {"env_error", "environment_error", "environment_cannot_answer"}
# How each judge use says "this holds", "this does not" and "I did not decide" (judge.py's _USES).
JUDGE_HOLDS = {"pass", "equivalent", "acceptable", "good_reference"}
JUDGE_FAILS = {"fail", "not_equivalent", "unacceptable", "bad_reference"}
_JUDGE_WORDS = {True: "pass", False: "fail", None: "abstain"}
CAUSES = {"candidate", "environment", "simulated_user", "undetermined"}
# The item kinds the Spec writer puts on an atom's target (spec/items.py). The state kinds and the
# sanity item are read as the end state's cells; the event kinds are decided here, off the calls
# and the log. Either way the item's gate flag decides, never the atom's kind (D330).
STATE_ITEMS = ("row_is", "row_new", "row_keeps", "nothing_else")
EVENT_ITEMS = ("called", "not_called", "before", "said")
CONFIRM_TURN = "confirm_turn"


# --- loading a stored Run ---

def load_run(source: Any) -> Run:
    """Read a Run from a JSONL path, a dict or a Run; header lines, event lines and a footer all work."""
    if isinstance(source, Run):
        return source
    if isinstance(source, dict):
        return Run.model_validate(source)
    return load_run_jsonl(source)


def _judge_says(result: Any) -> Optional[bool]:
    """Does this judge result say the atom holds? None means the judge did not decide (D76).

    A JudgeResult's verdict is a word, not a flag: `fail` and `abstain` are both non-empty strings,
    so reading one as a bool passed every judged Run. bool() is never the answer here, and an
    abstain is never a pass: D76 sends it to a person.
    """
    if isinstance(result, bool):
        return result
    data = result if isinstance(result, dict) else {
        key: getattr(result, key, None) for key in ("pass", "passed", "verdict")}
    for key in ("pass", "passed"):
        if isinstance(data.get(key), bool):
            return data[key]
    word = str(data.get("verdict") or "").strip().lower()
    if word in JUDGE_HOLDS:
        return True
    if word in JUDGE_FAILS:
        return False
    return None  # abstain, undetermined, an unknown word, or a shape this code does not read


def _env_marks(run: Run, flagged_tools: Iterable[str]) -> list[str]:
    """D88 code-first marks: assisted, fact_unavailable, a flagged or low-fidelity tool, an overlay miss."""
    flagged = set(flagged_tools or ())
    marks: list[str] = ["env_mark:assisted"] if run.assisted else []
    for event in run.events:
        payload = event.payload or {}
        found = []
        if event.assisted or payload.get("assisted") or event.route == "llm":
            found.append("env_mark:assisted")
        tags = payload.get("tags") or []
        if (payload.get("fact_unavailable") or payload.get("tag") == "fact_unavailable"
                or "fact_unavailable" in tags):
            found.append("env_mark:fact_unavailable")
        if payload.get("overlay_miss"):
            found.append("env_mark:overlay_miss")
        if payload.get("name") in flagged:
            found.append(f"env_mark:flagged_tool:{payload['name']}")
        marks.extend(m for m in found if m not in marks)
    return marks


def _termination(run: Run) -> str:
    if run.termination_reason:
        return run.termination_reason
    for event in reversed(run.events):
        if event.type == "stop":
            return str((event.payload or {}).get("termination_reason") or "")
    return ""


def _env_error(run: Run) -> bool:
    """An infrastructure failure inside the Environment, not a Candidate mistake."""
    if _termination(run).lower() in ENV_ERROR_REASONS:
        return True
    return any(event.type == "error" and ((event.payload or {}).get("environment")
               or (event.payload or {}).get("class") == "env_error") for event in run.events)


def _cannot_answer_tool(run: Run) -> str:
    """The tool name on the Run's cannot-answer event, empty when the record holds none."""
    for event in reversed(run.events):
        if event.type != "tool_result":
            continue
        payload = event.payload or {}
        error = payload.get("error") or {}
        if (event.route == "cannot_answer" or error.get("class") == "cannot_answer"
                or payload.get("reason") == "environment_cannot_answer"):
            return str(payload.get("name") or (error.get("payload") or {}).get("tool") or "")
    return ""


def _named_cause(cause_result: Any, notes: list[str]) -> Optional[str]:
    """The cause a judge named for this failure, with its cited spans kept beside it (D88)."""
    if cause_result is None:
        return None
    if isinstance(cause_result, str):
        word, spans = cause_result, []
    elif isinstance(cause_result, dict):
        word, spans = str(cause_result.get("verdict") or ""), cause_result.get("cited_spans") or []
    else:
        word = str(getattr(cause_result, "verdict", "") or "")
        spans = getattr(cause_result, "cited_spans", None) or []
    word = word.strip().lower()
    if word not in CAUSES:
        return None
    for span in spans:
        notes.append(f"judge_span:{span}")
    return word


def _score_predicate(atom: Atom, context: AtomContext, notes: list[str],
                     unresolved_ids: set[str]) -> Optional[bool]:
    refused = gate(atom.predicate_src) if atom.predicate_src else []
    if not atom.predicate_src:
        notes.append(f"atom_without_predicate:{atom.id}")
        return None
    if refused:
        notes.append(f"atom_rejected:{atom.id}:{refused[0]}")
        return None
    context.marking = atom.kind != "forbidden"
    try:
        return _evaluate(atom.predicate_src, context.env())
    except Unresolved as open_pair:
        unresolved_ids.add(atom.id)
        notes.append(f"atom_unresolved:{atom.id}:{open_pair.column}")
    except Exception as error:
        notes.append(f"atom_error:{atom.id}:{type(error).__name__}")
    finally:
        context.marking = True
    return None


def _judge_opinion(atom: Atom, judge_results: Optional[dict], notes: list[str]) -> tuple[Optional[bool], bool]:
    """What the judge said of this atom (None when it did not run or abstained), and whether it ran."""
    if judge_results is None or atom.id not in judge_results:
        notes.append(f"judge_atom_unevaluated:{atom.id}")
        return None, False
    opinion = _judge_says(judge_results[atom.id])
    notes.append(f"judge_reported:{atom.id}:{_JUDGE_WORDS[opinion]}")
    if opinion is None:
        notes.append(f"judge_abstained:{atom.id}")
    return opinion, True


def _calls_with(context: AtomContext, tool: str, args: Optional[dict]) -> list[dict]:
    """The calls of `tool` that answered without an error and carry every one of `args`."""
    return [call for call in context.calls if call["name"] == tool and not call["error"]
            and all(context.c((call["args"] or {}).get(k)) == context.c(v) for k, v in (args or {}).items())]


def _event_item_holds(target: dict, context: AtomContext) -> bool:
    """One event item on one Run: a call made or not made (with its arguments), an order, a value said.

    An order whose later call never happened holds, as confirm_before_write does: whether the call had
    to happen is the end state's question. A value is said when an assistant message contains it.
    """
    kind = target["kind"]
    if kind == "called":
        return bool(_calls_with(context, target["tool"], target.get("args")))
    if kind == "not_called":
        return not _calls_with(context, target["tool"], target.get("args"))
    if kind == "said":
        texts = [context.t(text) for _, text in context.assistant]

        def stated(value: Any) -> bool:
            needle = context.t(value)
            return bool(needle) and any(needle in text for text in texts)
        return (all(stated(v) for v in target.get("values") or [])
                and not any(stated(v) for v in target.get("not_values") or []))
    if target["first"] == CONFIRM_TURN:
        return context.confirmed_before_first_write(target["then"])
    then = _calls_with(context, target["then"], None)
    if not then:
        return True
    first = _calls_with(context, target["first"], None)
    return bool(first) and first[0]["idx"] < then[0]["idx"]


class _Item(NamedTuple):
    """One item's result, the name a failure or an open gate reports, and whether an open gate blocks."""
    result: ItemResult
    name: str
    blocks: bool = True


def _item(item_id: str, kind: str, source: Any, holds: Optional[bool], name: str, why: Optional[str] = None,
          blocks: bool = True, **judged: Any) -> _Item:
    return _Item(ItemResult(id=item_id, kind=kind, gate=source.gate, weight=source.weight, holds=holds,
                            why=why, **judged), name, blocks)


def _judge_item(atom: Atom, run: Run, judge_results: Optional[dict], notes: list[str]) -> tuple[_Item, bool]:
    """A judge item (D328): the item judge's score decides it like a code item, gate or scored.

    A Run with no turn the item's evidence names scores 0 by code, judged or not. Otherwise an
    unjudged item (the judge failed or was not asked) is open: a gate leaves the Run not verdicted,
    never failed, and a scored one masks the score.
    """
    result = (judge_results or {}).get(atom.id)
    row = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
    if row.get("score") is None and not has_evidence(run, atom.target or {}):
        row = {"score": 0, "why": NO_EVIDENCE}
    score = row.get("score")
    notes.append(f"judge_item:{atom.id}:{'unjudged' if score is None else score}")
    holds = None if score is None else score == 1
    return _item(atom.id, "judge", atom, holds, atom.id, row.get("why") or atom.description,
                 score=score, truncated=bool(row.get("truncated"))), result is not None


def _atom_items(verifier: Verifier, context: AtomContext, judge_results: Optional[dict],
                notes: list[str]) -> tuple[list[_Item], bool]:
    """Every atom of one Verifier as an item, and whether a judge answered any (judge_used).

    One scorer (G3): every atom that is not a Hard rule and not a judge atom is read off its
    structured target by kullback/runner/target.py, the same interpreter the gates call. Hard
    rules keep their compiled predicate source, which is policy code by nature; judge atoms are
    answered by judge.py. An atom with no target kind (unit fixtures only; every stored atom
    carries one) still evaluates its predicate. An item the Spec writer wrote is decided by its own
    gate flag: a state item is read as the end state's cells, an event item off the calls and the
    log, a judge item by its score. An older allowed atom is a permission, not an item: it is
    evaluated for the writes it covers and reports nothing. A forbidden atom holds as an item
    when its forbidden state is absent. An item-shaped judge atom is settled by the item judge's
    score, gate or scored, and stays open only when it went unjudged. An older judge atom is never
    settled in a gate: a gate one stays open (D76), a scored one carries the judge's opinion.
    """
    items: list[_Item] = []
    judge_used = False
    unresolved_ids: set[str] = set()
    fn = _target.canon_fn(context.rules if context.rules is not None else context._canon)
    tools = _target.scored_write_tools(verifier, context.run, context.write_tools)
    effects = _target.write_effects(context.run, tools, fn)
    asked = set(_target.question_keys(context.run, effects, fn))
    said = set(_target.communicate_values(context.run, fn))
    # Hard atoms run last: without a write-tool set write_calls() is what the other atoms covered,
    # so a hard atom placed first in the Verifier would see an empty list and hold vacuously.
    for atom in sorted(verifier.atoms, key=lambda a: a.kind == "hard"):
        if is_item(atom):
            item, used = _judge_item(atom, context.run, judge_results, notes)
            judge_used = judge_used or used
            items.append(item)
            continue
        kind = (atom.target or {}).get("kind")
        if kind in STATE_ITEMS:
            continue
        if kind in EVENT_ITEMS:
            items.append(_item(atom.id, "event", atom, _event_item_holds(atom.target, context),
                               f"gate:{kind}:{atom.id}", atom.description))
            continue
        if atom.judge:
            opinion, used = _judge_opinion(atom, judge_results, notes)
            judge_used = judge_used or used
            holds = None if atom.gate else opinion
            items.append(_item(atom.id, "judge", atom, holds, atom.id, atom.description,
                               blocks=atom.kind in MUST_HOLD))
            continue
        if atom.kind == "hard":
            raw = _score_predicate(atom, context, notes, unresolved_ids)
        else:
            payload = _target.atom_payload(atom)
            if not payload.get("kind"):
                raw = _score_predicate(atom, context, notes, unresolved_ids)
            else:
                try:
                    raw = _target.atom_holds(atom, context.run, fn, tools,
                                             effects=effects, asked=asked, said=said)
                except Exception as error:
                    notes.append(f"atom_error:{atom.id}:{type(error).__name__}")
                    raw = None
        if atom.kind == "allowed":
            continue
        holds = raw if atom.kind != "forbidden" or raw is None else not raw
        # An atom that could not be checked leaves the Run not verdicted; a counted pass here
        # would hide a Verifier defect or an unsettled pair (D76, D79, D219).
        blocks = atom.kind in MUST_HOLD or atom.kind == "forbidden" or atom.id in unresolved_ids
        items.append(_item(atom.id, "atom", atom, holds, atom.id,
                           atom.description or atom.predicate_src or atom.kind, blocks))
    if unresolved_ids:
        notes.append(f"atoms_unresolved={len(unresolved_ids)}")
    order = {atom.id: i for i, atom in enumerate(verifier.atoms)}
    items.sort(key=lambda item: order.get(item.result.id, 0))
    return items, judge_used


# --- the gates: end state, sanity, forbidden list, conduct (D316, D329) ---

def _forbidden_happened(rule: Forbidden, context: AtomContext, moved: dict) -> Optional[bool]:
    """Did this forbidden write or Run-ending call happen? Read off the calls and the diff."""
    if rule.tool is not None:
        hits = [call for call in context.calls if call["name"] == rule.tool and not call["error"]]
        if rule.row_id is not None:
            hits = [call for call in hits if any(context.c(v) == context.c(rule.row_id)
                                                 for v in (call["args"] or {}).values())]
        if hits or rule.kind == "end_call" or rule.table is None:
            return bool(hits)
    if rule.table is None:
        return None
    unsettled = False
    for key, change in moved.items():
        table, _, row_id = key.partition(".")
        if table != rule.table or (rule.row_id is not None and row_id != rule.row_id):
            continue
        if rule.field is None:
            return True
        field = change["fields"].get(rule.field)
        if field is None:
            continue
        if field.get("unresolved"):
            unsettled = True
        elif rule.value is None or context.c(rule.value) == field["after"]:
            return True
    return None if unsettled else False


def _conduct_holds(rule: Conduct, context: AtomContext) -> Optional[bool]:
    if rule.kind == "confirm_before_write":
        return None if rule.tool is None else context.confirmed_before_first_write(rule.tool)
    if rule.kind == "called":
        return None if rule.tool is None else context.called(rule.tool)
    message = (rule.source.ptr or {}).get("message")
    if not message:
        return None
    errors = [call["error"] for call in context.calls if call["name"] == rule.tool and call["error"]]
    said = [text for _, text in context.assistant] + [str(e.get("message") if isinstance(e, dict) else e)
                                                      for e in errors]
    return any(context.t(message) in context.t(text) for text in said)


def _best_end_state(verifier: Verifier, context: AtomContext) -> tuple[EndState, list, Match]:
    """The end state the Run is read against: any one matching whole, else the closest (D329).

    Closest is an end state whose gate cells and sanity all hold, then all cells held, then no
    cell failed (only unsettled), then the most cell weight held.
    """
    best = None
    for state in verifier.expected:
        cells = cell_results(state, context)
        rest = collateral(state, context)
        held = sum(cell.weight for cell, ok, _ in cells if ok)
        gates = all(ok for cell, ok, _ in cells if cell.gate) and rest.ok is True
        key = (gates, all(ok for _, ok, _ in cells) and rest.ok is True, all(ok for _, ok, _ in cells),
               all(ok is not False for _, ok, _ in cells), held)
        if best is None or key > best[0]:
            best = (key, state, cells, rest)
    return best[1], best[2], best[3]


def _state_items(verifier: Verifier, context: AtomContext) -> list[_Item]:
    """The end state's cells and the sanity item; with no end state the sanity item is read later."""
    if not verifier.expected:
        return []
    _, cells, rest = _best_end_state(verifier, context)
    items = [_item(f"state:{cell.table}.{cell.row_id}.{cell.field}", "state", cell, ok, f"gate:expected:{why}", why)
             for cell, ok, why in cells]
    why = rest.why[0] if rest.why else None
    items.append(_Item(ItemResult(id="sanity", kind="sanity", gate=True, holds=rest.ok, why=why),
                       f"gate:sanity:{why}"))
    return items


def _event_items(verifier: Verifier, context: AtomContext) -> list[_Item]:
    """The forbidden list and the conduct rules as items, read off the calls, the diff and the log."""
    items: list[_Item] = []
    moved = context.diff() if verifier.forbidden else {}
    for i, rule in enumerate(verifier.forbidden):
        happened = _forbidden_happened(rule, context, moved)
        where = f"gate:forbidden:{i}:{rule.tool or ''}:{rule.table or ''}.{rule.row_id or ''}.{rule.field or ''}"
        items.append(_item(f"forbidden:{i}", "event", rule, None if happened is None else not happened, where))
    for i, rule in enumerate(verifier.conduct):
        where = f"gate:conduct:{i}:{rule.kind}:{rule.tool or ''}"
        items.append(_item(f"conduct:{i}", "event", rule, _conduct_holds(rule, context), where))
    return items


def _gates(verifier: Verifier, context: AtomContext) -> tuple[Optional[str], Optional[str]]:
    """The first state, sanity, forbidden or conduct gate that failed and the first open one, named."""
    items = [item for item in _state_items(verifier, context) + _event_items(verifier, context)
             if item.result.gate]
    failed = next((item.name for item in items if item.result.holds is False), None)
    unsettled = next((item.name for item in items if item.result.holds is None), None)
    return failed, unsettled


def _classify(run: Run, context: AtomContext, cause_result: Any, marks: list[str], names: list[str],
              passed: bool, is_env_error: bool, not_verdicted: bool,
              notes: list[str]) -> tuple[str, Optional[str], bool]:
    """The Run's class, its cause and whether the Environment is suspected, from the marks alone."""
    if is_env_error:
        return "env_error", "environment", True
    if not_verdicted:
        # An atom the Verifier could not evaluate is an immature Verifier, not a broken Environment:
        # design section 6 calls this state "Task not verdicted", so it is not blamed on the
        # Environment and does not count as a Candidate failure either.
        return "not_verdicted", "undetermined", False
    if passed:
        return "pass", None, False
    # A transfer changes nothing even where mine.py classed the transfer tool as a write (D46).
    acting = [call for call in context.write_calls() if not is_transfer(call["name"])]
    transferred = not acting and (
        any(is_transfer(name) for name in names) or _termination(run).lower() in GAVE_UP
    )
    klass = "transferred_without_acting" if transferred else "fail"
    cause = _named_cause(cause_result, notes)  # code marks it, the judge names the cause (D88)
    suspected = bool(marks) or cause == "environment"
    if cause is None and not suspected:
        notes.append("cause_pending_judge")
    return klass, cause, suspected


def _extra_write_outcome(verifier: Verifier, context: AtomContext) -> Optional[tuple[str, str]]:
    _has_targets = any(
        _target.atom_payload(a).get("kind") in ("write", "write_value")
        and a.kind != "forbidden" for a in verifier.atoms)
    if _has_targets:
        _fn = _target.canon_fn(context.rules if context.rules is not None else context._canon)
        _tools = _target.scored_write_tools(verifier, context.run, context.write_tools)
        _effects = _target.write_effects(context.run, _tools, _fn)
        _extra = _target._extra_write(
            verifier, _effects,
            _tools if _target.judge_write_tools(verifier) else context.write_tools)
        if _extra is not None:
            return _extra, f"failing_atom:{_extra}: write not required and not allowed by any atom"
        return None
    extras = context.extra_writes()
    if extras:
        _extra = f"extra_write:{extras[0]['name']}"
        return _extra, f"failing_atom:{_extra}: write not required and not allowed by any atom"
    return None


def _sanity_by_writes(verifier: Verifier, context: AtomContext, atoms: list[_Item]) -> Optional[_Item]:
    """The sanity item of a Verifier with no end state: no write that no atom declares (D329).

    An open atom may be the one that would have declared the write, so beside one the item is open
    too, and it blocks nothing of its own: the open atom already leaves the Run not verdicted.
    """
    if verifier.expected:
        return None
    if any(item.result.holds is None and item.result.kind == "atom" and item.blocks for item in atoms):
        return _Item(ItemResult(id="sanity", kind="sanity", gate=True, holds=None, why="an atom is open"),
                     "sanity", blocks=False)
    extra = _extra_write_outcome(verifier, context)
    holds = extra is None
    name, why = (None, None) if holds else extra
    return _Item(ItemResult(id="sanity", kind="sanity", gate=True, holds=holds, why=why), name or "sanity")


def _note_comparisons(context: AtomContext, verifier: Verifier, workdir: Any, run_id: str,
                      notes: list[str]) -> bool:
    judged = False
    for comparison in context.comparisons:
        judged = judged or bool(getattr(comparison, "judge_used", False))
        if getattr(comparison, "route", None) == UNRESOLVED:
            notes.append(f"semantic_unresolved:{comparison.key}")
        if workdir is not None:
            record_use(workdir, comparison, run_id, verifier.task_id)
    return judged


def _decide_outcome(items: list[_Item], context: AtomContext, notes: list[str]) -> tuple[Optional[str], bool]:
    """The first gate item that failed, else the first open gate item that blocks (not verdicted).

    A definite failure anywhere wins over an open gate (D316); a scored item never fails the Run.
    """
    gates = [item for item in items if item.result.gate]
    failed = next((item for item in gates if item.result.holds is False), None)
    if failed is not None:
        if failed.result.kind in ("atom", "judge"):
            notes.append(f"failing_atom:{failed.name}: {failed.result.why}")
        elif failed.result.kind == "sanity" and failed.result.why and failed.result.why.startswith("failing_atom:"):
            notes.append(failed.result.why)
        else:
            notes.append(f"failing_atom:{failed.name}")
        return failed.name, False
    for item in items:
        if not item.result.gate and item.result.holds is False:
            notes.append(f"scored_fail:{item.result.id}")
    open_gate = next((item for item in gates if item.result.holds is None and item.blocks), None)
    if open_gate is not None:
        if open_gate.result.kind in ("atom", "judge"):
            notes.append(f"not_verdicted:{open_gate.name}: an item could not be evaluated")
        else:
            notes.append(f"not_verdicted:{open_gate.name}")
        return open_gate.name, True
    if context.write_tools is None:
        notes.append("side_effect_check_skipped")
    return None, False


def _score(items: list[_Item], passed: bool, decided: bool, notes: list[str]) -> Optional[float]:
    """0 when a gate failed; past the gates the weighted mean of every item, 1.0 with none (D329).

    The gate items all held, so they count in full, and together they hold at least half the
    weight (agreed 2026-09-27): a Run that got the end state right never scores under one half,
    and one that did nothing else right never scores above what the scored items allow.
    None, never a false 0, when the Run is not verdicted or an env error, or a scored item is open.
    """
    if not decided:
        return None
    if not passed:
        return 0.0
    scored = [item.result for item in items if not item.result.gate]
    open_items = [result.id for result in scored if result.holds is None]
    if open_items:
        notes.append(f"score_masked:{open_items[0]}")
        return None
    rest = sum(result.weight for result in scored)
    gates = max(sum(item.result.weight for item in items if item.result.gate), rest)
    if not gates + rest:
        return 1.0
    return (gates + sum(result.weight for result in scored if result.holds)) / (gates + rest)


# What the reward of a Run is, in the words a package manifest and a report carry (D329).
REWARD_SHAPE = {
    "pass": "every gate item holds",
    "score": "0 when a gate item fails, else the weighted mean of every item with the gate items held and "
             "holding at least half the weight; 1.0 with no scored item",
    "masked": "no pass and no score when the Run is not verdicted or an environment error; no score when a "
              "scored item is open",
    "sanity": "a gate item for every Task: nothing outside the declared rows changed",
}


def item_counts(verifier: Verifier) -> dict:
    """How many gate and scored items one Verifier holds, the sanity item included, read without a Run.

    Several end states are alternatives (any one may match), so the cells of the largest one count.
    """
    flags: list[bool] = [True]  # the sanity item
    if verifier.expected:
        flags += [cell.gate for cell in max((state.cells for state in verifier.expected), key=len)]
    flags += [rule.gate for rule in verifier.forbidden] + [rule.gate for rule in verifier.conduct]
    flags += [atom.gate for atom in verifier.atoms
              if (atom.target or {}).get("kind") not in STATE_ITEMS
              and (atom.kind != "allowed" or is_item(atom) or (atom.target or {}).get("kind") in EVENT_ITEMS)]
    return {"gate": sum(flags), "scored": len(flags) - sum(flags), "end_states": len(verifier.expected)}


def _side_effect_count(verifier: Verifier, context: AtomContext) -> int:
    if not _target.judge_write_tools(verifier):
        return context.writes_count()
    tools = _target.scored_write_tools(verifier, context.run, context.write_tools)
    return sum(1 for call in context.calls if not call["error"] and call["name"] in tools)


def verdict(run_jsonl: Any, verifier: Verifier, canon: Any = None, judge_results: Optional[dict] = None,
            *, environment: Any = None, runner_version: Optional[str] = None,
            reference_path: Optional[Iterable[str]] = None, write_tools: Optional[Iterable[str]] = None,
            flagged_tools: Iterable[str] = (), schema: Any = None, cause_result: Any = None,
            rules: Any = None, equivalence: Any = None, workdir: Any = None,
            verdict_version: str = VERDICT_VERSION) -> Verdict:
    """Pass or fail one stored Run on its End state; never calls a model, judge atoms arrive as results (D76).

    The Verifier's gates decide first (D316): one of its expected end states matches with nothing
    else moved, no forbidden write or call happened, and its conduct shows in the event log. A gate
    that cannot be settled leaves the Run not verdicted naming the pair. Then the atoms, as before.
    A Verifier with no expected end state and no forbidden or conduct rule is checked by its atoms
    alone, exactly as before, so every stored Verifier keeps working.

    `cause_result` is judge.py's answer for this Run's failure cause (D88); code marks the Run,
    the judge names the cause, and neither is computed here.
    """
    run = load_run(run_jsonl)
    context = AtomContext(run, canon, write_tools, schema, rules=rules, equivalence=equivalence)
    notes: list[str] = []
    atoms, judge_used = _atom_items(verifier, context, judge_results, notes)
    items = _state_items(verifier, context) + _event_items(verifier, context) + atoms
    sanity = _sanity_by_writes(verifier, context, atoms)
    items += [sanity] if sanity is not None else []

    # A semantic pair the judge settled makes this a judged Verdict (D84), and a pair nobody has
    # settled is named so the report can put it in front of a person rather than bury it.
    compared = _note_comparisons(context, verifier, workdir, run.run_id, notes)
    judge_used = judge_used or compared

    failing_atom, not_verdicted = _decide_outcome(items, context, notes)

    marks = _env_marks(run, flagged_tools)
    is_env_error = _env_error(run)
    if _termination(run) == "environment_cannot_answer":
        # G28: name the tool the Environment could not answer, so the record tells this
        # sibling apart from a body fault (G27), which never ends the Run and carries
        # error class body_fault on a continuing Run instead.
        notes.append(f"environment_cannot_answer:{_cannot_answer_tool(run)}")
    passed = failing_atom is None and not is_env_error and not not_verdicted
    names = [call["name"] for call in context.calls]
    side_effects = _side_effect_count(verifier, context)

    klass, cause, suspected = _classify(
        run, context, cause_result, marks, names, passed, is_env_error, not_verdicted, notes
    )
    score = _score(items, passed, not (is_env_error or not_verdicted), notes)

    notes.extend(marks)
    notes.append(f"side_effects={side_effects}")
    notes.append(f"tool_calls={len(names)}")
    same_path = None if reference_path is None else names == list(reference_path)

    return Verdict(
        run_id=run.run_id,
        env_id=getattr(environment, "env_id", None) or run.env_id,
        # None, not "0": a placeholder string is truthy, so it would walk past regrade's presence
        # check and score a Run against versions nobody ever copied (D97).
        schema_version=getattr(environment, "schema_version", None),
        tools_version=getattr(environment, "tools_version", None),
        policy_version=getattr(environment, "policy_version", None),
        verifier_version=verifier.verifier_version,
        verdict_version=verdict_version,
        runner_version=runner_version,
        **{"pass": passed, "class": klass},
        failing_atom=failing_atom,
        same_path=same_path,
        cause=cause,
        judge_used=judge_used,
        environment_suspected=suspected,
        notes=notes,
        score=score,
        items=[item.result for item in items],
    )
