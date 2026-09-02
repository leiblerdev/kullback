"""Decide one Run's pass or fail from its End state against a Task's Verifier, in code only (D43, D46, D94)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kullback.runner.atom_context import AtomContext, _evaluate, gate
from kullback.runner.canon import record_use
from kullback.runner.records import Atom, Run, Verdict, Verifier, load_run_jsonl

# AtomContext, gate and _evaluate live in runner/atom_context.py: they are what a Verdict evaluates
# an atom's predicate against, and the confinement gate they share with gates/confinement.py's
# constraint gate (runner/confinement.py) is easier to find next to that world model than buried in
# this module's own top.

VERDICT_VERSION = "1"
MUST_HOLD = {"required", "question", "communicate", "hard"}
TRANSFER_HINTS = ("transfer", "escalate", "handoff", "hand_off")
GAVE_UP = {"transfer", "transferred", "agent_transfer", "gave_up", "no_action"}
ENV_ERROR_REASONS = {"env_error", "environment_error"}
# How each judge use says "this holds", "this does not" and "I did not decide" (judge.py's _USES).
JUDGE_HOLDS = {"pass", "equivalent", "acceptable", "good_reference"}
JUDGE_FAILS = {"fail", "not_equivalent", "unacceptable", "bad_reference"}
CAUSES = {"candidate", "environment", "simulated_user", "undetermined"}


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


def _is_transfer(name: str) -> bool:
    return any(hint in (name or "").lower() for hint in TRANSFER_HINTS)


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


def _evaluate_atoms(verifier: Verifier, context: AtomContext, judge_results: Optional[dict],
                    notes: list[str]) -> tuple[list[Atom], list[Atom], bool]:
    """Every atom of one Verifier: the ones that failed, the ones nobody could check, judge_used."""
    failures: list[Atom] = []
    unevaluable: list[Atom] = []
    judge_used = False
    # Hard atoms run last: without a write-tool set write_calls() is what the other atoms covered,
    # so a hard atom placed first in the Verifier would see an empty list and hold vacuously.
    for atom in sorted(verifier.atoms, key=lambda a: a.kind == "hard"):
        holds: Optional[bool] = None
        refused = gate(atom.predicate_src) if atom.predicate_src and not atom.judge else []
        if atom.judge:
            if judge_results is None or atom.id not in judge_results:
                notes.append(f"judge_atom_unevaluated:{atom.id}")
            else:
                judge_used = True
                holds = _judge_says(judge_results[atom.id])
                if holds is None:
                    notes.append(f"judge_abstained:{atom.id}")
        elif not atom.predicate_src:
            notes.append(f"atom_without_predicate:{atom.id}")
        elif refused:
            notes.append(f"atom_rejected:{atom.id}:{refused[0]}")
        else:
            context.marking = atom.kind != "forbidden"
            try:
                holds = _evaluate(atom.predicate_src, context.env())
            except Exception as error:  # a broken atom is a Verifier defect, not a Candidate failure
                notes.append(f"atom_error:{atom.id}:{type(error).__name__}")
            finally:
                context.marking = True
        if holds is None:
            # An atom that had to hold and could not be checked leaves the Run not verdicted; a
            # counted pass here would hide a Verifier defect or an unrun judge (D76, D79).
            if atom.kind in MUST_HOLD:
                unevaluable.append(atom)
            continue
        if atom.kind in MUST_HOLD and not holds:
            failures.append(atom)
        elif atom.kind == "forbidden" and holds:
            failures.append(atom)
    return failures, unevaluable, judge_used


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
    acting = [call for call in context.write_calls() if not _is_transfer(call["name"])]
    transferred = not acting and (
        any(_is_transfer(name) for name in names) or _termination(run).lower() in GAVE_UP
    )
    klass = "transferred_without_acting" if transferred else "fail"
    cause = _named_cause(cause_result, notes)  # code marks it, the judge names the cause (D88)
    suspected = bool(marks) or cause == "environment"
    if cause is None and not suspected:
        notes.append("cause_pending_judge")
    return klass, cause, suspected


def verdict(run_jsonl: Any, verifier: Verifier, canon: Any = None, judge_results: Optional[dict] = None,
            *, environment: Any = None, runner_version: Optional[str] = None,
            reference_path: Optional[Iterable[str]] = None, write_tools: Optional[Iterable[str]] = None,
            flagged_tools: Iterable[str] = (), schema: Any = None, cause_result: Any = None,
            rules: Any = None, equivalence: Any = None, workdir: Any = None,
            verdict_version: str = VERDICT_VERSION) -> Verdict:
    """Pass or fail one stored Run on its End state; never calls a model, judge atoms arrive as results (D76).

    `cause_result` is judge.py's answer for this Run's failure cause (D88); code marks the Run,
    the judge names the cause, and neither is computed here.
    """
    run = load_run(run_jsonl)
    context = AtomContext(run, canon, write_tools, schema, rules=rules, equivalence=equivalence)
    notes: list[str] = []
    failures, unevaluable, judge_used = _evaluate_atoms(verifier, context, judge_results, notes)

    for comparison in context.comparisons:
        # A semantic pair the judge settled makes this a judged Verdict (D84), and a pair nobody has
        # settled is named so the report can put it in front of a person rather than bury it.
        judge_used = judge_used or bool(getattr(comparison, "judge_used", False))
        if getattr(comparison, "route", None) == "unresolved":
            notes.append(f"semantic_unresolved:{comparison.key}")
        if workdir is not None:
            record_use(workdir, comparison, run.run_id, verifier.task_id)

    order = {atom.id: i for i, atom in enumerate(verifier.atoms)}
    failures.sort(key=lambda a: order.get(a.id, 0))
    unevaluable.sort(key=lambda a: order.get(a.id, 0))

    failing_atom = None
    not_verdicted = False
    if failures:
        first = failures[0]
        failing_atom = first.id
        notes.append(f"failing_atom:{first.id}: {first.description or first.predicate_src or first.kind}")
    elif unevaluable:
        first = unevaluable[0]
        failing_atom, not_verdicted = first.id, True
        notes.append(f"not_verdicted:{first.id}: a {first.kind} atom could not be evaluated")
    elif context.write_tools is not None:
        # A write on a Task whose Verifier asks for none is an extra write by definition, so the
        # check runs on the write-tool set alone and not on whether an atom happened to call wrote().
        extras = context.extra_writes()
        if extras:
            failing_atom = f"extra_write:{extras[0]['name']}"
            notes.append(f"failing_atom:{failing_atom}: write not required and not allowed by any atom")
    else:
        notes.append("side_effect_check_skipped")

    marks = _env_marks(run, flagged_tools)
    is_env_error = _env_error(run)
    passed = failing_atom is None and not is_env_error and not not_verdicted
    names = [call["name"] for call in context.calls]
    side_effects = context.writes_count()

    klass, cause, suspected = _classify(
        run, context, cause_result, marks, names, passed, is_env_error, not_verdicted, notes
    )

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
    )
