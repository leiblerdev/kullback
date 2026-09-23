"""The Examiner's probe, re-roll and variant runners over the runner tool (D120).

The derivation takes three callables it cannot build itself: the loophole probe (D79 check 6),
the second-path re-rolls (D112, D133) and the synthesised variants (D199). The runner is a tool
of both agents and its inputs are the Builder's (D120), so this module rebuilds those callables
over the runner tool with the workdir as the environment: the Examiner gets scored Runs without
touching the compiled side, which the runner reads on its behalf (D123).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner import budget
from kullback.runner import tool as runner_tool
from kullback.runner.records import UserRules, read_json, run_path
from kullback.runner.target import as_run
from kullback.runner.world.environment import BuiltEnvironment
from kullback.user import ends as user_ends
from kullback.user.rules import goal_write_set
from kullback.user.simulated import SimulatedUser
from kullback.user.value_strip import value_strip


def _seed_ids(anchor: Any, task_id: str, run_ids: Iterable[str]) -> list[str]:
    """The Task's Runs minus its anchor: held-out runs never seed a Reference (D81)."""
    run_ids = list(run_ids)
    if anchor is None:
        return run_ids
    seed_runs = getattr(anchor, "seed_runs", None)
    if callable(seed_runs):
        return list(seed_runs(task_id, run_ids))
    held = set((anchor.get("held_out", {}) or {}).get(task_id, []))
    return [run_id for run_id in run_ids if run_id not in held]


def _replay_rows(workdir: Any, task_id: str) -> dict:
    """The Task's replay rows by run id, tolerant of a missing or torn file."""
    body = read_json(Path(workdir) / "replays.json", {}) or {}
    rows = (body.get(task_id) or {}) if isinstance(body, dict) else {}
    if isinstance(rows, list):
        rows = {str(row.get("run_id") or row.get("trace_id")): row
                for row in rows if isinstance(row, dict)}
    return rows if isinstance(rows, dict) else {}


def _reference_id(workdir: Any, task_id: str, run_ids: Iterable[str], anchor: Any) -> Optional[str]:
    """The confirmed seed replay with user rules on disk, earliest first (D81, D112).

    The seed set is the Task's Runs minus the anchor's held-out Runs; among the confirmed rows
    the Reference is the first one whose rules file exists, the way the Builder's re-roll runner
    picks the rules its Simulated user answers from.
    """
    seeds = set(_seed_ids(anchor, task_id, run_ids))
    rows = _replay_rows(workdir, task_id)
    confirmed = [key for key, row in sorted(rows.items())
                 if key in seeds and isinstance(row, dict)
                 and row.get("confirmed") and row.get("path")]
    if not confirmed:
        return None
    rules_dir = Path(workdir) / "user_rules"
    for key in confirmed:
        trace_id = str(rows[key].get("trace_id") or key)
        if (rules_dir / f"{trace_id}.json").is_file():
            return trace_id
    first = rows[confirmed[0]]
    return str(first.get("trace_id") or confirmed[0])


def _rules_for(workdir: Any, reference_id: Optional[str]) -> Optional[UserRules]:
    """The Reference's user rules, or nothing where the Task has none to answer from."""
    if reference_id is None:
        return None
    path = Path(workdir) / "user_rules" / f"{reference_id}.json"
    if not path.is_file():
        return None
    try:
        return UserRules.model_validate(read_json(path))
    except ValueError:
        return None


def _make_user(workdir: Any, task_id: str, anchor: Any, router: Any) -> Optional[SimulatedUser]:
    """The Simulated user of the Task's confirmed Reference, or nothing without rules.

    Built from the derivation's own inputs, the replay rows, the rules and the recordings, over
    the state reader of the Router the runner built: the Examiner never opens the compiled side
    itself (D123). The vocabulary is the generic core, the way a Run without a mined one reads.
    """
    env = BuiltEnvironment(workdir)
    task = env.task(task_id)
    reference_id = _reference_id(workdir, task_id, task.run_ids, anchor)
    rules = _rules_for(workdir, reference_id)
    if rules is None:
        return None
    writes = env.write_tools()
    traces = env.traces()
    reference = traces.get(reference_id) if reference_id is not None else None
    members = [traces[run_id] for run_id in task.run_ids if run_id in traces]
    return SimulatedUser(
        rules, starting_state_reader=router.state, write_tools=writes,
        goal_writes=goal_write_set(reference, writes) if reference is not None else None,
        answer_strip=value_strip(members) if members else None)


# Every model call of a Run is priced under this stage, the Builder's and the Examiner's alike (F29).
RUNNER_STAGE = "runner"


def _priced(model: Any, workdir: Path) -> Any:
    """The model a Run plays, priced into budget.json under the runner stage, never twice.

    The candidate is tested under the production setting, so its context is never capped (D65).
    """
    if model is None or isinstance(model, budget.BudgetedModel):
        return model
    return budget.BudgetedModel(model, stage=RUNNER_STAGE, workdir=workdir,
                                model_id=getattr(model, "name", None), cap_context=False)


def runners_for(workdir: Any, *, reroll_model: Any = None, anchor: Any = None) -> dict:
    """The derivation's three runners over the runner tool, with the workdir as environment (D120).

    Returns `run_probe(model, verifier)`, `run_rerolls(task_id, count, prefix)` and
    `run_variant(task_id, calls, run_id, transcript=())` with the exact signatures and return
    shapes the derivation consumes, plus `run_variant.write_runs_index` for the caller after its
    gather. Every row's path is resolved against the workdir, so it opens from any cwd; the
    Examiner's re-roll file stores it relative again. Nothing here opens the compiled side; the runner reads it, the Examiner does not
    (D123).
    """
    root = Path(workdir)
    reroll_model = _priced(reroll_model, root)

    def run_probe(model: Any, verifier: Any) -> Any:
        """Check 6's loophole probe as the derivation takes it: `run_probe(model, verifier)` (D79).

        The probe report is the runner's; the derivation scores Runs, so what it gets is the Run
        the report was written from, loaded off the probe file. The stand-in route is refused
        inside the probe (G28: a scored probe is never answered by a model).
        """
        report = runner_tool.probe(
            root, verifier.task_id, _priced(model, root), verifier=verifier, workdir=root,
            make_user=lambda router: _make_user(root, verifier.task_id, anchor, router))
        return as_run(run_path(root, report.path))

    def run_rerolls(task_id: str, count: int, prefix: str) -> list[dict]:
        """Second-path re-rolls as the derivation takes them: rows bought through the runner (D112).

        The seed runs are filtered through the anchor the way the Builder filtered them (D81:
        held-out runs never seed a Reference), and each row carries where its Run landed, how the
        Run ended and how its user ended it (D133). Without a re-roll model there is nothing to
        buy with, which is refused loudly rather than bought silently.
        """
        if reroll_model is None:
            raise ValueError("the Examiner has no model to re-roll with: pass reroll_model")
        reports = runner_tool.reroll(
            root, task_id, reroll_model, count=count, prefix=prefix, workdir=root,
            make_user=lambda router: _make_user(root, task_id, anchor, router))
        rows = []
        for report in reports:
            path = str(run_path(root, report.path))
            try:
                user_end: Optional[str] = user_ends.end_of_run(as_run(path))
            except (OSError, ValueError, TypeError):
                user_end = report.user_end
            rows.append({"run_id": report.run_id, "path": path,
                         "termination_reason": report.termination_reason,
                         "user_end": user_end if user_end is not None else report.user_end})
        return rows

    def run_variant(task_id: str, calls: Any, run_id: str, transcript: Any = ()) -> Optional[dict]:
        """A rewritten call path replayed from the Task's Starting state (D199).

        Costs no model call, since the Trace it drives was written by code. Answers with the Run
        that came out, as the row the derivation keeps; where the Run ended is the caller's to
        read. The runs index over the replayed files is left to the caller, which writes it once
        after its gather through `run_variant.write_runs_index`.
        """
        report = runner_tool.replay_calls(
            root, task_id, list(calls), run_id=run_id,
            transcript=tuple(transcript), workdir=root)
        if not report.path:
            return None
        path = str(run_path(root, report.path))
        try:
            run = as_run(path)
        except (OSError, ValueError, TypeError):
            return None
        return {"run_id": report.run_id, "path": path,
                "termination_reason": run.termination_reason, "crashed": None}

    def write_runs_index() -> Path:
        """The runs index over every replayed Run file, for the caller after its gather."""
        return runner_tool.write_runs_index(root)

    run_variant.write_runs_index = write_runs_index  # type: ignore[attr-defined]
    return {"run_probe": run_probe, "run_rerolls": run_rerolls, "run_variant": run_variant}


__all__ = ["runners_for"]
