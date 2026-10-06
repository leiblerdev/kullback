"""The refuse ruling (the solvability judge as code, D128) and what a trusted Verifier is (D126, D319).

D319: a Verifier carrying expected end states is trusted by code alone (`code_ruling`), and every
Task gets a tier: trusted, unconfirmed, refused or pending. No model's judgment and no probe pool
is read on that path, and the faithful replay is read only as the Run the end states describe,
never as evidence that the recorded agent was right. What follows to the next marker is the
pre-D319 rule, kept as `_legacy_trusted` this release so its count prints beside the tiers, and
still ruling a Verifier that carries no expected end state.

Pre-D319:

A Task may be refused only when no frontier Run of it finished: no confirmed replay and no re-roll
of any round with a success termination. The rule reads re-rolls already paid for, costs no Run and
never awards a pass. A trusted Verifier passed the D79 suite (`task_status` says `verifier_passed`),
scores no pass on every probe in its pool, is the last accepted version of its history, loosens past
no frontier Run (the loosening gate is run again here over the history cut at that version, so an
accepted flag written without the gate is not trust, D127), its false rejection over the held-out
pool is under D133's threshold, and its Task is not refused; round_end's "Tasks with a trusted
Verifier" is this ruling's count.

D194 made the false-rejection number a step in the chain rather than a number riding beside it. It
was measured here from the start and never read: a Verifier whose required atoms reject every
held-out Run that reached the Reference recognises no path but its own seeds, so what it checks is
one path and not the Task, and calling that trusted overstated the count. A Task with nothing held
out is not evidence either way, so it keeps `no_pool` in `false_rejection_ruling` and is decided by
the other steps. Every ruling carries the fraction (`false_rejection`), the pool size
(`false_rejection_pool`) and that per-Task word, so a reader sees the denominator behind the rate.

A suite failure names every check behind it (D198): the ones that failed and the ones that had no
input with the reason each gate gave, in the suite's fixed order, so a reader can tabulate the Tasks
that stop here instead of reading one sentence that says only that they stopped.

A Task whose suite failed only by checks that had no input is ruled on every other step first, and
held by the not-run checks only when all of them pass; `not_run_only` names those Tasks, so a
proposal on one is not refused for a check nobody could run (F45).

D281 adds provenance, where the caller names the workdir: every seed Run the accepted version
records must resolve, through the Run loader, to a Run of the same Task. A seed read from a file
another Task wrote, a file holding two Runs, or a file nobody can find again is evidence nobody can
attribute, so the version is not trusted, the reason names the seeds, and the history is kept for
the Examiner's next round to derive again. `untrusted_seeds` names each such Task's seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

from kullback.gates.loosening import (
    accepted_versions,
    as_history,
    discarded_runs,
    false_rejection,
    finished_run_ids,
    legitimate_runs,
    loosening_gate,
    over_strict,
)
from kullback.gates.probes import (
    as_pool,
    as_verifier,
    probe_scores,
    version_hash,
    write_tools_of,
)
from kullback.gates.verifier_suite import ALT_PATH_NOT_RUN, D79_STAGES, empty_run, skipped_steps
from kullback.runner.atom_context import AtomContext
from kullback.runner.canon import load_rules
from kullback.runner.expected import contradicts_user, unasked_writes
from kullback.runner.gate_support import _get, gate
from kullback.runner.records import (
    EXAM_DIR,
    EndState,
    EntitySchema,
    GateResult,
    ProbePool,
    Run,
    Verifier,
    VerifierHistory,
    exam_verifier_path,
    load_exam_history,
    load_exam_task_runs,
    load_probe_pools,
    load_task_run,
    read_json,
    run_path,
)
from kullback.runner.target import check_run
from kullback.runner.verdict import _gates as verdict_gates

# A check the suite could not run, and why it had no input. The stage names the status row carries
# are the suite's; the trusted ruling speaks the D79 check names a person reads (D173).
NOT_RUN_REASON = {"second_path_passes": ALT_PATH_NOT_RUN}
# What the false-rejection step says about a Task that held nothing out: no held-out Run reached the
# Reference, so the number is not a zero rate and not a failure, it is an absent measurement (D194).
NO_POOL = "no_pool"
# The Examiner's own re-roll rows (second-path batches), beside the workdir's re-roll file (D133).
EXAMINER_REROLLS = ("examiner", "rerolls.json")
RUNS_DIR = "runs"
#: Why a Task waiting in the Reference pool is untrusted: no recording left standing to derive from.
NEEDS_REFERENCE = "needs a Reference"
#: The pool file under the exam folder, written by examiner.reference and read here; the name is
#: owned here since gates never reach into the examiner.
POOL_FILE = "reference_pool.json"


def pooled_tasks(workdir: Any) -> list[str]:
    """The sorted ids of the Tasks waiting on a Reference, or empty without a workdir."""
    if workdir is None:
        return []
    body = read_json(Path(workdir) / EXAM_DIR / POOL_FILE, {}) or {}
    return sorted(body) if isinstance(body, dict) else []


def finished_runs(task_id: str, replays: dict, rerolls: dict) -> list[str]:
    """The run_ids of the confirmed replays and the finished re-rolls of one Task."""
    return finished_run_ids(task_id, replays, rerolls)


def refuse_gate(refusals: dict[str, dict], replays: dict, rerolls: dict) -> GateResult:
    """A refusal is admitted only when no frontier Run of the Task finished; a failure names the Runs."""
    refused: list[str] = []
    rejected: list[str] = []
    failures: list[str] = []
    for task_id in sorted(refusals or {}):
        finished = finished_runs(task_id, replays, rerolls)
        if finished:
            rejected.append(task_id)
            failures.append(f"task {task_id}: the frontier finished it: {', '.join(finished)}")
        else:
            refused.append(task_id)
    return gate("refuse", failures, refused=refused, rejected=rejected)


def _reason_of(refusal: Any) -> str:
    return str(_get(refusal, "reason", "") or "")


def _suite_reason(row: Any, skipped: list[str]) -> str:
    """Why the suite did not pass: every check that failed and every check that had no input, each
    named, in the fixed order of the suite (D198), and a check nobody could run never read as a check
    the Verifier failed (D173).

    A single-Reference Task fails `second_path_passes` because there is no second path to score, not
    because the Verifier turned one away, and the two ask for different repairs: more Runs of the
    Task (the Examiner's `reroll`, then `derive`) against a looser Verifier. The rule is unchanged
    either way, so a single-path Task is still untrusted; what changes is what the reason says and
    what `checks_not_run` in the metrics lets a reader act on.

    Until D198 a suite failure with nothing skipped said only "the D79 suite did not pass", and on
    one live build 25 Tasks stopped there with no way for a reader to group them by what went wrong.
    A check with no input says why in its own words, off the row the derivation wrote, because the
    gate that skipped it is the only thing that knows; `NOT_RUN_REASON` is the fallback for a row
    written before those words were recorded.
    """
    if _get(row, "derive_again", None):
        # The Reference was re-picked after the suite ran (reference_check), so the reason is that.
        return str(_get(row, "derive_again", ""))
    checks = _get(row, "checks", None) or {}
    given = _get(row, "not_run_reasons", None) or {}
    named = []
    for name in D79_STAGES.values():
        if checks.get(name) or (name not in checks and name not in skipped):
            continue  # a name the row does not mention is no evidence about the Verifier either way
        if name in skipped:
            why = str(given.get(name) or NOT_RUN_REASON.get(name) or "")
            named.append(f"{name} not run" + (f" ({why})" if why else ""))
        else:
            named.append(f"{name} failed")
    if not named:
        return "the D79 suite did not pass"
    return "the D79 suite did not pass: " + ", ".join(named)


def _held_by_not_run(row: Any, skipped: list[str]) -> bool:
    """The suite did not pass only because checks had no input: every check that ran passed (F45).

    The Task stays untrusted, the reason names each check not run and why, and a proposal ruled on
    it is not refused for that: the checks nobody could run say nothing against the atoms proposed.
    """
    checks = _get(row, "checks", None) or {}
    return bool(skipped) and all(ok for name, ok in checks.items() if name not in skipped)


def _is_accepted_version(verifier: Verifier, history: Optional[Any]) -> bool:
    """The current file is the last accepted row of the Task's history, hash for hash."""
    if history is None:
        return False
    accepted = accepted_versions(history)
    return bool(accepted) and accepted[-1].content_hash == version_hash(verifier)


def _loosens_past_the_frontier(task_id: str, history: Any, task_runs: dict, replays: dict, rerolls: dict,
                               canon_rules: Any, sigs: list) -> Optional[str]:
    """The loosening gate's first failure over the Task's history cut at its last accepted row, or None.

    The gate compares the newest row against the accepted one before it, so the history is cut at
    the current version: a rejected attempt after it is not what the file holds.
    """
    hist = as_history(history)
    last = max(i for i, row in enumerate(hist.versions) if row.accepted)
    cut = hist.model_copy(update={"versions": hist.versions[:last + 1]})
    ruling = loosening_gate({task_id: cut}, task_runs, replays, rerolls, canon_rules, sigs)
    return None if ruling.passed else ruling.failures[0]


def _run_rows(index: Any) -> dict[str, list[dict]]:
    """Per Task, the rows of a replay index (a dict per Task) or a re-roll index (a list per Task)."""
    out: dict[str, list[dict]] = {}
    for task_id, rows in (index or {}).items() if isinstance(index, dict) else ():
        values = rows.values() if isinstance(rows, dict) else rows if isinstance(rows, list) else ()
        out[task_id] = [row for row in values if isinstance(row, dict) and row.get("path")]
    return out


def seed_files(workdir: Any, replays: dict, rerolls: dict) -> dict[str, dict[str, Path]]:
    """Per Task, the file each run id names: the replay and re-roll rows, then the Examiner's rows."""
    examiner = read_json(Path(workdir).joinpath(*EXAMINER_REROLLS), {}) or {}
    files: dict[str, dict[str, Path]] = {}
    for index in (replays, rerolls, examiner):
        for task_id, rows in _run_rows(index).items():
            named = files.setdefault(task_id, {})
            for row in rows:
                for key in ("run_id", "trace_id"):
                    if row.get(key):
                        named.setdefault(str(row[key]), run_path(workdir, row["path"]))
    return files


def unattributed_seeds(verifier: Verifier, workdir: Any, files: dict[str, dict[str, Path]]) -> list[str]:
    """The seeds of this version that do not load, through the Run loader, as Runs of its Task (D281).

    A seed no row names is looked for under the Task's own runs folder, where the runner writes a
    replay or a variant; one found nowhere is not attributable either.
    """
    task_id = verifier.task_id
    named = files.get(task_id) or {}
    bad = []
    for seed in verifier.seed_run_ids:
        path = named.get(seed) or Path(workdir) / RUNS_DIR / task_id / f"{seed}.jsonl"
        try:
            load_task_run(path, task_id)
        except (OSError, ValueError):
            bad.append(seed)
    return bad


def _passing_probes(verifier: Verifier, pool: Any, canon_rules: Any, write_tools: Any,
                    runs: Optional[list[Run]] = None) -> list[str]:
    """The probes of the Task's pool the Verifier scores a pass that skipped a step, none without a pool.

    The same reading as check 6: a passing probe that made every call the seeds all made before their
    first demanded outcome solved the Task, and it is no reason to withhold trust. The seeds are read
    off the Task's Runs by the Verifier's seed ids; with none of them to hand, a passing probe
    withholds trust as it always did.
    """
    if pool is None:
        return []
    pool = as_pool(pool)
    scores = probe_scores(verifier, pool, canon_rules, write_tools)
    wanted = set(verifier.seed_run_ids)
    seeds = [run for run in runs or [] if run.run_id in wanted]
    return [probe.probe_id for probe in pool.probes if scores[probe.probe_id]
            and (not seeds or skipped_steps(verifier, probe.run, seeds, canon_rules))]


def _pool_ruling(held: dict) -> str:
    """The false-rejection number in words, or NO_POOL when no held-out Run was scored."""
    return NO_POOL if not held["held_out"] else f"{held['fraction']:.2f} of {held['held_out']} held-out Runs"


def _skipped_stages(row: Any) -> list[str]:
    """The checks the Task's status row says did not run, by their D79 names."""
    return [D79_STAGES.get(stage, stage) for stage in (_get(row, "not_run", None) or [])]


def _unattributed_seeds(verifier: Verifier, workdir: Any, files: dict) -> list[str]:
    """The seeds that are not Runs of the Task, none without a workdir to check them in."""
    return unattributed_seeds(verifier, workdir, files) if workdir is not None else []


def _legacy_trusted(task_status: dict, verifiers: list[Verifier], probes: dict[str, ProbePool],
                    history: dict[str, VerifierHistory], refusals: dict[str, dict], task_runs: dict[str, list[Run]],
                    replays: dict, rerolls: dict, canon_rules: Any, sigs: list, *,
                    workdir: Any = None) -> GateResult:
    """The trusted rule before D319, kept for this release only so the drop shows beside the new count.

    It reads the probe pool, the suite row (which counts a model's loophole probe) and the history,
    and nothing here decides trust any more: `trusted_gate` takes its count as `legacy_trusted` and
    its false-rejection numbers, which it measures and never gates on. Given the workdir, the seeds of each accepted version are checked for provenance (D281), and a
    Task waiting in the Reference pool is untrusted for want of a Reference.
    """
    write_tools = write_tools_of(sigs)
    files = seed_files(workdir, replays, rerolls) if workdir is not None else {}
    foreign_seeds: dict[str, list[str]] = {}
    # D133's number is over the Runs that did the job, so the discarded recordings are out of the
    # pool (D173); the loosening rule below reads the whole pool, which is a different question.
    legitimate = legitimate_runs(replays, rerolls, discarded_runs(task_status))
    admitted = refuse_gate(refusals, replays, rerolls).metrics["refused"]
    refused = {task_id: _reason_of((refusals or {})[task_id]) for task_id in admitted}
    pooled = {task_id: NEEDS_REFERENCE for task_id in pooled_tasks(workdir)}
    trusted: list[str] = []
    untrusted: dict[str, str] = {}
    fractions: dict[str, Optional[float]] = {}
    pool_sizes: dict[str, int] = {}
    pool_says: dict[str, str] = {}
    not_run: dict[str, list[str]] = {}
    not_run_only: list[str] = []
    probes_passing = 0
    failures: list[str] = []
    for verifier in sorted(map(as_verifier, verifiers or ()), key=lambda v: v.task_id):
        task_id = verifier.task_id
        passing = _passing_probes(verifier, (probes or {}).get(task_id), canon_rules, write_tools,
                                  (task_runs or {}).get(task_id))
        probes_passing += len(passing)
        held = false_rejection(verifier, (task_runs or {}).get(task_id, []), legitimate.get(task_id, set()),
                               canon_rules, write_tools)
        fractions[task_id] = held["fraction"]
        pool_sizes[task_id] = held["held_out"]
        pool_says[task_id] = _pool_ruling(held)
        row = (task_status or {}).get(task_id) or {}
        skipped = _skipped_stages(row)
        if skipped:
            not_run[task_id] = skipped
        suite_passed = bool(_get(row, "verifier_passed", False))
        only_not_run = not suite_passed and _held_by_not_run(row, skipped)
        if task_id in pooled:
            reason = pooled[task_id]
        elif not suite_passed and not only_not_run:
            reason = _suite_reason(row, skipped)
        elif passing:
            reason = f"probe {passing[0]} scores a pass"
        elif not _is_accepted_version(verifier, (history or {}).get(task_id)):
            reason = f"version {version_hash(verifier)} is not an accepted version"
        elif unattributed := _unattributed_seeds(verifier, workdir, files):
            foreign_seeds[task_id] = unattributed
            reason = (f"version {version_hash(verifier)} was derived from seeds that are not Runs of this Task: "
                      f"{', '.join(unattributed)}")
        elif (loosened := _loosens_past_the_frontier(task_id, history[task_id], task_runs or {}, replays or {},
                                                     rerolls or {}, canon_rules, sigs)) is not None:
            reason = f"version {version_hash(verifier)} loosens past the frontier: {loosened}"
        elif task_id in refused:
            reason = "the Task is refused"
        elif over_strict(held):
            reason = (f"false_rejection {pool_says[task_id]}: the required atoms reject every held-out Run that "
                      "reached the Reference, so the Verifier checks one path and not the Task")
        elif only_not_run:
            reason = _suite_reason(row, skipped)
            not_run_only.append(task_id)
        else:
            trusted.append(task_id)
            continue
        untrusted[task_id] = reason
        failures.append(f"task {task_id}: {reason}")
    return gate("trusted", failures, trusted=trusted, untrusted=untrusted, probes_passing=probes_passing,
                false_rejection=fractions, false_rejection_pool=pool_sizes, false_rejection_ruling=pool_says,
                refused=refused, checks_not_run=not_run, not_run_only=not_run_only,
                untrusted_seeds=foreign_seeds)


#: The tiers a Task's row carries (D319). Trusted is decided by code alone; unconfirmed is a value
#: nobody sourced, a pair nobody settled or a gate that answered unknown; refused is D128; pending is
#: a defect ruled back to the Spec, a check that did not pass or no Reference to read yet.
TIERS = ("trusted", "unconfirmed", "refused", "pending")


def unsupported_cells(state: EndState) -> list[str]:
    """Each cell of an end state whose value or row carries no source, as table.row or table.row.field."""
    out = []
    for cell in state.cells:
        if cell.row_source is None or (cell.field is not None and cell.source is None):
            out.append(f"{cell.table}.{cell.row_id}" + (f".{cell.field}" if cell.field else ""))
    return out


def faithful_references(verifier: Verifier, task_runs: dict, replays: dict) -> list[Run]:
    """The seed Runs of this version that are confirmed (faithful) replays of a kept recording.

    The replay says the Environment reproduced the recording; it says nothing about whether the
    recorded agent was right, and it is read here only as the Run the expected end states describe.
    """
    rows = (replays or {}).get(verifier.task_id) or {}
    confirmed = {str(_get(row, "run_id", "")) for row in (rows.values() if isinstance(rows, dict) else rows)
                 if _get(row, "confirmed", False)}
    wanted = set(verifier.seed_run_ids) & confirmed
    return [run for run in (task_runs or {}).get(verifier.task_id) or [] if run.run_id in wanted]


def _context(run: Run, canon_rules: Any, write_tools: Any, schema: Any = None) -> AtomContext:
    """A semantic column (by the schema) with no equivalence table to settle it comes back unsettled."""
    return AtomContext(run, canon_rules, write_tools or None, schema, rules=canon_rules)


def _unknown(name: str) -> str:
    """A gate that could not settle in words: the column kind for an end state pair, else the gate."""
    _, found, column = name.partition("unsettled:")
    return f"unsettled {column}" if found else f"unknown {name}"


def _empty_run_passes(verifier: Verifier, reference: Run, canon_rules: Any, write_tools: Any) -> Optional[str]:
    """Why the empty Run (no write, none of the required conduct) is not failed, or None when it fails.

    It fails when an atom fails it or a gate does (D295: a Task whose expected diff is empty still
    demands its conduct, so a Verifier only an empty end state speaks for is not evidence).
    """
    empty = empty_run(reference)
    passed, _ = check_run(verifier, empty, canon_rules, write_tools=write_tools)
    failed, unsettled = verdict_gates(verifier, _context(empty, canon_rules, write_tools))
    if failed or not passed:
        return None
    return "the empty Run passes" + (f" ({_unknown(unsettled)})" if unsettled else "")


class CodeRuling(NamedTuple):
    tier: str
    reason: str
    unsupported: int
    contradicts: list[str] = []
    unasked: list[str] = []


def code_ruling(verifier: Verifier, references: list[Run], row: Any, canon_rules: Any,
                write_tools: Any, schema: Any = None) -> CodeRuling:
    """(tier, reason, unsupported cells, consistency flags) by code alone: no model, no probe pool (D319).

    In order: an end state fully sourced; every faithful replay of the Reference matching one
    (False is a Verifier defect ruled back to the Spec, None leaves the Task unconfirmed, and the
    forbidden list and conduct answer too, so an unknown there blocks the same way); the empty Run
    failing; the wrong Run failing (D286) and the leak check clean (D287), off the suite's row.
    The count is the fewest unsupported cells of any one end state. The cells that contradict the
    user's named values and the Reference's writes nobody asked for (D321) ride on the ruling as
    flags and never decide the tier (D322).
    """
    if not verifier.expected:
        return CodeRuling("unconfirmed", "no expected end state", 0)
    missing = [unsupported_cells(state) for state in verifier.expected]
    count = min(len(cells) for cells in missing)
    sourced = [state for state, cells in zip(verifier.expected, missing, strict=True) if not cells]
    if not sourced:
        fewest = min(missing, key=len)
        more = f" and {count - 1} more" if count > 1 else ""
        return CodeRuling("unconfirmed", f"unsupported cell {fewest[0]}{more}", count)
    if not references:
        return CodeRuling("pending", "no faithful replay of a kept Reference", count)
    # Sourced says a value was available; these two say whether it is consistent with what the
    # user asked, reported as flags and never gating (D322).
    request = _context(references[0], canon_rules, write_tools, schema)
    turns = [text for _, text in request.user]
    contradicts = sorted({cell for state in sourced
                          for cell in contradicts_user(state, turns, request.start_state, canon_rules)})
    unasked = [write.split("@")[0] for write in unasked_writes(references[0], turns, write_tools)]

    def ruled(tier: str, reason: str = "") -> CodeRuling:
        return CodeRuling(tier, reason, count, contradicts, unasked)

    gated = verifier.model_copy(update={"expected": sourced})
    for run in references:
        failed, unsettled = verdict_gates(gated, _context(run, canon_rules, write_tools, schema))
        if failed:
            return ruled("pending", f"Verifier defect, ruled back to the Spec: faithful replay {run.run_id} fails {failed}")
        if unsettled:
            return ruled("unconfirmed", f"{_unknown(unsettled)} on faithful replay {run.run_id}")
    if why := _empty_run_passes(verifier, references[0], canon_rules, write_tools):
        return ruled("pending", why)
    checks = _get(row, "checks", None) or {}
    for name, words in (("plausible_wrong_fails", "the wrong Run is not failed"),
                        ("leak_check_clean", "the leak check is not clean")):
        if not checks.get(name):
            return ruled("pending", f"{words} ({name})")
    return ruled("trusted")


def trusted_gate(task_status: dict, verifiers: list[Verifier], probes: dict[str, ProbePool],
                 history: dict[str, VerifierHistory], refusals: dict[str, dict], task_runs: dict[str, list[Run]],
                 replays: dict, rerolls: dict, canon_rules: Any, sigs: list, *,
                 workdir: Any = None, schema: Any = None, spec_tiers: Optional[dict] = None) -> GateResult:
    """A failure per Task with a Verifier that is not trusted, its tier and the first reason (D319).

    `spec_tiers` (task id -> (tier, row), `spec.trust.workdir_tiers`) are the one ruling where the
    workdir has Specs (D322): they replace trusted, untrusted and the tier, and the rule below stays
    for the legacy column only.

    A refused Task is refused; a Task waiting in the Reference pool, or whose seeds are not Runs of
    it (D281), is pending; a Verifier carrying expected end states is ruled by `code_ruling`. One
    with none (written before D315) keeps the pre-D319 ruling and its words this release, and its
    Task is named in `legacy_only`. The pre-D319 rule runs
    beside it as `legacy_trusted` (its probe and false-rejection numbers are reported, never read
    here); the held-out false rejection (D133) is "valid other solutions failing" in the table.
    """
    legacy = _legacy_trusted(task_status, verifiers, probes, history, refusals, task_runs, replays, rerolls,
                             canon_rules, sigs, workdir=workdir)
    write_tools = write_tools_of(sigs)
    files = seed_files(workdir, replays, rerolls) if workdir is not None else {}
    refused = dict(legacy.metrics["refused"])
    pooled = set(pooled_tasks(workdir))
    trusted: list[str] = []
    untrusted: dict[str, str] = {}
    tiers: dict[str, str] = {}
    unsupported: dict[str, int] = {}
    foreign_seeds: dict[str, list[str]] = {}
    flags: dict[str, int] = {}
    failures: list[str] = []
    for verifier in sorted(map(as_verifier, verifiers or ()), key=lambda v: v.task_id):
        task_id = verifier.task_id
        count = 0
        if not verifier.expected:
            # Until the Spec writes end states for every Verifier, an atom-only one keeps the old
            # ruling and its words, counted in `legacy_only` so its count is never read as code's.
            old = legacy.metrics["untrusted"].get(task_id)
            tiers[task_id] = "trusted" if old is None else "refused" if task_id in refused else "pending"
            unsupported[task_id] = 0
            if old is None:
                trusted.append(task_id)
            else:
                untrusted[task_id] = old
                failures.append(f"task {task_id}: {old}")
                if task_id in legacy.metrics["untrusted_seeds"]:
                    foreign_seeds[task_id] = legacy.metrics["untrusted_seeds"][task_id]
            continue
        if task_id in refused:
            tier, reason = "refused", "the Task is refused"
        elif task_id in pooled:
            tier, reason = "pending", NEEDS_REFERENCE
        elif unattributed := _unattributed_seeds(verifier, workdir, files):
            foreign_seeds[task_id] = unattributed
            tier, reason = "pending", (f"version {version_hash(verifier)} was derived from seeds that are not Runs "
                                       f"of this Task: {', '.join(unattributed)}")
        else:
            ruling = code_ruling(verifier, faithful_references(verifier, task_runs, replays),
                                 (task_status or {}).get(task_id) or {}, canon_rules, write_tools, schema)
            tier, reason, count = ruling.tier, ruling.reason, ruling.unsupported
            flags[task_id] = len(ruling.contradicts) + len(ruling.unasked)
        tiers[task_id] = tier
        unsupported[task_id] = count
        if tier == "trusted":
            trusted.append(task_id)
            continue
        untrusted[task_id] = f"{tier}: {reason}"
        failures.append(f"task {task_id}: {untrusted[task_id]}")
    metrics = dict(legacy.metrics, trusted=trusted, untrusted=untrusted, untrusted_seeds=foreign_seeds,
                   trust_tier=tiers, unsupported_cells=unsupported, legacy_trusted=list(legacy.metrics["trusted"]),
                   legacy_only=sorted(v.task_id for v in map(as_verifier, verifiers or ()) if not v.expected),
                   consistency_flags=flags)
    if spec_tiers:
        metrics.update(_spec_ruling(spec_tiers, refused))
        failures = [f"task {task_id}: {why}" for task_id, why in sorted(metrics["untrusted"].items())]
    return gate("trusted", failures, **metrics)


def _spec_ruling(spec_tiers: dict[str, Any], refused: dict) -> dict:
    """The Spec's tiers as the one trusted ruling (D322): trusted stays trusted, every other tier is
    untrusted with its word and the first failing gate or reason; refused Tasks stay refused."""
    trusted, untrusted, tiers, flags = [], {}, {}, {}
    for task_id, (tier, row) in sorted(spec_tiers.items()):
        tiers[task_id] = tier
        flags[task_id] = len(row.get("contradicts") or ()) + len(row.get("unasked") or ())
        if tier == "trusted" and task_id not in refused:
            trusted.append(task_id)
        else:
            untrusted[task_id] = f"{tier}: {row.get('failing') or row.get('reason') or ''}".rstrip(": ")
    return {"trusted": trusted, "untrusted": untrusted, "trust_tier": tiers, "consistency_flags": flags}


def tier_counts(metrics: dict) -> dict[str, int]:
    """The tiers side by side with the legacy count and the held-out false rejection, as numbers.

    "Valid other solutions failing" is D133's number over every Task's pool, unchanged: the held-out
    Runs that reached the Reference and that the required atoms reject, of all held out.
    """
    tiers = list((metrics.get("trust_tier") or {}).values())
    fractions = metrics.get("false_rejection") or {}
    pools = metrics.get("false_rejection_pool") or {}
    return {**{tier: tiers.count(tier) for tier in TIERS},
            "legacy_trusted": len(metrics.get("legacy_trusted") or ()),
            "valid_other_failing": sum(round((fractions.get(t) or 0.0) * int(n or 0)) for t, n in pools.items()),
            "held_out": sum(int(n or 0) for n in pools.values()),
            "consistency_flags": sum(1 for n in (metrics.get("consistency_flags") or {}).values() if n)}


def trust_row(counts: dict) -> str:
    """One line: trusted, unconfirmed, refused and pending, the legacy count, the over-strictness and the
    Tasks carrying a consistency flag (D322: counted, never gating)."""
    return (f"trusted {counts.get('trusted', 0)} | unconfirmed {counts.get('unconfirmed', 0)} | "
            f"refused {counts.get('refused', 0)} | pending {counts.get('pending', 0)} | "
            f"legacy rule trusted {counts.get('legacy_trusted', 0)} | valid other solutions failing "
            f"{counts.get('valid_other_failing', 0)} of {counts.get('held_out', 0)} held-out Runs | "
            f"consistency flags {counts.get('consistency_flags', 0)}")


def _live_verifiers(root: Path) -> list[dict]:
    """Each Task's live Verifier: the Examiner's proposal where it wrote one, else the derived file."""
    verifiers: list[dict] = []
    for path in sorted((root / "verifiers").glob("*.json")) if (root / "verifiers").is_dir() else []:
        proposal = exam_verifier_path(root, path.stem)
        for source in (proposal, path) if proposal.is_file() else (path,):
            try:
                verifiers.append(json.loads(source.read_text(encoding="utf-8")))
                break
            except (OSError, ValueError):
                continue
    return verifiers


def _refusal_files(root: Path) -> dict[str, Any]:
    refusals: dict[str, Any] = {}
    for folder in (root / "refusals", root / "env" / "refusals"):
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            try:
                refusals.setdefault(path.stem, json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return refusals


def workdir_trusted_ruling(workdir: Any, spec_tiers: Optional[dict] = None) -> GateResult:
    """The trusted ruling over a workdir's live files, with provenance: the one trusted number (D281).

    The Builder's status and the round's snapshot both read trust here, so they cannot disagree: the
    Examiner's artefacts where it wrote them (F22), the refusals, the replay and re-roll rows, and
    the workdir for the seed provenance step. Files not there yet read as empty. Where the workdir
    has Specs the caller passes their tiers (`spec.trust.workdir_ruling`, D322), since the gates sit
    below the Spec and cannot read it themselves.
    """
    root = Path(workdir)
    task_status = read_json(root / "task_status.json", None) or {}
    return trusted_gate(task_status if isinstance(task_status, dict) else {}, _live_verifiers(root),
                        load_probe_pools(root), load_exam_history(root), _refusal_files(root),
                        load_exam_task_runs(root), read_json(root / "replays.json", None) or {},
                        read_json(root / "rerolls.json", None) or {}, load_rules(root / "canon-rules.json"),
                        read_json(root / "tool_sigs.json", None) or [], workdir=root, schema=schema_of(root),
                        spec_tiers=spec_tiers)


def schema_of(root: Path) -> Optional[EntitySchema]:
    """The workdir's column classes, so a semantic pair reads as unsettled rather than as different."""
    try:
        return EntitySchema.model_validate(read_json(root / "schema.json", None) or {})
    except ValueError:
        return None
