"""Every Task's objective difficulty, and the buckets a round reports trusted coverage over (D209).

A build reports one trusted count and says nothing about what kind of Task that count holds. Two
builds with the same count can be a set of one-write single-path Tasks and a set of three-write
multi-tool ones, and nothing in the round line or the scorecard tells them apart. This module reads
a difficulty record off what the harness already stored, with no model call and no name of any
customer's world in it:

  write_count            the writes the Task's End state requires: the entity-level write atoms of
                         the live Verifier, one per written entity and not one per field
  tools_touched          how many distinct tools the Reference Run called
  branching_depth        how many rewrites of the Reference's call path D199's rule finds, uncapped:
                         the independent adjacent pairs, the reads that can be made again and the
                         reads nothing later used
  second_paths           how many further ways to the same End state the harness confirmed, which is
                         the Task's Reference count less the Reference itself (D79 check 5, D189)
  question_atoms         the atoms of the question kind: what the Task had to ask before it wrote
  held_out_solve_rate    the share of the held-out pool the Verifier accepts (D133), which is one
                         less the false-rejection fraction, and None where nothing was held out
  judge_agreement        one less the judge disagreement rate over the Task's own judged items
                         (D185), and None where no judge row names the Task

A Task's bucket is the triple of three of them, banded so the bands are fixed and hold on any
corpus: writes 0, 1, 2 or 3 and more; tools 0, 1, 2 or 3 and more; paths 1 or 2 and more. Nothing
here selects, shifts or targets a distribution; the buckets are the measurement a selection rule
would need first, and the second half of the knob is written down in docs/todo.md.

`difficulty.json` in the workdir holds the record and the bucket per Task, rewritten each round, and
carries `no_record`: the Tasks that have no record and why, so a bucket table is never read as
covering Tasks it never saw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.examiner import lifecycle
from kullback.examiner import variants as variants_mod
from kullback.gates import verifier_suite
from kullback.gates.probes import write_tools_of
from kullback.runner.records import Verifier, read_json, read_jsonl, write_json

# The artifact this module writes, and its shape's version. Bumped when a record's fields or a
# bucket's boundaries change, so a file written under an older meaning reads as an older file
# rather than as today's numbers.
FILE_NAME = "difficulty.json"
FORMAT = 1

# The atom target kind that counts as one write. A write atom names a written entity; the value
# atoms under it name fields of the same write, and counting those would call one write with four
# fields harder than three writes with one.
WRITE_KIND = "write"
QUESTION_KIND = "question"

# Where each band stops counting and starts saying "and more". They are counts, so they mean the
# same thing on every corpus; a Task with a Reference always has at least one path.
WRITE_CAP = 3
TOOL_CAP = 3
PATH_CAP = 2

NO_VERIFIER = "no Verifier was derived for this Task"
NO_REFERENCE_RUN = "the Reference Run is not on disk"
# D208: the Verifier this Task had claims an End state derived from a Reference the Task no longer
# holds, so it is not its difficulty either. It says so under its own phrase rather than reading as
# a Task that never had one, because the two are different states to a person reading the table.
RETIRED_VERIFIER = "the Verifier this Task had was retired with the Reference it was derived from"


def _band(count: int, cap: int) -> str:
    """One count as its band: the number itself below the cap, and "<cap>+" at it or above."""
    count = max(0, int(count))
    return str(count) if count < cap else f"{cap}+"


def bucket_key(write_count: int, tools_touched: int, paths: int) -> str:
    """The bucket the three counts fall in, as the one short name every reader spells the same way."""
    return f"w{_band(write_count, WRITE_CAP)}t{_band(tools_touched, TOOL_CAP)}p{_band(paths, PATH_CAP)}"


# --- the counts a record is made of -------------------------------------------------

def write_count(verifier: Verifier) -> int:
    """The writes the Verifier requires: its entity-level write atoms."""
    return sum(1 for atom in verifier.atoms
               if verifier_suite.atom_payload(atom).get("kind") == WRITE_KIND)


def question_count(verifier: Verifier) -> int:
    """The Verifier's atoms of the question kind: what the Task had to ask before it acted."""
    return sum(1 for atom in verifier.atoms if atom.kind == QUESTION_KIND)


def tools_touched(calls: Iterable[dict]) -> int:
    """How many distinct tools the Run called; a call with no name is not a call."""
    return len({str(call.get("name")) for call in calls if call.get("name")})


def branching_depth(calls: list[dict], write_tools: Iterable[str]) -> int:
    """How many rewrites of this call path D199's rule finds, uncapped.

    `variants.variants` caps what it offers the replay, because each rewrite costs a replay; the
    count wanted here is the structure and not the budget, so the three generators are asked
    directly and their answers added. A path of one call has no adjacent pair and no rewrite.
    """
    calls = [call for call in calls if call.get("name")]
    if len(calls) < 2:
        return 0
    return (len(variants_mod.reorderings(calls, write_tools))
            + len(variants_mod.inserted_reads(calls, write_tools))
            + len(variants_mod.dropped_reads(calls, write_tools)))


def solve_rate(fraction: Any) -> Optional[float]:
    """The share of the held-out pool the Verifier accepts, off D133's false-rejection fraction.

    None stays None: a Task that held nothing out has no rate, and reading an empty pool as a
    perfect one is the mistake this returns None to avoid.
    """
    if fraction is None:
        return None
    try:
        return 1.0 - float(fraction)
    except (TypeError, ValueError):
        return None


def judge_agreement(rows: Iterable[dict], task_id: str) -> Optional[float]:
    """One less the disagreement rate over the judge rows that name this Task (D185).

    None where no row names it, which is every build whose judging nobody read back: an unread
    agreement is not a perfect one.
    """
    mine = [row for row in rows or () if str(row.get("item_id") or "") == task_id]
    if not mine:
        return None
    return 1.0 - (sum(1 for row in mine if row.get("disagreement")) / len(mine))


def second_paths(status_row: Any) -> int:
    """How many further ways to the End state the Task's derivation confirmed.

    The Reference count on the status row is the Reference plus every Recording that reached its End
    state, bought (D189) or written from the Reference's own calls (D199), so the second paths are
    that count less one. A row that carries none reads as none, never as one.
    """
    row = status_row if isinstance(status_row, dict) else {}
    try:
        return max(0, int(row.get("references") or 0) - 1)
    except (TypeError, ValueError):
        return 0


def record_for(verifier: Verifier, *, calls: Optional[list[dict]] = None, write_tools: Iterable[str] = (),
               status_row: Any = None, false_rejection: Any = None,
               judge_rows: Iterable[dict] = ()) -> dict:
    """One Task's difficulty record, every field a count or a rate off what is already stored."""
    calls = list(calls or [])
    paths = 1 + second_paths(status_row)
    writes, tools = write_count(verifier), tools_touched(calls)
    return {
        "task_id": verifier.task_id,
        "write_count": writes,
        "tools_touched": tools,
        "branching_depth": branching_depth(calls, write_tools),
        "second_paths": paths - 1,
        "question_atoms": question_count(verifier),
        "held_out_solve_rate": solve_rate(false_rejection),
        "judge_agreement": judge_agreement(judge_rows, verifier.task_id),
        "bucket": bucket_key(writes, tools, paths),
    }


# --- the table over the records -----------------------------------------------------

def _mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def bucket_rows(records: Iterable[dict], trusted_ids: Iterable[str] = ()) -> list[dict]:
    """One row per bucket that holds a Task: how many Tasks, how many trusted, the mean solve rate.

    The mean is over the Tasks of the bucket that have a rate at all; a bucket where nothing was held
    out reports None, which is what a reader has to see before believing a rate of one.
    """
    trusted = set(trusted_ids or ())
    grouped: dict[str, list[dict]] = {}
    for record in records or ():
        grouped.setdefault(str(record.get("bucket") or ""), []).append(record)
    rows = []
    for bucket, group in sorted(grouped.items()):
        rates = [float(r["held_out_solve_rate"]) for r in group if r.get("held_out_solve_rate") is not None]
        rows.append({
            "bucket": bucket,
            "tasks": len(group),
            "trusted": sum(1 for r in group if str(r.get("task_id")) in trusted),
            "solve_rate": _mean(rates),
            "rated": len(rates),
        })
    return rows


def round_summary(rows: Iterable[dict]) -> str:
    """The bucket table as one piece of a round line: trusted over Tasks per bucket that holds any."""
    parts = [f"{row['bucket']} {row['trusted']}/{row['tasks']}" for row in rows or () if row.get("tasks")]
    return " ".join(parts) if parts else "none"


def _rate_cell(row: dict) -> str:
    """A bucket's mean held-out solve rate, or what it says instead of a number it does not have."""
    if row.get("solve_rate") is None:
        return "no held-out pool"
    return f"{float(row['solve_rate']):.0%} over {row.get('rated', 0)} Tasks"


def markdown_table(rows: Iterable[dict], no_record: int = 0) -> list[str]:
    """The bucket table as Markdown, the same lines a report and the CLI read both print.

    The bucket name reads `w<writes>t<tools>p<paths>`, each part a band and not a raw count, so two
    builds on two corpora name the same bucket the same way.
    """
    rows = list(rows or ())
    lines = ["| bucket | Tasks | trusted | mean held-out solve rate |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row.get('bucket', '')} | {row.get('tasks', 0)} | {row.get('trusted', 0)} | "
                     f"{_rate_cell(row)} |")
    if not rows:
        lines.append("| none |  |  |  |")
    if no_record:
        lines += ["", f"{no_record} Tasks carry no difficulty record: they have no Verifier, or the Run "
                      "their Verifier was derived from is not on disk."]
    return lines


# --- reading a workdir --------------------------------------------------------------

def _verifiers(workdir: Path, task_status: Optional[dict] = None) -> tuple[dict[str, Verifier], dict[str, str]]:
    """The live Verifiers of a workdir by Task, and the Tasks whose Verifier is retired (D208).

    What sits under verifiers/ is not the live set. A Verifier derived from a Reference the Task no
    longer holds was retired, and reading one off disk here would put a Task in a bucket on a claim
    it has withdrawn, and count it trusted or not against that claim. The lifecycle accessor every
    other consumer reads through answers which are live, so a workdir an older build left with
    orphaned files reads the same as one the retirement step has been over.
    """
    on_disk: list[Verifier] = []
    for path in sorted((workdir / "verifiers").glob("*.json")):
        try:
            record = Verifier.model_validate(read_json(path, {}) or {})
        except (OSError, ValueError, TypeError):
            continue
        on_disk.append(record)
    live, retired = lifecycle.partition(on_disk, task_status)
    return ({record.task_id: record for record in live},
            {str(row["task_id"]): RETIRED_VERIFIER for row in retired})


def _keep(out: dict[str, str], workdir: Path, row: Any) -> None:
    """One row's Run file, by run_id; a path relative to the workdir is resolved against it."""
    if not isinstance(row, dict) or not row.get("run_id") or not row.get("path"):
        return
    path = Path(str(row["path"]))
    out.setdefault(str(row["run_id"]), str(path if path.is_absolute() else workdir / path))


def run_paths(workdir: Path) -> dict[str, str]:
    """Every Run on disk this workdir knows a path for, by run_id: the replays and the re-rolls.

    The Reference is named by run_id on references.json and its file is named by the replay or
    re-roll row that produced it, so the two are joined here once for every Task. The re-roll rows
    live in two places, beside a Task's Runs and in the Examiner's own file, and both are read: a
    Task whose Reference is a re-roll has its Reference in one of them.
    """
    out: dict[str, str] = {}
    for rows in (read_json(workdir / "replays.json", {}) or {}).values():
        for row in (rows or {}).values():
            _keep(out, workdir, row)
    for path in sorted((workdir / "runs").glob("*/rerolls.json")):
        for row in (read_json(path, {}) or {}).get("runs") or ():
            _keep(out, workdir, row)
    for rows in (read_json(workdir / "examiner" / "rerolls.json", {}) or {}).values():
        for row in rows or ():
            _keep(out, workdir, row)
    return out


def reference_path(verifier: Verifier, references: dict, paths: dict[str, str]) -> Optional[str]:
    """The file of the Run this Verifier was derived from, or None where none of them is on disk.

    The Verifier's own seed run ids are asked first and references.json second. The two agree while a
    Task keeps its Reference, and they part where a later round could not confirm one: the file
    holds the row of the round in hand, the Verifier holds the Runs it was actually derived from, and
    the record is about the Verifier that is live.
    """
    named = [str(run_id) for run_id in verifier.seed_run_ids or ()]
    named += [str(row["run_id"]) for row in (references.get(verifier.task_id) or {}).get("references") or []
              if isinstance(row, dict) and row.get("run_id")]
    for run_id in named:
        path = paths.get(run_id)
        if path and Path(path).is_file():
            return path
    return None


def _last_counts(workdir: Path) -> dict:
    rounds = read_json(workdir / "rounds.json", []) or []
    last = rounds[-1] if rounds else {}
    return dict((last.get("counts") if isinstance(last, dict) else {}) or {})


def _trusted_ruling(workdir: Path) -> dict:
    """The metrics of the last trusted ruling in the ledger, which is where a finished build's
    false-rejection rows and trusted ids are when no round record carries them."""
    rows = read_json(workdir / "gates.json", []) or []
    for row in reversed(rows if isinstance(rows, list) else []):
        if isinstance(row, dict) and row.get("stage") == "trusted":
            return dict(row.get("metrics") or {})
    return {}


def compute(workdir: Any, *, false_rejection: Optional[dict] = None,
            trusted_ids: Optional[Iterable[str]] = None) -> dict:
    """The difficulty record and the bucket of every Task of one workdir, plus the Tasks with none.

    `false_rejection` and `trusted_ids` are the round's own numbers when a driver has them in hand;
    without them they are read back off the last round record, and off the last trusted ruling where
    the rounds carry none, so a finished build reads the same as a running one.
    """
    workdir = Path(workdir)
    counts = _last_counts(workdir)
    ruling = _trusted_ruling(workdir) if (false_rejection is None or trusted_ids is None) else {}
    if false_rejection is None:
        false_rejection = dict(counts.get("false_rejection") or ruling.get("false_rejection") or {})
    if trusted_ids is None:
        trusted_ids = list(counts.get("trusted_ids") or ruling.get("trusted") or [])
    status = read_json(workdir / "task_status.json", {}) or {}
    references = read_json(workdir / "references.json", {}) or {}
    verifiers, retired = _verifiers(workdir, status)
    write_tools = write_tools_of(read_json(workdir / "tool_sigs.json", []) or [])
    judge_rows = read_jsonl(workdir / "judge_pairs.jsonl")
    paths = run_paths(workdir)
    records: list[dict] = []
    no_record: dict[str, str] = {}
    for task_id in sorted(set(status) | set(verifiers) | set(retired)):
        verifier = verifiers.get(task_id)
        if verifier is None:
            # The reason a Task has none is a fixed phrase, never the status row's own words: that
            # row quotes the customer's world back, and this file is read into reports.
            no_record[task_id] = retired.get(task_id, NO_VERIFIER)
            continue
        path = reference_path(verifier, references, paths)
        if path is None:
            no_record[task_id] = NO_REFERENCE_RUN
            continue
        try:
            calls = variants_mod.call_results(verifier_suite.as_run(path))
        except (OSError, ValueError, TypeError):
            no_record[task_id] = NO_REFERENCE_RUN
            continue
        records.append(record_for(verifier, calls=calls, write_tools=write_tools,
                                  status_row=status.get(task_id), false_rejection=false_rejection.get(task_id),
                                  judge_rows=judge_rows))
    return {"format": FORMAT, "tasks": records, "no_record": no_record,
            "buckets": bucket_rows(records, trusted_ids)}


def write_records(workdir: Any, body: dict) -> Path:
    """difficulty.json as this round leaves it."""
    return write_json(Path(workdir) / FILE_NAME, body)


def read_records(workdir: Any) -> dict:
    """difficulty.json as the last round left it, or an empty body where there is none."""
    body = read_json(Path(workdir) / FILE_NAME, None)
    return body if isinstance(body, dict) else {"format": FORMAT, "tasks": [], "no_record": {}, "buckets": []}


def refresh(workdir: Any, *, false_rejection: Optional[dict] = None,
            trusted_ids: Optional[Iterable[str]] = None) -> dict:
    """Compute the records over a workdir and write them; the body is answered back."""
    body = compute(workdir, false_rejection=false_rejection, trusted_ids=trusted_ids)
    write_records(workdir, body)
    return body


__all__ = ["FILE_NAME", "FORMAT", "NO_REFERENCE_RUN", "NO_VERIFIER", "PATH_CAP", "RETIRED_VERIFIER",
           "TOOL_CAP", "WRITE_CAP",
           "branching_depth", "bucket_key", "bucket_rows", "compute", "judge_agreement", "markdown_table",
           "question_count", "read_records", "record_for", "refresh", "reference_path", "round_summary",
           "run_paths", "second_paths", "solve_rate", "tools_touched", "write_count", "write_records"]
