"""Code mutants of each passing Reference, scored under the Task's Verifier: report only (spec/mutants.py).

Writes exam/mutants/<task>.json in the workdir and prints one line per Task, its id masked to the last
four characters: built, killed, survived, open and wrong-reason kills, then the same per operator. A Task
whose Reference does not pass its Verifier is not scored and says so.

    uv run python scripts/mutants.py .work [--task <id> ...] [--cap 40]
"""

from __future__ import annotations

import argparse

from kullback.spec.mutants import CAP, OPERATORS, workdir_mutants


def _cells(tally: dict) -> str:
    return (f"built {tally['built']} killed {tally['killed']} survived {tally['survived']} "
            f"open {tally['open']} wrong {tally['wrong_reason']}")


def line(task_id: str, report: dict) -> str:
    """One Task's counts, its id masked, never a row id or value."""
    head = f"{task_id[-4:]} "
    if report.get("status") != "scored":
        return head + {"reference_fails": "reference fails, not scored",
                       "reference_not_verdicted": "reference not verdicted, not scored"}.get(
                           report.get("status"), str(report.get("status")))
    parts = [_cells(report), f"dropped {report['dropped']}"]
    parts += [f"{op}: {_cells(report['by_operator'][op])}" for op in OPERATORS if report["by_operator"][op]["built"]]
    return head + " | ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir")
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--cap", type=int, default=CAP)
    args = parser.parse_args(argv)
    reports = workdir_mutants(args.workdir, args.task, cap=args.cap)
    for task_id, report in sorted(reports.items()):
        print(line(task_id, report))
    scored = [r for r in reports.values() if r.get("status") == "scored"]
    total = {key: sum(r[key] for r in scored) for key in ("built", "killed", "survived", "open", "wrong_reason")}
    print(f"tasks {len(reports)} scored {len(scored)} | {_cells(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
