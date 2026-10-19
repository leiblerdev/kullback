"""Live A/B of one repair session across two workdirs.

One arm per process: build the live model the way kullback.cli does,
price it under the examiner stage behind a hard spend ceiling, run one
examine session over the listed items, then write before/after metrics.

Usage:
    python scripts/experiments/examiner_ab.py WORKDIR --model MODEL_ID \
        --tasks IDS_FILE --cap USD --metrics OUT_JSON \
        --edit-tools EDIT,PROPOSE --dry-tool DRY_RUN_TOOL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--cap", required=True, type=float)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--edit-tools", default="edit_verifier,propose_verifier")
    parser.add_argument("--dry-tool", default="try_atoms")
    parser.add_argument("--without-dry-run", action="store_true",
                        help="Hide the dry tool from the Examiner at run time")
    return parser.parse_args(argv)


def read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def totals_usd(workdir: Path, stage: str) -> tuple[float, float, int]:
    totals = read_json(workdir / "budget.json", {}) or {}
    stages = totals.get("stages") or {}
    bucket = stages.get(stage) or {}
    total = totals.get("total") or {}
    return (float(total.get("usd") or 0.0),
            float(bucket.get("usd") or 0.0),
            int(bucket.get("calls") or 0))


def suite_passing(workdir: Path, ids: list[str]) -> dict[str, bool]:
    status = read_json(workdir / "task_status.json", {}) or {}
    return {tid: bool((status.get(tid) or {}).get("verifier_passed")) for tid in ids}


def parse_session_calls(workdir: Path, first_line: int, edit_tools: set[str],
                        dry_tool: str) -> dict:
    edits = {"calls": 0, "accepted": 0, "refused": 0, "tasks_accepted": []}
    dry = {"calls": 0, "accepted": 0, "undecided": 0, "refused": 0}
    trusted_on_accept: list[str] = []
    bus = workdir / "bus.jsonl"
    try:
        lines = bus.read_text().splitlines()[first_line:]
    except OSError:
        lines = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("agent") != "examiner":
            continue
        event = record.get("event") or {}
        if event.get("type") != "tool_execution_end":
            continue
        name = event.get("tool_name")
        result = event.get("result") or {}
        details = result.get("details") or {}
        content = result.get("content") or ""
        if name in edit_tools:
            edits["calls"] += 1
            if "retry_ask" not in details and "accepted" in content:
                edits["accepted"] += 1
                task_id = details.get("task_id")
                if task_id and task_id not in edits["tasks_accepted"]:
                    edits["tasks_accepted"].append(task_id)
                rulings = details.get("rulings") or []
                if any(r.get("name") == "trusted" and r.get("accepted") for r in rulings):
                    if task_id and task_id not in trusted_on_accept:
                        trusted_on_accept.append(task_id)
            else:
                edits["refused"] += 1
        elif dry_tool and name == dry_tool:
            dry["calls"] += 1
            verdict = details.get("accepted", result.get("accepted"))
            if verdict is True:
                dry["accepted"] += 1
            elif verdict == "not decided":
                dry["undecided"] += 1
            else:
                dry["refused"] += 1
    edits["tasks_accepted"] = sorted(edits["tasks_accepted"])
    return {"edits": edits, "dry": dry, "trusted_on_accept": sorted(trusted_on_accept)}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    workdir = args.workdir
    ids = [line.strip() for line in args.tasks.read_text().splitlines() if line.strip()]
    edit_tools = {name.strip() for name in args.edit_tools.split(",") if name.strip()}

    from kullback.ai.provider import MemoModel, live_model
    from kullback.runner import budget

    before_total, before_stage, before_calls = totals_usd(workdir, "examiner")
    before_suite = suite_passing(workdir, ids)
    try:
        first_line = len((workdir / "bus.jsonl").read_text().splitlines())
    except OSError:
        first_line = 0

    # Built the way kullback.cli builds it, then priced under the examiner
    # stage. The ledger already holds earlier spend, so the hard ceiling sits
    # one cap above the current total rather than at the cap itself.
    live = live_model(args.model, None)
    ceiling = budget.Ceiling(usd=before_total + args.cap, workdir=workdir)
    ceiling.require_priced(args.model)
    model = budget.BudgetedModel(
        MemoModel(live, workdir), stage="examiner", workdir=workdir,
        model_id=args.model, ceiling=ceiling,
        prompt_cache_key=f"kullback-examab-{workdir.name}",
    )

    from kullback.examiner.session import examine

    arm_saw_dry_tool = True
    if args.without_dry_run:
        import kullback.examiner.domain_tools as domain_tools_mod
        import kullback.examiner.prompt as prompt_mod
        import kullback.examiner.session as session_mod
        base_domain_tools = domain_tools_mod.domain_tools

        def domain_tools_without_dry(root):
            return [tool for tool in base_domain_tools(root)
                    if tool.name != args.dry_tool]

        session_mod.domain_tools = domain_tools_without_dry
        domain_tools_mod.domain_tools = domain_tools_without_dry
        base_tools_section = prompt_mod.tools_section

        def tools_section_without_dry():
            return "\n".join(line for line in base_tools_section().splitlines()
                             if args.dry_tool not in line)

        prompt_mod.tools_section = tools_section_without_dry
        arm_saw_dry_tool = False
        assert args.dry_tool not in prompt_mod.tools_section(), "dry tool still in prompt"

    stopped = "done"
    error = None
    started = time.monotonic()
    try:
        examine(workdir, task_ids=ids, model=model,
                reroll_model=None, probe_model=model)
    except budget.BudgetExceeded as exc:
        stopped = f"ceiling: {exc}"
    except Exception as exc:  # noqa: BLE001 - metrics must land whatever happens
        error = f"{type(exc).__name__}: {exc}"
        stopped = "error"
    wall_s = time.monotonic() - started

    after_total, after_stage, after_calls = totals_usd(workdir, "examiner")
    after_suite = suite_passing(workdir, ids)
    calls = parse_session_calls(workdir, first_line, edit_tools, args.dry_tool)
    metrics = {
        "workdir": str(workdir),
        "items": ids,
        "arm": "without" if args.without_dry_run else "with",
        "arm_saw_dry_tool": arm_saw_dry_tool,
        "stopped": stopped,
        "error": error,
        "spend": {
            "stage_usd": after_stage - before_stage,
            "total_usd": after_total - before_total,
            "stage_calls": after_calls - before_calls,
        },
        "suite": {
            "passing_before": sorted(t for t, p in before_suite.items() if p),
            "passing_after": sorted(t for t, p in after_suite.items() if p),
        },
        **calls,
    }
    args.metrics.write_text(json.dumps(metrics, indent=1))
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
