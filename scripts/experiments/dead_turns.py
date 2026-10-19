"""Replay the turns where the rule-driven user died, through the agent user, and count what it did.

    uv run python scripts/experiments/dead_turns.py --workdir <clone>:40 --workdir <clone>:20 \
        --model <provider/model> --ceiling-usd 3 --seed 7 --out results.json

A dead turn is a user turn the rule-driven user answered with its goal again (source `goal`
tagged goal_restated) right after a Candidate turn that asked a question, in a stored re-roll
(runs/second-path-*.jsonl) whose successful writes differ from its Reference's: it wrote nothing,
or something else. For each sampled dead turn the Task's agent user is rebuilt exactly as a
switch-on re-roll builds it (`builder.run_user._make_user` with a user model, over the Task's own
Router), handed the stored transcript up to that Candidate turn with its earlier turns preset, and
asked for the next turn once. No Candidate runs.

The workdir is written to: the user's bus and the priced calls land there under the `user` stage.
Point it at a clone (`cp -c -R`), never at the build's own workdir. The output carries counts, ids
and tags only, never a turn's text, so it can be shared without customer strings.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from kullback.ai.provider import live_model
from kullback.builder import run_user as R
from kullback.examiner.session import _load_anchor
from kullback.runner import budget
from kullback.runner import tool as runner_tool
from kullback.runner.records import canonical_json
from kullback.runner.world.environment import BuiltEnvironment
from kullback.user import account as account_mod
from kullback.user import rules as rules_mod

RUN_GLOB = "second-path-*.jsonl"
ANSWERED, ANSWERED_FLOOR, RESTATED, ASKED_END = "answered", "answered_by_floor", "restated", "asked_end"
# How close a turn's words are to the goal's before it reads as the goal said again.
RESTATE_OVERLAP = 0.5


def _events(path: Path) -> tuple[dict, list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    head = rows[0] if rows and "type" not in rows[0] else {}
    tail = rows[-1] if rows and "type" not in rows[-1] else {}
    return head, [row for row in rows if row.get("type")], tail


def _run_writes(events: list[dict], writes: set[str]) -> set[tuple[str, str]]:
    """The successful writes of a stored Run, as (tool, canonical arguments)."""
    args = {e["payload"].get("id"): e["payload"].get("args") or {}
            for e in events if e["type"] == "tool_call"}
    out = set()
    for event in events:
        payload = event["payload"]
        if event["type"] == "tool_result" and payload.get("name") in writes and not payload.get("error"):
            out.add((payload["name"], canonical_json(args.get(payload.get("id"), {}))))
    return out


def _reference_writes(trace: Any, writes: set[str]) -> set[tuple[str, str]]:
    return {(call.name, canonical_json(call.args)) for call in (trace.tool_calls if trace else ())
            if call.name in writes and call.error is None}


def _said(event: dict) -> str:
    reply = event["payload"].get("reply") or {}
    return (reply.get("content") or "").strip()


def _transcript(events: list[dict]) -> list[dict]:
    """The Runner's messages rebuilt from its events: spoken turns, calls and their results."""
    out: list[dict] = []
    for event in events:
        payload = event["payload"]
        if event["type"] == "user_turn":
            out.append({"role": "user", "content": payload.get("text") or ""})
        elif event["type"] == "model_call":
            reply = payload.get("reply") or {}
            out.append({"role": "assistant", "content": reply.get("content"),
                        "tool_calls": reply.get("tool_calls") or []})
        elif event["type"] == "tool_result":
            message = {"role": "tool", "tool_call_id": payload.get("id"), "name": payload.get("name"),
                       "content": json.dumps(payload.get("result"), default=str)}
            for key in ("error", "write_effect"):
                if payload.get(key) is not None:
                    message[key] = payload[key]
            out.append(message)
    return out


def dead_turns(root: Path) -> dict[str, list[dict]]:
    """Every dead turn of a workdir's failed re-rolls, by Task: the Run file and the event index."""
    env = BuiltEnvironment(root)
    anchor = _load_anchor(root)
    writes = set(env.write_tools())
    traces = env.traces()
    references: dict[str, set] = {}
    out: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((root / "runs").glob(RUN_GLOB)):
        head, events, _ = _events(path)
        task_id = head.get("task_id")
        if not task_id:
            continue
        if task_id not in references:
            task = env.task(task_id)
            ref = R._reference_id(root, task_id, task.run_ids, anchor)
            references[task_id] = _reference_writes(traces.get(ref) if ref else None, writes)
        if _run_writes(events, writes) == references[task_id]:
            continue
        asked = ""
        for index, event in enumerate(events):
            if event["type"] == "model_call" and _said(event):
                asked = _said(event)
            elif event["type"] == "user_turn":
                sources = event["payload"].get("sources") or {}
                if sources.get(rules_mod.GOAL) == rules_mod.GOAL_RESTATED and "?" in asked:
                    out[task_id].append({"run": path.name, "event": index})
                asked = ""
    return out


def sample(found: dict[str, list[dict]], count: int, seed: int, per_task: int = 2) -> list[dict]:
    """`count` dead turns spread over as many Tasks as possible, at most `per_task` each, seeded."""
    rng = random.Random(seed)
    tasks = sorted(found)
    rng.shuffle(tasks)
    pools = {task: rng.sample(found[task], len(found[task])) for task in tasks}
    picked: list[dict] = []
    for round_index in range(per_task):
        for task in tasks:
            if len(picked) >= count:
                return picked
            if round_index < len(pools[task]):
                picked.append({"task_id": task, **pools[task][round_index]})
    return picked


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _restates(text: str, goal: str) -> bool:
    said, wanted = _words(text), _words(goal)
    return bool(said and wanted) and len(said & wanted) / len(said | wanted) >= RESTATE_OVERLAP


def replay_one(root: Path, item: dict, model: Any, corpus: Any, anchor: Any) -> dict:
    """One dead turn through the agent user; what it did, in counts and tags."""
    _, events, _ = _events(root / "runs" / item["run"])
    prior = events[:item["event"]]
    env = BuiltEnvironment(root)
    router = runner_tool._router_for(env, item["task_id"], 0)
    user = R._make_user(root, item["task_id"], anchor, router, model, corpus)
    row: dict = {"task_id": item["task_id"], "run": item["run"], "event": item["event"]}
    if user is None:
        return {**row, "outcome": "no_user"}
    earlier = [e["payload"].get("text") or "" for e in prior if e["type"] == "user_turn"]
    user._turn = len(earlier)
    user.box.said = list(earlier)
    tools: list[str] = []
    stops: list[str] = []
    build = user.harness

    def watch(event: Any) -> None:
        kind = getattr(event, "type", "")
        if kind == "tool_execution_start":
            tools.append(event.tool_name)
        elif kind == "message_end" and getattr(event.message, "role", "") == "assistant":
            empty = not (getattr(event.message, "content", "") or "").strip()
            stops.append(f"{getattr(event.message, 'stop_reason', None)}{':empty' if empty else ''}")

    def watched(transcript=()):
        harness = build(transcript)
        harness.subscribe(watch)
        return harness

    user.harness = watched
    before = budget.load_totals(root)["stages"].get(R.USER_STAGE, budget.empty_bucket())
    text = user.reply(_transcript(prior))
    after = budget.load_totals(root)["stages"].get(R.USER_STAGE, budget.empty_bucket())
    payload = user.events[-1].payload if user.events else {}
    driver = payload.get("driver")
    sources = payload.get("sources") or {}
    goal = user.ctx.goal
    if user.box.requested is not None:
        outcome = ASKED_END
    elif sources.get(rules_mod.GOAL) == rules_mod.GOAL_RESTATED or _restates(text, goal):
        outcome = RESTATED
    else:
        outcome = ANSWERED if driver == "agent" else ANSWERED_FLOOR
    tags = sorted({v for v in sources.values()
                   if v in (*account_mod.CHOICE_SOURCES, account_mod.ACCOUNT)})
    if driver == "agent" and (payload.get("facts") or user.guards.facts_in(text)):
        tags.append("facts")
    return {**row, "outcome": outcome, "driver": driver, "tools": tools, "tags": tags, "stops": stops,
            "dropped": payload.get("agent_turn_dropped"), "requested_end": user.box.requested,
            "model_failed": user.counts.get("agent_user_model_failed", 0),
            **{key: after[key] - before[key] for key in ("calls", "input", "output", "cache_read",
                                                         "cache_write", "usd")}}


def summary(rows: list[dict]) -> dict:
    by: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by[row["corpus"]][row["outcome"]] += 1
        by[row["corpus"]]["turns"] += 1
        by[row["corpus"]]["usd"] += row.get("usd", 0.0)
        for tag in row.get("tags") or ():
            by[row["corpus"]]["tag:" + tag] += 1
        for name in row.get("tools") or ():
            by[row["corpus"]]["tool:" + name] += 1
        if row.get("dropped"):
            by[row["corpus"]]["drop:" + row["dropped"]] += 1
    out = {}
    for corpus, counts in by.items():
        counts = dict(counts)
        counts["answer_share"] = round(counts.get(ANSWERED, 0) / max(counts["turns"], 1), 3)
        out[corpus] = counts
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", action="append", required=True,
                        help="A clone of a build workdir and how many dead turns to take, as PATH:N.")
    parser.add_argument("--model", required=True, help="The user model, as provider/model.")
    parser.add_argument("--ceiling-usd", type=float, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true", help="Count and sample; call no model.")
    args = parser.parse_args(argv)
    spent = 0.0
    rows: list[dict] = []
    plan: dict = {}
    for spec in args.workdir:
        path, _, count = spec.rpartition(":")
        root = Path(path)
        found = dead_turns(root)
        picked = sample(found, int(count), args.seed, args.per_task)
        plan[root.name] = {"dead_turns": sum(map(len, found.values())), "tasks_with_dead_turns": len(found),
                           "sampled": len(picked), "sampled_tasks": len({p["task_id"] for p in picked}),
                           "sample": picked}
        if args.sample_only:
            continue
        # The ledger a ceiling charges is the workdir's whole build, so this run's allowance sits on
        # top of what the clone already records.
        already = float(budget.load_totals(root)["total"]["usd"] or 0.0)
        ceiling = budget.Ceiling(usd=already + max(args.ceiling_usd - spent, 0.0), spent=already)
        model = budget.BudgetedModel(live_model(args.model), stage=R.USER_STAGE, workdir=root,
                                     model_id=args.model, ceiling=ceiling, cap_context=True)
        corpus = R._Corpus(BuiltEnvironment(root))
        anchor = _load_anchor(root)
        for item in picked:
            if spent >= args.ceiling_usd:
                plan[root.name]["stopped_on_ceiling"] = True
                break
            row = replay_one(root, item, model, corpus, anchor)
            row["corpus"] = root.name
            rows.append(row)
            spent += row.get("usd", 0.0)
            print(json.dumps({k: row.get(k) for k in ("corpus", "task_id", "outcome", "dropped", "usd")}),
                  flush=True)
    body = {"model": args.model, "seed": args.seed, "plan": plan, "rows": rows,
            "summary": summary(rows), "spent_usd": round(spent, 4)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"summary": body["summary"], "spent_usd": body["spent_usd"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
