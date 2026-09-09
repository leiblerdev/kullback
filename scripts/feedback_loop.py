"""The Builder and Examiner feedback loop of one build, read off the records, as a markdown report.

    uv run python scripts/feedback_loop.py .work-retail

What the loop did, round by round: what the Examiner filed (kind, the verb it suggested, the Task),
whether the Builder called that verb on that Task in a later round, and whether the Task then moved
(entered the trusted list, or its Reference was confirmed); the tool mix of each agent per round; and
the blockers no finding named, ranked by the Tasks they cost, so a reader can see where the
Examiner's feedback points and where it is silent. Every number comes from a file under the workdir
(D66): findings.json, rounds.json, the two session.jsonl files, task_status.json, gates.json, and
the per-Task suite rows the build table already reads (`scripts/build_table.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_table import Build, table, task_rows  # noqa: E402

REPAIR_VERBS = ("repair_intent", "repair_recompile", "repair_refuse_task", "repair_escalate", "repair")


def _session(build: Build, agent: str) -> list[dict]:
    return build.files.lines(f"{agent}/session.jsonl")


def _calls(session: list[dict]) -> list[dict]:
    """Every tool call an agent made: name, arguments, timestamp."""
    out = []
    for event in session:
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            out.append({"name": call.get("name"), "arguments": call.get("arguments") or {},
                        "at": event.get("timestamp") or 0.0})
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_call"):
                    out.append({"name": block.get("name"), "arguments": block.get("input") or {},
                                "at": event.get("timestamp") or 0.0})
    return out


def _round_of(build: Build, at: float) -> Optional[int]:
    """The round a timestamp falls in, off rounds.json's clocks; None before the first or after the last."""
    for record in build.rounds:
        counts = record.get("counts") or {}
        start, end = counts.get("started_at"), counts.get("ended_at")
        if start is not None and end is not None and start <= at <= end:
            return int(record["round"])
    if build.rounds:
        last = build.rounds[-1]
        if at > (last.get("counts") or {}).get("ended_at", 0):
            return int(last["round"]) + 1  # a round still open when the records were read
    return None


def _findings(build: Build) -> list[dict]:
    raw = build.files.read("examiner/findings.json", []) or []
    items = raw if isinstance(raw, list) else list(raw.values())
    return [f for f in items if isinstance(f, dict)]


def _mentions(arguments: dict, task_id: Optional[str], tool: Optional[str]) -> bool:
    text = json.dumps(arguments, default=str)
    return bool((task_id and task_id in text) or (tool and re.search(rf"\b{re.escape(tool)}\b", text)))


def _trusted_by_round(build: Build) -> dict[int, set[str]]:
    return {int(r["round"]): set((r.get("counts") or {}).get("trusted_ids") or []) for r in build.rounds}


def finding_rows(build: Build) -> list[dict]:
    """One row per finding: what it asked, whether the Builder acted, whether the Task moved."""
    builder_calls = [dict(c, round=_round_of(build, c["at"])) for c in _calls(_session(build, "builder"))]
    trusted = _trusted_by_round(build)
    rows = []
    for finding in _findings(build):
        filed = int(finding.get("round") or 0)
        task_id, tool, verb = finding.get("task_id"), finding.get("tool"), finding.get("suggested") or "none"
        acted = [c for c in builder_calls
                 if (c["round"] or 0) > filed and c["name"] in REPAIR_VERBS and _mentions(c["arguments"], task_id, tool)]
        acted_verbs = Counter(c["name"] for c in acted)
        was_trusted = task_id in trusted.get(filed, set())
        later = [r for r in sorted(trusted) if r > filed]
        now_trusted = task_id in trusted[later[-1]] if later else None
        status = build.status.get(task_id) or {}
        rows.append({
            "finding_id": finding.get("finding_id"), "round": filed, "kind": finding.get("kind"),
            "task_id": task_id, "tool": tool, "suggested": verb, "status": finding.get("status"),
            "acted": ", ".join(f"{n} x{k}" if k > 1 else n for n, k in acted_verbs.items()) or "no",
            "same_verb": verb in acted_verbs,
            "moved": ("trusted now" if now_trusted and not was_trusted
                      else "reference confirmed" if status.get("reference_confirmed") and not now_trusted
                      else "no" if later else "no later round"),
            "text": str(finding.get("text") or "")[:120],
        })
    return rows


def tool_mix(build: Build) -> list[dict]:
    """Per round, what each agent called."""
    out = []
    for agent in ("builder", "examiner"):
        per_round: dict[int, Counter] = defaultdict(Counter)
        for call in _calls(_session(build, agent)):
            per_round[_round_of(build, call["at"]) or 0][call["name"]] += 1
        for round_no in sorted(per_round):
            counts = per_round[round_no]
            out.append({"agent": agent, "round": round_no, "calls": sum(counts.values()),
                        "mix": ", ".join(f"{name} {n}" for name, n in counts.most_common())})
    return out


def blockers(build: Build) -> dict[str, Counter]:
    """The losses no finding may have named, each ranked by the Tasks it costs."""
    named = {f.get("task_id") for f in _findings(build)} | {f.get("tool") for f in _findings(build)}
    assisted: Counter = Counter()
    other: Counter = Counter()
    for row in build.status.values():
        if row.get("reference_confirmed"):
            continue
        tools = row.get("assisted_tools") or []
        if tools:
            for tool in tools:
                assisted[tool] += 1
        else:
            reason = re.sub(r"\(.*", "", str(row.get("reason") or "")).strip()[:70] or "no reason recorded"
            other[reason] += 1
    checks: Counter = Counter()
    for row in task_rows(build):
        if row["confirmed"] and not row["trusted"]:
            for check in [c.strip() for c in str(row["checks"]).split(",") if c.strip()]:
                checks[check] += 1
    false_rejection: Counter = Counter()
    for ruling in build.gates:
        if ruling.get("stage") == "trusted":
            for value in (ruling.get("metrics") or {}).get("false_rejection", {}).values():
                false_rejection["not measured" if value is None else "1.0 (every held-out Run rejected)"
                                if value == 1.0 else "0.0 (none rejected)" if value == 0.0 else "between"] += 1
    return {"assisted_tools": Counter({f"{t}{' (named by a finding)' if t in named else ''}": n
                                       for t, n in assisted.items()}),
            "no_reference_other": other, "suite_checks": checks, "false_rejection": false_rejection}


def report(build: Build) -> list[str]:
    lines = [f"# Feedback loop: {build.workdir.name}", ""]
    rounds = build.rounds
    findings = finding_rows(build)
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in findings:
        by_round[row["round"]].append(row)
    body = []
    for record in rounds:
        n = int(record["round"])
        counts = record.get("counts") or {}
        filed = by_round.get(n, [])
        acted = sum(1 for r in filed if r["acted"] != "no")
        moved = sum(1 for r in filed if r["moved"] == "trusted now")
        body.append([str(n), str(counts.get("fidelity")), str(counts.get("trusted")),
                     str(len(filed)), str(acted), str(moved), str(len(record.get("pending_findings") or [])),
                     str(len(counts.get("repairs") or [])), str(record.get("exit") or "")])
    lines += ["## Rounds", ""] if rounds else ["No rounds.json: this workdir ran no round driver.", ""]
    if rounds:
        lines += table(["round", "fidelity", "trusted", "findings filed", "acted on later", "Task trusted later",
                        "pending at close", "repairs", "exit"], ["r"] * 9, body)
    lines += ["", "## What each agent called, per round", ""]
    mix = tool_mix(build)
    lines += table(["agent", "round", "calls", "mix"], ["l", "r", "r", "l"],
                   [[m["agent"], str(m["round"]), str(m["calls"]), m["mix"]] for m in mix]) if mix else ["No session records."]
    lines += ["", "## Every finding, and what came of it", ""]
    if findings:
        lines += table(["finding", "round", "kind", "Task", "suggested", "Builder acted", "same verb", "Task moved", "text"],
                       ["l", "r", "l", "l", "l", "l", "l", "l", "l"],
                       [[str(r["finding_id"]), str(r["round"]), str(r["kind"]), f"`{r['task_id']}`",
                         str(r["suggested"]), r["acted"], "yes" if r["same_verb"] else "no", r["moved"], r["text"]]
                        for r in findings])
        kinds = Counter(r["kind"] for r in findings)
        verbs = Counter(r["suggested"] for r in findings)
        lines += ["", f"{len(findings)} findings: kinds " + ", ".join(f"{k} {n}" for k, n in kinds.most_common())
                  + "; suggested verbs " + ", ".join(f"{k} {n}" for k, n in verbs.most_common())
                  + f"; acted on {sum(1 for r in findings if r['acted'] != 'no')}, Task trusted later "
                  + f"{sum(1 for r in findings if r['moved'] == 'trusted now')}."]
    else:
        lines += ["The Examiner filed no finding."]
    lines += ["", "## The losses, ranked, and whether a finding names them", ""]
    blocks = blockers(build)
    for title, key in (("Tasks with no Reference because a seed tool is assisted (a Task counts under each of its tools)", "assisted_tools"),
                       ("Tasks with no Reference for another reason", "no_reference_other"),
                       ("Tasks with a Reference whose Verifier failed the D79 suite, by check", "suite_checks"),
                       ("False rejection per Task (the trusted ruling)", "false_rejection")):
        counter = blocks[key]
        lines += [f"**{title}.** " + (", ".join(f"{k}: {n}" for k, n in counter.most_common()) if counter else "none") + "", ""]
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("workdir")
    args = parser.parse_args(argv)
    print("\n".join(report(Build(args.workdir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
