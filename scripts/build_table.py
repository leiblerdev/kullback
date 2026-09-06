"""One table per build (D139): the same rows for any workdir, in Markdown, on stdout.

Builds 1 to 8 were written up by hand in `docs/live-build.md` and no two printed the same numbers,
so "fidelity is not improving" was a feeling rather than a reading. This prints the table D139 asks
for from one build directory, and the output is meant to be pasted into that file unchanged.

The rows are the same every build. A row whose record this workdir does not hold prints `n/a` with
the record it would need, so a gap is visible instead of absent, and no number here is computed
from anything but the files under the workdir (D66): every detail line carries the file, and the
key or the run path inside it, that it was read from.

    uv run python scripts/build_table.py .work-retail
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from env_fidelity import CAUSE_OWNER  # noqa: E402
from env_fidelity import cause as fidelity_cause  # noqa: E402

# The Runner's own words for how one replayed call compared with the recording (runner/replay.py).
AGREES = ("same", "cosmetic", "both_refused")
PARTS = ("differs", "ours_refused", "theirs_refused", "unrecorded")

# replay.py stores a preview of each answer, not the answer, so a cause read off two long results
# would be read off two cut strings. Those calls are counted under this cause instead of guessed at.
UNREADABLE = "unreadable"
UNREADABLE_OWNER = ("n/a: replays.json keeps a 160 character preview of each answer, "
                    "which is not enough to name the cause")

# The counts D126 lets a round move, in the order the round table prints them (gates/round_end.py).
GATE_COUNTS = ("fidelity", "trusted", "refused_count", "assisted_runs", "probes_passing")

# The nine checks of the D79 suite, as the Examiner records them per Task in task_status.json.
D79_CHECKS = ("provenance_spans", "oracle_passes", "empty_fails", "unsolved_state_fails",
              "plausible_wrong_fails", "second_path_passes", "loophole_probe_fails",
              "mutation_flips", "leak_check_clean")

# A Python builtin raised out of a tool body is our error, whatever the model asked for: the
# corpus shows the customer's tool answering with a message (D67), and ours crashed instead.
BODY_EXCEPTIONS = ("NameError", "AttributeError", "KeyError", "TypeError", "IndexError",
                   "ZeroDivisionError", "ImportError", "UnboundLocalError")
PROVIDER_MARKS = ("ProviderError", "HTTP ", "RetryExhausted", "Timeout", "timed out")
# A Run that ended either of these ways ended the way the recording did, so it is not a failure.
FINISHED = ("user_stop", "transfer")


def na(record: str) -> str:
    """A number this workdir cannot answer, printed with the record that would answer it."""
    return f"n/a (needs {record})"


def share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def money(value: float) -> str:
    return f"${value:,.2f}"


def duration(ms: float) -> str:
    seconds = ms / 1000.0
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


def counted(rows: Iterable[str]) -> str:
    return ", ".join(rows) if rows else "none"


class Records:
    """Every file the table reads, and the name of each one this workdir does not hold."""

    def __init__(self, workdir: Any):
        self.workdir = Path(workdir)
        self.absent: list[str] = []

    def read(self, name: str, default: Any = None) -> Any:
        path = self.workdir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if name not in self.absent:
                self.absent.append(name)
            return default

    def lines(self, name: str) -> list[dict]:
        path = self.workdir / name
        if not path.is_file():
            if name not in self.absent:
                self.absent.append(name)
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def run_files(self) -> list[Path]:
        return sorted((self.workdir / "runs").rglob("*.jsonl"))

    def at(self, path: Path) -> str:
        """A file under the workdir as the pointer a reader can follow (D66)."""
        try:
            return str(path.relative_to(self.workdir))
        except ValueError:
            return str(path)


class Build:
    """One build directory, read once: the records the table needs and nothing else."""

    def __init__(self, workdir: Any):
        self.files = Records(workdir)
        self.workdir = self.files.workdir
        self.gates: list[dict] = self.files.read("gates.json", []) or []
        self.budget: dict = self.files.read("budget.json", {}) or {}
        self.rounds: list[dict] = self.files.read("rounds.json", []) or []
        self.status: dict = self.files.read("task_status.json", {}) or {}
        self.references: dict = self.files.read("references.json", {}) or {}
        self.replays: dict = self.files.read("replays.json", {}) or {}
        self.frozen: dict = self.files.read("tasks_frozen.json", {}) or {}
        self.scorecard: dict = self.files.read("scorecard.json", {}) or {}
        self.tool_builds: dict = self.files.read("tool_builds.json", {}) or {}
        self.environment: dict = self.files.read("environment.json", {}) or {}
        self.builder_session: list[dict] = self.files.lines("builder/session.jsonl")
        self.examiner_session: list[dict] = self.files.lines("examiner/session.jsonl")
        self.repairs: list[dict] = self._repairs()

    def _repairs(self) -> list[dict]:
        folder = self.workdir / "repairs"
        if not folder.is_dir():
            return []
        rows = []
        for path in sorted(folder.glob("*.jsonl")):
            for row in self.files.lines(f"repairs/{path.name}"):
                rows.append({**row, "file": f"repairs/{path.name}"})
        return rows

    def ruling(self, stage: str) -> Optional[dict]:
        """The last ruling a stage recorded in the ledger."""
        found = [row for row in self.gates if row.get("stage") == stage]
        return found[-1] if found else None

    def checks(self) -> list[dict]:
        """Every replayed call, with the Task, the Trace and its position carried along (D66)."""
        out = []
        for task_id, traces in self.replays.items():
            for trace_id, replay in (traces or {}).items():
                for pos, check in enumerate(replay.get("checks") or []):
                    out.append({**check, "task_id": task_id, "trace_id": trace_id, "pos": pos})
        return out


# --- the headline ------------------------------------------------------------


def coverage(build: Build) -> tuple[str, str]:
    card = (build.scorecard or {}).get("task_coverage") or {}
    if not card:
        return na("scorecard.json task_coverage"), "scorecard.json"
    frozen = list((build.frozen or {}).get("task_ids") or [])
    text = (f"{card.get('tasks_covered', 0)} of {card.get('tasks_total', 0)} frozen Tasks "
            f"({share(card.get('tasks_covered', 0), card.get('tasks_total', 0))}), "
            f"{card.get('runs_covered', 0)} of {card.get('runs_total', 0)} Runs "
            f"({share(card.get('runs_covered', 0), card.get('runs_total', 0))})")
    if frozen and len(frozen) != card.get("tasks_total"):
        text += f"; the frozen list holds {len(frozen)} ids and the scorecard counted " \
                f"{card.get('tasks_total')}"
    return text, "scorecard.json task_coverage, tasks_frozen.json task_ids"


def replay_totals(build: Build) -> tuple[dict, str]:
    """The Reference replay counts: the stage's own ruling when it landed one, else the replays."""
    ruling = build.ruling("replay_reference")
    if ruling and ruling.get("metrics"):
        return dict(ruling["metrics"]), "gates.json replay_reference metrics"
    totals: Counter = Counter()
    traces = confirmed = 0
    for per_task in build.replays.values():
        for replay in (per_task or {}).values():
            traces += 1
            confirmed += 1 if replay.get("confirmed") else 0
            for key, value in (replay.get("counts") or {}).items():
                if isinstance(value, int):
                    totals[key] += value
    if not traces:
        return {}, "replays.json"
    totals["traces"], totals["confirmed"] = traces, confirmed
    return dict(totals), "replays.json counts, summed over every replay"


def gate_names(build: Build) -> list[tuple[str, int, int]]:
    """Every gate in the ledger by name, with how many rulings it recorded and how many were green."""
    seen: dict[str, list[int]] = {}
    for row in build.gates:
        name = str(row.get("stage") or "?")
        counts = seen.setdefault(name, [0, 0])
        counts[0] += 1
        counts[1] += 1 if row.get("pass") else 0
    return [(name, total, green) for name, (total, green) in sorted(seen.items())]


def mechanic_calls(session: list[dict]) -> tuple[Counter, int]:
    """The tool calls in one agent's session, by verb, and how many assistant turns it took."""
    verbs: Counter = Counter()
    turns = 0
    for entry in session:
        if entry.get("type") != "message":
            continue
        message = entry.get("message") or {}
        if message.get("role") != "assistant":
            continue
        turns += 1
        for call in message.get("tool_calls") or []:
            verbs[str(call.get("name") or "?")] += 1
    return verbs, turns


def mechanic_row(build: Build) -> tuple[str, str]:
    verbs, turns = mechanic_calls(build.builder_session)
    if not build.builder_session:
        return na("builder/session.jsonl"), "builder/session.jsonl"
    if not verbs:
        return ("0 over 0 model turns; the session records no model turn, which is the code driver "
                "(D135)"), "builder/session.jsonl"
    listed = ", ".join(f"{verb} {count}" for verb, count in verbs.most_common())
    return f"{sum(verbs.values())} over {turns} model turns: {listed}", "builder/session.jsonl"


def repair_rows(build: Build) -> tuple[str, str, str]:
    """What the repair verbs did: the count, then the two rulings D139 asks for, or the gap."""
    if not build.repairs:
        none = ("0; the workdir has no repairs/ directory, and the code driver files no repair, "
                "so there is nothing to have moved a gate (D135)")
        return "0; the workdir has no repairs/ directory (D135)", none, none
    verbs = Counter(str(row.get("verb") or "?") for row in build.repairs)
    listed = ", ".join(f"{verb} {count}" for verb, count in verbs.most_common())
    gap = na("the round on a repair request and a gate ruling per round; repairs/*.jsonl records "
             "the verb, the target and a timestamp, and rounds.json carries no timestamp")
    return f"{len(build.repairs)} requested: {listed}", gap, gap


def spend(build: Build) -> dict:
    return dict((build.budget or {}).get("total") or {})


def context_row(build: Build, session: list[dict], name: str) -> tuple[str, str]:
    """The peak context fill for one agent, or the largest input the session does record."""
    peak = 0
    for entry in session:
        if entry.get("type") != "message":
            continue
        usage = (entry.get("message") or {}).get("usage") or {}
        peak = max(peak, int(usage.get("input") or 0) + int(usage.get("cache_read") or 0))
    record = "ContextStats.fill_at_turn_end, which no stage writes into the workdir"
    if not session:
        return na(f"{name}/session.jsonl and {record}"), f"{name}/session.jsonl"
    if peak:
        return (f"{na(record)}; the largest input a turn recorded is {peak:,} tokens",
                f"{name}/session.jsonl message usage")
    return na(record), f"{name}/session.jsonl"


def headline_rows(build: Build) -> list[tuple[str, str, str]]:
    """The rows D139 fixes: the same ones every build, in the same order."""
    totals, source = replay_totals(build)
    covered, coverage_source = coverage(build)
    gates = gate_names(build)
    green = sum(1 for _, total, ok in gates if total == ok)
    mechanic, mechanic_source = mechanic_row(build)
    requested, turned_green, stayed_red = repair_rows(build)
    total = spend(build)

    def counted_of(part: str, whole: str) -> str:
        if not totals or part not in totals or whole not in totals:
            return na(f"gates.json replay_reference metrics or replays.json counts ({part})")
        return f"{totals[part]:,} of {totals[whole]:,} ({share(totals[part], totals[whole])})"

    rows = [
        ("Tasks covered (D96)", covered, coverage_source),
        ("Traces confirming their Reference", counted_of("confirmed", "traces"), source),
        ("Writes replaying exactly", counted_of("writes_matched", "writes"), source),
        ("Reads differing in substance", counted_of("reads_semantic", "reads"), source),
        ("Gates green", f"{green} of {len(gates)} named below" if gates else na("gates.json"),
         "gates.json"),
        ("What the mechanic called", mechanic, mechanic_source),
        ("Repairs requested", requested, "repairs/"),
        ("Repairs that turned a red gate green", turned_green, "repairs/, gates.json"),
        ("Repairs that did not", stayed_red, "repairs/, gates.json"),
    ]
    if total:
        priced = int(total.get("calls") or 0) - int(total.get("unpriced_calls") or 0)
        rows += [
            ("Dollars", f"{money(float(total.get('usd') or 0.0))}", "budget.json total.usd"),
            ("Model calls", f"{int(total.get('calls') or 0):,} ({priced:,} priced, "
                            f"{int(total.get('unpriced_calls') or 0):,} unpriced, "
                            f"{int(total.get('memo_hits') or 0):,} memo hits)",
             "budget.json total"),
            ("Model call wall time", f"{duration(float(total.get('wall_ms') or 0.0))} summed over calls, "
                                     "which is not the build's own duration",
             "budget.json total.wall_ms"),
        ]
    else:
        rows += [("Dollars", na("budget.json total"), "budget.json"),
                 ("Model calls", na("budget.json total"), "budget.json"),
                 ("Model call wall time", na("budget.json total"), "budget.json")]
    rows.append(("Build duration", na("a start and an end timestamp; pipeline/state.json records "
                                      "the stage statuses and no clock"), "pipeline/state.json"))
    for name, session in (("Builder", build.builder_session), ("Examiner", build.examiner_session)):
        value, where = context_row(build, session, name.lower())
        rows.append((f"Peak context fill, {name}", value, where))
    return rows


# --- the detail: per tool ------------------------------------------------------


def check_cause(check: dict) -> str:
    """The env_fidelity cause behind one replayed call that missed, `none` when it agreed.

    A call the Runner counts as agreement gets no cause, `both_refused` included: env_fidelity
    splits two refusals whose wording differs into `error_prefix` and `error_message`, and the
    preview replays.json keeps is not always long enough to tell, so that split lives with the
    tool that can read the whole answer and not here.
    """
    verdict = str(check.get("verdict") or "")
    ours, theirs = str(check.get("ours") or ""), str(check.get("recorded") or "")
    if verdict not in PARTS or verdict == "unrecorded":
        return "none"
    if verdict == "theirs_refused":
        return fidelity_cause("only_theirs_errored", {}, {"error": theirs})
    if verdict == "ours_refused":
        return fidelity_cause("only_ours_errored", {"error": _message(ours)}, {"result": theirs})
    mine, real = _value(ours), _value(theirs)
    if mine is None or real is None:
        return UNREADABLE
    return fidelity_cause("result_differs", {"result": mine}, {"result": real})


def _message(preview: str) -> str:
    """The error text inside a preview of a ToolCallError, or the preview when it is not one."""
    value = _value(preview)
    if isinstance(value, dict) and "payload" in value:
        return str(value.get("payload"))
    return preview.strip('"')


def _value(preview: str) -> Any:
    if preview.endswith("..."):
        return None
    try:
        return json.loads(preview)
    except ValueError:
        return preview


def per_tool(build: Build) -> tuple[list[dict], dict[str, Counter]]:
    """One row per tool over every replayed call, and the calls that missed, by cause."""
    rows: dict[str, Counter] = defaultdict(Counter)
    kinds: dict[str, set] = defaultdict(set)
    causes: dict[str, Counter] = defaultdict(Counter)
    pointers: dict[str, str] = {}
    for check in build.checks():
        tool = str(check.get("tool") or "?")
        verdict = str(check.get("verdict") or "unrecorded")
        rows[tool][verdict] += 1
        rows[tool]["calls"] += 1
        kinds[tool].add(str(check.get("kind") or "?"))
        why = check_cause(check)
        if why != "none":
            causes[why][tool] += 1
            rows[tool][f"cause:{why}"] += 1
            pointers.setdefault(
                tool, f"replays.json {check['task_id']}/{check['trace_id']} checks[{check['pos']}]")
    assisted = set((build.environment or {}).get("assisted_tools") or [])
    out = []
    for tool in sorted(set(rows) | set(build.tool_builds)):
        counts = rows[tool]
        misses = {key[len("cause:"):]: value for key, value in counts.items() if key.startswith("cause:")}
        top = max(misses.items(), key=lambda kv: kv[1])[0] if misses else "none"
        attempts, reds = body_gates(build, tool)
        out.append({"tool": tool, "kind": "/".join(sorted(kinds[tool])) or "not called",
                    "counts": counts, "top_cause": top, "assisted": tool in assisted,
                    "attempts": attempts, "reds": reds,
                    "where": pointers.get(tool, "replays.json checks")})
    out.sort(key=lambda row: (-(row["counts"]["calls"] - sum(row["counts"][v] for v in AGREES)),
                              row["tool"]))
    return out, causes


def body_gates(build: Build, tool: str) -> tuple[str, str]:
    """How many bodies this tool took, and which gates were still red on the last one.

    The per-tool rulings in `gates.json` carry no tool name, so this is the one record that says
    which gate is red for which body.
    """
    nodes = (build.tool_builds.get(tool) or {}).get("nodes") or []
    if not nodes:
        return na(f"tool_builds.json {tool}"), na(f"tool_builds.json {tool}")
    last = nodes[-1]
    reds = [str(ruling.get("stage")) for ruling in last.get("gates") or [] if not ruling.get("pass")]
    return str(len(nodes)), counted(reds)


# --- the detail: per Task, per round, per repair, per failing Run ----------------


def task_rows(build: Build) -> list[dict]:
    """One row per Task on the frozen list: how far it got and what stopped it."""
    frozen = list((build.frozen or {}).get("task_ids") or []) or sorted(build.status)
    covered = {row.get("task_id"): row.get("reason")
               for row in ((build.scorecard or {}).get("task_coverage") or {}).get("uncovered") or []}
    rows = []
    for task_id in frozen:
        status = build.status.get(task_id) or {}
        checks = status.get("checks") or {}
        failed = [name for name in D79_CHECKS if checks.get(name) is False]
        confirmed = bool(status.get("reference_confirmed"))
        trusted = bool(status.get("verifier_passed"))
        rows.append({
            "task_id": task_id,
            "confirmed": confirmed,
            "trusted": trusted,
            "checks": ", ".join(failed) if checks else ("no D79 record" if not confirmed else "none"),
            "not_run": ", ".join(status.get("not_run") or ()),
            "atoms": atom_line(build, task_id),
            "reason": str(status.get("reason") or covered.get(task_id) or ""),
            "rank": 0 if (confirmed and not trusted) else (1 if not confirmed else 2),
            "where": f"task_status.json {task_id}",
        })
    rows.sort(key=lambda row: (row["rank"], row["task_id"]))
    return rows


def atom_line(build: Build, task_id: str) -> str:
    """The atoms a Task's Verifier requires, which is what a Verdict is measured on."""
    path = build.workdir / "verifiers" / f"{task_id}.json"
    try:
        verifier = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return na(f"verifiers/{task_id}.json")
    atoms = verifier.get("atoms") or []
    required = [atom for atom in atoms if atom.get("kind") == "required"]
    if not required:
        return f"{len(atoms)} atoms, none required"
    first = str(required[0].get("description") or required[0].get("id") or "?")
    return f"{len(required)} required of {len(atoms)}: {first}"


def round_rows(build: Build) -> list[dict]:
    """One row per round: which of D126's counts moved, what the round spent, how many turns."""
    rows = []
    previous: dict = {}
    turns = mechanic_calls(build.builder_session)[1] + mechanic_calls(build.examiner_session)[1]
    for pos, record in enumerate(build.rounds):
        counts = record.get("counts") or {}
        moved = [f"{key} {counts.get(key)} from {previous.get(key)}" for key in GATE_COUNTS
                 if previous and counts.get(key) != previous.get(key)]
        spent = counts.get("spend") or {}
        rows.append({
            "round": record.get("round", pos + 1),
            "counts": ", ".join(f"{key} {counts.get(key)}" for key in GATE_COUNTS),
            "moved": "first round" if not previous else counted(moved),
            "spend": f"builder {money(float(spent.get('builder') or 0.0))}, "
                     f"examiner {money(float(spent.get('examiner') or 0.0))}, "
                     f"total {money(float(spent.get('total') or 0.0))}" if spent
                     else na("rounds.json counts.spend"),
            "turns": "0, no model turn recorded" if turns == 0
                     else na("the round on a session entry; session.jsonl marks no round"),
            "exit": str(record.get("exit") or "the round did not exit"),
            "where": f"rounds.json[{pos}]",
        })
        previous = counts
    return rows


def repair_detail(build: Build) -> list[dict]:
    gap = na("a gate ruling per round; gates.json holds the last round only")
    return [{"verb": str(row.get("verb") or "?"), "target": str(row.get("target") or "?"),
             "before": gap, "after": gap, "where": str(row.get("file") or "repairs/")}
            for row in build.repairs]


# --- the detail: whose error a failing Run carries -------------------------------


# The rules that split a failing Run, in the order they are tried. Each names the side and what in
# the record decides it, so a line of the table can be walked back to the bytes it was read from.
RUN_RULES = (
    ("provider_error", "ours",
     "an error event naming the provider or the transport, so no model turn ever happened"),
    ("body_exception", "ours",
     "a tool result carrying a Python builtin exception, where the corpus shows a message (D67)"),
    ("replay_refused", "ours",
     "a replayed call we refused and the recording answered"),
    ("answer_differs", "ours",
     "a replayed call whose answer differs from the recorded one in substance"),
    ("simulated_user_had_no_answer", "ours",
     "a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact"),
    ("env_refused_the_candidate", "the model's",
     "the Environment refused what the model asked for, with the customer's own error class"),
    ("judge_failed_it", "unattributed",
     "the reference stage set the Run aside on the judge's words, which do not name a side"),
    ("unattributed", "unattributed", "no rule above matched what the Run recorded"),
)


def scan_run(path: Path) -> dict:
    """One Run file read for the few things the split needs, never held in memory whole."""
    found = {"provider": "", "exception": "", "refused": "", "fact_unavailable": False,
             "termination": "", "errors": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind, payload = row.get("type"), row.get("payload") or {}
        if kind == "error":
            found["errors"] += 1
            message = str(payload.get("message") or "")
            if not found["provider"] and any(mark in message for mark in PROVIDER_MARKS):
                found["provider"] = message
        elif kind == "tool_result":
            error = payload.get("error") or {}
            text = str(error.get("payload") or "")
            if text.split(":", 1)[0] in BODY_EXCEPTIONS and not found["exception"]:
                found["exception"] = text
            elif error and not found["refused"]:
                found["refused"] = f"{error.get('class') or error.get('class_')}: {text}"
        elif kind == "user_turn":
            tags = list(payload.get("tags") or ())
            if payload.get("fact_unavailable") or "fact_unavailable" in tags:
                found["fact_unavailable"] = True
        elif "termination_reason" in row:
            found["termination"] = str(row.get("termination_reason") or "")
    return found


def failing_runs(build: Build) -> list[dict]:
    """Every Run the records mark as failed, with the rule that says whose error it was.

    Failed means one of five things the records say outright: the replay did not confirm its Trace,
    the reference stage set the Run aside, the Run recorded an error event, it ended in neither of
    the two ways a recording ends, or a tool body raised a Python builtin at it. An error the
    customer's own tools raise is not a failure: the corpus is full of them and they replay.
    """
    replays: dict[str, dict] = {}
    for task_id, traces in build.replays.items():
        for replay in (traces or {}).values():
            replays[str(replay.get("run_id") or "")] = {**replay, "task_id": task_id}
    set_aside: dict[str, str] = {}
    for task_id, reference in build.references.items():
        for run_id, why in (reference.get("failed") or {}).items():
            set_aside[run_id] = f"{task_id}: {why}"
    rows = []
    for path in build.files.run_files():
        run_id = path.stem
        replay = replays.get(run_id)
        aside = set_aside.get(run_id)
        found = scan_run(path)
        reasons = list((replay or {}).get("reasons") or ())
        unconfirmed = replay is not None and not replay.get("confirmed")
        broke = (bool(found["errors"]) or bool(found["exception"])
                 or (found["termination"] and found["termination"] not in FINISHED))
        if not (unconfirmed or aside or broke):
            continue
        rule, detail = classify(found, reasons, aside)
        rows.append({"run_id": run_id, "kind": kind_of(run_id), "rule": rule,
                     "side": next(side for name, side, _ in RUN_RULES if name == rule),
                     "detail": detail, "where": build.files.at(path)})
    rows.sort(key=lambda row: ([name for name, _, _ in RUN_RULES].index(row["rule"]), row["run_id"]))
    return rows


def classify(found: dict, reasons: list[str], aside: Optional[str]) -> tuple[str, str]:
    """The first rule that matches, and the words in the record that matched it."""
    if found["provider"]:
        return "provider_error", found["provider"][:160]
    if found["exception"]:
        return "body_exception", found["exception"][:160]
    refused = [reason for reason in reasons if reason.endswith("ours_refused")]
    if refused:
        return "replay_refused", refused[0]
    differs = [reason for reason in reasons if reason.endswith("differs")]
    if differs:
        return "answer_differs", differs[0]
    if found["fact_unavailable"]:
        return "simulated_user_had_no_answer", "a user turn is tagged fact_unavailable"
    if found["refused"]:
        return "env_refused_the_candidate", found["refused"][:160]
    if aside:
        return "judge_failed_it", aside[:160]
    return "unattributed", "; ".join(reasons)[:160] or found["termination"] or "no reason recorded"


def kind_of(run_id: str) -> str:
    for prefix in ("replay", "reroll", "probe", "oracle", "candidate"):
        if run_id.startswith(prefix):
            return prefix
    return "run"


# --- rendering ----------------------------------------------------------------


def table(headers: list[str], aligns: list[str], rows: list[list[str]]) -> list[str]:
    line = "|" + "|".join(f" {head} " for head in headers) + "|"
    rule = "|" + "|".join({"r": "---:", "l": "---"}[a] for a in aligns) + "|"
    return [line, rule] + ["|" + "|".join(f" {cell} " for cell in row) + "|" for row in rows]


def capped(rows: list, limit: int, what: str) -> tuple[list, list[str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows, []
    return rows[:limit], ["", f"{len(rows) - limit} more {what} not listed; "
                              f"pass `--limit 0` to print every row."]


def render(build: Build, limit: int = 20) -> str:
    rows = headline_rows(build)
    covered = rows[0][1]
    lines = [f"# Build table: {build.workdir.name}", "",
             f"**Tasks covered: {covered}**", "",
             f"Read from `{build.workdir}` by `scripts/build_table.py` (D139). Every line names the "
             "file it was read from, and a number no record here holds is printed as `n/a` with the "
             "record it would need (D66).", ""]
    if build.files.absent:
        lines += [f"Records this workdir does not hold: {counted([f'`{n}`' for n in build.files.absent])}.", ""]
    lines += table(["row", "value", "read from"], ["l", "l", "l"],
                   [[name, value, f"`{where}`"] for name, value, where in rows])
    lines += ["", "## Gates", ""] + gates_section(build)
    lines += ["", "## Per tool, over the replayed calls", ""] + tool_section(build)
    lines += ["", "## Per Task", ""] + task_section(build, limit)
    lines += ["", "## Per round", ""] + round_section(build)
    lines += ["", "## Per repair verb", ""] + repair_section(build)
    lines += ["", "## Per failing Run: the model's error or ours", ""] + run_section(build, limit)
    return "\n".join(lines) + "\n"


def gates_section(build: Build) -> list[str]:
    names = gate_names(build)
    if not names:
        return [na("gates.json") + "."]
    rows = []
    for name, total, green in names:
        reds = [row for row in build.gates if row.get("stage") == name and not row.get("pass")]
        failures = reds[-1].get("failures") if reds else []
        first = str((failures or ["no failure text"])[0])[:150] if reds else ""
        rows.append([f"`{name}`", "green" if green == total else "red", f"{green} of {total}",
                     first, "`gates.json`"])
    return table(["gate", "ruling", "green rulings", "what the last red one says", "read from"],
                 ["l", "l", "r", "l", "l"], rows)


def tool_section(build: Build) -> list[str]:
    rows, causes = per_tool(build)
    if not rows:
        return [na("replays.json") + "."]
    body = []
    for row in rows:
        counts = row["counts"]
        agrees = sum(counts[word] for word in AGREES)
        body.append([f"`{row['tool']}`", row["kind"], f"{counts['calls']:,}",
                     f"{agrees:,}", f"{counts['differs']:,}", f"{counts['ours_refused']:,}",
                     f"{counts['theirs_refused']:,}", row["top_cause"], row["attempts"],
                     row["reds"], "yes" if row["assisted"] else "", f"`{row['where']}`"])
    lines = table(["tool", "kind", "calls", "agrees", "differs", "ours refused", "theirs refused",
                   "top cause", "bodies", "gates red on the last body", "assisted", "read from"],
                  ["l", "l", "r", "r", "r", "r", "r", "l", "r", "l", "l", "l"], body)
    lines += ["", "Agrees is `same`, `cosmetic` and `both_refused` together, which is how "
                  "`runner/replay.py` counts a call that agrees; a refusal on both sides whose "
                  "wording differs is agreement here and a miss to `env_fidelity.py`, which reads "
                  "the whole answer rather than the preview replays.json keeps.", "",
              "### Where the misses come from", ""]
    if not causes:
        return lines + ["No replayed call missed."]
    total = sum(sum(tools.values()) for tools in causes.values())
    body = []
    for why, tools in sorted(causes.items(), key=lambda kv: -sum(kv[1].values())):
        calls = sum(tools.values())
        body.append([why, f"{calls:,}", share(calls, total),
                     ", ".join(f"`{tool}` {count}" for tool, count in sorted(tools.items())),
                     CAUSE_OWNER.get(why, UNREADABLE_OWNER if why == UNREADABLE else ""),
                     "`replays.json`"])
    return lines + table(["cause", "calls", "share of the misses", "tools", "owner", "read from"],
                         ["l", "r", "r", "l", "l", "l"], body)


def task_section(build: Build, limit: int) -> list[str]:
    rows = task_rows(build)
    if not rows:
        return [na("tasks_frozen.json and task_status.json") + "."]
    shown, note = capped(rows, limit, "Tasks")
    body = [[f"`{row['task_id']}`", "yes" if row["confirmed"] else "no",
             "yes" if row["trusted"] else "no", row["checks"], row["not_run"] or "",
             row["atoms"], row["reason"][:150], f"`{row['where']}`"] for row in shown]
    lines = table(["Task", "Reference confirmed", "Verifier trusted", "D79 checks that failed",
                   "checks not run", "atoms", "reason", "read from"],
                  ["l", "l", "l", "l", "l", "l", "l", "l"], body)
    lines += ["", "Ordered by how far the Task got: a confirmed Reference whose Verifier the suite "
                  "refused first, then the Tasks with no confirmed Reference, then the covered ones."]
    return lines + note


def round_section(build: Build) -> list[str]:
    rows = round_rows(build)
    if not rows:
        return [na("rounds.json; this build ran no round driver") + "."]
    body = [[str(row["round"]), row["counts"], row["moved"], row["spend"], row["turns"],
             row["exit"], f"`{row['where']}`"] for row in rows]
    return table(["round", "gate counts (D126)", "moved", "spend", "turns", "exit", "read from"],
                 ["r", "l", "l", "l", "l", "l", "l"], body)


def repair_section(build: Build) -> list[str]:
    rows = repair_detail(build)
    if not rows:
        return ["No repair verb was called: the workdir has no `repairs/` directory, which is what "
                "a build under the code driver leaves (D135)."]
    return table(["verb", "called on", "ruling before", "ruling after", "read from"],
                 ["l", "l", "l", "l", "l"],
                 [[f"`{row['verb']}`", f"`{row['target']}`", row["before"], row["after"],
                   f"`{row['where']}`"] for row in rows])


def run_section(build: Build, limit: int) -> list[str]:
    rows = failing_runs(build)
    if not rows:
        return ["No Run failed, or no Run was written: " + na("runs/") + "."]
    by_rule: Counter = Counter(row["rule"] for row in rows)
    body = []
    for name, side, what in RUN_RULES:
        if not by_rule[name]:
            continue
        body.append([name, side, f"{by_rule[name]:,}", what, "`runs/`, `replays.json`, `references.json`"])
    lines = table(["rule", "side", "Runs", "what the record says", "read from"],
                  ["l", "l", "r", "l", "l"], body)
    ours = sum(count for name, side, _ in RUN_RULES for count in [by_rule[name]] if side == "ours")
    theirs = sum(count for name, side, _ in RUN_RULES for count in [by_rule[name]] if side == "the model's")
    rest = len(rows) - ours - theirs
    word = "Run" if len(rows) == 1 else "Runs"
    lines += ["", f"{len(rows):,} failing {word}: {ours:,} ours, {theirs:,} the model's, "
                  f"{rest:,} unattributed. The rules are tried in the order above and the first one "
                  "that matches decides the Run.", ""]
    shown, left = sample_by_rule(rows, limit)
    lines += table(["Run", "kind", "side", "rule", "what the record says", "read from"],
                   ["l", "l", "l", "l", "l", "l"],
                   [[f"`{row['run_id']}`", row["kind"], row["side"], row["rule"],
                     row["detail"], f"`{row['where']}`"] for row in shown])
    if left:
        lines += ["", f"{left:,} more failing Runs not listed; pass `--limit 0` to print every row."]
    return lines


def sample_by_rule(rows: list[dict], limit: int) -> tuple[list[dict], int]:
    """The first few Runs of every rule that fired, so no rule is crowded out by a louder one."""
    if limit <= 0:
        return rows, 0
    rules = [name for name, _, _ in RUN_RULES if any(row["rule"] == name for row in rows)]
    each = max(1, limit // max(1, len(rules)))
    shown = [row for name in rules
             for row in [r for r in rows if r["rule"] == name][:each]]
    return shown, len(rows) - len(shown)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path, help="The build directory to read; it is never written to.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Rows per long detail table; 0 prints every row.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.workdir.is_dir():
        raise SystemExit(f"{args.workdir} is not a directory")
    print(render(Build(args.workdir), args.limit), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
