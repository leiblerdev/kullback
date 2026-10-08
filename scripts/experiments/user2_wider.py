"""Wider live test of the agent user's two readings (fx-user2, part 2).

Selects Tasks whose stored second-path Runs show dead turns of the rule-driven user
and whose Reference Run the benchmark passed, then re-rolls each selected Task twice
under the agent user and twice under the rule user, and grades every new Run with the
stored Verifier plus a code-computed gold-action match from the benchmark sidecar.

Round 2 runs Candidate and user both on the shipping model at low reasoning effort,
one stream at a time. Selection is read-only against the smoke workdirs and costs
nothing. Re-rolls clone each corpus workdir with APFS clonefile under .work-scratch/
and stop at the spend cap. The benchmark sidecar is used for choosing Tasks and for
measuring only: no sidecar content is ever an input to the harness.

    uv run python scripts/experiments/user2_wider.py select [--seed N]
    uv run python scripts/experiments/user2_wider.py reroll --selection <json> [--cap-usd 12]
    uv run python scripts/experiments/user2_wider.py grade --selection <json>
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

AIRLINE = Path("/Users/krishuagarwal/.herdr/worktrees/kullback/overhaul-0922-smoke9/.work-airline")
RETAIL = Path("/Users/krishuagarwal/.herdr/worktrees/kullback/overhaul-0922-smoke9/.work-retail")
CORPORA = {"airline": AIRLINE, "retail": RETAIL}
SELECT_SEED = 20260925
PER_CORPUS = 15
REROLLS_PER_ARM = 2
CAP_USD = 12.0
SCRATCH = Path(".work-scratch")
DEST_TAG = "muse"

sys.path.insert(0, ".")
from kullback.examiner import runners as R  # noqa: E402
from kullback.examiner.session import _load_anchor  # noqa: E402
from kullback.runner.world.environment import BuiltEnvironment  # noqa: E402
from scripts import reference_agreement as agree_mod  # noqa: E402
from scripts.experiments import dead_turns as dead_mod  # noqa: E402


def eligible(root: Path) -> list[dict]:
    """Tasks with dead turns in stored second-path Runs whose Reference earned reward 1."""
    dead = dead_mod.dead_turns(root)
    reward_of = agree_mod.rewards(root)
    env = BuiltEnvironment(root)
    anchor = _load_anchor(root)
    out = []
    for task_id in sorted(dead):
        task = env.task(task_id)
        ref = R._reference_id(root, task_id, task.run_ids, anchor)
        if ref is None or reward_of.get(ref) != 1.0:
            continue
        out.append({"task_id": task_id, "reference": ref,
                    "dead_turns": len(dead[task_id])})
    return out


def select(seed: int = SELECT_SEED, per_corpus: int = PER_CORPUS) -> dict:
    """Fixed-seed pick of eligible Tasks, 15 per corpus."""
    rng = random.Random(seed)
    picked = {}
    for name, root in CORPORA.items():
        pool = eligible(root)
        rng.shuffle(pool)
        picked[name] = sorted(pool[:per_corpus], key=lambda row: row["task_id"])
    return {"seed": seed, "per_corpus": per_corpus, "corpora": picked}


def cmd_select(args: argparse.Namespace) -> int:
    result = select(seed=args.seed, per_corpus=args.per_corpus)
    for name, rows in result["corpora"].items():
        print(f"{name}: {len(rows)} eligible-and-picked")
        for row in rows:
            print(f"  {row['task_id']} ref={row['reference']} dead={row['dead_turns']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def clone(root: Path, dest: Path) -> None:
    """APFS copy-on-write clone; falls back to a copy where clonefile is refused."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "-R", "-c", str(root), str(dest)], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError:
        shutil.copytree(root, dest, symlinks=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="pick Tasks, read-only, no spend")
    sel.add_argument("--seed", type=int, default=SELECT_SEED)
    sel.add_argument("--per-corpus", type=int, default=PER_CORPUS)
    sel.add_argument("--out", default="")
    reroll = sub.add_parser("reroll", help="paired re-rolls on clones, spends")
    reroll.add_argument("--selection", required=True)
    reroll.add_argument("--manifest", default=".work-scratch/user2-muse-manifest.json")
    reroll.add_argument("--cap-usd", type=float, default=CAP_USD)
    reroll.add_argument("--tag", default=DEST_TAG)
    grade = sub.add_parser("grade", help="grade manifest Runs, no spend")
    grade.add_argument("--manifest", default=".work-scratch/user2-muse-manifest.json")
    grade.add_argument("--selection", default="")
    grade.add_argument("--out", default=".work-scratch/user2-muse-grades.json")
    args = parser.parse_args(argv)
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "reroll":
        return cmd_reroll(args)
    if args.cmd == "grade":
        return cmd_grade(args)
    return 0

MODEL_ID = "opencode-go/muse-spark-1.3-contributor"
EFFORT = "low"
RULE_PREFIX = "user2-rule"
AGENT_PREFIX = "user2-agent"


class EffortModel:
    """One live model with a fixed reasoning effort on every call.

    The Runner and the user harness both query with no config of their own, so the
    experiment setting lives here, on the script's side: a caller-set effort is kept,
    and only an unset one becomes the round's. Pricing and naming pass through to
    the inner model, so the spend lands under the same id.
    """

    def __init__(self, inner: object, effort: str = EFFORT):
        self.inner = inner
        self.effort = effort
        self.name = getattr(inner, "name", "model")

    def query(self, messages: list, tools: list | None = None, config: object = None):
        from kullback.ai.provider import ModelConfig

        settings = config.model_copy(deep=True) if config is not None else ModelConfig()
        if not settings.effort and not settings.reasoning_effort:
            settings.effort = self.effort
        return self.inner.query(messages, tools=tools, config=settings)


def spend_usd(root: Path) -> float:
    """Model spend so far in this workdir, over every priced stage."""
    try:
        stages = json.loads((root / "budget.json").read_text(encoding="utf-8"))["stages"]
    except (OSError, ValueError, KeyError):
        return 0.0
    return sum(float(stage.get("usd", 0) or 0) for stage in stages.values())


def _bus_lines(dest: Path) -> int:
    path = dest / "bus.jsonl"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def cmd_reroll(args: argparse.Namespace) -> int:
    from kullback.ai.provider import live_model

    tag = getattr(args, "tag", DEST_TAG)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    dest_of = {name: SCRATCH / f"user2-{tag}-{name}" for name in selection["corpora"]}
    manifest: dict = {"selection": args.selection, "runs": [], "stopped_at_cap": False,
                      "spend_usd": 0.0, "model": MODEL_ID, "effort": EFFORT, "tag": tag}
    bases: dict[str, float] = {}
    for name, rows in selection["corpora"].items():
        dest = dest_of[name]
        clone(CORPORA[name], dest)
        bases[name] = spend_usd(dest)
        anchor = _load_anchor(dest)
        rule = R.runners_for(dest, reroll_model=EffortModel(live_model(MODEL_ID)),
                             anchor=anchor, user_model=None)
        agent = R.runners_for(dest, reroll_model=EffortModel(live_model(MODEL_ID)),
                              anchor=anchor, user_model=EffortModel(live_model(MODEL_ID)))
        bus_mark = _bus_lines(dest)
        for row in rows:
            task_id = row["task_id"]
            for arm, runners, prefix in (("rule", rule, RULE_PREFIX),
                                         ("agent", agent, AGENT_PREFIX)):
                t0 = time.time()
                bought = runners["run_rerolls"](task_id, REROLLS_PER_ARM, prefix)
                t1 = time.time()
                for entry in bought:
                    manifest["runs"].append(
                        {"corpus": name, "workdir": str(dest), "task_id": task_id,
                         "arm": arm, **entry, "t0": t0, "t1": t1, "bus_mark": bus_mark})
                manifest["spend_usd"] = sum(
                    spend_usd(dest_of[seen]) - bases[seen]
                    for seen in selection["corpora"]
                    if dest_of[seen].is_dir())
                print(f"{name} {task_id} {arm}: "
                      f"{[e['run_id'] for e in bought]} spend={manifest['spend_usd']:.2f}",
                      flush=True)
                if manifest["spend_usd"] >= args.cap_usd:
                    manifest["stopped_at_cap"] = True
                    Path(args.manifest).write_text(json.dumps(manifest, indent=1),
                                                   encoding="utf-8")
                    print(f"cap {args.cap_usd} reached, stopping")
                    return 0
    Path(args.manifest).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done spend={manifest['spend_usd']:.2f} runs={len(manifest['runs'])}")
    return 0


def run_events(path: str) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    head = rows[0] if rows and "type" not in rows[0] else {}
    return head, [row for row in rows if row.get("type")]


def verifier_pass(workdir: Path, task_id: str, run_path: str) -> tuple[bool, str | None]:
    """The stored Verifier's verdict on one new Run, derived under the old rung order."""
    from kullback.runner import target as target_mod
    from kullback.runner.canon import CanonRules
    from kullback.runner.records import Verifier

    env = BuiltEnvironment(workdir)
    stored = Path(workdir) / "verifiers" / f"{task_id}.json"
    verifier = Verifier.model_validate(json.loads(stored.read_text(encoding="utf-8")))
    canon = CanonRules.model_validate(
        json.loads((Path(workdir) / "canon-rules.json").read_text(encoding="utf-8")))
    passed, failing = target_mod.check_run(verifier, run_path, canon,
                                            write_tools=env.write_tools())
    return passed, failing


def gold_actions(root: Path, reference: str) -> list[dict]:
    """The benchmark sidecar's assistant actions for the Reference trace, measuring only."""
    traces = {t["trace_id"]: t for t in
              (json.loads(p.read_text(encoding="utf-8"))
               for p in (root / "traces").glob("*.json")) if "trace_id" in t}
    digest = traces[reference]["hash"]
    fields = json.loads((root / "grader" / f"{digest}.json").read_text(encoding="utf-8"))["fields"]
    return [action for action in fields["evaluation_criteria"]["actions"]
            if action.get("requestor") == "assistant"]


def action_match(run_path: str, gold: list[dict]) -> dict:
    """Which gold actions the new Run made, by name with canonically equal arguments."""
    from kullback.runner.records import canonical_json

    made = []
    for line in Path(run_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "tool_call":
            continue
        payload = event.get("payload") or {}
        if payload.get("requestor") not in (None, "assistant"):
            continue
        made.append((payload.get("name"), canonical_json(payload.get("args") or {})))
    hits = []
    for action in gold:
        want = (action.get("name"), canonical_json(action.get("arguments") or {}))
        hits.append(want in made)
    return {"matched": sum(hits), "of": len(hits),
            "all_matched": bool(hits) and all(hits)}


def user_tool_counts(workdir: Path, t0: float, t1: float) -> dict:
    """The agent user's tool calls between two wall-clock marks, off the shared bus.

    One call appears in several bus events of its turn, so calls count once by id,
    read off the turn's own message.
    """
    from collections import Counter

    out: Counter = Counter()
    seen: set[str] = set()
    path = Path(workdir) / "bus.jsonl"
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("agent") != "user":
            continue
        if not (t0 <= float(row.get("recorded_at", 0)) <= t1):
            continue
        event = row.get("event") or {}
        if event.get("type") != "turn_end":
            continue
        for call in (event.get("message") or {}).get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            if call_id and call_id not in seen:
                seen.add(call_id)
                out[str(call.get("name") or "unknown")] += 1
    return dict(out)


def cmd_grade(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selection = json.loads(Path(manifest["selection"]).read_text(encoding="utf-8"))
    refs = {row["task_id"]: row["reference"]
            for rows in selection["corpora"].values() for row in rows}
    grades = []
    for entry in manifest["runs"]:
        workdir = Path(entry["workdir"])
        passed, failing = verifier_pass(workdir, entry["task_id"], entry["path"])
        gold = gold_actions(CORPORA[entry["corpus"]], refs[entry["task_id"]])
        match = action_match(entry["path"], gold)
        _, events = run_events(entry["path"])
        drops: dict[str, int] = {}
        for event in events:
            if event.get("type") != "user_turn":
                continue
            payload = event.get("payload") or {}
            if payload.get("agent_turn_dropped"):
                drops[payload["agent_turn_dropped"]] = drops.get(
                    payload["agent_turn_dropped"], 0) + 1
            for tag in payload.get("tags") or []:
                if str(tag).startswith("agent_user_"):
                    drops[str(tag)] = drops.get(str(tag), 0) + 1
        tools = user_tool_counts(workdir, entry["t0"], entry["t1"]) if entry["arm"] == "agent" else {}
        grades.append({**{k: entry[k] for k in
                          ("corpus", "task_id", "arm", "run_id", "termination_reason",
                           "user_end")},
                       "verifier_pass": passed, "verifier_failing": failing,
                       "gold_matched": match["matched"], "gold_of": match["of"],
                       "gold_all_matched": match["all_matched"],
                       "guard_drops": drops, "user_tools": tools})
        print(f"{entry['corpus']} {entry['task_id']} {entry['arm']} {entry['run_id']}: "
              f"verifier={passed} gold={match['matched']}/{match['of']} "
              f"end={entry['user_end']}", flush=True)
    Path(args.out).write_text(json.dumps(
        {"grades": grades, "spend_usd": manifest.get("spend_usd")}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

