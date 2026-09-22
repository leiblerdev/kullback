"""Replay stored Candidate Runs through reset and step and require the same world (G1).

For each stored Candidate Run with a rule-driven user and no stand-in route, across the given
workdirs: feed its own assistant messages back through `reset` (under the Run's seed) and `step`,
and require the same tool results, the same user turns, the same stop reason and the same Verdict.
A disagreement is a finding about determinism, not something papered over: each is listed by Run
id with the first event that differs.

Reads the workdirs in place and never writes into them. Episode Runs go under --outdir, which
defaults to .claude-scratch/episode-acceptance under this worktree.

Usage: uv run python scripts/episode_acceptance.py [--workdir ...] [--outdir ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORLD_EVENT_TYPES = ("tool_call", "tool_result", "user_turn", "stop", "error")


def load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """The footer and the event lines of one stored Run file."""
    footer: dict = {}
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        body = json.loads(line)
        if set(body) >= {"run_id", "termination_reason"} and "idx" not in body:
            footer = body
        elif isinstance(body, dict) and "idx" in body:
            events.append(body)
    return footer, events


def assistant_messages(events: list[dict]) -> list[dict]:
    """Each model call's assistant message in order: content plus tool calls."""
    out = []
    for event in events:
        if event.get("type") != "model_call":
            continue
        reply = (event.get("payload") or {}).get("reply") or {}
        out.append({"content": reply.get("content"),
                    "tool_calls": reply.get("tool_calls") or []})
    return out


def world_events(events: list[dict]) -> list[dict]:
    """The world half of a Run: everything except the policy's own calls."""
    return [e for e in events if e.get("type") in WORLD_EVENT_TYPES]


def compare_world(stored: list[dict], replayed: list[dict]) -> tuple[bool, str]:
    """Whether the two world halves agree, else the first event that differs."""
    if len(stored) != len(replayed):
        return False, f"event count {len(stored)} against {len(replayed)}"
    for position, (left, right) in enumerate(zip(stored, replayed, strict=True)):
        for key in ("type", "payload", "route"):
            if left.get(key) != right.get(key):
                return False, f"event {position}: {key} differs"
        if bool(left.get("assisted", False)) != bool(right.get("assisted", False)):
            return False, f"event {position}: assisted differs"
    return True, ""


def verdict_outcome(result) -> dict:
    return result.model_dump(mode="json", exclude={"run_id"})


def stored_seed(footer: dict) -> int:
    seed = footer.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("stored Run lacks an integer seed")
    return seed


def _parse_args() -> argparse.Namespace:
    """The command line: repeatable workdirs, an output folder, an optional Run cap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", action="append", required=True,
                        help="A built workdir to replay; repeatable.")
    parser.add_argument("--outdir", default=".claude-scratch/episode-acceptance")
    parser.add_argument("--limit", type=int, default=0, help="Cap Runs per workdir (0 means all).")
    return parser.parse_args()


class _Tally:
    """The counts, skips and disagreements one run of the script accumulates."""

    def __init__(self, outdir: Path, limit: int):
        self.outdir = outdir
        self.limit = limit
        self.total = 0
        self.agree = 0
        self.disagreements: list[str] = []
        self.skipped: dict[str, int] = {}

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def disagree(self, text: str) -> None:
        self.disagreements.append(text)

    def report(self) -> int:
        """The summary lines and the exit code: 0 only when every replayed Run agrees."""
        print(f"replayed {self.total} Runs, {self.agree} agree, {len(self.disagreements)} disagree")
        print(f"skipped: {self.skipped}")
        for line in self.disagreements:
            print(f"disagree: {line}")
        return 0 if self.total > 0 and self.agree == self.total and not self.disagreements else 1


def _workdir_setup(root: Path, tally: _Tally) -> dict | None:
    """One workdir's replay inputs, or None where it refuses to replay (recorded)."""
    from kullback.builder import compile_env
    from kullback.runner import canon
    from kullback.runner.records import EntitySchema, ToolSig
    from kullback.runner.world import BuiltEnvironment

    name = root.name
    env = BuiltEnvironment(root)
    schema = EntitySchema.model_validate(json.loads((root / "schema.json").read_text()))
    sigs = [ToolSig.model_validate(s)
            for s in json.loads((root / "tool_sigs.json").read_text())]
    bodies = json.loads((root / "bodies.json").read_text())
    if isinstance(bodies, dict) and "bodies" in bodies:
        bodies = bodies["bodies"]
    if compile_env.module_source(schema, sigs, bodies) != env.toolkit_source():
        tally.disagree(f"{name}: rendered toolkit differs from env/ files; refusing to replay")
        return None
    canon_rules = canon.load_rules(root / "canon-rules.json")
    write_tools = {s.name for s in sigs if getattr(s, "kind", None) == "write"}
    flagged = {s.name for s in sigs if getattr(s, "unclassified", False)}
    return {"root": root, "name": name, "env": env,
            "score_kwargs": dict(canon=canon_rules, judge_results=None, schema=schema,
                                 write_tools=write_tools or None, flagged_tools=flagged)}


def _task_verifier(ctx: dict, task_dir: Path, tally: _Tally) -> tuple | None:
    """One task folder's id and stored Verifier, or None where it is skipped (recorded)."""
    from kullback.runner.records import Verifier

    task_id = task_dir.name
    verifier_path = ctx["root"] / "verifiers" / f"{task_id}.json"
    if not verifier_path.is_file():
        tally.skip("no stored Verifier")
        return None
    verifier = Verifier.model_validate(json.loads(verifier_path.read_text()))
    return task_id, verifier


def _select_attempt(task_id: str, stem: str, tally: _Tally) -> str | None:
    """Whether a Run file names a Candidate attempt, or None where it is skipped (recorded)."""
    if stem.startswith(("replay-", "synth-", "second-")):
        tally.skip("not a Candidate Run")
        return None
    if stem.startswith("reroll-"):
        attempt = stem.rsplit("-", 1)[-1]
    elif stem.startswith(f"{task_id}-"):
        attempt = stem[len(task_id) + 1:]
    else:
        tally.skip("not a Candidate Run")
        return None
    if not attempt.isdigit():
        tally.skip("no attempt number")
        return None
    return attempt


def _eligible_run(path: Path, tally: _Tally) -> tuple | None:
    """One Run file's footer and events, or None where it is skipped (recorded)."""
    footer, events = load_jsonl(path)
    if not events:
        tally.skip("empty file")
        return None
    routes = {e.get("route") for e in events if e.get("type") == "tool_result"}
    if "llm" in routes:
        tally.skip("stand-in route")
        return None
    return footer, events


def _check_task(env, task_id: str, tally: _Tally):
    """The Task to replay, or None where it is skipped (recorded, uncounted)."""
    try:
        task = env.task(task_id)
    except Exception:
        tally.skip("no Task file")
        tally.total -= 1
        return None
    if env.rules(task) is None:
        tally.skip("no user rules")
        tally.total -= 1
        return None
    if env.agent_driven(task_id):
        tally.skip("agent-driven user")
        tally.total -= 1
        return None
    return task


def _replay_run(ctx: dict, tally: _Tally, task_id: str, task, path: Path,
                footer: dict, events: list[dict], verifier) -> bool:
    """Replay one stored Run and compare world, stop and Verdict; disagreements recorded."""
    from kullback.runner.verdict import load_run, verdict
    from kullback.runner.world import Episode

    run_id = footer.get("run_id") or path.stem
    episode = Episode(ctx["env"], outdir=tally.outdir / ctx["name"] / task_id)
    try:
        # The stored seed is the world input; an attempt suffix is not a seed.
        # Missing or malformed seeds are reported rather than invented.
        episode.reset(task_id, stored_seed(footer))
        done = None
        for message in assistant_messages(events):
            done = episode.step(message)
            if done.done:
                break
    except Exception as exc:
        tally.disagree(f"{run_id}: replay raised {type(exc).__name__}: {exc}")
        return False
    replayed_footer, replayed_events = load_jsonl(episode.run_path())
    if any(replayed_footer.get(key) != footer.get(key)
           for key in ("seed", "starting_state", "end_state")):
        tally.disagree(f"{run_id}: seed or world state differs")
        return False
    ok, why = compare_world(world_events(events), world_events(replayed_events))
    if not ok:
        tally.disagree(f"{run_id}: {why}")
        return False
    if (done is None and not episode._state.stopped) or \
            replayed_footer.get("termination_reason") != footer.get("termination_reason"):
        tally.disagree(
            f"{run_id}: stop reason {replayed_footer.get('termination_reason')} "
            f"against {footer.get('termination_reason')}")
        return False
    first = verdict(load_run(path), verifier, **ctx["score_kwargs"])
    second = verdict(load_run(episode.run_path()), verifier, **ctx["score_kwargs"])
    if verdict_outcome(first) != verdict_outcome(second):
        tally.disagree(f"{run_id}: verdict {first.class_} against {second.class_} "
                       f"(stored failing {first.failing_atom}, "
                       f"replayed failing {second.failing_atom})")
        return False
    return True


def main() -> int:
    args = _parse_args()
    tally = _Tally(Path(args.outdir), args.limit)
    for root in [Path(w) for w in args.workdir]:
        ctx = _workdir_setup(root, tally)
        if ctx is None:
            continue
        examined = 0
        for task_dir in sorted((root / "runs").iterdir()) if (root / "runs").is_dir() else []:
            if not task_dir.is_dir():
                continue
            selected = _task_verifier(ctx, task_dir, tally)
            if selected is None:
                continue
            task_id, verifier = selected
            for path in sorted(task_dir.glob("*.jsonl")):
                if _select_attempt(task_id, path.stem, tally) is None:
                    continue
                selected_run = _eligible_run(path, tally)
                if selected_run is None:
                    continue
                footer, events = selected_run
                if tally.limit and examined >= tally.limit:
                    break
                examined += 1
                tally.total += 1
                task = _check_task(ctx["env"], task_id, tally)
                if task is None:
                    continue
                if _replay_run(ctx, tally, task_id, task, path, footer, events, verifier):
                    tally.agree += 1
    return tally.report()


def replayed_as_jsonl(events: list[dict]) -> list[dict]:
    """Episode record events into the JSONL event shape the comparison reads."""
    out = []
    for event in events:
        out.append({"idx": event.get("idx"), "type": event.get("type"),
                    "payload": event.get("payload") or {}, "route": event.get("route")})
    return out


if __name__ == "__main__":
    sys.exit(main())
