"""Every build reports its trusted Tasks split by whether the Reference is right.

Measurement only: the build never reads the benchmark's own reward, this script
reads it afterwards from the sidecars set aside next to the workdir and prints
the split. Kept: of the trusted Tasks, the share resting on a wrong Reference.
Also printed, with no sidecar at all, is the proxy the harness can use itself:
for each trusted Task, whether independent re-runs reached the Reference's End
state, and how well that proxy tracks the sidecar split. A Task counts as right,
wrong or mixed only when every Reference has a sidecar reward; any unscored
Reference leaves the Task unknown.

    python scripts/measure/trust_split.py <workdir> [--round N] [--json out.json]

Inputs, read the way the build's own records write them: the trusted Task ids
come from the round file (`rounds/<n>/tasks.json`, latest round by default),
each Task's Reference run ids from its status row (`reference_run_ids`), the
run-to-recording map and the re-run evidence from the references record, the
needs-a-Reference pool from its pool file, and refusals from the round file.
The sidecar location and the reward field inside it are arguments with the
current layout as defaults. The script writes nothing inside the workdir, and
refuses a result path inside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from kullback.spec.split import (  # noqa: F401  (re-exported: the script's readers use these names)
    CLASSES,
    MIXED,
    REWARD_FIELD,
    RIGHT,
    SIDECAR_DIR,
    TRACES_DIR,
    UNKNOWN,
    WRONG,
    classify,
    sidecar_rewards,
)

AGREE = "agree"
DISAGREE = "disagree"
NONE = "none"
PROXIES = (AGREE, DISAGREE, NONE)

POOL_PATH = "exam/reference_pool.json"
STATUS_PATHS = ("task_status.json", "exam/task_status.json")
REFERENCES_PATHS = ("references.json", "exam/references.json")
LEDGER_PATHS = ("gates.json", "exam/gates.json")


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback



def _first_existing(workdir: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        path = workdir / name
        if path.is_file():
            return path
    return None


def round_names(workdir: Path) -> list[str]:
    rounds = workdir / "rounds"
    if not rounds.is_dir():
        return []
    names = [p.name for p in rounds.iterdir() if p.is_dir() and (p / "tasks.json").is_file()]
    return sorted(names, key=lambda n: (len(n), n))


def trusted_from_round(workdir: Path, name: Optional[str]) -> tuple[str, list[str], list[str]]:
    """(round, trusted ids, refused ids) out of the build's own round file.

    A missing round file is an error, and so is one that cannot be read:
    an unreadable snapshot never reports an empty round.
    """
    names = round_names(workdir)
    if not names:
        raise FileNotFoundError(f"no rounds/*/tasks.json under {workdir}")
    if name is not None and name not in names:
        raise FileNotFoundError(f"no round {name} under {workdir}; have {', '.join(names)}")
    chosen = name if name is not None else names[-1]
    path = workdir / "rounds" / chosen / "tasks.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable round file {path}") from exc
    rows = doc.get("rows") or [] if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"malformed round file {path}")
    trusted = sorted(str(r.get("task_id")) for r in rows if isinstance(r, dict) and r.get("trusted"))
    refused = sorted(str(r.get("task_id")) for r in rows if isinstance(r, dict) and r.get("refused"))
    return chosen, trusted, refused


def trusted_from_ledger(workdir: Path) -> tuple[list[str], list[str]]:
    """(trusted ids, refused ids) out of a trusted gate ruling, where one is kept."""
    for name in LEDGER_PATHS:
        rulings = _read(workdir / name, None)
        if not isinstance(rulings, list):
            continue
        for ruling in rulings:
            if not isinstance(ruling, dict) or ruling.get("stage") != "trusted":
                continue
            metrics = ruling.get("metrics") or {}
            trusted = metrics.get("trusted") or []
            refused = metrics.get("refused") or {}
            return sorted(map(str, trusted)), sorted(map(str, refused))
    return [], []


def pool_size(workdir: Path, pool: str) -> tuple[int, bool]:
    """(tasks waiting on a Reference, whether the pool file was there to say)."""
    body = _read(workdir / pool, None)
    if body is None:
        return 0, False
    return (len(body) if isinstance(body, dict) else 0), True


def proxy(reached: int, failed: int, reruns: int) -> str:
    """The harness-only evidence for one trusted Task.

    One independent re-run reaching the Reference's End state corroborates it,
    so a reach beats a miss; a Task is flagged only when nothing ever
    reproduced the End state, and named as having no evidence when no re-run
    ran at all.
    """
    if reached:
        return AGREE
    if failed:
        return DISAGREE
    if reruns:
        return DISAGREE
    return NONE


def measure(
    workdir: Path,
    round_name: Optional[str] = None,
    sidecar_dir: str = SIDECAR_DIR,
    reward_field: str = REWARD_FIELD,
    traces_dir: str = TRACES_DIR,
    pool: str = POOL_PATH,
) -> dict:
    try:
        chosen, trusted, refused = trusted_from_round(workdir, round_name)
        source = f"rounds/{chosen}/tasks.json"
    except FileNotFoundError:
        chosen, trusted, refused = "", *trusted_from_ledger(workdir)
        source = "trusted gate ruling"
        if not trusted and not refused:
            raise
    status_path = _first_existing(workdir, STATUS_PATHS)
    if status_path is None:
        raise FileNotFoundError(f"no task_status.json under {workdir}")
    status = _read(status_path, {}) or {}
    refs_path = _first_existing(workdir, REFERENCES_PATHS)
    if refs_path is None:
        raise FileNotFoundError(f"no references.json under {workdir}")
    refs = _read(refs_path, {}) or {}

    sidecars = workdir / sidecar_dir if not Path(sidecar_dir).is_absolute() else Path(sidecar_dir)
    traces = workdir / traces_dir if not Path(traces_dir).is_absolute() else Path(traces_dir)
    reward_of, sidecar_files = sidecar_rewards(sidecars, reward_field, traces)

    classes: dict[str, list[str]] = {c: [] for c in CLASSES}
    proxy_counts = {p: 0 for p in PROXIES}
    cross = {c: {p: 0 for p in PROXIES} for c in (RIGHT, WRONG)}
    rows = []
    for task_id in trusted:
        row = status.get(task_id) or {}
        ref_ids = [str(r) for r in (row.get("reference_run_ids") or [])]
        ref_row = refs.get(task_id) or {}
        trace_of = {str(r.get("run_id")): r.get("trace_id")
                    for r in (ref_row.get("references") or []) + (ref_row.get("recordings") or [])
                    if r.get("run_id")}
        rewards = [(rid, reward_of[tid]) for rid in ref_ids
                   if (tid := trace_of.get(rid)) is not None and tid in reward_of]
        unscored = len(ref_ids) - len(rewards)
        verdict = classify([r for _, r in rewards])
        classes[verdict].append(task_id)
        refset = set(ref_ids)
        failset = {str(k) for k in (ref_row.get("failed") or {})}
        reruns = [r for r in (ref_row.get("recordings") or [])
                  if r.get("kind") != "recording" and r.get("run_id")]
        reached = sum(1 for r in reruns if str(r["run_id"]) in refset)
        failed = sum(1 for r in reruns if str(r["run_id"]) in failset)
        # A re-run the record neither kept nor failed is evidence of nothing;
        # it still ran, so it counts against "none".
        other = len(reruns) - reached - failed
        mark = proxy(reached, failed, len(reruns))
        proxy_counts[mark] += 1
        if verdict in cross:
            cross[verdict][mark] += 1
        rows.append({"task_id": task_id, "class": verdict, "proxy": mark,
                     "reference_runs": ref_ids,
                     "rewards": [{"run_id": rid, "reward": r} for rid, r in rewards],
                     "unscored_references": unscored,
                     "reruns_reached": reached, "reruns_failed": failed,
                     "reruns_unresolved": other})
    scored = len(classes[RIGHT]) + len(classes[WRONG]) + len(classes[MIXED])
    pool_n, pool_found = pool_size(workdir, pool)
    return {"workdir": str(workdir), "round": chosen, "source": source,
            "trusted": trusted, "trusted_count": len(trusted),
            "classes": classes,
            "wrong_share_scored": (len(classes[WRONG]) / scored) if scored else None,
            "wrong_share_trusted": (len(classes[WRONG]) / len(trusted)) if trusted else None,
            "scored_count": scored,
            "rows": rows,
            "pool_size": pool_n, "pool_found": pool_found, "pool_path": pool,
            "refused": refused, "refused_count": len(refused),
            "proxy_counts": proxy_counts, "cross": cross,
            "cross_excluded": {"mixed": len(classes[MIXED]), "unknown": len(classes[UNKNOWN])},
            "sidecar": {"dir": str(sidecars), "files": sidecar_files,
                        "scored_traces": len(reward_of), "missing": not sidecars.is_dir()}}


def _share(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}% ({num}/{den})" if den else "n/a"


def report(result: dict) -> str:
    classes = result["classes"]
    trusted = result["trusted_count"]
    scored = result["scored_count"]
    lines = [f"trust split: {result['workdir']} ({source_word(result)})",
             f"trusted Tasks: {trusted}",
             "",
             "sidecar split (benchmark's own reward over each trusted Task's References):"]
    for cls in CLASSES:
        ids = classes[cls]
        lines.append(f"  {cls:<7} {len(ids):>4}  {_share(len(ids), scored if cls != UNKNOWN else trusted)}"
                     + (f"  {', '.join(ids)}" if ids else ""))
    lines.append(f"wrong share: {_share(len(classes[WRONG]), scored)} of scored"
                 + (f", {_share(len(classes[WRONG]), trusted)} of trusted" if scored != trusted else ""))
    if result["sidecar"]["missing"]:
        lines.append(f"no sidecar at {result['sidecar']['dir']}: classes are all unknown")
    lines += ["",
              f"needs-a-Reference pool: {result['pool_size']}"
              + ("" if result["pool_found"] else f" (no {result['pool_path']}, read as empty)"),
              f"refused: {result['refused_count']}"
              + (f"  {', '.join(result['refused'])}" if result["refused"] else ""),
              "",
              "proxy (harness-only: did independent re-runs reach the Reference End state):"]
    counts = result["proxy_counts"]
    for mark in PROXIES:
        ids = [r["task_id"] for r in result["rows"] if r["proxy"] == mark]
        lines.append(f"  {mark:<8} {counts[mark]:>4}  {_share(counts[mark], trusted)}"
                     + (f"  {', '.join(ids)}" if ids else ""))
    lines += ["",
              "sidecar x proxy (mixed/unknown excluded: "
              f"{result['cross_excluded']['mixed']} mixed, {result['cross_excluded']['unknown']} unknown):",
              f"  {'':<5} {'agree':>6} {'disagree':>8} {'none':>6}"]
    for cls in (RIGHT, WRONG):
        cell = result["cross"][cls]
        lines.append(f"  {cls:<5} {cell[AGREE]:>6} {cell[DISAGREE]:>8} {cell[NONE]:>6}")
    return "\n".join(lines) + "\n"


def source_word(result: dict) -> str:
    return f"round {result['round']}, {result['source']}" if result["round"] else result["source"]


def _inside(child: Path, parent: Path) -> bool:
    """Whether one path sits inside another, following links on both sides."""
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--round", dest="round_name", default=None,
                        help="Round file to read trusted Tasks from (default: latest).")
    parser.add_argument("--sidecar-dir", default=SIDECAR_DIR,
                        help="Benchmark sidecars, absolute or under the workdir "
                             f"(default: {SIDECAR_DIR}).")
    parser.add_argument("--reward-field", default=REWARD_FIELD,
                        help="Dotted reward field inside each sidecar's fields "
                             f"(default: {REWARD_FIELD}).")
    parser.add_argument("--traces-dir", default=TRACES_DIR,
                        help="Recordings, for sidecars that carry no trace id "
                             f"(default: {TRACES_DIR}).")
    parser.add_argument("--pool", default=POOL_PATH,
                        help=f"Needs-a-Reference pool file under the workdir (default: {POOL_PATH}).")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the machine-readable result here (outside the workdir).")
    args = parser.parse_args(argv)
    if args.json is not None and _inside(args.json, args.workdir):
        print(f"refusing to write {args.json} inside the workdir")
        return 2
    try:
        result = measure(args.workdir, args.round_name, args.sidecar_dir,
                         args.reward_field, args.traces_dir, args.pool)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    print(report(result), end="")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
