"""Every build reports its trusted Tasks split by whether the Reference is right.

Measurement only: the build never reads the benchmark's own reward, this script
reads it afterwards from the sidecars set aside next to the workdir and prints
the split. Kept: of the trusted Tasks, the share resting on a wrong Reference.
A Task counts as right, wrong or mixed only when every Reference has a sidecar
reward; any unscored Reference leaves the Task unknown.

    python scripts/measure/trust_split.py <workdir> [--json out.json]

Inputs, read the way the build's own records write them: the trusted Task ids
come from the latest round file (`rounds/<n>/tasks.json`), each Task's
Reference run ids from its status row (`reference_run_ids`), the run-to-trace
map from the references record, and refusals from the round file.
The sidecar location and the reward field inside it are arguments with the
current layout as defaults. The script writes nothing inside the workdir, and
refuses a result path inside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

RIGHT = "right"
WRONG = "wrong"
MIXED = "mixed"
UNKNOWN = "unknown"
CLASSES = (RIGHT, WRONG, MIXED, UNKNOWN)

SIDECAR_DIR = "grader"
TRACES_DIR = "traces"
REWARD_FIELD = "reward_info.reward"
STATUS_PATHS = ("task_status.json", "exam/task_status.json")
REFERENCES_PATHS = ("references.json", "exam/references.json")


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _dig(doc: Any, dotted: str) -> Any:
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


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


def trusted_from_round(workdir: Path) -> tuple[str, list[str], list[str]]:
    """(round, trusted ids, refused ids) out of the latest round file.

    A missing round file is an error, and so is one that cannot be read:
    an unreadable snapshot never reports an empty round.
    """
    names = round_names(workdir)
    if not names:
        raise FileNotFoundError(f"no rounds/*/tasks.json under {workdir}")
    chosen = names[-1]
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


def trace_hashes(traces_dir: Optional[Path]) -> dict[str, str]:
    """Recording hash -> trace id, for sidecars that carry no trace id."""
    mapping: dict[str, str] = {}
    if traces_dir is None or not traces_dir.is_dir():
        return mapping
    for path in sorted(traces_dir.glob("*.json")):
        trace = _read(path, {}) or {}
        if trace.get("hash") and trace.get("trace_id"):
            mapping[str(trace["hash"])] = str(trace["trace_id"])
    return mapping


def reward_from_doc(doc: Any, reward_field: str) -> Optional[float]:
    """The benchmark's own reward out of one sidecar, None where it is absent."""
    fields = doc.get("fields") or {}
    reward = _dig(fields, reward_field)
    if reward is None:
        reward = fields.get("reward", doc.get("reward"))
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def sidecar_rewards(
    sidecar_dir: Path, reward_field: str, traces_dir: Optional[Path]
) -> tuple[dict[str, float], int]:
    """trace id -> the benchmark's own reward, plus how many sidecars were read.

    The trace id is read off the sidecar itself; where the sidecar carries none,
    the file stem is resolved through the traces. The reward is the dotted field
    inside the sidecar's fields, falling back to a top-level reward.
    """
    reward_of: dict[str, float] = {}
    files = sorted(sidecar_dir.glob("*.json")) if sidecar_dir.is_dir() else []
    hash_to_trace = trace_hashes(traces_dir)
    for path in files:
        doc = _read(path, {}) or {}
        trace_id = doc.get("trace_id") or hash_to_trace.get(path.stem)
        if not trace_id:
            continue
        reward = reward_from_doc(doc, reward_field)
        if reward is None:
            continue
        reward_of[str(trace_id)] = reward
    return reward_of, len(files)


def classify(rewards: list[float]) -> str:
    if not rewards:
        return UNKNOWN
    if all(r == 1.0 for r in rewards):
        return RIGHT
    if all(r == 0.0 for r in rewards):
        return WRONG
    return MIXED


def load_records(workdir: Path) -> tuple[dict, dict]:
    """(status rows, references record) out of the build's own records."""
    status_path = _first_existing(workdir, STATUS_PATHS)
    if status_path is None:
        raise FileNotFoundError(f"no task_status.json under {workdir}")
    refs_path = _first_existing(workdir, REFERENCES_PATHS)
    if refs_path is None:
        raise FileNotFoundError(f"no references.json under {workdir}")
    return _read(status_path, {}) or {}, _read(refs_path, {}) or {}


def trace_of(ref_row: dict) -> dict[str, Any]:
    """run id -> trace id for one Task's References and recordings."""
    runs = (ref_row.get("references") or []) + (ref_row.get("recordings") or [])
    return {str(r.get("run_id")): r.get("trace_id") for r in runs if r.get("run_id")}


def rewards_for(
    ref_ids: list[str], traces: dict[str, Any], reward_of: dict[str, float]
) -> tuple[list[tuple[str, float]], int]:
    """(scored (run, reward) pairs, References with no sidecar reward)."""
    scored = [(rid, reward_of[tid]) for rid in ref_ids
              if (tid := traces.get(rid)) is not None and tid in reward_of]
    return scored, len(ref_ids) - len(scored)


def row_for(
    task_id: str, status: dict, refs: dict, reward_of: dict[str, float]
) -> tuple[str, dict]:
    """(class, row) for one trusted Task, its References scored off the sidecars.

    Any Reference without a sidecar reward leaves the Task unknown: a partial
    score never counts as right, wrong or mixed.
    """
    row = status.get(task_id) or {}
    ref_ids = [str(r) for r in (row.get("reference_run_ids") or [])]
    scored, unscored = rewards_for(ref_ids, trace_of(refs.get(task_id) or {}), reward_of)
    if unscored:
        verdict = UNKNOWN
    else:
        verdict = classify([r for _, r in scored])
    return verdict, {"task_id": task_id, "class": verdict,
                     "reference_runs": ref_ids,
                     "rewards": [{"run_id": rid, "reward": r} for rid, r in scored],
                     "unscored_references": unscored}


def measure(
    workdir: Path,
    sidecar_dir: str = SIDECAR_DIR,
    reward_field: str = REWARD_FIELD,
    traces_dir: str = TRACES_DIR,
) -> dict:
    chosen, trusted, refused = trusted_from_round(workdir)
    status, refs = load_records(workdir)

    sidecars = workdir / sidecar_dir if not Path(sidecar_dir).is_absolute() else Path(sidecar_dir)
    traces = workdir / traces_dir if not Path(traces_dir).is_absolute() else Path(traces_dir)
    reward_of, sidecar_files = sidecar_rewards(sidecars, reward_field, traces)

    classes: dict[str, list[str]] = {c: [] for c in CLASSES}
    rows = []
    for task_id in trusted:
        verdict, row = row_for(task_id, status, refs, reward_of)
        classes[verdict].append(task_id)
        rows.append(row)
    scored = len(classes[RIGHT]) + len(classes[WRONG]) + len(classes[MIXED])
    return {"workdir": str(workdir), "round": chosen, "source": f"rounds/{chosen}/tasks.json",
            "trusted": trusted, "trusted_count": len(trusted),
            "classes": classes,
            "wrong_share_scored": (len(classes[WRONG]) / scored) if scored else None,
            "wrong_share_trusted": (len(classes[WRONG]) / len(trusted)) if trusted else None,
            "scored_count": scored,
            "rows": rows,
            "refused": refused, "refused_count": len(refused),
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
              f"refused: {result['refused_count']}"
              + (f"  {', '.join(result['refused'])}" if result["refused"] else "")]
    return "\n".join(lines) + "\n"


def source_word(result: dict) -> str:
    return f"round {result['round']}, {result['source']}"


def _inside(child: Path, parent: Path) -> bool:
    """Whether one path sits inside another, following links on both sides."""
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--sidecar-dir", default=SIDECAR_DIR,
                        help="Benchmark sidecars, absolute or under the workdir "
                             f"(default: {SIDECAR_DIR}).")
    parser.add_argument("--reward-field", default=REWARD_FIELD,
                        help="Dotted reward field inside each sidecar's fields "
                             f"(default: {REWARD_FIELD}).")
    parser.add_argument("--traces-dir", default=TRACES_DIR,
                        help="Recordings, for sidecars that carry no trace id "
                             f"(default: {TRACES_DIR}).")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the machine-readable result here (outside the workdir).")
    args = parser.parse_args(argv)
    if args.json is not None and _inside(args.json, args.workdir):
        print(f"refusing to write {args.json} inside the workdir")
        return 2
    try:
        result = measure(args.workdir, args.sidecar_dir,
                         args.reward_field, args.traces_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    print(report(result), end="")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
