"""Judge items on stored Runs: the Reference's and the fresh Runs of each Task (D328 calibration).

    PYTHONPATH=$PWD uv run python scripts/judge_calibrate.py ITEMS RUNS_ROOT OUT --model ID [--rulings FILE]

ITEMS maps "<corpus>/<task_id>" to a list of items {id, question, anchor, evidence, gate?, weight?}.
RUNS_ROOT/<corpus>/runs holds <task_id>/replay-*.jsonl (the Reference) and <task_id>-<n>*.jsonl
(fresh Runs). RULINGS, when given, maps "<corpus>/<task_id>/<item_id>/<run_name>" to "met" or
"not_met" from the Examiner; agreement is counted only where a ruling exists. Nothing here decides
training: it prints the score distribution, the why strings, the truncated count, agreement and
spend, and writes one JSONL row per judged item to OUT.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from kullback.ai.provider import live_model
from kullback.judge.items import judge_item, tally
from kullback.runner.records import Atom
from kullback.runner.verdict import load_run


def runs_of(root: Path, key: str) -> list[tuple[str, Path]]:
    """(name, path) for the Reference Run and every fresh Run of one Task."""
    corpus, task = key.split("/", 1)
    folder = root / corpus / "runs"
    found = [("reference", p) for p in sorted((folder / task).glob("replay-*.jsonl"))[:1]]
    found += [(p.stem.removeprefix(task + "-"), p) for p in sorted(folder.glob(f"{task}-*.jsonl"))]
    return found


def atom_of(item: dict) -> Atom:
    target = {k: item[k] for k in ("question", "anchor", "evidence") if k in item}
    return Atom(id=item["id"], kind="communicate", judge=True, target=target,
                gate=bool(item.get("gate", False)), weight=float(item.get("weight", 1.0)))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("items")
    parser.add_argument("runs_root")
    parser.add_argument("out")
    parser.add_argument("--model", required=True, help="provider/model id, priced by runner/budget.py")
    parser.add_argument("--rulings")
    args = parser.parse_args(argv)
    items = {k: v for k, v in json.loads(Path(args.items).read_text()).items() if not k.startswith("_")}
    rulings = json.loads(Path(args.rulings).read_text()) if args.rulings else {}
    model = live_model(args.model)
    results, agree = [], Counter()
    with Path(args.out).open("w", encoding="utf-8") as out:
        for key, rows in items.items():
            for run_name, path in runs_of(Path(args.runs_root), key):
                run = load_run(path)
                for item in rows:
                    result = judge_item(model, atom_of(item), run, model_id=args.model)
                    results.append(result)
                    ruling = rulings.get(f"{key}/{item['id']}/{run_name}")
                    if ruling is not None and result.score is not None:
                        agree["agree" if (ruling == "met") == (result.score == 1) else "disagree"] += 1
                    row = {"task": key, "run": run_name, **result.model_dump()}
                    out.write(json.dumps(row) + "\n")
                    print(f"{key} {run_name} {item['id']} score={result.score} why={result.why!r}", flush=True)
    summary = tally(results)
    summary["by_run_kind"] = dict(Counter(
        ("reference" if r.startswith("reference") else "fresh", s)
        for r, s in ((json.loads(line)["run"], json.loads(line)["score"])
                     for line in Path(args.out).read_text().splitlines())).most_common())
    summary["agreement_with_rulings"] = dict(agree) or "no Examiner anchor ruling present"
    print(json.dumps({k: (str(v) if k == "by_run_kind" else v) for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
