"""A sealed dev/test split of trusted tasks, written once.

Prompts are tuned on dev only; vf_measure.py reports test numbers only with
--final. Stratified by corpus and by Reference right or wrong (plus the mixed
and unknown strata, so the split keeps their shares too).

    uv run python scripts/experiments/vf_split.py split.json --workdir LABEL=PATH \
        [--workdir LABEL=PATH ...] [--seed 7] [--dev-fraction 0.3]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vf_lib as L  # noqa: E402

FORMAT = 1


def parse_workdir(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"--workdir takes LABEL=PATH, got {spec!r}")
    return label, Path(path)


def split_ids(ids: list[str], dev_fraction: float, rng: random.Random) -> tuple[list[str], list[str]]:
    """Shuffled ids into dev and test; a lone id tunes (dev), never tests alone."""
    order = sorted(ids)
    rng.shuffle(order)
    if len(order) < 2:
        return order, []
    n_dev = max(1, min(len(order) - 1, round(len(order) * dev_fraction)))
    return order[:n_dev], order[n_dev:]


def build(corpora: dict[str, Path], seed: int, dev_fraction: float) -> dict:
    """The split body: dev and test ids per corpus, strata kept whole."""
    rng = random.Random(seed)
    dev: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    strata: dict[str, dict[str, int]] = {}
    for label in sorted(corpora):
        workdir = corpora[label]
        trusted = L.trusted_ids(workdir)
        class_of = L.classes(workdir, workdir / "grader")
        stratum: dict[str, list[str]] = {}
        for task_id in trusted:
            stratum.setdefault(class_of.get(task_id, L.UNKNOWN), []).append(task_id)
        strata[label] = {name: len(ids) for name, ids in sorted(stratum.items())}
        dev_ids, test_ids = [], []
        for name in sorted(stratum):
            d, t = split_ids(stratum[name], dev_fraction, rng)
            dev_ids.extend(d)
            test_ids.extend(t)
        dev[label], test[label] = sorted(dev_ids), sorted(test_ids)
    return {"format": FORMAT, "seed": seed, "dev_fraction": dev_fraction,
            "corpora": {label: str(corpora[label]) for label in sorted(corpora)},
            "strata": strata, "dev": dev, "test": test}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="split file to write; refused when it already exists")
    parser.add_argument("--workdir", action="append", default=[],
                        help="LABEL=PATH, one per corpus; trusted ids and classes read off each")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    args = parser.parse_args(argv)
    if not args.workdir:
        parser.error("at least one --workdir LABEL=PATH is required")
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}: the split is written once", file=sys.stderr)
        return 1
    try:
        corpora = dict(parse_workdir(spec) for spec in args.workdir)
    except ValueError as exc:
        parser.error(str(exc))
    for label, workdir in corpora.items():
        if not workdir.is_dir():
            parser.error(f"{label}: {workdir} is not a directory")
    body = build(corpora, args.seed, args.dev_fraction)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
