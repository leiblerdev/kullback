"""Builds tests/fixtures/tau2_retail_small.json: the first 3 simulations of a raw tau2 retail file plus the tasks they reference.

Nothing else is changed. Grader fields stay in, so ingest can be tested stripping them.

    uv run python tests/fixtures/make_tau2_retail_small.py \
        ../data/raw/claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json
"""

import json
import sys
from pathlib import Path

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json"
)
N_SIMULATIONS = 3


def build(source: Path, out: Path, n: int = N_SIMULATIONS) -> dict:
    raw = json.loads(source.read_text(encoding="utf-8"))
    simulations = raw["simulations"][:n]
    wanted = {s["task_id"] for s in simulations}
    tasks = [t for t in raw["tasks"] if t["id"] in wanted]
    small = {
        "timestamp": raw["timestamp"],
        "info": raw["info"],
        "tasks": tasks,
        "simulations": simulations,
    }
    out.write_text(json.dumps(small, ensure_ascii=False, indent=1), encoding="utf-8")
    return small


if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SOURCE
    dest = Path(__file__).with_name("tau2_retail_small.json")
    result = build(src, dest)
    print(
        f"{dest}: {len(result['simulations'])} simulations, {len(result['tasks'])} tasks, "
        f"{dest.stat().st_size} bytes, from {src}"
    )
