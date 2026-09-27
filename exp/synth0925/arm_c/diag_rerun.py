"""Diagnose the partial full run: re-run its rejected slots with attempt logging.

Same seed and slot plan, so each slot reuses its scenario and level. Model
sampling is not pinned, so trajectories differ; the mechanism distribution is
what matters. Writes nothing to tasks.jsonl or the synthetic store ids of the
partial run (ID_SALT isolates). Counts and names only, never values.

Usage: diag_rerun.py [slots] [tag]  (e.g. "0,1,2" "a"; default: all rejected, "diag")
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exp.synth0925.arm_c import run_arm_c as R
from exp.synth0925.arm_c.cost import JUDGE_ID, SYNTH_ID, Priced, Share

PARTIAL = R.OUT / "tasks-full-partial.jsonl"


def main() -> None:
    t0 = time.time()
    rows = [json.loads(line) for line in PARTIAL.read_text().splitlines() if line.strip()]
    rej_slots = [i for i, r in enumerate(rows) if r["status"] != "admitted"]
    if len(sys.argv) > 1:
        rej_slots = [int(x) for x in sys.argv[1].split(",") if x.strip() != ""]
        tag = sys.argv[2] if len(sys.argv) > 2 else "x"
    else:
        tag = "diag"
    print(f"partial rows: {len(rows)}, redriving slots: {rej_slots}", flush=True)
    R.ID_SALT = ":diag"
    R.ATTEMPTS_PATH = R.OUT / f"attempts-{tag}.jsonl"
    if R.ATTEMPTS_PATH.exists():
        R.ATTEMPTS_PATH.unlink()
    share = Share(R.CEIL)
    synth = Priced(SYNTH_ID, R.CEIL)
    judge = Priced(JUDGE_ID, R.CEIL)
    scenarios = [json.loads(line) for line in (R.OUT / "scenarios.jsonl").read_text().splitlines()
                 if line.strip()]
    from kullback import synthesise as S
    world = S.World(R.WORK)
    plan = R.slots(60)
    for slot in rej_slots:
        share.refuse_if_broke()
        spec = plan[slot]
        scen = scenarios[slot % len(scenarios)]
        try:
            task = R.build_task(world, synth, judge, share, scen,
                                spec["level"], spec["kind"], slot)
        except RuntimeError as error:
            print(f"slot {slot} stopped: {error}", flush=True)
            break
        print(f"slot {slot} {spec['level']} -> {task['status']} rev {task.get('revisions')}", flush=True)
    print(f"diag {tag} done in {time.time() - t0:.0f}s, model {share.model_usd:.4f} USD, "
          f"calls {share.model_calls}", flush=True)


if __name__ == "__main__":
    main()
