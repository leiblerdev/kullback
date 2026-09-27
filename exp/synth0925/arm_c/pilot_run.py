"""Pilot / full driver for the fixed arm C loop: anchor facts, coherent binding.

Usage: pilot_run.py <limit> <tag>  (e.g. 10 r1pilot, then 60 r1full).
Appends to tasks.jsonl from the first missing slot. Fresh line ids via ID_SALT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exp.synth0925.arm_c import run_arm_c as R
from exp.synth0925.arm_c.cost import JUDGE_ID, SYNTH_ID, Priced, Share


def main() -> None:
    if "," in (sys.argv[1] if len(sys.argv) > 1 else ""):
        slotspec, out_name, tag = sys.argv[1], sys.argv[2], sys.argv[3]
        t0 = time.time()
        R.ID_SALT = ":r1"
        R.TASKS_PATH = R.OUT / out_name
        if R.TASKS_PATH.exists():
            R.TASKS_PATH.unlink()
        R.ATTEMPTS_PATH = R.OUT / f"attempts-{tag}.jsonl"
        if R.ATTEMPTS_PATH.exists():
            R.ATTEMPTS_PATH.unlink()
        share = Share(R.CEIL)
        synth = Priced(SYNTH_ID, R.CEIL)
        judge = Priced(JUDGE_ID, R.CEIL)
        from kullback import synthesise as S
        world = S.World(R.WORK)
        done = 0
        for slot in [int(x) for x in slotspec.split(",")]:
            share.refuse_if_broke()
            if R.run_slot(world, synth, judge, share, slot) is None:
                break
            done += 1
        wall = time.time() - t0
        state = {"model_usd": share.model_usd, "search_usd": share.search_usd,
                 "model_calls": share.model_calls, "searches": share.searches,
                 "fetches": share.fetches, "wall_s": wall}
        import json
        (R.OUT / f"spend-{tag}.json").write_text(json.dumps(state, indent=1))
        print(f"slice {tag}: {done} slots in {wall:.0f}s, model {share.model_usd:.4f} USD",
              flush=True)
        return
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tag = sys.argv[2] if len(sys.argv) > 2 else "r1"
    t0 = time.time()
    R.ID_SALT = ":r1"
    R.ATTEMPTS_PATH = R.OUT / f"attempts-{tag}.jsonl"
    share = Share(R.CEIL)
    synth = Priced(SYNTH_ID, R.CEIL)
    judge = Priced(JUDGE_ID, R.CEIL)
    rows = R.phase_tasks(share, synth, judge, limit)
    wall = time.time() - t0
    state = {"model_usd": share.model_usd, "search_usd": share.search_usd,
             "model_calls": share.model_calls, "searches": share.searches,
             "fetches": share.fetches, "wall_s": wall}
    import json
    (R.OUT / f"spend-{tag}.json").write_text(json.dumps(state, indent=1))
    from collections import Counter
    print(f"pilot {tag}: {len(rows)} rows, " +
          str(dict(Counter((r['level'], r['status']) for r in rows))), flush=True)
    print(f"done in {wall:.0f}s, model {share.model_usd:.4f} USD, calls {share.model_calls}",
          flush=True)


if __name__ == "__main__":
    main()
