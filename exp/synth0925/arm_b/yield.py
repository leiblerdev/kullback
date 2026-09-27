"""Full yield pass: retarget every minable template, collect Decline units."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import retarget
import units

from kullback import synthesise as S

R = units.R
OUT = R.parent


def main(workdir: Path = R, outdir: Path | None = None) -> None:
    od = outdir or OUT
    bind = units.Binding(workdir)
    world = S.World(workdir)
    d = json.loads((od / "units.json").read_text())
    ts = d["templates"]
    idx = units.trace_index(workdir)
    for t in ts:
        calls = units.trace_calls(workdir, idx, t["source_task_id"]) or []
        t["read_results"] = [c.get("result") for c in calls
                             if c.get("error") is None and c.get("name") not in d["write_tools"]]

    do_units, do_fail = [], []
    for i, t in enumerate(ts):
        primaries = {str(w["primary"]["row"]) for w in t["writes"]}
        if len(primaries) != 1:
            do_fail.append({"source": t["source_task_id"], "why": "multi-row-template"})
            continue
        u = retarget.retarget_do(bind, world, t, f"mine:{i}")
        (do_units if u else do_fail).append(
            u if u else {"source": t["source_task_id"], "why": "no-accepting-row"})
        if i % 20 == 0:
            print(f"do {i}/{len(ts)}", flush=True)

    dec_units = []
    for i, t in enumerate(ts):
        if len(t["writes"]) != 1:
            continue
        try:
            ds = retarget.decline_candidates(bind, world, t, f"dec:{i}")
        except Exception as e:  # noqa: BLE001 - one bad template must not stop mining
            print("decline error", t["source_task_id"], type(e).__name__, str(e)[:100])
            continue
        dec_units.extend(ds)
        if i % 20 == 0:
            print(f"dec {i}/{len(ts)}", flush=True)

    out = {"do_units": do_units, "do_fail": do_fail, "decline_units": dec_units}
    (od / "yields.json").write_text(json.dumps(out, indent=1))
    print(f"do_ok={len(do_units)} do_fail={len(do_fail)} decline={len(dec_units)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else R,
         Path(sys.argv[2]) if len(sys.argv) > 2 else None)
