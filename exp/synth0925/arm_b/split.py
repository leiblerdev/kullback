"""Split multi-row templates into per-row units and retarget each."""
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
    ts = {t["source_task_id"]: t for t in d["templates"]}
    idx = units.trace_index(workdir)
    y = json.loads((od / "yields.json").read_text())
    have = {(u["source_task_id"], tuple(sorted(u["rows"]))) for u in y["do_units"]}
    multis = [f for f in y["do_fail"] if f["why"] == "multi-row-template"]
    print(f"multi-row templates: {len(multis)}", flush=True)
    added, failed = [], []
    for m in multis:
        t = ts[m["source"]]
        calls = units.trace_calls(workdir, idx, t["source_task_id"]) or []
        t = dict(t)
        t["read_results"] = [c.get("result") for c in calls
                             if c.get("error") is None and c.get("name") not in d["write_tools"]]
        groups: dict[str, list] = {}
        for w in t["writes"]:
            groups.setdefault(str(w["primary"]["row"]), []).append(w)
        for row, ws in sorted(groups.items()):
            sub = dict(t, writes=ws)
            try:
                u = retarget.retarget_do(bind, world, sub, f"split:{m['source']}:{row}")
            except Exception as e:  # noqa: BLE001 - one bad group must not stop mining
                print("split error", m["source"], row, type(e).__name__, str(e)[:100])
                continue
            if u and (u["source_task_id"], tuple(sorted(u["rows"]))) not in have:
                u["split_from"] = row
                added.append(u)
                have.add((u["source_task_id"], tuple(sorted(u["rows"]))))
            elif not u:
                failed.append({"source": m["source"], "why": f"split-no-row:{row}"})
    print(f"split added={len(added)} failed={len(failed)}", flush=True)
    y["do_units"].extend(added)
    y["do_fail"].extend(failed)
    (od / "yields.json").write_text(json.dumps(y, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else R,
         Path(sys.argv[2]) if len(sys.argv) > 2 else None)
