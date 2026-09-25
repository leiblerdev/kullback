"""Generate a fixed list of scenario keys. Usage: gen_keys.py <outdir> <keys.json>."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import synthesize as synth_mod

outdir = Path(sys.argv[1])
keys = json.loads(Path(sys.argv[2]).read_text())
clusters = {e["skill"]: e
            for e in json.loads((outdir / "clusters.json").read_text())}
evidence = json.loads((outdir / "evidence.json").read_text())
model = synth_mod.make_synth(outdir)
spent = 0.0
for key in keys:
    skill, _, level = key.partition("::")
    scenario, cost = synth_mod.scenario_for(
        model, skill, clusters[skill].get("evidence", ""), level)
    spent += cost
    part = Path(sys.argv[3])
    try:
        body = json.loads(part.read_text())
    except (OSError, ValueError):
        body = {}
    body[key] = {"scenario": scenario,
                 "write_calls": synth_mod.cluster_writes(clusters[skill], evidence),
                 "cost_usd": cost}
    part.write_text(json.dumps(body, indent=1))
    print(f"{key}: {'ok' if scenario else 'EMPTY'} spent {spent:.3f}", flush=True)
