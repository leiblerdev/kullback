"""Pre-generate one scenario per (cluster, level): static input for workers."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import synthesize as synth_mod

outdir = Path(sys.argv[1])
levels = ("easy", "medium", "hard")
clusters = json.loads((outdir / "clusters.json").read_text())
evidence = json.loads((outdir / "evidence.json").read_text())
scenarios = {}
spent = 0.0
try:
    old = json.loads((outdir / "scenarios.json").read_text())
except (OSError, ValueError):
    old = {}
model = synth_mod.make_synth(outdir)
for entry in clusters:
    for level in levels:
        key = f"{entry['skill']}::{level}"
        if key in old and old[key].get("scenario"):
            scenarios[key] = old[key]
            if "write_calls" not in scenarios[key]:
                calls = old[key].get("write_call", "")
                scenarios[key]["write_calls"] = [calls] if calls else []
            spent += old[key].get("cost_usd", 0.0)
            continue
        scenario, cost = synth_mod.scenario_for(
            model, entry["skill"], entry.get("evidence", ""), level)
        spent += cost
        scenarios[key] = {"scenario": scenario,
                          "write_calls": synth_mod.cluster_writes(entry, evidence),
                          "cost_usd": cost}
        (outdir / "scenarios.json").write_text(json.dumps(scenarios, indent=1))
        print(f"{key}: {'ok' if scenario else 'EMPTY'} spent {spent:.3f}", flush=True)
print(f"scenarios: {len(scenarios)} spent {spent:.3f}")
