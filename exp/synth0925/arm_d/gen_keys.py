"""Generate a fixed list of scenario keys. Usage: gen_keys.py <outdir> <keys.json> <part.json> [clustertag]."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import synthesize as synth_mod


def main() -> None:
    outdir = Path(sys.argv[1])
    keys = json.loads(Path(sys.argv[2]).read_text())
    part = Path(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    clusters = {entry["skill"]: entry
                for entry in json.loads((outdir / f"clusters{tag}.json").read_text())}
    evidence = json.loads((outdir / "evidence.json").read_text())
    model = synth_mod.make_synth(outdir)
    spent = 0.0
    for key in keys:
        skill, _, level = key.partition("::")
        scenario, cost = synth_mod.scenario_for(
            model, skill, clusters[skill].get("evidence", ""), level)
        spent += cost
        try:
            body = json.loads(part.read_text())
        except (OSError, ValueError):
            body = {}
        body[key] = {"scenario": scenario,
                     "write_calls": synth_mod.cluster_writes(clusters[skill], evidence),
                     "cost_usd": cost}
        part.write_text(json.dumps(body, indent=1))
        print(f"{key}: {'ok' if scenario else 'EMPTY'} spent {spent:.3f}", flush=True)


if __name__ == "__main__":
    main()
