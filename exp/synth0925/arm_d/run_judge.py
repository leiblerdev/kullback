"""Run the agentic judge over every evidence bundle, saving incrementally."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import judge as judge_mod

workdir = Path(sys.argv[1])
outdir = Path(sys.argv[2])
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
evidence = json.loads((outdir / "evidence.json").read_text(encoding="utf-8"))
done = {}
if (outdir / "judge_results.json").is_file():
    try:
        for row in json.loads((outdir / "judge_results.json").read_text()):
            done[row["run_id"]] = row
    except ValueError:
        pass
model = judge_mod.make_model()
reader = judge_mod.Reader(workdir, evidence)
rows = []
spent = 0.0
for item in evidence:
    if item["run_id"] in done:
        rows.append(done[item["run_id"]])
        spent += done[item["run_id"]].get("cost_usd", 0.0)
        continue
    if spent >= 20.0:
        break
    if len(rows) - len(done) >= max_new:
        break
    row = judge_mod.judge_one(model, reader, item)
    spent += row.get("cost_usd", 0.0)
    rows.append(row)
    (outdir / "judge_results.json").write_text(json.dumps(rows, indent=1))
    print(f"judged {len(rows)}/{len(evidence)} {row['cause']} "
          f"spent {spent:.2f} USD", flush=True)
(outdir / "judge_results.json").write_text(json.dumps(rows, indent=1))
print(f"done {len(rows)}/{len(evidence)} spent {spent:.2f} USD")
