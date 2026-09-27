"""Judge evidence indices [start, end) into a part file. Merge parts after."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import judge as judge_mod

workdir = Path(sys.argv[1])
outdir = Path(sys.argv[2])
start, end = int(sys.argv[3]), int(sys.argv[4])
part = Path(sys.argv[5])
evidence = json.loads((outdir / "evidence.json").read_text())
model = judge_mod.make_model()
reader = judge_mod.Reader(workdir, evidence)
rows = []
spent = 0.0
for item in evidence[start:end]:
    if spent >= 20.0:
        break
    row = judge_mod.judge_one(model, reader, item)
    spent += row.get("cost_usd", 0.0)
    rows.append(row)
    part.write_text(json.dumps(rows, indent=1))
    print(f"{part.name} {len(rows)}/{end - start} {row['cause']} "
          f"spent {spent:.2f} USD", flush=True)
print(f"{part.name} done {len(rows)}/{end - start} spent {spent:.2f} USD")
