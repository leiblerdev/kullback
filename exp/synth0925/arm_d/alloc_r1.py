"""Allocate 60 r1 attempts: at least 3 per cluster (one per level), then fill
to 60 by weakness count, levels balanced toward 20/20/20. Writes r1slice-N.json."""
import collections
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
clusters = json.loads((outdir / "clusters-r1.json").read_text())
verdicts = json.loads((outdir / "judge_results.json").read_text())
weak = [r for r in verdicts if r.get("cause") == "candidate_weakness"]
size = {c["skill"]: len([r for r in weak if r["run_id"] in set(c["member_run_ids"])])
        for c in clusters}

jobs = []
for entry in clusters:
    for level in ("easy", "medium", "hard"):
        jobs.append({"cluster": entry["skill"], "level": level})
left = 60 - len(jobs)
order = sorted(clusters, key=lambda c: -size.get(c["skill"], 0))
levels = ("easy", "medium", "hard")
asked = collections.Counter(j["level"] for j in jobs)
idx = 0
while left > 0 and order:
    entry = order[idx % len(order)]
    level = min(levels, key=lambda lv: asked.get(lv, 0))
    jobs.append({"cluster": entry["skill"], "level": level})
    asked[level] += 1
    left -= 1
    idx += 1
print("jobs:", len(jobs), dict(collections.Counter(
    (j["cluster"][:30], j["level"]) for j in jobs)))
per = collections.Counter(j["cluster"] for j in jobs)
print("per cluster:", {k[:30]: v for k, v in per.items()})
for worker in range(nworkers):
    part = jobs[worker::nworkers]
    (outdir / f"r1slice-{worker}.json").write_text(json.dumps(part, indent=0))
    print(worker, len(part), dict(collections.Counter(j["level"] for j in part)))
