"""Build one static slice of (cluster, level) tasks. Workers append to their
own jsonl; admitted Tasks store under the shared synthetic/ with distinct ids.
Usage: build_slice.py <workdir> <outdir> <slice.json> <worker> [filetag]
With filetag (e.g. -r1), clusters, scenarios and slice outputs use it."""
import json
import sys
from pathlib import Path

from exp.synth0925.arm_d import bind as bind_mod
from exp.synth0925.arm_d import synthesize as synth_mod
from kullback import synthesise as S
from kullback.gates import verifier_suite


def main() -> None:
    workdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    spec = json.loads(Path(sys.argv[3]).read_text())
    worker = sys.argv[4]
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    run(workdir, outdir, spec, worker, tag=tag)


SEED = "synth0925-d"


def run(workdir, outdir, spec, worker, tag: str = "") -> None:
    clusters = {entry["skill"]: entry
                for entry in json.loads((outdir / f"clusters{tag}.json").read_text())}
    evidence = json.loads((outdir / "evidence.json").read_text())
    scenarios = json.loads((outdir / f"scenarios{tag}.json").read_text())
    world = S.World(workdir)
    body = S.load_graph(workdir)
    observed = body.get("args") or {}
    fn = verifier_suite.canon_fn(world.canon_rules)
    model = synth_mod.make_synth(workdir)
    dest = outdir / f"tasks{tag}-slice-{worker}.jsonl"
    for pos, job in enumerate(spec):
        entry = clusters[job["cluster"]]
        level = job["level"]
        key = f"{entry['skill']}::{level}"
        scenario = scenarios.get(key, {}).get("scenario") or {}
        write_calls = scenarios.get(key, {}).get("write_calls") or []
        excluded = synth_mod.member_rows(entry, evidence)
        run_walk = bind_mod.make_run_walk(body, bind_mod.make_bind(), excluded=excluded)
        ident = f"{SEED}:{worker}:{entry['skill']}:{level}:{pos}"
        line = synth_mod.build_task(world, body, observed, run_walk, entry,
                                    write_calls, level, scenario, model,
                                    ident, fn, SEED, pos)
        with open(dest, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        print(f"{worker} {pos + 1}/{len(spec)}: {level} {line['status']} "
              f"rev {line['revisions']} {(line.get('reasons') or '')[:70]}", flush=True)


if __name__ == "__main__":
    main()
