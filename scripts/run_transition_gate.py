"""Grade every stored write body through run_gates with the transition gate on.

Code only, no model calls. Each workdir argument is opened read only; the sandboxes run
under a scratch directory given by --scratch (never inside the workdir). Prints per tool
the chain outcome, the transition metrics, and the transition gate's own verdict beside
the chain's (stored bodies usually fall at an earlier gate, which says nothing about what
the transition gate rules on them):

    uv run python scripts/run_transition_gate.py --scratch <dir> <workdir> [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transition_common import derive_evidence, derive_worlds, load_basics

from kullback.builder import compile_env
from kullback.builder import effects as effects_mod
from kullback.builder import sandbox as sandbox_mod


def grade_tool(basics: dict, maps: dict, workdir: Path, scratch: Path, tool: str,
               calls: list, observed: dict) -> dict:
    """One write tool through the chain and through the transition gate on its own."""
    evidence = effects_mod.replay_evidence({tool: observed.get(tool, [])})
    source = compile_env.module_source(basics["schema"], [basics["by_name"][tool]],
                                       {tool: basics["bodies"][tool]})
    box = sandbox_mod.Sandbox(source, basics["db"], scratch / workdir.name / tool,
                              call_states=maps["states"], call_tasks=maps["call_tasks"],
                              timeout=600.0)
    gates = sandbox_mod.run_gates(source, box, calls, [], basics["schema"],
                                  rules=basics["rules"], sig=basics["by_name"][tool],
                                  transition_evidence=evidence)
    failed = [gate for gate in gates if not gate.passed]
    transition = next((gate for gate in gates
                       if gate.stage == sandbox_mod.TRANSITION_STAGE), None)
    alone = sandbox_mod.gate_transition(box, calls, basics["schema"], evidence,
                                        rules=basics["rules"])
    return {"refused": bool(failed),
            "refused_by": failed[0].stage if failed else None,
            "transition": dict(transition.metrics) if transition is not None else {},
            "alone": {"passed": alone.passed, **dict(alone.metrics)},
            "first_failure": failed[0].failures[:1] if failed else []}


def grade_workdir(workdir: Path, scratch: Path) -> dict:
    basics = load_basics(workdir)
    maps = derive_worlds(basics)
    shown = derive_evidence(workdir, basics, maps["worlds"])
    out: dict[str, dict] = {}
    for tool in sorted(shown["calls_by_tool"]):
        if tool not in basics["writes"]:
            continue
        calls = [call for call in shown["calls_by_tool"][tool] if basics["bodies"].get(tool)]
        if not calls:
            continue
        out[tool] = grade_tool(basics, maps, workdir, scratch, tool, calls,
                               shown["observed"])
    return {"workdir": str(workdir), "per_tool": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True)
    parser.add_argument("workdirs", nargs="+")
    args = parser.parse_args(argv)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    refused_total = 0
    for raw in args.workdirs:
        result = grade_workdir(Path(raw), scratch)
        print(f"== {result['workdir']}")
        for tool in sorted(result["per_tool"]):
            row = result["per_tool"][tool]
            print(f"  {tool}: refused={row['refused']} refused_by={row['refused_by']} "
                  f"transition={row['transition']} alone={row['alone']}")
            if not row["alone"]["passed"]:
                refused_total += 1
    print(f"REFUSED_BY_TRANSITION_TOOLS={refused_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
