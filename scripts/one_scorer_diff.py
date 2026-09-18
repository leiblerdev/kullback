"""One-scorer differential (G3 acceptance): the Verdict against check_run on stored Runs.

Takes workdir paths as arguments, no corpus names. Read only, no model calls, nothing
written into any workdir. For every stored Run of every Task it shows:

- atom disagreements between the patched Verdict and check_run on every non-judge atom,
  with Hard atoms agreeing as defect == defect (hard_holds None against a raising
  predicate); non-Hard atoms must show zero disagreements by construction;
- the Verdicts that changed against the predicate-based (origin/main) reading, which must
  be the known flips plus only what the Hard-defect unification explains.

Usage:
    PYTHONPATH=<worktree> <venv>/bin/python scripts/one_scorer_diff.py <workdir> ...
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from kullback.examiner import lifecycle
from kullback.report.load import load_tool_sigs, run_from_jsonl
from kullback.runner import target as T
from kullback.runner.atom_context import AtomContext, _evaluate, gate
from kullback.runner.boundary import runner_version as compute_runner_version
from kullback.runner.canon import Unresolved, load_rules
from kullback.runner.records import EntitySchema, Environment, Task, Verifier
from kullback.runner.verdict import MUST_HOLD, verdict


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load(path: Path, model: type) -> Any:
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _old_predicate_decisions(verifier: Verifier, atoms: list, context: AtomContext) -> dict:
    """The origin/main reading: every non-judge atom off its predicate source."""
    out: dict = {}
    for atom in sorted(atoms, key=lambda a: a.kind == "hard"):
        holds: Optional[bool] = None
        refused = gate(atom.predicate_src) if atom.predicate_src else []
        if not atom.predicate_src:
            out[atom.id] = ("unevaluable", "no_predicate_src")
            continue
        if refused:
            out[atom.id] = ("unevaluable", f"gate_refused:{refused[0]}")
            continue
        context.marking = atom.kind != "forbidden"
        try:
            holds = _evaluate(atom.predicate_src, context.env())
        except Unresolved:
            out[atom.id] = ("unevaluable", "unresolved_pair")
            context.marking = True
            continue
        except Exception as error:
            out[atom.id] = ("error", type(error).__name__)
            context.marking = True
            continue
        finally:
            context.marking = True
        if holds is None:
            out[atom.id] = ("unevaluable", "no_answer")
        elif atom.kind in MUST_HOLD and not holds:
            out[atom.id] = ("fail", None)
        elif atom.kind == "forbidden" and holds:
            out[atom.id] = ("fail", None)
        else:
            out[atom.id] = ("pass", None)
    return out


def _old_verdict_state(atoms: list, decisions: dict, context: AtomContext) -> str:
    order = {atom.id: i for i, atom in enumerate(atoms)}
    states = [(order[a.id], decisions.get(a.id, ("unevaluable", "missing"))[0]) for a in atoms]
    if any(state == "fail" for _, state in states):
        return "fail"
    if any(state in ("unevaluable", "error") for _, state in states):
        return "not_verdicted"
    if context.write_tools is not None and context.extra_writes():
        return "fail"
    return "pass"


def measure(workdir: Path) -> dict:
    env_path = workdir / "environment.json"
    environment = _load(env_path, Environment) if env_path.is_file() else None
    schema_path = workdir / "schema.json"
    schema = _load(schema_path, EntitySchema) if schema_path.is_file() else None
    sigs = load_tool_sigs(workdir)
    write_tools = {s.name for s in sigs if s.kind == "write"} or None
    rules_path = workdir / "canon-rules.json"
    rules = load_rules(rules_path) if rules_path.is_file() else None

    tasks = [_load(p, Task) for p in sorted((workdir / "tasks").glob("*.json"))]
    status = _json(workdir / "task_status.json") or {}
    all_verifiers = []
    for path in sorted((workdir / "verifiers").glob("*.json")):
        try:
            all_verifiers.append(_load(path, Verifier))
        except Exception:
            pass
    live_ids = {v.task_id for v in lifecycle.live(all_verifiers, status)}
    by_task = {v.task_id: v for v in all_verifiers}

    atom_cells: Counter = Counter()
    new_hard_defects: list = []
    flips: list = []
    runs_scored = 0
    pairs = 0
    nonhard_pairs = 0
    nonhard_disagreements = 0
    judge_skipped = 0

    for task in tasks:
        verifier = by_task.get(task.id)
        if verifier is None:
            continue
        atoms = [a for a in verifier.atoms if not a.judge]
        judge_skipped += len(verifier.atoms) - len(atoms)
        if not atoms:
            continue
        paths = sorted((workdir / "runs" / task.id).glob("*.jsonl"))
        probe = workdir / "probes" / f"probe-{task.id}.jsonl"
        if probe.is_file():
            paths.append(probe)
        for path in paths:
            run = run_from_jsonl(path)
            if run is None:
                continue
            runs_scored += 1
            fn = T.canon_fn(rules)
            tools = T.scored_write_tools(verifier, run, write_tools)
            effects = T.write_effects(run, tools, fn)
            asked = set(T.question_keys(run, effects, fn))
            said = set(T.communicate_values(run, fn))
            context = AtomContext(run, fn, write_tools, schema, rules=rules)
            old = _old_predicate_decisions(verifier, atoms, context)
            nonjudge = verifier.model_copy(update={"atoms": atoms})
            for atom in atoms:
                pairs += 1
                if atom.kind == "hard":
                    new_side = T.hard_holds(atom, run, tools, fn)
                    old_state, _ = old.get(atom.id, ("unevaluable", "missing"))
                    old_side = ("defect" if old_state in ("error", "unevaluable") else old_state)
                    if new_side is None and old_side == "pass":
                        atom_cells[("hard:pass", "hard:pass")] += 1
                        continue
                    new_state = ("defect" if new_side is None
                                 else ("pass" if new_side else "fail"))
                    atom_cells[(f"hard:{new_state}", f"hard:{old_side}")] += 1
                    if new_side is None:
                        new_hard_defects.append({"task_id": task.id, "atom_id": atom.id,
                                                 "old": old_side})
                    continue
                nonhard_pairs += 1
                new_holds = T.atom_holds(atom, run, fn, tools,
                                         effects=effects, asked=asked, said=said)
                if new_holds is None:
                    continue
                new_state = "pass" if new_holds else "fail"
                one = verifier.model_copy(update={"atoms": [atom]})
                try:
                    passed, failing = T.check_run(one, run, rules, write_tools=write_tools)
                    target_state = "pass" if (passed or failing != atom.id) else "fail"
                except Exception as error:
                    target_state = f"error:{type(error).__name__}"
                if new_state != target_state:
                    nonhard_disagreements += 1
                atom_cells[(f"nonhard:{new_state}", f"target:{target_state}")] += 1
            new_record = verdict(run, nonjudge, rules, write_tools=write_tools, schema=schema,
                                 rules=rules, environment=environment)
            new_state = ("pass" if new_record.passed else
                         ("not_verdicted" if new_record.class_ == "not_verdicted" else "fail"))
            old_state = _old_verdict_state(atoms, old, context)
            if new_state != old_state:
                flips.append({"task_id": task.id, "file": path.name,
                              "direction": f"{old_state}_to_{new_state}",
                              "new_failing_atom": new_record.failing_atom})

    return {
        "workdir": workdir.name,
        "verifiers_live": len(live_ids),
        "runs_scored": runs_scored,
        "judge_atoms_skipped": judge_skipped,
        "atom_run_pairs": pairs,
        "nonhard_pairs": nonhard_pairs,
        "nonhard_disagreements": nonhard_disagreements,
        "atom_cells": {f"{a}/{b}": n for (a, b), n in sorted(atom_cells.items())},
        "hard_defects": len(new_hard_defects),
        "verdict_flips": flips,
    }


def main(argv: list[str]) -> int:
    out: dict = {"code_runner_version": compute_runner_version(
        Path(__import__("kullback").__file__).parent).runner_version,
        "code_path": str(Path(__import__("kullback").__file__).parent),
        "corpora": []}
    for name in [a for a in argv if not a.startswith("--")]:
        workdir = Path(name)
        try:
            out["corpora"].append(measure(workdir))
        except Exception:
            out["corpora"].append({"workdir": workdir.name, "fatal": traceback.format_exc()})
    print(json.dumps(out, indent=1, sort_keys=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
