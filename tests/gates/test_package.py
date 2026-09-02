"""What the gates package is: no model call, no agent, no Builder, and one registry naming every gate (D122)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import kullback.gates as gates
from kullback.runner.records import GateResult

GATES_DIR = Path(gates.__file__).resolve().parent
FORBIDDEN_PACKAGES = ("kullback.ai", "kullback.agent", "kullback.builder", "kullback.examiner",
                      "kullback.cli", "kullback.tui", "kullback.report")
GATE_FILES = sorted(GATES_DIR.rglob("*.py"))


def _imports(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            out += [base] + [f"{base}.{alias.name}" for alias in node.names]
    return out


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_no_gate_module_imports_the_provider_layer_or_an_agent(path: Path):
    """D122: a gate is code no agent can write and no model is consulted for; the import list is
    the first place that would show, and lint-imports checks the same line at the package level."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for name in _imports(tree):
        for forbidden in FORBIDDEN_PACKAGES:
            assert not (name == forbidden or name.startswith(forbidden + ".")), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_no_gate_module_calls_a_model(path: Path):
    """`query(` is how every Model in kullback.ai is asked; nothing under gates/ says it, in any form."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        assert callee not in ("query", "stream", "complete"), f"{path.name} line {node.lineno} calls {callee}()"
    source = path.read_text(encoding="utf-8")
    assert "query(" not in source, f"{path.name} names a model call"


def test_the_registry_names_every_gate_once_and_every_one_lives_in_the_package():
    names = [spec.name for spec in gates.GATES]
    assert len(names) == len(set(names))
    for spec in gates.GATES:
        module = inspect.getmodule(spec.fn)
        assert module is not None and module.__name__.startswith("kullback.gates."), spec.name
        # A gate rules under its own name, or it is a sequence and names every stage it may return,
        # or its name is dotted because the plain one is taken and the stem is what it rules under.
        assert spec.rulings and (spec.name in spec.rulings or len(spec.rulings) > 1
                                 or spec.name.split(".", 1)[0] in spec.rulings), spec
        assert all(isinstance(a, str) and a for a in spec.artifacts), spec
        assert spec.over
    assert gates.gate_named("scorecard").fn is gates.scorecard_gate
    with pytest.raises(KeyError):
        gates.gate_named("no-such-gate")


def test_the_registry_covers_every_stage_the_gates_package_rules_under():
    """Every stage name a function in this package returns is registered, so a later phase can run
    'every gate' generically and a report can find the gate behind one of these stage names."""
    rulings = {stage for spec in gates.GATES for stage in spec.rulings}
    for stage in ("ingest", "mine", "compile_tools.parses", "compile_tools.replay_fidelity", "compile_policy",
                  "build_environment", "build_user_rules", "replay_reference", "derive_verifier", "leak_check",
                  "budget", "scorecard", "confined", "verifier_oracle", "verifier_unfinished_run"):
        assert stage in rulings, stage
    assert set(gates.D79_STAGES) <= rulings


def _stage_literals(path: Path, callees: tuple[str, ...]) -> set[str]:
    """Every stage name a ruling written in this file rules under.

    Two shapes: the `gate("...")` and `_gate("...")` helpers, whose first argument is the name, and a
    `GateResult(stage="...")` built by hand, which the helpers' scan would walk straight past.
    """
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if callee in callees and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.add(first.value)
        if callee == "GateResult":
            for keyword in node.keywords:
                if keyword.arg == "stage" and isinstance(keyword.value, ast.Constant) \
                        and isinstance(keyword.value.value, str):
                    out.add(keyword.value.value)
    return out


def test_the_stages_the_build_records_outside_the_registry_are_none():
    """Every stage in a build's gates.json is a registered gate. Until phase 4 twelve were not: the
    six sandbox rulings lived in builder/sandbox.py beside the subprocess they read, and six more
    were rulings build.py made inline with gate_support.gate(). The sandbox rulings are now
    functions over what the subprocess returned (gates/tool_runs.py) and the inline ones functions
    over the stage's artifact (gates/stages.py), so a new inline ruling cannot appear in the Builder
    without a registered gate behind it, and the tool_result hook can run any of them. The scan
    covers every module of the Builder, not only the two the rulings left, because a ruling can be
    written anywhere in the package; what a whole build actually leaves on disk is pinned in
    tests/builder/test_extension.py, which is the same claim measured rather than read."""
    import kullback.builder as builder

    builder_dir = Path(builder.__file__).resolve().parent
    recorded: set[str] = set()
    for path in sorted(builder_dir.rglob("*.py")):
        recorded |= _stage_literals(path, ("gate", "_gate"))
    registered = {stage for spec in gates.GATES for stage in spec.rulings}
    assert recorded - registered == set()
    for stage in ("cluster", "compile_tools", "intent", "rerolls", "tau2_export", "vocabulary",
                  "parses", "executes_on_s0", "deterministic", "non_trivial", "replay_fidelity", "refuses_unknown"):
        assert stage in registered, stage


def test_the_artifact_bindings_name_artifacts_the_build_declares():
    """A gate bound to an artifact is run by the hook over the pipeline's store, so the names have
    to be ones a stage in build.py releases; `gates_over` finds the gates for one artifact."""
    produced = set()
    for spec in gates.GATES:
        produced.update(spec.artifacts)
    declared = {"traces", "sigs", "schema", "categories", "tasks", "canon_rules", "db", "overlays", "assumptions",
                "synthetic_rows", "bodies", "assisted_tools", "constraints", "policy_text", "lessons_applied",
                "lessons_set_aside", "intents", "vocabulary", "user_rules", "environment", "replays", "rerolls",
                "verifiers", "task_status"}
    assert produced <= declared, produced - declared
    assert [spec.name for spec in gates.gates_over("tasks")] == ["cluster"]
    assert gates.gates_over("no-such-artifact") == ()


def test_every_gate_returns_the_one_ruling_record():
    """A gate over nothing is still a ruling, stage and all, so a hook can log it unchanged."""
    over_nothing = {
        "ingest": ([],), "mine": ([], []), "compile_tools": ({},), "compile_policy": ([],),
        "build_user_rules": ([],), "replay_reference": ({},), "gate_a_oracle_replay": ([],),
        "derive_verifier": ({},), "verdict": ({},), "candidate_runs": ([],), "budget": ({"stages": {}},),
        "regrade": ([],), "confined": ("",), "compile_policy.confined": ("def check(case):\n    return True\n",),
        "cluster": ([],), "compile_tools.bodies": ({},), "intent": ({},), "vocabulary": ({},),
        "tau2_export": ([],), "rerolls": ({}, 3), "derive_verifier.tasks": ({},),
        "parses": ("",), "executes_on_s0": ([], []), "deterministic": ([], [], []), "non_trivial": ([], []),
        "refuses_unknown": ([], []),
    }
    for name, args in over_nothing.items():
        out = gates.gate_named(name).fn(*args)
        results = out if isinstance(out, list) else [out]
        assert results, name
        for result in results:
            assert isinstance(result, GateResult), name
            assert result.stage in gates.gate_named(name).rulings, (name, result.stage)
