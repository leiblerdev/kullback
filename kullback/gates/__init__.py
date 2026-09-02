"""Every accept-or-reject check over what the Builder or the Examiner made, and no model call (D122).

A gate is a plain function over an artifact (or two) that returns a `GateResult`: the ruling's
`stage` is the gate's name, `passed` is the ruling, `failures` are the reasons in words a person or
an agent reads, `metrics` the numbers behind it. Nothing here executes a Run, reads a file it was
not handed, calls a model or keeps state between calls, so any gate can later run inside an
agent's `tool_result` hook (phase 4) exactly as the pipeline calls it today. The package is
written by people, hashed per release beside `RunnerVersion` (`gates_version`), importable by both
agents and writable by neither; it imports `kullback.runner` for the records and the primitives
it rules with, and never `kullback.builder`, `kullback.examiner` or `kullback.agent`.

`GATES` lists every gate in the order the build meets them, so a later phase can run "every gate"
generically; `rulings` names the `stage` values one call may return, which for the D79 suite is
nine and for the compile-tools sequence up to five. `artifacts` names, for a gate whose evidence is
whole build artifacts, which ones in argument order: the Builder's tool_result hook (phase 4) looks
a produced artifact up with `gates_over` and runs every gate bound to it over the pipeline's store,
which is how a ruling reaches the model in the tool result without the tool having to know the gate.
A gate over evidence a stage gathers on the way (a sandbox run, a probe) has no binding and is run
by the stage alone.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from kullback.gates import artifacts, confinement, fidelity, scorecard, stages, tool_runs, verifier_suite
from kullback.gates.artifacts import (
    D79_CHECKS,
    GRADER_FIELDS,
    LEAK_MIN_LENGTH,
    TAU2_FILES,
    VERDICT_GOLDEN_CHECKS,
    VERDICT_VERSIONS,
    audit_gate,
    budget_gate,
    candidate_runs_gate,
    compile_tools_gates,
    deterministic_gate,
    environment_gate,
    executes_gate,
    ingest_gate,
    leak_gate,
    mine_gate,
    non_trivial_gate,
    parses_gate,
    policy_gate,
    regrade_gate,
    setup_review_gate,
    user_rules_gate,
    verdict_golden_gate,
    verifier_gate,
)
from kullback.gates.confinement import (
    gate_confined,
    predicate_confinement,
    predicate_confinement_gate,
    source_confinement,
)
from kullback.gates.fidelity import (
    oracle_replay_gate,
    reference_replay_gate,
    replay_fidelity_gate,
    replay_match,
    summarize,
    unconfirmed_reason,
)
from kullback.gates.scorecard import FROZEN_TASKS_NAME, freeze_tasks, task_coverage
from kullback.gates.scorecard import scorecard as scorecard_gate
from kullback.gates.stages import (
    cluster_gate,
    compile_tools_gate,
    intent_gate,
    rerolls_gate,
    task_verifiers_gate,
    tau2_export_gate,
    vocabulary_gate,
)
from kullback.gates.tool_runs import (
    TOOL_RUN_STAGES,
    body_deterministic_gate,
    body_executes_gate,
    body_non_trivial_gate,
    body_parses_gate,
    body_refuses_unknown_gate,
    body_replay_fidelity_gate,
)
from kullback.gates.verifier_suite import (
    D79_STAGES,
    check_run,
    d79_results,
    load_run,
    loophole_probe,
    unfinished_run,
    validate_verifier,
    wrong_run,
)
from kullback.runner.records import GateResult


class GateSpec(NamedTuple):
    """One registered gate: its name, what it rules on, the function, the stages it may return, and
    the build artifacts it takes in argument order when its evidence is whole artifacts."""
    name: str
    over: str
    fn: Callable[..., Any]
    rulings: tuple[str, ...]
    artifacts: tuple[str, ...] = ()


def _spec(name: str, over: str, fn: Callable[..., Any], *rulings: str,
          artifacts: tuple[str, ...] = ()) -> GateSpec:
    """A gate named by its one ruling, or a sequence naming every ruling, or a dotted name whose stem
    is the ruling it returns when the plain name is taken (`compile_tools.bodies` rules under
    `compile_tools`)."""
    return GateSpec(name, over, fn, rulings or (name,), tuple(artifacts))


GATES: tuple[GateSpec, ...] = (
    _spec("ingest", "the Traces after ingest (D66)", ingest_gate, artifacts=("traces",)),
    _spec("mine", "the mined ToolSigs and the calls they rest on", mine_gate),
    _spec("cluster", "the Tasks the Runs were clustered into, each holding at least one Run", cluster_gate,
          artifacts=("tasks", "categories")),
    _spec("confined", "one generated tool body, before it runs anywhere", gate_confined),
    _spec("parses", "one generated tool body as text: is it Python", body_parses_gate),
    _spec("executes_on_s0", "what the sandbox got back running the recorded calls on their Starting states",
          body_executes_gate),
    _spec("deterministic", "two fresh sandbox runs of the same recorded calls (D39)", body_deterministic_gate),
    _spec("non_trivial", "the sandbox's answers across argument sets, against the recorded ones",
          body_non_trivial_gate),
    _spec("replay_fidelity", "the sandbox's answers against the recorded results, column by column (D73, D84)",
          body_replay_fidelity_gate),
    _spec("refuses_unknown", "the sandbox's answers to a reference the world does not hold",
          body_refuses_unknown_gate),
    _spec("compile_tools", "the evidence a compiled tool body produced, five gates in order", compile_tools_gates,
          "compile_tools.parses", "compile_tools.executes", "compile_tools.deterministic",
          "compile_tools.non_trivial", "compile_tools.replay_fidelity"),
    _spec("compile_tools.replay_fidelity", "the recorded calls replayed through a tool body (D80)",
          replay_fidelity_gate),
    _spec("compile_tools.bodies", "every compiled body and the assisted tools; a tool with no body fails the stage",
          compile_tools_gate, "compile_tools", artifacts=("bodies", "assisted_tools")),
    _spec("compile_policy.confined", "one constraint predicate, before it runs anywhere", predicate_confinement_gate),
    _spec("compile_policy", "the compiled Constraints and their tests", policy_gate, artifacts=("constraints",)),
    _spec("intent", "one grounded Intent per Task (D47)", intent_gate, artifacts=("intents",)),
    _spec("vocabulary", "the derived Vocabulary and what the web added to it (D115)", vocabulary_gate,
          artifacts=("vocabulary",)),
    _spec("build_environment", "the Environment's tau2 shape and its db", environment_gate),
    _spec("tau2_export", "the overlay conflicts the one db.json of the tau2 export cannot carry (D74)",
          tau2_export_gate),
    _spec("build_user_rules", "the Simulated user's rules (D44)", user_rules_gate, artifacts=("user_rules",)),
    _spec("replay_reference", "every Trace of every Task replayed through the built tools (D108)",
          reference_replay_gate, artifacts=("replays",)),
    _spec("rerolls", "the re-rolled Runs of the frontier per Task (D112)", rerolls_gate),
    _spec("gate_a_oracle_replay", "a Reference's own calls replayed, seed and held-out apart", oracle_replay_gate),
    _spec("verifier_suite", "one Verifier against its Reference, its re-runs and the probe Runs (D79, D119)",
          validate_verifier, *D79_STAGES),
    _spec("derive_verifier", "the D79 results of one Verifier, every check run and passed", verifier_gate),
    _spec("derive_verifier.tasks", "every Task's status after its Verifier met the D79 suite", task_verifiers_gate,
          "derive_verifier"),
    _spec("leak_check", "what reaches the Candidate, against the Verifier's constants (D89)", leak_gate),
    _spec("verdict", "the Verdict golden files (design section 6)", verdict_golden_gate),
    _spec("setup_review", "the prominent Tasks against the setup review (D48)", setup_review_gate),
    _spec("candidate_runs", "a batch of Candidate Runs", candidate_runs_gate),
    _spec("budget", "the spend totals of a Candidate batch (D65, D85)", budget_gate),
    _spec("audit", "the blind audit sample per Task (D48)", audit_gate),
    _spec("regrade", "the Verdicts a regrade produced (D97)", regrade_gate),
    _spec("scorecard", "the build directory (D62, D80, D96)", scorecard_gate),
)


def gate_named(name: str) -> GateSpec:
    """The registered gate of this name; a name nothing is registered under is a KeyError."""
    for spec in GATES:
        if spec.name == name:
            return spec
    raise KeyError(name)


def gates_over(artifact: str) -> tuple[GateSpec, ...]:
    """Every gate whose evidence names this build artifact, in registry order (phase 4's hook)."""
    return tuple(spec for spec in GATES if artifact in spec.artifacts)


__all__ = [
    "D79_CHECKS", "D79_STAGES", "FROZEN_TASKS_NAME", "GATES", "GRADER_FIELDS", "LEAK_MIN_LENGTH",
    "TAU2_FILES", "TOOL_RUN_STAGES", "VERDICT_GOLDEN_CHECKS", "VERDICT_VERSIONS", "GateResult", "GateSpec",
    "artifacts", "audit_gate", "body_deterministic_gate", "body_executes_gate", "body_non_trivial_gate",
    "body_parses_gate", "body_refuses_unknown_gate", "body_replay_fidelity_gate", "budget_gate",
    "candidate_runs_gate", "check_run", "cluster_gate", "compile_tools_gate", "compile_tools_gates",
    "confinement", "d79_results", "deterministic_gate", "environment_gate", "executes_gate", "fidelity",
    "freeze_tasks", "gate_confined", "gate_named", "gates_over", "ingest_gate", "intent_gate", "leak_gate",
    "load_run", "loophole_probe", "mine_gate", "non_trivial_gate", "oracle_replay_gate", "parses_gate",
    "policy_gate", "predicate_confinement", "predicate_confinement_gate", "reference_replay_gate",
    "regrade_gate", "replay_fidelity_gate", "replay_match", "rerolls_gate", "scorecard", "scorecard_gate",
    "setup_review_gate", "source_confinement", "stages", "summarize", "task_coverage", "task_verifiers_gate",
    "tau2_export_gate", "tool_runs", "unconfirmed_reason", "unfinished_run", "user_rules_gate",
    "validate_verifier", "verdict_golden_gate", "verifier_gate", "verifier_suite", "vocabulary_gate",
    "wrong_run",
]
