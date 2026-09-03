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

Phase 5 adds the Examiner's gates (D127, D133, D128, D126): `probe_pool` and `probe_admission` over
the monotone probe pools, `loosening` and `false_rejection` over the Verifier histories and the
legitimate pool of frontier Runs, `refuse` over the refusals, and `trusted`, whose count is the
round's "Tasks with a trusted Verifier"; `round_end` holds the counts and the three exits, and
`ledger.GateLedger` is the one class both agents write gates.json through.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.gates import (
    artifacts,
    confinement,
    fidelity,
    ledger,
    loosening,
    probes,
    round_end,
    scorecard,
    stages,
    tool_runs,
    trust,
    verifier_suite,
)
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
from kullback.gates.ledger import GateLedger
from kullback.gates.loosening import (
    accepted_versions,
    false_rejection,
    false_rejection_gate,
    legitimate_runs,
    loosening_gate,
    newly_passed,
)
from kullback.gates.probes import (
    PROBE_STOP,
    consecutive_failed,
    probe_admission_gate,
    probe_pool_gate,
    probe_scores,
    version_hash,
)
from kullback.gates.round_end import GATE_COUNTS, done, exit_for, round_counts, stalled
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
from kullback.gates.trust import finished_runs, refuse_gate, trusted_gate
from kullback.gates.verifier_suite import (
    D79_STAGES,
    HELPERS_SRC,
    check_run,
    d79_results,
    load_run,
    loophole_probe,
    unfinished_run,
    validate_verifier,
    wrong_run,
)
from kullback.runner.records import GateResult

# --- what both agents share of the gates: the paths no agent writes, the ruling a tool result carries ---

PROTECTED = ("kullback/gates", "kullback/runner")
# A path segment `gates/` or `runner/`, on its own or under kullback/, at the start of a value or
# after a separator; `gates.json`, `runs/` and `runner_version.json` do not match. The second
# alternative is the package directory named with nothing after it (`rm -r kullback/gates`), which
# the first would miss for want of a trailing separator; it asks for the `kullback/` prefix, so a
# bare word `gates` or `runner` in an argument is still an ordinary string.
PROTECTED_PATH = re.compile(
    r"(?:^|[\s\"'=:,(\[{/\\])"
    r"(?:(?:kullback[/\\])?(?:gates|runner)[/\\]|kullback[/\\](?:gates|runner)(?=$|[\s\"',)\]}]))")


def first_string(value: Any, match: Callable[[str], bool]) -> Optional[str]:
    """The first string in a value (walked through dicts, lists and tuples) that `match` accepts."""
    if isinstance(value, str):
        return value if match(value) else None
    if isinstance(value, dict):
        items: Any = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return None
    for item in items:
        found = first_string(item, match)
        if found is not None:
            return found
    return None


def names_protected_path(value: Any) -> Optional[str]:
    """The first string in a value that names a path under the gates or the Runner, the two packages
    no agent writes (D122); the Builder's and the Examiner's tool_call hooks both refuse on it."""
    return first_string(value, lambda text: PROTECTED_PATH.search(text) is not None)


class Ruling(BaseModel):
    """One gate's answer as a tool result carries it: named, decided, the reasons in `failures`."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    passed: bool
    failures: list[str] = Field(default_factory=list)


def ruling_of(result: GateResult) -> Ruling:
    return Ruling(stage=result.stage, passed=bool(result.passed), failures=list(result.failures))


def ruling_line(label: str, rulings: Any) -> str:
    """One ruling per name, pass or fail with the first reason: the line the model reads."""
    return f"{label}: " + "; ".join(
        f"{r.stage} {'pass' if r.passed else 'fail'}" + (f" ({r.failures[0]})" if r.failures and not r.passed else "")
        for r in rulings)


def rulings_over(store: dict, produced: Iterable[str]) -> list[GateResult]:
    """Every registered gate bound to one of these artifacts, run over a store, in registry order;
    a gate whose artifacts the store does not all hold is skipped. Both agents' tool_result hooks
    call this over their plan's store with what the tool said it produced."""
    out: list[GateResult] = []
    seen: set[str] = set()
    for artifact in produced:
        for spec in gates_over(artifact):
            if spec.name in seen or any(name not in store for name in spec.artifacts):
                continue
            seen.add(spec.name)
            out.append(spec.fn(*[store[name] for name in spec.artifacts]))
    return out


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
    _spec("probe_pool", "every probe in a Task's pool scores no pass on its current Verifier", probe_pool_gate,
          artifacts=("verifiers", "probes", "canon_rules", "sigs")),
    _spec("probe_admission", "a Task is open to a new probe until three consecutive probes were already rejected",
          probe_admission_gate, artifacts=("probes", "verifiers")),
    _spec("loosening", "a new Verifier version newly passes only the Reference, a frontier re-roll or a production Run",
          loosening_gate, artifacts=("history", "task_runs", "replays", "rerolls", "canon_rules", "sigs")),
    _spec("false_rejection", "the held-out frontier Runs the required atoms wrongly fail, per Task",
          false_rejection_gate, artifacts=("verifiers", "task_runs", "replays", "rerolls", "canon_rules", "sigs")),
    _spec("refuse", "a Task is refused only when no frontier Run of it finished", refuse_gate,
          artifacts=("refusals", "replays", "rerolls")),
    _spec("trusted", "a Verifier is trusted when it passed the suite, rejects every probe, is an accepted version "
          "and its Task is not refused", trusted_gate,
          artifacts=("task_status", "verifiers", "probes", "history", "refusals", "task_runs", "replays", "rerolls",
                     "canon_rules", "sigs")),
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
    "D79_CHECKS", "D79_STAGES", "FROZEN_TASKS_NAME", "GATES", "GATE_COUNTS", "GRADER_FIELDS", "HELPERS_SRC",
    "LEAK_MIN_LENGTH", "PROBE_STOP", "PROTECTED", "PROTECTED_PATH", "TAU2_FILES", "TOOL_RUN_STAGES", "VERDICT_GOLDEN_CHECKS",
    "VERDICT_VERSIONS", "GateLedger", "GateResult", "GateSpec", "Ruling", "accepted_versions",
    "artifacts", "audit_gate", "body_deterministic_gate", "body_executes_gate", "body_non_trivial_gate",
    "body_parses_gate", "body_refuses_unknown_gate", "body_replay_fidelity_gate", "budget_gate",
    "candidate_runs_gate", "check_run", "cluster_gate", "compile_tools_gate", "compile_tools_gates",
    "confinement", "consecutive_failed", "d79_results", "deterministic_gate", "done", "environment_gate",
    "executes_gate", "exit_for", "false_rejection", "false_rejection_gate", "fidelity", "finished_runs", "first_string",
    "freeze_tasks", "gate_confined", "gate_named", "gates_over", "ingest_gate", "intent_gate", "leak_gate",
    "ledger", "legitimate_runs", "load_run", "loophole_probe", "loosening", "loosening_gate", "mine_gate", "names_protected_path",
    "newly_passed", "non_trivial_gate", "oracle_replay_gate", "parses_gate", "policy_gate",
    "predicate_confinement", "predicate_confinement_gate", "probe_admission_gate", "probe_pool_gate",
    "probe_scores", "probes", "reference_replay_gate", "refuse_gate", "regrade_gate", "replay_fidelity_gate",
    "replay_match", "rerolls_gate", "round_counts", "round_end", "ruling_line", "ruling_of", "rulings_over", "scorecard", "scorecard_gate",
    "setup_review_gate", "source_confinement", "stages", "stalled", "summarize", "task_coverage",
    "task_verifiers_gate", "tau2_export_gate", "tool_runs", "trust", "trusted_gate", "unconfirmed_reason",
    "unfinished_run", "user_rules_gate", "validate_verifier", "verdict_golden_gate", "verifier_gate",
    "verifier_suite", "version_hash", "vocabulary_gate", "wrong_run",
]
