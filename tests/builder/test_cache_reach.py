"""No stage delegates to code its cache key does not hash (G29).

For every builder stage, the first-party modules reachable from its run and gate closures
(through the helpers they call) must be among the modules its code version hashes, and
every own file helper they call must be among its hashed helpers. The walk lives in
kullback.builder.cache_reach; this file pins the contract and guards the walk itself
against silent misses.
"""

from __future__ import annotations

from pathlib import Path

from kullback.builder import cache_reach as reach

# Delegations the walk must report. A subset per stage, chosen where a miss would be
# silent: a gate ruling, an own file helper, a frozen module, a lazy import.
MUST_REACH: dict[str, set[str]] = {
    "ingest": {"kullback.builder.ingest", "kullback.gates.artifacts", "kullback.runner.records"},
    "mine": {"kullback.builder.mine", "kullback.gates.artifacts", "kullback.runner.records"},
    "readers": {"kullback.builder.readers", "kullback.builder.templates", "kullback.gates.stages",
                "kullback.runner.records"},
    "cluster": {"kullback.builder.cluster", "kullback.builder.compile_env", "kullback.gates.scorecard",
                "kullback.gates.stages", "kullback.runner.records"},
    "canon_rules": {"kullback.runner.canon"},
    "starting_state": {"kullback.builder.compile_env", "kullback.builder.readers",
                       "kullback.gates.tool_runs", "kullback.runner.canon", "kullback.runner.records"},
    "compile_tools": {"kullback.builder.memory", "kullback.builder.transaction",
                      "kullback.builder.cluster", "kullback.builder.repair",
                      "kullback.gates.tool_runs", "kullback.runner.records"},
    "compile_policy": {"kullback.builder.policy", "kullback.gates.artifacts", "kullback.runner.records"},
    "judge_lessons": {"kullback.builder.memory", "kullback.runner.records"},
    "vocabulary": {"kullback.builder.vocabulary", "kullback.gates.stages", "kullback.runner.records"},
    "user_rules": {"kullback.user.rules", "kullback.builder.vocabulary",
                   "kullback.gates.artifacts", "kullback.runner.records"},
    "build_environment": {"kullback.builder.compile_env", "kullback.builder.readers",
                          "kullback.gates.artifacts", "kullback.gates.stages",
                          "kullback.runner.records"},
    "replay_reference": {"kullback.builder.compile_env", "kullback.runner.replay",
                         "kullback.runner.route", "kullback.gates.fidelity",
                         "kullback.gates.tool_runs", "kullback.runner.canon",
                         "kullback.runner.judge", "kullback.runner.records",
                         "kullback.builder.effects", "kullback.builder.repair",
                         "kullback.gates.verifier_suite"},
    "intent": {"kullback.builder.intent", "kullback.builder.parallel",
               "kullback.gates.stages", "kullback.runner.records"},
    "rerolls": {"kullback.builder.compile_env", "kullback.runner.loop", "kullback.runner.route",
                "kullback.user.rules", "kullback.builder.intent", "kullback.user.agent",
                "kullback.user.context", "kullback.builder.vocabulary", "kullback.runner.records"},
}

MUST_REACH_HELPERS: dict[str, set[str]] = {
    "cluster": {"_grouping"},
    "canon_rules": {"_rows_of"},
    "compile_tools": {"callers_by_tool", "trace_worlds", "holdout_world", "evidence_calls",
                      "replay_lesson", "attribute_fidelity"},
    "compile_policy": {"_policy_text"},
    "user_rules": {"_vocab_of"},
    "build_environment": {"with_synthetic_rows"},
    "replay_reference": {"holdout_world", "with_synthetic_rows", "replay_failures_of",
                         "_write_runs_index"},
    "intent": {"_read_intents"},
    "rerolls": {"_reroll_key", "_candidate_task_ctx", "_candidate_run_once", "_user_driver",
                "_tool_definitions", "_vocab_from", "_write_runs_index"},
}

MUST_REACH_CONSTS: dict[str, set[str]] = {
    "cluster": {"TASK_SPLIT"},
    "canon_rules": {"CANON_RULES"},
    "starting_state": {"WORLD_PROVENANCE_FILE"},
    "compile_tools": {"KEPT_BODY_DIR", "REPLAYS_FILE", "FIDELITY_REASONS", "REPLAY_AGREED",
                        "REPLAY_LESSON_HEAD"},
    "replay_reference": {"REPLAY_EVIDENCE_FILE", "EQUIVALENCE_FILE", "EFFECTS_FILE"},
    "rerolls": {"REROLL_SEED", "REROLL_TURNS", "REROLL_KEY_FORMAT", "RUN_SEED_KIND",
                "_JSON_TYPES", "REROLL_RECORD", "REROLL_KEY_NOTE"},
}


# hashed by construction; pin the set so an edit to that function fails loudly here.
EVIDENCE_COMMITTED = {"replay_failures", "replay_failures_of", "replay_difference",
                      "replay_lesson", "evidence_calls", "is_evidence_call", "after_write_calls"}


def test_no_stage_reaches_past_its_key():
    """Reachable modules, helpers and constants minus hashed ones are empty, for every stage."""
    failures = []
    for stage, factory in reach.STAGE_FACTORIES.items():
        missed = reach.missing(factory)
        if missed.modules or missed.helpers or missed.constants:
            failures.append(f"{stage}: modules={sorted(missed.modules)} "
                            f"helpers={sorted(missed.helpers)} constants={sorted(missed.constants)}")
    assert not failures, "stages delegate to code their key does not hash:\n" + "\n".join(failures)


def test_walk_sees_known_delegations():
    """The walk reports the delegations a silent miss would hide."""
    failures = []
    for stage, factory in reach.STAGE_FACTORIES.items():
        found = reach.reachable(factory)
        missing_modules = MUST_REACH.get(stage, set()) - found.modules
        missing_helpers = MUST_REACH_HELPERS.get(stage, set()) - found.helpers
        missing_consts = MUST_REACH_CONSTS.get(stage, set()) - found.constants
        if missing_modules or missing_helpers or missing_consts:
            failures.append(f"{stage}: walk missed modules={sorted(missing_modules)} "
                            f"helpers={sorted(missing_helpers)} constants={sorted(missing_consts)}")
    assert not failures, "the walk went blind:\n" + "\n".join(failures)


def test_stage_keys_cover_the_second_hop():
    """The import closure of what each stage reaches, minus exemptions, is inside its key."""
    failures = []
    for stage, factory in reach.STAGE_FACTORIES.items():
        found = reach.reachable(factory)
        keyed = reach.hashed(factory)
        covered, needed = set(), set()
        for module in keyed.modules:
            covered.add(module)
            covered |= reach.import_closure(module)
        for module in found.modules:
            needed.add(module)
            needed |= reach.import_closure(module)
        missing = needed - covered - reach.EXEMPT
        if missing:
            failures.append(f"{stage}: second hop missing={sorted(missing)}")
    assert not failures, "a key misses what a hashed module imports:\n" + "\n".join(failures)


def test_module_hash_covers_the_closure():
    """The key hash moves with closure bytes, deterministically, honoring exemptions."""
    from kullback.builder import build as build_module
    from kullback.builder import mine
    from kullback.runner.records import content_hash

    assert reach.import_closure("kullback.builder.mine"), "pin needs a nonempty closure"
    plain = content_hash(Path(mine.__file__).read_bytes())[:16]
    assert build_module._module_hash(mine) != plain
    assert build_module._module_hash(mine) == build_module._module_hash(mine)
    assert "kullback.runner.canon" in reach.import_closure("kullback.builder.mine")
    assert reach.closure_hash(mine, exempt=frozenset({"kullback.runner.canon"})) != \
        reach.closure_hash(mine)
    assert "kullback.agent.__init__" in reach.closure_files("kullback.builder.repair"), \
        "importing a submodule executes its package init"


def test_two_modules_that_import_each_other_keep_their_own_hashes():
    """A cycle gives both modules one closure, so only the module's own name tells them apart.

    The tool compiler imports the sandbox, and the sandbox reaches the compiler again through
    the effects module, so their closures are the same 42 files. A key that delegates to one of
    them has to say which, or an edit that should rebuild one stage rebuilds the other as well
    and the key stops naming the module it carries.
    """
    from kullback.builder import compile_env, sandbox

    assert reach.closure_files("kullback.builder.compile_env") == \
        reach.closure_files("kullback.builder.sandbox"), "pin needs the cycle to still be there"
    assert reach.closure_hash(compile_env) != reach.closure_hash(sandbox)


def test_stage_registry_matches_the_graph():
    """The walk covers exactly the factories the stages graph calls, no more and no fewer."""
    assert reach.factories_in_stages() == set(reach.STAGE_FACTORIES.values())


def test_evidence_version_commits_its_helpers():
    """_evidence_version hashes exactly the helpers the key relies on it for."""
    assert reach.evidence_commit() == EVIDENCE_COMMITTED


def test_every_stage_hashes_something():
    """No stage passes vacuously: reachable and hashed are both non empty."""
    failures = []
    for stage, factory in reach.STAGE_FACTORIES.items():
        found, keyed = reach.reachable(factory), reach.hashed(factory)
        if not found.modules:
            failures.append(f"{stage}: walk found nothing reachable")
        if not keyed.modules:
            failures.append(f"{stage}: key hashes no module")
    assert not failures, "\n".join(failures)
