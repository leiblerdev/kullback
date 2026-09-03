<!-- Written 2026-09-03 by a review fleet: 28 agents over the union of every open branch. A finding that a test proves nothing was only kept when a verifier broke the behaviour the test names, in a private copy of the tree, and the test still passed. -->

# Kullback test suite review

25,500 lines, 81 files, 1,648 test functions, 1,878 collected tests. Every finding below was
checked by a second reader who tried to refute it, most by breaking the implementation and
re-running the suite. The cross-file duplication claims, which arrived unverified, were read
before being repeated; what did not survive that read is listed as dropped.

## Verdict

This is a serious suite, not a coverage-number suite. The house rules it works under are real and
mostly kept: no model is ever called, the three offline models sit at the provider boundary, and
almost every test is named for a behaviour in a sentence rather than for the function it calls.
Most of what it asserts, it asserts sharply, and the hard parts of the product are genuinely
pinned. Verdict classification, canonicalization, routing, the gate registry, the import
boundary, replay fidelity, the agent harness and the session tree all have tests that fail when
you break them, and several mutations we tried during this review were killed instantly by tests
in the right file with the right name.

What you cannot yet do is read a green run as proof that the shipped rules hold. Two things get in
the way. First, a recurring pattern of assertions that cannot fail: a partition property that
every possible split satisfies, an `assert "the" in source`, a comparison of a list against
itself, an expected hash re-derived by calling the hash function, a handler that raises an
exception the production code swallows. Around twenty-six tests are carried by at least one such
line, and in nine of them the line is the whole test, so the rule they are named for is held by
nothing. Second, a long tail of branches that can be deleted with the suite fully green: the
held-out call rule behind D51 and D75, the twin check in `synth.verify`, the z statistic and alpha
table behind D132, four of the seven environment-suspicion triggers in `verdict._env_marks`, the
builder-tree index rebuild, four of the six ways a replay fails to confirm a Reference, the
`CanonRules` argument to `check_run`, the corpus demotion `derive_all` runs, and four of the read
tool's eleven kinds. We removed each of those and the suite reported 1,878 passed.

The other three axes are healthier but uneven. Naming is good on average and bad in clusters: a
handful of files still name the function instead of the claim, roughly twenty names promise more
than the body checks, and four test names are used twice for different rules, so a failure line
does not say which rule broke. Duplication is the axis with the most obvious cheap wins: one
fixture path is spelled three ways in eight places while the documented `conftest.py` fixture that
provides it is used by exactly one file, and several shared helpers exist in two copies that have
drifted into contradicting each other. Placement drifts against the project's own one-module-one-
test-file rule in about a dozen spots, and two of the largest modules in the package,
`builder/sandbox.py` at 369 lines and `gates/tool_runs.py` at 292, have no test file at all and
are exercised through a re-export from a neighbour. One test double has bent production code:
`cli._score` reads the D97 regrade gate through `getattr(gated, "passed", True)`, failing open for
anything that is not a `GateResult`, and it does that only because a fixture patches `cli._entry`.
That is the single place where a test has made the product worse.

None of this is rot. It is what a suite looks like when it was written fast alongside a moving
design, by many hands, with the review pressure on the code rather than on the tests. The fixes
are nearly all local: assert the rule instead of a property of the rule, add the missing branch,
move the test to the file the rule lives in, delete the second copy of the helper.

## Findings, ranked by what they cost

### 1. Assertions that cannot fail, on rules that decide customer-visible outcomes

The dominant shape, 26 sites. In the nine below the vacuous line is the whole test, so the named
rule is pinned by nothing anywhere in the repository.

- `tests/builder/test_compile_env.py:390`
  `test_split_calls_puts_every_call_in_exactly_one_of_the_shown_and_held_out_splits_and_leaves_neither_empty`
  asserts a partition property that any split satisfies. Mutating `compile_env.py:996-1003` from
  every-third to every-second, and turning the short-corpus fallback into "hold nothing back",
  left the file green (101 passed) and the rest of the suite green (1,777 passed). D51 and D75 rest
  on which calls are held out, and nothing states it. Fix: for six calls assert
  `shown == calls[0], calls[1], calls[3], calls[4]` and `held_out == calls[2], calls[5]`, and add a
  two-call case asserting `held_out == calls[-1:]`.
- `tests/runner/test_verdict.py:287` names the error-event scan, but the run also carries
  `termination_reason="env_error"`, so `_env_error` returns True before the scan runs. Replacing
  the whole `any(event.type == "error" ...)` line with `return False` left the full suite green.
  Fix: split into one run whose only signal is the error event, one whose only signal is the
  reason, and one for the `payload["class"] == "env_error"` spelling.
- `tests/builder/test_synth.py:217` claims in its name and its comment to catch a twin, but no
  assertion reads `checks["twins"]` and the fixture cannot produce one (changing the id changes the
  fingerprint). Replacing the `checks["twins"].append(...)` at `synth.py:904` with `pass` left the
  full suite green. Fix: split the test, and add one that copies an observed user, changes only
  `address.city`, and asserts `"repeats an observed row"` appears in `checks["twins"]`.
- `tests/builder/test_skills.py:33` and its sibling only reach the `sd == 0` shortcuts, so the D132
  z statistic and the alpha threshold table are never exercised. Setting `z = 0.0` and replacing
  the threshold table with `{}.get(alpha, 1.96)` left the file green. Fix: add diff sets with real
  variance on each side of 1.96, and one case pinning that `alpha=0.01` needs 2.576.
- `tests/builder/test_repair.py:16` `test_verbs_record_requests` never asserts anything about what
  was recorded. Reducing `repair._record`'s body line to `{}` left the file green. Fix: read each
  verb's JSONL back and assert verb, target and arguments; split one test per verb.
- `tests/ai/test_pricing.py:148` (and the same pattern at :110 and :120) rests on a handler that
  raises `AssertionError`, but `pricing._fetch` swallows every exception. Setting `stale = True`,
  the exact regression the test is named for, left all 19 tests passing. Fix: count calls in the
  handler and assert the count, rather than raising inside it.
- `tests/builder/test_policy.py:146` claims an abbreviation guard and a time string. Setting
  `policy._ABBREVIATIONS = ()` left all 45 tests passing, and the forbidden time string is one
  `split_policy` cannot produce. Fix: use `split_policy("Cancel it, e.g. Order #W1. Then confirm.")`,
  which does isolate the guard, and drop the unreachable case.
- `tests/examiner/test_derive.py:118` names D43's "agreed in some re-runs is allowed" rule but its
  case is decided by provenance. Replacing the agreement test with `agreed = True` left the full
  suite green. Fix: make the case turn on a value that differs across the good Runs.
- `tests/test_rounds.py:242` asserts `allowance_remaining is not None and <= 0`, which the line
  that seeds it at the start of the beat already satisfies. Deleting the whole watcher branch from
  `Loop._watched` left the test passing. Fix: give the beat a nonzero allowance, write a real
  `budget.json` total from a subscriber, and assert the recorded sequence falls.

Seventeen more sites where a vacuous line sits beside real assertions, so the cost is a misleading
test rather than an unguarded rule: `tests/builder/test_memory.py:888` (`!= ""`, and the
`data = {}` mutation in `record_lesson` that erases every other tool's lessons survives the whole
suite), `test_memory.py:536` (pins a constant no production code reads),
`tests/builder/test_skills.py:69` (compares a list against itself),
`tests/builder/test_repair.py:63` (only the restore path, so an unconditional restore passes),
`tests/agent/test_context_tools.py:244` (expected digest re-derived by calling `content_hash`;
returning a constant from it leaves the whole suite green),
`tests/test_rounds.py:613` and `:387`, `tests/runner/test_records.py:147` and `:332`,
`tests/builder/test_intent.py:300`, `tests/builder/test_pipeline.py:649`,
`tests/builder/test_compile_env.py:932` and `:1383`, `tests/builder/test_mine.py:516`,
`tests/test_cli.py:306`, `tests/gates/test_verifier_suite.py:132`,
`tests/examiner/test_derive.py:401`.

### 2. Rules that can be deleted with the suite green

35 sites where a mutation removed shipped behaviour and nothing failed. The ones that would cost
most if they regressed:

- `kullback/builder/memory.py` index recovery. Making `_read_index` return `{"nodes": {}}` for both
  the corrupt and the missing file, so `_rebuild_index` is dead, left 1,878 passed. `head`,
  `open_proposals` and `accepted_single_rounds` all read that index, so D82's one-change-per-round
  rule drops silently. Add a test at `tests/builder/test_memory.py:152`.
- `kullback/runner/verdict.py:63-72`. Four of the seven `_env_marks` triggers, the `llm` route, the
  Run-level `assisted` flag, `payload["assisted"]`, and the `tag`/`tags` spellings of
  `fact_unavailable`, are covered by nothing. The `llm` route is a live D88 signal.
  `tests/runner/test_verdict.py:328`, widen the parametrize.
- `kullback/runner/replay.py:245-251`. Four of the six ways a replay fails to confirm a Reference
  (unmade, gaps, crashed, stand-in answer) are produced by no test. `tests/runner/test_replay.py:12`.
- `kullback/gates/verifier_suite.py:704`. Changing `fn = canon_fn(canon)` to `canon_fn(None)`,
  so the scorer ignores the customer's canon rules, left the suite green, while
  `gates/probes.py:49`, `gates/loosening.py:62` and `examiner/tools.py:481` always pass rules in.
  This is the Verifier and the Verdict silently comparing under different rules.
  `tests/gates/test_verifier_suite.py:194`.
- `kullback/examiner/stage.py` `final_constraints`. Replacing the demote call with a no-op, so no
  rule is ever demoted and no residual is built, left the suite green; the only test of the D76
  corpus demotion is an `is_file()` check on its output. `tests/examiner/test_stage.py:49`.
- `kullback/examiner/reference.py` `judge_groups`. Replacing the `except Exception` with a bare
  `raise`, so a broken judge takes the Task down, left the suite green. D110 leans on that branch.
  `tests/examiner/test_reference.py:87`.
- `kullback/examiner/tools.py` `_read`. Four of the read tool's eleven kinds (gates, rerolls,
  replays, references) return a constant with the suite green. `tests/examiner/test_tools.py:341`.
- `kullback/builder/cluster.py:241`. Replacing the world split with `[cluster]` left 689 passed:
  the D74 world split is tested only on `split_by_world` in isolation, never through `cluster_runs`.
  `tests/builder/test_cluster.py:510`.
- `kullback/builder/compile_env.py` `lookup_rows`. The `schema.homes` lookup, the 2000-character
  truncation, and the unknown-table, not-found and table-shape answers all mutate green. D117 names
  the first two as shipped. `tests/builder/test_compile_env.py:1516`.
- `kullback/runner/route.py:131`. The `__tool_type__` marker branch of `_is_tool` is unreachable in
  the whole suite because every test Router is given `tool_sigs`; production reaches it whenever
  `tool_sigs.json` is empty. `tests/runner/test_route.py:200`.
- `kullback/report.py`. The entire budget branch of `stage_statuses` (per-stage usd, `cache_share`,
  `memo_hits`) and `_cell`'s pipe escaping both delete green; the three files that import
  `kullback.report` are the only tests that could catch either.
  `tests/test_report.py:669` and `:167`.
- `kullback/cli.py:457`. The `tui` command has no test: renaming the entry point and mangling every
  forwarded option left `tests/test_cli.py` and `tests/test_tui.py` green (75 passed). Both
  "seven commands" docstrings are wrong. `tests/test_cli.py:70`.
- `scripts/env_fidelity.py`. `ours()` and `recorded_calls()` have no test at all, including the
  write detection whose past bug the module records in a comment; and `verdict()`'s payload
  unwrapping deletes green. `tests/test_env_fidelity.py:1` and `:13`.

Also in this bucket, smaller: `mine.is_scalar_result`'s mixed case, `intent._written_value` and
`_earlier_view`'s two guards, `user_sim`'s agent-refusal walk-away half, `search.py`'s ten-URL
batching, page truncation and all three `close()` paths, `artifacts.candidate_runs_gate`'s `seeds`
argument and `mine_gate`'s empty-schema guard, `confinement.DENIED_ATTRS` and the three extras in
`DENIED_BUILTINS`, `messages.to_wire`'s surviving error turn, `stream.py`'s `CancelledError`
escape, `scorecard._not_gradeable`'s dict and task-id branches, `canon.save_table`'s key ordering,
`ingest.py:102`'s bare-simulation `tau2_native` branch, `context.py:793`'s data fence around the
summarizer prompt, and `SessionStore._take`'s missing-entry guard.

### 3. A test double that bent production code

`tests/test_cli.py:18`. The `fake_modules` fixture patches `cli._entry`, a module internal, and
production was changed to keep the stub working: `cli.py:212` reads the D97 regrade gate through
`getattr(gated, "passed", True)`, which fails open for anything that is not a `GateResult`.
Restoring `if not gated.passed:` fails two tests with `'dict' object has no attribute 'passed'`,
which is the proof. CONTRIBUTING is explicit that a fake belongs at the provider boundary or at a
parameter production already takes. The shipped refusal rule is separately pinned by
`test_a_verdict_missing_its_runner_version_is_refused_not_counted`, so nothing is unverified;
what is wrong is the shape of the product. Fix: give `_score` a `regrade_gate` parameter defaulting
to the real one, restore the strict read, and delete the comment naming the fixture.

Two smaller cases of the same shape in `tests/test_rounds.py:613` and `:652`, both monkeypatching
`builder_agent.drive_tool`. At `:613` the patched `is_error=True` is inert, since the `BuildError`
actually comes from `plan.last` being None, so the branch the test names is never taken. At `:652`
a bare `object()` stands in for the pipeline result where a real failing action was available; we
wrote the mock-free version and it passes.

### 4. Names that promise more than the body checks

22 sites. Cost is misdirection during triage: a red line names a rule the test does not check, and
a green line reads as coverage that is not there.

Worst: `tests/gates/test_package.py:153` is named for every registered gate but iterates a
hand-written table, and 8 of the 40 gates in `GATES` are never called, so a newly registered gate
is silently skipped (all 8 do have rulings elsewhere, so nothing is unproven; what is missing is
the completeness guarantee the name gives). `tests/runner/test_boundary.py:69` advertises thirteen
evasions of the D89 boundary but ten rows pass on an incidental `imports importlib` failure, and
disabling the dynamic-import-call rule fails exactly one of the thirteen; the getattr row's own
evasion is not detected at all. `tests/gates/test_scorecard.py:207` is named for a Task set aside
but sets aside a Run id, and both the Task-id branch and the dict form of `not_gradeable.json`
delete green. `tests/test_report.py:204` claims both coverage numbers and asserts the Task count
plus an assisted-run line; the Run-weighted share, D96's second headline number, is asserted
nowhere in the report slice. `tests/builder/test_search.py:119` says the live switch builds the
providers, but `providers_from_env` never reads `LIVE_ENV_VAR`, and the live-on branch of
`search_for` returns `None` with the suite green.

Then: `tests/test_e2e.py:454` (only `passed is False`, which cannot tell discrimination from
inability to evaluate), `tests/runner/test_verdict.py:140` (accepts either of two classes when one
is the only reachable answer), `tests/runner/test_canon.py:70` (the corpus does contain a float),
`tests/runner/test_route.py:169` (promises the End state, builds no Router),
`tests/builder/test_ingest.py:387` (the word "reused" it searches for is the simulation's own trace
id) and `:543`, `tests/builder/test_compile_env.py:1186` (a comma check no shrinking implementation
would trip), `tests/examiner/test_derive.py:367` (the rude Run fails at `c0`, never reaching the
judge atom) and `tests/examiner/test_tools.py:140` (a comment describing an assertion that is not
there), `tests/test_rounds.py:462` (claims a protected entry is released; nothing is ever
protected in that scenario), `tests/builder/test_pipeline.py:483` ("refuses" for a filter that
returns an empty list), `tests/builder/test_memory.py:757` (names one required field, asserts
three), `tests/gates/test_package.py:72`, `tests/test_e2e.py:406`,
`tests/gates/test_verifier_suite.py:132` (docstring claims the per-exec copy of the builtins
allowlist is verified; the rule is refused by `confine` before exec, so the claim rests on an
assertion that cannot fail).

Separately, five names are the function list rather than a claim: `tests/builder/test_skills.py:10`
and `:33`, `tests/gates/test_scorecard.py:67` and `:77`, `tests/examiner/test_derive.py:390` and
`:100`, `tests/builder/test_ingest.py:125`.

### 5. Four test names used twice for different rules

A failure line naming one of these does not say which rule broke. All four verified by reading both
bodies.

- `test_a_tool_call_naming_a_path_under_gates_or_runner_is_blocked_and_becomes_an_error_result` at
  `tests/builder/test_extension.py:195` and `tests/examiner/test_extension.py:102`. Different hooks
  (`no_agent_writes_gates_or_runner` versus `examiner_reads_only_its_surface`), different tools
  (`write_file` versus `finding`), different decisions (D122 versus D123). Rename the examiner copy.
- `test_replay_fidelity_reports_success_and_error_separately` at
  `tests/builder/test_compile_env.py:362` and `tests/gates/test_fidelity.py:16`. Two unrelated
  functions reaching opposite verdicts: the first calls `ce.gate_replay_fidelity` (a re-export of
  `sandbox.py:250`) and asserts `passed is True` with `success_calls`/`error_calls` metrics; the
  second calls `gates/fidelity.py:34`'s `replay_fidelity_gate` and asserts `passed is False` with
  nested `success`/`error` dicts. Rename the builder copy.
- `test_summarize_counts_traces_tasks_and_calls` at `tests/gates/test_fidelity.py:103` and
  `test_summarize_counts_traces_tasks_and_calls_and_names_the_common_miss` at `:147`. Six identical
  assertion lines; the second's last two lines re-assert what `:112` already owns. Correction to the
  original claim: `:103` is not a strict subset, it also asserts `reads` and `reads_cosmetic`, which
  `:147` does not. Fix: drop the two `unconfirmed_reason` lines from `:147`, keep both count sets.
- `test_load_run_reads_header_plus_event_lines` / `test_load_run_reads_header_events_and_footer` at
  `tests/gates/test_verifier_suite.py:45` and `tests/runner/test_verdict.py:499`, plus the one-line
  pair at `:53` and `:507`. `verifier_suite.load_run` is `return load_run_jsonl(path)` and its own
  docstring says "one reader is enough". Meanwhile `records.load_run_jsonl` has no test of its own,
  and `verdict.load_run`'s `Run` and `dict` passthrough branches are exercised by nothing (grep
  confirms both calls in that file pass a path). Fix: move the reader claims to
  `tests/runner/test_records.py` against `load_run_jsonl`, keep `verifier_suite.py:59` (it is about
  `S.start_state`), and add one test for the two branches `verdict.load_run` adds.

### 6. One fixture path, three spellings, eight hand-rolled copies

`tests/conftest.py:47` defines a session-scoped `tau2_small_path`, and DEVELOPING.md lists it. Only
`tests/builder/test_ingest.py` uses it. Everything else recomputes the path:
`Path(request.config.rootpath) / "tests" / "fixtures" / "tau2_retail_small.json"` at
`tests/builder/test_extension.py:42`, `tests/test_rounds.py:40`, `tests/examiner/worlds.py:95`,
`tests/builder/test_parallel.py:113`, `tests/builder/test_build.py:44` and `:213`, and
`tests/test_e2e.py:163`; and `Path(__file__).resolve().parents[1] / "fixtures" / ...` at
`tests/builder/test_mine.py:89`. `tests/builder/test_extension.py:41` and `tests/test_rounds.py:39`
are byte-identical `_fixture` functions under the same name. Fix: promote the path to a module-level
constant in `tests/conftest.py` (`FIXTURES` is already there) so module-scoped fixtures can import
it, and delete the eight copies.

### 7. Shared helpers duplicated, two of them into contradiction

Eleven shapes, roughly thirty sites, in a suite that already has the convention CONTRIBUTING
prescribes (`tests/<package>/<name>_fixtures.py`, honoured in nine places).

- `a_constraint` at `tests/gates/test_artifacts.py:226` and `tests/gates/test_confinement.py:17`.
  The two disagree about the case shape: artifacts wraps in `pre_state`, confinement does not, and
  `gates/artifacts.py:246` runs a case as `func(case.get("pre_state") or {}, ...)` with a comment
  recording that build 8 failed every compiled constraint over exactly this. The confinement copy's
  cases would raise `KeyError`; they never do, because all three uses either override
  `predicate_src` with an escape refused before it runs or read `.predicate_src` alone. Delete the
  copy or replace it with a `PREDICATE_SRC` constant.
- `oracle_lines` at `tests/runner/test_verdict.py:51` and `tests/runner/test_regrade.py:18`. The
  same cancel-W123 Run under one name with different parameters (`reason` versus `run_id`) and
  different wording, so a change to one leaves the other behind. Both then hand-roll a JSONL writer
  (`test_verdict.py:117-125`, `test_regrade.py:53-58`) while `tests/conftest.py:117`'s
  `write_run_jsonl` does exactly that and is used by nothing but `make_recorded_model`.
- `a_run` at `tests/gates/test_artifacts.py:449` and `tests/gates/test_scorecard.py:15`, byte for
  byte identical, in a package that already holds `verifier_fixtures.py` and `examiner_fixtures.py`.
  `tests/test_report.py:51`'s `a_run` is a genuinely different contract; leave it.
- `_reply` and the async drain, four copies and two incompatible contracts.
  `tests/test_rounds.py:43` and `tests/builder/test_extension.py:30` are byte-identical `_reply`
  taking `(name, args)` with auto-assigned ids; `tests/examiner/test_extension.py:32` is the same
  name taking `(cid, name, args)`. The drain is duplicated at `tests/test_rounds.py:48`,
  `tests/builder/test_extension.py:35` and `tests/agent/conftest.py:14`, and reappears as `_run` at
  `tests/examiner/test_extension.py:37`. `tests/agent/conftest.py` already has the sanctioned pair.
- `make_trace` and `cancel_trace` across `tests/builder/test_cluster.py:32` and `:61`,
  `tests/builder/test_intent.py:22` and `:37`, `tests/builder/test_user_sim.py:43`. The intent copy
  is a strict subset of the cluster one (no errors, no requestor); the user_sim one takes
  `(role, content)` pairs and no tool calls at all. The `cancel_trace` pair produce different
  transcripts under one name.
- `Order`, `DB`, `Toolkit` and the world dict at `tests/runner/test_route.py:39-75` against
  `tests/runner/replay_fixtures.py:23-58`, whose own docstring says the other packages import from
  it "rather than each standing up a second copy of the same world". Correction to the original
  claim: `Order`, `DB` and the world dict are identical, but the two `Toolkit`s are not near-twins;
  the fixtures one is parametrized (`cancelled`, `total_as`) and has `check_balance`, the route one
  has `get_db_hash`. Share `Order`, `DB` and `world`, keep the two toolkits.
- `_world` at `tests/gates/test_round_end.py:15` and `tests/gates/test_trust.py:53`. Correction to
  the original claim: these are not identical. Both build the same eleven keys from the same
  `examiner_fixtures` helpers, but the round_end copy adds a second failing Task and takes
  `**update`, and the trust copy passes `"success"` reasons to `reroll_row`. The shared shape is
  worth one parametrized function in `examiner_fixtures.py`; the differences must become arguments,
  not be flattened away.
- `_harness` at `tests/examiner/test_tools.py:31` and `tests/examiner/test_extension.py:43` (the
  first is the second with arguments dropped), and byte-identical `_read` at
  `tests/examiner/test_tools.py:23` and `tests/examiner/test_stage.py:24`. `tests/examiner/worlds.py`
  already holds `drive`, `events_of` and `probe_runner_over`.
- Seven test files import fixtures out of another test module, which CONTRIBUTING forbids by name:
  `tests/runner/test_loop.py:8`, `tests/builder/test_build.py:16`,
  `tests/builder/test_parallel.py:17`, `tests/builder/test_extension.py:12`,
  `tests/test_rounds.py:17`, `tests/examiner/worlds.py:14` (a fixtures module importing a test
  module across packages), and `tests/runner/test_budget.py:203` with
  `tests/runner/test_regrade.py:234` importing `import_closure` from `test_verdict`. Two of these
  reach up into `tests/test_e2e.py`, so importing it as a library re-runs its module-level setup.

Nothing here was dropped as false; two entries were corrected as noted, and both corrections make
the fix larger, not smaller.

### 8. Tests in the wrong file, and two large modules with no test file at all

`kullback/builder/sandbox.py` (369 lines) and `kullback/gates/tool_runs.py` (292) have no test
file. `compile_env.py:35-40` re-exports four sandbox gates with a comment saying it is "for tests",
and `tests/builder/test_compile_env.py` reaches them at lines 343, 350, 357, 364, 376, 715, 717,
741, 758, 796, 809 and 1475. `kullback/ai/messages.py` (114 lines) and `kullback/agent/events.py`
likewise have no file; their only tests sit in `tests/ai/test_stream.py:140` and
`tests/agent/test_harness.py:155`.

Tests asserting another package's module from a neighbour's file:
`tests/builder/test_parallel.py` (three of six tests drive `runner/budget.py` and `ai/provider.py`,
and are the only thread-safety tests those modules have),
`tests/gates/test_confinement.py:45` (three claims about `runner/confinement.py`'s `confine`, whose
own file is eleven lines and one test),
`tests/builder/test_extension.py:221` (the only tests of `gates.names_protected_path`),
`tests/builder/test_compile_env.py:1328` and `:1340` (the only tests of `mine_schema`'s nested-home
mining), `tests/agent/conftest.py` (`harness.drive_tool` and `DriverModel` are core code tested
only from the two application packages).

### 9. Redundant tests

Ten sites where deleting one copy loses nothing.
`tests/builder/test_compile_env.py:1189` is `:441` with a different scripted reply that
`model.calls == []` proves is never consumed; `:1377` is covered by `:1282`.
`tests/builder/test_pipeline.py:121` is a strict subset of `:134` (the encoder dispatches on
`isinstance` only, so a top-level record and a nested one hit the same branch), and `:664`
re-plants a parametrized row from `tests/runner/test_boundary.py:55`.
`tests/agent/test_context_tools.py:288` walks the same path as `:203`; we could not construct a
mutation that kills one and spares the other. `tests/builder/test_policy.py:255` makes two
assertions its neighbours at `:180` and `:311` already make, one strictly stronger.
`tests/examiner/test_derive.py:401`'s only live claim duplicates `:125`.
`tests/test_e2e.py:261` is a weaker version of `tests/builder/test_ingest.py:434` over the same
fixture; keeping the e2e copy is defensible only if it gains the sidecar assertion it lacks, since
it greps the on-disk trace text while the ingest test dumps in-memory models.
`tests/test_e2e.py:549` repeats `tests/runner/test_boundary.py:13`'s call with weaker assertions,
and `tests/builder/test_pipeline.py:649` runs the same gate a third time. Note that CONTRIBUTING
names `tests/test_e2e.py` as the file that asserts D89 and D91, so this triplication is at least
partly deliberate; the judgment call is whether the top-level reminder earns a third run. The
`test_pipeline.py:649` copy is not defensible either way: we planted a live dynamic Builder import
inside `pipeline.py` itself and it stayed green, because `runner/boundary.py` scans only
`SCANNED_PACKAGES = ("runner", "ai")`.

### 10. Smaller consistency drift

`tests/builder/test_cluster.py:488` and `:496` read the full raw corpus without
`@pytest.mark.slow`, while the two equivalent tests in `test_mine.py` carry it (nothing deselects
the marker today, so it is documentation).
`tests/gates/test_probes.py:22, 35, 95, 96` and `tests/runner/test_records.py:320` use bug class
names that do not exist in `examiner/skills.py`'s `BUG_CLASSES`, so the failure strings the gate
builds do not read like a real ruling.
`tests/builder/test_search.py:15`'s transport helper lacks the docstring its two identical siblings
in `tests/ai/` carry, and the module docstring names `shared/search.py`, a path that does not exist.
`tests/agent/test_context.py:50`'s comment states 116 characters where the real rendering is 147.
`tests/runner/test_judge.py:150`'s comment states the opposite of what `TestModel` does.
`tests/runner/test_records.py:285` points `PYTHONPATH` one directory above the repository root; the
subprocess works only because the package is installed in the venv.
`tests/runner/replay_fixtures.py:104` exports an `events` helper no test calls, behind a `noqa` on
`tests/test_replay.py:9` hiding four unused imports.
`tests/fixtures/make_tau2_retail_small.py:14`'s `DEFAULT_SOURCE` resolves to `<repo>/../data/raw`
while `scripts/fetch_tau2_traces.sh` writes `<repo>/data/raw`, so the no-argument rebuild command
DEVELOPING.md documents fails on this layout.
`kullback/agent/loop.py:289`'s `user_message` is public, unexported, uncalled and untested.
CONTRIBUTING says the suite is 1,815 tests; it collects 1,878.

## The four axes

### Consistency

Good where it is written down, drifting where it is not. The suite has a real convention, stated in
CONTRIBUTING step 2, and honours it in nine places with `*_fixtures.py` modules. It breaks it in
about twenty: seven test files import fixtures out of another test module (two of them reaching up
into `tests/test_e2e.py`, so importing it as a library re-runs its module setup), a dozen tests
assert another package's module from a neighbour's file, and four modules including two of the
largest in the package have no test file at all. The smaller drift is a long tail: one `slow`
marker convention followed by two files and ignored by two, bug class names invented in one file
while the real ones are used in the next, one transport helper missing the docstring both its twins
carry, a stale module path in a docstring, and a test count in CONTRIBUTING that is 63 low. None of
this breaks anything. All of it makes the suite harder to navigate than it needs to be, and it is
what lets a duplicate helper drift into contradicting the original, which has already happened
twice.

### Correctness, meaning whether the tests test the right thing

This is the weakest axis and the reason for the ranking above. The suite is not a suite of dummies:
we broke a lot of implementation during this review and most mutations were caught, often
immediately, by a test in the right file with a name that told us what we had broken. But the
failure mode is real and it is concentrated. Twenty-six tests contain at least one assertion that
cannot fail, and in nine the vacuous line is the whole test. Thirty-five shipped branches can be
deleted with all 1,878 tests green, several of them rules the design documents name by decision
number: the held-out call rule (D51, D75), the world split (D74), the skill gate statistic (D132),
the corpus demotion (D76), the environment-suspicion triggers (D88), the index recovery behind D82.
The pattern behind most of these is the same mistake made repeatedly: the test asserts a property
that the rule implies rather than the rule, or sets up a case where a second guard reaches the same
answer first, so the named guard is never the one under test. That is a fixable habit, not a
structural problem, but until it is fixed a green run is evidence that nothing obvious broke, not
that the rules hold.

### Duplication

The most tractable axis. Eleven distinct shapes are copied across roughly thirty sites, and in
every case the project already has the mechanism that would remove the copy: `tests/conftest.py`
holds the fixture path and a JSONL writer that nothing uses, `tests/agent/conftest.py` holds the
canonical `reply` and `collect`, `tests/runner/replay_fixtures.py` holds the order world,
`tests/examiner/worlds.py` holds the examiner helpers, and `tests/gates/` already has two fixtures
modules. The cost is mostly navigation, with two exceptions worth taking seriously: `a_constraint`
has drifted into two copies that disagree about the shape of a constraint test case, where the
wrong one survives only because its predicate never runs, and `oracle_lines` has drifted into two
Runs under one name that no longer describe the same transcript. Test-level redundancy is a
smaller problem, ten near-duplicate tests, and two of those three e2e copies are arguably
deliberate under CONTRIBUTING's own wording.

### Naming

Above average and better than most suites, with two specific failures. The house rule, a sentence
stating the behaviour, is followed by the large majority of the 1,648 test functions, and the good
names are genuinely good: they say what the rule is, and reading a list of them tells you what the
module does. The first failure is overreach: roughly twenty names claim more than the body checks,
usually by naming two things and asserting one, and a name that promises a rule nobody checks is
worse than no test, because it reads as coverage. The second is collision: four names are used
twice for different rules in different files, so a failure line does not identify the rule. Five
names still list the functions called instead of the claim, which is drift rather than a settled
local style, since the same files' later tests are full sentences.

## What to fix first

1. `tests/test_cli.py:18` and `kullback/cli.py:212`. Give `_score` a `regrade_gate` parameter and
   restore `if not gated.passed:`. This is the only place a test has made the product worse.
2. The nine hollow tests in finding 1. Each is a one-test fix and each currently leaves a named
   rule unpinned. Start with `test_compile_env.py:390` (D51/D75) and `test_verdict.py:287` (D88).
3. The deletable rules in finding 2 that would be silent in production if they regressed, in this
   order: the memory index rebuild, `verifier_suite.check_run`'s canon rules, `verdict._env_marks`,
   `reference.judge_groups`' failing judge, and `replay.py`'s four unreported failure modes.
4. Rename the four colliding test names (finding 5) and fix the worst name-versus-body mismatches
   (`test_package.py:153`, `test_boundary.py:69`, `test_scorecard.py:207`, `test_report.py:204`,
   `test_search.py:119`).
5. Collapse the fixture path to one constant in `tests/conftest.py` and delete the eight copies.
   Mechanical, no behaviour change, and it removes the `_fixture` twins.
6. Delete the `a_constraint` copy in `tests/gates/test_confinement.py` and unify `oracle_lines`.
   These are the two duplicates that have drifted into disagreement.
7. Create `tests/builder/test_sandbox.py` and `tests/gates/test_tool_runs.py`, and move the sandbox
   and body-fidelity assertions out of `test_compile_env.py`. Then `tests/ai/test_messages.py` and
   `tests/agent/test_events.py`.
8. Move the seven cross-test-module imports into `*_fixtures.py` modules, starting with the two
   that reach into `tests/test_e2e.py`.
9. The rest of finding 2's coverage gaps, the remaining naming fixes, and the ten redundant tests.
   None of these is urgent; do them as you touch the files.
