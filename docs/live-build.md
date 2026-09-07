# The first live build (2026-08-29)

Retail, 456 tau2 simulations, `openai/gpt-5.6-luna`. Nothing in the Builder above `starting_state`
had ever run against a real model, and every one of the findings below is something no unit test
could have reached, because the thing under test was either the request the provider gets or what
the model writes.

## What broke, in the order it broke

**1. `max_tokens` is not the field the gpt-5 family takes.** The very first call came back HTTP 400:
`Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens'
instead.` The same family also refuses any temperature but its own default. `OpenAIModel` now picks
the field from the wire id rather than from a list of model names, because a list of names is stale
the week after it is written and this failure kills a build on its first call.

**2. Tool calls and reasoning cannot be combined on `/v1/chat/completions`.** `Function tools with
reasoning_effort are not supported for gpt-5.6-luna. To use function tools, use /v1/responses or set
reasoning_effort to 'none'.` We were not sending a `reasoning_effort` at all: the endpoint applies
its own default, so not sending one is not enough and it has to be turned off by name. A caller that
did ask for an effort keeps it and gets the error, because quietly downgrading what they asked for
is a worse answer than the error. This is on the path the agentic judges and the Candidate loop use,
not the Builder's, so it would have surfaced one stage later.

**3. The context window default refused a call the model would have taken.** `gpt-5.6-luna` was not
in `CONTEXT_WINDOWS`, so it took the 200,000 default, so D65's cap was 80,000 tokens, and a
`compile_tools` prompt of 89,645 was refused. A 500,000 token prompt was then accepted live, so the
table now carries 400,000 as a measured floor rather than a number read off a page.

**4. The evidence cap and the context cap were measuring different things.** `compile_tool` counts
the characters of its message contents; `budget.py` counts the tokens of the JSON the request will
carry, which is larger by the envelope and by every escaped quote and newline in a tool result. A
prompt of 358,580 content characters was under the 320,000 character cap on nothing and over the
80,000 token cap at 89,645. No constant reconciles two different measures. Worse, the refusal the
authoritative one raises left `compile_tool` and killed the whole build: one tool with a large
corpus took the other thirteen with it. It is now caught where the stage already had a word for a
tool it could not write, so the tool comes out assisted (D49) and the build carries on.

**5. A relative `--workdir` broke every sandbox execution.** The sandbox subprocess runs with `cwd`
inside its own directory and was handed a relative path to its own runner, which the child then
resolved against that directory a second time:

    can't open file '.../.work-retail/tools/cancel_pending_order/attempt_3/sandbox/
                     .work-retail/tools/cancel_pending_order/attempt_3/sandbox/run_tool.py'

Eleven of sixteen tools failed the `executes_on_s0` gate this way, and all sixteen came out
assisted. Every test passes `tmp_path`, which is already absolute, so nothing caught it. `Sandbox`
now resolves its directory once.

**6. `compile_tools` hashed neither of the modules it delegates to.** R42 puts the bytes of
`policy.py`, `memory.py` and `verifier.py` into their stages' cache keys. `compile_tools`, which does
essentially all of its work in `compile_env` and `sandbox`, had no `code_version` at all, so the fix
above would have left every broken body in the cache for `--iterate` to hand straight back.

## What worked first time

- Ingest: 456 runs, 3,220 tool calls, 65 errors, 0 rejected, gate pass, in 3 seconds.
- Mine, cluster, canon and starting state: no model, no surprises, same numbers as the offline runs.
- Prompt caching: 337,810 of 405,357 input tokens were cache reads on the second build. The per
  stage `prompt_cache_key` is doing what docs/prompt-caching.md says it should.
- The confinement gate caught the model writing `eval` into `calculate` and refused the body. That
  is the gate doing its job on real model output for the first time.
- The budget wrapper counted every call, and refused the oversized one, without being asked twice.

## Still open

- **No price for this model.** `budget.PRICES` has no `openai/gpt-5.6-*` row, so every call is
  counted under `unpriced_calls` and `usd` stays 0.00. That is D86 behaving correctly rather than
  quietly pricing at zero, but it also means `--ceiling-usd` cannot be used with this model: a
  ceilinged build raises `UnpricedModel`. The number has to come from a person; it is not something
  to guess.
- **Unbounded retry evidence.** When no shown call is named in a failure, `_evidence_for` sends
  every shown call. For `get_order_details` on this corpus that is 815,521 characters. It was a
  symptom of finding 5 rather than a cause, so it is recorded and not yet changed.

## Environment fidelity against the real toolkit (2026-08-29)

`scripts/env_fidelity.py` replays recorded calls against both our model-written body and the real
tau2 tool, on the real seed database, and compares what came back and whether the state moved.
The reference runs in its own venv under its own dependencies (`scripts/tau2_reference.py`) so
that litellm and pandas stay out of the harness. Retail, 20 distinct calls per tool:

**Agreement 64.3%** on the 157 calls that could be scored; 98 more belong to the 5 tools the confinement gate refused to load.

| tool | calls | same | differs | only ours failed | other message | not loaded |
|---|---:|---:|---:|---:|---:|---:|
| `calculate` | 19 | 0 | 19 | 0 | 0 | 0 |
| `cancel_pending_order` | 20 | 20 | 0 | 0 | 0 | 0 |
| `exchange_delivered_order_items` | 20 | 0 | 0 | 0 | 0 | 20 |
| `find_user_id_by_email` | 19 | 15 | 0 | 0 | 4 | 0 |
| `find_user_id_by_name_zip` | 20 | 18 | 0 | 0 | 2 | 0 |
| `get_item_details` | 9 | 0 | 0 | 9 | 0 | 0 |
| `get_order_details` | 20 | 0 | 0 | 0 | 0 | 20 |
| `get_product_details` | 20 | 0 | 0 | 0 | 0 | 20 |
| `get_user_details` | 20 | 20 | 0 | 0 | 0 | 0 |
| `list_all_product_types` | 1 | 1 | 0 | 0 | 0 | 0 |
| `modify_pending_order_address` | 18 | 0 | 0 | 0 | 0 | 18 |
| `modify_pending_order_items` | 20 | 0 | 0 | 0 | 0 | 20 |
| `modify_pending_order_payment` | 1 | 0 | 0 | 1 | 0 | 0 |
| `modify_user_address` | 8 | 8 | 0 | 0 | 0 | 0 |
| `return_delivered_order_items` | 20 | 19 | 0 | 0 | 1 | 0 |
| `transfer_to_human_agents` | 20 | 0 | 0 | 20 | 0 | 0 |

What each column is telling us:

- **same (101)**: `cancel_pending_order`, `get_user_details`, `modify_user_address`, the two `find_user_id_*`
  and `return_delivered_order_items` behave as the real tool does, result and effect, on calls the
  model never saw. That is the claim the harness makes, and this is the first time it has been true.
- **not loaded (98)**: five bodies reached for `getattr` or `__dict__` and the confinement gate refused
  them. The gate is right; the prompt never said the rule. It says it now (`_confinement_block`),
  generated from the gate's own constants so the two cannot drift.
- **`calculate` differs (19)**: ours returns `{"value": -121.2}`, the real tool returns `-121.2`. A scalar
  result is being described to the model as a one-column object somewhere between mine and the prompt.
- **only ours failed (30)**: `get_item_details` raises `Item not found` where items are nested under
  `products[...].variants`; `transfer_to_human_agents` errors on every call; `modify_pending_order_payment`
  called `.get` on a pydantic row. All three are the model being told less than it needed about the
  shape of the world.
- **other message (7)**: ours raises `Error: User not found`, the real tool `User not found`. The
  model copied tau2's transport prefix out of the trace into the exception.

### The same number split by cause (25 calls per tool, 297 recorded)

The per-tool table says where; the cause table says why and which Builder stage owns it.
`env_fidelity.py` now names a cause per missed call and prints both denominators, because the
headline flatters: the five refused tools are the biggest single miss and the scored number
does not contain them.

- Agreement on the 179 calls that loaded: **65.4%**.
- Agreement over every recorded call, refused tools counted as misses: **39.4%**.

| cause | calls | share of recorded | tools | owner |
|---|---:|---:|---|---|
| confinement | 118 | 39.7% | `exchange_delivered_order_items`, `get_order_details`, `get_product_details`, `modify_pending_order_address`, `modify_pending_order_items` | compile_tools: `getattr` or `__dict__` in the body |
| missing_import | 25 | 8.4% | `transfer_to_human_agents` | compile_tools: the body calls `re.findall` and never imports `re`; `re` is on the allowed list, the module preamble does not import it |
| result_shape | 19 | 6.4% | `calculate` | compile_tools: `{"value": -121.2}` where the real tool returns `"-121.2"`; same number |
| schema_shape | 9 | 3.0% | `get_item_details` | mine: we mined a top-level `items` table; the real db nests items under `products[id].variants`, so on the real seed our body finds no `items`. The nesting is in the traces (5,615 `variants` sightings) and the miner now reads it (D106) |
| error_prefix | 8 | 2.7% | `find_user_id_by_email`, `find_user_id_by_name_zip`, `return_delivered_order_items` | compile_tools: `Error: ` copied from the transport into the message |
| row_access | 1 | 0.3% | `modify_pending_order_payment` | compile_tools: `.get` on a pydantic row |

Five of the six causes are the model being told less than it needed. The confinement block,
scalar results, no `.get` and dict-typed column samples were already in `compile_env.py`; D106
added the rest: the error prefix is now read off the corpus (`shared_error_prefix`) instead of a
constant, a NameError on an allowed module is turned into a one-line import hint in the retry, and
the sixth cause, `schema_shape`, turned out to have signal in the traces after all: every
`get_product_details` result nests items under `variants` keyed by `item_id`, so the miner now
records `items -> products.variants` as the table's home, tells the body to look there first, and
folds the standalone rows into it (9 of 9 on retail). On the first build this was the one miss the
real seed exposed and our own `db.json` hid: on our db the `items` table existed and the tool
answered. The rebuild will say whether the six are closed; none of these rules names a tau2 table,
tool or message, which is the overfit check the next benchmark corpus has to pass (D51, D106).

## Second live build (2026-08-29, evening): 88.2 percent, and what the number hides

Rebuilt with `--iterate`, `openai/gpt-5.6-luna`, grown to the real seed's counts
(`--grow users=500 --grow orders=1000 --grow products=50`): 217 calls, 27 minutes, 16 tools,
seven marked assisted by the gates. `scripts/env_fidelity.py` over 25 calls per tool, 297 recorded:

- Agreement on the 297 calls that loaded: **88.2%** (255 same, 7 both refused). No tool was refused by
  the confinement gate this time, so the two denominators are the same number.
- First build: 65.4% on 179 loaded calls, 39.4% over every recorded call.

| cause | calls | share of recorded | tools | what it is |
|---|---:|---:|---|---|
| result_shape | 19 | 6.4% | `calculate` | still `{"value": x}` for a scalar; the prompt's scalar rule did not take on this tool |
| body_error | 9 | 3.0% | `get_item_details` | the body looks up a top-level `items` table; see below |
| value | 4 | 1.3% | `modify_pending_order_items` | a different answer with no shape or error explanation |
| error_message | 3 | 1.0% | `find_user_id_by_name_zip` | both refused, the wording differs |

Closed since the first build: confinement (118), missing_import (25), error_prefix (8),
row_access (1). Open: result_shape on one tool, and `get_item_details`, which is not the D106 fix
failing but the D106 fix never running. The `mine` stage was served from the cache: its code version
hashed only the closure in `build.py`, not `mine.py`, so the schema this Environment was built on is
the one mined before D106, with `items` as a table of nine rows and no home under
`products.variants`. Every stage now hashes the modules it delegates to (D109), and the third build
is the one that measures D106.

### The Verifier stage had never derived anything

`task_status.json` from this build: 205 Tasks, every one `reference_confirmed: False`,
`verifiers: 0`, Task coverage 0 of 205, and the scorecard reads "Run ... was not replayed" 456
times. The stage read Runs from `runs/<task>/`, which only `harness run` writes; the oracle replay
that design section 6 calls Gate A lived in `tests/test_e2e.py` and nowhere in `src/`. So the
Environment was measured tool by tool against tau2's toolkit and never once against its own
Traces end to end, and no Task had a pass condition.

D108 puts the replay in the build: `runner/replay.py` drives the loop with the Trace's own
assistant and user turns over the built tools, scores every routed answer against the recorded
one, and a Trace whose writes all agree and whose reads never differ in substance confirms its
Reference. The Verifier stage derives from the confirmed seed replays and runs the whole D79 suite:
the wrong Run is built by code from the Reference, the second path is the Task's second confirmed
Trace, the leak check reads the Intent and the user rules, and the loophole probe is one Run per
Task by the model in the Task's own world (`--probe-limit` caps it). On the offline fixture two of
three Tasks confirm and their Verifiers pass seven of eight checks, failing only the second path,
because a single-Trace Task has none until the k re-runs of D78 exist. The third live build is the
first with the stage in it; its numbers go below when it finishes.

## Builds 7 and 8 (2026-08-29 and 2026-09-02): the first Tasks with a Verdict

Build 7 was the last on the old layout, build 8 the first on the rebuilt one (`kullback build --iterate --workers 8`, same corpus, same model, runner frozen first). Build 8 took 44 minutes. The two builds answer the same question, "how many of the 205 Tasks can be graded", and the second is the first with an answer above zero.

| | build 7 | build 8 |
|---|---:|---:|
| Traces that confirm their Reference | 316 of 456 | 343 of 456 |
| Tasks with a confirmed Reference | 151 | 164 |
| writes that replay exactly | 429 of 575 | 472 of 575 |
| reads that differ in substance | 38 of 2,645 | 38 of 2,645 |
| frontier re-rolls | 453 over 151 Tasks | 483 over 161 Tasks |
| Verifiers derived | 76 | 131 |
| Verifiers that pass the D79 suite | 0 | 20 |
| Tasks covered | 0 of 205 | 20 of 205 (56 of 456 Runs) |

The scorecard gate passes for the first time. Coverage is 9.8% of Tasks and 12.3% of Runs, and the 185 uncovered Tasks split into 74 with no confirmed Reference and 111 whose Verifier failed the suite. Three causes account for most of both halves.

**Float noise is the largest fidelity miss.** Of the 102 write results that differed from the recording across the 36 Tasks blocked on `modify_pending_order_items`, 74 differ only past the sixth decimal: a refund of `30.180000000000007` recorded against `30.180000000000064` from our body, the same sum added in a different order. The canonicalization rules carry a `number_precision` knob and the canon stage left it null, so writes compare exactly and the gate calls the Trace unconfirmed. The remaining 28 are real: the body picks a different item variant (the options and the price differ, so the refund does), and `calculate` raises `NameError: name 'decimal' is not defined` on four Tasks because the body uses a module it never imports and the `executes_on_s0` gate never reached that branch.

**The policy gate has never run a predicate.** `compile_policy` fails on every compiled constraint with `check() missing 2 required positional arguments: 'write_call' and 'transcript'`, in build 7 and in build 8. The compiled predicates take `(pre_state, write_call, transcript)`, as the stage's prompt and its own static check require; `gates/artifacts.py`'s `_run_predicate` calls `func(case)` with the whole case dict. The unit test passes because its fixture predicate takes `case`. So the 26 compiled constraints have never been exercised by the gate, only by the stage's own case runner, and the gate's ruling on policy has been a false fail on every live build.

**Sixty-five Verifiers require nothing.** Every Verifier that passes the empty Run has exactly one required atom, "the Run makes at most 0 write calls": the Tasks where the frontier only read and answered. Under D43 the End state of such a Run is what the user was told, and no stage writes that atom yet, so the suite rejects all 65 and the loophole probe passes 58 of them. This is the Examiner's first job (phase 5): atoms over the user-facing effects, judged, so a read-only Task has a pass condition that an empty Run fails.

The suite's failures by check, over the 131 Verifiers: `mutation_flips` 106, `unsolved_state_fails` 75, `empty_fails` 65, `plausible_wrong_fails` 65, `loophole_probe_fails` 58, `second_path_passes` 12, `leak_check_clean` 1. The 65 no-write Verifiers sit inside every one of the first five counts; 41 Verifiers with write atoms still fail `mutation_flips`, which is the next thing to read.

Where the time goes: the re-roll stage is 3 model calls per Task, 7 per Run, with a fresh sample each time, so a rebuild whose Environment did not change still re-rolls every Task. Build 8 wrote 483 re-roll Runs in about 15 minutes at 8 workers. Caching re-rolls by (Task, Environment hash, seed) so `--iterate` reuses them when the Environment is unchanged is on the todo.

### Where build 8's Runs failed: the model's errors and ours

Every failed Run, replay and re-roll in build 8, read and sorted. The split matters because a model error is evidence and a harness error is noise that hides evidence.

**Ours, largest first.**

1. The Simulated user cannot say yes and refuses facts it was never told. Of the 105 Tasks whose recording wrote something, 69 got three re-rolls that wrote nothing, and 247 of the re-rolls with a recorded write ended without one. The transcripts show why: the user opens with the zip alone, the agent asks for a name and the user answers "I do not have my name" (63 re-rolls never authenticated this way); the user gives the zip of the address it is moving to instead of the one on file (27); and when the agent asks the one question every write needs, "do you confirm", the user answers "I do not have an answer for that" (58 re-rolls, authenticated, every action listed, nothing confirmed). One user even answered an address question with an order id. D44 says facts the recorded user gave are repeated verbatim and unasked questions get a representative answer; a confirmation is the most representative answer there is, and the name is a fact in every recording. Until this is fixed the re-rolls corroborate nothing, the judge sees "B: no writes" on every Task, and D111 has no samples.
2. The judge fails what it cannot see. Seventeen Tasks read "the judge failed every End state". In eleven the reason is "without evidence of the required authentication and explicit confirmation": the judge is handed the Intent, the verifier's output and the End state, with read-only tools over the state and no transcript, so that evidence is absent by construction and D93 says abstain, not fail. In three the judge grades the opening request instead of the Intent: the Intent says "exchange desk lamp only" because the recorded user changed their mind mid-conversation, the frontier did exactly that, and the judge writes "the user requested exchanges for both". Two or three of the seventeen look like real frontier errors (an extra exchange on an unrelated order, one order updated of three requested) and deserve the fail.
3. Float noise on 74 of 102 differing writes (above).
4. Three tool bodies. `calculate` raises `NameError: decimal` on 49 results and 33 replayed reads. `modify_pending_order_address` refuses with "Order is not pending" after an item modification on the same order, 11 writes, where the recording accepted; the corpus is the truth here, whatever rule the body inferred. `get_order_details` raises a bare `KeyError` on an unknown id, 155 results, where the corpus shows "Error: Order not found"; the miner recorded that error shape and the body ignored it.
5. The policy gate's arity bug (above).
6. Stale files. 111 re-roll files dated 2026-08-29 sit under `runs/` with a provider HTTP 400 ("messages: empty array") from a build that seeded re-rolls without a confirmed recording. Build 8 did not write them and its gate did not count them, but anything that globs `runs/` does.

**The model's own, in the re-rolls.** The re-roll model drops the `#` from order ids: 132 lookups where the user had said "#W…" and the call went out as "W…", 1,204 recorded calls that never did, and 102 re-rolls that never recovered. It also calls the name lookup with empty names instead of asking, 122 of 410 calls. Both are what this model does with a tool schema that carries no description of the id format, so the second half of each belongs to us: the traces declare the descriptions and the sigs carry none.

**The model's own, in the recordings.** 51 replayed reads are `both_refused`: the recorded frontier looked up a user by a wrong email or name and got "User not found", and our body said the same. 96 replays end in a transfer to a human because the recording did. These are the customer's real failures replayed faithfully, and the fidelity check treats them as agreement, which is right.

In short: the Environment replays the customer's own Runs well (2,556 of 2,561 reads agree, 458 of 575 writes agree exactly and 74 more agree past the sixth decimal), and nearly every failure on the way to a Verdict is on our side of the line, in the Simulated user and the judge, before the model gets to make its own.

## Builds 9 and 10 (2026-09-03): a crash and a stall, found on 2026-09-06

Neither was written up when it ran; both were found in worktrees under `/private/tmp` three days later, which is the reason for D139 (one table per build, printed by a script, into this file every time).

Build 9, retail, `opencode-go/glm-5.3-flash`, 14:17. Crashed in `compile_tools` after five attempts each ended in a read timeout on the provider (`RetryExhausted`). Eight calls, no tool body written, the intent gate red with 0 of 86 Tasks grounded because nothing downstream ran. It says nothing about the harness beyond the fact that a provider timeout kills a build instead of marking the tool assisted.

Build 10, airline, `openai/gpt-5.6-luna`, 18:18, 200 traces, 86 Tasks, two rounds. 1,899 calls, 0.65 dollars after repricing (the file `budget.json.mispriced` beside it is the same ledger at the reseller's rate). Round 1: 32 Tasks with a Reference, 2 trusted Verifiers, 41 fidelity rulings. Round 2 moved no count and the loop exited stalled. The red lights are build 8's: one tool body crashes on 20 of 24 calls, one answers every call the same way, three tools miss the replay bar (`payment_methods` and `available_seats, prices` differ; one write tool at 0 of 33), the tau2 export finds 86 overlay conflicts, `compile_policy` still fails every predicate (the arity was fixed on 3 September; the gate then raised `NameError: called_before` because it ran the rule without the helpers the compiler's sandbox prepends, fixed 6 September; the 118 stored constraints now pass the gate with 78 compiled), and the D79 suite passes 2 of 32 Verifiers. Nothing in the loop could act on any of it: round 2 was the same build again. That is the evidence behind D135 and D136.

## Build 11 (2026-09-06): the D135 experiment, code driver against the model driving

D135 to D138 call this "build 9"; the number moved once the worktree builds of 3 September above were found and written up as 9 and 10. Both arms ran on retail with `openai/gpt-5.6-luna`, on two copies of build 8's workdir (`.work-b9-code`, `.work-b9-agent`), so the caches were equally warm, under a 25 dollar ceiling with eight workers, on commit a011af6 with the Runner re-frozen (runner version 28d8274). Both tables were printed by `scripts/build_table.py` and the numbers below are copied from them; dollars are this build's spend on top of build 8's 6.49.

| | build 8 (code driver) | build 11, code driver | build 11, model driving |
|---|---:|---:|---:|
| Tasks covered | 20 of 205 (9.8%) | 48 of 205 (23.4%) | 61 of 205 (29.8%) |
| Traces confirming their Reference | 343 of 456 (75.2%) | 347 of 456 (76.1%) | 423 of 456 (92.8%) |
| Writes replaying exactly | 472 of 575 (82.1%) | 452 of 575 (78.6%) | 546 of 575 (95.0%) |
| Gates green | 9 of 15 | 9 of 15 | 10 of 15 |
| Rounds | 2, stalled | 2, stalled | 2, stalled |
| Dollars | 6.49 | 1.61 | 1.92 |
| Mechanic calls | none | none | 65 over 25 turns: repair_refuse_task 41, repair_recompile 7, status 6, build 6, replay 4, compile_tool 1 |

**What moved coverage from 20 to 48 on the same driver** is the three hand fixes of D136 (the policy gate running predicates like the sandbox, the Simulated user answering confirmations and known facts from the recording, the judge abstaining on evidence it was not given): `compile_policy` went green, and the Tasks with a confirmed Reference more than doubled. None of that is the driver.

**What the model driving bought on top** is one tool body. Both arms recompiled every tool (the compiler module hash changed, so the cache did not serve build 8's bodies). The compiler's fourth attempt at `exchange_delivered_order_items` in the code arm crashed on all 67 calls (`AttributeError: 'dict' object has no attribute 'available'`), and the compiler kept it anyway: `compile_env.compile_tool` keeps the last attempt's body when none passes, not the best one, and the third attempt had passed 44 of 44 replayed writes. That one body cost the code arm 94 replay misses and 38 Tasks whose seed Trace calls it ("an assisted tool whose body..." in `task_status.json`). The model arm's compiler got a passing body on its fourth attempt, and the tool replays 95 of 95. The 13 point gap between the arms is that body, not the mechanic: every one of the mechanic's 48 repair requests changed nothing (below).

**Why both arms stalled after two rounds.** Five things on our side, none of them a gate being wrong:

1. The recompile hint goes nowhere. `repair_recompile(name, hint)` writes the hint to `tool_lessons.json` and reruns `compile_tools` narrowed to the tool, but nothing reads the lesson into the compiler prompt (`memory.lesson_for` has no caller) and the stage's cache identity does not include it, so all seven recompiles printed "5 stages, 5 from cache".
2. The refuse verb is inert. `repair_refuse_task` appends to `repairs/repair_refuse_task.jsonl` and nothing reads that file; `rounds.json` ends with `"refused": {}`. The status tool suggested it for every `intent` red light, so the mechanic called it 41 times.
3. Intents are one shot. 160 of 205 Tasks fail the intent gate in build 8 and in both arms (145 "noun phrases with no span", 22 "not evidenced in every Run"), and an ungrounded Intent is a Task with no Verdict. `write_intent` queries once and records the ungrounded phrases as the reason; nothing feeds them back for a second try.
4. The Examiner's round 2 nudge says "read the rulings and act" while the code requires a derive call every round. The model did what the nudge said (read the rulings, filed one finding, answered), so round 2 was recorded as "examiner failed: the model never called derive('all')".
5. The compiler keeps the last failing attempt instead of the best one (the exchange body above).

The table's own blind spots showed too: build duration, context fill, per-round spend after round 1 and the cause of 22 "unreadable" replay misses (a 160 character preview in `replays.json`) all printed `n/a`, as `docs/todo.md` predicted.

**Read as an experiment.** The driver did not decide this build; the fixes before it and one compiler coin flip did. The mechanic read the red lights correctly (it asked for exactly the recompiles and the intent repairs the gates named) and had no verb that worked. So the comparison D135 asks for is not yet made: it needs the five fixes above, then the same two arms again. Nothing here says the graph was the problem, and nothing yet says the model driving is worth its extra 0.31 dollars.

### The mechanic's session in the model arm, call by call

Read from `.work-b9-agent/builder/session.jsonl` and `examiner/session.jsonl`. Times are seconds from the first Builder message.

| when | who | what | what came back |
|---:|---|---|---|
| 3s | Builder | `status` | 380 red lights over 30 tools and 184 Tasks (build 8's rulings, copied in); the first 25 shown: `calculate` NameError on every call, `exchange_delivered_order_items` accepts an unknown payment method, `get_user_details` and `modify_pending_order_items` differ on hard columns, 13 intent lines |
| 5s | Builder | `repair_recompile` x4 (calculate, exchange, get_user_details, modify_pending_order_items), each with a hint | compile_tools ran (first compile in this workdir), the hint reached nothing |
| 83s | Builder | `status` | 372 red lights |
| 91s | Builder | `build environment` and 25 `repair_refuse_task` in one batch | the build ran the pipeline (intent, replay, rerolls); every refuse: "recorded (request in repair_refuse_task.jsonl)" |
| 1182s | Builder | `status` | 191 red lights over 3 tools and 169 Tasks: `calculate` 4 replay misses, `get_user_details` 1, `modify_pending_order_items` 5, and 169 intent lines of which 25 shown |
| 1187s | Builder | `repair_recompile` x3 with new hints, `repair_refuse_task` x15 | compile_tools ran again without the hint; the same bodies came back; every refuse recorded |
| 1204s | Builder | `build`, `replay task_2e134f9291e5`, `status`, one more refuse, `status` | 13 of 13 stages from cache; 186 red lights; 186 red lights |
| 1217s | Builder | answered with no tool call | round 1 Builder beat ends |
| 0s | Examiner | `derive all` | 175 Verifiers over 205 Tasks, 61 passed the D79 suite |
| 829s | Examiner | `read rerolls`, `read gates`, three `finding`s | each finding says the intent grounding is too strict; each carries `suggested: replay`, because `Finding.suggested` only allows compile_tool, replay, reroll or none |
| 837s | Examiner | answered "no verifier action remains until Builder replay repairs" | round 1 ends: fidelity 194, trusted 61 |
| 2060s | Builder | round 2 steer, then `status`, `build` | 439 red lights (the Examiner's rulings now included); 13 of 13 stages from cache |
| 2066s | Builder | finding 1: `replay`, `build` | 7 of 8 and 13 of 13 from cache |
| 2078s | Builder | finding 2: `replay`, `build` | same |
| 2090s | Builder | finding 3: `replay`, `compile_tool modify_pending_order_items`, `build` | 8 of 8, 5 of 5, 13 of 13 from cache |
| 887s | Examiner | round 2 steer "read the rulings and act", `read gates`, one `finding` | never called derive; the round is recorded as failed and the loop exits stalled |

Nothing the mechanic did could change an artifact: the recompile hint never reached the compiler, refusing a Task reached no gate, the intent red lights pointed at refusing, the Examiner's finding vocabulary had no word for "rewrite the Intent", and the round-2 steer to the Examiner did not ask for the one call the code required. The model read the lights right every time.

### Before build 12: what the grill of the same evening changed (2026-09-06)

Read back from the session above, six harness changes, all on our side (D136), none a model wrote:

1. `status` shows every red light, grouped by gate and failure kind, with a zoom (D140). The build 11 mechanic saw 25 lines of 380.
2. The Examiner's finding names the Builder verb and carries the hint, `repair_intent` and `repair_recompile` included (D141). The Runner is re-frozen for it.
3. The stall rule counts artifact changes and repair effects, not gate counts alone; a repair that changes nothing says so, with the reason (D142).
4. The Intent splitter keeps numbers, money and ids whole and grounds in two steps; `write_intent` gets three tries with feedback (D143). On build 8's stored Intents: 37 grounded before, 137 after, of 205.
5. The Builder and Examiner prompts are in the GEPA order with general examples (D144).
6. The table's three blind spots have records now: round timestamps for the duration, the context fill per round, and a `difference` record on every replay miss, so "unreadable" is gone from the causes (D139: the record first, then the number).

Build 12 runs both arms again on fresh copies of build 8's workdir.

## Build 13 (2026-09-06 to 2026-09-07): the model driving on the D145 to D153 code, 16 hours, dead in round 2

Launched on a fresh copy of the retail workdir with `--agent`, `openai/gpt-5.6-luna` driving both sessions, eight workers, a 25 dollar ceiling, on the working tree after D153 (the triage skill, the fact miner, the cache effect, the driver's build) and before D154. D154 to D158 were committed while it ran; the process kept its imports, so what follows is the D153 code. The table is `docs/builds/build-13-model-driving.md`; build 12's code arm (`docs/builds/build-12-code-driver.md`) is the comparison, the model arm of build 12 was still running when this was written.

| | build 12, code driver | build 13, model driving |
|---|---:|---:|
| Tasks covered (trusted) | 75 of 205 (36.6%) | 73 of 205 (35.6%) |
| Traces confirming their Reference | 432 of 456 (94.7%) | 334 of 456 (73.2%) |
| Writes replaying exactly | 550 of 575 (95.7%) | 430 of 575 (74.8%) |
| Rounds | 2, stalled | 2, died (D161) |
| Dollars | 8.65 | 11.96 (the cache saved 8.97) |
| Model calls | 17,952 | 27,549 |
| Build duration | 49m 27s | 16h 0m 21s |
| Mechanic calls | none | 127 over 132 turns: status 75, repair_recompile 27, repair_intent 16, replay 6, build 3 |

**The trust funnel, round 1.** 205 Tasks, 156 with every replayed Trace matching its recording, 142 with a confirmed Reference, 73 past the D79 suite, 73 trusted. The three stages lose Tasks for different reasons and only the first is the model's:

- Fidelity, 150 differing calls of 2,977 replayed: 126 are one `KeyError` in the body of the tool that replaces items on an order, 22 more are the same failure read back later in the Trace (the order still holding the old item), one is a balance off by 72 downstream of the same missed write, and none is float noise. The body memorised three literal item ids from the recorded calls into a dict instead of reading the products table. The mechanic asked for 19 recompiles of that one tool and none converged. Build 12's compiler drew a body that reads the table, which is the whole gap in the fidelity rows above: the same coin flip as build 11's exchange body. D162 makes it a gate: a body may not hold a literal that matches an id pattern, a Starting state row id, or a recorded argument value.
- Reference confirmation, 63 Tasks without one: 29 where the recordings disagree on the End state, 16 where the judge failed every candidate, 9 blocked by the crash above, 8 stale rows (the replays confirm on disk, `task_status.json` was not refreshed after the repair that fixed the read), one with every recording breaking a Hard constraint.
- The suite, 69 confirmed and not trusted: `mutation_flips` 30, `leak_check_clean` 26, `second_path_passes` 21 (never run: every re-roll failed, the D158 closing rule is not in this build), `unsolved_state_fails` 17, and 8 read-only Tasks failing five checks at once because their Verifier says only "no writes" and mines no answer atom. D156 and D157 address the state and the leak; the answer atom for a read-only Task is still open.

**Where 16 hours went.** The Builder's round 1 beat took 25 minutes (280 seconds of status and repairs, then a 20 minute build). The Examiner's `derive` took 3 hours 6 minutes on its first call and 10 hours 36 minutes on its second, the same `derive all` over the same artifacts: `derive_all` runs the Tasks one after another, the suite's loophole probe is one model Run per Task (173 probe Runs on disk) and nothing is memoised between two calls in the same round. That is 13.7 of the 16 hours in one serial function while eight workers sat idle; the fix is per-Task derivation in parallel under `--workers` with a cache keyed by the Verifier's inputs, listed in the todo.

**What the triage skill changed.** Build 12's model arm called `status` 4 times in 59 calls and repaired blind; this mechanic called it 75 times in 127, once per Task before every repair, zooming on the tool or Task the finding named (D150). The repairs it then asked for were the right ones (`repair_recompile` on the tools with red replay lights, `repair_intent` on the Tasks the intent gate named) and two Intents grounded in round 2 (`task_03d78b3f5cea` after four tries, `task_b72a88c48881` on the first). The other 27 repairs changed nothing they were asked to change, 19 of them the one body above: a right diagnosis and a verb whose only lever is another sample from the same model, which is what D162 replaces with a rule. The fact miner's effect on re-rolls cannot be read from this build: 438 of 468 re-rolls finished (93.6%) against 564 of 582 (96.9%) in build 12's model arm, but the runs ended on the D77 closing rule and the D151 name facts are not what stopped them.

**How it died.** Round 2's Builder beat built the target, then acted on the follow-up findings with narrowed recompiles and answered. `execute` replaces the store with the last run's artifacts, so the Examiner's handover lacked the Constraints and `derive` raised `KeyError: 'constraints'`; the round is recorded as failed and the loop exited "stalled". D153 had covered the beat with no build call; this is the beat with a build followed by repairs. D161 has the driver build the target whenever the last run was narrowed or aimed elsewhere.

**Also found.** The Examiner's context fill is recorded at 608.9 percent of the window at a turn end, with 5 fallback compactions: its `read` tool returns a whole record unclamped (`read gates` came back as the full rulings list), so one result can exceed the window six times over. And the table's "repairs that turned a red gate green" row still cannot place 16 of the 43 requests, the blind spot `docs/todo.md` records.

**Read as an experiment.** The model driving cost 3.31 dollars and 15 hours more than the code driver for two fewer trusted Tasks, and none of the difference is the driver: one memorised body, one serial stage, one partial store. What the mechanic did right, the triage before every repair and the Intent repairs, is now the harness's; what it could not do, write a body that generalises when its own model keeps memorising, becomes a gate. The next build runs on D155 to D162 and the parallel derive.
