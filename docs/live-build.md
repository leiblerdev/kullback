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
