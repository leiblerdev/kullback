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
