# First slice, offline part: numbers from the real tau2 retail corpus

Scope: `harness-design.md` section 11 steps 1 and 2, the Task clustering that step 5 rests on, the
shared Starting state of step 3, and the Gate A oracle replay of step 4. No model calls were made
anywhere in this run, so every LLM hook in the pipeline (`mine`'s kind classifier and result-schema
filler, `cluster`'s namer, `intent`, `compile_env`'s tool bodies, `policy`'s predicates) ran with
`model=None`. Nothing under `../data/raw` or `../vendor` was modified.

Input file: `../data/raw/claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json`,
sha256 `ed41dbd18c080154156484e3a0122c095e324a11367a640d88e15956daed7b9d`, 24,908,843 bytes.

Everything below was produced from the working directory
`.` unless stated.
Scratch scripts live in
`$SCRATCH`
(referred to below as `$SP`).

Baseline before any of this: `uv run pytest -q` in `harness/` is 756 passed, 0 failed.

---

## Step 1. Ingest

```
uv run python -m harness.cli ingest \
  ../data/raw/claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json \
  --workdir ../data/work
```

Output line: `ingest tau2_native: 456 runs, 3591 tool calls, 109 errors, 0 truncated, gate pass`.

| Metric | Value |
|---|---|
| Traces (Runs) written | 456 |
| Tool calls | 3,591 |
| Turns | 13,486 |
| Tool-call errors | 109 |
| Truncated results | 0 |
| Unresolved calls (no result and no error) | 0 |
| `gate_ingest` | pass, no failures |
| Grader sidecars written | 456 |
| Format detected | `tau2_native` |

Error classes, all assigned by code rule (`classified_by: rule`, D67): `not_found_entity` 60,
`business_error` 49. No `unknown`, so `unknown_error_flags` is empty.

### Hand count against the raw JSON

`uv run python $SP/handcount.py` walks the raw file directly, importing nothing from `harness`.

Corpus level: 456 simulations, 114 tasks, 3,591 `tool_calls` across all messages (all with
`requestor: assistant`), roles assistant 6,743 / user 3,152 / tool 3,591, and 109 tool messages with
`error: true`. Every one of those four numbers equals what ingest reported.

Three Runs checked call by call:

| Raw sim index | id | task_id / trial | Raw messages | Raw tool calls | Trace turns | Trace tool calls | Call names identical |
|---|---|---|---|---|---|---|---|
| 0 | `4bec2b80-1781-4799-a103-037acd71715d` | 6 / 0 | 28 | 6 | 28 | 6 | yes |
| 1 | `ee6707bb-dbcb-4356-a43d-d91194bcf895` | 2 / 0 | 34 | 10 | 34 | 10 | yes |
| 2 | `526bdc8f-ca8e-4f0d-8406-daa54fa6c3f1` | 1 / 0 | 28 | 6 | 28 | 6 | yes |

Each trace's `raw_ptr` points at the right `sim_index`, and the ordered call names match the raw
messages one for one (for example sim 0: `find_user_id_by_name_zip, get_user_details,
get_order_details, get_product_details, get_product_details, exchange_delivered_order_items`).

### Gate checks

- Hash stability (D66): a second ingest into a separate workdir produced an identical list of 456
  trace hashes. Command: `uv run python -m harness.cli ingest <same file> --workdir data/work2`, then
  a list comparison of `ingest_summary.json`; result `True`.
- Grader stripping (D66, D89): `grep -lE '"reward_info"|"action_checks"|"nl_assertions"|"evaluation_criteria"' ../data/work/traces/*.json`
  returns 0 files. The fields are in `../data/work/grader/<trace_id>.json` instead, 456 of them.

### One thing the export does not carry

`tools_declared` is empty on all 456 traces: the Sierra export has no `tools` list as sent. All 456
traces share one system prompt. Consequence for step 2: there is no declared schema to merge, so
every `ToolSig` is `source: observed` and D72's `declared` flag is false everywhere.

---

## Step 2. Mine

```
uv run python $SP/step2_mine.py
```

which calls `mine_tools(traces, model=None)` and `mine_schema(traces, db_json_path=None, model=None)`
over the 456 traces and writes `../data/work/tool_sigs/*.json` and `../data/work/entity_schema.json`.
`db_json_path` is deliberately `None`: feeding tau2's own `db.json` in would make step 4's comparison
circular.

`gate_tools`: pass. `{tools: 15, thin: 0, unclassified: 2, writes: 7}`. Every ToolSig has at least
three observed calls, so none needed the `llm` flag.

Ground truth for the comparison came from
`../vendor/tau2-bench/src/tau2/domains/retail/tools.py`, read by AST (`$SP/tau2_sigs.py`), not by
import. tau2 marks tools with `@is_tool(ToolType.READ | WRITE | GENERIC)`; the retail toolkit has 16
such methods (plus one `THINK` tool commented out).

### Per tool

| Tool | tau2 kind | mined kind | Arg names match | Calls | Errors | Result fields | Note |
|---|---|---|---|---|---|---|---|
| calculate | generic | read | yes (`expression`) | 20 | 0 | 1 | **kind mismatch**, flagged `unclassified` |
| cancel_pending_order | write | write | yes | 117 | 1 | 14 | name prefix rule |
| exchange_delivered_order_items | write | write | yes | 149 | 13 | 14 | name prefix rule |
| find_user_id_by_email | read | read | yes | 124 | 28 | 1 | name prefix rule |
| find_user_id_by_name_zip | read | read | yes | 386 | 26 | 1 | name prefix rule |
| get_item_details | read | **not mined** | n/a | 0 | 0 | n/a | **never called in the corpus** |
| get_order_details | read | read | yes | 1,315 | 4 | 14 | name prefix rule |
| get_product_details | read | read | yes | 492 | 0 | 3 | name prefix rule |
| get_user_details | read | read | yes | 449 | 0 | 6 | name prefix rule |
| list_all_product_types | read | read | yes (none) | 29 | 0 | 50 | name prefix rule |
| modify_pending_order_address | write | write | yes | 88 | 0 | 14 | observed effect |
| modify_pending_order_items | write | write | yes | 169 | 18 | 14 | observed effect |
| modify_pending_order_payment | write | write | yes | 4 | 0 | 14 | name prefix rule |
| modify_user_address | write | write | yes | 39 | 0 | 6 | name prefix rule |
| return_delivered_order_items | write | write | yes | 190 | 19 | 14 | observed effect |
| transfer_to_human_agents | generic | read | yes (`summary`) | 20 | 0 | 1 | **kind mismatch**, flagged `unclassified` |

### Matches and mismatches

- Tools: 15 of tau2's 16 mined. The one gap, `get_item_details`, appears zero times in the raw file
  (`grep -c get_item_details` on the raw JSON returns 0). Mining did the designed thing: it flagged
  nothing and synthesized nothing.
- Argument names: 15 of 15 exact, both directions. No missing argument, no extra argument, on any
  tool. Argument names come from observed calls only, so a never-supplied optional argument would be
  invisible; on this corpus none was.
- Read/write class: 13 of 15 exact. Every one of tau2's 7 write tools is mined as `write`, and every
  one of its 7 read tools as `read`. The set of write tools is identical to tau2's, which is what the
  Category partition in step 3 depends on.
- The 2 mismatches are both tau2 `GENERIC` tools, `calculate` and `transfer_to_human_agents`. The
  code rules propose read or write from name prefixes only; neither name matches a prefix, so both
  fell through to D70's default, `kind: read` with `kind_confidence: low` and `unclassified: true`,
  reason "no name rule matched, default read and unclassified (D70)". That is the decided behaviour,
  not a silent error: both land in the setup review. Worth noting anyway: `propose_kind` in
  `mine.py` has no path that returns `generic` at all, so a generic tool can only reach that class
  through the LLM hook (`classify_kind`, which does allow `generic`), and that hook did not run here.
  Neither tool touches state, so classifying them read is harmless for the Verdict.
- Evidence for the three `observed effect` classifications is real inverse-replay evidence, for
  example `modify_pending_order_address` was called write because of an observed change in
  `get_order_details.address.address1/address2/zip`.

### Schema

`mine_schema` found exactly tau2's three tables (`orders`, `products`, `users`) and 23 columns.
Against tau2's `data_model.py`:

- `users`: 6 columns, identical set to tau2's `User` model.
- `products`: 3 columns, identical set to tau2's `Product` model.
- `orders`: 14 columns, identical set to tau2's `Order` model, including all 7 fields that are
  optional and absent from the shipped `db.json` (`cancel_reason`, `exchange_items`,
  `exchange_new_items`, `exchange_payment_method_id`, `exchange_price_difference`, `return_items`,
  `return_payment_method_id`). Those were recovered from post-write results.

Id patterns learned: `orders.order_id` `^#W\d{7}$`, `products.product_id` `^\d{10}$`,
`users.user_id` and `orders.user_id` `^[A-Za-z]+_[A-Za-z]+_\d+$`. All three match tau2's real ids.
Two patterns are loose (`orders.exchange_payment_method_id` and `orders.return_payment_method_id` are
`^[A-Za-z0-9_]+$`), which is honest given how few values were seen.

---

## Step 3. Cluster

```
uv run python $SP/step3_cluster.py
uv run python $SP/step3b.py
```

`cluster_runs(traces, tool_sigs, threshold=..., model=None)`. Truth for the comparison is the raw
file's `task_id`: 114 task ids, 4 trials each, 456 Runs.

At the shipped default `DEFAULT_THRESHOLD = 0.3`:

| Metric | Value |
|---|---|
| Categories | 23 |
| Tasks | 74 |
| Unguarded Tasks (fewer than 3 Runs, D81) | 12 |
| Single-Run Tasks | 9 |
| Largest Task | 58 Runs |
| Task size histogram | 1:9, 2:3, 3:11, 4:26, 5:4, 6:2, 7:2, 8:8, 10:1, 11:2, 12:1, 18:1, 19:1, 24:1, 31:1, 58:1 |

Written to `../data/work/tasks/*.json` and `../data/work/categories.json`.

### Agreement with tau2's task_id groupings

The headline number asked for, "same task_id, 4 trials, should land in one Task":

| Threshold | Tasks | Pair agreement (recall) | Pair precision | Pair F1 | task_id groups intact | Purity |
|---|---|---|---|---|---|---|
| 0.2 | 41 | **0.868** | 0.080 | 0.146 | 87 / 114 (76.3%) | 0.318 |
| **0.3 (shipped)** | **74** | **0.836** | **0.165** | **0.276** | **80 / 114 (70.2%)** | **0.531** |
| 0.4 | 96 | 0.844 | 0.380 | 0.524 | 83 / 114 (72.8%) | 0.697 |
| 0.5 | 119 | 0.827 | 0.573 | 0.677 | 79 / 114 (69.3%) | 0.805 |
| 0.6 | 141 | 0.787 | 0.664 | **0.720** | 74 / 114 (64.9%) | 0.866 |
| 0.7 | 173 | 0.694 | 0.695 | 0.695 | n/a | n/a |

Pairs: 114 groups of 4, so 684 same-task_id pairs. At 0.3, 572 of 684 land together.

Read the agreement number with the precision number beside it. Pair agreement alone is maximised by
merging everything, and at 0.3 that is close to what happens: 165 of every 1,000 pairs the clustering
calls "same Task" actually share a tau2 task_id, and one Task holds 58 Runs from many different task
ids. The `DEFAULT_THRESHOLD = 0.3` comment in `cluster.py` says it was calibrated on the 3-run
fixture and that "the full slice revises this number". The full slice revises it: **0.6 is the best
of the values tried on pair F1 (0.720 against 0.276), at the cost of more unguarded Tasks (52 against
12)**. This is a proposal, not a change: I did not edit `cluster.py`, which the cluster_intent agent
owns.

### The ceiling on this number

Category membership is decided by the confirmed write-tool signature, which is upstream of the
threshold. Only **91 of 114 tau2 task ids (79.8%) have all 4 trials writing through the same tool
set** on this corpus: on 23 task ids the agent wrote differently across trials (a failed trial, an
extra `modify_`), so those 4 trials cannot land in one Task at any threshold. So 79.8% is the ceiling
for "task_id groups intact", and the observed 70.2% at 0.3 is 88% of that ceiling. That is a
property of the traces, not a bug in `cluster.py`, and it is arguably the correct behaviour: a trial
that wrote something different is a different end state.

Categories are threshold-independent, 23 of them, one per observed write-tool signature (including
one empty signature for Runs that wrote nothing).

---

## Step 4. Starting state, compared with tau2's own db.json

```
uv run python $SP/step4_state.py     # build and compare
uv run python $SP/step4b.py          # the same comparison under strict equality
```

`build_starting_state(traces, schema, ../data/work/state, tasks=..., tool_sigs=...)` over all 456
traces. Inverse replay (D33): a row's shared value is the latest sighting no write had yet touched.

| Table | Rows the traces touch | Rows in tau2's db.json | Touched rows found in tau2 | Exact match | Exact-match rate | Coverage of the tau2 table |
|---|---|---|---|---|---|---|
| orders | 161 | 1,000 | 161 | 161 | **100%** | 16.1% |
| products | 38 | 50 | 38 | 38 | **100%** | 76.0% |
| users | 53 | 500 | 53 | 53 | **100%** | 10.6% |
| **total** | **252** | **1,550** | **252** | **252** | **100%** | 16.3% |

No row was invented: every one of the 252 rows the traces touch exists in tau2's db.json, and none is
outside it. `assumptions.json` is empty, meaning no row was ever seen only after a write, so inverse
replay never had to keep a post-state. 74 Task overlays were written (D74).

**First 5 mismatches: there are none.** All 252 rows match field for field.

There is one difference, and it is uniform rather than a per-row mismatch, so it is stated here
rather than in a list of five. Under byte-strict `==` the rate is orders 0 / 161, products 38 / 38,
users 53 / 53. The reason is key presence, not value: our 161 order rows carry all 7 optional Order
fields explicitly as `null` (`cancel_reason`, `return_items`, `return_payment_method_id`,
`exchange_items`, `exchange_new_items`, `exchange_payment_method_id`, `exchange_price_difference`),
because `get_order_details` returns the full model, while tau2's shipped `db.json` omits them. Every
value compares equal once an absent key and an explicit `null` are treated as the same thing, which
is the treatment used for the 100% figures above. It matters for the tau2 export: emitted rows will
be a superset of tau2's keys, which `RetailDB.model_validate` accepts (confirmed in step 5, both
databases replay identically).

---

## Step 5. Gate A, oracle replay through tau2's own tools

Environment: a throwaway venv outside the repository, with tau2 installed non-editable from the
vendored source so nothing is written into `../vendor`.

```
uv venv $SP/tau2env --python 3.12
uv pip install --python $SP/tau2env/bin/python ./vendor/tau2-bench     # run from monitoring-tool/

uv run python $SP/step5a_select.py                                     # pick the Runs, dump their calls
$SP/tau2env/bin/python $SP/step5b_replay.py ../data/work/state/db.json      $SP/gateA_ours.json
$SP/tau2env/bin/python $SP/step5b_replay.py \
    ../vendor/tau2-bench/data/tau2/domains/retail/db.json $SP/gateA_tau2db.json
uv run python $SP/step5c_compare.py ours ; uv run python $SP/step5c_compare.py tau2db
uv run python $SP/step5d_endstate.py ours ; uv run python $SP/step5d_endstate.py tau2db
```

The replay process constructs a fresh `RetailDB` per Run from the given database, wraps it in tau2's
real `RetailTools`, and calls the Reference's own tool calls with their recorded arguments in order.
The comparison process is separate and runs in the harness venv, so `harness.shared.canon` does the
canonicalization (D39) and the column classes come from the mined `EntitySchema` (D73).

Selection (D81): 20% of each Task's Runs held out, at least one, from the 74 Tasks at threshold 0.3;
Tasks marked unguarded contribute only to the seed pool. Seed pool 356 Runs, held-out pool 100. From
those, 20 seed Runs across 20 distinct Tasks and 10 held-out Runs across 10 distinct Tasks were taken
by walking one Task at a time, so no sample is one Task repeated. 257 tool calls in total.

### Result, replaying against our built Starting state

| | Seed (20 Runs) | Held out (10 Runs) |
|---|---|---|
| Tool calls replayed | 168 | 89 |
| Successful write calls | 37 | 20 |
| **Write match rate after canon** | **37 / 37 = 100%** | **20 / 20 = 100%** |
| Successful read calls | 125 | 65 |
| Read match rate after canon | 125 / 125 = 100% | 65 / 65 = 100% |
| **Semantic read mismatches (canon differs, class `semantic`, would need a judge)** | **0** | **0** |
| Hard read mismatches | 0 | 0 |
| Calls the trace recorded as errors | 6 | 4 |
| Errors reproduced as errors by tau2 | 6 / 6 = 100% | 4 / 4 = 100% |
| Runs where every write reproduced | 20 / 20 | 10 / 10 |
| End-state rows written by the Run | 33 | 17 |
| **End state match after canon** | **33 / 33 = 100%** | **17 / 17 = 100%** |

Gate A passes on both splits, seed and held-out reported separately as the design asks.

### Control

Replaying the same 30 Runs against tau2's own shipped `db.json` instead of ours gives **identical
numbers in every cell**. Our 252-row Starting state is as good as tau2's 1,550-row one for the
Reference calls these Runs make. Nothing in the sample depends on a row our inverse replay dropped.

### Two things the raw numbers hide

1. **Error text carries a prefix.** All 10 recorded errors reproduce as errors with the right cause,
   but the recorded payload is tau2's harness string `"Error: User not found"` while the tool itself
   raises `"User not found"`. All 10 of 10 differ by exactly that `Error: ` prefix and nothing else.
   That is a harness wrapper, not a fidelity failure, but a generated tool that copies the observed
   payload verbatim would double the prefix. Worth a canon rule or an ingest strip.
2. **The End state number in the table is over written rows, deliberately.** The first version of
   this check compared the replayed database against the last sighting of each row anywhere in the
   trace, and got 94 / 95 on the seed split. The single mismatch was
   `users.ethan_garcia_1261.payment_methods` in Run `8e32436d-cd3b-49f6-b32a-e2bd92b3db82`: the trace
   read the user at call 1 (gift card balance 86.00) and then spent 41.92 of it at call 7 without
   ever reading the user again. The replay's 44.08 is correct and the reference was stale. Counting
   only rows observed after the trace's last write leaves 8 comparable rows on the seed split and 0
   on the held-out split, which is too thin to be a gate. The number reported above instead compares,
   for every row a Run wrote, the database at the end of the Run against the last write result that
   returned that row, which is what the trace actually states about its End state: 50 rows across the
   30 Runs, all matching. Both framings are in `$SP/gateA_result_*.json`.

---

## What could not run, and why

| Section 11 step | Status | Why |
|---|---|---|
| 1 ingest | ran | |
| 2 mine | ran | LLM hooks skipped (`model=None`); no tool needed them, since none was thin and none lacked a result schema |
| Task clustering (needed by 5) | ran | `name_task` skipped, so Tasks have ids but no names |
| Starting state part of 3 | ran | |
| 3 `compile_env` tool bodies and `policy` | **not run** | The tool body is written by an LLM (`write_tool_body(model, ...)`) and each policy predicate is proposed by an LLM. With `model=None` there is nothing to gate, so the five compile gates, the 30-held-out-call replay fidelity per generated tool, and "the five files load in tau2's harness" are all untested here. Gate A below used tau2's own `tools.py`, exactly as the task specified, so it tests the Starting state and the canonicalizer, not our generated tools. |
| 4 Gate A | ran, against tau2's tools | |
| 5 `verifier` for one Task | **not run** | Needs k frontier re-runs of a Candidate model and LLM Provenance classification. |
| 6 Runner with a cheaper model | **not run** | Model calls. |
| 7 `verdict` vs tau2 reward | **not run** | Depends on 5 and 6. The grader sidecars needed for it are on disk (456 of them) and untouched. |
| 8 `regrade` | **not run** | Depends on 7. |
| OTel-format repeat | **not run** | The OTel copy of these trajectories is not in `../data/raw`; only the two tau2-native files are. |

Also not run: `intent.py` (grounded Intent needs an LLM to write the line), and the second raw file
`claude-sonnet-4-5_enabled_retail_gpt-5.2_4trials.json`, which was left alone since the slice names
the Claude 3.7 file.

## Findings worth a decision

1. `DEFAULT_THRESHOLD = 0.3` in `cluster.py` is wrong on the full corpus: pair F1 0.276 against 0.720
   at 0.6, purity 0.53 against 0.87. D97 said the slice revises this number; here is the number.
   Owner: the cluster_intent agent.
2. `propose_kind` in `mine.py` cannot return `generic`. tau2's two generic tools both land on D70's
   read-and-flagged default. Harmless for the Verdict on this domain, but the rule table has a hole.
   Owner: the mine agent.
3. Recorded error payloads carry a `Error: ` prefix the tool itself does not raise. Either strip it
   at ingest or teach the canonicalizer, before a generated tool copies it. Owner: ingest or canon.
4. The End state a trace states is the row its writes returned, not the last row it read. Any Gate A
   implementation that uses "last sighting" will show false mismatches (1 in 95 here). Worth writing
   down where `validate.py` implements Gate A.
5. Our order rows carry the 7 optional Order fields as explicit `null`; tau2's `db.json` omits them.
   Equal in value, different in bytes. The tau2 export should say which shape it emits.

## Files produced

- `../data/work/` (not committed): `raw/<sha256>.json`, `traces/` 456 files, `grader/` 456 files,
  `ingest_summary.json`, `tool_sigs/` 15 files, `entity_schema.json`, `tasks/` 74 files,
  `categories.json`, `state/db.json`, `state/assumptions.json`, `state/overlays/`.
- `$SP/`: `handcount.py`, `tau2_sigs.py`, `slice_common.py`, `step2_mine.py`, `step3_cluster.py`,
  `step3b.py`, `step4_state.py`, `step4b.py`, `step5a_select.py`, `step5b_replay.py`,
  `step5c_compare.py`, `step5d_endstate.py`, and the JSON results beside them.
