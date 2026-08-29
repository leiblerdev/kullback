# Cross-domain check: airline and telecom against retail (2026-08-28)

The question: is the offline slice (ingest, mine, cluster, Starting state, Gate A) tuned to tau2 retail, or does it hold on a domain it never saw? The answer is that parts of it are tuned to retail, in five concrete places listed under Judgement. Nothing in the harness was changed or retuned for this run; the default settings (`DEFAULT_THRESHOLD = 0.4` in `cluster.py`, D100 similarity) were used as they are.

Files used, matching the retail slice's choice (one claude-3-7-sonnet file per domain):

- airline: `data/raw/claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json`, sha256 `40a2c6a246eab27db5cdefda895fbe7f44be78a23d300817248534aa606d66de`
- telecom: `data/raw/claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json`, sha256 `f49e540896fe91ab8631647f02eb777ef6fed5a6ebec545f43e71d0504e227b2`

Workdirs: `data/work/xdomain-airline/`, `data/work/xdomain-telecom/`. Retail numbers are the ones in the README's measured section, except the cluster F1 at the 0.4 default, recomputed read-only against retail's existing workdir because the older table predates the D100 similarity fix.

Two bugs in the retail-derived Gate A scripts had to be fixed to get honest numbers at all, both in the scratch scripts and not in the harness: `plain()` only called `.model_dump()` on the outermost value, so airline's `search_direct_flight` (a bare list of pydantic models) serialized as repr strings and produced false mismatches; and the compare and end-state scripts assumed every DB dump is `{table: {row_id: row}}`, which crashed on telecom's `TelecomDB` where every table is a bare list.

## Ingest

| Metric | Retail | Airline | Telecom |
|---|---|---|---|
| Simulations (hand count) | 456 | 200 | 456 |
| Tasks (hand count) | 114 | 50 | 114 |
| Tool calls (hand count, all requestors) | 3,591 | 1,678 | 8,742 |
| by requestor `assistant` | 3,591 | 1,678 | 3,121 |
| by requestor `user` | 0 | 0 | 5,621 |
| Tool-call errors (hand count) | 109 | 27 | 149 |
| Ingest: runs / tool calls / errors / truncated | 456/3,591/109/0 | 200/1,678/27/0 | 456/8,742/149/0 |
| `gate_ingest` | pass | pass | fail |

Every hand count equals ingest's own count on all three domains. Ingest counting is domain-agnostic and held without change.

Telecom's ingest gate fails on one thing: trace `ed66fbbc-ee0c-4504-9ab4-9e53e01ea86f` (`termination_reason: max_steps`) ends with an assistant `get_details_by_id` call that has no tool result and no error, so `gate_ingest` reports "tool call ... has no result and no error" (1 of 8,742 calls). That is a raw-data artifact (the source simulation hit its step budget mid-call), and the gate is right to catch it. The 456 traces are still written.

The domain fact ingest surfaces without flagging: telecom's trajectories interleave two tool-calling actors in one trace, the assistant and the simulated user operating their own phone through a separate `user_tools.py` toolkit (`check_network_status`, `toggle_roaming`, `run_speed_test`, 20 more). Retail and airline have only the `assistant` requestor. `ToolCall.requestor` is preserved on ingest but nothing in `mine.py`, `cluster.py` or `compile_env.py` filters or branches on it. Every downstream telecom number below traces back to this.

## Step 2: ToolSigs mined against the real tools.py

| Metric | Retail (16 real tools) | Airline (14 real tools) | Telecom (13 real tools) |
|---|---|---|---|
| Tools mined | 15 | 14 | 38 |
| Real tools never mined (never called) | 1 | 0 | 4 (`disable_roaming`, `get_customer_by_id`, `get_customer_by_name`, `suspend_line`) |
| Mined names absent from real tools.py | 0 | 0 | 29 |
| Argument names, exact match | 15/15 | 14/14 | 9/9 (tools present in both) |
| Kind (read/write/generic), exact match | 13/15 | 12/14 | 8/9 |
| `gate_tools` | pass | pass | fail |

Airline mismatches: `book_reservation` and `send_certificate` are writes in tau2 and mined as read. Unlike retail's two mismatches (stateless generic tools landing on a harmless read default), these are real state-changing tools. Neither name matches `WRITE_PREFIXES = ("cancel_", "modify_", "update_", "return_", "exchange_")` in `mine.py`, and neither has an observed-effect signal (`book_reservation` creates a reservation that did not exist to be read before; `send_certificate` was never bracketed by two identical reads). So the airline Category write signature never includes either tool, and every booking and certificate issuance is invisible to Task clustering.

Telecom mismatch: `send_payment_request` (write in tau2, mined read). The other five called write tools (`enable_roaming`, `refuel_data`, `resume_line`, and two never called) were classified write entirely through the observed-effect fallback (D68), since none of telecom's write verbs (`suspend_`, `resume_`, `send_`, `enable_`, `disable_`, `refuel_`) match `WRITE_PREFIXES`.

The other 29 telecom "tools" are `user_tools.py` actions. One of them, `grant_app_permission`, is mined as write (observed effect on `check_app_status.value`) and lands in the Category write signature (`enable_roaming, grant_app_permission, refuel_data, resume_line`): the simulated user's own phone action pollutes the signature that is meant to describe what the agent did to the account. `gate_tools` fails on telecom (thin: 1, unclassified: 29) as a mechanical consequence.

Schema: `mine_schema` recovered 2 of 3 tables for airline (`reservations`, `users`, missing `flights`) and 4 of 5 for telecom (`customers`, `devices`, `lines`, `plans`, missing `bills`).

## Step 3: cluster F1 against tau2 task_id groups

| Threshold | Retail F1 | Airline F1 | Telecom F1 |
|---|---|---|---|
| 0.3 | 0.719 | 0.801 | 0.169 |
| 0.4 (default) | 0.717 | 0.788 | 0.207 |
| 0.5 | 0.710 | 0.718 | 0.217 |
| 0.6 | 0.685 | 0.659 | 0.233 |

Information only; the default was not changed. Airline at the default (0.788) is comparable to retail (0.717); 66.0% of airline's 50 task_ids have all trials sharing one write signature (the clustering ceiling), close to retail's 79.8%. Telecom collapses to a quarter of that at every threshold, from two compounding causes: the `grant_app_permission` contamination above, and a lower ceiling (only 56.1% of telecom's 114 task_ids share one write signature) on a corpus that is combinatorial personas over a handful of scenario types with a shared tool vocabulary, which gives token Jaccard little signal.

## Step 4: Starting state against the real database

| | Retail | Airline | Telecom |
|---|---|---|---|
| Tables recovered / real tables | 3/3 | 2/3 (`flights` missing) | 4/5 (`bills` missing) |
| Rows touched and found in the real DB | 252/252 | 147/148 exact | 6/6 |
| Exact match rate | 100% | 99.3% | 66.7% (mostly byte format, one semantic) |
| New rows created by traces (correctly absent from seed) | 0 | 3 | 0 |

Airline: `reservations` 114/114 exact (3 more are new bookings, correctly not counted); `users` 33/34. The one mismatch (`yara_garcia_1905.payment_methods`, gift card 338.0 versus seed 152.0) is best explained by the kind mismatch above: `book_reservation` and `send_certificate` are mined as read, so inverse replay never marks the rows they touch as "after a write" and can pick the wrong sighting. This is an inference from the two findings together, not re-traced call by call.

`flights` is never recovered despite 338 search calls. Cause, read from the code: `_is_id()` in `mine.py` (`name == "id" or name.endswith("_id")`) does not recognise `flight_number` as an id column, so `_table_of()` never proposes a `flights` table.

Telecom: only 1 of 4 customers, 1 of 9 devices, 3 of 9 lines and 1 of 5 plans appear in the corpus at all (it is built around one default customer, `C1001`/`L1002`). `plans` 1/1 exact; `lines` 2/3; `customers` and `devices` 0/1, purely a datetime separator difference (`"2025-01-15 10:30:00"` versus `"...T..."`) that the retail-tuned `norm()` does not paper over. The one semantic mismatch, `lines.L1002.data_used_gb` (15.1 built versus 8.7 real), traces to a structural fact: all 2,285 telecom tasks (100%, checked directly) carry `initial_state.initialization_actions` in `tasks.json`, real tool calls applied to the shared DB before the conversation starts and invisible to the ingested trajectory. Retail and airline have zero such tasks. The "one shared, trace-reconstructible world" Starting-state model is the wrong model for this domain, not just an imperfect one.

## Step 5: Gate A, oracle replay

Airline (seed 20 Runs, held-out 10 Runs):

| | Seed, ours | Seed, control (tau2 db) | Held-out, ours | Held-out, control |
|---|---|---|---|---|
| Write match | 10/27 | 27/27 | 4/14 | 14/14 |
| Read match | 129/183 | 183/183 | 51/74 | 74/74 |
| Errors reproduced | 3/3 | 3/3 | 5/5 | 5/5 |
| End state (write-returned rows) | 7/19 | 19/19 | 3/10 | 10/10 |

The control is 100% in every cell on both splits, matching retail, and the `Error: ` prefix finding reproduces. Every "ours" loss traces to the missing `flights` table: `get_flight_status`, `search_*` and flight-touching writes raise "Flight <n> not found" against the correctly empty table, and the failure cascades through the rest of that Run's writes. Constructing `FlightDB` from the built db.json at all required backfilling `flights: {}` by hand in the scratch script.

Telecom (seed 20 Runs, held-out 10 Runs):

| | Seed, ours | Seed, control | Held-out, ours | Held-out, control |
|---|---|---|---|---|
| Write match | 0/16 | 7/16 | 0/6 | 3/6 |
| Read match | 69/332 | 77/332 | 31/152 | 37/152 |
| Errors reproduced | 8/8 | 8/8 | n/a (0 recorded) | n/a (0 recorded) |
| End state (write-returned rows) | not measurable | not measurable | not measurable | not measurable |
| Calls with no matching method on `TelecomTools` | 240/356 | 240/356 | 105/158 | 105/158 |

Even the control is far from 100%: 240 of 356 seed calls and 105 of 158 held-out calls are the simulated user's own actions with no method on `TelecomTools`, and they fail identically on either database. Of the tools that do match, the control's shortfall is the initialization_actions gap again: replay never applies a task's pre-conversation setup, so a legitimate difference shows up even against tau2's own file.

End state could not be measured for telecom on either database. Telecom's write tools mostly return a confirmation string (`enable_roaming` returns "Roaming enabled successfully") or an id-less summary dict (`refuel_data` returns `message`, `new_data_refueling_gb`, `charge`, no `line_id`), never the updated row the way every retail and airline write does. `extract_rows` and `match_table` can only recognise a row they are handed, so there is nothing to extract. That is a convention difference, not a Starting-state defect.

## What crashed

1. Airline, constructing `FlightDB` from the built Starting state: pydantic `ValidationError`, `flights` field required. Cause: `_is_id()` misses `flight_number`, so `flights` never becomes a table, and tau2's `FlightDB` rejects a db.json without it. Worked around in the scratch script only.
2. Telecom, indexing the replayed database by row id: `AttributeError: 'list' object has no attribute 'get'`. Not a harness bug; `TelecomDB` stores every table as `List[Model]`, not `Dict[str, Model]` like the other domains. Fixed in the scratch comparison script.
3. Telecom, constructing `TelecomDB`: same failure mode as 1 (`bills` required, never mined), pre-empted by the backfill. The root cause differs: `Bill` rows do have an `_id` key (`bill_id`), but `_table_of()` walks a row's `_id` keys and returns the first whose singular form is a token in the tool's name. `Bill` also carries `customer_id`, `customer` is a token of `get_bills_for_customer` and `bill` is not, so every bill row is filed under `customers`. Confirmed directly: the mined `customers` table carries `bill_id`, `period_start`, `total_due`, `line_items` and `status`, none of which are `Customer` fields.

## Judgement

Tuned to retail, with evidence:

- `WRITE_PREFIXES` and `READ_PREFIXES` in `mine.py` are retail's own verb vocabulary. They transfer to airline's `cancel_` and `update_` by coincidence and miss `book_` and `send_` (two real writes lost); they miss essentially all of telecom's write vocabulary (one of six called write tools lost even after the observed-effect fallback).
- `_is_id()` (`name == "id" or name.endswith("_id")`) is retail's id convention; airline's `flight_number` defeats it completely (0% of the `flights` table recovered despite 338 calls).
- `_table_of()`'s foreign-key tie-break (does an id's singular form appear in the tool name) breaks with two or more `_id` keys on one row; telecom loses the whole `bills` table to `customers` because of it.
- Nothing reads `ToolCall.requestor`. Invisible on retail and airline (one actor each), and the single largest cause of telecom's numbers: 38 mined tools instead of 13, a user action in the Category signature, cluster F1 at a quarter of retail's, 240 of 356 Gate A calls failing as `missing_tool` even against tau2's own database.
- Every retail and airline write tool returns the full updated row, and the End-state logic assumes this without saying so. Telecom's write tools return a string or an id-less dict, and the End-state check finds nothing on this domain on either database.

Held without change:

- Ingest: counts, format detection, `gate_ingest` and trace hashing matched a hand count exactly on all three domains, and telecom's one gate failure is a correct catch.
- `harness.shared.canon` and the Gate A method: 100% on airline's control, matching retail, once the two comparison-script bugs were fixed.
- The observed-effect kind classifier (D68) recovers five of six telecom write tools and two of airline's that the prefix rule misses. It is the one piece of `mine.py`'s kind logic that visibly compensates for the prefix list's retail shape.
- `gate_tools` and `gate_ingest` are domain-agnostic and correctly went red on telecom and green on retail and airline. The gates work; the miners feeding them are what is tuned.

What this means for the next build, in order: read `requestor` and mine only the agent's calls (user-side tools become part of the user simulator, not the Environment); replace the prefix lists with the observed-effect classifier as the primary signal and a small learned verb list as the fallback; widen `_is_id()` to any column that is unique per row and referenced by other rows, not only `_id` suffixes; make `_table_of()` prefer the id whose singular matches the tool's object noun and fall back to the id that is unique within the result; accept write results that are confirmation strings or partial dicts by reading the End state from the next read of the same row instead of from the write's return value; and support per-task initialization actions as a Starting-state overlay. Each of these has a test to write against airline and telecom first, then retail must still pass unchanged.

## Rerun from a committed script (2026-08-29)

The check above was run from scratch scripts kept outside the repository, and two bugs in them had
to be found before the numbers were honest. That is not a good place for the measurement that
decides whether a fix worked, so the check is now `scripts/xdomain_check.py`: one command, no model
call, ingest through Starting state, compared against the domain's own `tools.py` and seed database.

    uv run python scripts/xdomain_check.py retail airline telecom

Rebuilding it surfaced two more conventions the scratch scripts had wrong, both about absence and
neither a defect in the harness:

- **Null and absent are the same field.** Our rows are built from tool results, which write a
  declared-optional field as `null`; the seed file leaves it out. Counting that as a mismatch cost
  retail 161 of its 252 rows. With it settled, retail is 252/252 again.
- **The row key is not a row field.** A dict-shaped table is keyed by row id and a list-shaped one
  is not, and the old script wrote the key into the row before comparing, so every telecom row
  differed from the real one by an `id` field our harness never claimed.

Two numbers in the table above do not survive the corrected comparison, and both were measurement
artifacts rather than harness defects:

- Airline `users.yara_garcia_1905` is an exact match, not a mismatch. Our row carries
  `gift_card_6941833` at 152.0, which is the seed value; the 338.0 above came from the scratch
  script, not from the build. Airline is 148/148 exact on the rows it touched, plus 3 new bookings
  correctly absent from the seed.
- Telecom is 5 of 6 exact once the datetime separator is settled, not 4 of 6. The one real mismatch
  is `lines.L1002.roaming_enabled` (we build `false`, the seed holds `true`), which is the
  `initialization_actions` gap and nothing else.

Baseline at this commit, with the requestor filter in and nothing else fixed:

| | Retail | Airline | Telecom |
|---|---|---|---|
| Tool calls, of which by the assistant | 3,591 / 3,591 | 1,678 / 1,678 | 8,742 / 3,121 |
| Tools mined against real | 15 / 16 | 14 / 14 | 21 / 13 |
| Argument names exact | 15/15 | 14/14 | 9/9 |
| Kind exact | 15/15 | 12/14 | 8/9 |
| Cluster pair F1 (ceiling) | 0.717 (0.798) | 0.788 (0.660) | 0.177 (0.658) |
| Tables recovered | 3/3 | 2/3 | 4/5 |
| Rows exact / mismatched | 252 / 0 | 148 / 0 | 5 / 1 |

Two movements against the first run are the requestor filter's, and they go in opposite directions.
Telecom's clustering ceiling rose from 0.561 to 0.658, because the write signature no longer carries
the simulated user's own phone actions. Its pair F1 fell from 0.207 to 0.177, because a Run whose
only writes were the user's now has an empty write signature, and every such Run lands in one
category: precision 0.106. Removing the wrong tools from the signature exposed how many telecom Runs
have no agent write at all, which the contamination had been hiding.

Telecom still mines 21 tools against 13 real, and the 8 extra are not a filter failure. The
assistant in this export really does call the user's phone tools: `check_network_status` 53 times,
`run_speed_test` 5, `toggle_data` and `reboot_device` twice each, out of 266, 480, 258 and 615 calls
the user makes to the same tools. The requestor filter is doing exactly what it should; what is left
is a domain where the two toolkits overlap in the agent's own hands. `gate_tools` flags all but one
of them as thin, which is the right outcome for a tool seen twice.

Retail kind is now 15/15 rather than 13/15: D98's generic-name rule landed after the first run and
picks out `calculate` and `transfer_to_human_agents`, the two that used to fall to a read default.

## After the fixes (2026-08-29)

Four of the seven follow-ups are in, recorded as D101 to D104. Every number below is one command
(`uv run python scripts/xdomain_check.py retail airline telecom`) with nothing tuned per domain.

| | Retail | Airline | Telecom |
|---|---|---|---|
| Kind exact, before | 15/15 | 12/14 | 8/9 |
| Kind exact, after | **15/15** | **14/14** | **9/9** |
| Tables recovered, before | 3/3 | 2/3 | 4/5 |
| Tables recovered, after | **3/3** | **3/3** | **5/5** |
| Rows exact, after | 252 | 148 | 8 |
| Fields exact (of fields both rows carry) | 1559/1559 | 2393/2393 | 83/84 |
| Cluster pair F1, before | 0.717 | 0.788 | 0.177 |
| Cluster pair F1, after | 0.717 | 0.756 | 0.178 |

Retail did not move on anything, which is the condition each of these fixes had to meet.

Three things the corrected comparison says that the first run could not:

- **Airline's `flights` is recovered and its rows are right where the tool shows them.** The 121 rows
  the strict row count calls mismatches differ by nesting, not by value: tau2 keeps per-date state
  under `dates[date]` and `search_direct_flight` hands it back flat, so our row carries `date`,
  `prices`, `available_seats` and `status` at the top level. Every field both rows carry is exact.
- **13 more airline flights are tagged synthetic rows** (D40), filled from the observed rows for ids
  the traces named but never showed. They are no longer scored against the real database, because
  they were never a claim about it.
- **Telecom's one remaining row mismatch is `lines.L1002.roaming_enabled`**, which is the
  `initialization_actions` gap and nothing else.

Airline's cluster F1 falling from 0.788 to 0.756 is the price of D101 and it is in the decision log:
two real writes now enter the Category signature and split Runs that a missing write had held
together. The signature is what the Verifier rests on, so a more truthful signature is worth more
than the third of a point.

Still open from the list above: write results that are confirmation strings need the End state read
from the next read of the same row; per-task initialization actions need a Starting-state overlay
that carries values rather than only a version hash; and `norm()` in the comparison script settles
the datetime separator, which is a measurement convention rather than a harness fix.
