# Rows whose identity is more than one column (airline-domain corpus, 119 Tasks)

The rule, the code it lives in, and one round measured against the round it replaces. Field names
and masked shapes only; no customer values.

Workdirs: `.work-airline` is the round before (read only), `.work-airline-keys` is its copy rebuilt
on this branch. One round, `--target replay_reference --max-rounds 1 --workers 8
--model openai/gpt-5.6-luna --ceiling-usd 10`, code driving.

## The rule

A table's rows repeat an id inside one trace, with no write in between, and their hard values
differ: one id is standing for more than one row and the single key is losing all but the last of
them. What joins the key is a column of the table that is also an argument of the call that
returned the rows and whose argument value differs between the two calls. The key is the
intersection over every conflicting pair of the corpus, so one odd pair cannot add a column and a
table whose repeats no argument explains keeps its single key.

Four refusals, each with a test in an invented domain (`tests/builder/test_mine.py`, a `docks`
table let out per shift):

| Refusal | Why |
|---|---|
| a change after a write | that is the after_write rule (D74); a sighting on or after a write to that id is not compared |
| a soft column | only `hard` columns (D73) count, both for the difference that opens the case and for the column that joins the key |
| a column that is not an argument of the returning call | the corpus has not shown the tool being asked for one version rather than the other |
| a table with one version per id | it has no repeat to explain |

Row ids stay strings: a composite id is the key columns' values joined by the separator the schema
names, so nothing under `kullback/runner/` reads a row differently.

### The partial sighting

A row seen with only part of its key is not a new row and not a contradiction. A key column the row
leaves out or answers null is taken from the arguments of the call that returned it, which is what
keys a search answering rows for the date it was given. What neither the row nor the call supplies
is empty in that part of the key, and such a sighting is folded into every row whose known key
columns match: it fills only the columns those rows were never shown holding, so it can add what the
keyed sightings did not show and can never overwrite what they did. Where no keyed row matches it is
kept as the partial sighting it is, under its own partial key, because dropping it would lose an
observation and inventing the missing part would be a guess (D41). One assumption line per fold.
This corpus folded none: every sighting of the keyed table came from a call that named the missing
part in its arguments.

## Files changed

| File | What |
|---|---|
| `kullback/builder/mine.py` | `composite_keys` (the rule), `write_tool_names`, and `mine_schema` naming the key on the schema |
| `kullback/gates/tool_runs.py` | `key_fields`, `key_separator`, `row_key`, `partial_key`; `match_table` keys a row by them and takes the call's arguments |
| `kullback/builder/compile_env.py` | `extract_rows` passes the arguments through; `referenced_ids` and `add_synthetic_rows` work in whole keys; `fold_partial_rows`; the tables block shows the key's form |
| `kullback/builder/sandbox.py` | the new row helpers re-exported beside the old ones |
| `kullback/builder/build.py` | the mine stage passes the sigs' own write tools |
| `kullback/runner/records.py` | `EntitySchema.composite_keys` and `EntitySchema.key_separator`, both defaulted |
| `tests/builder/test_mine.py`, `tests/builder/test_compile_env.py` | the rule and the world it builds, in an invented domain |

## Tables that got a composite key

| Table | Key | Rows before | Rows after |
|---|---|---|---|
| flights | `flight_number` then `date`, joined with a pipe | 122 (122 observed, 0 synthetic) | 187 (160 observed, 27 synthetic) |
| reservations | its id column alone, unchanged | 99 | 99 |
| users | its id column alone, unchanged | 33 | 33 |

One table of the three, which is the one the corpus shows repeating an id.

The 27 synthetic rows are (id, date) pairs a status call names in its arguments and no search ever
returned. They did not exist before because a bare id was already in the world; with the whole key
they are ids the traces reference and the world does not hold, so D40 fills them, tagged, and a Run
that reads one is reported assisted. The 160 observed rows keep `date: null` exactly as the
recordings show them; the date lives in the key, not in the column.

## Tasks at replay fidelity

| | Tasks with a confirmed replay | Tasks with a Reference | Trusted |
|---|---|---|---|
| the eight recorded rounds before | 47, 47, 60, 60, 64, 64, 56, 56 | 33 to 39 | 1 to 8 |
| this round | 63 | 62 | 19 |

Fidelity 56 to 63 against the round this replaces, and one below the corpus's best of 64. References
38 to 62 and trusted 5 to 19 are the larger movements, and they are the ones the round's own counts
report.

## Recorded calls matched, per tool

Replay of the Reference Runs (`replays.json`, verdict `same` over every recorded call of the
corpus).

| Tool | Before | After |
|---|---|---|
| search_direct_flight | 275/291 | 287/291 |
| search_onestop_flight | 18/78 | 18/78 |
| get_flight_status | 0/112 | 51/112 |
| get_reservation_details | 489/489 | 489/489 |
| get_user_details | 180/210 | 180/210 |
| list_all_airports | 0/3 | 0/3 |
| calculate | 100/100 | 100/100 |
| transfer_to_human_agents | 66/66 | 66/66 |
| **reads, all tools** | **1128/1349** | **1191/1349** |
| book_reservation | 0/29 | 0/29 |
| cancel_reservation | 43/47 | 43/47 |
| update_reservation_flights | 16/59 | 0/59 |
| update_reservation_baggages | 14/20 | 6/20 |
| update_reservation_passengers | 9/9 | 6/9 |
| **writes, all tools** | **82/164** | **55/164** |

Bodies assisted: 6 of 13 before and 6 of 13 after, the same six.

The corpus-level per-tool ruling this round records (D171 `tool_fidelity.json`, which the earlier
workdir predates): search_direct_flight 246 of 249 calls replayed, get_flight_status 47 of 97,
search_onestop_flight 18 of 70, update_reservation_flights 0 of 36, book_reservation 0 of 24,
list_all_airports 0 of 2, every other tool all of its calls. 56 of the 119 Tasks are blocked by a
differing call of their own.

## What moved, and what the numbers cost

The instruction reached the bodies. All three bodies that touch the keyed table build the key the
tables block describes, one as `id + "|" + date` and two by walking the table and splitting the key
on the separator. Nothing had to be repaired into them.

Reads gained 63 calls. The search tool answers the rows of the date it was asked for instead of the
last date the world saw, 275 to 287 of 291. The status tool goes from 0 of 112 to 51 of 112: before,
its body answered null on every call because the row it looked up did not exist under the key it
built; now the row exists.

Three costs, all of them the change's own and all measurable:

1. The keyed write refuses a pair the world does not hold. 15 of the 59 recorded
   update_reservation_flights calls now raise not_found, because the write names its flight
   references inside a nested list of arguments and `referenced_ids` reads only top-level arguments,
   so those (id, date) pairs are never filled. Before, a lookup by the id alone always found some
   row, which is why the number was higher and the row was the wrong one. This is the clearest next
   step: read referenced ids out of nested argument structures as well.
2. Four of the 291 search calls answer a synthetic row where the recording answered an empty list:
   a status call named an (id, date) pair, D40 filled it, and the search then finds it. The Run that
   reads one is already reported assisted.
3. The status per date is still not in the world. The status tool answers a bare string, which the
   miner attributes to no column, so every row of the table carries the same status value and the
   body invents the rest; the `non_trivial` gate catches it and the tool stays assisted. The
   composite key gives the row its identity, not its value.

The other write movements (baggages 14 to 6, passengers 9 to 6, flights 16 to 0) are the round's own
bodies: every body is rewritten when the schema changes, and those three tools' kept bodies scored
worse this round than last. update_reservation_flights' last attempt fell at the memorised-values
gate on a literal id, which is D162's refusal and not about keys.

## Cost

| | |
|---|---|
| spend, this round | 0.3703 USD (compile_policy 0.2197, compile_tools 0.1240, loophole_probe 0.0256, reference_judge 0.0010) |
| wall time | 509 s, exit `max_rounds` |
| stages that recomputed | mine, cluster, canon_rules, starting_state, compile_tools, compile_policy, replay_reference and everything after; the schema is an input to all of them |

## Two things this change touched that it was meant to leave alone

`EntitySchema` gains two defaulted fields, not one: the schema has to name the separator as well as
the key columns, and there is nothing on the record a separator could hide in. `kullback/runner/` is
frozen, so the runner needs re-freezing before the next build; nothing in the build path compares
the recorded RunnerVersion, so the build did not refuse and nothing was re-frozen here.

`kullback/gates/tool_runs.py` is edited because it holds the shared row helpers (`match_table`,
`id_field`), so keying a table by a composite key lands there whatever else changes. No ruling
changed: the seven gate functions are untouched, and `verifier_suite._entity` reads a write's id out
of its arguments and never forms a database key, so no gate needed the key.

`synth.grow` (the `--grow` flag, D107) still composes new rows keyed by the id column alone, so a
table with a composite key cannot be grown correctly. This build passes no `--grow`.

## The retail fixture is unchanged

`mine_schema` over `tests/fixtures/tau2_retail_small.json` names no composite key, asserted by a
test, and the whole e2e fixture build is byte-identical in its keying: every table there is keyed
by its id column alone, so `key_fields` answers the one-element list every reader saw before.
