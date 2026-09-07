# Build table: .work-b12-agent

**Tasks covered: 65 of 205 frozen Tasks (31.7%), 161 of 456 Runs (35.3%)**

Read from `.work-b12-agent` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 65 of 205 frozen Tasks (31.7%), 161 of 456 Runs (35.3%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 421 of 456 (92.3%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 536 of 575 (93.2%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 5 of 2,645 (0.2%) | `gates.json replay_reference metrics` |
| Gates green | 12 of 16 named below | `gates.json` |
| What the mechanic called | 59 over 50 model turns: repair_intent 30, repair_recompile 14, build 10, status 4, repair_escalate 1 | `builder/session.jsonl` |
| Repairs requested | 47 requested: repair_intent 30, repair_recompile 16, repair_escalate 1 | `repairs/` |
| Repairs that turned a red gate green | 1 of 47: repair_intent on task_000ffeb205cd (round 4, loosening went red to green); 20 not placed: n/a (needs the round on a repair request and the rulings of both the round before it and its own; repairs/*.jsonl records round and gates_by_round.json the rulings per round) | `repairs/ round, gates_by_round.json` |
| Repairs that did not | 26 of 47: repair_escalate on task_7ef2dc85ffaf (round 6), repair_intent on task_03d78b3f5cea (round 2), repair_intent on task_000ffeb205cd (round 2), repair_intent on task_55bde35c8b79 (round 2), repair_intent on task_3c4e4938561e (round 8), and 21 more; 20 not placed: n/a (needs the round on a repair request and the rulings of both the round before it and its own; repairs/*.jsonl records round and gates_by_round.json the rulings per round) | `repairs/ round, gates_by_round.json` |
| Dollars | $14.53; the cache saved $0.00: 105,087,272 cache-read tokens at the cache rate, 2,726 memo hits sent nothing; without it $14.53 | `budget.json total.usd, total.cache_saved_usd` |
| Model calls | 40,943 (37,456 priced, 3,487 unpriced, 2,726 memo hits) | `budget.json total` |
| Model call wall time | 19h 39m 45s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | 19h 29m 7s over 10 rounds, 2026-09-06T14:49:04+00:00 to 2026-09-07T10:18:12+00:00 | `rounds.json counts.started_at, counts.ended_at` |
| Peak context fill, Builder | 28.4% of the window at a turn end, over 50 turns in 10 rounds | `rounds.json counts.context_fill` |
| Peak context fill, Examiner | 237.3% of the window at a turn end, over 49 turns in 10 rounds | `rounds.json counts.context_fill` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `build_user_rules` | green | 1 of 1 |  | `gates.json` |
| `compile_policy` | green | 1 of 1 |  | `gates.json` |
| `confined` | green | 16 of 16 |  | `gates.json` |
| `derive_verifier` | red | 0 of 1 | task task_55bde35c8b79: the D79 suite did not pass | `gates.json` |
| `deterministic` | green | 16 of 16 |  | `gates.json` |
| `executes_on_s0` | green | 16 of 16 |  | `gates.json` |
| `loosening` | green | 1 of 1 |  | `gates.json` |
| `non_trivial` | green | 16 of 16 |  | `gates.json` |
| `parses` | green | 16 of 16 |  | `gates.json` |
| `probe_pool` | green | 1 of 1 |  | `gates.json` |
| `refuses_unknown` | green | 5 of 5 |  | `gates.json` |
| `replay_fidelity` | red | 26 of 29 | return_delivered_order_items({"item_ids": ["6065192424"], "order_id": "#W9571698", "payment_method_id": "credit_card_1565124"}): expected error busine | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_2e134f9291e5: modify_pending_order_items write: differs | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |
| `trusted` | red | 0 of 1 | task task_000ffeb205cd: the D79 suite did not pass | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `modify_pending_order_items` | write | 139 | 119 | 14 | 3 | 3 | value | 4 | replay_fidelity | yes | `replays.json task_2e134f9291e5/2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35 checks[10]` |
| `modify_pending_order_address` | write | 76 | 65 | 0 | 11 | 0 | body_error | 1 | none |  | `replays.json task_24633c648d01/284cbf5a-0965-4f63-a02d-ce06f4dce316 checks[9]` |
| `return_delivered_order_items` | write | 148 | 140 | 0 | 0 | 8 | real_errored | 4 | executes_on_s0 | yes | `replays.json task_09638ac8e2b2/5235d473-52bc-4e26-82a5-74f1a3513ce0 checks[6]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | value | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | value | 4 | executes_on_s0 | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `calculate` | read | 33 | 33 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 2 | none |  | `replays.json checks` |
| `exchange_delivered_order_items` | write | 95 | 95 | 0 | 0 | 0 | none | 4 | none |  | `replays.json checks` |
| `find_user_id_by_email` | read | 123 | 123 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `find_user_id_by_name_zip` | read | 366 | 366 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_item_details` | read | 24 | 24 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_product_details` | read | 414 | 414 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `list_all_product_types` | read | 19 | 19 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_pending_order_payment` | write | 4 | 4 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_user_address` | write | 35 | 35 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `transfer_to_human_agents` | read | 38 | 38 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |

Agrees is `same`, `cosmetic` and `both_refused` together, which is how `runner/replay.py` counts a call that agrees; a refusal on both sides whose wording differs is agreement here and a miss to `env_fidelity.py`, which reads the whole answer rather than the preview replays.json keeps.

### Where the misses come from

| cause | calls | share of the misses | tools | owner | read from |
|---|---:|---:|---|---|---|
| value | 19 | 43.2% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_items` 14 | compile_tools: a different answer with no shape or error explanation | `replays.json` |
| body_error | 14 | 31.8% | `modify_pending_order_address` 11, `modify_pending_order_items` 3 | compile_tools: the body raised where the real tool answered | `replays.json` |
| real_errored | 11 | 25.0% | `modify_pending_order_items` 3, `return_delivered_order_items` 8 | reference: the real tool refused where ours answered | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_000ffeb205cd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_000ffeb205cd` |
| `task_039604eb2f69` | yes | no | leak_check_clean |  | 6 required of 38: modify_pending_order_items writes #W5061109 | the Verifier did not pass the D79 suite | `task_status.json task_039604eb2f69` |
| `task_03c32a57c05e` | yes | no | second_path_passes | verifier_alt_path | 16 required of 61: cancel_pending_order writes #W5995614 | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_041a2ef3294c` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_041a2ef3294c` |
| `task_06efc5b06acb` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_06efc5b06acb` |
| `task_09ba7fddcd5f` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09ba7fddcd5f` |
| `task_0d00a5249f02` | yes | no | second_path_passes | verifier_alt_path | 5 required of 44: exchange_delivered_order_items writes #W9502127 | the Verifier did not pass the D79 suite | `task_status.json task_0d00a5249f02` |
| `task_1211c761ec81` | yes | no | second_path_passes | verifier_alt_path | 6 required of 76: modify_pending_order_address writes #W4082615 | the Verifier did not pass the D79 suite | `task_status.json task_1211c761ec81` |
| `task_1a0555df39f1` | yes | no | second_path_passes | verifier_alt_path | 5 required of 43: exchange_delivered_order_items writes #W2378156 | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1a6f39478206` | yes | no | leak_check_clean |  | 9 required of 43: return_delivered_order_items writes #W5490111 | the Verifier did not pass the D79 suite | `task_status.json task_1a6f39478206` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_24633c648d01` | yes | no | second_path_passes | verifier_alt_path | 5 required of 72: modify_pending_order_address writes #W5270061 | the Verifier did not pass the D79 suite | `task_status.json task_24633c648d01` |
| `task_2eba497a317d` | yes | no | leak_check_clean |  | 14 required of 66: modify_pending_order_address writes #W3730488 | the Verifier did not pass the D79 suite | `task_status.json task_2eba497a317d` |
| `task_37d1178f5430` | yes | no | second_path_passes | verifier_alt_path | 9 required of 56: return_delivered_order_items writes #W6239298 | the Verifier did not pass the D79 suite | `task_status.json task_37d1178f5430` |
| `task_3f7d47009c93` | yes | no | second_path_passes, leak_check_clean | verifier_alt_path | 6 required of 47: modify_user_address writes FATIMA_TAYLOR_3452 | the Verifier did not pass the D79 suite | `task_status.json task_3f7d47009c93` |
| `task_3ff6b56d38f7` | yes | no | mutation_flips, leak_check_clean |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_3ff6b56d38f7` |
| `task_447f2fc94897` | yes | no | second_path_passes | verifier_alt_path | 14 required of 52: exchange_delivered_order_items writes #W2466703 | the Verifier did not pass the D79 suite | `task_status.json task_447f2fc94897` |
| `task_44854ce70e66` | yes | no | second_path_passes | verifier_alt_path | 10 required of 76: modify_pending_order_address writes #W3730488 | the Verifier did not pass the D79 suite | `task_status.json task_44854ce70e66` |
| `task_4589c978e041` | yes | no | second_path_passes | verifier_alt_path | 11 required of 52: exchange_delivered_order_items writes #W3792453 | the Verifier did not pass the D79 suite | `task_status.json task_4589c978e041` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

| round | gate counts (D126) | moved | artifacts changed | spend | turns | exit | read from |
|---:|---|---|---|---|---|---|---|
| 1 | fidelity 153, trusted 49, refused_count 0, assisted_runs 0, probes_passing 0 | first round | first round | builder $1.58, examiner $0.16, total $1.75 | builder 9, examiner 9, total 18 | the round did not exit | `rounds.json[0]` |
| 2 | fidelity 153, trusted 49, refused_count 0, assisted_runs 0, probes_passing 0 | none | intents | builder $0.01, examiner $0.05, total $0.06 | builder 10, examiner 4, total 14 | the round did not exit | `rounds.json[1]` |
| 3 | fidelity 153, trusted 48, refused_count 0, assisted_runs 0, probes_passing 0 | trusted 48 from 49 | bodies | builder $0.95, examiner $0.15, total $1.09 | builder 4, examiner 7, total 11 | the round did not exit | `rounds.json[2]` |
| 4 | fidelity 153, trusted 50, refused_count 0, assisted_runs 0, probes_passing 0 | trusted 50 from 48 | intents | builder $0.94, examiner $0.14, total $1.08 | builder 4, examiner 7, total 11 | the round did not exit | `rounds.json[3]` |
| 5 | fidelity 153, trusted 50, refused_count 0, assisted_runs 0, probes_passing 0 | none | none | builder $0.98, examiner $0.07, total $1.06 | builder 4, examiner 4, total 8 | the round did not exit | `rounds.json[4]` |
| 6 | fidelity 153, trusted 50, refused_count 0, assisted_runs 0, probes_passing 0 | none | bodies | builder $1.03, examiner $0.05, total $1.08 | builder 5, examiner 5, total 10 | the round did not exit | `rounds.json[5]` |
| 7 | fidelity 196, trusted 63, refused_count 0, assisted_runs 0, probes_passing 0 | fidelity 196 from 153, trusted 63 from 50 | none | builder $0.00, examiner $0.19, total $0.19 | builder 1, examiner 2, total 3 | the round did not exit | `rounds.json[6]` |
| 8 | fidelity 196, trusted 63, refused_count 0, assisted_runs 0, probes_passing 0 | none | bodies, intents | builder $1.30, examiner $0.13, total $1.43 | builder 10, examiner 4, total 14 | the round did not exit | `rounds.json[7]` |
| 9 | fidelity 196, trusted 64, refused_count 0, assisted_runs 0, probes_passing 0 | trusted 64 from 63 | none | builder $0.00, examiner $0.17, total $0.17 | builder 1, examiner 2, total 3 | the round did not exit | `rounds.json[8]` |
| 10 | fidelity 196, trusted 64, refused_count 0, assisted_runs 0, probes_passing 0 | none | none | builder $0.00, examiner $0.12, total $0.12 | builder 2, examiner 5, total 7 | stalled | `rounds.json[9]` |

## Per repair verb

| verb | called on | round | ruling before | ruling after | turned green | read from |
|---|---|---:|---|---|---|---|
| `repair_escalate` | `task_7ef2dc85ffaf` | 6 | red: derive_verifier, loosening, replay_reference, trusted | red: intent, replay_reference, trusted | none | `repairs/repair_escalate.jsonl` |
| `repair_intent` | `task_039604eb2f69` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_041a2ef3294c` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_06efc5b06acb` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_0866ed543e0c` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_3c4e4938561e` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_6677a476d3c7` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_b72a88c48881` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_c280d4c61cef` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_c3f2f72e504b` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_eb55d04caaa4` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: derive_verifier, intent, replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_000ffeb205cd` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: derive_verifier, intent, replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_55bde35c8b79` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: derive_verifier, intent, replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_000ffeb205cd` | 4 | red: derive_verifier, loosening, replay_reference, trusted | red: derive_verifier, intent, replay_reference, trusted | loosening | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_3c4e4938561e` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_6677a476d3c7` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_b72a88c48881` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_c280d4c61cef` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_c3f2f72e504b` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_eb55d04caaa4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_intent.jsonl` |
| `repair_recompile` | `calculate` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `calculate` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_address` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 3 | red: derive_verifier, intent, replay_reference, trusted | red: derive_verifier, loosening, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 5 | red: derive_verifier, intent, replay_reference, trusted | red: derive_verifier, loosening, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `calculate` | 6 | red: derive_verifier, loosening, replay_reference, trusted | red: intent, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | 6 | red: derive_verifier, loosening, replay_reference, trusted | red: intent, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 6 | red: derive_verifier, loosening, replay_reference, trusted | red: intent, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 6 | red: derive_verifier, loosening, replay_reference, trusted | red: intent, replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_address` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `return_delivered_order_items` | 8 | red: derive_verifier, intent, replay_reference, trusted | red: replay_reference, trusted | none | `repairs/repair_recompile.jsonl` |

The rulings are the ones the round before this repair's round left and the ones its own round left (`gates_by_round.json`); repairs made in the same round share what that round moved.

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| provider_error | ours | 12 | an error event naming the provider or the transport, so no model turn ever happened | `runs/`, `replays.json`, `references.json` |
| body_exception | ours | 13 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| replay_refused | ours | 11 | a replayed call we refused and the recording answered | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 13 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 120 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 78 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 148 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |
| unattributed | unattributed | 8 | no rule above matched what the Run recorded | `runs/`, `replays.json`, `references.json` |

403 failing Runs: 169 ours, 78 the model's, 156 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `reroll-task_55d0d0ade502-0` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-0.jsonl` |
| `reroll-task_55d0d0ade502-1` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-1.jsonl` |
| `replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35` | replay | ours | body_exception | UnboundLocalError: cannot access local variable 'difference' where it is not associated with a value | `runs/task_2e134f9291e5/replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35.jsonl` |
| `replay-2dfd6f1f-4d3d-443d-b55d-38edbd37bf76` | replay | ours | body_exception | IndexError: list index out of range | `runs/task_2e134f9291e5/replay-2dfd6f1f-4d3d-443d-b55d-38edbd37bf76.jsonl` |
| `replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_8e41df099d75/replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51.jsonl` |
| `replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_b72a88c48881/replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71.jsonl` |
| `replay-2535d17c-a600-42cf-b74f-86718674fe44` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_4fc41fab480d/replay-2535d17c-a600-42cf-b74f-86718674fe44.jsonl` |
| `replay-50c539c3-7f61-4ee4-aece-dddaaeee4886` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_52cace5aa209/replay-50c539c3-7f61-4ee4-aece-dddaaeee4886.jsonl` |
| `reroll-task_0866ed543e0c-0` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0866ed543e0c/reroll-task_0866ed543e0c-0.jsonl` |
| `reroll-task_0866ed543e0c-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0866ed543e0c/reroll-task_0866ed543e0c-1.jsonl` |
| `replay-77d71860-9c54-423d-bb0b-df62f4fcb332` | replay | the model's | env_refused_the_candidate | business_error: ValueError: New item 5753502325 not found or available | `runs/task_de3d8a4f9893/replay-77d71860-9c54-423d-bb0b-df62f4fcb332.jsonl` |
| `replay-af9b9687-2eb0-4f82-bfbf-c0c88620065f` | replay | the model's | env_refused_the_candidate | business_error: ValueError: User not found | `runs/task_7f32b58a1790/replay-af9b9687-2eb0-4f82-bfbf-c0c88620065f.jsonl` |
| `replay-010a23ef-3524-4dbf-ad95-81d4cf974a45` | replay | unattributed | judge_failed_it | task_dbf0c8930714: violates c_9f52317fb6df | `runs/task_dbf0c8930714/replay-010a23ef-3524-4dbf-ad95-81d4cf974a45.jsonl` |
| `replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5` | replay | unattributed | judge_failed_it | task_5ef4c9e9d199: judge: The intent was to return order #W9571698. State A returned only one item rather than the full order, and state C made no write. State  | `runs/task_5ef4c9e9d199/replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5.jsonl` |
| `replay-351db446-7efd-48ab-9269-b0f86e148505` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_5ef4c9e9d199/replay-351db446-7efd-48ab-9269-b0f86e148505.jsonl` |
| `replay-36611a4a-08df-4daf-8885-7890b314c654` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_197b67aea13a/replay-36611a4a-08df-4daf-8885-7890b314c654.jsonl` |

387 more failing Runs not listed; pass `--limit 0` to print every row.
