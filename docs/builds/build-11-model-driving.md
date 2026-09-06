# Build table: .work-b9-agent

**Tasks covered: 61 of 205 frozen Tasks (29.8%), 185 of 456 Runs (40.6%)**

Read from `.work-b9-agent` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 61 of 205 frozen Tasks (29.8%), 185 of 456 Runs (40.6%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 423 of 456 (92.8%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 546 of 575 (95.0%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 9 of 2,645 (0.3%) | `gates.json replay_reference metrics` |
| Gates green | 10 of 15 named below | `gates.json` |
| What the mechanic called | 65 over 25 model turns: repair_refuse_task 41, repair_recompile 7, status 6, build 6, replay 4, compile_tool 1 | `builder/session.jsonl` |
| Repairs requested | 48 requested: repair_refuse_task 41, repair_recompile 7 | `repairs/` |
| Repairs that turned a red gate green | n/a (needs the round on a repair request and a gate ruling per round; repairs/*.jsonl records the verb, the target and a timestamp, and rounds.json carries no timestamp) | `repairs/, gates.json` |
| Repairs that did not | n/a (needs the round on a repair request and a gate ruling per round; repairs/*.jsonl records the verb, the target and a timestamp, and rounds.json carries no timestamp) | `repairs/, gates.json` |
| Dollars | $8.42 | `budget.json total.usd` |
| Model calls | 17,118 (13,631 priced, 3,487 unpriced, 1,040 memo hits) | `budget.json total` |
| Model call wall time | 7h 25m 51s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | n/a (needs a start and an end timestamp; pipeline/state.json records the stage statuses and no clock) | `pipeline/state.json` |
| Peak context fill, Builder | n/a (needs ContextStats.fill_at_turn_end, which no stage writes into the workdir); the largest input a turn recorded is 15,830 tokens | `builder/session.jsonl message usage` |
| Peak context fill, Examiner | n/a (needs ContextStats.fill_at_turn_end, which no stage writes into the workdir); the largest input a turn recorded is 36,491 tokens | `examiner/session.jsonl message usage` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `build_user_rules` | green | 1 of 1 |  | `gates.json` |
| `compile_policy` | green | 1 of 1 |  | `gates.json` |
| `confined` | green | 1 of 1 |  | `gates.json` |
| `derive_verifier` | red | 0 of 1 | task task_55bde35c8b79: the D79 suite did not pass | `gates.json` |
| `deterministic` | green | 1 of 1 |  | `gates.json` |
| `executes_on_s0` | green | 1 of 1 |  | `gates.json` |
| `intent` | red | 0 of 1 | task task_000ffeb205cd: noun phrases not evidenced in every Run: canister vacuum cleaner (not in e8bd582d-90f1-4a3f-840a-0bc8eb5e80fd) | `gates.json` |
| `non_trivial` | green | 1 of 1 |  | `gates.json` |
| `parses` | green | 1 of 1 |  | `gates.json` |
| `refuses_unknown` | green | 6 of 6 |  | `gates.json` |
| `replay_fidelity` | red | 0 of 1 | modify_pending_order_items({"item_ids": ["3694871183"], "new_item_ids": ["6077640618"], "order_id": "#W5061109", "payment_method_id": "paypal_3742148" | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_2e134f9291e5: modify_pending_order_items write: differs | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |
| `trusted` | red | 0 of 1 | task task_000ffeb205cd: the D79 suite did not pass | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `modify_pending_order_items` | write | 139 | 122 | 17 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_2e134f9291e5/2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35 checks[11]` |
| `modify_pending_order_address` | write | 76 | 65 | 0 | 11 | 0 | body_error | 1 | none |  | `replays.json task_24633c648d01/284cbf5a-0965-4f63-a02d-ce06f4dce316 checks[9]` |
| `calculate` | read | 33 | 29 | 0 | 4 | 0 | body_error | 4 | replay_fidelity | yes | `replays.json task_623ba789b049/13875336-5ac2-4f14-a9f7-d0f911dff139 checks[6]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | unreadable | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `return_delivered_order_items` | write | 148 | 147 | 0 | 0 | 1 | real_errored | 4 | none |  | `replays.json task_d8954d758335/ad96d823-7edc-4d21-a815-25aac88cd134 checks[7]` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
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
| unreadable | 22 | 57.9% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_items` 17 | n/a: replays.json keeps a 160 character preview of each answer, which is not enough to name the cause | `replays.json` |
| body_error | 15 | 39.5% | `calculate` 4, `modify_pending_order_address` 11 | compile_tools: the body raised where the real tool answered | `replays.json` |
| real_errored | 1 | 2.6% | `return_delivered_order_items` 1 | reference: the real tool refused where ours answered | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_000ffeb205cd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 35: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_000ffeb205cd` |
| `task_039604eb2f69` | yes | no | second_path_passes | verifier_alt_path | 6 required of 44: modify_pending_order_items writes #W5061109 | the Verifier did not pass the D79 suite | `task_status.json task_039604eb2f69` |
| `task_03c32a57c05e` | yes | no | second_path_passes | verifier_alt_path | 7 required of 73: cancel_pending_order writes #W5995614 | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_041a2ef3294c` | yes | no | mutation_flips |  | 1 required of 55: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_041a2ef3294c` |
| `task_06efc5b06acb` | yes | no | second_path_passes, mutation_flips | verifier_alt_path | 1 required of 44: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_06efc5b06acb` |
| `task_09638ac8e2b2` | yes | no | mutation_flips |  | 1 required of 42: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09638ac8e2b2` |
| `task_09ba7fddcd5f` | yes | no | second_path_passes, mutation_flips | verifier_alt_path | 1 required of 39: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09ba7fddcd5f` |
| `task_0d00a5249f02` | yes | no | second_path_passes | verifier_alt_path | 5 required of 48: exchange_delivered_order_items writes #W9502127 | the Verifier did not pass the D79 suite | `task_status.json task_0d00a5249f02` |
| `task_11e0759b1041` | yes | no | second_path_passes | verifier_alt_path | 5 required of 80: modify_pending_order_address writes #W1092119 | the Verifier did not pass the D79 suite | `task_status.json task_11e0759b1041` |
| `task_1211c761ec81` | yes | no | second_path_passes | verifier_alt_path | 6 required of 80: modify_pending_order_address writes #W4082615 | the Verifier did not pass the D79 suite | `task_status.json task_1211c761ec81` |
| `task_126d2df50be8` | yes | no | mutation_flips |  | 1 required of 43: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_126d2df50be8` |
| `task_1a0555df39f1` | yes | no | unsolved_state_fails, second_path_passes, mutation_flips | verifier_alt_path | 1 required of 40: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | second_path_passes, mutation_flips | verifier_alt_path | 1 required of 48: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_24633c648d01` | yes | no | second_path_passes | verifier_alt_path | 5 required of 76: modify_pending_order_address writes #W5270061 | the Verifier did not pass the D79 suite | `task_status.json task_24633c648d01` |
| `task_37d1178f5430` | yes | no | second_path_passes | verifier_alt_path | 9 required of 60: return_delivered_order_items writes #W6239298 | the Verifier did not pass the D79 suite | `task_status.json task_37d1178f5430` |
| `task_3a82fb0dbd47` | yes | no | second_path_passes | verifier_alt_path | 10 required of 68: cancel_pending_order writes #W8855135 | the Verifier did not pass the D79 suite | `task_status.json task_3a82fb0dbd47` |
| `task_3bb9d4223b67` | yes | no | second_path_passes | verifier_alt_path | 5 required of 50: exchange_delivered_order_items writes #W7181492 | the Verifier did not pass the D79 suite | `task_status.json task_3bb9d4223b67` |
| `task_3c1a364cb85d` | yes | no | second_path_passes | verifier_alt_path | 6 required of 63: exchange_delivered_order_items writes #W3239882 | the Verifier did not pass the D79 suite | `task_status.json task_3c1a364cb85d` |
| `task_3f7d47009c93` | yes | no | second_path_passes | verifier_alt_path | 6 required of 51: modify_user_address writes FATIMA_TAYLOR_3452 | the Verifier did not pass the D79 suite | `task_status.json task_3f7d47009c93` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

| round | gate counts (D126) | moved | spend | turns | exit | read from |
|---:|---|---|---|---|---|---|
| 1 | fidelity 194, trusted 61, refused_count 0, assisted_runs 0, probes_passing 0 | first round | builder $1.75, examiner $0.18, total $1.92 | n/a (needs the round on a session entry; session.jsonl marks no round) | the round did not exit | `rounds.json[0]` |
| 2 | fidelity None, trusted None, refused_count None, assisted_runs None, probes_passing None | fidelity None from 194, trusted None from 61, refused_count None from 0, assisted_runs None from 0, probes_passing None from 0 | n/a (needs rounds.json counts.spend) | n/a (needs the round on a session entry; session.jsonl marks no round) | stalled | `rounds.json[1]` |

## Per repair verb

| verb | called on | ruling before | ruling after | read from |
|---|---|---|---|---|
| `repair_recompile` | `calculate` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `calculate` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_recompile.jsonl` |
| `repair_refuse_task` | `task_000ffeb205cd` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_018fc5cbf7e9` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_039604eb2f69` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_03d78b3f5cea` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_041a2ef3294c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_06efc5b06acb` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0866ed543e0c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_09638ac8e2b2` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0c9e2b8b91b2` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0d00a5249f02` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0da6dbdf8d5c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0eb898e0acde` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0fdb72dff11a` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_11e0759b1041` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1211c761ec81` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_126d2df50be8` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_12bf34ff4998` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1473d4c4b1ab` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_15472455ed4d` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_197b67aea13a` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1a6f39478206` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1c59c73758e6` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1f7556d168fa` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1fed0a63aabd` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_24633c648d01` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_000ffeb205cd` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_018fc5cbf7e9` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_039604eb2f69` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_03d78b3f5cea` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_041a2ef3294c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_06efc5b06acb` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0866ed543e0c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_09638ac8e2b2` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0c9e2b8b91b2` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0da6dbdf8d5c` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0eb898e0acde` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_0fdb72dff11a` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_11e0759b1041` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_1211c761ec81` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_126d2df50be8` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |
| `repair_refuse_task` | `task_2e134f9291e5` | n/a (needs a gate ruling per round; gates.json holds the last round only) | n/a (needs a gate ruling per round; gates.json holds the last round only) | `repairs/repair_refuse_task.jsonl` |

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| provider_error | ours | 15 | an error event naming the provider or the transport, so no model turn ever happened | `runs/`, `replays.json`, `references.json` |
| body_exception | ours | 2 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| replay_refused | ours | 15 | a replayed call we refused and the recording answered | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 17 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 2 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 10 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 26 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |
| unattributed | unattributed | 7 | no rule above matched what the Run recorded | `runs/`, `replays.json`, `references.json` |

94 failing Runs: 51 ours, 10 the model's, 33 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `reroll-task_55d0d0ade502-0` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-0.jsonl` |
| `reroll-task_55d0d0ade502-1` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-1.jsonl` |
| `reroll-task_612571611125-2` | reroll | ours | body_exception | TypeError: DomainTools.find_user_id_by_name_zip() got an unexpected keyword argument 'email' | `runs/task_612571611125/reroll-task_612571611125-2.jsonl` |
| `reroll-task_7399de413f06-0` | reroll | ours | body_exception | KeyError: 'W3916020' | `runs/task_7399de413f06/reroll-task_7399de413f06-0.jsonl` |
| `replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_8e41df099d75/replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51.jsonl` |
| `replay-13875336-5ac2-4f14-a9f7-d0f911dff139` | replay | ours | replay_refused | calculate read: ours_refused | `runs/task_623ba789b049/replay-13875336-5ac2-4f14-a9f7-d0f911dff139.jsonl` |
| `replay-2535d17c-a600-42cf-b74f-86718674fe44` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_4fc41fab480d/replay-2535d17c-a600-42cf-b74f-86718674fe44.jsonl` |
| `replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_2e134f9291e5/replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35.jsonl` |
| `reroll-task_0d00a5249f02-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0d00a5249f02/reroll-task_0d00a5249f02-1.jsonl` |
| `reroll-task_7ef2dc85ffaf-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_7ef2dc85ffaf/reroll-task_7ef2dc85ffaf-1.jsonl` |
| `replay-351db446-7efd-48ab-9269-b0f86e148505` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Payment method should be the original payment method | `runs/task_5ef4c9e9d199/replay-351db446-7efd-48ab-9269-b0f86e148505.jsonl` |
| `replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Payment method should be the original payment method | `runs/task_0da6dbdf8d5c/replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627.jsonl` |
| `replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5` | replay | unattributed | judge_failed_it | task_5ef4c9e9d199: judge: The intent required returning the more expensive tablet and refunding the credit card, but both states used a gift card as the refund  | `runs/task_5ef4c9e9d199/replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5.jsonl` |
| `replay-086b19c9-0244-46d4-b315-8c59fbc63301` | replay | unattributed | judge_failed_it | task_50b019d79a4c: judge: The intent was to modify the order by exchanging the luggage set for a coat, but both states only canceled the pending order. | `runs/task_50b019d79a4c/replay-086b19c9-0244-46d4-b315-8c59fbc63301.jsonl` |
| `replay-ad96d823-7edc-4d21-a815-25aac88cd134` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_d8954d758335/replay-ad96d823-7edc-4d21-a815-25aac88cd134.jsonl` |
| `reroll-task_7ef2dc85ffaf-2` | reroll | unattributed | unattributed | max_turns | `runs/task_7ef2dc85ffaf/reroll-task_7ef2dc85ffaf-2.jsonl` |

78 more failing Runs not listed; pass `--limit 0` to print every row.
