# Build table: .work-b12-code

**Tasks covered: 75 of 205 frozen Tasks (36.6%), 173 of 456 Runs (37.9%)**

Read from `.work-b12-code` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 75 of 205 frozen Tasks (36.6%), 173 of 456 Runs (37.9%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 432 of 456 (94.7%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 550 of 575 (95.7%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 5 of 2,645 (0.2%) | `gates.json replay_reference metrics` |
| Gates green | 10 of 15 named below | `gates.json` |
| What the mechanic called | 0 over 0 model turns; the session records no model turn, which is the code driver (D135) | `builder/session.jsonl` |
| Repairs requested | 0; the workdir has no repairs/ directory (D135) | `repairs/` |
| Repairs that turned a red gate green | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/ round, gates_by_round.json` |
| Repairs that did not | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/ round, gates_by_round.json` |
| Dollars | $8.65 | `budget.json total.usd` |
| Model calls | 17,952 (14,465 priced, 3,487 unpriced, 1,281 memo hits) | `budget.json total` |
| Model call wall time | 7h 34m 40s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | 49m 27s over 2 rounds, 2026-09-06T14:49:04+00:00 to 2026-09-06T15:38:32+00:00 | `rounds.json counts.started_at, counts.ended_at` |
| Peak context fill, Builder | 0.0%; the agent took no model turn, which is the code driver (D135) | `rounds.json counts.context_fill, counts.turns` |
| Peak context fill, Examiner | 0.0%; the agent took no model turn, which is the code driver (D135) | `rounds.json counts.context_fill, counts.turns` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `build_user_rules` | green | 1 of 1 |  | `gates.json` |
| `compile_policy` | green | 1 of 1 |  | `gates.json` |
| `confined` | green | 16 of 16 |  | `gates.json` |
| `derive_verifier` | red | 0 of 1 | task task_55bde35c8b79: the D79 suite did not pass | `gates.json` |
| `deterministic` | green | 16 of 16 |  | `gates.json` |
| `executes_on_s0` | green | 16 of 16 |  | `gates.json` |
| `intent` | red | 0 of 1 | task task_03d78b3f5cea: noun phrases with no span: calculate | `gates.json` |
| `non_trivial` | green | 16 of 16 |  | `gates.json` |
| `parses` | green | 16 of 16 |  | `gates.json` |
| `refuses_unknown` | green | 5 of 5 |  | `gates.json` |
| `replay_fidelity` | red | 26 of 29 | return_delivered_order_items({"item_ids": ["6065192424"], "order_id": "#W9571698", "payment_method_id": "credit_card_1565124"}): expected error busine | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_2e134f9291e5: modify_pending_order_items write: differs | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |
| `trusted` | red | 0 of 1 | task task_000ffeb205cd: the D79 suite did not pass | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `modify_pending_order_items` | write | 139 | 122 | 17 | 0 | 0 | value | 4 | replay_fidelity | yes | `replays.json task_2e134f9291e5/2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35 checks[11]` |
| `return_delivered_order_items` | write | 148 | 140 | 0 | 0 | 8 | real_errored | 4 | replay_fidelity | yes | `replays.json task_09638ac8e2b2/5235d473-52bc-4e26-82a5-74f1a3513ce0 checks[6]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | value | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | value | 4 | replay_fidelity | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `calculate` | read | 33 | 33 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `exchange_delivered_order_items` | write | 95 | 95 | 0 | 0 | 0 | none | 4 | none |  | `replays.json checks` |
| `find_user_id_by_email` | read | 123 | 123 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `find_user_id_by_name_zip` | read | 366 | 366 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_item_details` | read | 24 | 24 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_product_details` | read | 414 | 414 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `list_all_product_types` | read | 19 | 19 | 0 | 0 | 0 | none | 2 | none |  | `replays.json checks` |
| `modify_pending_order_address` | write | 76 | 76 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_pending_order_payment` | write | 4 | 4 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_user_address` | write | 35 | 35 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `transfer_to_human_agents` | read | 38 | 38 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |

Agrees is `same`, `cosmetic` and `both_refused` together, which is how `runner/replay.py` counts a call that agrees; a refusal on both sides whose wording differs is agreement here and a miss to `env_fidelity.py`, which reads the whole answer rather than the preview replays.json keeps.

### Where the misses come from

| cause | calls | share of the misses | tools | owner | read from |
|---|---:|---:|---|---|---|
| value | 22 | 73.3% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_items` 17 | compile_tools: a different answer with no shape or error explanation | `replays.json` |
| real_errored | 8 | 26.7% | `return_delivered_order_items` 8 | reference: the real tool refused where ours answered | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_000ffeb205cd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_000ffeb205cd` |
| `task_039604eb2f69` | yes | no | leak_check_clean |  | 6 required of 38: modify_pending_order_items writes #W5061109 | the Verifier did not pass the D79 suite | `task_status.json task_039604eb2f69` |
| `task_03c32a57c05e` | yes | no | second_path_passes | verifier_alt_path | 7 required of 69: cancel_pending_order writes #W5995614 | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_041a2ef3294c` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_041a2ef3294c` |
| `task_06efc5b06acb` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_06efc5b06acb` |
| `task_09ba7fddcd5f` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09ba7fddcd5f` |
| `task_0d00a5249f02` | yes | no | second_path_passes, leak_check_clean | verifier_alt_path | 5 required of 44: exchange_delivered_order_items writes #W9502127 | the Verifier did not pass the D79 suite | `task_status.json task_0d00a5249f02` |
| `task_1211c761ec81` | yes | no | second_path_passes | verifier_alt_path | 6 required of 76: modify_pending_order_address writes #W4082615 | the Verifier did not pass the D79 suite | `task_status.json task_1211c761ec81` |
| `task_1a0555df39f1` | yes | no | second_path_passes | verifier_alt_path | 5 required of 44: exchange_delivered_order_items writes #W2378156 | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_24633c648d01` | yes | no | leak_check_clean |  | 8 required of 64: modify_pending_order_address writes #W5270061 | the Verifier did not pass the D79 suite | `task_status.json task_24633c648d01` |
| `task_3c1a364cb85d` | yes | no | leak_check_clean |  | 6 required of 51: exchange_delivered_order_items writes #W3239882 | the Verifier did not pass the D79 suite | `task_status.json task_3c1a364cb85d` |
| `task_3f7d47009c93` | yes | no | second_path_passes | verifier_alt_path | 6 required of 47: modify_user_address writes FATIMA_TAYLOR_3452 | the Verifier did not pass the D79 suite | `task_status.json task_3f7d47009c93` |
| `task_3ff6b56d38f7` | yes | no | mutation_flips |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_3ff6b56d38f7` |
| `task_46a0d3f4d1e2` | yes | no | leak_check_clean |  | 3 required of 48: modify_user_address writes LUCAS_SANTOS_6600 | the Verifier did not pass the D79 suite | `task_status.json task_46a0d3f4d1e2` |
| `task_4c969e4c1fb8` | yes | no | second_path_passes, mutation_flips | verifier_alt_path | 1 required of 45: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_4c969e4c1fb8` |
| `task_4e59c76d1a2a` | yes | no | second_path_passes, leak_check_clean | verifier_alt_path | 6 required of 41: exchange_delivered_order_items writes #W7209932 | the Verifier did not pass the D79 suite | `task_status.json task_4e59c76d1a2a` |
| `task_55bde35c8b79` | yes | no | unsolved_state_fails, mutation_flips, leak_check_clean |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_55bde35c8b79` |
| `task_56470c280bfd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_56470c280bfd` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

| round | gate counts (D126) | moved | artifacts changed | spend | turns | exit | read from |
|---:|---|---|---|---|---|---|---|
| 1 | fidelity 199, trusted 75, refused_count 0, assisted_runs 0, probes_passing 0 | first round | first round | builder $1.74, examiner $0.22, total $1.96 | builder 0, examiner 0, total 0 | the round did not exit | `rounds.json[0]` |
| 2 | fidelity 199, trusted 75, refused_count 0, assisted_runs 0, probes_passing 0 | none | none | builder $0.00, examiner $0.07, total $0.07 | builder 0, examiner 0, total 0 | stalled | `rounds.json[1]` |

## Per repair verb

No repair verb was called: the workdir has no `repairs/` directory, which is what a build under the code driver leaves (D135).

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| body_exception | ours | 3 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 16 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 124 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 57 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 141 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |
| unattributed | unattributed | 14 | no rule above matched what the Run recorded | `runs/`, `replays.json`, `references.json` |

355 failing Runs: 143 ours, 57 the model's, 155 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `replay-2dfd6f1f-4d3d-443d-b55d-38edbd37bf76` | replay | ours | body_exception | IndexError: list index out of range | `runs/task_2e134f9291e5/replay-2dfd6f1f-4d3d-443d-b55d-38edbd37bf76.jsonl` |
| `replay-a4d65729-1856-4b33-9aec-d570ab79f364` | replay | ours | body_exception | IndexError: list index out of range | `runs/task_82620677bdf5/replay-a4d65729-1856-4b33-9aec-d570ab79f364.jsonl` |
| `reroll-task_82620677bdf5-0` | reroll | ours | body_exception | IndexError: list index out of range | `runs/task_82620677bdf5/reroll-task_82620677bdf5-0.jsonl` |
| `replay-2535d17c-a600-42cf-b74f-86718674fe44` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_4fc41fab480d/replay-2535d17c-a600-42cf-b74f-86718674fe44.jsonl` |
| `replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_2e134f9291e5/replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35.jsonl` |
| `replay-50c539c3-7f61-4ee4-aece-dddaaeee4886` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_52cace5aa209/replay-50c539c3-7f61-4ee4-aece-dddaaeee4886.jsonl` |
| `reroll-task_0d00a5249f02-0` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0d00a5249f02/reroll-task_0d00a5249f02-0.jsonl` |
| `reroll-task_0d00a5249f02-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0d00a5249f02/reroll-task_0d00a5249f02-1.jsonl` |
| `reroll-task_0d00a5249f02-2` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0d00a5249f02/reroll-task_0d00a5249f02-2.jsonl` |
| `replay-77d71860-9c54-423d-bb0b-df62f4fcb332` | replay | the model's | env_refused_the_candidate | business_error: ValueError: New item 5753502325 not found or available | `runs/task_de3d8a4f9893/replay-77d71860-9c54-423d-bb0b-df62f4fcb332.jsonl` |
| `replay-af9b9687-2eb0-4f82-bfbf-c0c88620065f` | replay | the model's | env_refused_the_candidate | business_error: ValueError: User not found | `runs/task_7f32b58a1790/replay-af9b9687-2eb0-4f82-bfbf-c0c88620065f.jsonl` |
| `replay-b6654b05-c74c-47d8-adf9-1d0bf510f89b` | replay | the model's | env_refused_the_candidate | business_error: ValueError: User not found | `runs/task_763ad00b0046/replay-b6654b05-c74c-47d8-adf9-1d0bf510f89b.jsonl` |
| `replay-03b3101c-5962-4ee1-9f0d-56d91ef69f49` | replay | unattributed | judge_failed_it | task_9fd8a96486be: judge: A returned items but did not reroute the pet bed shipment; B made no writes, so it did neither requested database update. | `runs/task_9fd8a96486be/replay-03b3101c-5962-4ee1-9f0d-56d91ef69f49.jsonl` |
| `replay-0a92a7fc-64f0-41b2-8447-b0a138d3f901` | replay | unattributed | judge_failed_it | task_dca12e38f942: judge: A exchanges the item for the identical item ID, so it does not effect an exchange. B performs no writes, so the requested exchange is  | `runs/task_dca12e38f942/replay-0a92a7fc-64f0-41b2-8447-b0a138d3f901.jsonl` |
| `replay-1590d7de-7cfb-4dfc-9d87-496030e948cb` | replay | unattributed | judge_failed_it | task_9ffb7baf2fb2: judge: State A made no writes, so it did not cancel both orders as requested. State B attempted cancellation of both orders; authentication a | `runs/task_9ffb7baf2fb2/replay-1590d7de-7cfb-4dfc-9d87-496030e948cb.jsonl` |
| `replay-351db446-7efd-48ab-9269-b0f86e148505` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_5ef4c9e9d199/replay-351db446-7efd-48ab-9269-b0f86e148505.jsonl` |
| `replay-36611a4a-08df-4daf-8885-7890b314c654` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_197b67aea13a/replay-36611a4a-08df-4daf-8885-7890b314c654.jsonl` |
| `replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627` | replay | unattributed | unattributed | return_delivered_order_items write: theirs_refused | `runs/task_0da6dbdf8d5c/replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627.jsonl` |

337 more failing Runs not listed; pass `--limit 0` to print every row.
