# Build table: .work-b9-code

**Tasks covered: 48 of 205 frozen Tasks (23.4%), 119 of 456 Runs (26.1%)**

Read from `.work-b9-code` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 48 of 205 frozen Tasks (23.4%), 119 of 456 Runs (26.1%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 347 of 456 (76.1%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 452 of 575 (78.6%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 5 of 2,645 (0.2%) | `gates.json replay_reference metrics` |
| Gates green | 9 of 15 named below | `gates.json` |
| What the mechanic called | 0 over 0 model turns; the session records no model turn, which is the code driver (D135) | `builder/session.jsonl` |
| Repairs requested | 0; the workdir has no repairs/ directory (D135) | `repairs/` |
| Repairs that turned a red gate green | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/, gates.json` |
| Repairs that did not | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/, gates.json` |
| Dollars | $8.11 | `budget.json total.usd` |
| Model calls | 16,207 (12,720 priced, 3,487 unpriced, 1,086 memo hits) | `budget.json total` |
| Model call wall time | 7h 10m 10s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | n/a (needs a start and an end timestamp; pipeline/state.json records the stage statuses and no clock) | `pipeline/state.json` |
| Peak context fill, Builder | n/a (needs ContextStats.fill_at_turn_end, which no stage writes into the workdir) | `builder/session.jsonl` |
| Peak context fill, Examiner | n/a (needs ContextStats.fill_at_turn_end, which no stage writes into the workdir) | `examiner/session.jsonl` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `build_user_rules` | green | 1 of 1 |  | `gates.json` |
| `compile_policy` | green | 1 of 1 |  | `gates.json` |
| `confined` | green | 16 of 16 |  | `gates.json` |
| `derive_verifier` | red | 0 of 1 | task task_55bde35c8b79: the D79 suite did not pass | `gates.json` |
| `deterministic` | green | 15 of 15 |  | `gates.json` |
| `executes_on_s0` | red | 15 of 16 | exchange_delivered_order_items({"item_ids": ["1615379700"], "new_item_ids": ["8106223139"], "order_id": "#W1304208", "payment_method_id": "paypal_1679 | `gates.json` |
| `intent` | red | 0 of 1 | task task_000ffeb205cd: noun phrases not evidenced in every Run: canister vacuum cleaner (not in e8bd582d-90f1-4a3f-840a-0bc8eb5e80fd) | `gates.json` |
| `non_trivial` | green | 15 of 15 |  | `gates.json` |
| `parses` | green | 16 of 16 |  | `gates.json` |
| `refuses_unknown` | green | 5 of 5 |  | `gates.json` |
| `replay_fidelity` | red | 26 of 28 | modify_pending_order_items({"item_ids": ["3694871183"], "new_item_ids": ["6077640618"], "order_id": "#W5061109", "payment_method_id": "paypal_3742148" | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_0866ed543e0c: exchange_delivered_order_items write: ours_refused | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |
| `trusted` | red | 0 of 1 | task task_000ffeb205cd: the D79 suite did not pass | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `exchange_delivered_order_items` | write | 95 | 1 | 0 | 94 | 0 | body_error | 4 | executes_on_s0 | yes | `replays.json task_0866ed543e0c/15c91efb-264e-49cf-bcd8-6d03a24f4a05 checks[7]` |
| `modify_pending_order_items` | write | 139 | 122 | 17 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_2e134f9291e5/2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35 checks[11]` |
| `modify_pending_order_address` | write | 76 | 65 | 0 | 11 | 0 | body_error | 1 | none |  | `replays.json task_24633c648d01/284cbf5a-0965-4f63-a02d-ce06f4dce316 checks[9]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | unreadable | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `return_delivered_order_items` | write | 148 | 147 | 0 | 0 | 1 | real_errored | 3 | none |  | `replays.json task_d8954d758335/ad96d823-7edc-4d21-a815-25aac88cd134 checks[7]` |
| `calculate` | read | 33 | 33 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
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
| body_error | 105 | 82.0% | `exchange_delivered_order_items` 94, `modify_pending_order_address` 11 | compile_tools: the body raised where the real tool answered | `replays.json` |
| unreadable | 22 | 17.2% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_items` 17 | n/a: replays.json keeps a 160 character preview of each answer, which is not enough to name the cause | `replays.json` |
| real_errored | 1 | 0.8% | `return_delivered_order_items` 1 | reference: the real tool refused where ours answered | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_000ffeb205cd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_000ffeb205cd` |
| `task_039604eb2f69` | yes | no | leak_check_clean |  | 6 required of 45: modify_pending_order_items writes #W5061109 | the Verifier did not pass the D79 suite | `task_status.json task_039604eb2f69` |
| `task_03c32a57c05e` | yes | no | second_path_passes | verifier_alt_path | 7 required of 76: cancel_pending_order writes #W5995614 | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_041a2ef3294c` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_041a2ef3294c` |
| `task_06efc5b06acb` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_06efc5b06acb` |
| `task_09638ac8e2b2` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09638ac8e2b2` |
| `task_09ba7fddcd5f` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09ba7fddcd5f` |
| `task_1a0555df39f1` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 41: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 40: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_2eba497a317d` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_2eba497a317d` |
| `task_37d1178f5430` | yes | no | second_path_passes | verifier_alt_path | 5 required of 46: return_delivered_order_items writes #W6239298 | the Verifier did not pass the D79 suite | `task_status.json task_37d1178f5430` |
| `task_3f7d47009c93` | yes | no | second_path_passes | verifier_alt_path | 6 required of 54: modify_user_address writes FATIMA_TAYLOR_3452 | the Verifier did not pass the D79 suite | `task_status.json task_3f7d47009c93` |
| `task_3ff6b56d38f7` | yes | no | mutation_flips |  | 1 required of 40: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_3ff6b56d38f7` |
| `task_44854ce70e66` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_44854ce70e66` |
| `task_4589c978e041` | yes | no | mutation_flips |  | 1 required of 44: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_4589c978e041` |
| `task_4ba1606f3306` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 40: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_4ba1606f3306` |
| `task_4c969e4c1fb8` | yes | no | second_path_passes | verifier_alt_path | 17 required of 67: modify_pending_order_address writes #W1092119 | the Verifier did not pass the D79 suite | `task_status.json task_4c969e4c1fb8` |
| `task_55bde35c8b79` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 40: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_55bde35c8b79` |
| `task_56470c280bfd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 38: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_56470c280bfd` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

| round | gate counts (D126) | moved | spend | turns | exit | read from |
|---:|---|---|---|---|---|---|
| 1 | fidelity 159, trusted 48, refused_count 0, assisted_runs 0, probes_passing 0 | first round | builder $1.42, examiner $0.16, total $1.58 | 0, no model turn recorded | the round did not exit | `rounds.json[0]` |
| 2 | fidelity 159, trusted 48, refused_count 0, assisted_runs 0, probes_passing 0 | none | builder $0.00, examiner $0.03, total $0.03 | 0, no model turn recorded | stalled | `rounds.json[1]` |

## Per repair verb

No repair verb was called: the workdir has no `repairs/` directory, which is what a build under the code driver leaves (D135).

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| provider_error | ours | 21 | an error event naming the provider or the transport, so no model turn ever happened | `runs/`, `replays.json`, `references.json` |
| body_exception | ours | 133 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| replay_refused | ours | 11 | a replayed call we refused and the recording answered | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 17 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 68 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 33 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 124 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |

407 failing Runs: 250 ours, 33 the model's, 124 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `reroll-task_55d0d0ade502-0` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-0.jsonl` |
| `reroll-task_55d0d0ade502-1` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_55d0d0ade502/reroll-task_55d0d0ade502-1.jsonl` |
| `replay-08003115-be07-4cf4-9bf0-7fee49d3c8d4` | replay | ours | body_exception | AttributeError: 'dict' object has no attribute 'available' | `runs/task_623ba789b049/replay-08003115-be07-4cf4-9bf0-7fee49d3c8d4.jsonl` |
| `replay-0a92a7fc-64f0-41b2-8447-b0a138d3f901` | replay | ours | body_exception | AttributeError: 'dict' object has no attribute 'available' | `runs/task_dca12e38f942/replay-0a92a7fc-64f0-41b2-8447-b0a138d3f901.jsonl` |
| `replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_8e41df099d75/replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51.jsonl` |
| `replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_b72a88c48881/replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71.jsonl` |
| `replay-2535d17c-a600-42cf-b74f-86718674fe44` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_4fc41fab480d/replay-2535d17c-a600-42cf-b74f-86718674fe44.jsonl` |
| `replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_2e134f9291e5/replay-2796fbd6-1a2b-45bc-831d-5c7b0e0e4c35.jsonl` |
| `reroll-task_018fc5cbf7e9-0` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_018fc5cbf7e9/reroll-task_018fc5cbf7e9-0.jsonl` |
| `reroll-task_018fc5cbf7e9-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_018fc5cbf7e9/reroll-task_018fc5cbf7e9-1.jsonl` |
| `replay-351db446-7efd-48ab-9269-b0f86e148505` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Payment method does not belong to the order | `runs/task_5ef4c9e9d199/replay-351db446-7efd-48ab-9269-b0f86e148505.jsonl` |
| `replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Payment method does not belong to the order | `runs/task_0da6dbdf8d5c/replay-3b57483e-c84b-46d0-b2b3-c0c7d5f17627.jsonl` |
| `replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5` | replay | unattributed | judge_failed_it | task_5ef4c9e9d199: judge: The intent required returning only the more expensive tablet and refunding the credit card. A and B used a gift card, and B returned f | `runs/task_5ef4c9e9d199/replay-03281b9d-4a28-4fa1-bd9c-23b48dbc1ae5.jsonl` |
| `replay-086b19c9-0244-46d4-b315-8c59fbc63301` | replay | unattributed | judge_failed_it | task_50b019d79a4c: judge: The intent was to swap the luggage set for a coat. A and B canceled the order instead, and C made no database update, so none performe | `runs/task_50b019d79a4c/replay-086b19c9-0244-46d4-b315-8c59fbc63301.jsonl` |

393 more failing Runs not listed; pass `--limit 0` to print every row.
