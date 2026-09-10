---
archived: true
archived_on: 2026-09-10
---
# Build table: .work-retail

**Tasks covered: 20 of 205 frozen Tasks (9.8%), 56 of 456 Runs (12.3%)**

Read from `.work-retail` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

Records this workdir does not hold: `rounds.json`, `builder/session.jsonl`, `examiner/session.jsonl`.

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 20 of 205 frozen Tasks (9.8%), 56 of 456 Runs (12.3%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 343 of 456 (75.2%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 472 of 575 (82.1%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 38 of 2,645 (1.4%) | `gates.json replay_reference metrics` |
| Gates green | 8 of 15 named below | `gates.json` |
| What the mechanic called | n/a (needs builder/session.jsonl) | `builder/session.jsonl` |
| Repairs requested | 0; the workdir has no repairs/ directory (D135) | `repairs/` |
| Repairs that turned a red gate green | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/, gates.json` |
| Repairs that did not | 0; the workdir has no repairs/ directory, and the code driver files no repair, so there is nothing to have moved a gate (D135) | `repairs/, gates.json` |
| Dollars | $6.49 | `budget.json total.usd` |
| Model calls | 11,543 (8,056 priced, 3,487 unpriced, 992 memo hits) | `budget.json total` |
| Model call wall time | 4h 51m 7s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | n/a (needs a start and an end timestamp; pipeline/state.json records the stage statuses and no clock) | `pipeline/state.json` |
| Peak context fill, Builder | n/a (needs builder/session.jsonl and ContextStats.fill_at_turn_end, which no stage writes into the workdir) | `builder/session.jsonl` |
| Peak context fill, Examiner | n/a (needs examiner/session.jsonl and ContextStats.fill_at_turn_end, which no stage writes into the workdir) | `examiner/session.jsonl` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `build_user_rules` | green | 1 of 1 |  | `gates.json` |
| `compile_policy` | red | 0 of 1 | c_6d8ea589e4cf: pos case raised TypeError: check() missing 2 required positional arguments: 'write_call' and 'transcript' | `gates.json` |
| `confined` | green | 16 of 16 |  | `gates.json` |
| `derive_verifier` | red | 0 of 1 | task task_55bde35c8b79: the D79 suite did not pass | `gates.json` |
| `deterministic` | green | 15 of 15 |  | `gates.json` |
| `executes_on_s0` | red | 15 of 16 | calculate({"expression": "2621.77 - 2508.06"}) raised NameError: name 'decimal' is not defined | `gates.json` |
| `intent` | red | 0 of 1 | task task_000ffeb205cd: noun phrases not evidenced in every Run: canister vacuum cleaner (not in e8bd582d-90f1-4a3f-840a-0bc8eb5e80fd) | `gates.json` |
| `non_trivial` | green | 15 of 15 |  | `gates.json` |
| `parses` | green | 16 of 16 |  | `gates.json` |
| `refuses_unknown` | red | 5 of 6 | exchange_delivered_order_items({"item_ids": ["1615379700"], "new_item_ids": ["8106223139"], "order_id": "#W1304208", "payment_method_id": "paypal_1679 | `gates.json` |
| `replay_fidelity` | red | 26 of 28 | modify_pending_order_items({"item_ids": ["3254583681"], "new_item_ids": ["2635605237"], "order_id": "#W5061109", "payment_method_id": "paypal_3742148" | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_039604eb2f69: modify_pending_order_items write: differs | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |
| `vocabulary` | green | 1 of 1 |  | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `modify_pending_order_items` | write | 139 | 47 | 92 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_039604eb2f69/d9c609f4-5be4-4caa-b48c-4cd8b9d28c02 checks[4]` |
| `calculate` | read | 33 | 0 | 0 | 33 | 0 | missing_import | 4 | executes_on_s0 | yes | `replays.json task_018fc5cbf7e9/af9889a2-59b6-4084-93db-0d92b790db8a checks[4]` |
| `modify_pending_order_address` | write | 76 | 65 | 0 | 11 | 0 | body_error | 1 | none |  | `replays.json task_24633c648d01/284cbf5a-0965-4f63-a02d-ce06f4dce316 checks[9]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | unreadable | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | unreadable | 4 | replay_fidelity | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `exchange_delivered_order_items` | write | 95 | 95 | 0 | 0 | 0 | none | 4 | refuses_unknown | yes | `replays.json checks` |
| `find_user_id_by_email` | read | 123 | 123 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `find_user_id_by_name_zip` | read | 366 | 366 | 0 | 0 | 0 | none | 2 | none |  | `replays.json checks` |
| `get_item_details` | read | 24 | 24 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_product_details` | read | 414 | 414 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `list_all_product_types` | read | 19 | 19 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_pending_order_payment` | write | 4 | 4 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_user_address` | write | 35 | 35 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `return_delivered_order_items` | write | 148 | 148 | 0 | 0 | 0 | none | 3 | none |  | `replays.json checks` |
| `transfer_to_human_agents` | read | 38 | 38 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |

Agrees is `same`, `cosmetic` and `both_refused` together, which is how `runner/replay.py` counts a call that agrees; a refusal on both sides whose wording differs is agreement here and a miss to `env_fidelity.py`, which reads the whole answer rather than the preview replays.json keeps.

### Where the misses come from

| cause | calls | share of the misses | tools | owner | read from |
|---|---:|---:|---|---|---|
| unreadable | 97 | 68.8% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_items` 92 | n/a: replays.json keeps a 160 character preview of each answer, which is not enough to name the cause | `replays.json` |
| missing_import | 33 | 23.4% | `calculate` 33 | compile_tools: the body used a module it never imported | `replays.json` |
| body_error | 11 | 7.8% | `modify_pending_order_address` 11 | compile_tools: the body raised where the real tool answered | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_000ffeb205cd` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_000ffeb205cd` |
| `task_03c32a57c05e` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_03d78b3f5cea` | yes | no | unsolved_state_fails, second_path_passes, mutation_flips | verifier_alt_path | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_03d78b3f5cea` |
| `task_041a2ef3294c` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_041a2ef3294c` |
| `task_06efc5b06acb` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_06efc5b06acb` |
| `task_09638ac8e2b2` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09638ac8e2b2` |
| `task_09ba7fddcd5f` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09ba7fddcd5f` |
| `task_0d00a5249f02` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_0d00a5249f02` |
| `task_0da6dbdf8d5c` | yes | no | mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_0da6dbdf8d5c` |
| `task_0fdb72dff11a` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_0fdb72dff11a` |
| `task_1211c761ec81` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1211c761ec81` |
| `task_126d2df50be8` | yes | no | mutation_flips |  | 1 required of 37: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_126d2df50be8` |
| `task_12bf34ff4998` | yes | no | mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_12bf34ff4998` |
| `task_15472455ed4d` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_15472455ed4d` |
| `task_197b67aea13a` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_197b67aea13a` |
| `task_1a0555df39f1` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 32: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | empty_fails, unsolved_state_fails, plausible_wrong_fails, loophole_probe_fails, mutation_flips |  | 1 required of 30: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_37d1178f5430` | yes | no | mutation_flips |  | 1 required of 33: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_37d1178f5430` |
| `task_3c1a364cb85d` | yes | no | mutation_flips |  | 1 required of 31: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_3c1a364cb85d` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

n/a (needs rounds.json; this build ran no round driver).

## Per repair verb

No repair verb was called: the workdir has no `repairs/` directory, which is what a build under the code driver leaves (D135).

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| provider_error | ours | 111 | an error event naming the provider or the transport, so no model turn ever happened | `runs/`, `replays.json`, `references.json` |
| body_exception | ours | 192 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| replay_refused | ours | 10 | a replayed call we refused and the recording answered | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 76 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 48 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 23 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 129 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |

589 failing Runs: 437 ours, 23 the model's, 129 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `reroll-task_018fc5cbf7e9-0` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_018fc5cbf7e9/reroll-task_018fc5cbf7e9-0.jsonl` |
| `reroll-task_018fc5cbf7e9-1` | reroll | ours | provider_error | ProviderError: openai/gpt-5.6-luna: HTTP 400: Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead. | `runs/task_018fc5cbf7e9/reroll-task_018fc5cbf7e9-1.jsonl` |
| `replay-079cf9b3-d7c9-438e-a87a-a3697ac47b23` | replay | ours | body_exception | NameError: name 'decimal' is not defined | `runs/task_5cd55b38f1a9/replay-079cf9b3-d7c9-438e-a87a-a3697ac47b23.jsonl` |
| `replay-13875336-5ac2-4f14-a9f7-d0f911dff139` | replay | ours | body_exception | NameError: name 'decimal' is not defined | `runs/task_623ba789b049/replay-13875336-5ac2-4f14-a9f7-d0f911dff139.jsonl` |
| `replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_8e41df099d75/replay-0886d470-7d2e-4e1c-9fbc-e4f3ae752a51.jsonl` |
| `replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71` | replay | ours | replay_refused | modify_pending_order_address write: ours_refused | `runs/task_b72a88c48881/replay-196c0f80-487e-48d9-bb0d-68f3e8b98c71.jsonl` |
| `replay-05a7b05d-c2f5-4f72-a30e-73960cffad45` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_2eba497a317d/replay-05a7b05d-c2f5-4f72-a30e-73960cffad45.jsonl` |
| `replay-0608dfd1-affd-4965-8c34-8a2159d213f5` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_9e2ffb60be01/replay-0608dfd1-affd-4965-8c34-8a2159d213f5.jsonl` |
| `reroll-task_0866ed543e0c-0` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0866ed543e0c/reroll-task_0866ed543e0c-0.jsonl` |
| `reroll-task_0866ed543e0c-1` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_0866ed543e0c/reroll-task_0866ed543e0c-1.jsonl` |
| `replay-266babd2-6f0b-4fb1-94eb-e708e24f2c24` | replay | the model's | env_refused_the_candidate | business_error: ValueError: User not found | `runs/task_12bf34ff4998/replay-266babd2-6f0b-4fb1-94eb-e708e24f2c24.jsonl` |
| `replay-281cc608-8938-49cd-a753-e8fa367ba295` | replay | the model's | env_refused_the_candidate | business_error: ValueError: User not found | `runs/task_9bdba5bb7858/replay-281cc608-8938-49cd-a753-e8fa367ba295.jsonl` |
| `replay-010a23ef-3524-4dbf-ad95-81d4cf974a45` | replay | unattributed | judge_failed_it | task_dbf0c8930714: violates c_9f52317fb6df, c_ea9943e77c85 | `runs/task_dbf0c8930714/replay-010a23ef-3524-4dbf-ad95-81d4cf974a45.jsonl` |
| `replay-0317e4c5-bdc3-4c78-addb-3cf1c751f5cb` | replay | unattributed | judge_failed_it | task_5cd55b38f1a9: judge: A performed the requested cancellations and return but did so without the required user authentication and explicit confirmation befor | `runs/task_5cd55b38f1a9/replay-0317e4c5-bdc3-4c78-addb-3cf1c751f5cb.jsonl` |

575 more failing Runs not listed; pass `--limit 0` to print every row.
