# Build table: .work-b13

**Tasks covered: 73 of 205 frozen Tasks (35.6%), 164 of 456 Runs (36.0%)**

Read from `.work-b13` by `scripts/build_table.py` (D139). Every line names the file it was read from, and a number no record here holds is printed as `n/a` with the record it would need (D66).

| row | value | read from |
|---|---|---|
| Tasks covered (D96) | 73 of 205 frozen Tasks (35.6%), 164 of 456 Runs (36.0%) | `scorecard.json task_coverage, tasks_frozen.json task_ids` |
| Traces confirming their Reference | 334 of 456 (73.2%) | `gates.json replay_reference metrics` |
| Writes replaying exactly | 430 of 575 (74.8%) | `gates.json replay_reference metrics` |
| Reads differing in substance | 5 of 2,645 (0.2%) | `gates.json replay_reference metrics` |
| Gates green | 8 of 11 named below | `gates.json` |
| What the mechanic called | 127 over 132 model turns: status 75, repair_recompile 27, repair_intent 16, replay 6, build 3 | `builder/session.jsonl` |
| Repairs requested | 43 requested: repair_recompile 27, repair_intent 16 | `repairs/` |
| Repairs that turned a red gate green | 0 of 43; 16 not placed: n/a (needs the round on a repair request and the rulings of both the round before it and its own; repairs/*.jsonl records round and gates_by_round.json the rulings per round) | `repairs/ round, gates_by_round.json` |
| Repairs that did not | 27 of 43: repair_intent on task_03d78b3f5cea (round 2), repair_intent on task_6735d433a7b4 (round 2), repair_intent on task_67ec25252b30 (round 2), repair_intent on task_b72a88c48881 (round 2), repair_intent on task_e780df33f5d4 (round 2), and 22 more; 16 not placed: n/a (needs the round on a repair request and the rulings of both the round before it and its own; repairs/*.jsonl records round and gates_by_round.json the rulings per round) | `repairs/ round, gates_by_round.json` |
| Dollars | $11.96; the cache saved $8.97: 77,926,560 cache-read tokens at the cache rate, 1,402 memo hits sent nothing; without it $20.93 | `budget.json total.usd, total.cache_saved_usd` |
| Model calls | 27,549 (24,062 priced, 3,487 unpriced, 1,402 memo hits) | `budget.json total` |
| Model call wall time | 12h 57m 41s summed over calls, which is not the build's own duration | `budget.json total.wall_ms` |
| Build duration | 16h 0m 21s over 2 rounds, 2026-09-06T17:14:32+00:00 to 2026-09-07T09:14:54+00:00 | `rounds.json counts.started_at, counts.ended_at` |
| Peak context fill, Builder | 22.0% of the window at a turn end, over 132 turns in 2 rounds | `rounds.json counts.context_fill` |
| Peak context fill, Examiner | 608.9% of the window at a turn end, over 12 turns in 2 rounds | `rounds.json counts.context_fill` |

## Gates

| gate | ruling | green rulings | what the last red one says | read from |
|---|---|---:|---|---|
| `confined` | green | 1 of 1 |  | `gates.json` |
| `deterministic` | green | 1 of 1 |  | `gates.json` |
| `executes_on_s0` | green | 1 of 1 |  | `gates.json` |
| `intent` | red | 0 of 1 | task task_e780df33f5d4: noun phrases not evidenced in every Run: change (not in 91a368d6-f490-471b-b05b-1138a7ac9d86) | `gates.json` |
| `non_trivial` | green | 1 of 1 |  | `gates.json` |
| `parses` | green | 1 of 1 |  | `gates.json` |
| `refuses_unknown` | green | 6 of 6 |  | `gates.json` |
| `replay_fidelity` | red | 0 of 1 | modify_pending_order_items({"item_ids": ["3694871183"], "new_item_ids": ["6077640618"], "order_id": "#W5061109", "payment_method_id": "paypal_3742148" | `gates.json` |
| `replay_reference` | red | 0 of 1 | task task_018fc5cbf7e9: modify_pending_order_items write: ours_refused | `gates.json` |
| `rerolls` | green | 1 of 1 |  | `gates.json` |
| `tau2_export` | green | 1 of 1 |  | `gates.json` |

## Per tool, over the replayed calls

| tool | kind | calls | agrees | differs | ours refused | theirs refused | top cause | bodies | gates red on the last body | assisted | read from |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `modify_pending_order_items` | write | 139 | 5 | 8 | 126 | 0 | body_error | 4 | replay_fidelity | yes | `replays.json task_018fc5cbf7e9/28740ae1-fb3d-4159-97ba-11c7e966fdd5 checks[4]` |
| `modify_pending_order_address` | write | 76 | 65 | 11 | 0 | 0 | value | 1 | none |  | `replays.json task_24633c648d01/284cbf5a-0965-4f63-a02d-ce06f4dce316 checks[9]` |
| `get_order_details` | read | 1,204 | 1,200 | 4 | 0 | 0 | value | 1 | none |  | `replays.json task_2e134f9291e5/2dfd6f1f-4d3d-443d-b55d-38edbd37bf76 checks[11]` |
| `get_user_details` | read | 424 | 423 | 1 | 0 | 0 | value | 4 | replay_fidelity | yes | `replays.json task_4fc41fab480d/2535d17c-a600-42cf-b74f-86718674fe44 checks[10]` |
| `calculate` | read | 33 | 33 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `cancel_pending_order` | write | 78 | 78 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `exchange_delivered_order_items` | write | 95 | 95 | 0 | 0 | 0 | none | 4 | none |  | `replays.json checks` |
| `find_user_id_by_email` | read | 123 | 123 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `find_user_id_by_name_zip` | read | 366 | 366 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_item_details` | read | 24 | 24 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `get_product_details` | read | 414 | 414 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `list_all_product_types` | read | 19 | 19 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_pending_order_payment` | write | 4 | 4 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `modify_user_address` | write | 35 | 35 | 0 | 0 | 0 | none | 1 | none |  | `replays.json checks` |
| `return_delivered_order_items` | write | 148 | 148 | 0 | 0 | 0 | none | 3 | none |  | `replays.json checks` |
| `transfer_to_human_agents` | read | 38 | 38 | 0 | 0 | 0 | none | 2 | none |  | `replays.json checks` |

Agrees is `same`, `cosmetic` and `both_refused` together, which is how `runner/replay.py` counts a call that agrees; a refusal on both sides whose wording differs is agreement here and a miss to `env_fidelity.py`, which reads the whole answer rather than the preview replays.json keeps.

### Where the misses come from

| cause | calls | share of the misses | tools | owner | read from |
|---|---:|---:|---|---|---|
| body_error | 126 | 84.0% | `modify_pending_order_items` 126 | compile_tools: the body raised where the real tool answered | `replays.json` |
| value | 24 | 16.0% | `get_order_details` 4, `get_user_details` 1, `modify_pending_order_address` 11, `modify_pending_order_items` 8 | compile_tools: a different answer with no shape or error explanation | `replays.json` |

## Per Task

| Task | Reference confirmed | Verifier trusted | D79 checks that failed | checks not run | atoms | reason | read from |
|---|---|---|---|---|---|---|---|
| `task_039604eb2f69` | yes | no | leak_check_clean |  | 6 required of 48: modify_pending_order_items writes #W5061109 | the Verifier did not pass the D79 suite | `task_status.json task_039604eb2f69` |
| `task_03c32a57c05e` | yes | no | second_path_passes | verifier_alt_path | 7 required of 79: cancel_pending_order writes #W5995614 | the Verifier did not pass the D79 suite | `task_status.json task_03c32a57c05e` |
| `task_09638ac8e2b2` | yes | no | mutation_flips |  | 1 required of 48: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_09638ac8e2b2` |
| `task_0d00a5249f02` | yes | no | leak_check_clean |  | 5 required of 50: exchange_delivered_order_items writes #W9502127 | the Verifier did not pass the D79 suite | `task_status.json task_0d00a5249f02` |
| `task_0eb898e0acde` | yes | no | leak_check_clean |  | 6 required of 50: modify_pending_order_items writes #W5199551 | the Verifier did not pass the D79 suite | `task_status.json task_0eb898e0acde` |
| `task_1a0555df39f1` | yes | no | second_path_passes | verifier_alt_path | 5 required of 53: exchange_delivered_order_items writes #W2378156 | the Verifier did not pass the D79 suite | `task_status.json task_1a0555df39f1` |
| `task_1a6f39478206` | yes | no | leak_check_clean |  | 9 required of 53: return_delivered_order_items writes #W5490111 | the Verifier did not pass the D79 suite | `task_status.json task_1a6f39478206` |
| `task_1c59c73758e6` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 43: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1c59c73758e6` |
| `task_1f7556d168fa` | yes | no | second_path_passes, mutation_flips | verifier_alt_path | 1 required of 54: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_1f7556d168fa` |
| `task_2eba497a317d` | yes | no | leak_check_clean |  | 14 required of 76: modify_pending_order_address writes #W3730488 | the Verifier did not pass the D79 suite | `task_status.json task_2eba497a317d` |
| `task_3ff6b56d38f7` | yes | no | mutation_flips, leak_check_clean |  | 1 required of 43: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_3ff6b56d38f7` |
| `task_42b60c50e545` | yes | no | leak_check_clean |  | 6 required of 53: modify_pending_order_items writes #W4082615 | the Verifier did not pass the D79 suite | `task_status.json task_42b60c50e545` |
| `task_43972b9469d4` | yes | no | leak_check_clean |  | 5 required of 51: modify_pending_order_items writes #W8661412 | the Verifier did not pass the D79 suite | `task_status.json task_43972b9469d4` |
| `task_44854ce70e66` | yes | no | second_path_passes | verifier_alt_path | 10 required of 86: modify_pending_order_address writes #W3730488 | the Verifier did not pass the D79 suite | `task_status.json task_44854ce70e66` |
| `task_4589c978e041` | yes | no | second_path_passes | verifier_alt_path | 6 required of 53: exchange_delivered_order_items writes #W7181492 | the Verifier did not pass the D79 suite | `task_status.json task_4589c978e041` |
| `task_46a0d3f4d1e2` | yes | no | leak_check_clean |  | 3 required of 58: modify_user_address writes LUCAS_SANTOS_6600 | the Verifier did not pass the D79 suite | `task_status.json task_46a0d3f4d1e2` |
| `task_4ba1606f3306` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 44: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_4ba1606f3306` |
| `task_4c969e4c1fb8` | yes | no | mutation_flips |  | 1 required of 47: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_4c969e4c1fb8` |
| `task_4e59c76d1a2a` | yes | no | second_path_passes, leak_check_clean | verifier_alt_path | 6 required of 51: exchange_delivered_order_items writes #W7209932 | the Verifier did not pass the D79 suite | `task_status.json task_4e59c76d1a2a` |
| `task_55bde35c8b79` | yes | no | unsolved_state_fails, mutation_flips |  | 1 required of 43: the Run makes at most 0 write calls | the Verifier did not pass the D79 suite | `task_status.json task_55bde35c8b79` |

Ordered by how far the Task got: a confirmed Reference whose Verifier the suite refused first, then the Tasks with no confirmed Reference, then the covered ones.

185 more Tasks not listed; pass `--limit 0` to print every row.

## Per round

| round | gate counts (D126) | moved | artifacts changed | spend | turns | exit | read from |
|---:|---|---|---|---|---|---|---|
| 1 | fidelity 189, trusted 73, refused_count 0, assisted_runs 0, probes_passing 0 | first round | first round | builder $2.23, examiner $0.33, total $2.56, cache saved $3.35 | builder 11, examiner 8, total 19 | the round did not exit | `rounds.json[0]` |
| 2 | fidelity None, trusted None, refused_count None, assisted_runs None, probes_passing None | fidelity None from 189, trusted None from 73, refused_count None from 0, assisted_runs None from 0, probes_passing None from 0 | bodies, intents | builder $2.91, examiner $0.00, total $2.91, cache saved $5.61 | builder 121, examiner 4, total 125 | stalled | `rounds.json[1]` |

## Per repair verb

| verb | called on | round | ruling before | ruling after | turned green | read from |
|---|---|---:|---|---|---|---|
| `repair_intent` | `task_03d78b3f5cea` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_000ffeb205cd` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_018fc5cbf7e9` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_039604eb2f69` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_03d78b3f5cea` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_6735d433a7b4` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_67ec25252b30` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_b72a88c48881` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_b72a88c48881` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_intent` | `task_e780df33f5d4` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_intent.jsonl` |
| `repair_recompile` | `calculate` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `exchange_delivered_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 1 | no round ran before this one | red: derive_verifier, intent, replay_reference, trusted | n/a (needs gates_by_round.json) | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `get_user_details` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `list_all_product_types` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |
| `repair_recompile` | `modify_pending_order_items` | 2 | red: derive_verifier, intent, replay_reference, trusted | red: intent, replay_fidelity, replay_reference | none | `repairs/repair_recompile.jsonl` |

The rulings are the ones the round before this repair's round left and the ones its own round left (`gates_by_round.json`); repairs made in the same round share what that round moved.

## Per failing Run: the model's error or ours

| rule | side | Runs | what the record says | read from |
|---|---|---:|---|---|
| body_exception | ours | 139 | a tool result carrying a Python builtin exception, where the corpus shows a message (D67) | `runs/`, `replays.json`, `references.json` |
| answer_differs | ours | 8 | a replayed call whose answer differs from the recorded one in substance | `runs/`, `replays.json`, `references.json` |
| simulated_user_had_no_answer | ours | 17 | a user turn tagged fact_unavailable, where D44 says the recorded user gave the fact | `runs/`, `replays.json`, `references.json` |
| env_refused_the_candidate | the model's | 42 | the Environment refused what the model asked for, with the customer's own error class | `runs/`, `replays.json`, `references.json` |
| judge_failed_it | unattributed | 130 | the reference stage set the Run aside on the judge's words, which do not name a side | `runs/`, `replays.json`, `references.json` |
| unattributed | unattributed | 3 | no rule above matched what the Run recorded | `runs/`, `replays.json`, `references.json` |

339 failing Runs: 164 ours, 42 the model's, 133 unattributed. The rules are tried in the order above and the first one that matches decides the Run.

| Run | kind | side | rule | what the record says | read from |
|---|---|---|---|---|---|
| `replay-006f1488-f87e-4215-9e9e-27dc6b43268a` | replay | ours | body_exception | KeyError: '5320792178' | `runs/task_a10529127f1e/replay-006f1488-f87e-4215-9e9e-27dc6b43268a.jsonl` |
| `replay-05a7b05d-c2f5-4f72-a30e-73960cffad45` | replay | ours | body_exception | KeyError: '2216662955' | `runs/task_2eba497a317d/replay-05a7b05d-c2f5-4f72-a30e-73960cffad45.jsonl` |
| `replay-0608dfd1-affd-4965-8c34-8a2159d213f5` | replay | ours | body_exception | KeyError: '5052031638' | `runs/task_9e2ffb60be01/replay-0608dfd1-affd-4965-8c34-8a2159d213f5.jsonl` |
| `replay-0caa666a-bffb-4103-9ceb-05bc74b50b83` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_91bcf9206039/replay-0caa666a-bffb-4103-9ceb-05bc74b50b83.jsonl` |
| `replay-1b6e4d4e-6a71-4e29-a8a6-67a439c27ee7` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_91bcf9206039/replay-1b6e4d4e-6a71-4e29-a8a6-67a439c27ee7.jsonl` |
| `replay-3a56c3aa-ee2f-44dc-86c5-3f430ef7193c` | replay | ours | answer_differs | modify_pending_order_items write: differs | `runs/task_e36a312d163c/replay-3a56c3aa-ee2f-44dc-86c5-3f430ef7193c.jsonl` |
| `reroll-task_37d1178f5430-2` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_37d1178f5430/reroll-task_37d1178f5430-2.jsonl` |
| `reroll-task_46a0d3f4d1e2-2` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_46a0d3f4d1e2/reroll-task_46a0d3f4d1e2-2.jsonl` |
| `reroll-task_48ae3c959915-0` | reroll | ours | simulated_user_had_no_answer | a user turn is tagged fact_unavailable | `runs/task_48ae3c959915/reroll-task_48ae3c959915-0.jsonl` |
| `replay-2c000c43-e180-4af3-9342-691c5b3c2317` | replay | the model's | env_refused_the_candidate | not_found_entity: Error: Order not found | `runs/task_000ffeb205cd/replay-2c000c43-e180-4af3-9342-691c5b3c2317.jsonl` |
| `replay-5269dac6-ab05-48aa-a857-38c9c107143b` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Payment method does not match order payment method | `runs/task_48ae3c959915/replay-5269dac6-ab05-48aa-a857-38c9c107143b.jsonl` |
| `replay-ad96d823-7edc-4d21-a815-25aac88cd134` | replay | the model's | env_refused_the_candidate | business_error: ValueError: Order is not delivered | `runs/task_d8954d758335/replay-ad96d823-7edc-4d21-a815-25aac88cd134.jsonl` |
| `replay-010a23ef-3524-4dbf-ad95-81d4cf974a45` | replay | unattributed | judge_failed_it | task_dbf0c8930714: violates c_9f52317fb6df, c_ea9943e77c85 | `runs/task_dbf0c8930714/replay-010a23ef-3524-4dbf-ad95-81d4cf974a45.jsonl` |
| `replay-075cc208-d1a5-4739-9f12-866246a18c34` | replay | unattributed | judge_failed_it | task_8f8ef5e89736: judge: A made no database update, so it did not return the laptop. B canceled a pending order instead of processing the requested laptop retu | `runs/task_8f8ef5e89736/replay-075cc208-d1a5-4739-9f12-866246a18c34.jsonl` |
| `replay-0a044a76-b5c8-4e87-943b-3fa3bbfed547` | replay | unattributed | judge_failed_it | task_37d1178f5430: judge: A and B only record returns and do not accomplish the requested tracking-number lookup or shipping-address and color changes; B also r | `runs/task_37d1178f5430/replay-0a044a76-b5c8-4e87-943b-3fa3bbfed547.jsonl` |
| `reroll-task_a5e0012de413-2` | reroll | unattributed | unattributed | max_turns | `runs/task_a5e0012de413/reroll-task_a5e0012de413-2.jsonl` |
| `reroll-task_d251e0d84e94-0` | reroll | unattributed | unattributed | max_turns | `runs/task_d251e0d84e94/reroll-task_d251e0d84e94-0.jsonl` |
| `reroll-task_ddc432876575-0` | reroll | unattributed | unattributed | max_turns | `runs/task_ddc432876575/reroll-task_ddc432876575-0.jsonl` |

321 more failing Runs not listed; pass `--limit 0` to print every row.
