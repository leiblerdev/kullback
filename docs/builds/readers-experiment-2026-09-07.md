# Readers for prose results, declared or free (D176 candidate)

Two builds of one customer corpus, one per arm of the open question: whether the model's reader
proposal must also declare, per column, stored or derived, and per tool the shapes that assert
nothing, with the gate holding those declarations to the recording (`--readers-gate declared`,
arm A), or whether the proposal is free and replay fidelity is the only check
(`--readers-gate replay`, arm B).

Both arms ran from one worktree on copies of the same workdir, taken before either ran, of a
corpus of 456 recordings, 183 frozen Tasks and 36 tools, 27 of them the simulated user's own.
Command per arm, no `--iterate`:

    uv run kullback build -w <copy> --target replay_reference --model openai/gpt-5.6-luna \
        --workers 8 --ceiling-usd 10 --max-rounds 1 --readers-gate <declared|replay>

Both builds finished (exit `max_rounds`, no failed stage). Nothing was re-frozen: no file under
`kullback/runner/` changed, so the frozen-runner check never came up.

What the requestor's own tools left to read: 27 tools, 5,287 prose calls, 181 distinct result
strings, 175 masked shapes. That 175 is the denominator of every gate count below.

## 1. The proposal, and what the gate caught

| | arm A, `declared` | arm B, `replay` |
|---|---|---|
| attempts to pass | 2 of 4 | 1 of 4 |
| kept assisted | no | no |
| table | one, one row per Task, row id the requestor's name | same |
| columns proposed | 28, all declared `stored` | 29 |
| readers | 27, one per tool | 27 |
| acknowledgement shapes declared | 7, over 5 tools | not asked for |
| write-only tools declaring a change | 0 | not asked for |
| derived columns declared | 0 | not asked for |
| shapes the gate failed, attempt 0 | 16 of 175 | 0 of 175 |
| shapes the gate failed, attempt 1 | 0 of 175 | n/a |

Every one of arm A's 16 failures was the same rule: the reader read nothing out of the shape and
the shape was not declared an acknowledgement. They fell over 5 tools: an APN settings check (3
shapes), an app permission check (4), a permission grant (4), a payment (1), a speed test (4).
The second attempt read a column out of all 16 of them: the speed test's rate and quality band,
the granted permission, the APN name and the MMSC setting, the permission list. Its 7 declared
acknowledgements are other shapes, the ones that genuinely assert nothing ("App 'X' not found on
this phone.", "No payment request has been made.", "Speed test failed: No Connection."). So the
rule's effect was to turn 16 silent readers into readers that read, and none of the 16 into a
declaration.

The two other declared rules never fired. Rule 2 (a derived column equals its derivation over the
stored columns walked to that point) had nothing to check, because the model declared all 28
columns stored, the status bar and the MMS capability included, which the grill expected to be
derived. Rule 3 (a tool that only acknowledges names a column it is seen to change) had nothing to
check either, because no tool of this corpus answers only acknowledgements: every write here
reports the new state in the same string. So on this corpus the declared arm's whole effect was
rule 1, one extra attempt, and 7 shapes marked as asserting nothing.

Arm B passed the first reply: over 181 distinct results, every reader parsed, ran and answered a
dict or None. The `replay` gate never rejected anything, so the free arm's cost is one model call.

## 2. Tasks at replay fidelity

Both arms: 0 Tasks confirmed, as before the stage existed. The mechanism moved calls, not Tasks.

| | base | arm A | arm B |
|---|---|---|---|
| Tasks confirmed | 0 | 0 | 0 |
| frozen denominator | 183 | 183 | 183 |
| Tasks after clustering | 183 | 365 | 393 |
| recordings replayed | 456 | 456 | 456 |
| writes matched | 658 of 2,577 (25.5%) | 1,540 (59.8%) | 1,630 (63.3%) |
| reads scored in substance | 2,696 of 5,155 | 2,902 | 2,756 |

A Task confirms only when a whole recording replays, and the median recording here makes ten or
more device calls, so one wrong column anywhere in it costs the Task. Writes matched is the number
that moved: it more than doubled in both arms.

The Task count is the other thing that moved. A revealed row splits worlds the way a contradicted
row does (D74), so two recordings that read one device column differently before either wrote are
now different Tasks: 183 becomes 365 under arm A and 393 under arm B. The frozen denominator stays
183, so coverage is still measured against the same 183, but every per-Task record in these two
workdirs is over a finer split. That split is a real consequence of the mechanism, not an artifact
of the arm, and it is the thing to decide about before D176 is recorded.

## 3. Per tool, recorded calls matched

Read from `replays.json`: a call counts as matched when the Runner ruled `same`, `cosmetic` or
`both_refused`. Call counts are the recording's own and are the same in all three columns.

| tool | side | calls | base | arm A | arm B |
|---|---|---|---|---|---|
| get_details_by_id | assistant | 1153 | 0.0% | 0.0% | 0.0% |
| get_customer_by_phone | assistant | 459 | 100.0% | 100.0% | 100.0% |
| enable_roaming | assistant | 164 | 100.0% | 100.0% | 100.0% |
| refuel_data | assistant | 146 | 100.0% | 100.0% | 100.0% |
| transfer_to_human_agents | assistant | 124 | 100.0% | 100.0% | 100.0% |
| get_data_usage | assistant | 121 | 91.7% | 100.0% | 0.0% |
| get_bills_for_customer | assistant | 79 | 0.0% | 0.0% | 0.0% |
| send_payment_request | assistant | 67 | 100.0% | 100.0% | 100.0% |
| resume_line | assistant | 53 | 0.0% | 0.0% | 0.0% |
| can_send_mms | user | 714 | 1.5% | 81.7% | 81.7% |
| check_network_status | user | 462 | 10.0% | 37.7% | 41.8% |
| run_speed_test | user | 424 | 46.2% | 43.2% | 40.1% |
| check_status_bar | user | 403 | 1.2% | 46.4% | 74.4% |
| reboot_device | user | 337 | 44.2% | 29.4% | 17.5% |
| check_apn_settings | user | 319 | 68.3% | 76.2% | 76.5% |
| toggle_roaming | user | 248 | 20.6% | 8.5% | 5.2% |
| toggle_airplane_mode | user | 245 | 47.8% | 47.8% | 41.2% |
| set_network_mode_preference | user | 223 | 73.5% | 0.4% | 73.5% |
| check_app_permissions | user | 198 | 39.9% | 12.6% | 1.0% |
| toggle_data | user | 197 | 11.7% | 18.3% | 3.6% |
| grant_app_permission | user | 188 | 95.7% | 0.5% | 95.7% |
| check_wifi_calling_status | user | 163 | 39.9% | 100.0% | 100.0% |
| reset_apn_settings | user | 154 | 85.1% | 85.1% | 85.1% |
| reseat_sim_card | user | 147 | 0.0% | 29.9% | 26.5% |
| check_sim_status | user | 135 | 34.8% | 34.8% | 49.6% |
| check_network_mode_preference | user | 126 | 66.7% | 98.4% | 99.2% |
| check_installed_apps | user | 111 | 100.0% | 100.0% | 96.4% |
| toggle_wifi_calling | user | 98 | 100.0% | 100.0% | 100.0% |
| check_payment_request | user | 91 | 36.3% | 72.5% | 76.9% |
| toggle_data_saver_mode | user | 75 | 45.3% | 16.0% | 14.7% |
| check_data_restriction_status | user | 64 | 87.5% | 98.4% | 100.0% |
| disconnect_vpn | user | 63 | 57.1% | 22.2% | 15.9% |
| make_payment | user | 62 | 1.6% | 90.3% | 91.9% |
| check_vpn_status | user | 57 | 93.0% | 100.0% | 100.0% |
| check_wifi_status | user | 21 | 100.0% | 100.0% | 100.0% |
| check_app_status | user | 8 | 50.0% | 25.0% | 0.0% |

Totals over the requestor's own 27 tools: base 2,013 of 5,333 calls (37.7%), arm A 2,679 (50.2%),
arm B 3,036 (56.9%). Over the assistant's 9: base 1,071 of 2,366 (45.3%), arm A 1,081 (45.7%),
arm B 960 (40.6%).

The three tools the brief named as the earlier build's worst all move: the MMS capability check
from 1.5% to 81.7% in both arms, the status bar from 1.2% to 46.4% (A) and 74.4% (B), the SIM
reseat from 0% to 29.9% (A) and 26.5% (B). The two it named as the best are where arm A loses
them: the permission grant falls from 95.7% to 0.5% and the network mode write from 73.5% to 0.4%,
in arm A only.

Both losses are body defects downstream of the column set, not gate failures. Arm A proposed the
status bar's battery as one stored column and no column for the bar itself, and its permission
grant body then answered "Status Bar: 80" where the recording answers the assembled bar; arm B
proposed the same battery column but its body assembles the bar from the other columns and matches
on 95.7%. Arm A's network mode body raised `TypeError: can only concatenate str (not "int") to
str` on 222 of 223 calls. Neither is something either gate looks at: both arms declare or free,
the reader was correct and the body was not.

The other regressions (a roaming toggle, a data saver toggle, a VPN disconnect, a device reboot,
a data toggle in arm B) share one cause: the tool's answer restates the whole status bar, so
a body that has the row still has to reproduce every column in it, and one wrong column costs the
call. Both arms carry it, and arm A is ahead of arm B on all five; only one of the five (a data
toggle, 11.7% to 18.3%) is ahead of the base build as well.

One assistant-side regression is noise rather than arm: the data usage lookup, an assistant tool
whose body compile_tools rewrote in both builds, answers floats and nulls in arm B where the
recording answers strings (0.0%), and matches on 100% in arm A. It reads the customer's own tables,
not the revealed row.

## 4. The revealed row, and who reads it

Both arms named one table and gave it one row per Task, keyed by the requestor. Arm A filled 14.4
of its 28 columns per recording before the first write, arm B 13.5 of 29; the rest of the row is
left to the Environment's own defaults, which is why a signal or network type read only later in a
recording is often wrong at the start.

Bodies reading that table: 28 of 36 tools in arm A, 27 in arm B. All 27 of the requestor's own
tools read it in arm A, 26 in arm B. One assistant-side tool reads it in both: the roaming enable
that the assistant makes and the requestor's own roaming check then reads. That is the cross-side
link the one-world decision buys, and it holds in both arms.

`compile_tools`' own replay of the recorded calls through each body (`tool_fidelity.json`, D80)
agrees with the reference replay: 3,613 of 7,196 calls in arm A, 3,830 of 7,277 in arm B; 24 of 36
bodies kept assisted in arm A, 25 in arm B.

## 5. Cost and wall time

Both builds ran at once on one machine, so the wall times are of two builds sharing it.

| | arm A | arm B |
|---|---|---|
| wall time | 8m 23s | 9m 15s |
| spend, whole build | $0.735 | $0.701 |
| of which the readers stage | $0.021, 2 attempts | $0.015, 1 attempt |
| of which compile_tools | $0.307, 321 calls | $0.275, 290 calls |
| of which compile_policy | $0.407, 416 calls | $0.411, 418 calls |

Spend is this build's, over the copied workdir's own totals; the model memo carried the policy and
the intent work from the copy where the prompts had not moved. Neither arm came near the $10
ceiling. The readers stage is the cheapest model stage in the build by an order of magnitude, and
the declared arm's extra attempt costs about half a cent.

## 6. What this says about the open question

Read straight, the numbers favour neither arm as strongly as the design difference suggests, and
two of them cut against the recommendation the grill went in with:

1. The declared gate's distinctive rules did nothing here. All 28 columns came back stored, and no
   tool of this corpus only acknowledges, so rules 2 and 3 never ran. What did run, rule 1, is the
   one rule the free gate could also carry: it is a check on the reader, not on a declaration.
2. The free arm ended ahead on the requestor's own calls, 56.9% against 50.2%, and its lead is two
   tools where arm A's body could not assemble the status bar. That is a body-writing difference,
   downstream of a column set the gate had no opinion about, so it is not evidence that the free
   gate is better either. Repeated with another seed it could as easily fall the other way.
3. Neither arm confirmed a Task, so the headline number the experiment was to move is unmoved. The
   mechanism's real gain is in writes matched, 25.5% to 59.8% and 63.3%.
4. Both arms fragment the Task set, 183 into 365 and 393. Any decision to keep the mechanism has
   to say what happens to that: either the revealed row stays out of `trace_worlds`, or a Task is
   allowed to hold recordings that started in different device states, or the split is accepted and
   the frozen denominator carries the comparison.

The cheapest reading is that the declarations are not what pays here, and that the next thing to
fix is one level down: the status bar, which 12 tools restate and every arm gets wrong at the
start, and the starting values for the columns no recording reads before its first write.

## 7. How to reproduce

    cp -R .work-telecom .work-readers-a && cp -R .work-telecom .work-readers-b
    uv run kullback build -w .work-readers-a --target replay_reference \
        --model openai/gpt-5.6-luna --workers 8 --ceiling-usd 10 --max-rounds 1 \
        --readers-gate declared
    uv run kullback build -w .work-readers-b ... --readers-gate replay

Every number above is read from files under the two workdirs: `readers.json` (the proposal, its
attempts and the failures per attempt), `replays.json` (per tool, per call verdicts and Task
confirmation), `tool_fidelity.json` (compile_tools' own replay), `tasks.json` and
`tasks_frozen.json` (the Task counts), `budget.json` (spend, against the copied workdir's totals),
`bodies.json` (which bodies read the revealed table).
