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

## 8. Follow-up: one gate, mined write effects, corpus fills

After the two arms above, the declared arm and the `--readers-gate` flag were removed: the code
holds one path, the free one, and arm A's rule 1 survives as a reported metric rather than a
refusal, so the stage ruling now says how many shapes of each tool parse to nothing
(`silent_shapes`, `silent_by_tool`) without failing anything.

Two faults from sections 2 and 6 were fixed:

1. The requestor's own writes now close columns. A write tool's changed columns are mined the way
   `mine.observed_effects` credits assistant writes: the columns whose parsed value differs between
   the last reading before the call and the first reading after it, the call's own result counting
   as a reading after it, in at least one recording. Those columns close at that tool's calls in
   every walk, so a requestor write ends a column the way a first reading does. The miner's
   `kind` in `tool_sigs.json` says which tools are writes.
2. Columns no recording reads before its first write are filled from the corpus: the commonest
   pre-write value of that column across recordings. Each fill is recorded as an assumption of the
   Starting state, the way `compile_env.build_starting_state` records its own, and the count is in
   the stage ruling. A column no recording reads before any write anywhere stays unset and is
   named in the ruling.

Re-run on `.work-readers-b` with the same command, model and ceiling as before
(`--target replay_reference --max-rounds 1`).

### Tasks at replay fidelity

| | base | arm B before | arm B after |
|---|---|---|---|
| Tasks confirmed | 0 | 0 | 0 |
| frozen denominator | 183 | 183 | 183 |
| Tasks after clustering | 183 | 393 | 221 |
| recordings replayed | 456 | 456 | 456 |
| writes matched | 658 of 2,577 (25.5%) | 1,630 (63.3%) | 1,301 (50.5%) |
| reads scored in substance | 2,696 of 5,155 | 2,756 | 2,842 |

### Per tool, recorded calls matched

Same source and rule as section 3: a call counts as matched when the Runner ruled `same`,
`cosmetic` or `both_refused`, and the call counts are the recording's own. `kind` is the miner's
label, which the new closing rule reads.

| tool | side | kind | calls | base | before | after |
|---|---|---|---|---|---|---|
| get_details_by_id | assistant | read | 1153 | 0.0% | 0.0% | 0.0% |
| get_customer_by_phone | assistant | read | 459 | 100.0% | 100.0% | 100.0% |
| enable_roaming | assistant | write | 164 | 100.0% | 100.0% | 100.0% |
| refuel_data | assistant | write | 146 | 100.0% | 100.0% | 100.0% |
| transfer_to_human_agents | assistant | generic | 124 | 100.0% | 100.0% | 100.0% |
| get_data_usage | assistant | read | 121 | 91.7% | 0.0% | 100.0% |
| get_bills_for_customer | assistant | read | 79 | 0.0% | 0.0% | 100.0% |
| send_payment_request | assistant | write | 67 | 100.0% | 100.0% | 100.0% |
| resume_line | assistant | read | 53 | 0.0% | 0.0% | 0.0% |
| can_send_mms | user | write | 714 | 1.5% | 81.7% | 79.8% |
| check_network_status | user | write | 462 | 10.0% | 41.8% | 2.6% |
| run_speed_test | user | write | 424 | 46.2% | 40.1% | 44.6% |
| check_status_bar | user | write | 403 | 1.2% | 74.4% | 34.0% |
| reboot_device | user | read | 337 | 44.2% | 17.5% | 15.4% |
| check_apn_settings | user | read | 319 | 68.3% | 76.5% | 72.7% |
| toggle_roaming | user | read | 248 | 20.6% | 5.2% | 0.0% |
| toggle_airplane_mode | user | read | 245 | 47.8% | 41.2% | 44.9% |
| set_network_mode_preference | user | read | 223 | 73.5% | 73.5% | 73.5% |
| check_app_permissions | user | read | 198 | 39.9% | 1.0% | 37.9% |
| toggle_data | user | write | 197 | 11.7% | 3.6% | 8.1% |
| grant_app_permission | user | read | 188 | 95.7% | 95.7% | 95.7% |
| check_wifi_calling_status | user | read | 163 | 39.9% | 100.0% | 39.9% |
| reset_apn_settings | user | read | 154 | 85.1% | 85.1% | 0.6% |
| reseat_sim_card | user | read | 147 | 0.0% | 26.5% | 24.5% |
| check_sim_status | user | read | 135 | 34.8% | 49.6% | 51.9% |
| check_network_mode_preference | user | read | 126 | 66.7% | 99.2% | 57.1% |
| check_installed_apps | user | read | 111 | 100.0% | 96.4% | 100.0% |
| toggle_wifi_calling | user | read | 98 | 100.0% | 100.0% | 100.0% |
| check_payment_request | user | read | 91 | 36.3% | 76.9% | 69.2% |
| toggle_data_saver_mode | user | read | 75 | 45.3% | 14.7% | 26.7% |
| check_data_restriction_status | user | read | 64 | 87.5% | 100.0% | 87.5% |
| disconnect_vpn | user | read | 63 | 57.1% | 15.9% | 7.9% |
| make_payment | user | read | 62 | 1.6% | 91.9% | 91.9% |
| check_vpn_status | user | read | 57 | 93.0% | 100.0% | 8.8% |
| check_wifi_status | user | read | 21 | 100.0% | 100.0% | 100.0% |
| check_app_status | user | read | 8 | 50.0% | 0.0% | 50.0% |

Totals over the requestor's own 27 tools: base 2,013 of 5,333 (37.7%), before 3,036 (56.9%), after
2,421 (45.4%). Over the assistant's 9: base 1,071 of 2,366 (45.3%), before 960 (40.6%), after 1,160
(49.0%). The 33 calls of malformed or invented tool names match in all three.

The three tools the brief named as the earlier build's worst keep most of their gain but lose some
of it: the MMS capability check 1.5% to 81.7% to 79.8%, the status bar 1.2% to 74.4% to 34.0%, the
SIM reseat 0% to 26.5% to 24.5%. The two it named as the best are unmoved at 95.7% and 73.5%, as
they were in the free arm before. The two assistant-side lookups that gained (a data usage lookup
0% to 100%, a bills lookup 0% to 100%) read the customer's own tables, not the revealed row: their
bodies were rewritten in this run and the move is body-writing, not the closing rule.

### The split, before and after

Measured over the frozen 183-Task grouping, by walking each recording and asking which revealed
columns two recordings of one frozen Task disagree on at the point the overlay is taken:

| | before | after |
|---|---|---|
| frozen Tasks whose recordings disagree on a revealed column | 89 of 183 | 27 of 183 |
| distinct columns that split a Task | 21 | 8 |
| revealed columns read per recording, before its first write | 13.5 | 4.6 |
| Tasks after clustering, in the build | 393 | 221 |

Before, the columns that split the most Tasks were the cellular network type (61 Tasks), the
cellular signal (55), the mobile data switch (47), airplane mode (42), the cellular connection
(36), the network mode preference (35), the SIM card status (31), the MMSC setting and the roaming
switch (27 each), the app permission list (24), Wi-Fi calling (21), the speed quality and the
speed result (20 each), the granted permission and the app name (16 each). After, eight columns
remain: the roaming switch (21 Tasks), airplane mode (12), a payment request (3), the network mode
preference and the mobile data switch (2 each), the data saver switch, the VPN status and a
payment made (1 each).

Of the first readings the old rule took in the disagreeing Tasks, 1,925 of 2,201 came after a call
of a tool now credited with changing that column. That is the fault named in the follow-up: the
requestor's own writes never closed a column, so a reading taken after the requestor had already
changed the device was recorded as its starting value.

### Assumptions and unset columns

The proposal named 29 columns. 24 of them are read by no recording before that recording's first
write and are filled from the corpus, each fill written into the Starting state's assumptions, for
example: the table's row for the requestor, column `cellular_signal`, not read before a write in
407 of 456 recordings, starting on the value 20 of the 49 recordings that did read it before
writing showed. (The wording of that sentence was corrected after this run; it changes no number
here.) Five columns are read
before a write in no recording at all and stay unset: `cellular_connection`, `messaging_mms`,
`wifi_calling`, `speed_test_result`, `speed_quality`. Both counts are in the stage ruling
(`filled_columns`, `unset_columns`), and the unset names are listed there.

The reported silent metric found nothing on this corpus: `silent_shapes` 0, `silent_by_tool` empty.
The proposal passed on the first attempt, not assisted. That is the same reply as the free arm
before the change, served from the model memo, so the readers stage cost nothing in this run.

### Cost and wall time

| | arm B before | arm B after |
|---|---|---|
| wall time | 9m 15s | 5m 02s |
| spend, whole build | $0.701 | $0.281 |
| of which the readers stage | $0.015, 1 attempt | $0.000, memo hit |
| of which compile_tools | $0.275, 290 calls | $0.281, 421 calls |
| of which compile_policy | $0.411, 418 calls | $0.000, memo hit |

`compile_tools`' own replay (`tool_fidelity.json`): 3,377 of 6,515 calls, against 3,830 of 7,277
before; 25 of 36 bodies kept assisted, the same as before.

### Where this disagrees with the instruction

The closing rule does what it was asked to do, and the split halves, but on this corpus it costs
call fidelity on the requestor's side, 56.9% down to 45.4%, and the cause is the miner's `kind`,
not the rule.

Five of the requestor's tools carry `kind == "write"` here. One of them is a real toggle. The other
four are device *checks* that only report state, and several real toggles carry `kind == "read"`
(see the `kind` column above). Each credited tool is mined as changing 12 to 16 columns, because a
check's result string restates the whole device state and any column that drifted between two of
its calls is credited to it. A check therefore closes almost the whole row at its first call, which
is why revealed columns read per recording fell from 13.5 to 4.6, why 24 of 29 columns had to be
filled from corpus modes, and why the tools whose starting value now comes from a mode rather than
from the recording lose fidelity: the roaming toggle to 0%, the APN reset from 85.1% to 0.6%, the
VPN check from 100% to 8.8%, the network status check from 41.8% to 2.6%.

A narrower variant was probed before settling: credit a column only when the write is the single
write call in the interval between the two readings. It does not help (4.6 columns per recording
becomes 5.0), because the mislabelled checks are the only calls in most intervals.

So the honest reading is that the fault named in the follow-up is real and the fix is the right
shape, but it lands on a miner label that is wrong for this corpus, and the two changes together
trade one measured number (the split, 393 to 221) against another (requestor-side matched calls,
3,036 to 2,421). Whichever way D176 is recorded, the `kind` classifier is the next thing to look
at: as long as a read-only check can be labelled a write, any rule that lets writes close columns
will close too many.

## 9. Follow-up: a lookup's row is homed by the id the call asked for

Section 3's worst line was the assistant-side lookup at 0.0% of 1,153 recorded calls, and section
8 left it there. The cause was in the miner, not in the readers stage: a row that comes back with
several `_id` columns and a tool name that names no entity got no home, so `extract_rows` dropped
it, the world held none of those rows, and every recorded call of the tool replayed as a
`KeyError` on the id it was given.

Three rules were added.

1. **Homing by the id the call asked for** (`mine._home_of`). A row a call answered is homed in
   the entity whose id column holds one of the values the call passed. Where several of the row's
   ids were passed, the column whose name equals the argument's name wins, else the first in the
   row's own order. Two narrowings keep the rule from taking a table nothing names: an id the tool
   name says is only the address is ignored (`_address_of`, the entity after the first preposition
   in the name), so a call that lists a parent's children still homes the children; and a column
   whose name carries no id suffix is ignored, so a search filter that `id_columns` happens to call
   an id cannot name a table after itself. The noun rule, the distinct-across-siblings rule and the
   only-id rule stay as fallbacks, unchanged and in that order. Composite keys keep working: homing
   picks the table, keying picks the row.
2. **Revealed columns are classed the way mined ones are, and only hard columns split a Task.**
   Revealed columns already ran through `propose_column_class`; what did not hold was the other
   half, that `compile_env.trace_worlds` and `readers.merge_worlds` hashed every column of a row
   rather than its `hard` ones. Both now consult the classes (`merge_worlds` with no schema still
   counts every column, which is what a caller with no classes to hand can say). One class rule was
   added for any corpus: a numeric column whose sightings never repeat a value (distinct equals
   sampled, sampled at least `MIN_UNIQUE_VALUES`, 5) is a reading the world takes when it is asked,
   not a fact it stores, so it is exempt at low confidence and the setup review sees it.
3. **An early `--target` stops without the Examiner.** `examiner.stage.missing_inputs` names the
   derivation inputs a target did not build (`tasks`, `sigs`, `constraints`); when any is missing
   the Examiner beat returns before opening a session and the round exits `target_built` with a
   note saying so, instead of `derive failed: KeyError: 'constraints'`. A ceiling still reports as
   `ceiling`, because that is the reason that has to be reported (D86).

Where each tool's rows were homed and by which rule is written to `row_homes.json` by the mine
stage, per tool: the tables, the rule's own sentence, the rows homed by it, and the rows left
unhomed.

Re-run on a fresh copy of the base workdir with the same command, model and ceiling as before
(`--target replay_reference --max-rounds 1 --workers 8 --ceiling-usd 10`). The before column is the
previous workdir as it stands on disk, which is a later run than the table in section 8.

### Tasks at replay fidelity

| | before | after |
|---|---|---|
| Tasks confirmed | 0 | 8 |
| frozen denominator | 183 | 183 |
| Tasks after clustering | 371 | 373 |
| recordings replayed | 456 | 456 |
| writes matched | 1,121 of 2,466 (45.5%) | 1,307 of 2,264 (57.7%) |
| reads scored in substance | 2,396 of 5,266 | 1,777 of 5,468 |

The headline number the experiment set out to move is moved: the corpus goes from no Task at
replay fidelity to 8 of 373, against a frozen denominator of 183.

### Per tool, recorded calls matched

Same source and rule as sections 3 and 8: `replays.json`, a call counts as matched when the Runner
ruled `same`, `cosmetic` or `both_refused`, and the call counts are the recording's own.

| tool | side | calls | before | after |
|---|---|---|---|---|
| get_details_by_id | assistant | 1153 | 0.0% | 79.3% |
| get_customer_by_phone | assistant | 459 | 100.0% | 100.0% |
| enable_roaming | assistant | 164 | 100.0% | 100.0% |
| refuel_data | assistant | 146 | 100.0% | 100.0% |
| transfer_to_human_agents | assistant | 124 | 100.0% | 100.0% |
| get_data_usage | assistant | 121 | 100.0% | 0.0% |
| get_bills_for_customer | assistant | 79 | 0.0% | 0.0% |
| send_payment_request | assistant | 67 | 100.0% | 100.0% |
| resume_line | assistant | 53 | 0.0% | 0.0% |
| can_send_mms | user | 714 | 81.7% | 81.7% |
| check_network_status | user | 462 | 43.5% | 56.9% |
| run_speed_test | user | 424 | 52.1% | 46.0% |
| check_status_bar | user | 403 | 34.0% | 76.7% |
| reboot_device | user | 337 | 7.1% | 39.2% |
| check_apn_settings | user | 319 | 96.6% | 58.3% |
| toggle_roaming | user | 248 | 11.7% | 23.4% |
| toggle_airplane_mode | user | 245 | 0.0% | 51.0% |
| set_network_mode_preference | user | 223 | 73.5% | 7.2% |
| check_app_permissions | user | 198 | 92.4% | 92.4% |
| toggle_data | user | 197 | 1.0% | 38.6% |
| grant_app_permission | user | 188 | 0.5% | 0.5% |
| check_wifi_calling_status | user | 163 | 98.8% | 98.8% |
| reset_apn_settings | user | 154 | 85.1% | 30.5% |
| reseat_sim_card | user | 147 | 30.6% | 25.2% |
| check_sim_status | user | 135 | 74.8% | 74.8% |
| check_network_mode_preference | user | 126 | 98.4% | 100.0% |
| check_installed_apps | user | 111 | 100.0% | 100.0% |
| toggle_wifi_calling | user | 98 | 100.0% | 100.0% |
| check_payment_request | user | 91 | 33.0% | 74.7% |
| toggle_data_saver_mode | user | 75 | 0.0% | 45.3% |
| check_data_restriction_status | user | 64 | 96.9% | 98.4% |
| disconnect_vpn | user | 63 | 39.7% | 60.3% |
| make_payment | user | 62 | 91.9% | 0.0% |
| check_vpn_status | user | 57 | 100.0% | 100.0% |
| check_wifi_status | user | 21 | 100.0% | 100.0% |
| check_app_status | user | 8 | 12.5% | 25.0% |

Totals: assistant side 1,081 of 2,366 (45.7%) to 1,874 (79.2%); the requestor's own 27 tools 2,877
of 5,333 (53.9%) to 3,091 (58.0%); over everything 3,991 of 7,732 (51.6%) to 4,998 (64.6%). The 33
calls of malformed or invented tool names match in both.

`compile_tools`' own replay (`tool_fidelity.json`) agrees on the lookup: 881 of 1,055 evidence
calls, against 0 of 1,057 before. 23 of 36 bodies stayed assisted, the same count as before.

Three tools move the other way, and none of the three is homing. Two assistant-side lookups that
read the customer's own tables went 100% to 0% and a payment tool 91.9% to 0%: their bodies were
rewritten in this run (`tool_fidelity.json` has them at 0 of 49 and 0 of 57 evidence calls, so the
body is wrong before any world question), and two requestor-side tools that lost ground, the
network mode preference at 73.5% to 7.2% and the APN reset at 85.1% to 30.5%, are the same
body-writing variance section 8 saw in the other direction. That is the noise floor of one round
of body writing on this corpus, and it is large.

### What the lookup still misses

Of the 239 calls of the lookup that still differ, all 239 are the Starting state not holding the id
the call asks for: the world holds 4 rows of one table, 2 of another and 1 of a third, and the
recordings ask for ids outside that set, so the tool answers `not_found_entity` where the recording
answered a row. That is D40 and `--grow` territory, not homing. The two assistant-side tools still
at 0% are the same shape: one is skipped whole by D74's after-write rule (53 of 53 recorded calls,
`tool_builds.json` `after_write_skipped`), so it gets no evidence calls and no per-call fidelity
rows at all, and the other reads a table the world holds one row of.

### Schema and the split

| | before | after |
|---|---|---|
| mined columns of the table the lookup homes into | 5 | 14 |
| rows of that table in the Starting state | 1 | 2 |
| revealed columns, hard of the proposal's 29 | 29 | 28 |
| frozen Tasks whose recordings disagree on a hard column of a mined row | 0 of 183 | 0 of 183 |
| frozen Tasks whose recordings disagree on a revealed hard column | 85 of 183 | 86 of 183 |
| distinct revealed columns that split a Task | 19 | 17 |

The Task count does not fall to 183, and the reason is worth stating plainly, because the brief
expected it to. Hashing only hard columns changes nothing on the mined rows: no frozen Task's
recordings disagree on a hard column of a mined row, before or after. Every split on this corpus
comes from the revealed row, and 17 of its 28 hard columns still split a Task, led by the cellular
signal (51 Tasks), airplane mode (48), the mobile data switch (47), the cellular network type (40),
the network mode preference (35), the cellular connection (31) and the SIM card status (29). Those
are genuine disagreements: two recordings of one frozen Task really did start with the device in
different states, and one world with a per-Task overlay can pin one of them (D74). The new class
rule removes exactly the columns it was written to remove, a measured reading that never repeats a
value, which was splitting 20 Tasks before and splits none now; the rest of the movement between 19
and 17 columns is the readers proposal naming its columns slightly differently in the two runs.

So the split is not a classification bug that a rule can close. Either the revealed row stays out
of `trace_worlds`, or a Task is allowed to hold recordings that started in different device states,
or the frozen denominator carries the comparison, which is what section 6 already said and what
this run does not change.

### Cost and wall time

| | before | after |
|---|---|---|
| wall time | not recorded | 8m 53s |
| spend, this build over the copy | $1.226 | $0.702 |
| of which compile_tools | $0.801 | $0.276 |
| of which compile_policy | $0.411 | $0.407 |
| of which the readers stage | $0.015, 1 attempt | $0.015, 1 attempt |

Neither run came near the $10 ceiling. The round exited `max_rounds` with 10 findings still open,
the largest of them the three assisted tools above.

### How to reproduce

    cp -R .work-telecom .work-readers-c
    uv run kullback build -w .work-readers-c --target replay_reference \
        --model openai/gpt-5.6-luna --workers 8 --ceiling-usd 10 --max-rounds 1

Numbers above are read from `row_homes.json` (the homing decisions), `replays.json` (per tool, per
call verdicts and Task confirmation), `tool_fidelity.json` (compile_tools' own replay), `gates.json`
(the stage rulings), `schema.json` and `db.json` (the tables), `readers.json` (the proposal) and
`budget.json` (spend, against the copied workdir's totals).

## 10. Follow-up: the render, and the round trip as the gate

Failure pattern 5: the requestor's device tools end their results with a status line composed out of
several other columns, the readers parse that line into columns, and the compiled bodies write it
back with a word the row never held. Section 9 left five of those tools between 0% and 77%.

Decided with the founder (2026-09-07 evening) and built here: the readers proposal gains, per tool,
a **render**, `def render(row) -> str`, the inverse of the reader, and the gate holds the two
together by round trip. For every recorded result of a tool, `render` of the row that result was
answered from, walked exactly the way `starting_row` walks it and completed with the fills, has to
be that result again character for character. Failures come back one line per masked shape, the
same feedback shape readers already had, for up to 4 attempts; the proposal with the fewest failing
shapes is kept and marked assisted, as before. Renders run in the same subprocess sandbox as the
readers, in one module under one confinement gate. A tool whose result is an acknowledgement renders
a constant; a word the sentence states that no column holds is computed inside the render out of the
columns that do, and gets no column of its own. The body skill gained one paragraph saying the
render is the answer (a read tool answers `render(row)`, a write tool applies its effect and then
answers `render(row)`), and the body writer sees the render source in its prompt for those tools.

Re-run on a fresh copy of the third corpus's workdir with the same command, model and ceiling as
section 9 (`--target replay_reference --max-rounds 1 --workers 8 --ceiling-usd 10`). The before
column is the section 9 workdir as it stands on disk.

### The round trip, per attempt

| | attempt 0 | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|---|
| renders proposed, of 27 tools | 27 | 15 | 27 | 27 |
| columns proposed | 29 | 29 | 29 | 30 |
| round trips asked | 2,670 | 0 | 2,986 | 3,303 |
| round trips that held | 1,192 | 0 | 1,780 | 3,189 |
| masked shapes failing, of 175 | 135 | 175 | 96 | 14 |
| shapes the readers read nothing out of | 51 | 0 | 27 | 0 |

Attempt 1 is the one interesting failure of the loop: the model answered with renders for 15 of the
27 tools, so 12 tools were reported missing a render and the round trip never ran. The feedback
named each of the 12 and attempt 2 proposed all 27 again. Attempts 0, 2 and 3 are a clean descent:
44.6%, 59.6%, 96.5% of the recorded results written back out of the row they came from. The kept
proposal is attempt 3's, marked assisted because 14 masked shapes still fail after the fourth
attempt, all of them on two tools whose results begin with a line the row does not determine.

### Tasks at replay fidelity

| | before | after |
|---|---|---|
| Tasks confirmed | 3 | 3 |
| frozen denominator | 183 | 183 |
| Tasks after clustering | 373 | 365 |
| recordings replayed | 456 | 456 |
| writes matched | 1,307 of 2,264 (57.7%) | 1,482 of 2,466 (60.1%) |
| reads matched | 3,691 of 5,468 | 4,103 of 5,266 |

### Per tool, recorded calls matched

Same source and rule as sections 3, 8 and 9: `replays.json`, a call counts as matched when the
Runner ruled `same`, `cosmetic` or `both_refused`, and the call counts are the recording's own.
`render` says whether the kept proposal carried a render for that tool.

| tool | side | render | calls | before | after |
|---|---|---|---|---|---|
| get_details_by_id | assistant | no | 1153 | 79.3% | 94.5% |
| can_send_mms | user | yes | 714 | 81.7% | 81.7% |
| check_network_status | user | yes | 462 | 56.9% | 47.4% |
| get_customer_by_phone | assistant | no | 459 | 100.0% | 100.0% |
| run_speed_test | user | yes | 424 | 46.0% | 62.5% |
| check_status_bar | user | yes | 403 | 76.7% | 84.1% |
| reboot_device | user | yes | 337 | 39.2% | 31.5% |
| check_apn_settings | user | yes | 319 | 58.3% | 71.2% |
| toggle_roaming | user | yes | 248 | 23.4% | 40.7% |
| toggle_airplane_mode | user | yes | 245 | 51.0% | 52.7% |
| set_network_mode_preference | user | yes | 223 | 7.2% | 74.0% |
| check_app_permissions | user | yes | 198 | 92.4% | 82.3% |
| toggle_data | user | yes | 197 | 38.6% | 20.3% |
| grant_app_permission | user | yes | 188 | 0.5% | 78.2% |
| enable_roaming | assistant | no | 164 | 100.0% | 100.0% |
| check_wifi_calling_status | user | yes | 163 | 98.8% | 98.8% |
| reset_apn_settings | user | yes | 154 | 30.5% | 85.1% |
| reseat_sim_card | user | yes | 147 | 25.2% | 31.3% |
| refuel_data | assistant | no | 146 | 100.0% | 100.0% |
| check_sim_status | user | yes | 135 | 74.8% | 87.4% |
| check_network_mode_preference | user | yes | 126 | 100.0% | 98.4% |
| transfer_to_human_agents | assistant | no | 124 | 100.0% | 100.0% |
| get_data_usage | assistant | no | 121 | 0.0% | 0.0% |
| check_installed_apps | user | yes | 111 | 100.0% | 100.0% |
| toggle_wifi_calling | user | yes | 98 | 100.0% | 100.0% |
| check_payment_request | user | yes | 91 | 74.7% | 36.3% |
| get_bills_for_customer | assistant | no | 79 | 0.0% | 0.0% |
| toggle_data_saver_mode | user | yes | 75 | 45.3% | 45.3% |
| send_payment_request | assistant | no | 67 | 100.0% | 100.0% |
| check_data_restriction_status | user | yes | 64 | 98.4% | 98.4% |
| disconnect_vpn | user | yes | 63 | 60.3% | 73.0% |
| make_payment | user | yes | 62 | 0.0% | 1.6% |
| check_vpn_status | user | yes | 57 | 100.0% | 50.9% |
| resume_line | assistant | no | 53 | 0.0% | 0.0% |
| check_wifi_status | user | yes | 21 | 100.0% | 100.0% |
| check_app_status | user | yes | 8 | 25.0% | 25.0% |

Totals: the requestor's own 27 tools, all of which carry a render, 3,091 of 5,333 (58.0%) to 3,502
(65.7%); the assistant's 9, none of which carry one, 1,874 of 2,366 (79.2%) to 2,050 (86.6%); over
everything 4,998 of 7,732 (64.6%) to 5,585 (72.2%). The 33 calls of malformed or invented tool names
match in both.

`compile_tools`' own replay (`tool_fidelity.json`): 4,909 of 6,993 evidence calls, against 4,617 of
6,984 before. 25 of 36 bodies kept assisted, against 23 before.

### The tools pattern 5 named, and the ones that moved most

| tool | calls | before | after |
|---|---|---|---|
| a permission grant | 188 | 0.5% | 78.2% |
| a network mode preference setter | 223 | 7.2% | 74.0% |
| an APN reset | 154 | 30.5% | 85.1% |
| a roaming toggle | 248 | 23.4% | 40.7% |
| a speed test | 424 | 46.0% | 62.5% |
| an APN check | 319 | 58.3% | 71.2% |
| a VPN disconnect | 63 | 60.3% | 73.0% |
| a SIM status check | 135 | 74.8% | 87.4% |
| a status bar check | 403 | 76.7% | 84.1% |
| a SIM reseat | 147 | 25.2% | 31.3% |
| an airplane toggle | 245 | 51.0% | 52.7% |
| a reboot | 337 | 39.2% | 31.5% |
| a network status check | 462 | 56.9% | 47.4% |
| an app permission check | 198 | 92.4% | 82.3% |
| a data toggle | 197 | 38.6% | 20.3% |
| a payment request check | 91 | 74.7% | 36.3% |
| a VPN status check | 57 | 100.0% | 50.9% |

Over all 27 of the requestor's tools, 10 move up by more than five points, 6 move down by more than
five, and 11 stay within five. The five the pattern itself named split two up and three down.

All 14 masked shapes the render never satisfied after four attempts belong to one tool, the reboot.
Its results open with a line the row does not determine and end with the composed line, so the
render writes the second half and not the first, and it is one of the six that fall. The other five
that fall carry no failing shape at all, so their loss is body writing rather than the render.

### Cost and wall time

| | before (section 9) | after |
|---|---|---|
| wall time | 8m 53s | 15m 05s |
| spend, this build over the copy | $0.702 | $0.789 |
| of which the readers stage | $0.015, 1 attempt | $0.047, 4 attempts |
| of which compile_tools | $0.276 | $0.350, 526 calls |
| of which compile_policy | $0.407 | $0.387, 404 calls |

The round exited `max_rounds`, no stage failed, and neither run came near the $10 ceiling. The whole
cost of the render is the readers stage's three extra attempts, $0.032 and about two minutes.

### Where this disagrees with the instruction

The brief expected the render to move the five tools it named. It moves two of them up and three of
them down, and the tools it moves furthest are four the brief did not name, one of which went from
1 matched call in 188 to 147. The requestor's side gains 411 calls net and the corpus stays at 3
Tasks confirmed, because the Tasks that hold those tools are split by the revealed row the same way
section 9 measured, and one round of body writing still moves single tools by tens of calls in both
directions. So the render does what it was built to do (96.5% of recorded results written back out
of the row, up from 44.6% on the first attempt) and it buys real fidelity, but it does not on its
own turn a Task at replay fidelity, and it is not the only thing that moved between these two runs.

### The nested-argument fill (part 3 of the same brief)

`compile_env.referenced_ids` now walks lists and dicts at any depth for a value under a table's
first key field, reading the remaining key parts from the object the value sat in rather than from
the top level. Measured over the three corpora on disk, against the old top-level-only rule:

| corpus | referenced ids, old | new | new only | of those, not in the world |
|---|---|---|---|---|
| first (retail) | 255 | 255 | 0 | 0 |
| second (nested list writes, D179) | 173 | 220 | 47 | 17, all one table |
| third (device corpus) | 4 | 4 | 0 | 0 |

So the rule fires exactly where D179 said it would and nowhere else, and it changed no number in
either of the two builds above: both ran with 0 synthetic rows before and after. On the first
corpus 9 referenced ids are still not in the world, but their table has no observed row at all, and
D40 leaves such a table empty rather than inventing a shape for it. That is the rule as written,
not something this change touched.

The third corpus's lookup goes 914 of 1,153 (79.3%) to 1,090 (94.5%) in this run, and it is not
this fill: the 239 calls section 9 left were `not_found` on ids the world does hold under other
tables, and after this run 63 calls differ on one missing column of a row the body did find. The
body was rewritten this round. On the first corpus the four lookups are unmoved (1,200 of 1,204,
423 of 424, 414 of 414, 24 of 24).

### How to reproduce

    cp -R .work-readers-c .work-readers-d
    uv run kullback build -w .work-readers-d --target replay_reference \
        --model openai/gpt-5.6-luna --workers 8 --ceiling-usd 10 --max-rounds 1

Numbers are read from `readers.json` (the attempts, the renders, the round trips), `replays.json`
(per call verdicts and Task confirmation), `tool_fidelity.json` (compile_tools' own replay),
`schema.json` and `db.json` (the tables) and `budget.json` (spend, against the copied workdir's
totals).
