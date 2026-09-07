# D79 suite failures, decomposed to root cause (.work-b14 retail, .work-airline)

Method: both workdirs are read only and both builds were still moving, so `task_status.json` was
snapshotted first and every ruling was reproduced from the `examiner/cache/<task>/<key>.json` entry
whose `status.checks` matches the snapshot row (that entry carries the Verifier, the References and
the confirmation record together). The suite was then re-run over those bytes with the current code
in `wt-telecom` (`verifier_suite.validate_verifier`'s own gate functions), with `CanonRules` built
from `canon-rules.json` and write tools from `tool_sigs.json` (`kind == "write"`), which reproduces
every recorded check exactly. Counts are Tasks, at the snapshot.

Snapshot totals: b14 205 Tasks, 140 References confirmed, 95 trusted, 45 failing the suite.
airline 119 Tasks, 38 References, 2 trusted, 36 failing. (The prompt's numbers, 135/99/36 and 39/2,
are the same builds a few minutes earlier; the shape of the failures is unchanged.)

## Ranked summary, by Tasks the cause costs

| # | Root cause | b14 | airline | Fix sits in | Reachable by a repair verb today |
|---|---|---|---|---|---|
| 1 | The Reference makes no write, so the Verifier is a write cap of 0 plus Hard atoms plus communicate atoms. Nothing in it can be mutated and little of it can be failed | 27 | 29 | (b) suite, then (a) derivation | No for `mutation_flips`; partly for the others (`repair` can add atoms) |
| 2 | The Task has one Reference, so `verifier_alt_path` is not run and the check is scored as a fail | 20 | 10 | (c) Reference, and (b) the check conflates "not run" with "too narrow" | Yes: the Examiner's `reroll` then `derive` |
| 3 | `false_rejection` counts as legitimate the very Runs the D111 rule discarded that round | 47 | 8 | (b) the gate's pool | No |
| 4 | A system_derived constant reaches the Simulated user's rules or the Intent | 4 | 5 | (a) derivation and (d) the user-rules miner | Partly: `repair_intent` for the Intent half |
| 5 | Genuine over-strictness on a held-out anchor Run: communicate, question and list-valued write_value atoms | 8 | 1 | (a) derivation | Yes: the Examiner's `repair` drop, under the loosening gate |

## mutation_flips

Every failing Task in both builds has the same shape: `req_write == 0` and `entity_count == 0` on
the Reference. Stuck atoms per Task: the one `entity_count` atom and every mutable `hard` atom
(34 of 39 in b14, 34 to 59 in airline; the judge atoms carry no code and are not mutated).

| Cause | b14 | airline | Where |
|---|---|---|---|
| `entity_count` with `count` 0: `_mutant` computes `max(count - 1, 0)`, so the mutant is the same atom and cannot flip | 27 | 28 | (b) `verifier_suite._mutant` |
| `hard` atoms: `_HARD_WRAPPER` skips every call whose name is in `read_tools` and not in `write_tools`, so a Run with no write call judges nothing and `check()` returns True. Replacing the rule with `_NEVER_HOLDS` still returns True | 27 | 28 | (b) `verifier_suite._HARD_WRAPPER`, and the mutation gate that reads its answer |
| Not reproduced at the snapshot (build moved under the read) | 0 | 1 | n/a |

The communicate atoms of these same Verifiers do flip, so the Verifier is not wholly inert; the
gate fails on the two atom kinds that cannot be made to say anything false. Note the second-order
consequence: on a read-only Task no compiled Hard constraint is evaluated at all, so the policy is
unchecked there whatever the mutation gate says.

Fix: (b). Either `_mutant` returns None for a cap of 0 (nothing to mutate is not the same as an
atom that cannot be failed) or, better, the wrapper reports "judged no call" the way `hard_holds`
already refuses to read None as "the constraint held". The Examiner's `repair` cannot reach either:
it only drops and adds atoms, and dropping the cap plus the Hard atoms leaves a Verifier of
communicate atoms alone, which is a loosening the loosening gate has to rule on.

## second_path_passes

| Cause | b14 | airline | Where |
|---|---|---|---|
| `verifier_alt_path` not run: `suite_for` passes `alt_path_run=paths[1] if len(paths) > 1 else None`, and the Task has one Reference | 20 | 10 | (c) Reference selection |
| The second path exists and fails on an atom | 0 | 0 | n/a |

Not one Task in either build has a real second path that the Verifier rejects. Every one is a
missing input, confirmed by `not_run` carrying `verifier_alt_path` on all 20 and all 10 rows.
Why the Task has one Reference: in b14, 17 of 20 had 2 to 5 candidate Runs and the D111 rule
discarded all but one (the judge ruled on 16 of the 20); 3 Tasks had a single candidate Run at all.
In airline, 8 of 10 had candidates discarded, 2 had a single Run.

Fix: (c) plus (b). The Examiner's `reroll` verb can roll more Runs of the Task and `derive` again,
which is the acting move available today. Separately the suite should distinguish "not run" from
"too narrow" in the trusted ruling: a check with no input is not evidence of a narrow Verifier, and
today it is counted the same way.

## unsolved_state_fails

| Cause | b14 | airline | Where |
|---|---|---|---|
| Read-only Reference, communicate atoms present: `unfinished_run` cuts before the last tool call, and the facts were already stated in an earlier assistant turn, so the cut Run still passes | 7 | 14 | (b) `verifier_suite.unfinished_run` |
| Read-only Reference, no communicate atom: the Verifier is the cap plus Hard atoms, which the cut Run satisfies trivially | 3 | 6 | (a) derivation, (b) suite |
| Not reproduced at the snapshot | 0 | 1 | n/a |

All 10 and all 21 have `req_write == 0`. The cut rule prefers a tool call over an assistant turn,
which is right for a Task whose outcome is a write and wrong for a Task whose outcome is the answer:
for a Verifier with no required write the cut should fall before the final answer turn.

Fix: (b) in `unfinished_run`. The Examiner's `repair` cannot move the cut; adding a communicate atom
for a fact stated only in the closing turn would fix a subset, and that is (a) work the derivation
should do rather than the model.

## empty_fails, plausible_wrong_fails, loophole_probe_fails

The same 3 Tasks in b14 and the same 6 in airline fail all three, and no other Task fails any of them.

| Cause | b14 | airline | Where |
|---|---|---|---|
| The Verifier is `entity_count` 0 plus Hard atoms and nothing else: the empty Run, the muted wrong Run and the probe all satisfy it | 3 | 6 | (a) derivation |

Why no communicate atom was derived, over the read-only Tasks that fail any check: in b14, 2 Tasks
have no Reference that states a fact read from the world at all (prose answers, or a transfer), and
1 Task has References that each state facts with an empty intersection; the other 24 keep 1 to 17
facts. In airline, 4 state nothing and 3 have an empty intersection; the other 22 keep 1 to 14.
So two distinct derivation causes: `communicate_values` only recognises id-shaped or numeric tokens
that also appear in an earlier tool result, and D43's intersection across every Reference collapses
to nothing as soon as one Reference phrases its answer without those tokens.

Fix: (a). D156 already says a no-write recording has an outcome; the derivation needs an atom that
carries it (what the answer asserted, not only which id-shaped tokens it repeated). The Examiner's
`repair` can add atoms and so could patch a Task at a time, which is the model doing the
derivation's job.

## leak_check_clean

| Cause | b14 | airline | Where |
|---|---|---|---|
| A `write_value` atom, provenance system_derived, on the `order_id` field, whose value appears in the Simulated user's rules | 3 | 0 | (d) the user-rules miner |
| A `communicate` atom, provenance system_derived, whose token appears in the Simulated user's rules | 1 | 5 | (d) the user-rules miner |
| A `communicate` atom, provenance system_derived, whose token appears in the Intent | 2 | 0 | (a)/(d), the Intent miner |

Failures, not Tasks: b14 has 6 leaks over 4 Tasks, airline 5 over 5 Tasks. Every leaked constant is
system_derived, so the value came out of a tool result and no user of any seed recording said it,
which is exactly what D157 tightened. What still reaches the Simulated user is an id or a number the
recording's user never spoke.

Fix: (d) with (a). `repair_intent` is the Builder verb for the Intent half (2 leaks in b14); there is
no repair verb over `user_rules`, so the 4 and 5 leaks into the Simulated user's rules are not
actionable today and would need the user-fact miner to refuse a value no user turn carries.

## false_rejection == 1.0

Recomputed with `loosening.false_rejection` over every Run under `runs/<task>/`, with the legitimate
pool built the way `legitimate_runs` builds it (confirmed replays plus finished re-rolls).
b14: 50 Tasks at 1.0, 112 rejected held-out Runs. airline: 9 Tasks, 22 rejected Runs.
42 of the 50 b14 Tasks have a trusted Verifier, so this number is the one that survives the suite.

| What the rejected held-out Run actually was | b14 Runs | airline Runs | Correct rejection |
|---|---|---|---|
| A recording the same round's D111 rule discarded, judge reason (it did not do the job) | 98 | 17 | Yes |
| A recording the D111 rule discarded for violating a compiled constraint | 3 | 4 | Yes |
| The held-out anchor Run (D81), a genuine frontier path | 10 | 1 | Mostly no, see below |
| A re-roll not in the confirmation record | 1 | 0 | Unknown |

By Task: 39 of the 50 b14 Tasks reject only Runs their own confirmation record lists under
`failed`, another 8 reject a mix of those and the anchor, and 3 reject only Runs outside the record.
In airline 8 of the 9 reject only discarded Runs. This is the gate's pool, not the Verifier:
`legitimate_runs` counts a replay as legitimate when it is `confirmed`, which means the Environment
reproduced it, not that it did the job. The Reference rule discarded those same Runs minutes earlier.

The 10 anchor Runs in b14, which are the real held-out frontier, break down as:

| Rejecting atom | Runs | Was the Run correct |
|---|---|---|
| `communicate`: the anchor made the same writes but never stated one token the References all stated | 6 | Yes, wrongly rejected |
| `write_value` on `item_ids`: the required canonical key is a longer list than the anchor's | 1 | Arguably: a different write, not a different path |
| `question` on `field:order_id`: the anchor asked about `reason` and took a confirm instead, the user having volunteered the id | 1 | Yes, wrongly rejected |
| `write`: the anchor made no write, or one write where the References made two | 2 | No, correctly rejected |

The single airline anchor case is a `communicate` atom on a Run that terminated as a transfer and
stated nothing, so it is correctly rejected.

Fix: (b) first, the pool. Subtracting the confirmation's `failed` Runs from `legitimate_runs` would
drop b14 from 50 Tasks at 1.0 to about 4 and airline from 9 to 1, and would make the number mean
what D133 wants it to mean. Then (a): of the genuinely held-out Runs, the communicate atom is the
over-strict one (6 of 10 in b14), followed by the question atom and exact-list `write_value`
matching. The Examiner's `repair` can drop those atoms today and the loosening gate will allow it,
since the Run it newly passes is in the legitimate pool.

## Two notes worth carrying out of this

1. The whole failing population of both builds is two shapes. Read-only Tasks (27 of 45 failing in
   b14, 29 of 36 in airline) fail because a Verifier with no write has nothing the suite can falsify;
   single-Reference Tasks (20 and 10) fail because a check had no input. Nothing in either build
   failed because a Verifier was wrong about a Run that was right, except the 8 anchor cases above.
2. `_HARD_WRAPPER` returning True when it judged no call means a compiled Hard constraint is silently
   unchecked on every read-only Task. The mutation gate is the only thing that notices, and it
   reports it as a Verifier defect rather than as a constraint that never ran.
