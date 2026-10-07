# The writer's disagreement with the Reference, against the sidecar class (D327)

Measured on 2026-10-07, code only, 0 USD, on the vf-live copies of retail and airline (read only). Script:
`.claude/herdr/fix-1007/fx-trust-measure.py`, which tiers every Task with `spec.trust.tier_of_task` exactly as
`intent_tiers` does and reads `writer_disagrees` and `fresh_runs` off each row. Not pre-registered: the flag does
not gate, so there was no rule to hold it to; the numbers are reported as they came out.

## Cannot show

- One build per corpus, one Spec writer model. The writer's disagreement is a property of that model's reading.
- The sidecar class is the reward of the kept recordings; 22 retail and 36 airline Tasks have none and are left
  out. Mixed (5 retail, 1 airline) is counted with right, as the analysis did.
- Airline has 4 trusted Tasks and none on a wrong Reference: its trusted cells say nothing.
- No Task was replayed or re-run here; a Task with fresh Runs on disk is as the vf-live build left it.

## Not run apart from failed

| corpus | replay_only | not run (no fresh Run on disk) | fresh Runs all failed |
|---|---|---|---|
| retail | 52 | 40 | 12 |
| airline | 4 | 3 | 1 |

The analysis counted 43 retail not run; this script finds the Runs by file name under runs/ as the analysis did, and
gets 40. The trust row now prints "replay_only N (not run M)".

## The flag against the class

The flag (`writer_disagrees` set) is the demand half: a required write demand whose tool the Reference never
called, or whose row is in no diffed row. A row half was built and measured beside it (a diffed row outside the
action tables that no write demand names) and dropped at the fold, since it fired on right References only.
Cells are Tasks: flag set or not by Reference wrong or not.

| corpus, Tasks | flag, wrong | flag, right | no flag, wrong | no flag, right | precision | recall |
|---|---|---|---|---|---|---|
| retail, trusted | 1 | 0 | 5 | 38 | 1.00 | 0.17 |
| retail, all with a class | 22 | 5 | 27 | 147 | 0.82 | 0.45 |
| airline, trusted | 0 | 0 | 0 | 4 | n/a | n/a |
| airline, all with a class | 3 | 3 | 12 | 76 | 0.50 | 0.20 |

Which half fires (Tasks): retail, a demanded tool never called 22 wrong and 3 right, a demanded row not diffed 0
wrong and 2 right, a diffed row no demand names 0 wrong and 19 right; airline 3 and 2, 0 and 1, 4 and 17.

## What it says

- The demand half reproduces the analysis (18 wrong against 1 right there, 22 against 3 here, by the diff rather
  than by call arguments): where the writer demands a write the Reference never called, the Reference is usually
  wrong. On airline the same half is a coin toss on 6 Tasks.
- The dropped row half was noise: it fired on right References (world-made rows such as a balance moved by a
  refund, rows a write changes beside the one it names), costing precision and finding no wrong Reference on retail.
- The 6 wrong-trusted retail Tasks: 1 carries the flag. On the other 5 the blind writer demanded what the Reference
  did (a call-argument match finds the same 1 of 6), so the writer does not see what is wrong with them. The flag
  does not close the gap that trust by agreement leaves.

## What gating on it would cost

Gating on the flag would have withheld 1 wrong and 0 right of the 44 trusted retail Tasks (13.6 percent wrong,
6 of 44, becomes 5 of 43, 11.6 percent) and none of the 4 trusted airline Tasks. Over all classed Tasks it would hold
back 5 right retail and 3 right airline References with the 25 wrong ones. That does not reach the 10 percent rule;
the flag stays counted and never gates.
