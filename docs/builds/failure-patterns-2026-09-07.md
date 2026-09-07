# What every model did wrong, read off five builds (2026-09-07)

Founder: "watch the output of every model sir, builder, examiner, judges and note for the failure patterns." Four read-only reviews of the records of `.work-b14` (retail-domain corpus, 205 Tasks, 3 rounds), `.work-airline` and `.work-airline-keys` (119 Tasks), `.work-telecom` and `.work-readers-b` (183 Tasks, 371 after the readers stage split them). Counts only, digits masked where a value could identify a customer. Each pattern names the fix that is now on a branch (`lookup-homing`, `loop-fixes`, `derive-fixes`) or the decision that already closed it.

## Ranked by the Tasks they cost

| # | Model | Pattern | Evidence | Fix |
|---|---|---|---|---|
| 1 | Verifier derivation | A communicate atom requires any token the recorded answer stated that also sits in a tool result, whether or not the Intent asked for it; a held-out Run that solves the Task without repeating it is rejected | about 70 percent of exported assertions on the first two corpora carry such a number; false rejection mean 0.74 over 98 Tasks, 63 Tasks at 1.0 (b14 round 3); 0.85 on airline | derive-fixes A: required only when the Intent or a user turn asked for the fact |
| 2 | Reference judge | Fails a recording the corpus itself rewarded, because it sees End states only and a Task resolved by an answer looks abandoned; then derive caps the Verifier at 0 writes | b14: 15 of 59 joinable judge-failed recordings, 10 Tasks; airline 18 of 22, 14 Tasks; 55 of 200 b14 Verifiers open with "at most 0 write calls" while other recordings wrote; the judge abstained 0 times in 126 judgements | derive-fixes B: the judge sees what each Run told the user; no 0-cap when another confirmed recording wrote |
| 3 | Body writer (assistant lookup) | A row with several id columns and a tool noun matching no table is never homed, so every lookup raises KeyError | telecom: 1,057 of 1,057 calls, all 183 Tasks | lookup-homing A: a row is homed by the id the call asked for |
| 4 | Readers stage | Every revealed column is class hard, so a value that never repeats (a speed test) splits Tasks | 183 Tasks became 371 | lookup-homing B: revealed columns take the D73 classes |
| 5 | Body writer (prose results) | The status-bar line appended by the device tools is derived from other columns; bodies render "none" where the recording says a state word | airplane toggle 0 of 241, data toggle 1 of 192, reboot 10 of 329 calls | open: a derived rendering per tool, gated by round trip over the recordings (reopens D176's declared arm) |
| 6 | Verifier derivation | A literal entity id as the first required atom rejects a second legitimate path | 138 of 200 b14 Verifiers; second_path_passes fails 14 Tasks on b14, 41 of 62 on airline-keys | derive-fixes D: ten cases measured before any rule changes |
| 7 | Examiner | One bare read of the replays (8.95M characters) put the context at 13 times the window and the round ended with no finding | telecom round 1; b14 at 9 times the window in 2 of 3 rounds | D175 (read clamp and index), D178 (floor cut, real window, search) |
| 8 | Examiner | A ruling name passed as the finding's kind is rejected and the finding is lost | b14: 17 of 28 finding calls, 13 about real replay mismatches | loop-fixes 1: ruling names map to kinds |
| 9 | Examiner | The same Task repaired four times against the same failing check, no throttle | airline: 13 of 13 repairs rejected | loop-fixes 2: two strikes, then refuse or re-roll |
| 10 | Reference grouping | Two recordings that wrote the same items in another list order land in two End states and the Task loses its Reference | 12 of b14's 25 disagreements | derive-fixes C |
| 11 | Builder | A bodies-only round regresses Tasks that were trusted; nothing re-checks them | b14 trusted 95 to 66, fidelity 188 to 147 in one round; airline 8 to 5 | D174 (a recompile replaces only by beating); loop-fixes 8: the round delta in every headline |
| 12 | Body writer | Recompile hints chase one recorded call at a time | one retail tool recompiled 6 times in a round | loop-fixes 6: hints accumulate, every failing shape shown |
| 13 | Body writer | A kept assisted body that ignores its arguments (one constant refusal for every call) is not told apart from one that misses an edge | telecom: 26 of 36 tools assisted, all without one passing attempt | loop-fixes 7: hardcoded bodies marked and refused |
| 14 | Gate text | A raised body's failure text is the bare exception, no table or column named | replay_fidelity is the largest gate bucket on every build (18 to 83) | loop-fixes 5: the feedback names the table, the id's absence, the arguments |
| 15 | Body writer | A catalog listing answers only the categories the world's rows hold | retail: 19 of 19 calls differ, 15 categories only on the recorded side | open: a no-argument tool whose results never change is a world constant |
| 16 | Body writer | A write's payment-history entry has the wrong polarity (refund for payment) | 52 of 76 differing retail checks | body feedback (loop-fixes 5, 6); one regression fixture with both directions |
| 17 | Policy compiler | A constraint with code but no test rate against confirmed recordings is indistinguishable from one that passed | 27 to 40 percent of constraints on every build | derive-fixes E |
| 18 | Intent grounding | Short idiomatic phrases fail literal grounding | telecom 16 of 183, elsewhere under 2 percent | derive-fixes F: vocabulary before grounding on every corpus |
| 19 | Examiner | probe, refuse and reroll were never called on any build; two of five prompt examples are about them | 0 calls in three sessions | loop-fixes 9: examples rewritten to what the records need |

## Controls that held

No Intent on any build names a tool function. Bodies never reached the host past the sandbox. The reference judge never accepted a recording that violated a compiled constraint (hard-constraint failures 6, 7, 1 against judge failures 215, 99, 3). The readers stage passed on its first attempt with no silent shape; its readers were not the blocker, the bodies that ignored what they revealed were.

## What could not be told from the records

Per-Run rejected ids behind the false-rejection fraction (only the Task-level fraction is persisted; derive-fixes D reconstructs ten). Whether the judge's abstention path (D93) is alive, since no build produced an abstention. Who filed the findings on the two builds whose Examiner never ran a turn (they are D170's findings by rule; the session file does not say so). Whether the trusted drop on b14 was the Examiner's repairs or the Builder's recompiles.
