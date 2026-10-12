# Deferred work

What the old builds taught is in learnings.md. The papers and repos the founder sent are in papers.md. The
list was cleared on 2026-09-22 for the overhaul and rebuilt on 2026-09-24 after the build published as build-20260924.

## Next

- Deep search over the failures of both build-20260924 builds: Examiner findings, refused tool calls, repeated failing
  calls, turns per repair and stop reasons. Find where the model struggles and propose tools, skills and prompts
  that raise the Verifier and trusted counts. Workdirs:
  /Users/krishuagarwal/.herdr/worktrees/kullback/overhaul-0922-smoke9/.work-retail and .work-airline.
- Synthetic Task generation, on the D224 to D226 path: grounded scenarios, augmented seeds and traps, then
  generated Tasks hardened the way TauForge augments, validates and hardens. Every generated Task needs concise
  instructions and a high quality Verifier (founder, 2026-09-23).
- Two patches still wait in docs/frozen-patches now that the freeze is open (D314): confinement-holes (five of
  its holes are still open, but its module-shape rule refuses today's generated skeleton, so it needs a rebase
  of its own) and speed-3 (the spend ledger batching, written against a budget.py that has since moved).
- Re-roll batches. One Builder run call plays at most RUNS_PER_CALL = 20 fresh Runs, which is why builds re-roll
  10 to 20 Tasks at a time. Measure whether a larger cap or parallel Runs helps.
- Cheaper runner model experiment. Record the Run kind in the ledger first, since second-path Runs dominate the
  runner's spend.
- The airline body's hand-rolled id rule should go back to the context's new_id in the next build.
- Independent check on the rebuilt world: the replay only confirms recorded calls; nothing checks the world
  against evidence outside the recording (review 2026-10-06).

## From the failure research of build-20260924 (2026-09-25)

- Route checks. A Verifier may demand an ordering ("identity looked up before the first read of that user's data"),
  derived from what every seed did, as a Hard atom over `called_before`. Deferred by the founder on 2026-09-25: the
  Verifier scores results only for now. Evidence: 6 Tasks where the loophole probe skipped the lookup every seed made
  (report group G3, retail 1c8acd, 559243, 67c669, d81010, a53732, airline 3d9770).
- Ending action feedback. When every re-roll fails only on the ending action and ends in the same End state as the
  Reference, loosen the ending action from required to allowed. Evidence: airline 3b00d4, where 9 of 9 correct
  declines failed on the transfer the recording made (report, "The ending action").
- Hand-off summary text. 32 retail re-rolls differ from the Reference only in a free summary field. Measure what
  fails them before choosing a fix.

## Before the first training run (founder, 2026-09-25)

- Compare every built Task with the benchmark's test tasks, one by one: which test task each Task came from, and
  how close each synthetic Task sits to a test task (same request, same rows changed). The held-out anchor
  (builder/world_tools.py draw_anchor) holds out recordings inside a Task, not Tasks, so today every benchmark
  task the traces cover is built and would be trained on. Decide from the comparison how to split by task
  before ingest, and write the held-out task ids into the package manifest. Do this when training is about to start.

## Priority order after the overhaul (founder, 2026-09-23)

Item 1, every trace passes, is met by build-20260924 (replay fidelity 1.00 on retail and airline). Synthetic Tasks moved
to Next. What remains:

1. Kitaru (ZenML), https://www.zenml.io/product/kitaru: build evals on top of the built Environment. From the page,
   Kitaru is a replay-based evaluation tool that records, replays and evaluates agents against frozen production
   sessions before a change ships.
2. The difficulty knob ("TaskPilot and FrogNano: what a difficulty knob would look like in Kullback"). difficulty.py
   measures the buckets; the second half is a selection rule that shifts Task difficulty as training proceeds.

## Sandbox

A generated tool body runs in a subprocess with its imports blocked, which reduces the blast radius but is not a
security boundary (builder/sandbox.py, gates/confinement.py, gates/artifacts.py). A real sandbox for model-written
code is still open.

## Builder memory

Deferred by the founder on 2026-09-23: "memory is totally different now, we will talk about it later." The reading
of omp, pi and tau and the candidate decisions M1 to M7 are in .claude/reports/memory-management-2026-09-23.md (repo
checkout, untracked). Nothing of it is built. Revisit when the founder describes the new memory design.

## Feedback mining from traces (founder, 2026-09-23, production direction)

The recorded traces and the Environments built from them already hold what customers ask for (Intents), what they
know and hand over (typed facts), what they ask (question atoms), where policy stops them (refusals), where they
give up (terminations) and what they want that the tools cannot do (off-path requests). Report it in aggregate
only, with personal data stripped, never as a view of one customer. Not built before synthetic Tasks.

## Environment comparison

The Verifier scores the End state and what the user was told, never the route, so a right output reached by a
wrong method passes by design. An Environment that scores the route gives a different refusal rate, so a
comparison across Environments is fair only once every Environment scores the same thing. Forbidden routes are
Hard atoms written per Task.

## Route checking (founder, 2026-09-25)

Deferred: "how the tools should run is the route check, maybe don't do that, add it in todo". Today a Verifier scores
the End state and what the user was told. Two route checks are parked here, for recorded and synthetic Tasks alike:
the tools that should run for a Task, and a no-detour check that no write outside the Task's fix touched a row the
Task names. The second catches a detour that undoes itself (change a field, change it back); a detour that leaves a
row behind (place an order, then cancel it) is already caught by the End state. Evidence that the class is large:
PAE (arXiv 2603.03116) finds 27 to 78 percent of reported tau-bench successes broke a required procedure.
