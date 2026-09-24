# Deferred work

What the old builds taught is in learnings.md. The papers and repos the founder sent are in papers.md. The
list was cleared on 2026-09-22 for the overhaul and rebuilt on 2026-09-24 after smoke 9.

## Next

- Deep search over the smoke 9 failures in both builds: Examiner findings, refused tool calls, repeated failing
  calls, turns per repair and stop reasons. Find where the model struggles and propose tools, skills and prompts
  that raise the Verifier and trusted counts. Workdirs:
  /Users/krishuagarwal/.herdr/worktrees/kullback/overhaul-0922-smoke9/.work-retail and .work-airline.
- Synthetic Task generation, on the D224 to D226 path: grounded scenarios, augmented seeds and traps, then
  generated Tasks hardened the way TauForge augments, validates and hardens. Every generated Task needs concise
  instructions and a high quality Verifier (founder, 2026-09-23).
- Re-freeze the runner under the new hash before the next build, because the recorded tool-context feed changed
  kullback/runner. Verdicts from before and after the re-freeze are not comparable (D61). The patches still waiting
  in docs/frozen-patches (safe-write, speed-1, speed-3, confinement-holes and the others tests name) land in the
  same re-freeze, including the speed-1 variant gate that examiner/stage.py keeps serial until then.
- The two commits on overhaul-0922/pr2 after PR #121 merged (1d9203b, de52f22) are not on main yet.
- Re-roll batches. One Builder run call plays at most RUNS_PER_CALL = 20 fresh Runs, which is why builds re-roll
  10 to 20 Tasks at a time. Measure whether a larger cap or parallel Runs helps.
- Cheaper runner model experiment. Record the Run kind in the ledger first, since second-path Runs dominate the
  runner's spend.
- The airline body's hand-rolled id rule should go back to the context's new_id in the next build.

## Priority order after the overhaul (founder, 2026-09-23)

Item 1, every trace passes, is met by smoke 9 (replay fidelity 1.00 on retail and airline). Synthetic Tasks moved
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
