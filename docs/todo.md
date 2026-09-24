# Deferred work

Cleared on 2026-09-22. The harness is being rebuilt around one event bus, a Builder that reads and writes its Environment through tools, and dense per-call feedback; every item of the old list is superseded by that rebuild or has to be re-earned against it. What the old builds taught is in learnings.md. The papers and repos the founder sent are in papers.md.

## Builder memory

- Deferred by the founder on 2026-09-23: "memory is totally different now, we will talk about it later."
  The reading of omp, pi and tau and the candidate decisions M1 to M7 are in
  .claude/reports/memory-management-2026-09-23.md (repo checkout, untracked). Nothing of it is built.
  Revisit when the founder describes the new memory design.

## Priority order after the overhaul (founder, 2026-09-23)

1. Every trace passes first. Founder: "all the traces pass (a high fidelity environment for retail and
   airline...) that makes it quite ready". Replay fidelity 1.0 and every Reference confirmed on retail
   and airline, and the loop must notice when the Simulated user side is what fails, not only the
   Environment (F34, F39).
2. Kitaru (ZenML), https://www.zenml.io/product/kitaru: build evals on top of the built Environment.
   From the page, Kitaru is a replay-based evaluation tool: record, replay and evaluate agents against
   frozen production sessions before shipping a change.
3. The difficulty knob ("TaskPilot and FrogNano: what a difficulty knob would look like in Kullback")
   and synthetic Task generation. Founder: concise instructions and high quality Verifiers are required
   for every generated Task.
4. Augment seeds: new scenarios, and traps hardened for the specific Environment, either written by
   hand or extracted from the traces.
5. Generate Tasks from scenarios grounded in the Environment. D224 to D226 already name the mechanism;
   this is the "scenario generation and seed augmentation" line in harness-design.md, not new work.
6. Generate and harden the Tasks: the TauForge stages augment, validate, harden, the last steps of the
   four this loop already runs under other names.

## Feedback mining from traces (founder, 2026-09-23, production direction)

- The recorded traces and the Environments built from them already hold what customers ask for (Intents), what
  they know and hand over (typed facts), what they ask (question atoms), where policy stops them (refusals), where
  they give up (terminations) and what they want that the tools cannot do (off-path requests).
- Reported in aggregate only, with personal data stripped; never a view of one customer.
- Not built before fidelity, the quality pass and synthetic Tasks.

## Environment comparison

Right output by a wrong method is scored as a pass by design: the Verifier scores the End state and what the user was told, never the route. An Environment that scores the route gives a different refusal rate, so a comparison across Environments is fair only once every Environment scores the same thing. Forbidden routes are Hard atoms written per Task.
