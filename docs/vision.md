# Vision and the source of truth

Written 2026-10-16 from the founder's words of 2026-10-15 and 2026-10-16 (D365). This page says what the product is,
what it takes in, what it treats as true, and which parts exist today. Where it and docs/design-philosophy.md disagree
on the source of truth, this page holds.

## The product

You tell an agent the behaviour you want from your own agent. It builds the evals and the training data for that
behaviour: a realistic Environment, Tasks inside it, and a Verifier for each Task. A human reviews a sample of the
evals before any number goes out. Every model you care about is then benchmarked on the same evals, and the model you
pick is trained on the same set, in the same place, without the data leaving the loop. In the founder's words: "an
agent which asks you the right behaviour you want and then we generate the evals or the same and then the human sees
some of the evals and then we benchmark all the models across it and then train the model then and there".

## Inputs

Traces are one input, not the whole. Beside them sit the database structure, the business knowledge graph (how the
customer's entities, rules and processes relate) and the policy document the agent works under. From these we build
the most realistic Environment we can: the rows of the world, the tools with their effects, and simulated users who
know what a real user of that business would know. That part is what Kullback does today from traces alone, at
measured replay fidelity.

Then we augment the Environment heavily with synthetic Tasks. They are generated from the real world (what happens in
this kind of business, found by search against the domain, never by occurrence in our recordings) and bound to the
Environment's real rows, so a synthetic Task asks for real work over real data. This step is designed and unbuilt. The
plan for breadth is docs/scaling.md (axis 1, D224), the plan for rows the traces never show is docs/synthetic-rows.md,
and the founder's principles of 2026-10-11 set the rules a synthesiser must follow: a synthetic Task has no Reference
and is never tested against a valid Run, its Verifier comes from the scenario, realism is judged against the domain,
and difficulty rises only by more work in the world, never by a vaguer ask or a stricter check.

The nearest published designs are read in .claude/reports/synthetic-data-papers-2026-10-15.md: the AWS AgentCore post,
which grows verifiable customer-service Tasks from the environment, policy and database alone and checks target state
values, and Trajectory2Task, which turns explored trajectories into Tasks with ambiguous, changing or infeasible intent.

## The source of truth

Three things are true for a Task, and nothing else is.

The ask is what the user came for. It can branch on the world or the policy ("if that cannot be changed, cancel it",
"if it is under the limit, exchange it, else return it"); such a branch is still the ask, and the Verifier resolves it
against the world and the policy. The policy is the customer's document of rules. The world is the rows, the tools and
their effects.

Everything else derives from these three: the Intent, the Verifier, the mutants, trust, and the Examiner's rulings.
The mutants measure how tightly a Verifier fits the ask, not how tightly it fits one recorded Run.

The Reference is for fidelity only. A recorded conversation that replays in our world with the same calls and the same
effects tells us the world is rebuilt right: what passed in the customer's environment will pass in ours. That is all
it tells us. The Reference never enters the Verifier, the repair of a Verifier, or the mutants. When a Reference fails
a Verifier, that is a finding about the Reference, not a reason to change the check.

A trace also records how its agent behaved, and users react to that behaviour. A transfer asked for after the agent
failed, a user giving up, a new ask improvised late because the first one went nowhere: these are recorded as evidence
about the agent and never gate a Verifier. They are reactions to one agent's conduct, not the ask.

Benchmark gold and benchmark scripts are a ruler for us. We use them to measure how right our Verifiers are on public
corpora. They are never an input to a build, because a customer has no gold.

This supersedes the sentence "The trace is the only truth" in docs/design-philosophy.md. The trace is one input; its
user side is the main evidence of the ask; its agent side is evidence about one agent and nothing more. The byte-level
provenance that section describes still holds: every derived value still points back to the bytes it came from.

## Why

The heldout 100 of 2026-10-12 (.claude/herdr/truth-1008/heldout-report.md, sections Result at 100 and The wrong
Verifiers) drew 100 unseen Tasks, 50 per corpus, and had one blind rater per corpus label each Verifier against the
benchmark gold. The labels were 43 right, 27 partial, 9 wrong and 21 no_verifier. It passed every pre-registered bar.
Of the 63 trusted Verifiers, 8 were labelled wrong (13 percent).

The nine wrong Verifiers are where the source of truth shows. Read against the founder's rule of 2026-10-15, six gated
the Reference agent's failures as the user voiced them: the Verifier copied an outcome the recorded agent produced
(a refusal, a no-match statement, a cancellation left out, a detail the user only asked for along the way) instead of
what the user came for. Two came through the repair path that answers a failing Reference, where the writer loosened
checks until the wrong Reference passed. One was a misread. The report's own grouping (section Patterns) cuts the same
nine by mechanism rather than by source, and its loosening search names three of the nine as reached through repairs
(d01d by a fails_reference ruling, d79a and 4770 by other rulings), so the six, two and one split is a reading of the
report, not a count it makes. Both readings point the same way: a Verifier that leans on the Reference inherits the
Reference's mistakes, and agreement between fresh Runs and the Reference cannot see them, because fresh Runs agree with
a wrong Reference as readily as with a right one. All four trusted Tasks on a wrong Reference were labelled wrong.

What this evidence cannot show: one rater per corpus with no second rater, one run with no replicate, the rater of the
same model family as the writer and the Examiner, and a draw and a code version that both differ from the earlier
full50 measurement, so no gain over full50 can be split between the fixes and easier Tasks.

## Evals and training data

The eval is the held-out Tasks mined from traces. They are the customer's real asks, in the customer's real world, and
nothing else can stand in for them. Synthetic Tasks are training data. They are measured on the mined held-out set,
never on themselves: a model trained on synthetic Tasks is scored on the held-out mined Tasks, and a synthesiser is
judged by what its Tasks do to that score.

A human sees a sample of the evals before any benchmark number is reported. The intent is that the sample shows the
Task, the Verifier and what the Verifier would accept, so a reviewer can say whether a right agent passes and a wrong
one fails; the step itself is not built (see The loop). A number over evals nobody has looked at is not reported.

## The loop

1. A behaviour statement: the policy plus the asks, refusals and stressors the customer wants covered.
2. An Environment rebuilt from the inputs, at measured fidelity.
3. Tasks and their Verifiers, mined from traces and synthesised from the real world.
4. A human review of a sample of the evals.
5. A benchmark of every model on the reviewed evals.
6. Training on the same set.
7. Harder Tasks, chosen from the trained model's pass rate, and the loop again.

What exists today: the Environment rebuilt from traces with replay fidelity; the Intent and the Verifier written from
the Intent, the policy and the world (D330); trust decided by constructed Runs that must fail (D333) and by mutants
(D356, D362); the Harbor export (D363); and a benchmark of any model by running that export in Harbor.

What does not exist yet: a shape for the behaviour statement; the synthesiser; the human review step; the training
loop; and a measure of realism for synthetic Tasks. Clean-ask (D366 onward, its own stream) takes the Reference out of
the Verifier, the repair path and the mutants, as the source of truth above requires.

## What this does not change

The world is still rebuilt from traces, row by row and tool by tool, with every value pointing back to its bytes.
Fidelity is still measured by replaying recorded conversations, and the fidelity target stands. Trust still rests on
constructed Runs that a right Verifier must fail, a do-nothing Run and a stray write. The judge still answers one item
at a time, 0 or 1, and fails closed when it cannot answer. What changes is only what those mechanisms are measured
against: the ask, the policy and the world, never the recorded agent's outcome.
