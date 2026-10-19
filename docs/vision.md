# Vision and the source of truth

What the product is, what it takes in, what it treats as true, and what exists today (D365). Where this page and
[design-philosophy.md](design-philosophy.md) disagree on the source of truth, this page holds. Terms are in the
[glossary](glossary.md).

## The product

You tell an agent the behaviour you want from your own agent. It builds the evals and the training data for that
behaviour: a realistic Environment, Tasks inside it, and a Verifier for each Task. A human reviews a sample of the
evals, every model you care about is benchmarked on them, and the model you pick is trained on the same set.

In the founder's words: "an agent which asks you the right behaviour you want and then we generate the evals or
the same and then the human sees some of the evals and then we benchmark all the models across it and then train
the model then and there".

## Inputs

Traces are one input. Beside them sit the database structure, the business knowledge graph (how the customer's
entities, rules and processes relate) and the policy document the agent works under. From these we build the
most realistic Environment we can. Today Kullback builds it from traces alone, at measured replay fidelity.

Synthetic Tasks then widen the Environment. They come from the real world (what happens in this kind of business,
found by search against the domain) and are bound to the Environment's real rows. [scaling.md](scaling.md) has
the plan and the rules a synthetic Task follows.

## The source of truth

Three things are true for a Task, and nothing else is.

- The ask: what the user came for. It can branch on the world or the policy ("if that cannot be changed, cancel
  it"); the branch is still the ask, and the Verifier resolves it against the world and the policy.
- The policy: the customer's document of rules.
- The world: the rows, the tools and their effects.

Everything else derives from these three: the Intent, the Verifier, the mutants, trust and the Examiner's
rulings. The mutants measure how tightly a Verifier fits the ask, not how tightly it fits one recorded Run.

The Reference is for fidelity only. A recorded conversation that replays in our world with the same calls and
effects shows the world is rebuilt right. The Reference never enters the Verifier, its repair or the mutants.
When a Reference fails a Verifier, that is a finding about the Reference, not a reason to change the check.

A trace also records how its agent behaved, and users react to that behaviour: a transfer asked for after the
agent failed, a user giving up, a new ask improvised late. These are evidence about one agent's conduct and never
gate a Verifier.

Benchmark gold and benchmark scripts are a ruler. We use them to measure how right our Verifiers are on public
corpora, never as an input to a build, because a customer has no gold.

This replaces "The trace is the only truth" in design-philosophy.md. The byte-level provenance it describes
still holds: every derived value points back to the bytes it came from.

## Evals and training data

The eval is the held-out Tasks mined from traces: the customer's real asks in the customer's real world.
Synthetic Tasks are training data. They are scored on the mined held-out set, never on themselves. A number over
evals nobody has looked at is not reported.

## The loop

1. A behaviour statement: the policy plus the asks, refusals and stressors the customer wants covered.
2. An Environment rebuilt from the inputs, at measured fidelity.
3. Tasks and Verifiers, mined from traces and synthesised from the real world.
4. A human review of a sample of the evals.
5. A benchmark of every model on the reviewed evals.
6. Training on the same set.
7. Harder Tasks, chosen from the trained model's pass rate, and the loop again.

## What exists today

- The Environment rebuilt from traces, with replay fidelity.
- The Intent, and the Verifier written from the Intent, the policy and the world (D330).
- Trust by constructed Runs that must fail (D333) and by mutants (D356, D362).
- The Harbor export (D363), so any model can be benchmarked by running it in Harbor.
- Scenarios and synthetic Tasks born from them (`kullback synth`, D372 to D389).
- A training agent (`kullback train`).

Not built yet: a shape for the behaviour statement, the human review step, and a measure of realism for
synthetic Tasks.
