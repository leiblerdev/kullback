# Design philosophy

This is the reasoning behind Kullback in one place: what we built, why, and what we chose not to build. The decision log (`decision-log.md`, D01 to D97) has every individual choice with the alternative it beat; the ADRs in `adr/` hold the four that are hard to reverse. This document is the shape of the whole.

## Where it starts

Three sentences from the founder set the direction and everything below is downstream of them: "it should be simple enough and gets the job done in high quality", "as efficient and simple as possible", and "care much more about generalizability than the overfitting".

Simple means every stage can be explained in one sentence and the first version of each stage is the plainest one that keeps the trust gates honest. It does not mean small at any cost; where complexity buys something we can name (per-Task overlays, the repair loop, two judges), it goes in, and the reason is written next to it.

Generalizability means every stage is measured on runs it did not see. A number on the seed runs is printed beside the number on the held-out runs, always, and a gap between them is reported as overfitting rather than averaged away.

High quality means we build for the hardest runs first: long, multi-turn, many tool calls, off the recorded path. The common single-call run is then the easy case of the same machinery, not a separate path with its own bugs.

## What we did, and why

### The trace is the only truth

Every customer file is stored byte for byte and hashed before anything reads it. Every derived value, from a tool signature to a column class to a line of an intent, carries a pointer back to the byte range it came from. When a number in the report is wrong, the chain from the number to the raw bytes is short and mechanical.

The reason is trust. The product's claim is "on your runs, this model does the job". A customer can only believe that if they can follow any figure back to their own data. It also removes a whole class of argument: if something cannot be pointed back to a trace, it is not in the environment, full stop.

### Verdicts come from the end state after re-execution

A candidate model is run through the conversation in the rebuilt environment, and the verdict is computed from what changed in the world and what the user was told, compared with what the frontier model's successful runs changed. The route the model took, the order of its calls, and the text of its reasoning never move a verdict. (ADR-0004.)

Static replay of a logged run with a different model is unsound past the first divergent action: a 2026 CMU study found that only 3 to 8 percent of later recorded states remain valid once an early action changes. Path matching measures imitation, not competence, and penalizes every valid alternative route. Reasoning text is not gradable from text alone: chain-of-thought faithfulness sits at 20 to 40 percent and falls as models get stronger, and editing only the reasoning inflates judge false positives by up to 90 percent. Every durable agent benchmark grades the world, and so do we.

"End state" is defined by effects, and effects fall on two things: the world (writes) and the user (what they were told or asked). A question the frontier asked in every successful re-run is a required atom; a write whose value came from the user's answer has to hold the answer given in the candidate's own run. What remains "path" is exactly what has no effect.

### Code decides, models propose

The verdict is computed by code over the run's event log and the Task's verifier. Language models appear at defined points and only in defined roles: writing a tool body that then has to pass five gates, proposing a column class that a rule verifies or a re-run overrides, naming a cluster whose membership code decided, and answering a question that code could not compile into a predicate.

Where a model judges, two agentic judges do it independently, each with read tools over the starting and end state, each required to run at least one tool check and cite a span. If they disagree, the run goes to a queue for a person. A judge can remove a run from the bar or widen a verifier for every candidate. It can never award a pass. (ADR-0003, D92.)

The reason is asymmetric cost. A conservative error keeps a Task on the frontier model and costs the customer some savings. A liberal error moves a Task that then regresses in production and costs the customer's trust in the whole report. The machinery is built to make only the first kind of mistake, which also means the reported savings are a floor, not an estimate.

### Only replicas count toward the bar

The bar is the frontier model's score on the customer's own inputs. Only re-executions of recorded runs feed it. Synthesized scenarios, however well grounded, are a later phase for coverage, hardening and post-training, and are reported separately if at all. (ADR-0005.)

A synthesized task is not the customer's input. Letting it into the verdict changes the claim from "safe on your traffic" to "safe on tasks we invented", in either direction. Weighting it lower does not help; any weight makes the bar a blend the customer cannot audit.

### Gates, never blends

Every stage ends in a code gate that passes or fails, with the failure attached to the artifact and handed back to the stage that produced it, for a bounded number of retries. There is no score that mixes a good environment with a weak verifier into a single number. Gate A demands 100 percent of recorded writes reproduced after canonicalization and zero unexplained read mismatches, on the held-out runs as well as the seed. A verifier enters the pool only after the recorded run passes it, an empty run fails it, a plausible wrong run fails it, and a grep finds no leaked reference.

A run that needed a model stand-in for any tool call is reported as assisted and gets no counted verdict. A Task with too few runs to hold anything out is marked unguarded. A tool nobody could classify is flagged and blocks the review. None of these are averaged in. The report opens with whether the environment was built at all before it shows a single per-Task number.

The reason is that a blended score hides exactly the failure it should expose. The one number a customer would look at would be a mixture of environment fidelity, verifier validity and model quality, and no one could say which had moved.

### Data follows trust, one rung at a time

The environment is built from traces alone first. Every entity a run read becomes a row; the shared world is reconstructed by replaying the corpus backwards, and each Task carries an overlay of the rows its runs actually saw, in the version they saw. Unseen entities return "not found" and the run is reported assisted. Tool definitions that already travel in the traces are parsed from day one. Schema, policy documents and finally a database snapshot are asked for later, each rung justified by the assisted share the previous rung left. (ADR-0006, D74.)

The first customer conversation must not begin with a data-sharing agreement. Traces are already leaving their system to a tracing vendor; a snapshot is a different conversation, and the assisted share is the evidence that makes it a reasonable one.

### Two programs, one set of records, and a wall between them

The Builder writes the environment; the Runner executes runs and computes verdicts. They share the record definitions and nothing else. The Runner has no import path to the Builder, and the candidate model can see nothing from the verifier: not its atoms, not the grader fields of a benchmark trace, not the reference. A test scans the imports and fails the suite if the wall is breached. (D89, D91.)

The loop that advances one turn is a single function, written so that a `reset` and `step` wrapper for any environment standard is packaging rather than redesign. The tau2-bench file shape came first because it is the one place we hold ground truth for the reconstruction; the wrappers come after the slice passes its gates. (D90.)

### The report shows, the person decides

The report opens with whether the environment was built (gates, scorecard, assisted tools, unguarded Tasks, overlays), then gives each Task's numbers and a suggestion. The routing plan is written from what the person decides, not from the suggestion. When the spend ceiling is hit, the build stops where it is and reports as it is; nothing is quietly switched to a cheaper path to finish. (D85, D86.)

### The Builder improves itself, slowly and under watch

The Builder may edit its own prompts, rules and code, one change per round. Each edit is a node in a memory tree, and an evaluator outside the loop accepts or rejects it on the held-out runs. Batches of changes are allowed only after twenty accepted single rounds, with bisection on a reject. Lessons from one customer travel to the next as a file, and the Builder has to argue the relevance of each lesson, with evidence, before applying it; lessons it sets aside are listed in the report. (D82, D87.)

## What we did not do, and why

We do not grade the path. A different tool sequence that reaches a satisfying end state passes. Path differences are reason codes on the report.

We do not grade reasoning text. It is monitored for fabricated observations only.

We do not aggregate step-level matches into a run score. Agents recover from most local errors, only about 16 percent of steps are outcome critical, and all-or-nothing aggregation is systematically pessimistic.

We do not let a judge overrule an end-state check or award a pass, and we do not take a third judge to break a tie on a reference. A disputed reference sets the Task aside as not gradable until a person resolves it. (D93.)

We do not count synthesized scenarios toward the bar, and we do not synthesize rows for entities the traces never showed and then treat them as real. Synthetic rows are tagged and their runs are assisted.

We do not cap or compact the candidate's context. The candidate sees the production setting: the system prompt, the tool definitions and the conversation the customer's agent gave it, with production's own compaction if the traces show one. (D65.)

We did not build an event bus, a plugin system, MCP inside the core, subagents, a UI framework, or session trees for runs. A fork is a new run with a parent id. Each of those would be a second way to do something the records already do.

We did not train a user simulator. The simulated user is rules from the recorded user: facts exact, style representative, and when a fact is neither in the trace nor in the world, it says so and the run is marked. Prompt-only simulators err 40 to 47 percent of the time in the published measurements; tool-constrained ones about 16 percent; rules grounded in the recorded facts are the version of that we can audit. (D44, D77.)

We did not keep the benchmark secret to make it last. The counter-argument in the research is that expert curation, not secrecy, extends a benchmark's life; our default is no human in the loop until the disagreement queue shows its size. This one is open. (R29 against D92.)

We did not write the sandbox yet. Compiled tools run in a subprocess for the compile gates and in process for the Runner. That is a blast-radius reducer, not a security boundary, and it is the first item on the list before any customer trace is compiled.

## Where the evidence disagrees with us

We keep these in the open because a design that hides its counter-evidence is asking to be trusted rather than checked.

EnvFactory's ablation found that half trajectory match plus half final-state equivalence beat either alone; BFCL multi-turn requires the ground-truth call sequence to be a subset of the executed calls; AgentScaler requires exact call match for read-only tasks. We hold to end state only, and we record the dissent in ADR-0004.

The synthetic-data research wanted a light path check on top of the end state. We rejected it (D94) for the same reason.

The re-run count k that fixes the required and allowed atom sets is an assumption (A30, "stabilizes within ten re-runs") until the experiment on tau2 retail decides it.

The first clustering default was wrong on real data: threshold 0.3 gave F1 0.28 against tau2's task groups where 0.6 gives 0.72. The design said the slice would revise the defaults, and it did.

The design budgeted 2,700 to 3,700 lines; the first build is about 8,900, and every module says what it carries beyond its band. Whether the bands or the code change is an open call.

## How to argue with any of this

Find the D number, read the alternative it beat, and open an issue that names it. A change of decision is a new entry in the decision log; an ADR is only for a decision that is hard to reverse, surprising without context, and the result of a real trade-off. Numbers that contradict a decision are the strongest argument, and `SLICE_RESULTS.md` is where they go, with the script that produced them.
